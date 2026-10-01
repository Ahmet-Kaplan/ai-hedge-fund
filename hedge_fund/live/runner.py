"""The paper fund's two daily steps: submit (morning) and reconcile (next morning).

submit plans and sends market orders on rebalance days (sent before the
open, they fill at the opening price); every guard that can stop it runs
before any analyst is asked or any order is sent.
reconcile records the previous session's fills and closing NAV. Alpaca is the
source of truth for the book; the ledger is the record.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from hedge_fund.brokers.alpaca import Account, OrderResult
from hedge_fund.data.protocol import DataClient
from hedge_fund.data.sessions import NEW_YORK
from hedge_fund.fund import Fund, FundSpec
from hedge_fund.live.calendar import ORDER_CUTOFF, RECONCILE_DEADLINE, client_order_prefix, is_rebalance_day, ny_midnight
from hedge_fund.live.ledger import Ledger, NavRow
from hedge_fund.live.plan import LivePlan, plan_rebalance
from hedge_fund.paths import KILL_PATH
from hedge_fund.pipeline.run_cycle import exact_marks

logger = logging.getLogger(__name__)

DRAWDOWN_HALT = 0.15   # halt and flatten at a 15% fall from peak NAV



class PaperAccount(Protocol):
    """What the runner needs from Alpaca — AlpacaPaperClient, or a test fake."""

    def calendar(self, start: str, end: str) -> list[str]: ...
    def list_orders(self, after: str) -> list[OrderResult]: ...
    def positions(self) -> dict[str, int]: ...
    def account(self) -> Account: ...
    def submit_order(self, ticker: str, side: Literal["buy", "sell"], quantity: int, client_order_id: str) -> OrderResult: ...


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

SubmitStatus = Literal[
    "killed", "not_trading_day", "too_late", "already_submitted", "halted",
    "not_rebalance_day", "dry_run", "submitted", "flattening",
]


class SubmitResult(BaseModel):
    status: SubmitStatus
    session: str
    detail: str = ""
    plan: LivePlan | None = None
    orders: list[OrderResult] = Field(default_factory=list)


def submit(
    fund: Fund, universe: list[str], client: PaperAccount, data_client: DataClient,
    ledger: Ledger, *, now: datetime, dry_run: bool = False, force_rebalance: bool = False,
    kill_path: Path = KILL_PATH,
) -> SubmitResult:
    """Plan and send today's orders, or say exactly why not.

    A dry run skips the calendar, clock and rebalance-day guards (it never
    sends anything, so the plan can be inspected any day) and never halts.
    force_rebalance trades today even mid-period (e.g. to start the fund);
    every other guard still applies.
    """
    now = now.astimezone(NEW_YORK)
    session = now.date().isoformat()

    def done(status: SubmitStatus, detail: str = "", **extra) -> SubmitResult:
        logger.info("submit %s: %s %s", session, status, detail)
        return SubmitResult(status=status, session=session, detail=detail, **extra)

    if kill_path.exists():
        return done("killed", f"{kill_path} exists; delete it to trade again")
    sessions = client.calendar((now.date() - timedelta(days=10)).isoformat(), session)
    prefix = client_order_prefix(fund.spec.name, session)
    if not dry_run:
        if not sessions or sessions[-1] != session:
            return done("not_trading_day")
        if now.time() > ORDER_CUTOFF:
            return done("too_late", f"after {ORDER_CUTOFF:%H:%M} ET; too close to the close to send day orders")
        if any(o.client_order_id.startswith(prefix) for o in client.list_orders(after=ny_midnight(session))):
            return done("already_submitted")

    breach = _drawdown_breach(ledger)
    if breach and not dry_run and not ledger.is_halted():
        ledger.halt(breach)
    halted = ledger.is_halted() or breach is not None
    positions = client.positions()
    if halted:
        if not positions:
            return done("halted", breach or ledger.halted_path.read_text().strip())
        status: SubmitStatus = "flattening"
    else:
        previous = [s for s in sessions if s < session]
        scheduled = bool(previous) and is_rebalance_day(session, previous[-1], fund.spec.rebalance)
        if not dry_run and not force_rebalance and not scheduled:
            return done("not_rebalance_day")
        status = "submitted"

    # Planning can raise (data or LLM outage) — that aborts before any order exists.
    plan = plan_rebalance(fund, universe, positions, client.account().cash, data_client, session, flatten=halted)
    payload = {"status": status, "plan": plan.model_dump(mode="json"), "orders": []}
    if dry_run:
        ledger.write_plan(session, payload, dry_run=True)
        return done("dry_run", f"would have been {status}: {len(plan.orders)} orders", plan=plan)

    ledger.write_plan(session, payload)   # the plan is on disk before anything is sent
    orders: list[OrderResult] = []
    for ticker, side, quantity, client_id in _executable(plan, prefix):
        orders.append(client.submit_order(ticker, side, quantity, client_id))
        payload["orders"] = [o.model_dump(mode="json") for o in orders]
        ledger.write_plan(session, payload)
    rejected = sum(o.status == "rejected" for o in orders)
    return done(status, f"{len(orders)} orders sent, {rejected} rejected", plan=plan, orders=orders)


def retry_rejected(
    client: PaperAccount, ledger: Ledger, *, now: datetime, kill_path: Path = KILL_PATH,
) -> list[OrderResult]:
    """Re-send today's rejected orders (e.g. blocked by a since-filled manual order).

    Same quantities and client order ids as the original plan; the plan file is
    updated in place. Refuses when killed, when there is no plan for today, or
    after the order cutoff.
    """
    now = now.astimezone(NEW_YORK)
    session = now.date().isoformat()
    if kill_path.exists():
        raise ValueError(f"{kill_path} exists (KILL switch on); not sending anything")
    if now.time() > ORDER_CUTOFF:
        raise ValueError(f"after {ORDER_CUTOFF:%H:%M} ET; too close to the close to send day orders")
    payload = ledger.read_plan(session)
    if payload is None:
        raise ValueError(f"no plan submitted for {session}; nothing to retry")
    retried: list[OrderResult] = []
    for i, order in enumerate(payload["orders"]):
        if order["status"] != "rejected":
            continue
        result = client.submit_order(order["ticker"], order["side"], order["quantity"], order["client_order_id"])
        payload["orders"][i] = result.model_dump(mode="json")
        ledger.write_plan(session, payload)
        retried.append(result)
        logger.info("retry %s %s %d: %s %s", order["ticker"], order["side"], order["quantity"], result.status, result.reason or "")
    return retried


def _executable(plan: LivePlan, prefix: str) -> list[tuple[str, Literal["buy", "sell"], int, str]]:
    """Orders as the broker accepts them: a long↔short flip becomes close-then-open.

    Alpaca won't take one order that crosses zero (sell 10 while long 5); it
    must be sell 5 to close, then sell 5 to open the short.
    """
    out = []
    for order in plan.orders:
        held = plan.positions.get(order.ticker, 0)
        delta = order.quantity if order.side == "buy" else -order.quantity
        crosses = held != 0 and (held + delta) * held < 0
        if crosses:
            out.append((order.ticker, order.side, abs(held), prefix + order.ticker))
            out.append((order.ticker, order.side, order.quantity - abs(held), prefix + order.ticker + "-open"))
        else:
            out.append((order.ticker, order.side, order.quantity, prefix + order.ticker))
    return out


def _drawdown_breach(ledger: Ledger) -> str | None:
    """A halt reason if the latest NAV is DRAWDOWN_HALT or more below its peak."""
    peak, latest = ledger.peak_equity(), ledger.latest_equity()
    if not peak or latest is None or latest > (1 - DRAWDOWN_HALT) * peak:
        return None
    return (f"drawdown halt: equity {latest:,.2f} is {1 - latest / peak:.1%} below peak {peak:,.2f}. "
            "Run `aihf-paper resume` to trade again.")


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------

class FillRecord(BaseModel):
    client_order_id: str
    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    filled_qty: int
    status: str
    fill_price: float | None = None
    reference_price: float | None = None
    slippage_bps: float | None = None      # positive = cost vs the sizing close


class ReconcileResult(BaseModel):
    session: str
    nav: NavRow
    fills: list[FillRecord] = Field(default_factory=list)
    mismatches: list[str] = Field(default_factory=list)


def session_to_reconcile(client: PaperAccount, now: datetime) -> str:
    """The most recent completed session — the only one whose closing book is still observable."""
    now = now.astimezone(NEW_YORK)
    today = now.date().isoformat()
    sessions = client.calendar((now.date() - timedelta(days=10)).isoformat(), today)
    if today in sessions and now.time() >= RECONCILE_DEADLINE:
        raise ValueError("after the 09:30 ET open today's fills may already be booked; "
                         "reconcile the previous session before 09:30 ET (14:30 UK) on a trading day")
    past = [s for s in sessions if s < today]
    if not past:
        raise ValueError("no completed trading session in the last 10 days")
    return past[-1]


def reconcile(
    spec: FundSpec, client: PaperAccount, data_client: DataClient, ledger: Ledger, *, session: str,
) -> ReconcileResult:
    """Record `session`'s fills and closing NAV. Safe to re-run; the NAV row is replaced."""
    prefix = client_order_prefix(spec.name, session)
    orders = [o for o in client.list_orders(after=ny_midnight(session)) if o.client_order_id.startswith(prefix)]
    plan = ledger.read_plan(session)
    reference = plan["plan"]["marks"] if plan else {}
    fills = [_fill_record(o, reference.get(o.ticker)) for o in orders]
    if fills:
        ledger.write_fills(session, {"session": session, "fills": [f.model_dump(mode="json") for f in fills]})

    positions = client.positions()
    cash = client.account().cash
    closes = exact_marks(sorted(set(positions) | {spec.benchmark}), session, data_client)
    long_ = sum(s * closes[t] for t, s in positions.items() if s > 0)
    short = sum(-s * closes[t] for t, s in positions.items() if s < 0)
    equity = cash + long_ - short
    row = NavRow(
        date=session, equity=round(equity, 2), cash=round(cash, 2),
        long_exposure=round(long_, 2), short_exposure=round(short, 2),
        gross=round((long_ + short) / equity, 6) if equity > 0 else 0.0,
        benchmark_close=closes[spec.benchmark],
    )
    ledger.upsert_nav(row)

    mismatches = _mismatches(plan["plan"]["positions"], fills, positions) if plan and fills else []
    for message in mismatches:
        logger.warning(message)
    logger.info("reconcile %s: equity %.2f, %d fills", session, row.equity, len(fills))
    return ReconcileResult(session=session, nav=row, fills=fills, mismatches=mismatches)


def _fill_record(order: OrderResult, reference: float | None) -> FillRecord:
    slippage = None
    if order.filled_avg_price is not None and reference:
        direction = 1 if order.side == "buy" else -1
        slippage = round(direction * (order.filled_avg_price - reference) / reference * 10_000, 2)
    return FillRecord(
        client_order_id=order.client_order_id, ticker=order.ticker, side=order.side,
        quantity=order.quantity, filled_qty=order.filled_qty, status=order.status,
        fill_price=order.filled_avg_price, reference_price=reference, slippage_bps=slippage,
    )


def _mismatches(start: dict[str, int], fills: list[FillRecord], actual: dict[str, int]) -> list[str]:
    expected = dict(start)
    for f in fills:
        expected[f.ticker] = expected.get(f.ticker, 0) + (f.filled_qty if f.side == "buy" else -f.filled_qty)
    return [
        f"{t}: ledger expects {expected.get(t, 0)} shares, Alpaca holds {actual.get(t, 0)}; trusting Alpaca"
        for t in sorted(set(expected) | set(actual))
        if expected.get(t, 0) != actual.get(t, 0)
    ]
