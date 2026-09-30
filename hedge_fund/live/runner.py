"""The paper fund's two daily steps: submit (morning) and reconcile (next morning).

submit plans and sends market-on-close orders on rebalance days; every guard
that can stop it runs before any analyst is asked or any order is sent.
reconcile records the previous session's fills and closing NAV. Alpaca is the
source of truth for the book; the ledger is the record.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from hedge_fund.brokers.alpaca import Account, OrderResult
from hedge_fund.data.protocol import DataClient
from hedge_fund.data.sessions import NEW_YORK
from hedge_fund.fund import Fund, FundSpec
from hedge_fund.live.calendar import MOC_CUTOFF, client_order_prefix, is_rebalance_day, ny_midnight
from hedge_fund.live.ledger import Ledger, NavRow
from hedge_fund.live.plan import LivePlan, plan_rebalance
from hedge_fund.paths import KILL_PATH
from hedge_fund.pipeline.run_cycle import exact_marks

logger = logging.getLogger(__name__)

DRAWDOWN_HALT = 0.15   # halt and flatten at a 15% fall from peak NAV

# Past this time on a trading day, that day's MOC fills may already be in the
# account, so "current book" would no longer be the previous session's book.
_RECONCILE_DEADLINE = time(15, 50)


class PaperAccount(Protocol):
    """What the runner needs from Alpaca — AlpacaPaperClient, or a test fake."""

    def calendar(self, start: str, end: str) -> list[str]: ...
    def list_orders(self, after: str) -> list[OrderResult]: ...
    def positions(self) -> dict[str, int]: ...
    def account(self) -> Account: ...
    def submit_moc(self, ticker: str, side: Literal["buy", "sell"], quantity: int, client_order_id: str) -> OrderResult: ...


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
    """Plan and send today's market-on-close orders, or say exactly why not.

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
        if now.time() > MOC_CUTOFF:
            return done("too_late", f"after {MOC_CUTOFF:%H:%M} ET; market-on-close orders are closed")
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
    for order in plan.orders:
        orders.append(client.submit_moc(order.ticker, order.side, order.quantity, prefix + order.ticker))
        payload["orders"] = [o.model_dump(mode="json") for o in orders]
        ledger.write_plan(session, payload)
    rejected = sum(o.status == "rejected" for o in orders)
    return done(status, f"{len(orders)} orders sent, {rejected} rejected", plan=plan, orders=orders)


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
    if today in sessions and now.time() >= _RECONCILE_DEADLINE:
        raise ValueError("after 15:50 ET today's market-on-close fills may already be booked; "
                         "reconcile the previous session tomorrow before 15:50 ET")
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
