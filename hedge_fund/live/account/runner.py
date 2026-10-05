"""The live account's daily steps: reconcile (record yesterday) and submit (trade today).

submit never assesses anything: the satellite copies the paper fund's latest
plan, scaled by the agent share. Every guard runs before any order is sent,
and the plan is written to disk before the first order.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from hedge_fund.brokers.alpaca import LiveOrder
from hedge_fund.crypto.rules import ma100
from hedge_fund.data.protocol import DataClient
from hedge_fund.data.sessions import NEW_YORK
from hedge_fund.live.account.ledger import LiveLedger, LiveNavRow
from hedge_fund.live.account.orders import PlannedOrder, check_orders, size_orders, to_limits
from hedge_fund.live.account.settings import LiveSettings
from hedge_fund.live.account.target import account_book, affordable_satellite, latest_paper_weights, target_book
from hedge_fund.live.calendar import ORDER_CUTOFF, is_rebalance_day, ny_midnight
from hedge_fund.live.ledger import Ledger
from hedge_fund.paths import KILL_PATH
from hedge_fund.pipeline.run_cycle import exact_marks

logger = logging.getLogger(__name__)

LiveStatus = Literal["killed", "not_confirmed", "needs_dry_run", "not_trading_day", "too_late",
                     "already_submitted", "nothing_to_do", "dry_run", "submitted"]


class LiveSubmitResult(BaseModel):
    status: LiveStatus
    session: str
    detail: str = ""
    target: dict[str, float] = Field(default_factory=dict)
    orders: list[PlannedOrder] = Field(default_factory=list)
    results: list[LiveOrder] = Field(default_factory=list)
    trend: str = "off"
    exposure: dict[str, int] = Field(default_factory=dict)        # ma100 per coin: 1 in, 0 out
    shadow: list[PlannedOrder] = Field(default_factory=list)      # crypto orders ma100 would place


CryptoMarks = Callable[[list[str], str], dict[str, float]]
CryptoHistory = Callable[[list[str], str], dict[str, list[float]]]   # daily closes through a UTC day, oldest first
CryptoQuotes = Callable[[list[str]], dict[str, tuple[float, float]]]  # (bid, ask) now
MA_DAYS = 100
MAX_LIMIT_TRIES = 2   # a third consecutive attempt goes at market (spec §15)


def _is_crypto(ticker: str) -> bool:
    return "/" in ticker


def _marks(names, day: str, data_client: DataClient, crypto_marks: CryptoMarks | None) -> dict[str, float]:
    """Stock closes from the data client; crypto pairs from crypto_marks (UTC daily closes)."""
    stocks = sorted(n for n in names if not _is_crypto(n))
    crypto = sorted(n for n in names if _is_crypto(n))
    marks = exact_marks(stocks, day, data_client) if stocks else {}
    if crypto:
        if crypto_marks is None:
            raise ValueError(f"crypto holdings or targets {crypto} need a crypto price source")
        marks.update(crypto_marks(crypto, day))
    return marks


def reconcile_live(settings: LiveSettings, client, data_client: DataClient, ledger: LiveLedger, *, session: str,
                   crypto_marks: CryptoMarks | None = None) -> LiveNavRow:
    """Record `session`'s closing value, deposits and satellite return; halt a trailing satellite."""
    rows = ledger.live_nav_rows()
    previous = max((r.date for r in rows if r.date < session), default=None)
    since = previous or (datetime.fromisoformat(session) - timedelta(days=30)).date().isoformat()
    ledger.add_flows(client.cash_flows(after=since))
    holdings = client.holdings()
    cash = client.account().cash
    core = settings.core_ticker
    closes = _marks(set(holdings) | {core}, session, data_client, crypto_marks)
    core_value = holdings.get(core, 0.0) * closes[core]
    satellite_value = sum(q * closes[t] for t, q in holdings.items() if t != core and not _is_crypto(t))
    crypto_value = sum(q * closes[t] for t, q in holdings.items() if _is_crypto(t))

    satellite_return = None
    before = ledger.holdings_before(session)
    if before:
        prev_day, prev_holdings = before
        sat = {t: q for t, q in prev_holdings.items() if t != core and not _is_crypto(t) and q}
        if sat:
            start = exact_marks(sorted(sat), prev_day, data_client)
            end = exact_marks(sorted(sat), session, data_client)
            base = sum(abs(q) * start[t] for t, q in sat.items())
            satellite_return = sum(q * (end[t] - start[t]) for t, q in sat.items()) / base

    row = LiveNavRow(
        date=session, equity=round(cash + core_value + satellite_value + crypto_value, 2), cash=round(cash, 2),
        net_flow=round(ledger.net_flow(previous, session), 2), core_value=round(core_value, 2),
        satellite_value=round(satellite_value, 2), crypto_value=round(crypto_value, 2),
        core_close=closes[core], satellite_return=satellite_return,
    )
    ledger.upsert_live_nav(row)
    ledger.save_holdings(session, holdings)

    since_review = ledger.last_review_date()
    excess = ledger.satellite_excess(since_review) if since_review else None
    if excess is not None and excess <= -settings.satellite_halt_relative and not ledger.satellite_halted():
        ledger.halt_satellite(f"satellite trails the core by {-excess:.1%} since {since_review}; "
                              "it is sold into the core until a review sets a new share")
        logger.warning("satellite halted: trails the core by %.1f%%", -excess * 100)
    logger.info("reconcile %s: equity %.2f, net flow %.2f", session, row.equity, row.net_flow)
    return row


def submit_live(
    settings: LiveSettings, client, data_client: DataClient, ledger: LiveLedger, paper: Ledger, *,
    now: datetime, dry_run: bool = False, kill_path: Path = KILL_PATH, crypto_marks: CryptoMarks | None = None,
    crypto_history: CryptoHistory | None = None, crypto_quotes: CryptoQuotes | None = None,
) -> LiveSubmitResult:
    now = now.astimezone(NEW_YORK)
    session = now.date().isoformat()

    def done(status: LiveStatus, detail: str = "", **extra) -> LiveSubmitResult:
        logger.info("live submit %s: %s %s", session, status, detail)
        return LiveSubmitResult(status=status, session=session, detail=detail, **extra)

    if kill_path.exists():
        return done("killed", f"{kill_path} exists")
    if not settings.confirm_live and not dry_run:   # a dry run is how you check before confirming
        return done("not_confirmed", "set confirm_live: true in live.yaml to allow real orders")
    if not dry_run and not ledger.dry_run_done():
        return done("needs_dry_run", "run `aihf-live submit --dry-run` once and review it first")
    sessions = client.calendar((now.date() - timedelta(days=10)).isoformat(), session)
    prefix = f"live-{session}-"
    if not dry_run:
        if not sessions or sessions[-1] != session:
            return done("not_trading_day")
        if now.time() > ORDER_CUTOFF:
            return done("too_late", f"after {ORDER_CUTOFF:%H:%M} ET")
        if any(o.client_order_id.startswith(prefix) for o in client.list_orders(after=ny_midnight(session))):
            return done("already_submitted")
    previous = [s for s in sessions if s < session]
    if not previous:
        raise ValueError("no completed session in the last 10 days to price the account")
    mark_day = previous[-1]

    # The bot's crypto orders from earlier days that are still open: cancelled before sizing (in every
    # mode, so switching ma100 off leaves nothing behind); a dry run only looks.
    stale = [o for o in client.open_orders() if _is_crypto(o.ticker)
             and o.client_order_id.startswith("live-") and not o.client_order_id.startswith(prefix)]
    if stale and not dry_run:
        client.cancel_orders([o.order_id for o in stale])
    missed = {o.ticker: o.side for o in stale}

    holdings = client.holdings()
    cash = client.account().cash
    paper_weights, paper_session, fresh = latest_paper_weights(paper, session, settings.stale_plan_days)
    share = 0.0 if ledger.satellite_halted() else settings.agent_share
    core = settings.core_ticker
    # Price the paper fund's names only when the satellite can hold them: a core-only
    # run must not fail because some paper name lacks a close.
    wanted = (set(paper_weights) | set(_last_fresh_satellite(ledger))) if share > 0 else set()
    crypto_core = set(settings.crypto_core) if settings.crypto_share > 0 else set()
    names = sorted(set(holdings) | {core} | wanted | crypto_core)
    marks = _marks(names, mark_day, data_client, crypto_marks)
    equity = cash + sum(q * marks[t] for t, q in holdings.items())
    shorts_ok = settings.shorts_enabled and equity >= settings.short_min_equity
    if fresh:
        satellite = affordable_satellite(
            paper_weights, share, shorts_ok=shorts_ok, max_name=settings.satellite_max_name_pct, core_ticker=core,
            satellite_dollars=share * (1 - settings.crypto_share) * equity,
            min_position_usd=settings.satellite_min_position_usd)
    else:   # the paper fund missed its run: hold the satellite as last set
        satellite = _last_fresh_satellite(ledger) if share > 0 else {}
    target = account_book(target_book(satellite, core), settings)
    rebalance = not holdings or is_rebalance_day(session, mark_day, settings.rebalance)
    sizing = dict(rebalance=rebalance, min_order_usd=settings.min_order_usd, min_trade_pct=settings.min_trade_pct,
                  cash_buffer_pct=settings.cash_buffer_pct)

    trend = settings.crypto_trend if crypto_core else "off"
    exposure: dict[str, int] = {}
    shadow: list[PlannedOrder] = []
    trend_target, out_held = target, set()
    if trend != "off":
        if crypto_history is None:
            raise ValueError("crypto_trend needs a crypto price history source")
        signal_day = (now.astimezone(timezone.utc).date() - timedelta(days=1)).isoformat()
        history = crypto_history(sorted(crypto_core), signal_day)
        short = [c for c in sorted(crypto_core) if len(history.get(c, [])) < MA_DAYS]
        if short:   # never read missing data as "below the average" and sell
            raise ValueError(f"ma100 needs {MA_DAYS} daily closes through {signal_day}; too few for {', '.join(short)}")
        exposure = {c: ma100(history[c]) for c in sorted(crypto_core)}
        trend_target = {t: w * exposure.get(t, 1) for t, w in target.items()}
        out_held = {c for c, e in exposure.items() if e == 0 and holdings.get(c, 0) > 0}
    if trend == "ma100":
        target = trend_target
    orders = size_orders(holdings, cash, marks, target, also_rebalance=out_held if trend == "ma100" else set(), **sizing)
    if trend == "shadow":
        shadow = [o for o in size_orders(holdings, cash, marks, trend_target, also_rebalance=out_held, **sizing)
                  if _is_crypto(o.ticker)]

    quotes: dict[str, tuple[float, float]] = {}
    limits: dict[str, dict] = {}
    if trend == "ma100":
        before = ledger.crypto_limits()
        market = set()
        for o in orders:
            if not _is_crypto(o.ticker):
                continue
            prior = before.get(o.ticker)
            tries = prior["tries"] + 1 if prior and prior["side"] == o.side and missed.get(o.ticker) == o.side else 1
            if tries > MAX_LIMIT_TRIES:
                market.add(o.ticker)
            else:
                limits[o.ticker] = {"side": o.side, "tries": tries}
        if limits:
            if crypto_quotes is None:
                raise ValueError("ma100 limit orders need a crypto quote source")
            quotes = crypto_quotes(sorted(limits))
        orders = to_limits(orders, quotes, market=market)
    check_orders(orders, holdings, cash, marks, {core} | set(settings.crypto_core), shorts_ok=shorts_ok, max_name=settings.satellite_max_name_pct)
    payload = {"session": session, "rebalance": rebalance, "equity": equity, "cash": cash, "holdings": holdings,
               "marks": marks, "paper_plan": paper_session, "paper_fresh": fresh, "agent_share": share,
               "satellite": satellite, "target": target, "crypto_trend": trend, "exposure": exposure,
               "quotes": quotes, "shadow": [o.model_dump() for o in shadow],
               "orders": [o.model_dump() for o in orders], "results": []}
    trend_info = dict(trend=trend, exposure=exposure, shadow=shadow)
    if dry_run:
        ledger.write_plan(session, payload, dry_run=True)
        ledger.mark_dry_run_done()
        return done("dry_run", f"{len(orders)} orders", target=target, orders=orders, **trend_info)
    if trend == "ma100" or ledger.crypto_limits():
        ledger.save_crypto_limits(limits)
    if not orders:
        return done("nothing_to_do", target=target, **trend_info)

    ledger.write_plan(session, payload)
    results: list[LiveOrder] = []
    for o in orders:
        cid = f"{prefix}{o.ticker}-{o.side}"
        if o.limit_price is not None:
            results.append(client.limit_order(o.ticker, o.side, o.qty, o.limit_price, cid))
        elif o.dollars is not None:
            results.append(client.buy_notional(o.ticker, o.dollars, cid))
        elif o.qty is not None:
            results.append(client.sell_qty(o.ticker, o.qty, cid))
        else:
            results.append(client.trade_shares(o.ticker, o.side, o.shares, cid))
        payload["results"] = [r.model_dump() for r in results]
        ledger.write_plan(session, payload)
    rejected = sum(r.status == "rejected" for r in results)
    return done("submitted", f"{len(results)} orders, {rejected} rejected", target=target, orders=orders, results=results,
                **trend_info)


def _last_fresh_satellite(ledger: LiveLedger) -> dict[str, float]:
    for _, plan in reversed(ledger.satellite_plans()):
        if plan.get("paper_fresh"):
            return plan.get("satellite", {})
    return {}
