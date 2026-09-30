"""Plan one live rebalance: the backtest's decision path, sized against a real book.

Same assessment, blend, risk clamp and order arithmetic as execute_decision;
the differences are only where the book comes from (the broker account, passed
in) and the sizing price (the cutoff close — the execution close is still in
the future when orders go out).
"""

from __future__ import annotations

from math import isfinite

from pydantic import BaseModel

from hedge_fund.brokers.models import Order, Position
from hedge_fund.data.protocol import DataClient
from hedge_fund.data.sessions import previous_day
from hedge_fund.fund import Fund
from hedge_fund.pipeline.execution import build_orders
from hedge_fund.pipeline.models import DecisionRecord
from hedge_fund.pipeline.run_cycle import _mark_prices, assess_fund, check_projected_book


class LivePlan(BaseModel):
    session: str                        # execution session (orders fill at its close)
    cutoff: str                         # data cutoff: the day before the session
    decision: DecisionRecord | None     # None when flattening
    marks: dict[str, float]             # reference closes used for sizing
    equity: float
    cash: float
    positions: dict[str, int]           # the book the plan started from
    targets: dict[str, float]
    orders: list[Order]
    flatten: bool = False


def plan_rebalance(
    fund: Fund, universe: list[str], positions: dict[str, int], cash: float,
    data_client: DataClient, session: str, *, flatten: bool = False,
) -> LivePlan:
    """Orders that move `positions` to the fund's targets (or to flat) at `session`'s close."""
    cutoff = previous_day(session)
    decision = None if flatten else assess_fund(fund, cutoff, data_client, universe)
    targets = {} if decision is None else {t: w for t, w in decision.final_weights.items() if w != 0}

    marks = {t: decision.marks[t] for t in targets} if decision is not None else {}
    missing = sorted(t for t in positions if t not in marks)
    extra, skipped = _mark_prices(missing, cutoff, data_client)
    if skipped:
        raise ValueError(f"cannot price held names {', '.join(s.ticker for s in skipped)} as of {cutoff}")
    marks.update(extra)

    equity = cash + sum(shares * marks[t] for t, shares in positions.items())
    if not isfinite(equity) or equity <= 0:
        raise ValueError(f"{fund.spec.name}: equity {equity} must be finite and positive to size orders")
    held = {t: Position(ticker=t, shares=s) for t, s in positions.items() if s}
    orders = build_orders(targets, held, marks, equity)
    check_projected_book(orders, dict(positions), marks, equity, fund.spec.risk)
    return LivePlan(
        session=session, cutoff=cutoff, decision=decision, marks=marks, equity=equity,
        cash=cash, positions=dict(positions), targets=targets, orders=orders, flatten=flatten,
    )
