"""Turn a target book into live orders, and refuse orders that break the account's limits.

Buys are dollar amounts (fractional shares), funded only by cash already in
the account. Long sells are fractional quantities. Shorts and covers are
whole shares. Sells are listed first.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel

_TOLERANCE = 0.005   # 0.5% of equity: price moves between sizing and fill


class PlannedOrder(BaseModel):
    ticker: str
    side: Literal["buy", "sell"]
    dollars: float | None = None   # notional buy
    qty: float | None = None       # fractional long sell
    shares: int | None = None      # whole-share short / cover


def size_orders(
    holdings: dict[str, float], cash: float, marks: dict[str, float], target: dict[str, float], *,
    rebalance: bool, min_order_usd: float, min_trade_pct: float, cash_buffer_pct: float,
) -> list[PlannedOrder]:
    """Orders that move `holdings` toward `target` (weights of equity).

    On a rebalance day every name is realigned (dust under the larger of
    min_order_usd and min_trade_pct of equity skipped, closes never skipped).
    Otherwise only uninvested cash is put to work, toward the most
    underweight names. Buys never exceed cash × (1 − cash_buffer_pct).
    """
    equity = cash + sum(q * marks[t] for t, q in holdings.items())
    if equity <= 0:
        return []
    floor = max(min_order_usd, min_trade_pct * equity)
    sells: list[PlannedOrder] = []
    covers: list[PlannedOrder] = []
    wanted: dict[str, float] = {}
    for t in sorted(set(target) | set(holdings)):
        weight, q, mark = target.get(t, 0.0), holdings.get(t, 0.0), marks[t]
        delta = weight * equity - q * mark
        if weight < 0 or q < 0:                                  # short side: whole shares
            if not rebalance:
                continue
            target_shares = math.trunc(weight * equity / mark) if weight < 0 else 0
            change = target_shares - round(q)
            if change < 0 and -change * mark >= floor:
                sells.append(PlannedOrder(ticker=t, side="sell", shares=-change))
            elif change > 0 and (target_shares == 0 or change * mark >= floor):
                covers.append(PlannedOrder(ticker=t, side="buy", shares=change))
            continue
        if delta < 0 and q > 0 and rebalance:
            closing = weight <= 0
            if closing or -delta >= floor:
                sells.append(PlannedOrder(ticker=t, side="sell", qty=round(q if closing else min(q, -delta / mark), 9)))
        elif delta > 0 and (not rebalance or delta >= floor):
            wanted[t] = delta

    budget = cash * (1 - cash_buffer_pct) - sum(o.shares * marks[o.ticker] for o in covers)
    total = sum(wanted.values())
    buys: list[PlannedOrder] = []
    if budget >= min_order_usd and total > 0:
        scale = min(1.0, budget / total)
        for t in sorted(wanted):
            dollars = math.floor(wanted[t] * scale * 100) / 100
            if dollars >= min_order_usd:
                buys.append(PlannedOrder(ticker=t, side="buy", dollars=dollars))
    return sells + covers + buys


def check_orders(
    orders: list[PlannedOrder], holdings: dict[str, float], cash: float, marks: dict[str, float],
    core_ticker: str, *, shorts_ok: bool, max_name: float,
) -> None:
    """Raise if these orders would overspend cash, short without permission, or oversize a name."""
    equity = cash + sum(q * marks[t] for t, q in holdings.items())
    spend = sum(o.dollars for o in orders if o.dollars) + sum(o.shares * marks[o.ticker] for o in orders if o.shares and o.side == "buy")
    if spend > cash + 1e-6:
        raise ValueError(f"orders spend ${spend:,.2f} but the account has ${cash:,.2f} cash")
    projected = dict(holdings)
    for o in orders:
        if o.dollars:
            change = o.dollars / marks[o.ticker]
        elif o.qty:
            change = -o.qty
        else:
            change = o.shares if o.side == "buy" else -o.shares
        projected[o.ticker] = projected.get(o.ticker, 0.0) + change
    for t, q in projected.items():
        if q < -1e-9 and not shorts_ok:
            raise ValueError(f"{t}: would short without shorts enabled (margin and ${2000:,}+ equity)")
        if t != core_ticker and equity > 0 and abs(q * marks[t]) / equity > max_name + _TOLERANCE:
            raise ValueError(f"{t}: would be {abs(q * marks[t]) / equity:.1%} of the account (max {max_name:.0%})")
