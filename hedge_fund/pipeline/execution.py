"""Execution — turn target weights into delta orders.

Targets are the complete statement of the desired book: any held name absent
from the targets has an implicit target of zero, so close orders fall out of
the same arithmetic as everything else. Pure function; the broker does the
filling.

Later: Almgren-Chriss optimal execution, market impact, fill probability.
"""

from __future__ import annotations

from hedge_fund.brokers.models import Order, Position


def build_orders(
    target_weights: dict[str, float],
    positions: dict[str, Position],
    marks: dict[str, float],
    equity: float,
    min_trade_pct: float = 0.0,
) -> list[Order]:
    """Diff the target book against the broker's current book.

    Sizing: target_shares = int(weight * equity / mark) — floor toward zero,
    never overshoot the target; sub-share dust stays in cash and is
    re-evaluated next cycle. Orders below one share are not emitted.

    Ordering: all sells first, then buys, alphabetical within each group —
    deterministic, and sells free the cash that buys consume within the
    same cycle.

    Dust: an order that opens or adds to a position (moves it away from
    zero) is skipped when its notional is below min_trade_pct of equity;
    a trade that reduces or closes a position is never skipped, so the
    book can always move back inside its limits.

    A KeyError on marks here means a pipeline bug upstream (run_cycle prices
    every tradeable and held name before calling this) — let it raise.
    """
    sells: list[Order] = []
    buys: list[Order] = []

    for ticker in sorted(set(target_weights) | set(positions)):
        mark = marks[ticker]
        target_shares = int(target_weights.get(ticker, 0.0) * equity / mark)
        current_shares = positions[ticker].shares if ticker in positions else 0
        delta = target_shares - current_shares
        if delta == 0:
            continue
        adds_exposure = abs(target_shares) > abs(current_shares) and target_shares * current_shares >= 0
        if adds_exposure and abs(delta) * mark < min_trade_pct * equity:
            continue
        order = Order(
            ticker=ticker,
            side="buy" if delta > 0 else "sell",
            quantity=abs(delta),
            price=mark,
        )
        (buys if delta > 0 else sells).append(order)

    return sells + buys
