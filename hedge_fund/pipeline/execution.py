"""Execution — turn target weights into delta orders.

Targets are the complete statement of the desired book: any held name absent
from the targets has an implicit target of zero, so close orders fall out of
the same arithmetic as everything else. Pure function; the broker does the
filling.

Later: Almgren-Chriss optimal execution, market impact, fill probability.
"""

from __future__ import annotations

import hashlib

from hedge_fund.brokers.models import Order, Position

# Venues cap this (Alpaca: 48 characters). Keep the alphabet to characters
# every venue accepts, so an id is never the reason an order is rejected.
_MAX_CLIENT_ORDER_ID = 48


def client_order_id(fund: str, session: str, sequence: int, order: Order) -> str:
    """A deterministic, venue-safe id for one order.

    Same fund + session + sequence + ticker + side always yields the same id,
    so replaying an execution cannot place a second order: a venue that
    supports client order ids hands back the first one instead, and the
    in-process brokers recognize it too. Different funds never collide (the
    fund is hashed in), and the readable prefix keeps logs greppable.
    """
    digest = hashlib.sha256(
        f"{fund}|{session}|{sequence}|{order.ticker}|{order.side}".encode()
    ).hexdigest()[:12]
    tail = f"{session}-{sequence:03d}-{order.side[0]}-{digest}"
    ticker = "".join(c for c in order.ticker.upper() if c.isalnum())[: max(1, 47 - len(tail))]
    return f"{ticker}-{tail}"[:_MAX_CLIENT_ORDER_ID]


def stamp_client_order_ids(fund: str, session: str, orders: list[Order]) -> list[Order]:
    """Return *orders* with `client_order_id` set, in order.

    The sequence is the order's position in the list, and `build_orders`
    emits deterministically (sells first, then buys, alphabetical within each
    group), so ids are stable across runs.
    """
    return [
        order.model_copy(update={"client_order_id": client_order_id(fund, session, i, order)})
        for i, order in enumerate(orders)
    ]


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

    `min_trade_pct` skips a trade worth less than that fraction of equity, but
    **only when it would grow a position**. Shrinking trades always go out: a
    cost control must never be a reason to hold risk you meant to shed, and a
    small trim is exactly what a cap or a drawdown brake asks for.

    Ordering: all sells first, then buys, alphabetical within each group —
    deterministic, and sells free the cash that buys consume within the
    same cycle.

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
        if min_trade_pct > 0 and abs(target_shares) > abs(current_shares):
            if abs(delta) * mark < min_trade_pct * equity:
                continue
        order = Order(
            ticker=ticker,
            side="buy" if delta > 0 else "sell",
            quantity=abs(delta),
            price=mark,
        )
        (buys if delta > 0 else sells).append(order)

    return sells + buys
