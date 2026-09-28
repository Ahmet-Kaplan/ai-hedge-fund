"""SimBroker — deterministic simulated broker for backtests.

Fills every order completely, exactly at the order's reference price. That
determinism is the point: given the same orders, a backtest replays to the
same book. Commissions are charged inside place_order, where they change fills
without touching the pipeline; slippage stays a declared future addition in the
same place.

Bookkeeping — cash, signed shares, weighted-average cost basis, realized P&L —
lives in `PositionBook`, shared with PaperBroker so the two offline venues
cannot disagree about what a fill cost or what it earned.

Margin is not modeled: cash may go negative and stays visible. With an
unlevered mandate (gross_target <= 1), sells-before-buys ordering, and
floor-toward-zero sizing, a long book won't get there — but nothing here
pretends to enforce it.
"""

from __future__ import annotations

from math import isfinite

from hedge_fund.brokers.book import PositionBook
from hedge_fund.brokers.models import Commission, Fill, Order, Position


class SimBroker:
    """In-memory broker: signed positions carrying a weighted-average cost
    basis, plus a cash balance.

    Backtests always construct with cash only and let place_order accumulate
    state across ticks. Live-clock paper runs use PaperBroker instead.
    """

    venue = "sim"

    def __init__(
        self,
        cash: float,
        positions: dict[str, int] | None = None,
        commission: Commission | None = None,
        cost_basis: dict[str, float] | None = None,
    ) -> None:
        self._book = PositionBook(cash, positions, commission, cost_basis)
        # client_order_id -> Fill. Replayed instructions return the original
        # fill instead of moving the book twice.
        self._receipts: dict[str, Fill] = {}

    def positions(self) -> dict[str, Position]:
        return self._book.positions()

    def cash(self) -> float:
        return self._book.cash

    def realized_pnl(self) -> float:
        """Cumulative realized P&L, gross of commission."""
        return self._book.realized_pnl

    def place_order(self, order: Order) -> Fill:
        if not isfinite(order.price) or order.price <= 0:
            raise ValueError(
                f"cannot fill {order.ticker} at price {order.price} — "
                "the caller must price every order"
            )

        cid = order.client_order_id
        if cid is not None and cid in self._receipts:
            return self._receipts[cid]

        movement = self._book.apply(order.ticker, order.side, order.quantity, order.price)
        fill = Fill(
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            commission=movement.commission,
            realized_pnl=movement.realized_pnl,
        )
        if cid is not None:
            self._receipts[cid] = fill
        return fill

    def receipts(self) -> dict[str, Fill]:
        """client_order_id -> Fill for everything this broker has executed."""
        return dict(self._receipts)
