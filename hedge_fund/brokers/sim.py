"""SimBroker — deterministic simulated broker for backtests.

Fills every order completely, exactly at the order's reference price. That
determinism is the point: given the same orders, a backtest replays to the
same book. Costs (a per-fill commission and a short-borrow accrual) come out
of cash; fill prices stay exact so replays remain deterministic.

Margin is not modeled: cash may go negative and stays visible. With an
unlevered mandate (gross_target <= 1), sells-before-buys ordering, and
floor-toward-zero sizing, a long book won't get there — but nothing here
pretends to enforce it.
"""

from __future__ import annotations

from math import isfinite

from hedge_fund.brokers.models import Fill, Order, Position


class SimBroker:
    """In-memory broker: signed positions plus a cash balance."""

    def __init__(self, cash: float, commission_bps: float = 0.0) -> None:
        self._cash = cash
        self._shares: dict[str, int] = {}
        self._commission_rate = commission_bps / 10_000
        self._costs = 0.0

    def positions(self) -> dict[str, Position]:
        return {
            t: Position(ticker=t, shares=s)
            for t, s in self._shares.items()
            if s != 0
        }

    def cash(self) -> float:
        return self._cash

    def costs(self) -> float:
        """Total commission and borrow charged so far, in dollars."""
        return self._costs

    def place_order(self, order: Order) -> Fill:
        if not isfinite(order.price) or order.price <= 0:
            raise ValueError(
                f"cannot fill {order.ticker} at price {order.price} — "
                "the caller must price every order"
            )

        if order.side == "buy":
            self._shares[order.ticker] = self._shares.get(order.ticker, 0) + order.quantity
            self._cash -= order.quantity * order.price
        else:
            self._shares[order.ticker] = self._shares.get(order.ticker, 0) - order.quantity
            self._cash += order.quantity * order.price
        self._charge(order.quantity * order.price * self._commission_rate)

        if self._shares[order.ticker] == 0:
            del self._shares[order.ticker]

        return Fill(
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
        )

    def accrue_borrow(self, marks: dict[str, float], days: int, borrow_bps_annual: float) -> float:
        """Charge the borrow fee on every short for `days` calendar days at
        `marks`; returns the fee. `marks` must price every short."""
        short_notional = sum(-s * marks[t] for t, s in self._shares.items() if s < 0)
        fee = short_notional * borrow_bps_annual / 10_000 * days / 365
        self._charge(fee)
        return fee

    def _charge(self, fee: float) -> None:
        self._cash -= fee
        self._costs += fee
