"""PaperBroker — a TradingVenue with a real order lifecycle and no real money.

Orders are accepted (or rejected) on submit and filled when the broker is
advanced to a later session with `process(session)`, through the same
`SimulatedExecution` cost/liquidity model the backtester uses:

    submit  -> ACCEPTED | REJECTED     (validation, cash and short checks)
    process -> FILLED | PARTIALLY_FILLED (gtc keeps the rest open)
               | EXPIRED (day order remainder) | REJECTED (no price / liquidity)
    cancel  -> CANCELLED (any non-terminal order)

Submitting the same client order id twice returns the existing order — the
duplicate-order guard. `reconcile` compares the broker's books with an
external ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hedge_fund.core.orders import OrderRequest, OrderState, OrderStatus
from hedge_fund.systematic.ledger import Ledger


class BrokerUnavailable(ConnectionError):
    """Transient venue failure (network, gateway down). Safe to retry."""


@dataclass
class Reconciliation:
    ok: bool
    cash_difference: float
    position_differences: dict[str, float] = field(default_factory=dict)


class PaperBroker:
    live = False

    def __init__(self, cash: float, execution, *, allow_short: bool = False) -> None:
        self.execution = execution
        self.book = Ledger(cash=cash, instruments=execution.instruments, allow_short=allow_short)
        self.orders: dict[str, OrderState] = {}

    # -- TradingVenue ------------------------------------------------------

    def cash(self) -> float:
        return self.book.cash

    def positions(self) -> dict[str, float]:
        return dict(self.book.positions)

    def order(self, client_order_id: str) -> OrderState:
        return self.orders[client_order_id]

    def open_orders(self) -> list[OrderState]:
        return [o for o in self.orders.values() if not o.status.terminal]

    def submit(self, order: OrderRequest) -> OrderState:
        order = order.with_client_id()
        if order.client_order_id in self.orders:            # idempotent resubmission
            return self.orders[order.client_order_id]
        state = OrderState(request=order)
        self.orders[order.client_order_id] = state
        reason = self._validate(order)
        state.transition(OrderStatus.REJECTED if reason else OrderStatus.ACCEPTED, reason)
        return state

    def cancel(self, client_order_id: str) -> OrderState:
        state = self.orders[client_order_id]
        if not state.status.terminal:
            state.transition(OrderStatus.CANCELLED, "cancelled by client")
        return state

    # -- simulation clock ---------------------------------------------------

    def process(self, session: str) -> list[OrderState]:
        """Fill open orders decided before *session* at its execution event."""
        touched = []
        for state in sorted(self.open_orders(), key=lambda s: (s.request.side != "sell", s.request.symbol)):
            req = state.request
            if session <= req.decision_session:
                continue
            remaining = req.model_copy(update={"quantity": state.remaining})
            if req.side == "sell" and not self.book.allow_short:
                held = self.book.quantity(req.symbol)
                if held <= 0:
                    state.transition(OrderStatus.CANCELLED if state.filled_quantity else OrderStatus.EXPIRED,
                                     "nothing left to sell")
                    touched.append(state)
                    continue
                remaining = remaining.model_copy(update={"quantity": min(remaining.quantity, held)})
            sim = self.execution.execute(remaining, session)
            if sim.status is OrderStatus.REJECTED or not sim.fills:
                if state.status is OrderStatus.ACCEPTED:
                    state.transition(OrderStatus.REJECTED, sim.reject_reason)
                else:
                    state.transition(OrderStatus.EXPIRED, sim.reject_reason)
                touched.append(state)
                continue
            for f in sim.fills:
                inst = self.book.instruments.get(f.symbol)
                if f.side == "buy" and inst.notional(f.quantity, f.price) + f.commission > self.book.cash + 1e-9:
                    state.transition(OrderStatus.EXPIRED if state.filled_quantity else OrderStatus.CANCELLED,
                                     "insufficient cash at execution")
                    break
                self.book.apply_fill(f.model_copy(update={"client_order_id": req.client_order_id}))
                state.add_fill(f.model_copy(update={"client_order_id": req.client_order_id}))
            if state.status is OrderStatus.PARTIALLY_FILLED and req.time_in_force == "day":
                state.transition(OrderStatus.EXPIRED, "day order remainder expired")
            touched.append(state)
        return touched

    def _validate(self, order: OrderRequest) -> str | None:
        if order.order_type not in ("market", "moo", "moc"):
            return f"order type {order.order_type} not supported by the paper broker"
        if order.side == "sell" and not self.book.allow_short and self.book.quantity(order.symbol) <= 0:
            return "short selling is not allowed"
        if order.side == "buy":
            inst = self.book.instruments.get(order.symbol)
            if inst.notional(order.quantity, order.reference_price) > self.book.cash * 1.10:
                return "insufficient cash"
        return None

    # -- reconciliation -------------------------------------------------------

    def reconcile(self, positions: dict[str, float], cash: float, *, tolerance: float = 1e-6) -> Reconciliation:
        names = set(positions) | set(self.book.positions)
        diffs = {s: self.book.quantity(s) - positions.get(s, 0.0) for s in names
                 if abs(self.book.quantity(s) - positions.get(s, 0.0)) > tolerance}
        cash_diff = self.book.cash - cash
        return Reconciliation(ok=not diffs and abs(cash_diff) <= max(tolerance, 0.01), cash_difference=cash_diff,
                              position_differences=diffs)
