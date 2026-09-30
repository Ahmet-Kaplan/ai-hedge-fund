"""Orders and fills with an explicit lifecycle, shared by simulation, paper and live.

The older `hedge_fund.brokers.models.Order/Fill` (whole shares, fill-or-raise)
stay for the Buffett pipeline. These richer models carry what a real venue
returns — partial fills, rejections, costs — and the audit fields that let a
trade be explained after the fact.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OrderStatus(str, Enum):
    NEW = "new"                    # created, not yet sent
    ACCEPTED = "accepted"          # venue acknowledged
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    @property
    def terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED)


_ALLOWED = {
    OrderStatus.NEW: {OrderStatus.ACCEPTED, OrderStatus.REJECTED},
    OrderStatus.ACCEPTED: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED,
                           OrderStatus.REJECTED, OrderStatus.EXPIRED},
    OrderStatus.PARTIALLY_FILLED: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED,
                                   OrderStatus.EXPIRED},
}


class InvalidTransition(ValueError):
    pass


class OrderRequest(BaseModel):
    """What the engine wants to trade, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    side: Literal["buy", "sell"]
    quantity: float = Field(gt=0)
    order_type: Literal["market", "limit", "moc", "moo"] = "market"
    limit_price: float | None = None
    time_in_force: Literal["day", "gtc", "opg", "cls"] = "day"
    decision_session: str = Field(description="session whose data produced the decision")
    reference_price: float = Field(gt=0, description="price the decision assumed (as-of close)")
    strategy: str = ""
    reason: dict = Field(default_factory=dict, description="audit: signals, config hash, risk decision")
    client_order_id: str = ""

    def idempotency_key(self) -> str:
        """Same decision, same instrument, same side => same key (duplicate guard)."""
        payload = json.dumps({"s": self.symbol, "side": self.side, "d": self.decision_session,
                              "st": self.strategy, "q": round(self.quantity, 8)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    def with_client_id(self) -> OrderRequest:
        return self if self.client_order_id else self.model_copy(update={"client_order_id": self.idempotency_key()})


class FillEvent(BaseModel):
    """One execution against an order, with its cost breakdown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    client_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float = Field(gt=0)
    price: float = Field(gt=0, description="all-in execution price incl. spread and impact")
    session: str
    mid_price: float = Field(gt=0, description="reference price at the execution event (open/close)")
    commission: float = 0.0
    spread_cost: float = 0.0
    impact_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.commission + self.spread_cost + self.impact_cost

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side == "buy" else -self.quantity


class OrderState(BaseModel):
    """Mutable lifecycle view of one order at a broker."""

    request: OrderRequest
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    fills: list[FillEvent] = Field(default_factory=list)
    reject_reason: str | None = None

    @property
    def remaining(self) -> float:
        return max(self.request.quantity - self.filled_quantity, 0.0)

    def transition(self, new: OrderStatus, reason: str | None = None) -> None:
        if new not in _ALLOWED.get(self.status, set()):
            raise InvalidTransition(f"{self.status.value} -> {new.value} is not allowed")
        self.status = new
        if reason:
            self.reject_reason = reason

    def add_fill(self, fill: FillEvent) -> None:
        if self.status not in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
            raise InvalidTransition(f"cannot fill an order that is {self.status.value}")
        if fill.quantity > self.remaining + 1e-9:
            raise InvalidTransition("fill exceeds the remaining quantity")
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.transition(OrderStatus.FILLED if self.remaining <= 1e-9 else OrderStatus.PARTIALLY_FILLED)
