"""Broker data models — positions, orders, fills.

Two order verbs, not four: positions are signed share counts, so "short" is
just selling past zero and "cover" is buying back toward it. Long/short
labeling is a display concern, not an execution one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Position(BaseModel):
    """Signed share count in one ticker. Negative = short."""

    ticker: str
    shares: int


class Order(BaseModel):
    """An instruction to trade. `price` is the reference price the caller
    computed (the as-of close): SimBroker and PaperBroker fill at that mark
    (PaperBroker may delay the fill), a live broker fills at its own quote —
    the Fill always carries the truth.

    `client_order_id` is the caller's deterministic id for this instruction
    (see `hedge_fund.pipeline.execution.stamp_client_order_ids`). A venue that
    supports it returns the *existing* order instead of creating a second one,
    which is what makes a retry after an ambiguous failure safe. Venues that
    ignore it still honour the id in-process (`PaperBroker`, `SimBroker`).
    """

    ticker: str
    side: Literal["buy", "sell"]
    quantity: int = Field(gt=0)
    price: float
    client_order_id: str | None = None


class Fill(BaseModel):
    """An executed order at its actual fill price."""

    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    price: float
