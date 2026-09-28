"""Broker data models — positions, orders, fills.

Two order verbs, not four: positions are signed share counts, so "short" is
just selling past zero and "cover" is buying back toward it. Long/short
labeling is a display concern, not an execution one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Commission(BaseModel):
    """What a fill costs to execute: a per-ticket charge plus a per-share rate.

    The two shapes real schedules combine. Both default to zero, so a broker
    nobody configured prices exactly as it did before commissions existed and
    an existing backtest replays to the same book, to the cent.

    Ported from PR #19, which had this right.
    """

    per_trade: float = Field(default=0.0, ge=0.0)
    per_share: float = Field(default=0.0, ge=0.0)

    def charge(self, quantity: int) -> float:
        return self.per_trade + self.per_share * quantity


class Position(BaseModel):
    """Signed share count in one ticker. Negative = short.

    `cost_basis` is the weighted-average entry price and stays positive on both
    sides — the sign already lives in `shares`, and carrying it twice would only
    create a way for the two to disagree.

    `None` means *unknown*, not zero: a book seeded from a receipt knows the
    share counts but not what they were bought for, and reporting 0.0 would
    manufacture a profit on every closing trade. Fill.realized_pnl is None in
    that case for the same reason.
    """

    ticker: str
    shares: int
    cost_basis: float | None = None


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
    """An executed order at its actual fill price.

    `commission` and `realized_pnl` ride along because the Fill is what the
    caller keeps: re-deriving either afterwards would mean reconstructing the
    basis the broker already held at fill time, and after a crossing order that
    basis is gone.

    `realized_pnl` is None — not 0.0 — when the closed shares had no known
    basis. "This trade made nothing" and "nobody can say what this trade made"
    are different statements, and the audit trail should not confuse them.
    """

    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    price: float
    commission: float = 0.0
    realized_pnl: float | None = None
