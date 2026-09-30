"""Instruments — what can be traded, independent of any broker or data vendor.

The engine reasons in instruments, not tickers: an equity share, an ETF, an FX
pair, a futures contract and a crypto coin differ in quantity rules (whole
shares, fractional units, lots, contracts), contract multipliers, quote
currency and trading calendar. Everything downstream (sizing, execution,
risk) asks the instrument how to round a quantity and what a price is worth.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    FX = "fx"
    FUTURE = "future"
    CRYPTO = "crypto"
    OTHER = "other"


class Instrument(BaseModel):
    """A tradable instrument and its quantity/price conventions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    asset_class: AssetClass = AssetClass.EQUITY
    currency: str = "USD"
    multiplier: float = Field(1.0, gt=0, description="notional per unit per 1.0 of price (futures)")
    quantity_step: float = Field(1.0, gt=0, description="smallest tradable increment (1 = whole units)")
    min_quantity: float = Field(0.0, ge=0, description="smallest order size accepted by the venue")
    min_notional: float = Field(0.0, ge=0, description="smallest order value accepted by the venue")
    tick_size: float = Field(0.01, gt=0)
    shortable: bool = True
    calendar: str = "XNYS"

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("symbol must not be empty")
        return v

    @property
    def fractional(self) -> bool:
        return self.quantity_step < 1.0

    def notional(self, quantity: float, price: float) -> float:
        return quantity * price * self.multiplier

    def round_quantity(self, quantity: float) -> float:
        """Round toward zero to a whole number of quantity steps.

        Never rounds up: sizing must not overshoot a target. Returns 0.0 when
        the result is below the venue minimum.
        """
        if not math.isfinite(quantity):
            raise ValueError("quantity must be finite")
        steps = math.floor(abs(quantity) / self.quantity_step + 1e-9)
        q = round(steps * self.quantity_step, 10)
        if q < self.min_quantity:
            return 0.0
        return math.copysign(q, quantity) if q else 0.0

    def quantity_for_notional(self, notional: float, price: float) -> float:
        if price <= 0 or not math.isfinite(price):
            raise ValueError(f"{self.symbol}: price must be finite and positive")
        return self.round_quantity(notional / (price * self.multiplier))


def equity(symbol: str, *, fractional: bool = False, **kw) -> Instrument:
    """A US-listed share (or ETF with asset_class=ETF); fractional if the broker allows."""
    return Instrument(symbol=symbol, quantity_step=0.000001 if fractional else 1.0, **kw)


class InstrumentRegistry:
    """Symbol -> Instrument. Unknown symbols default to whole-share US equities."""

    def __init__(self, instruments: list[Instrument] | None = None, *, default_fractional: bool = False) -> None:
        self._by_symbol = {i.symbol: i for i in instruments or []}
        self._fractional = default_fractional

    def get(self, symbol: str) -> Instrument:
        key = symbol.strip().upper()
        if key not in self._by_symbol:
            self._by_symbol[key] = equity(key, fractional=self._fractional)
        return self._by_symbol[key]

    def add(self, instrument: Instrument) -> None:
        self._by_symbol[instrument.symbol] = instrument

    def __contains__(self, symbol: str) -> bool:
        return symbol.strip().upper() in self._by_symbol
