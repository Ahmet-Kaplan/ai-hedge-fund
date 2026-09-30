"""Ledger: cash, positions and every cash flow, in real (as-of) units.

Positions are held in the units that actually traded on the day (real
shares, lots, contracts). Corporate actions are applied at the start of the
session they take effect, before that session's trading and valuation:

    split     quantity *= factor on the split's effective session
    dividend  cash += quantity_held * dividend_per_share on the ex-date
              (shorts pay it); only positions held into the ex-date count

Both come from the market panel's as-of view of that session, so nothing is
known before its date. Every cash movement is kept as a `CashFlow` so NAV can
be reconciled to fills + costs + dividends exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from hedge_fund.core.instruments import InstrumentRegistry
from hedge_fund.core.orders import FillEvent


@dataclass(frozen=True)
class CashFlow:
    session: str
    kind: Literal["trade", "commission", "dividend", "deposit"]
    symbol: str | None
    amount: float                 # + into cash, - out of cash


@dataclass
class Ledger:
    cash: float
    instruments: InstrumentRegistry = field(default_factory=InstrumentRegistry)
    positions: dict[str, float] = field(default_factory=dict)
    flows: list[CashFlow] = field(default_factory=list)
    fills: list[FillEvent] = field(default_factory=list)
    allow_short: bool = False

    def __post_init__(self) -> None:
        self.initial_cash = self.cash
        self.flows.append(CashFlow("", "deposit", None, self.cash))

    # ------------------------------------------------------------------

    def quantity(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def apply_fill(self, fill: FillEvent) -> None:
        inst = self.instruments.get(fill.symbol)
        new_qty = self.quantity(fill.symbol) + fill.signed_quantity
        if new_qty < -1e-9 and not (self.allow_short and inst.shortable):
            raise ValueError(f"{fill.symbol}: fill would open a short position; shorting is not allowed")
        gross = inst.notional(fill.quantity, fill.price)
        trade_amount = -gross if fill.side == "buy" else gross
        self.cash += trade_amount - fill.commission
        self.flows.append(CashFlow(fill.session, "trade", fill.symbol, trade_amount))
        if fill.commission:
            self.flows.append(CashFlow(fill.session, "commission", fill.symbol, -fill.commission))
        if abs(new_qty) < 1e-9:
            self.positions.pop(fill.symbol, None)
        else:
            self.positions[fill.symbol] = new_qty
        self.fills.append(fill)

    def apply_corporate_actions(self, view) -> list[CashFlow]:
        """Splits then dividends effective on `view.session`, for held names."""
        session, out = view.session, []
        held = [s for s in sorted(self.positions) if s in view.tickers]
        if not held:
            return out
        splits = view.bars("split", tickers=held, lookback=1)
        divs = view.bars("dividend", tickers=held, lookback=1)
        if not len(splits) or splits.index[-1] != session:
            return out
        for s in held:
            factor = float(splits[s].iloc[-1])
            if math.isfinite(factor) and factor > 0 and factor != 1.0:
                self.positions[s] = self.positions[s] * factor
            dps = float(divs[s].iloc[-1])
            if math.isfinite(dps) and dps:
                amount = self.positions[s] * dps * self.instruments.get(s).multiplier
                self.cash += amount
                flow = CashFlow(session, "dividend", s, amount)
                self.flows.append(flow)
                out.append(flow)
        return out

    # ------------------------------------------------------------------

    def market_value(self, marks: dict[str, float]) -> float:
        total = 0.0
        for s, q in self.positions.items():
            if s not in marks or not math.isfinite(marks[s]):
                raise ValueError(f"{s}: no mark to value the position")
            total += self.instruments.get(s).notional(q, marks[s])
        return total

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + self.market_value(marks)

    def total(self, kind: str) -> float:
        return sum(f.amount for f in self.flows if f.kind == kind)

    def reconcile(self) -> float:
        """Cash implied by the flow journal minus actual cash (should be 0)."""
        return sum(f.amount for f in self.flows) - self.cash
