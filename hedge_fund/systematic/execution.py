"""Simulated execution: next-session fills with commission, spread and impact.

Timing rule (strict): an order decided on data of session D can only fill at
an execution event of a *later* session — the next session's open
(`NEXT_OPEN`) or close (`NEXT_CLOSE`). There is no implicit default in the
engine: the convention is a required argument, and a request to fill at or
before the decision session raises `LookAheadExecution`.

Prices come from the market panel's as-of view of the execution session, so
the fill price uses the split basis in force on that day (real shares).

Costs, each configurable:

    commission   max(min, per_share * qty + bps * notional)
    half-spread  half_spread_bps of the reference price, paid on every fill
    impact       impact_coef * sigma_daily * sqrt(order_notional / ADV_notional)
                 (square-root law); sigma and ADV use only sessions *before*
                 the execution session, so an open fill never peeks at that
                 day's volume or close.

Liquidity: an order may take at most `max_participation` of ADV notional.
The rest of a day order expires (partial fill). Zero/unknown ADV, an
untradable bar (halt, delisting) or a missing price rejects the order.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hedge_fund.core.instruments import InstrumentRegistry
from hedge_fund.core.orders import FillEvent, OrderRequest, OrderState, OrderStatus


class FillTiming(str, Enum):
    NEXT_OPEN = "next_open"
    NEXT_CLOSE = "next_close"


class LookAheadExecution(ValueError):
    """An order would fill at or before the session whose data decided it."""


class CostModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    commission_per_share: float = Field(0.0, ge=0)
    commission_bps: float = Field(0.0, ge=0)
    commission_min: float = Field(0.0, ge=0)
    half_spread_bps: float = Field(2.0, ge=0)
    impact_coef: float = Field(0.1, ge=0)
    max_participation: float = Field(0.10, gt=0, le=1.0)
    adv_lookback: int = Field(20, ge=1)
    vol_lookback: int = Field(20, ge=2)
    default_daily_vol: float = Field(0.02, gt=0, description="used when too little history exists")

    def commission(self, quantity: float, notional: float) -> float:
        raw = self.commission_per_share * abs(quantity) + self.commission_bps / 1e4 * abs(notional)
        return max(self.commission_min, raw) if quantity else 0.0


ZERO_COST = CostModel(half_spread_bps=0.0, impact_coef=0.0, max_participation=1.0)


class SimulatedExecution:
    """ExecutionModel over a MarketPanel. Deterministic: same inputs, same fills."""

    def __init__(self, panel, costs: CostModel, timing: FillTiming,
                 instruments: InstrumentRegistry | None = None) -> None:
        if not isinstance(timing, FillTiming):
            raise TypeError("timing must be an explicit FillTiming (next_open or next_close)")
        self.panel, self.costs, self.timing = panel, costs, timing
        self.instruments = instruments or InstrumentRegistry()

    # ------------------------------------------------------------------

    def liquidity(self, symbol: str, session: str) -> tuple[float, float]:
        """(ADV notional, daily vol) from sessions strictly before *session*."""
        prior = [s for s in self.panel.sessions_through(session) if s < session]
        if not prior:
            return 0.0, self.costs.default_daily_vol
        view = self.panel.as_of(prior[-1])
        close = view.bars("close", tickers=[symbol], lookback=max(self.costs.adv_lookback, self.costs.vol_lookback) + 1)[symbol]
        volume = view.bars("volume", tickers=[symbol], lookback=len(close))[symbol]
        dollar = (close * volume).dropna().tail(self.costs.adv_lookback)
        adv = float(dollar.mean()) if len(dollar) else 0.0
        rets = np.log(close.dropna()).diff().dropna().tail(self.costs.vol_lookback)
        vol = float(rets.std(ddof=1)) if len(rets) >= 2 else self.costs.default_daily_vol
        if not math.isfinite(vol) or vol <= 0:
            vol = self.costs.default_daily_vol
        return (adv if math.isfinite(adv) else 0.0), vol

    def reference_price(self, symbol: str, session: str) -> float:
        view = self.panel.as_of(session)
        if view.session != session:
            return float("nan")
        field = "open" if self.timing is FillTiming.NEXT_OPEN else "close"
        frame = view.bars(field, tickers=[symbol], lookback=1)
        return float(frame.iloc[-1, 0]) if len(frame) and frame.index[-1] == session else float("nan")

    def execute(self, order: OrderRequest, session: str) -> OrderState:
        if session <= order.decision_session:
            raise LookAheadExecution(
                f"{order.symbol}: decided on {order.decision_session}, cannot fill on {session}")
        order = order.with_client_id()
        state = OrderState(request=order)
        instrument = self.instruments.get(order.symbol)
        mid = self.reference_price(order.symbol, session)
        if not math.isfinite(mid) or mid <= 0:
            state.transition(OrderStatus.REJECTED, f"no tradable {self.timing.value} price on {session}")
            return state
        adv, vol = self.liquidity(order.symbol, session)
        if adv <= 0:
            state.transition(OrderStatus.REJECTED, "no liquidity history (zero ADV)")
            return state
        cap_units = self.costs.max_participation * adv / (mid * instrument.multiplier)
        quantity = instrument.round_quantity(min(order.quantity, cap_units))
        if quantity <= 0:
            state.transition(OrderStatus.REJECTED, "order below minimum size after liquidity cap")
            return state
        notional = instrument.notional(quantity, mid)
        if notional < instrument.min_notional:
            state.transition(OrderStatus.REJECTED, f"notional {notional:.2f} below venue minimum")
            return state

        spread = self.costs.half_spread_bps / 1e4
        impact = self.costs.impact_coef * vol * math.sqrt(notional / adv)
        sign = 1.0 if order.side == "buy" else -1.0
        price = mid * (1.0 + sign * (spread + impact))
        if price <= 0:
            state.transition(OrderStatus.REJECTED, "cost model produced a non-positive price")
            return state
        state.transition(OrderStatus.ACCEPTED)
        state.add_fill(FillEvent(
            client_order_id=order.client_order_id, symbol=order.symbol, side=order.side,
            quantity=quantity, price=price, session=session, mid_price=mid,
            commission=self.costs.commission(quantity, notional),
            spread_cost=notional * spread, impact_cost=notional * impact,
        ))
        if state.status is OrderStatus.PARTIALLY_FILLED:
            state.transition(OrderStatus.EXPIRED, "remainder above participation limit")
        return state
