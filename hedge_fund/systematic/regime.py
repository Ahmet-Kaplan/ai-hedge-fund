"""Market regime — a transparent feature/filter, not a forecast.

`SimpleRegime` labels the market from one reference instrument (the
benchmark), using only the as-of view:

    trend        up / down / range   close vs. its long moving average, with a band
    volatility   high / low          current realized vol vs. its own trailing history
    risk         on / off            off when drawn down from the trailing high by more
                                     than `drawdown_off`, or trending down in high vol

Any detector implementing `classify(view) -> RegimeState` can replace it.
"""

from __future__ import annotations

import math
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class RegimeState(BaseModel):
    model_config = ConfigDict(frozen=True)

    session: str
    trend: Literal["up", "down", "range", "unknown"] = "unknown"
    volatility: Literal["high", "low", "unknown"] = "unknown"
    risk: Literal["on", "off", "unknown"] = "unknown"
    features: dict[str, float] = Field(default_factory=dict)

    def labels(self) -> set[str]:
        return {f"trend:{self.trend}", f"vol:{self.volatility}", f"risk:{self.risk}"}


class RegimeDetector(Protocol):
    def classify(self, view) -> RegimeState: ...


class SimpleRegimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reference: str = "SPY"
    trend_window: int = Field(200, ge=20)
    trend_band: float = Field(0.02, ge=0)
    vol_window: int = Field(20, ge=5)
    vol_history: int = Field(252, ge=40)
    high_vol_quantile: float = Field(0.7, gt=0, lt=1)
    drawdown_window: int = Field(252, ge=20)
    drawdown_off: float = Field(0.10, gt=0, lt=1)


class SimpleRegime:
    def __init__(self, config: SimpleRegimeConfig | None = None) -> None:
        self.config = config or SimpleRegimeConfig()

    def classify(self, view) -> RegimeState:
        c = self.config
        need = max(c.trend_window, c.vol_history + c.vol_window, c.drawdown_window) + 1
        px = view.bars("close", tickers=[c.reference], lookback=need)[c.reference].dropna()
        feats: dict[str, float] = {}
        trend = vol_label = risk = "unknown"
        if len(px) >= c.trend_window:
            sma = float(px.tail(c.trend_window).mean())
            gap = float(px.iloc[-1] / sma - 1)
            feats["trend_gap"] = gap
            trend = "up" if gap > c.trend_band else "down" if gap < -c.trend_band else "range"
        rets = np.log(px).diff().dropna()
        if len(rets) >= c.vol_history + c.vol_window:
            rolling = rets.rolling(c.vol_window).std().dropna() * math.sqrt(252)
            current = float(rolling.iloc[-1])
            threshold = float(rolling.tail(c.vol_history).quantile(c.high_vol_quantile))
            feats.update(realized_vol=current, vol_threshold=threshold)
            vol_label = "high" if current > threshold else "low"
        if len(px) >= c.drawdown_window:
            dd = float(px.iloc[-1] / px.tail(c.drawdown_window).max() - 1)
            feats["drawdown"] = dd
            off = dd <= -c.drawdown_off or (trend == "down" and vol_label == "high")
            risk = "off" if off else "on"
        return RegimeState(session=view.session, trend=trend, volatility=vol_label, risk=risk, features=feats)
