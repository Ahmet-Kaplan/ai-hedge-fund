"""Momentum — the 12-1 month return, scaled by the stock's own volatility.

Stocks that outperformed over the past year tend to keep outperforming for
months (Jegadeesh & Titman 1993). The most recent month is skipped because
over weeks returns tend to reverse. Dividing by trailing volatility makes a
steady +30% count for more than a wild +30% and tames momentum's crashes
after sharp market rebounds (Barroso & Santa-Clara 2015).

Price-only and deterministic: no LLM, and backtestable over any window the
price history covers.
"""

from __future__ import annotations

import math
import statistics
from datetime import date as _date
from datetime import timedelta

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel

_TRADING_DAYS = 252


class MomentumModel(QuantModel):
    investment_approach = "long_short"

    def __init__(self, *, lookback_days: int = 252, skip_days: int = 21) -> None:
        if not 0 <= skip_days < lookback_days:
            raise ValueError("skip_days must be at least 0 and shorter than lookback_days")
        self._lookback = lookback_days
        self._skip = skip_days

    @property
    def name(self) -> str:
        return "momentum"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        start = (_date.fromisoformat(date) - timedelta(days=int(self._lookback * 1.6) + 10)).isoformat()
        closes = [bar.close for bar in data_client.get_prices(ticker, start, date) if bar.time[:10] <= date]
        if len(closes) < self._lookback + 1:
            return Signal(model_name=self.name, ticker=ticker, date=date, value=0.0,
                          reasoning=f"abstained: {len(closes)} daily closes, need {self._lookback + 1}",
                          metadata={"abstained": True})
        window = closes[-(self._lookback + 1):]
        momentum = window[-1 - self._skip] / window[0] - 1
        daily = [b / a - 1 for a, b in zip(window, window[1:])]
        volatility = statistics.stdev(daily) * math.sqrt(_TRADING_DAYS)
        value = math.tanh(momentum / volatility) if volatility > 0 else 0.0
        return Signal(
            model_name=self.name, ticker=ticker, date=date, value=value,
            reasoning=f"12-1 month return {momentum:+.1%} at {volatility:.0%} annualized volatility",
            components={"return_12_1": momentum, "volatility": volatility},
            metadata={"return_12_1": momentum, "volatility": volatility},
        )
