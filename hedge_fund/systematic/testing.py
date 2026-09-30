"""In-memory synthetic market for tests: vendor-shaped, deterministic, no network.

Stores raw (as-traded) daily bars with split factors and cash dividends, and
serves them the way the Tiingo client does: `get_prices` split-adjusted over
the whole stored history, `raw_closes`, `split_events`, `dividends`, plus
curated `security_event`s. Satisfies `hedge_fund.core.MarketDataProvider`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from hedge_fund.data.models import Price
from hedge_fund.data.security_events import SecurityEvent


def weekdays(start: str, end: str) -> list[str]:
    d, out = date.fromisoformat(start), []
    while d <= date.fromisoformat(end):
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


@dataclass
class RawBar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    dividend: float = 0.0
    split: float = 1.0


class SyntheticMarket:
    def __init__(self, bars: dict[str, list[RawBar]] | None = None,
                 events: dict[str, SecurityEvent] | None = None) -> None:
        self.bars = {k.upper(): sorted(v, key=lambda b: b.date) for k, v in (bars or {}).items()}
        self.events = events or {}
        self.calls = 0

    # -- construction helpers ------------------------------------------

    def add(self, symbol: str, bars: list[RawBar]) -> SyntheticMarket:
        self.bars[symbol.upper()] = sorted(bars, key=lambda b: b.date)
        return self

    def add_series(self, symbol: str, days: list[str], closes, *, volume: float = 1_000_000,
                   gap: float = 0.0) -> SyntheticMarket:
        """Bars from a close path; open = previous close * (1 + gap)."""
        bars, prev = [], None
        for d, c in zip(days, closes):
            o = c if prev is None else prev * (1 + gap)
            bars.append(RawBar(d, o, max(o, c) * 1.005, min(o, c) * 0.995, float(c), volume))
            prev = c
        return self.add(symbol, bars)

    @staticmethod
    def random_walk(days: list[str], *, start: float = 100.0, drift: float = 0.0, vol: float = 0.01,
                    seed: int = 0) -> list[float]:
        rng = np.random.default_rng(seed)
        steps = rng.normal(drift, vol, len(days))
        return list(start * np.exp(np.cumsum(steps)))

    # -- vendor surface ------------------------------------------------

    def _factors(self, rows: list[RawBar]) -> list[float]:
        """Product of splits strictly after each row (vendor adjustment)."""
        out, acc = [0.0] * len(rows), 1.0
        for i in range(len(rows) - 1, -1, -1):
            out[i] = acc
            acc *= rows[i].split
        return out

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        self.calls += 1
        rows = self.bars.get(ticker.upper(), [])
        f = self._factors(rows)
        return [Price(open=r.open / k, high=r.high / k, low=r.low / k, close=r.close / k,
                      volume=int(round(r.volume * k)), time=f"{r.date}T00:00:00Z")
                for r, k in zip(rows, f) if start_date <= r.date <= end_date]

    def raw_closes(self, ticker, start_date, end_date):
        return {r.date: r.close for r in self.bars.get(ticker.upper(), []) if start_date <= r.date <= end_date}

    def split_events(self, ticker, start_date, end_date):
        return {r.date: r.split for r in self.bars.get(ticker.upper(), [])
                if start_date <= r.date <= end_date and r.split != 1.0}

    def dividends(self, ticker, start_date, end_date):
        return {r.date: r.dividend for r in self.bars.get(ticker.upper(), [])
                if start_date <= r.date <= end_date and r.dividend}

    def security_event(self, ticker):
        return self.events.get(ticker.upper())
