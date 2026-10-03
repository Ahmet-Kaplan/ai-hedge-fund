"""A point-in-time universe: the most liquid S&P 500 members as of each month.

For a backtest to be honest, the stocks it may pick on a past date must be
the ones an investor could have picked then — not today's winners. Each
month's universe is the S&P 500 as it stood at the start of that month
(sp500.py), ranked by median daily dollar volume over the previous ~3 months
of trading, using only data from before the month began. Dollar volume is a
standard liquidity-based proxy for size that needs nothing but prices.
"""

from __future__ import annotations

import logging
import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from hedge_fund.data import sp500
from hedge_fund.data.protocol import DataClient
from hedge_fund.data.store import MarketStore

logger = logging.getLogger(__name__)

_REFRESH_AFTER = timedelta(days=7)
_MIN_SESSIONS = 20


class PointInTimeUniverse:
    def __init__(
        self, store: MarketStore, data: DataClient, *, size: int = 100, lookback_sessions: int = 63,
        fetch: Callable[[], tuple[list[str], list[sp500.Change]]] = sp500.fetch,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.store, self.data = store, data
        self.size, self.lookback = size, lookback_sessions
        self._fetch, self._now = fetch, now
        self._history: tuple[list[str], list[sp500.Change]] | None = None
        self._by_month: dict[str, list[str]] = {}

    def __call__(self, as_of: str) -> list[str]:
        month_start = as_of[:7] + "-01"
        if month_start not in self._by_month:
            self._by_month[month_start] = self._rank(month_start)
        return self._by_month[month_start]

    def members(self, as_of: str) -> set[str]:
        current, changes = self.history()
        return sp500.members_as_of(as_of, current, changes)

    def candidates(self, start: str, end: str) -> list[str]:
        """Everyone who was a member at any point in [start, end]."""
        current, changes = self.history()
        names = sp500.members_as_of(start, current, changes)
        names |= {added for day, added, _ in changes if start < day.isoformat() <= end and added}
        return sorted(names)

    def prefetch(self, start: str, end: str) -> None:
        if hasattr(self.data, "prefetch_prices"):
            self.data.prefetch_prices(self.candidates(start, end), end)

    def history(self) -> tuple[list[str], list[sp500.Change]]:
        if self._history is None:
            current, changes, fetched_at = self.store.sp500()
            stale = fetched_at is None or self._now() - datetime.fromisoformat(fetched_at) > _REFRESH_AFTER
            if stale:
                try:
                    current, changes = self._fetch()
                    self.store.replace_sp500(current, changes, self._now().isoformat())
                except Exception as exc:
                    if fetched_at is None:
                        raise
                    logger.warning("S&P 500 membership: using stored copy from %s (%s)", fetched_at, exc)
            self._history = (current, changes)
        return self._history

    def _rank(self, month_start: str) -> list[str]:
        cutoff = (date.fromisoformat(month_start) - timedelta(days=1)).isoformat()
        window_start = (date.fromisoformat(cutoff) - timedelta(days=int(self.lookback * 1.6) + 10)).isoformat()
        liquidity: dict[str, float] = {}
        for ticker in self.members(month_start):
            bars = [bar for bar in self.data.get_prices(ticker, window_start, cutoff) if bar.time[:10] <= cutoff]
            bars = sorted(bars, key=lambda bar: bar.time)[-self.lookback:]
            if len(bars) >= _MIN_SESSIONS:
                liquidity[ticker] = statistics.median(bar.close * bar.volume for bar in bars)
        ranked = sorted(liquidity, key=lambda t: (-liquidity[t], t))
        return ranked[:self.size]
