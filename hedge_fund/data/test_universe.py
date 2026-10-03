"""PointInTimeUniverse tests — S&P membership from fixtures, prices from a fake client."""

from datetime import date, datetime, timedelta, timezone

import pytest

from hedge_fund.data.models import Price
from hedge_fund.data.store import MarketStore
from hedge_fund.data.universe import PointInTimeUniverse


class Prices:
    """Constant daily dollar volume per ticker: close 10, volume = volumes[ticker]."""

    def __init__(self, volumes, first_day=None):
        self.volumes = volumes
        self.first_day = first_day or {}
        self.prefetched = []

    def get_prices(self, ticker, start, end, **kwargs):
        out, day = [], date.fromisoformat(start)
        while day.isoformat() <= end:
            if day.weekday() < 5 and ticker in self.volumes and day.isoformat() >= self.first_day.get(ticker, ""):
                out.append(Price(open=10, high=10, low=10, close=10, volume=self.volumes[ticker], time=f"{day}T00:00:00Z"))
            day += timedelta(days=1)
        return out

    def prefetch_prices(self, tickers, end):
        self.prefetched.append(sorted(tickers))


CURRENT = ["AAA", "BBB", "NEW"]
CHANGES = [(date(2025, 3, 3), "NEW", "OLD")]


def universe(tmp_path, volumes, size=2, fetch_calls=None):
    def fetch():
        if fetch_calls is not None:
            fetch_calls.append(1)
        return CURRENT, CHANGES
    return PointInTimeUniverse(MarketStore(tmp_path / "m.db"), Prices(volumes), size=size, fetch=fetch,
                               now=lambda: datetime(2026, 9, 30, tzinfo=timezone.utc))


def test_top_members_by_dollar_volume_as_of_the_month(tmp_path):
    u = universe(tmp_path, {"AAA": 100, "BBB": 300, "OLD": 200, "NEW": 50})
    assert u("2025-02-14") == ["BBB", "OLD"]       # OLD still a member in February
    assert u("2025-04-15") == ["BBB", "AAA"]       # replaced by NEW (low volume) in March


def test_only_data_before_the_month_is_used(tmp_path):
    u = universe(tmp_path, {"AAA": 100, "BBB": 300, "OLD": 200, "NEW": 50})
    u.data.first_day["BBB"] = "2025-04-01"           # BBB only starts trading in April
    assert u("2025-04-15") == ["AAA", "NEW"]
    assert u("2025-05-15") == ["BBB", "AAA"]


def test_membership_history_is_stored_and_refreshed_weekly(tmp_path):
    calls = []
    universe(tmp_path, {"AAA": 1}, fetch_calls=calls)("2025-04-15")
    universe(tmp_path, {"AAA": 1}, fetch_calls=calls)("2025-04-15")
    assert len(calls) == 1


def test_candidates_cover_everyone_who_was_ever_a_member(tmp_path):
    u = universe(tmp_path, {"AAA": 1})
    assert u.candidates("2024-01-01", "2026-01-01") == ["AAA", "BBB", "NEW", "OLD"]
    u.prefetch("2024-01-01", "2026-01-01")
    assert u.data.prefetched == [["AAA", "BBB", "NEW", "OLD"]]
