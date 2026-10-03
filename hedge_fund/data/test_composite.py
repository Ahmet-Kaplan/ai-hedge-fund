"""CompositeDataClient tests — which source answers which question.

The routing is the whole contract: a capability the primary does not serve must
reach the fallback, and everything the primary *does* serve must not, or the
free run quietly becomes a paid run.
"""

from __future__ import annotations

import pytest

from hedge_fund.data.composite import CompositeDataClient


class Recorder:
    """Records every call it receives, with the arguments it was given."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)

        def record(*args, **kwargs):
            self.calls.append((item, args, kwargs))
            return f"{self.name}:{item}"

        return record

    def methods(self) -> list[str]:
        return [name for name, _, _ in self.calls]


@pytest.fixture
def pair():
    return Recorder("free"), Recorder("keyed")


def test_prices_and_fundamentals_come_from_the_primary(pair):
    primary, fallback = pair
    client = CompositeDataClient(primary, fallback)

    assert client.get_prices("AAPL", "2024-01-01", "2024-02-01") == "free:get_prices"
    client.get_financial_metrics("AAPL", "2024-02-01")
    client.get_company_facts("AAPL")
    client.get_earnings_history("AAPL")
    client.get_market_cap("AAPL", "2024-02-01")

    assert fallback.calls == []


def test_the_three_unserved_questions_go_to_the_fallback(pair):
    primary, fallback = pair
    client = CompositeDataClient(primary, fallback)

    assert client.get_news("AAPL", "2024-02-01") == "keyed:get_news"
    client.get_insider_trades("AAPL", "2024-02-01")
    client.get_earnings("AAPL")

    assert primary.calls == []
    assert fallback.methods() == ["get_news", "get_insider_trades", "get_earnings"]


def test_arguments_are_forwarded_unchanged(pair):
    """A dropped `start_date` or `limit` would silently widen a window."""
    primary, fallback = pair
    client = CompositeDataClient(primary, fallback)

    client.get_news("AAPL", "2024-02-01", "2024-01-01", 25)
    client.get_financial_metrics("AAPL", "2024-02-01", "annual", 3)

    assert fallback.calls[0] == ("get_news", ("AAPL", "2024-02-01", "2024-01-01", 25), {})
    assert primary.calls[0] == (
        "get_financial_metrics", ("AAPL", "2024-02-01", "annual", 3), {}
    )


def test_the_members_are_visible_for_inspection(pair):
    primary, fallback = pair
    client = CompositeDataClient(primary, fallback)
    assert client.primary is primary and client.fallback is fallback


def test_optional_passthroughs_are_only_forwarded_when_present(pair):
    """`prefetch_prices` and `coverage` are conveniences, not protocol: a
    primary without them must not make the composite raise."""
    primary, fallback = pair
    client = CompositeDataClient(primary, fallback)

    client.prefetch_prices(["AAPL"], "2024-02-01")
    assert primary.methods() == ["prefetch_prices"]
    assert client.coverage(["AAPL"], "2024-02-01") == "free:coverage"


def test_a_primary_without_the_optional_methods_is_tolerated():
    class Bare:
        def get_prices(self, *a, **k):
            return []

    client = CompositeDataClient(Bare(), Recorder("keyed"))

    client.prefetch_prices(["AAPL"], "2024-02-01")   # does not raise
    assert client.coverage(["AAPL"], "2024-02-01") is None


def test_the_composite_is_not_a_context_manager():
    """Resources are opened by `open_data_client`, because a wrapper such as
    CachedDataClient is not itself a context manager."""
    client = CompositeDataClient(Recorder("free"), Recorder("keyed"))
    assert not hasattr(client, "__enter__")
