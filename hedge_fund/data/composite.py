"""CompositeDataClient — one source for what it serves, another for the rest.

The free sources serve prices and SEC fundamentals and nothing else. A run that
staffs a news model therefore needs two sources at once, and the alternative is
worse than it looks: letting `get_news` answer "nothing" is indistinguishable
from "this company had no news", so `news_sentiment` and `news_analyst` would
quietly abstain on every ticker and the receipt would attribute it to the
company rather than to the data source.

So the split is explicit. The *primary* answers everything it can; a *fallback*
answers the capabilities the primary declares it does not serve. The primary
still decides what those are — `FreeDataClient` raises `DataSourceError` for its
three unserved methods — so there is one place to change when a source grows a
capability, rather than a list to keep in sync here.

    with open_data_client() as data:      # free prices + SEC, FD news
        run_cycle(fund, as_of, broker, data, universe)

Resource lifetime is *not* managed here — both members hold resources (the free
client a SQLite store, a keyed client an HTTP session) and not every wrapper is
itself a context manager, so `open_data_client` opens them with nested `with`
statements and this stays a pure router.
"""

from __future__ import annotations

from typing import Any

#: Methods the free sources do not serve. Read from the primary at call time
#: only to keep this list honest — see `_route`.
UNSERVED = ("get_news", "get_insider_trades", "get_earnings")


class CompositeDataClient:
    """Route each capability to the source that actually serves it."""

    def __init__(self, primary: Any, fallback: Any) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def primary(self) -> Any:
        return self._primary

    @property
    def fallback(self) -> Any:
        return self._fallback

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def get_prices(self, ticker: str, start_date: str, end_date: str, **kwargs: Any):
        return self._primary.get_prices(ticker, start_date, end_date, **kwargs)

    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10):
        return self._primary.get_financial_metrics(ticker, end_date, period, limit)

    def get_company_facts(self, ticker: str):
        return self._primary.get_company_facts(ticker)

    def get_earnings_history(self, ticker: str, limit: int = 12):
        return self._primary.get_earnings_history(ticker, limit)

    def get_market_cap(self, ticker: str, end_date: str):
        return self._primary.get_market_cap(ticker, end_date)

    def get_news(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000):
        return self._fallback.get_news(ticker, end_date, start_date, limit)

    def get_insider_trades(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000):
        return self._fallback.get_insider_trades(ticker, end_date, start_date, limit)

    def get_earnings(self, ticker: str):
        return self._fallback.get_earnings(ticker)

    # ------------------------------------------------------------------
    # Passthroughs the pipeline uses when present
    # ------------------------------------------------------------------

    def prefetch_prices(self, tickers: list[str], end_date: str) -> None:
        probe = getattr(self._primary, "prefetch_prices", None)
        if callable(probe):
            probe(tickers, end_date)

    def coverage(self, *args: Any, **kwargs: Any):
        probe = getattr(self._primary, "coverage", None)
        return probe(*args, **kwargs) if callable(probe) else None
