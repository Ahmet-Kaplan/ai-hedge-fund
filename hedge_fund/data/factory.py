"""The data client every entry point uses: `make_data_client()`.

Default provider "tiingo-edgar" — a CompositeDataClient:

    Tiingo     prices (split-adjusted), raw closes, splits, dividends
    SEC EDGAR  point-in-time fundamentals, filing dates, shares, company facts

It needs TIINGO_API_KEY and SEC_USER_AGENT and never touches Financial
Datasets. News, insider trades and earnings (used by PEAD and the event
study, not the Buffett baseline) have no free source here; they raise a
clear error unless a supplement is opted into explicitly with
HEDGE_FUND_DATA_SUPPLEMENT=financial-datasets.

HEDGE_FUND_DATA_PROVIDER=financial-datasets restores the previous, fully
FD-backed client (disk-cached).
"""

from __future__ import annotations

import os

from hedge_fund.data.edgar import USER_AGENT_ENV, EdgarClient
from hedge_fund.data.tiingo import API_KEY_ENV as TIINGO_KEY_ENV
from hedge_fund.data.tiingo import TiingoClient

PROVIDER_ENV = "HEDGE_FUND_DATA_PROVIDER"
SUPPLEMENT_ENV = "HEDGE_FUND_DATA_SUPPLEMENT"
TIINGO_EDGAR = "tiingo-edgar"
FINANCIAL_DATASETS = "financial-datasets"
DEFAULT_PROVIDER = TIINGO_EDGAR
PROVIDERS = (TIINGO_EDGAR, FINANCIAL_DATASETS)
FD_KEY_ENV = "FINANCIAL_DATASETS_API_KEY"


def data_provider(provider: str | None = None) -> str:
    name = (provider or os.environ.get(PROVIDER_ENV) or DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"{PROVIDER_ENV}={name!r}: expected one of {', '.join(PROVIDERS)}")
    return name


def _supplement() -> str | None:
    value = (os.environ.get(SUPPLEMENT_ENV) or "").strip().lower()
    if value in ("", "none"):
        return None
    if value != FINANCIAL_DATASETS:
        raise ValueError(f"{SUPPLEMENT_ENV}={value!r}: only {FINANCIAL_DATASETS!r} is supported")
    return value


def required_data_env(provider: str | None = None) -> list[str]:
    """Environment variables the selected data provider needs, in prompt order."""
    if data_provider(provider) == FINANCIAL_DATASETS:
        return [FD_KEY_ENV]
    needed = [TIINGO_KEY_ENV, USER_AGENT_ENV]
    if _supplement() == FINANCIAL_DATASETS:
        needed.append(FD_KEY_ENV)
    return needed


class CompositeDataClient:
    """DataClient routing each method to the source that owns that data."""

    def __init__(self, prices: TiingoClient, fundamentals: EdgarClient, supplemental=None) -> None:
        self.prices = prices
        self.fundamentals = fundamentals
        self.supplemental = supplemental

    def __enter__(self) -> CompositeDataClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        for part in (self.prices, self.fundamentals, self.supplemental):
            close = getattr(part, "close", None)
            if callable(close):
                close()

    def request_counts(self) -> dict[str, int]:
        """HTTP requests made so far, per source (cache hits cost nothing)."""
        counts = {"tiingo": self.prices.requests, "sec": self.fundamentals.requests}
        if self.supplemental is not None:
            counts["financial_datasets"] = getattr(self.supplemental, "requests", 0)
        return counts

    # prices -------------------------------------------------------------

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        return self.prices.get_prices(ticker, start_date, end_date, **kwargs)

    def raw_closes(self, ticker, start_date, end_date):
        return self.prices.raw_closes(ticker, start_date, end_date)

    def split_events(self, ticker, start_date, end_date):
        return self.prices.split_events(ticker, start_date, end_date)

    def dividends(self, ticker, start_date, end_date):
        return self.prices.dividends(ticker, start_date, end_date)

    # fundamentals -------------------------------------------------------

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        return self.fundamentals.get_financial_metrics(ticker, end_date, period, limit)

    def get_company_facts(self, ticker):
        return self.fundamentals.get_company_facts(ticker)

    def get_market_cap(self, ticker, end_date):
        return self.fundamentals.get_market_cap(ticker, end_date)

    # not covered by the free sources -----------------------------------

    def _supplement(self, what: str):
        if self.supplemental is None:
            raise NotImplementedError(
                f"{what} is not available from Tiingo + SEC EDGAR; set {SUPPLEMENT_ENV}={FINANCIAL_DATASETS} "
                f"(uses {FD_KEY_ENV}) or {PROVIDER_ENV}={FINANCIAL_DATASETS}")
        return self.supplemental

    def get_news(self, ticker, end_date, start_date=None, limit=1000):
        return self._supplement("news").get_news(ticker, end_date, start_date, limit)

    def get_insider_trades(self, ticker, end_date, start_date=None, limit=1000):
        return self._supplement("insider trades").get_insider_trades(ticker, end_date, start_date, limit)

    def get_earnings(self, ticker):
        return self._supplement("earnings").get_earnings(ticker)

    def get_earnings_history(self, ticker, limit=12):
        return self._supplement("earnings history").get_earnings_history(ticker, limit)


class _ClosingCache:
    """CachedDataClient(FDClient()) with the FD client's lifecycle."""

    def __init__(self, raw) -> None:
        from hedge_fund.data.cached import CachedDataClient
        self._raw = raw
        self._cached = CachedDataClient(raw)

    def __getattr__(self, name):
        return getattr(self._cached, name)

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        self._raw.close()

    @property
    def requests(self) -> int:
        return getattr(self._raw, "requests", 0)


def make_data_client(provider: str | None = None, **kwargs):
    """Build the configured DataClient; use it as a context manager.

    Keyword arguments reach the Tiingo/EDGAR clients for tests and tools:
    tiingo=TiingoClient(...), edgar=EdgarClient(...).
    """
    name = data_provider(provider)
    if name == FINANCIAL_DATASETS:
        from hedge_fund.data.client import FDClient
        return _ClosingCache(FDClient())
    tiingo = kwargs.pop("tiingo", None) or TiingoClient()
    edgar = kwargs.pop("edgar", None) or EdgarClient(price_source=tiingo)
    if kwargs:
        raise TypeError(f"unexpected arguments: {sorted(kwargs)}")
    supplemental = None
    if _supplement() == FINANCIAL_DATASETS:
        from hedge_fund.data.client import FDClient
        supplemental = _ClosingCache(FDClient())
    return CompositeDataClient(tiingo, edgar, supplemental)
