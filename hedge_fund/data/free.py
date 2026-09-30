"""FreeDataClient — the DataClient built on free sources and the local database.

Prices come from Alpaca market data, fundamentals and earnings from SEC
EDGAR; both are fetched into MarketStore once and read locally after:

- prices: the first request for a ticker downloads its history since 2016;
  later requests fetch only days the database doesn't hold yet, plus the last
  stored day as a check. Adjusted prices are rewritten after every split and
  dividend, so if that day no longer matches, the ticker's whole history is
  re-downloaded (otherwise a split would appear as a crash).
- SEC: a company is re-synced only when its stored copy is over 20 hours old
  AND the request is about a date on/after that sync (a backtest about July
  never refetches). If SEC is unreachable, a stored copy is used with a
  warning; with no stored copy the error propagates (fail loud).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable

from hedge_fund.data.alpaca_prices import AlpacaPriceSource
from hedge_fund.data.errors import DataSourceError
from hedge_fund.data.fundamentals import CONCEPTS, earnings_events, metrics_rows
from hedge_fund.data.models import CompanyFacts, CompanyNews, Earnings, EarningsRecord, FinancialMetrics, InsiderTrade, Price
from hedge_fund.data.sec import SecSource, sic_sector
from hedge_fund.data.sessions import completed_through
from hedge_fund.data.store import MarketStore

logger = logging.getLogger(__name__)

PRICE_HISTORY_START = "2016-01-01"   # Alpaca's consolidated daily history begins here
SEC_STALE_AFTER = timedelta(hours=20)
_FILING_FORMS = ["8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A"]


class FreeDataClient:
    def __init__(
        self, store: MarketStore, prices: AlpacaPriceSource, sec: SecSource,
        *, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._store = store
        self._prices = prices
        self._sec = sec
        self._now = now
        self._synced: set[str] = set()   # tickers already re-synced this session

    def __enter__(self) -> "FreeDataClient":
        return self

    def __exit__(self, *exc) -> None:
        self._store.close()

    # -- prices ----------------------------------------------------------------

    def get_prices(self, ticker: str, start_date: str, end_date: str, **kwargs) -> list[Price]:
        end = min(end_date, completed_through())
        if start_date > end:
            return []
        self._ensure_prices([ticker], start_date, end)
        return self._store.prices(ticker, start_date, end)

    def prefetch_prices(self, tickers: list[str], end_date: str) -> None:
        """Bring every ticker's history up to `end_date` in as few requests as possible."""
        self._ensure_prices(sorted(set(tickers)), PRICE_HISTORY_START, min(end_date, completed_through()))

    def _ensure_prices(self, tickers: list[str], start: str, end: str) -> None:
        windows: dict[tuple[str, str], list[str]] = {}
        for ticker in tickers:
            held = self._store.price_range(ticker)
            if held is None:
                windows.setdefault((min(start, PRICE_HISTORY_START), end), []).append(ticker)
                continue
            first, through = held
            if start < first:
                windows.setdefault((start, first), []).append(ticker)
            if end > through:
                windows.setdefault((through, end), []).append(ticker)   # overlaps one stored day
        for (lo, hi), group in windows.items():
            fetched = self._prices.fetch(group, lo, hi)
            raw = self._prices.fetch(group, lo, hi, adjustment="raw")
            for ticker in group:
                bars = fetched.get(ticker, [])
                held = self._store.price_range(ticker)
                if held and lo == held[1] and self._history_rewritten(ticker, lo, bars):
                    self._refresh_history(ticker, hi)
                    continue
                raw_closes = {bar.time[:10]: bar.close for bar in raw.get(ticker, [])}
                self._store.upsert_prices(ticker, bars, raw_closes)
                first = min(lo, held[0]) if held else lo
                through = max(hi, held[1]) if held else hi
                self._store.set_price_range(ticker, first, through)

    def _history_rewritten(self, ticker: str, day: str, bars: list[Price]) -> bool:
        stored = self._store.prices(ticker, day, day)
        fresh = [bar for bar in bars if bar.time[:10] == day]
        return bool(stored and fresh) and abs(fresh[0].close / stored[0].close - 1) > 1e-4

    def _refresh_history(self, ticker: str, end: str) -> None:
        logger.info("%s: adjusted history changed (split or dividend); re-downloading", ticker)
        bars = self._prices.fetch([ticker], PRICE_HISTORY_START, end).get(ticker, [])
        raw = self._prices.fetch([ticker], PRICE_HISTORY_START, end, adjustment="raw").get(ticker, [])
        self._store.delete_prices(ticker)
        self._store.upsert_prices(ticker, bars, {bar.time[:10]: bar.close for bar in raw})
        self._store.set_price_range(ticker, PRICE_HISTORY_START, end)

    def _close_on_or_before(self, ticker: str) -> Callable[[str], float | None]:
        # As traded, not split-adjusted: the share count it multiplies is as filed then.
        return lambda day: self._store.raw_close(ticker, day)

    # -- SEC -------------------------------------------------------------------

    def _cik(self, ticker: str, as_of: str) -> str:
        known = self._store.company(ticker)
        facts_at = self._store.sync_times(known["cik"])[0] if known else None
        if ticker not in self._synced:
            stale = facts_at is None or (
                self._now() - datetime.fromisoformat(facts_at) > SEC_STALE_AFTER and as_of >= facts_at[:10]
            )
            if stale:
                try:
                    self._sec.sync_company(ticker, self._store, now=self._now().isoformat())
                    self._synced.add(ticker)
                except DataSourceError as exc:
                    if facts_at is None:
                        raise
                    logger.warning("%s: using stored SEC data from %s (%s)", ticker, facts_at, exc)
        return self._store.company(ticker)["cik"]

    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10) -> list[FinancialMetrics]:
        if period != "ttm":
            raise ValueError(f"free data serves trailing-twelve-month ('ttm') metrics only, not {period!r}")
        cik = self._cik(ticker, end_date)
        facts = self._store.facts(cik, CONCEPTS, filed_lte=end_date)
        self._ensure_prices([ticker], PRICE_HISTORY_START, min(end_date, completed_through()))
        sic = self._store.company(ticker)["sic"]
        financial = sic is not None and 6000 <= sic <= 6799   # banks, insurers: no meaningful gross profit
        return metrics_rows(ticker, facts, self._close_on_or_before(ticker), limit, derive_gross_profit=not financial)

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        self._cik(ticker, self._now().date().isoformat())
        row = self._store.company(ticker)
        return CompanyFacts(
            ticker=ticker, name=row["name"], cik=row["cik"], sector=sic_sector(row["sic"]),
            industry=row["sic_description"], sic_code=str(row["sic"]) if row["sic"] is not None else None,
        )

    def get_earnings_history(self, ticker: str, limit: int = 12) -> list[EarningsRecord]:
        cik = self._cik(ticker, self._now().date().isoformat())
        eps = self._store.facts(cik, ["EarningsPerShareDiluted"], filed_lte="9999-12-31")["EarningsPerShareDiluted"]
        return earnings_events(ticker, eps, self._store.filings(cik, _FILING_FORMS), limit)

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        rows = self.get_financial_metrics(ticker, end_date, limit=1)
        return rows[0].market_cap if rows else None

    # -- not provided by the free sources (unused by the current analysts) ----

    def get_news(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000) -> list[CompanyNews]:
        return []

    def get_insider_trades(self, ticker: str, end_date: str, start_date: str | None = None, limit: int = 1000) -> list[InsiderTrade]:
        return []

    def get_earnings(self, ticker: str) -> Earnings | None:
        return None
