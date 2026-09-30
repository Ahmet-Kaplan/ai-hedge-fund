"""SEC EDGAR fundamentals behind the DataClient protocol.

    from hedge_fund.data.edgar import EdgarClient

    with EdgarClient(price_source=raw_prices) as sec:        # SEC_USER_AGENT from env
        rows = sec.get_financial_metrics("KO", "2020-02-28")  # point-in-time TTM rows

Point-in-time: a row exists from its filing's SEC acceptance date and holds
only what was filed by that date (see metrics.py). `get_financial_metrics`
returns rows filed on or before *end_date*, newest first — the same
contract FDClient implements with ``filing_date_lte``.

Network: data.sec.gov and www.sec.gov only, identified by SEC_USER_AGENT
("Name contact@example.com", required by SEC fair-access policy), at most
~8 requests/second. Responses are cached on disk (trimmed) under
~/.hedge-fund/cache/edgar/; a filing's cover page is fetched at most once
ever. Infrastructure failures raise EdgarClientError; 404 means "no data".

Scope: fundamentals and company facts. Prices come from *price_client*
(any DataClient) if given; news, insider trades and earnings are not
provided by this adapter and raise NotImplementedError.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta
from math import isfinite, prod
from pathlib import Path

import requests

from hedge_fund.data.edgar.concepts import trim_companyfacts, trim_submissions
from hedge_fund.data.edgar.cover import parse_cover_shares
from hedge_fund.data.edgar.facts import Filing, FactStore
from hedge_fund.data.edgar.identity import (
    SHARE_CLASSES,
    Identity,
    TickerResolver,
    normalize_ticker,
    shares_in_ticker_units,
)
from hedge_fund.data.edgar.metrics import FilingValues, flows, to_metrics
from hedge_fund.data.edgar.prices import RawPriceSource
from hedge_fund.data.models import CompanyFacts, FinancialMetrics
from hedge_fund.data.sessions import NEW_YORK
from hedge_fund.paths import CACHE_DIR

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = CACHE_DIR / "edgar"
USER_AGENT_ENV = "SEC_USER_AGENT"
_ANNUAL_FORMS = frozenset({"10-K", "10-KT"})

# One download per document at a time across threads (the TUI fans out).
_DOC_LOCKS: dict[str, threading.Lock] = {}
_DOC_LOCKS_GUARD = threading.Lock()


def _doc_lock(path: Path) -> threading.Lock:
    with _DOC_LOCKS_GUARD:
        return _DOC_LOCKS.setdefault(str(path.resolve()), threading.Lock())


class EdgarClientError(Exception):
    """An SEC request failed for infrastructure reasons (not "no data")."""

    def __init__(self, message: str, *, status_code: int | None = None, path: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.path = path


class EdgarConfigError(EdgarClientError):
    """SEC_USER_AGENT is missing or unusable."""


def validate_user_agent(value: str | None) -> str:
    ua = (value or "").strip()
    if not ua or "@" not in ua or len(ua) < 8:
        raise EdgarConfigError(
            f"{USER_AGENT_ENV} must identify you to the SEC, e.g. 'Jane Doe jane@example.com' "
            "(https://www.sec.gov/os/accessing-edgar-data). Set it in your shell or ~/.hedge-fund/.env."
        )
    return ua


def sic_division(sic: int | str | None) -> str | None:
    """SIC code -> SIC division name (the adapter's `sector`)."""
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return None
    for lo, hi, name in (
        (100, 999, "Agriculture, Forestry and Fishing"),
        (1000, 1499, "Mining"),
        (1500, 1799, "Construction"),
        (2000, 3999, "Manufacturing"),
        (4000, 4999, "Transportation, Communications and Utilities"),
        (5000, 5199, "Wholesale Trade"),
        (5200, 5999, "Retail Trade"),
        (6000, 6799, "Finance, Insurance and Real Estate"),
        (7000, 8999, "Services"),
        (9100, 9729, "Public Administration"),
        (9900, 9999, "Nonclassifiable"),
    ):
        if lo <= code <= hi:
            return name
    return None


class EdgarClient:
    DATA_URL = "https://data.sec.gov"
    WWW_URL = "https://www.sec.gov"
    _RETRY_DELAYS = (1, 4, 10)
    _MIN_INTERVAL = 0.125  # seconds between requests; SEC allows 10/s

    def __init__(
        self,
        user_agent: str | None = None,
        *,
        price_source: RawPriceSource | None = None,
        price_client=None,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        offline: bool = False,
        max_age_hours: float | None = 24.0,
        timeout: float = 30.0,
        today: str | None = None,
    ) -> None:
        self._user_agent = user_agent
        self._prices = price_source
        self._price_client = price_client
        self._dir = Path(cache_dir)
        self._offline = offline
        self._max_age = None if max_age_hours is None else timedelta(hours=max_age_hours)
        self._timeout = timeout
        self._today = today
        self._session: requests.Session | None = None
        self._last_request = 0.0
        self.requests = 0
        self._history = TickerResolver(history=None)
        self._current: TickerResolver | None = None
        self._stores: dict[tuple, FactStore | None] = {}
        self._filings: dict[tuple, list[Filing]] = {}
        self._values: dict[tuple[str, str], FilingValues] = {}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # DataClient protocol — fundamentals
    # ------------------------------------------------------------------

    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm",
                              limit: int = 10) -> list[FinancialMetrics]:
        """Rows for reports filed on or before *end_date*, newest first.

        period="ttm": one row per 10-Q/10-K (trailing twelve months).
        period="annual": 10-K rows only (TTM at a fiscal year end = the year).
        Unknown tickers and registrants without SEC fundamentals return [].
        """
        if period not in ("ttm", "annual"):
            raise ValueError(f"EdgarClient supports period='ttm' or 'annual', not {period!r}")
        ident = self.resolve(ticker, end_date)
        if ident is None:
            logger.warning("EDGAR: no SEC identity for %s", ticker)
            return []
        if not ident.has_fundamentals:
            logger.info("EDGAR: %s has no SEC fundamentals (%s)", ticker, ident.note)
            return []
        store = self._store(ident)
        if store is None:
            return []
        filings = [f for f in self._filings_of(ident, store)
                   if f.filed <= end_date and (period == "ttm" or f.form in _ANNUAL_FORMS)]
        filings.sort(key=lambda f: (f.filed, f.accn), reverse=True)
        classed = ident.cik in SHARE_CLASSES
        out = []
        for filing in filings[:max(limit, 0)]:
            fv = self._filing_values(ticker, store, filing)
            out.append(to_metrics(ticker.strip().upper(), fv, classed,
                                  self._split_factor(ticker, filing.filed, end_date)))
        return out

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        today = self._today_iso()
        ident = self.resolve(ticker, today)
        if ident is None or ident.cik is None:
            return None
        sub = self._submissions(ident.cik)
        if sub is None:
            return None
        sic = sub.get("sic")
        exchanges = [e for e in (sub.get("exchanges") or []) if e]
        return CompanyFacts(
            ticker=ticker.strip().upper(),
            is_active=ident.is_active(today),
            name=sub.get("name"),
            cik=str(ident.cik),
            sector=sic_division(sic),
            industry=sub.get("sicDescription"),
            category=sub.get("category"),
            exchange=exchanges[0] if exchanges else None,
            location=sub.get("location"),
            sec_filings_url=f"{self.WWW_URL}/cgi-bin/browse-edgar?action=getcompany&CIK={ident.cik:010d}",
            sic_code=str(sic) if sic else None,
            sic_industry=sub.get("sicDescription"),
            sic_sector=sic_division(sic),
        )

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        """Market cap of the latest report filed by *end_date* (point-in-time)."""
        rows = self.get_financial_metrics(ticker, end_date, limit=1)
        return rows[0].market_cap if rows else None

    # ------------------------------------------------------------------
    # DataClient protocol — delegated or unsupported
    # ------------------------------------------------------------------

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        if self._price_client is None:
            raise NotImplementedError("EdgarClient has no prices; pass price_client=<DataClient>")
        return self._price_client.get_prices(ticker, start_date, end_date, **kwargs)

    def get_news(self, ticker, end_date, start_date=None, limit=1000):
        raise NotImplementedError("EdgarClient does not provide news")

    def get_insider_trades(self, ticker, end_date, start_date=None, limit=1000):
        raise NotImplementedError("EdgarClient does not provide insider trades")

    def get_earnings(self, ticker):
        raise NotImplementedError("EdgarClient does not provide earnings")

    def get_earnings_history(self, ticker, limit=12):
        raise NotImplementedError("EdgarClient does not provide earnings history")

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def resolve(self, ticker: str, as_of: str | None = None) -> Identity | None:
        """Curated history first; SEC's current ticker map only on a miss."""
        ident = self._history.resolve(ticker, as_of)
        if ident is not None:
            return ident
        if self._current is None:
            self._current = TickerResolver(current=self._company_tickers(), history={})
        return self._current.resolve(ticker, as_of)

    # ------------------------------------------------------------------
    # Row construction
    # ------------------------------------------------------------------

    def _store(self, ident: Identity) -> FactStore | None:
        key = tuple(ident.lineage())
        if key not in self._stores:
            parts = []
            for cik, until in ident.lineage():
                doc = self._companyfacts(cik)
                if doc is not None:
                    parts.append(FactStore.from_companyfacts(doc, filed_until=until))
            self._stores[key] = FactStore.merged(parts) if parts else None
        return self._stores[key]

    def _filings_of(self, ident: Identity, store: FactStore) -> list[Filing]:
        key = tuple(ident.lineage())
        if key not in self._filings:
            self._filings[key] = store.filings()
        return list(self._filings[key])

    def _filing_values(self, ticker: str, store: FactStore, filing: Filing) -> FilingValues:
        key = (normalize_ticker(ticker), filing.accn)
        if key not in self._values:
            view = store.view(filing.filed)
            prior_end = view.prior_period_end(filing.report_period)
            fv = FilingValues(filing=filing, values=flows(view, filing.report_period),
                              prior=flows(view, prior_end) if prior_end else {})
            fv.shares, fv.shares_source = self._shares(ticker, store, view, filing)
            fv.price, fv.price_date = self._price_at(ticker, filing.filed)
            self._values[key] = fv
        return self._values[key]

    def _shares(self, ticker, store, view, filing) -> tuple[float | None, str | None]:
        """Shares outstanding for *filing*, in units of *ticker*'s class.

        Order: the filing's own dei cover value -> balance-sheet shares ->
        its rendered cover page (per class) -> weighted-average diluted.
        Multi-class registrants with unequal classes (Berkshire) use the
        cover page only: their single-value facts count one class.
        """
        classed = filing.cik in SHARE_CLASSES
        if not classed:
            cover = store.cover_shares(filing.accn)
            if cover:
                latest = max(f.end for f in cover)
                values = {f.val for f in cover if f.end == latest}
                if len(values) == 1 and next(iter(values)) > 0:
                    return next(iter(values)), "dei_cover"
            bs = view.instant("shares_outstanding_balance_sheet", filing.report_period)
            if bs:
                return bs, "balance_sheet"
        by_class = self._cover_page_shares(filing.cik, filing.accn)
        if by_class:
            converted = shares_in_ticker_units(filing.cik, ticker, by_class, filing.filed)
            if converted:
                return converted, "cover_page"
        if not classed:
            weighted = view.latest_duration("weighted_shares_diluted", filing.report_period)
            if weighted:
                return weighted, "weighted_average"
        return None, None

    def _price_at(self, ticker: str, day: str) -> tuple[float | None, str | None]:
        """Raw close on *day*, else the last one within the prior week."""
        if self._prices is None:
            return None, None
        start = (date.fromisoformat(day) - timedelta(days=7)).isoformat()
        closes = {d: c for d, c in self._prices.raw_closes(ticker, start, day).items()
                  if d <= day and c is not None and isfinite(c) and c > 0}
        if not closes:
            return None, None
        d = max(closes)
        return closes[d], d

    def _split_factor(self, ticker: str, filed: str, as_of: str) -> float:
        """Product of splits effective after the filing, up to *as_of*."""
        if self._prices is None or as_of <= filed:
            return 1.0
        start = (date.fromisoformat(filed) + timedelta(days=1)).isoformat()
        events = self._prices.split_events(ticker, start, as_of)
        return float(prod(f for d, f in events.items() if filed < d <= as_of and f and f > 0))

    # ------------------------------------------------------------------
    # SEC documents (cached)
    # ------------------------------------------------------------------

    def _company_tickers(self) -> dict[str, int]:
        data = self._cached("company_tickers.json.gz", lambda: self._get_json(f"{self.WWW_URL}/files/company_tickers.json"),
                            trim=lambda raw: {v["ticker"]: v["cik_str"] for v in raw.values()})
        return data or {}

    def _companyfacts(self, cik: int) -> dict | None:
        return self._cached(f"companyfacts/CIK{cik:010d}.json.gz",
                            lambda: self._get_json(f"{self.DATA_URL}/api/xbrl/companyfacts/CIK{cik:010d}.json"),
                            trim=trim_companyfacts)

    def _submissions(self, cik: int) -> dict | None:
        return self._cached(f"submissions/CIK{cik:010d}.json.gz",
                            lambda: self._get_json(f"{self.DATA_URL}/submissions/CIK{cik:010d}.json"),
                            trim=trim_submissions)

    def _cover_page_shares(self, cik: int | None, accn: str) -> dict[str | None, float] | None:
        if cik is None:
            return None
        rel = f"cover/{cik}/{accn}.json.gz"
        path = self._dir / rel
        if not path.exists() and self._offline:
            return None  # offline: cover pages are an enhancement, not a requirement
        url = f"{self.WWW_URL}/Archives/edgar/data/{cik}/{accn.replace('-', '')}/R1.htm"
        data = self._cached(rel, lambda: self._get_text(url), trim=parse_cover_shares, immutable=True)
        if not data:
            return None
        return {(None if k == "" else k): v for k, v in data.items()}

    def _cached(self, rel: str, fetch, trim, immutable: bool = False):
        path = self._dir / rel
        with _doc_lock(path):
            return self._cached_locked(path, rel, fetch, trim, immutable)

    def _cached_locked(self, path: Path, rel: str, fetch, trim, immutable: bool):
        hit = self._read(path)
        if hit is not None:
            fresh = immutable or self._offline or self._max_age is None
            if not fresh:
                try:
                    fresh = datetime.now(NEW_YORK) - datetime.fromisoformat(hit["fetched_at"]) <= self._max_age
                except (KeyError, TypeError, ValueError):
                    fresh = False
            if fresh:
                return hit["data"]
        if self._offline:
            raise EdgarClientError(f"offline and not cached: {rel}", path=rel)
        raw = fetch()
        data = None if raw is None else trim(raw)
        if isinstance(data, dict) and None in data:  # JSON keys must be strings
            data = {("" if k is None else k): v for k, v in data.items()}
        self._write(path, {"fetched_at": datetime.now(NEW_YORK).isoformat(), "data": data})
        return data

    @staticmethod
    def _read(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rt") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError, EOFError):
            return None  # corrupt entry -> refetch

    @staticmethod
    def _write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")
        with gzip.open(tmp, "wt") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        tmp.replace(path)

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _get_json(self, url: str):
        resp = self._request(url)
        return None if resp is None else resp.json()

    def _get_text(self, url: str) -> str | None:
        resp = self._request(url)
        return None if resp is None else resp.text

    def _request(self, url: str) -> requests.Response | None:
        """GET with SEC etiquette. Raises EdgarClientError on failure;
        returns None only for 404."""
        if self._offline:
            raise EdgarClientError(f"offline: refusing to fetch {url}", path=url)
        if self._session is None:
            ua = validate_user_agent(self._user_agent or os.environ.get(USER_AGENT_ENV))
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        for attempt, delay in enumerate((*self._RETRY_DELAYS, None)):
            wait = self._MIN_INTERVAL - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            self.requests += 1
            try:
                resp = self._session.get(url, timeout=self._timeout)
            except requests.RequestException as exc:
                raise EdgarClientError(f"GET {url} failed: {exc}", path=url) from exc
            if resp.status_code in (429, 500, 502, 503, 504):
                if delay is None:
                    raise EdgarClientError(
                        f"GET {url} still returning {resp.status_code} after {len(self._RETRY_DELAYS)} retries",
                        status_code=resp.status_code, path=url)
                logger.info("SEC %s for %s, retrying in %ss", resp.status_code, url, delay)
                time.sleep(delay)
                continue
            if resp.status_code == 404:
                return None
            if resp.status_code == 403:
                raise EdgarClientError(
                    f"GET {url} returned 403: SEC refused the request — check {USER_AGENT_ENV} "
                    "and stay under 10 requests/second", status_code=403, path=url)
            if resp.status_code >= 400:
                raise EdgarClientError(f"GET {url} returned {resp.status_code}: {resp.text[:200]}",
                                       status_code=resp.status_code, path=url)
            return resp
        raise AssertionError("unreachable: the final attempt returns or raises")

    def _today_iso(self) -> str:
        return self._today or datetime.now(NEW_YORK).date().isoformat()
