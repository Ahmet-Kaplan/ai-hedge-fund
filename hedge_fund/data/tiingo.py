"""Tiingo end-of-day prices: full history once per ticker, served locally.

    from hedge_fund.data.tiingo import TiingoClient

    with TiingoClient() as tiingo:                                  # TIINGO_API_KEY from env
        bars = tiingo.get_prices("AAPL", "2020-08-24", "2020-09-04")  # split-adjusted
        raw = tiingo.raw_closes("AAPL", "2020-08-24", "2020-09-04")   # as traded

Storage. The first request for a ticker downloads its entire daily history
(one HTTP request) and stores the *raw* bars — unadjusted OHLCV plus
Tiingo's splitFactor and divCash — gzipped under ~/.hedge-fund/cache/tiingo/.
Every later date range is sliced from that file. A stale file (older than
max_age) is extended once per process with only the missing days.

Adjustment. `get_prices` returns split-adjusted bars — the basis the
DataClient protocol (and the Financial Datasets feed before it) promises —
computed here from raw closes and split factors, never taken from Tiingo's
adjClose (which also folds in dividends). Because the whole series is
re-based from one stored raw history, bars fetched at different times can
never mix bases. A ticker's history is pinned for the life of the process,
so a backtest cannot see its basis shift mid-run.

`raw_closes` / `split_events` / `dividends` implement the RawPriceSource
that the EDGAR adapter needs for point-in-time market cap and P/E.

Symbols. Share-class tickers use Tiingo's dash form (BRK.B -> BRK-B), and a
delisted ticker whose history Tiingo files under a successor symbol is
aliased (FRC -> FRCB). A ticker Tiingo does not know returns no bars: the
pipeline skips names without prices instead of failing the run. Auth,
quota, network and server errors raise TiingoClientError.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from math import isfinite
from pathlib import Path

import requests

from hedge_fund.data.models import Price
from hedge_fund.data.sessions import NEW_YORK
from hedge_fund.paths import CACHE_DIR

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = CACHE_DIR / "tiingo"
API_KEY_ENV = "TIINGO_API_KEY"
HISTORY_START = "1962-01-01"

# Delisted tickers whose full history Tiingo files under a successor symbol.
SYMBOL_ALIASES: dict[str, str] = {
    "FRC": "FRCB",  # First Republic Bank; OTC successor after the May 2023 seizure
}


class TiingoClientError(Exception):
    """A Tiingo request failed for infrastructure reasons (auth, quota,
    network, server) — distinct from "this ticker has no data"."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def tiingo_symbol(ticker: str) -> str:
    t = ticker.strip().upper().replace("/", ".")
    t = SYMBOL_ALIASES.get(t, t)
    return t.replace(".", "-")


# Process-wide: one pinned history per (cache dir, symbol), one download at a time.
_HISTORIES: dict[tuple[str, str], dict | None] = {}
_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock(key: tuple[str, str]) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def clear_process_cache() -> None:
    """Forget pinned histories (tests; long-lived processes between runs)."""
    with _LOCKS_GUARD:
        _HISTORIES.clear()


class TiingoClient:
    BASE_URL = "https://api.tiingo.com"
    _RETRY_DELAYS = (5, 20, 60)

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        offline: bool = False,
        max_age_hours: float | None = 20.0,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._dir = Path(cache_dir)
        self._offline = offline
        self._max_age = None if max_age_hours is None else timedelta(hours=max_age_hours)
        self._timeout = timeout
        self._session: requests.Session | None = None
        self.requests = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> TiingoClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # DataClient: prices
    # ------------------------------------------------------------------

    def get_prices(self, ticker: str, start_date: str, end_date: str,
                   interval: str = "day", interval_multiplier: int = 1) -> list[Price]:
        """Split-adjusted daily OHLCV bars in [start_date, end_date]."""
        if interval != "day" or interval_multiplier != 1:
            raise ValueError("TiingoClient serves daily bars only")
        rows = self._rows(ticker)
        if not rows:
            return []
        factors = _forward_split_factors(rows)
        out = []
        for row, f in zip(rows, factors):
            d = row["date"]
            if start_date <= d <= end_date:
                out.append(Price(
                    open=row["open"] / f, high=row["high"] / f, low=row["low"] / f, close=row["close"] / f,
                    volume=int(round(row["volume"] * f)), time=f"{d}T00:00:00Z",
                ))
        return out

    # ------------------------------------------------------------------
    # RawPriceSource (EDGAR valuation) and corporate actions
    # ------------------------------------------------------------------

    def raw_closes(self, ticker: str, start_date: str, end_date: str) -> dict[str, float]:
        return {r["date"]: r["close"] for r in self._rows(ticker) if start_date <= r["date"] <= end_date}

    def split_events(self, ticker: str, start_date: str, end_date: str) -> dict[str, float]:
        return {r["date"]: r["splitFactor"] for r in self._rows(ticker)
                if start_date <= r["date"] <= end_date and r["splitFactor"] != 1.0}

    def dividends(self, ticker: str, start_date: str, end_date: str) -> dict[str, float]:
        """{ex-date: cash per share, as traded (not split-adjusted)}."""
        return {r["date"]: r["divCash"] for r in self._rows(ticker)
                if start_date <= r["date"] <= end_date and r["divCash"]}

    def is_stored(self, ticker: str) -> bool:
        """True if *ticker*'s history (or a remembered miss) is already on
        disk or pinned in this process — reading it costs no request."""
        symbol = tiingo_symbol(ticker)
        return (str(self._dir.resolve()), symbol) in _HISTORIES or self._path(symbol).exists()

    def history_range(self, ticker: str) -> tuple[str, str] | None:
        rows = self._rows(ticker)
        return (rows[0]["date"], rows[-1]["date"]) if rows else None

    # ------------------------------------------------------------------
    # History store
    # ------------------------------------------------------------------

    def _rows(self, ticker: str) -> list[dict]:
        symbol = tiingo_symbol(ticker)
        key = (str(self._dir.resolve()), symbol)
        with _lock(key):
            if key not in _HISTORIES:
                _HISTORIES[key] = self._load(symbol)
            hist = _HISTORIES[key]
        return hist["rows"] if hist else []

    def _path(self, symbol: str) -> Path:
        return self._dir / f"{symbol}.json.gz"

    def _load(self, symbol: str) -> dict | None:
        """Disk history, downloaded or extended if missing or stale."""
        path = self._path(symbol)
        cached = _read(path)
        if cached is not None and (self._offline or self._fresh(cached)):
            return cached if cached.get("found", True) else None
        if self._offline:
            raise TiingoClientError(f"offline and no cached history for {symbol}")
        if cached is not None and cached.get("found", True) and cached["rows"]:
            last = cached["rows"][-1]
            new = self._download(symbol, last["date"])  # overlap one day to verify continuity
            overlap = [r for r in (new or []) if r["date"] == last["date"]]
            if overlap and abs(overlap[0]["close"] / last["close"] - 1) > 1e-6:
                # raw history should never change; if it did, the provider
                # corrected it — take the whole series again, not a splice
                logger.warning("Tiingo revised %s history; re-downloading", symbol)
                new = self._download(symbol, HISTORY_START)
                rows = _merge([], new or [])
            else:
                rows = _merge(cached["rows"], new or [])
        else:
            new = self._download(symbol, HISTORY_START)
            rows = _merge([], new or [])
        hist = {"symbol": symbol, "found": bool(rows) or new is not None,
                "fetched_at": datetime.now(NEW_YORK).isoformat(), "rows": rows}
        _write(path, hist)
        return hist if rows else None

    def _fresh(self, cached: dict) -> bool:
        if self._max_age is None:
            return True
        try:
            return datetime.now(NEW_YORK) - datetime.fromisoformat(cached["fetched_at"]) <= self._max_age
        except (KeyError, TypeError, ValueError):
            return False

    def _download(self, symbol: str, start: str) -> list[dict] | None:
        """Raw daily rows from *start*; None if Tiingo does not know the symbol."""
        body = self._request(f"/tiingo/daily/{symbol.lower()}/prices", {"startDate": start, "format": "json"})
        if body is None:
            return None
        if not isinstance(body, list):
            raise TiingoClientError(f"unexpected Tiingo response for {symbol}: {str(body)[:200]}")
        return [_clean(r) for r in body]

    def _request(self, path: str, params: dict):
        if self._session is None:
            key = self._api_key or os.environ.get(API_KEY_ENV)
            if not key:
                raise TiingoClientError(f"{API_KEY_ENV} is not set")
            self._session = requests.Session()
            self._session.headers.update({"Authorization": f"Token {key}", "Content-Type": "application/json"})
        key = self._session.headers["Authorization"].split(" ", 1)[-1]
        for attempt, delay in enumerate((*self._RETRY_DELAYS, None)):
            self.requests += 1
            try:
                resp = self._session.get(self.BASE_URL + path, params=params, timeout=self._timeout)
            except requests.RequestException as exc:
                raise TiingoClientError(f"GET {path} failed: {_redact(str(exc), key)}") from exc
            if resp.status_code == 404:
                return None
            if resp.status_code in (429, 500, 502, 503, 504):
                if delay is None:
                    raise TiingoClientError(f"GET {path} still returning {resp.status_code} after "
                                            f"{len(self._RETRY_DELAYS)} retries", status_code=resp.status_code)
                logger.info("Tiingo %s for %s, retrying in %ss", resp.status_code, path, delay)
                time.sleep(delay)
                continue
            if resp.status_code >= 400:
                raise TiingoClientError(f"GET {path} returned {resp.status_code}: {_redact(resp.text[:200], key)}",
                                        status_code=resp.status_code)
            body = resp.json()
            if isinstance(body, dict) and "not found" in str(body.get("detail", "")).lower():
                return None
            return body
        raise AssertionError("unreachable: the final attempt returns or raises")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(r: dict) -> dict:
    def num(k, default=0.0):
        v = r.get(k)
        return float(v) if v is not None and isfinite(float(v)) else default
    split = num("splitFactor", 1.0)
    return {
        "date": str(r["date"])[:10],
        "open": num("open"), "high": num("high"), "low": num("low"), "close": num("close"),
        "volume": int(num("volume")), "divCash": num("divCash"),
        "splitFactor": split if split > 0 else 1.0,
    }


def _merge(old: list[dict], new: list[dict]) -> list[dict]:
    by_date = {r["date"]: r for r in old}
    by_date.update({r["date"]: r for r in new})
    return [by_date[d] for d in sorted(by_date) if by_date[d]["close"] > 0]


def _forward_split_factors(rows: list[dict]) -> list[float]:
    """For each bar, the product of split factors effective *after* it —
    dividing a raw price by it restates the price in today's share basis."""
    out = [1.0] * len(rows)
    running = 1.0
    for i in range(len(rows) - 1, -1, -1):
        out[i] = running
        running *= rows[i]["splitFactor"]
    return out


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError, EOFError):
        return None


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    with gzip.open(tmp, "wt") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    tmp.replace(path)


def _redact(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text

