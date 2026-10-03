"""Daily price bars from Alpaca's market data API — free with any Alpaca account.

Consolidated (SIP) daily bars, split- and dividend-adjusted, back to 2016.
One request covers up to 100 tickers; pages are followed until exhausted.
This is a fetcher only: FreeDataClient decides what to fetch and stores it.
"""

from __future__ import annotations

import os
import time
from typing import Callable

import requests

from hedge_fund.data.errors import DataSourceError
from hedge_fund.data.models import Price

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
_SYMBOLS_PER_REQUEST = 100

# The free market-data tier rate-limits per minute. Requests are spaced so the
# limit is rarely reached, and retried with backoff when it is anyway — the same
# shape the SEC source uses.
_RETRY_DELAYS = (1.0, 2.0, 4.0)
_MIN_INTERVAL = 0.35


class AlpacaPriceSource:
    def __init__(self, key_id: str | None = None, secret_key: str | None = None, *,
                 timeout: float = 30.0, session: requests.Session | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        key_id = key_id or os.environ.get("APCA_API_KEY_ID", "")
        secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY", "")
        if not key_id or not secret_key:
            raise ValueError("set APCA_API_KEY_ID and APCA_API_SECRET_KEY (Alpaca keys) for free price data")
        self._timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key})
        self._clock, self._sleep = clock, sleep
        self._last_request: float | None = None

    def _get(self, params: dict) -> dict:
        """One bars request, paced and retried.

        A backtest walks many tickers and pages, and the free tier rate-limits,
        so a single 429 must not end the run. Spaced requests first, then
        exponential backoff — the cheapest way to handle a rate limit is not to
        hit it. The clock and sleeper are injectable so a test can assert the
        retry without waiting for it, which is how SecSource is tested too.
        """
        for delay in (*_RETRY_DELAYS, None):
            if self._last_request is not None:
                wait = _MIN_INTERVAL - (self._clock() - self._last_request)
                if wait > 0:
                    self._sleep(wait)
            self._last_request = self._clock()
            try:
                resp = self._session.get(BARS_URL, params=params, timeout=self._timeout)
            except requests.RequestException as exc:
                if delay is None:
                    raise DataSourceError(f"Alpaca bars: {exc}") from exc
                self._sleep(delay)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if delay is None:
                    raise DataSourceError(
                        f"Alpaca bars: HTTP {resp.status_code}: {resp.text[:300]}",
                        status_code=resp.status_code,
                    )
                self._sleep(delay)
                continue
            if resp.status_code >= 400:
                raise DataSourceError(f"Alpaca bars: HTTP {resp.status_code}: {resp.text[:300]}",
                                      status_code=resp.status_code)
            return resp.json()
        raise AssertionError("unreachable")

    def fetch(self, tickers: list[str], start: str, end: str, adjustment: str = "all") -> dict[str, list[Price]]:
        """Daily bars in [start, end] per ticker, oldest first. Tickers without bars are absent.

        adjustment="all" (split + dividend adjusted) for returns; "raw" for
        prices as traded, which market caps need.
        """
        out: dict[str, list[Price]] = {}
        for i in range(0, len(tickers), _SYMBOLS_PER_REQUEST):
            chunk = tickers[i:i + _SYMBOLS_PER_REQUEST]
            params = {"symbols": ",".join(chunk), "timeframe": "1Day", "start": start, "end": end,
                      "adjustment": adjustment, "feed": "sip", "limit": 10000}
            while True:
                body = self._get(params)
                for ticker, bars in (body.get("bars") or {}).items():
                    out.setdefault(ticker, []).extend(_price(b) for b in bars)
                token = body.get("next_page_token")
                if not token:
                    break
                params = {**params, "page_token": token}
        return out


def _price(bar: dict) -> Price:
    # Daily bars are stamped at midnight New York time (04:00/05:00 UTC); the
    # trading date is the date part.
    return Price(open=bar["o"], high=bar["h"], low=bar["l"], close=bar["c"],
                 volume=int(bar["v"]), time=f"{bar['t'][:10]}T00:00:00Z")
