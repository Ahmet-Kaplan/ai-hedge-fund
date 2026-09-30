"""Daily price bars from Alpaca's market data API — free with any Alpaca account.

Consolidated (SIP) daily bars, split- and dividend-adjusted, back to 2016.
One request covers up to 100 tickers; pages are followed until exhausted.
This is a fetcher only: FreeDataClient decides what to fetch and stores it.
"""

from __future__ import annotations

import os

import requests

from hedge_fund.data.errors import DataSourceError
from hedge_fund.data.models import Price

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
_SYMBOLS_PER_REQUEST = 100


class AlpacaPriceSource:
    def __init__(self, key_id: str | None = None, secret_key: str | None = None, *,
                 timeout: float = 30.0, session: requests.Session | None = None) -> None:
        key_id = key_id or os.environ.get("APCA_API_KEY_ID", "")
        secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY", "")
        if not key_id or not secret_key:
            raise ValueError("set APCA_API_KEY_ID and APCA_API_SECRET_KEY (Alpaca keys) for free price data")
        self._timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key})

    def fetch(self, tickers: list[str], start: str, end: str) -> dict[str, list[Price]]:
        """Daily bars in [start, end] per ticker, oldest first. Tickers without bars are absent."""
        out: dict[str, list[Price]] = {}
        for i in range(0, len(tickers), _SYMBOLS_PER_REQUEST):
            chunk = tickers[i:i + _SYMBOLS_PER_REQUEST]
            params = {"symbols": ",".join(chunk), "timeframe": "1Day", "start": start, "end": end,
                      "adjustment": "all", "feed": "sip", "limit": 10000}
            while True:
                try:
                    resp = self._session.get(BARS_URL, params=params, timeout=self._timeout)
                except requests.RequestException as exc:
                    raise DataSourceError(f"Alpaca bars: {exc}") from exc
                if resp.status_code >= 400:
                    raise DataSourceError(f"Alpaca bars: HTTP {resp.status_code}: {resp.text[:300]}",
                                          status_code=resp.status_code)
                body = resp.json()
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
