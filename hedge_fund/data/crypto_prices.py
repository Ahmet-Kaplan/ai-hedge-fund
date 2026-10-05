"""Daily crypto bars from Alpaca's market data API — free, no key needed.

BTC, ETH and SOL go back to 2021-01-01. Bars are UTC days. Pairs use
Alpaca's order/data form (`BTC/USD`); positions come back as `BTCUSD`, and
`pair()` maps either to the first. Bars are cached in the MarketStore
`prices` table under the pair name and synced incrementally.
"""

from __future__ import annotations

from datetime import date, timedelta

import requests

from hedge_fund.data.errors import DataSourceError
from hedge_fund.data.models import Price
from hedge_fund.data.store import MarketStore

CRYPTO_BARS_URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
CRYPTO_QUOTES_URL = "https://data.alpaca.markets/v1beta3/crypto/us/latest/quotes"


def pair(symbol: str) -> str:
    """'BTCUSD', 'btc/usd' or 'BTC-USD' → 'BTC/USD'."""
    s = symbol.strip().upper().replace("-", "/")
    if "/" in s and s.endswith("/USD"):
        return s
    if "/" not in s and s.endswith("USD") and len(s) > 3:
        return f"{s[:-3]}/USD"
    raise ValueError(f"not a USD crypto pair: {symbol!r}")


class CryptoPriceSource:
    def __init__(self, *, timeout: float = 30.0, session: requests.Session | None = None) -> None:
        self._timeout = timeout
        self._session = session or requests.Session()

    def fetch(self, symbols: list[str], start: str, end: str) -> dict[str, list[Price]]:
        """Daily bars in [start, end] (UTC days) per pair, oldest first."""
        params = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": f"{start}T00:00:00Z",
                  "end": f"{end}T23:59:59Z", "limit": 10000, "sort": "asc"}
        out: dict[str, list[Price]] = {}
        while True:
            try:
                resp = self._session.get(CRYPTO_BARS_URL, params=dict(params), timeout=self._timeout)
            except requests.RequestException as exc:
                raise DataSourceError(f"Alpaca crypto bars: {exc}") from exc
            if resp.status_code >= 400:
                raise DataSourceError(f"Alpaca crypto bars: HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            for symbol, rows in (data.get("bars") or {}).items():
                out.setdefault(symbol, []).extend(
                    Price(open=r["o"], high=r["h"], low=r["l"], close=r["c"], volume=int(r.get("v") or 0), time=r["t"])
                    for r in rows)
            token = data.get("next_page_token")
            if not token:
                return out
            params["page_token"] = token


    def latest_quotes(self, symbols: list[str]) -> dict[str, tuple[float, float]]:
        """(best bid, best ask) per pair, right now."""
        try:
            resp = self._session.get(CRYPTO_QUOTES_URL, params={"symbols": ",".join(symbols)}, timeout=self._timeout)
        except requests.RequestException as exc:
            raise DataSourceError(f"Alpaca crypto quotes: {exc}") from exc
        if resp.status_code >= 400:
            raise DataSourceError(f"Alpaca crypto quotes: HTTP {resp.status_code}: {resp.text[:200]}")
        rows = resp.json().get("quotes") or {}
        out = {s: (float(rows[s]["bp"]), float(rows[s]["ap"])) for s in symbols if s in rows}
        bad = [s for s in symbols if s not in out or not 0 < out[s][0] < out[s][1]]
        if bad:
            raise DataSourceError(f"Alpaca crypto quotes: no usable bid/ask for {', '.join(bad)}")
        return out


def crypto_closes(store: MarketStore, source: CryptoPriceSource, symbols: list[str],
                  start: str, end: str) -> dict[str, dict[str, float]]:
    """Closes per pair per UTC day in [start, end], fetching only days not yet cached."""
    for symbol in symbols:
        synced = store.price_range(symbol)
        if synced and synced[0] <= start and synced[1] >= end:
            continue
        fetch_from = start if not synced or synced[0] > start else (date.fromisoformat(synced[1]) + timedelta(days=1)).isoformat()
        bars = source.fetch([symbol], fetch_from, end).get(symbol, [])
        store.upsert_prices(symbol, bars)
        first = min([start] + ([synced[0]] if synced else []))
        store.set_price_range(symbol, first, max(end, synced[1]) if synced else end)
    return {s: {p.time[:10]: p.close for p in store.prices(s, start, end)} for s in symbols}


def crypto_bars(store: MarketStore, source: CryptoPriceSource, symbols: list[str], start: str, end: str
                ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """(closes, highs, lows) per pair per UTC day — synced like crypto_closes."""
    crypto_closes(store, source, symbols, start, end)
    closes, highs, lows = {}, {}, {}
    for s in symbols:
        bars = store.prices(s, start, end)
        closes[s] = {p.time[:10]: p.close for p in bars}
        highs[s] = {p.time[:10]: p.high for p in bars}
        lows[s] = {p.time[:10]: p.low for p in bars}
    return closes, highs, lows
