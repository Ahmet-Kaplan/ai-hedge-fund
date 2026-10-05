"""Crypto bars: fake session, temp SQLite store, no network."""

import pytest

from hedge_fund.data.crypto_prices import CryptoPriceSource, crypto_closes, pair
from hedge_fund.data.errors import DataSourceError
from hedge_fund.data.store import MarketStore


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.text = payload, status, str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.headers, self.calls, self._responses = {}, [], list(responses)

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        return self._responses.pop(0)


def bar(day, close):
    return {"t": f"{day}T00:00:00Z", "o": close, "h": close, "l": close, "c": close, "v": 1.5}


def test_pair_normalizes_both_alpaca_forms():
    assert pair("BTCUSD") == pair("btc/usd") == "BTC/USD"
    with pytest.raises(ValueError):
        pair("BTCEUR")


def test_fetch_follows_pages():
    session = FakeSession(
        FakeResponse({"bars": {"BTC/USD": [bar("2021-01-01", 29000.0)]}, "next_page_token": "p2"}),
        FakeResponse({"bars": {"BTC/USD": [bar("2021-01-02", 32000.0)], "ETH/USD": [bar("2021-01-02", 700.0)]},
                      "next_page_token": None}),
    )
    out = CryptoPriceSource(session=session).fetch(["BTC/USD", "ETH/USD"], "2021-01-01", "2021-01-02")
    assert [p.close for p in out["BTC/USD"]] == [29000.0, 32000.0]
    assert out["ETH/USD"][0].time.startswith("2021-01-02")
    assert session.calls[1]["page_token"] == "p2"
    assert session.calls[0]["timeframe"] == "1Day"


def test_http_error_raises():
    with pytest.raises(DataSourceError):
        CryptoPriceSource(session=FakeSession(FakeResponse({"message": "no"}, 500))).fetch(["BTC/USD"], "2021-01-01", "2021-01-02")


def test_closes_are_cached_and_synced_incrementally(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    first = FakeSession(FakeResponse({"bars": {"BTC/USD": [bar("2021-01-01", 1.0), bar("2021-01-02", 2.0)]}}))
    assert crypto_closes(store, CryptoPriceSource(session=first), ["BTC/USD"], "2021-01-01", "2021-01-02") == \
        {"BTC/USD": {"2021-01-01": 1.0, "2021-01-02": 2.0}}
    second = FakeSession(FakeResponse({"bars": {"BTC/USD": [bar("2021-01-03", 3.0)]}}))
    closes = crypto_closes(store, CryptoPriceSource(session=second), ["BTC/USD"], "2021-01-01", "2021-01-03")
    assert closes["BTC/USD"]["2021-01-03"] == 3.0
    assert second.calls[0]["start"].startswith("2021-01-03")           # only the missing day was fetched
    untouched = FakeSession()
    crypto_closes(store, CryptoPriceSource(session=untouched), ["BTC/USD"], "2021-01-01", "2021-01-03")
    assert untouched.calls == []                                       # fully cached


def test_latest_quotes_are_bid_and_ask():
    session = FakeSession(FakeResponse({"quotes": {"BTC/USD": {"bp": 60_000.0, "ap": 60_010.5, "bs": 1, "as": 1}}}))
    assert CryptoPriceSource(session=session).latest_quotes(["BTC/USD"]) == {"BTC/USD": (60_000.0, 60_010.5)}
    assert session.calls[0] == {"symbols": "BTC/USD"}


def test_missing_or_crossed_quote_raises():
    session = FakeSession(FakeResponse({"quotes": {"BTC/USD": {"bp": 0, "ap": 1}}}))
    with pytest.raises(DataSourceError, match="ETH/USD"):
        CryptoPriceSource(session=session).latest_quotes(["BTC/USD", "ETH/USD"])
