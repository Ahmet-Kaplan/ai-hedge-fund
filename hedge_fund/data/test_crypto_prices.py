"""Crypto bars: fake session, temp SQLite store, no network."""

import pytest

from hedge_fund.data.crypto_prices import CryptoPriceSource, crypto_closes, pair
from hedge_fund.data.errors import DataSourceError
from hedge_fund.data.store import MarketStore


class FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self._payload, self.status_code, self.text = payload, status, str(payload)
        self.headers = headers or {}

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
        CryptoPriceSource(session=FakeSession(FakeResponse({"message": "no"}, 400))).fetch(["BTC/USD"], "2021-01-01", "2021-01-02")


def test_429_is_retried_then_succeeds():
    sleeps = []
    session = FakeSession(FakeResponse({"message": "too many requests."}, 429, {"Retry-After": "7"}),
                          FakeResponse({"message": "busy"}, 503),
                          FakeResponse({"bars": {"BTC/USD": [bar("2021-01-01", 29000.0)]}}))
    out = CryptoPriceSource(session=session, sleep=sleeps.append).fetch(["BTC/USD"], "2021-01-01", "2021-01-01")
    assert [p.close for p in out["BTC/USD"]] == [29000.0]
    assert sleeps == [7.0, 5.0]   # Retry-After honoured, then the normal backoff


def test_429_gives_up_after_the_retries():
    session = FakeSession(*[FakeResponse({"message": "too many requests."}, 429)] * 5)
    with pytest.raises(DataSourceError) as exc_info:
        CryptoPriceSource(session=session, sleep=lambda s: None).fetch(["BTC/USD"], "2021-01-01", "2021-01-01")
    assert exc_info.value.status_code == 429 and len(session.calls) == 5


def test_sends_data_keys_from_the_environment(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "kid")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "sec")
    session = FakeSession()
    CryptoPriceSource(session=session)
    assert session.headers == {"APCA-API-KEY-ID": "kid", "APCA-API-SECRET-KEY": "sec"}


def test_works_without_keys(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    session = FakeSession()
    CryptoPriceSource(session=session)
    assert session.headers == {}


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
