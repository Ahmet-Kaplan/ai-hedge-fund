"""AlpacaPriceSource tests — fake session, no network."""

import pytest

from hedge_fund.data.alpaca_prices import AlpacaPriceSource
from hedge_fund.data.errors import DataSourceError


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.headers = {}
        self.calls = []
        self._responses = list(responses)

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        return self._responses.pop(0)


def bar(day, close):
    return {"t": f"{day}T04:00:00Z", "o": close, "h": close, "l": close, "c": close, "v": 10}


def test_follows_pages_and_maps_bars():
    session = FakeSession(
        FakeResponse(payload={"bars": {"AAPL": [bar("2026-09-28", 1.0)]}, "next_page_token": "p2"}),
        FakeResponse(payload={"bars": {"AAPL": [bar("2026-09-29", 2.0)], "SPY": [bar("2026-09-29", 3.0)]},
                              "next_page_token": None}),
    )
    source = AlpacaPriceSource("k", "s", session=session)
    out = source.fetch(["AAPL", "SPY"], "2026-09-28", "2026-09-29")
    assert [(p.time, p.close) for p in out["AAPL"]] == [("2026-09-28T00:00:00Z", 1.0), ("2026-09-29T00:00:00Z", 2.0)]
    assert out["SPY"][0].close == 3.0
    assert session.calls[0]["symbols"] == "AAPL,SPY"
    assert session.calls[0]["feed"] == "sip" and session.calls[0]["adjustment"] == "all"
    assert session.calls[1]["page_token"] == "p2"
    assert session.headers == {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"}


def test_splits_large_symbol_lists():
    tickers = [f"T{i}" for i in range(150)]
    session = FakeSession(FakeResponse(payload={"bars": {}}), FakeResponse(payload={"bars": {}}))
    AlpacaPriceSource("k", "s", session=session).fetch(tickers, "2026-09-28", "2026-09-29")
    assert [len(c["symbols"].split(",")) for c in session.calls] == [100, 50]


def test_http_error_raises():
    session = FakeSession(FakeResponse(403, {"message": "forbidden"}))
    with pytest.raises(DataSourceError) as exc:
        AlpacaPriceSource("k", "s", session=session).fetch(["AAPL"], "2026-09-28", "2026-09-29")
    assert exc.value.status_code == 403


def test_requires_keys(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="APCA_API_KEY_ID"):
        AlpacaPriceSource(session=FakeSession())
