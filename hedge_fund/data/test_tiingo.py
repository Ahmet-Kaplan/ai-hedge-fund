"""TiingoClient against a fake Tiingo — synthetic bars, no network, no key.

No vendor data is committed: every series here is made up, shaped like the
Tiingo EOD API response (raw OHLCV + splitFactor + divCash per day).
"""

from __future__ import annotations

import threading
from datetime import date, timedelta

import pytest
import requests

from hedge_fund.data import tiingo as tiingo_mod
from hedge_fund.data.protocol import DataClient
from hedge_fund.data.tiingo import HISTORY_START, TiingoClient, TiingoClientError, tiingo_symbol

KEY = "fixture-tiingo-key-0123456789"


def bar(day, close, split=1.0, div=0.0, volume=1000):
    return {"date": f"{day}T00:00:00.000Z", "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": volume, "adjClose": close, "adjHigh": close, "adjLow": close,
            "adjOpen": close, "adjVolume": volume, "divCash": div, "splitFactor": split}


def weekdays(start, end):
    d, out = date.fromisoformat(start), []
    while d <= date.fromisoformat(end):
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


class FakeTiingo:
    """Serves {symbol: [bars]} honoring startDate; records every call."""

    def __init__(self, histories, scripted=None):
        self.histories = histories
        self.scripted = list(scripted or [])
        self.calls: list[tuple[str, dict]] = []
        self.headers: dict = {}
        self._lock = threading.Lock()

    def get(self, url, params=None, timeout=None):
        with self._lock:
            self.calls.append((url, dict(params or {})))
            if self.scripted:
                r = self.scripted.pop(0)
                if isinstance(r, Exception):
                    raise r
                return r
        symbol = url.rsplit("/daily/", 1)[1].split("/")[0]
        rows = self.histories.get(symbol.upper())
        if rows is None:
            return FakeResponse(404, {"detail": f"Error: Ticker '{symbol}' not found"})
        start = (params or {}).get("startDate", "1900-01-01")
        return FakeResponse(200, [r for r in rows if r["date"][:10] >= start])

    def close(self):
        pass


@pytest.fixture
def fake(monkeypatch):
    server = FakeTiingo({})

    def make_session():
        server.headers = {}
        return server

    monkeypatch.setattr(tiingo_mod.requests, "Session", make_session)
    monkeypatch.setattr(TiingoClient, "_RETRY_DELAYS", (0, 0, 0))
    tiingo_mod.clear_process_cache()
    yield server
    tiingo_mod.clear_process_cache()


def _client(tmp_path, **kw):
    kw.setdefault("api_key", KEY)
    return TiingoClient(cache_dir=tmp_path / "tiingo", **kw)


# 4-for-1 split effective 2020-08-31 (raw close drops from 500 to 125)
AAPL = [bar("2020-08-26", 496.0), bar("2020-08-27", 500.0, div=0.82), bar("2020-08-28", 500.0),
        bar("2020-08-31", 125.0, split=4.0), bar("2020-09-01", 130.0)]


def test_satisfies_protocol_prices(tmp_path, fake):
    assert callable(_client(tmp_path).get_prices)
    assert not isinstance(_client(tmp_path), DataClient)  # prices only; the composite is the DataClient


def test_full_history_downloaded_once_then_served_locally(tmp_path, fake):
    fake.histories["AAPL"] = AAPL
    c = _client(tmp_path)
    assert [p.time[:10] for p in c.get_prices("AAPL", "2020-08-27", "2020-08-31")] == \
        ["2020-08-27", "2020-08-28", "2020-08-31"]
    c.get_prices("AAPL", "2020-01-01", "2020-12-31")
    c.raw_closes("AAPL", "2020-08-01", "2020-09-30")
    assert c.requests == 1
    url, params = fake.calls[0]
    assert url == "https://api.tiingo.com/tiingo/daily/aapl/prices" and params["startDate"] == HISTORY_START
    # another client in the same process shares the pinned history
    other = _client(tmp_path)
    other.get_prices("AAPL", "2020-08-26", "2020-09-01")
    assert other.requests == 0
    # a new process reads the disk store
    tiingo_mod.clear_process_cache()
    fresh = _client(tmp_path)
    assert len(fresh.get_prices("AAPL", "2020-08-26", "2020-09-01")) == 5 and fresh.requests == 0


def test_split_adjusted_prices_and_raw_closes(tmp_path, fake):
    fake.histories["AAPL"] = AAPL
    c = _client(tmp_path)
    bars = {p.time[:10]: p for p in c.get_prices("AAPL", "2020-08-26", "2020-09-01")}
    assert bars["2020-08-28"].close == 125.0 and bars["2020-08-31"].close == 125.0
    assert bars["2020-08-28"].volume == 4000 and bars["2020-08-31"].volume == 1000
    assert bars["2020-08-27"].high == pytest.approx(500 * 1.01 / 4)
    assert c.raw_closes("AAPL", "2020-08-28", "2020-08-31") == {"2020-08-28": 500.0, "2020-08-31": 125.0}
    assert c.split_events("AAPL", "2020-01-01", "2020-12-31") == {"2020-08-31": 4.0}
    # dividends preserved as traded; adjusted prices are split-only, not dividend-adjusted
    assert c.dividends("AAPL", "2020-01-01", "2020-12-31") == {"2020-08-27": 0.82}
    assert bars["2020-08-27"].close == 125.0


def test_multiple_splits_compound(tmp_path, fake):
    fake.histories["XYZ"] = [bar("2021-01-04", 600.0), bar("2021-01-05", 300.0, split=2.0),
                             bar("2021-01-06", 100.0, split=3.0), bar("2021-01-07", 101.0)]
    closes = [p.close for p in _client(tmp_path).get_prices("XYZ", "2021-01-01", "2021-01-31")]
    assert closes == [100.0, 100.0, 100.0, 101.0]


def test_share_class_symbol_and_successor_alias(tmp_path, fake):
    assert tiingo_symbol("brk.b") == "BRK-B" and tiingo_symbol("FRC") == "FRCB"
    fake.histories["BRK-B"] = [bar("2016-08-05", 145.0)]
    fake.histories["FRCB"] = [bar("2023-04-27", 3.5), bar("2023-05-02", 0.3)]
    c = _client(tmp_path)
    assert c.raw_closes("BRK.B", "2016-08-01", "2016-08-31") == {"2016-08-05": 145.0}
    assert [p.time[:10] for p in c.get_prices("FRC", "2023-04-01", "2023-06-01")] == ["2023-04-27", "2023-05-02"]
    assert [u.rsplit("/daily/", 1)[1] for u, _ in fake.calls] == ["brk-b/prices", "frcb/prices"]


def test_unknown_ticker_is_empty_and_remembered(tmp_path, fake):
    c = _client(tmp_path)
    assert c.get_prices("ZZZZ", "2020-01-01", "2020-12-31") == []
    assert c.raw_closes("ZZZZ", "2020-01-01", "2020-12-31") == {}
    assert c.history_range("ZZZZ") is None
    tiingo_mod.clear_process_cache()
    assert _client(tmp_path).get_prices("ZZZZ", "2020-01-01", "2020-12-31") == []
    assert len(fake.calls) == 1  # the miss was stored


def test_stale_store_is_extended_incrementally(tmp_path, fake):
    fake.histories["SPY"] = [bar(d, 100.0) for d in weekdays("2024-01-02", "2024-01-12")]
    _client(tmp_path).get_prices("SPY", "2024-01-01", "2024-01-31")
    fake.histories["SPY"] += [bar(d, 101.0) for d in weekdays("2024-01-15", "2024-01-19")]
    tiingo_mod.clear_process_cache()
    c = _client(tmp_path, max_age_hours=0)
    assert len(c.get_prices("SPY", "2024-01-01", "2024-01-31")) == 14
    assert fake.calls[-1][1]["startDate"] == "2024-01-12"  # only the missing days, one-day overlap


def test_revised_history_is_downloaded_again_not_spliced(tmp_path, fake):
    fake.histories["SPY"] = [bar(d, 100.0) for d in weekdays("2024-01-02", "2024-01-12")]
    _client(tmp_path).get_prices("SPY", "2024-01-01", "2024-01-31")
    fake.histories["SPY"] = [bar(d, 50.0) for d in weekdays("2024-01-02", "2024-01-19")]
    tiingo_mod.clear_process_cache()
    bars = _client(tmp_path, max_age_hours=0).get_prices("SPY", "2024-01-01", "2024-01-31")
    assert {p.close for p in bars} == {50.0}
    assert fake.calls[-1][1]["startDate"] == HISTORY_START


def test_history_is_pinned_within_a_process(tmp_path, fake):
    fake.histories["SPY"] = [bar("2024-01-02", 100.0)]
    c = _client(tmp_path, max_age_hours=0)
    c.get_prices("SPY", "2024-01-01", "2024-01-31")
    fake.histories["SPY"] = [bar("2024-01-02", 100.0), bar("2024-01-03", 50.0, split=2.0)]
    assert [p.close for p in c.get_prices("SPY", "2024-01-01", "2024-01-31")] == [100.0]
    assert len(fake.calls) == 1


def test_concurrent_first_requests_download_once(tmp_path, fake):
    fake.histories["KO"] = [bar(d, 60.0) for d in weekdays("2024-01-02", "2024-03-29")]
    results = []

    def work():
        results.append(len(_client(tmp_path).get_prices("KO", "2024-01-01", "2024-03-31")))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [len(fake.histories["KO"])] * 8 and len(fake.calls) == 1


def test_auth_failure_raises_with_key_redacted(tmp_path, fake):
    fake.scripted = [FakeResponse(401, text=f"Invalid token {KEY}")]
    with pytest.raises(TiingoClientError) as exc:
        _client(tmp_path).get_prices("AAPL", "2020-01-01", "2020-12-31")
    assert exc.value.status_code == 401 and KEY not in str(exc.value) and "***" in str(exc.value)


def test_rate_limit_retries_then_raises(tmp_path, fake):
    fake.histories["AAPL"] = AAPL
    fake.scripted = [FakeResponse(429)]
    assert len(_client(tmp_path).get_prices("AAPL", "2020-08-01", "2020-09-30")) == 5
    tiingo_mod.clear_process_cache()
    fake.scripted = [FakeResponse(503)] * 4
    with pytest.raises(TiingoClientError, match="retries"):
        _client(tmp_path).get_prices("MSFT", "2020-08-01", "2020-09-30")


def test_network_error_raises(tmp_path, fake):
    fake.scripted = [requests.ConnectionError("down")]
    with pytest.raises(TiingoClientError, match="down"):
        _client(tmp_path).get_prices("AAPL", "2020-01-01", "2020-12-31")


def test_missing_key_raises_before_any_request(tmp_path, fake, monkeypatch):
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    with pytest.raises(TiingoClientError, match="TIINGO_API_KEY"):
        TiingoClient(cache_dir=tmp_path).get_prices("AAPL", "2020-01-01", "2020-12-31")
    assert fake.calls == []


def test_key_is_sent_as_header_not_in_url(tmp_path, fake):
    fake.histories["AAPL"] = AAPL
    _client(tmp_path).get_prices("AAPL", "2020-08-01", "2020-09-30")
    assert fake.headers["Authorization"] == f"Token {KEY}"
    assert all(KEY not in url and KEY not in str(params) for url, params in fake.calls)


def test_offline_uses_store_and_never_fetches(tmp_path, fake):
    fake.histories["AAPL"] = AAPL
    _client(tmp_path).get_prices("AAPL", "2020-08-01", "2020-09-30")
    tiingo_mod.clear_process_cache()
    off = _client(tmp_path, offline=True, max_age_hours=0)
    assert len(off.get_prices("AAPL", "2020-08-01", "2020-09-30")) == 5
    with pytest.raises(TiingoClientError, match="offline"):
        off.get_prices("MSFT", "2020-08-01", "2020-09-30")
    assert len(fake.calls) == 1


def test_non_daily_interval_rejected(tmp_path, fake):
    with pytest.raises(ValueError):
        _client(tmp_path).get_prices("AAPL", "2020-01-01", "2020-12-31", interval="week")
