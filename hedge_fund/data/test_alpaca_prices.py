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


class TickingClock:
    """Advances on every read, so the pacing interval is always already met."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


class TestRateLimiting:
    """The free tier rate-limits, and a backtest walks many tickers.

    Found by running the free path end to end: a real backtest died on
    `HTTP 429: too many requests`, because the SEC source retried and this one
    did not.
    """

    def _source(self, responses, sleeps, clock=None):
        """*clock* defaults to one that advances, so pacing adds no wait and the
        sleeps observed are the backoff alone."""
        return AlpacaPriceSource(
            "k", "s", session=FakeSession(*responses),
            clock=clock or TickingClock(), sleep=sleeps.append,
        )

    def test_a_429_is_retried_rather_than_ending_the_run(self):
        sleeps: list[float] = []
        source = self._source(
            [FakeResponse(429, {"message": "too many requests"}),
             FakeResponse(payload={"bars": {"AAPL": [bar("2026-09-28", 10.0)]}})],
            sleeps,
        )

        out = source.fetch(["AAPL"], "2026-09-01", "2026-09-30")

        assert [p.close for p in out["AAPL"]] == [10.0]
        assert sleeps == [1.0]           # backed off before retrying

    def test_a_5xx_is_retried_too(self):
        sleeps: list[float] = []
        source = self._source([FakeResponse(503, {}), FakeResponse(payload={"bars": {}})], sleeps)
        assert source.fetch(["AAPL"], "2026-09-01", "2026-09-30") == {}
        assert sleeps == [1.0]

    def test_the_backoff_grows_then_gives_up(self):
        sleeps: list[float] = []
        source = self._source([FakeResponse(429, {"message": "slow down"})] * 4, sleeps)

        with pytest.raises(DataSourceError, match="429"):
            source.fetch(["AAPL"], "2026-09-01", "2026-09-30")

        assert sleeps == [1.0, 2.0, 4.0]

    def test_a_client_error_is_not_retried(self):
        """401 or 404 will not fix itself, and retrying only delays the report."""
        sleeps: list[float] = []
        source = self._source([FakeResponse(401, {"message": "unauthorized"})], sleeps)

        with pytest.raises(DataSourceError, match="401"):
            source.fetch(["AAPL"], "2026-09-01", "2026-09-30")

        assert sleeps == []

    def test_requests_are_paced_so_the_limit_is_rarely_reached(self):
        """The cheapest way to handle a rate limit is not to hit it."""
        sleeps: list[float] = []
        source = self._source(
            [FakeResponse(payload={"bars": {}}), FakeResponse(payload={"bars": {}})],
            sleeps, clock=lambda: 100.0,          # frozen: the second request is immediate
        )

        source.fetch(["AAPL"], "2026-09-01", "2026-09-30")   # no wait: nothing before it
        source.fetch(["MSFT"], "2026-09-01", "2026-09-30")   # immediate → waits

        assert sleeps == [pytest.approx(0.35)]

    def test_a_retry_does_not_carry_a_stale_page_token_forward(self):
        """The retry re-sends the same request; only a good page advances."""
        sleeps: list[float] = []
        session = FakeSession(
            FakeResponse(429, {"message": "slow down"}),
            FakeResponse(payload={"bars": {"AAPL": [bar("2026-09-28", 1.0)]}}),
        )
        source = AlpacaPriceSource("k", "s", session=session,
                                   clock=TickingClock(), sleep=sleeps.append)

        source.fetch(["AAPL"], "2026-09-01", "2026-09-30")

        assert "page_token" not in session.calls[0]
        assert "page_token" not in session.calls[1]
