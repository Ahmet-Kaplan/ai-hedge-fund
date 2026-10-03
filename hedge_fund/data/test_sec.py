"""SecSource tests — fake session and clock, no network."""

import pytest

from hedge_fund.data.errors import DataSourceError
from hedge_fund.data.sec import SecSource, sec_ticker, sic_sector
from hedge_fund.data.store import MarketStore


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, routes):
        self.headers = {}
        self.urls = []
        self._routes = {k: list(v) for k, v in routes.items()}

    def get(self, url, timeout=None):
        self.urls.append(url)
        for key, responses in self._routes.items():
            if key in url:
                return responses.pop(0)
        raise AssertionError(f"unexpected url {url}")


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
           "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire"}}
SUBMISSIONS = {"name": "Apple Inc.", "sic": "3571", "sicDescription": "Electronic Computers",
               "filings": {"recent": {
                   "accessionNumber": ["a1", "a2", "a3"], "filingDate": ["2026-07-30", "2026-07-31", "2026-07-15"],
                   "reportDate": ["2026-07-30", "2026-06-27", ""], "form": ["8-K", "10-Q", "4"],
                   "items": ["2.02,9.01", "", ""]}}}
FACTS = {"facts": {
    "us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [
            {"start": "2026-03-29", "end": "2026-06-27", "val": 100, "accn": "a2", "fy": 2026, "fp": "Q3", "form": "10-Q", "filed": "2026-07-31"},
            {"start": "2026-03-29", "end": "2026-06-27", "val": 99, "accn": "a9", "fy": 2026, "fp": "Q3", "form": "8-K", "filed": "2026-07-30"},
        ]}},
        "SomethingWeDoNotUse": {"units": {"USD": [
            {"end": "2026-06-27", "val": 1, "accn": "a2", "fy": 2026, "fp": "Q3", "form": "10-Q", "filed": "2026-07-31"}]}},
    },
    "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": "2026-07-17", "val": 14594180000, "accn": "a2", "fy": 2026, "fp": "Q3", "form": "10-Q", "filed": "2026-07-31"}]}}},
}}


def source(routes, clock=None):
    clock = clock or FakeClock()
    session = FakeSession(routes)
    return SecSource("Test User test@example.org", session=session, clock=clock.time, sleep=clock.sleep), session, clock


def test_requires_user_agent(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        SecSource(session=FakeSession({}))


def test_ticker_and_sector_helpers():
    assert sec_ticker("brk.b") == "BRK-B"
    assert sic_sector(3571) == "Manufacturing"
    assert sic_sector(6022) == "Finance, Insurance & Real Estate"
    assert sic_sector(None) is None


def test_sync_stores_company_filings_and_only_used_concepts(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    sec, session, _ = source({"company_tickers": [FakeResponse(payload=TICKERS)],
                              "submissions/CIK0000320193": [FakeResponse(payload=SUBMISSIONS)],
                              "companyfacts/CIK0000320193": [FakeResponse(payload=FACTS)]})
    sec.sync_company("AAPL", store, now="2026-09-30T10:00:00+00:00")
    assert store.company("AAPL") == {"ticker": "AAPL", "cik": "320193", "name": "Apple Inc.", "sic": 3571,
                                     "sic_description": "Electronic Computers"}
    assert [f["form"] for f in store.filings("320193", ["8-K", "10-Q", "10-K"])] == ["8-K", "10-Q"]
    facts = store.facts("320193", ["NetIncomeLoss", "EntityCommonStockSharesOutstanding", "SomethingWeDoNotUse"], "2026-12-31")
    assert [(f.value, f.form) for f in facts["NetIncomeLoss"]] == [(100, "10-Q")]   # 8-K facts dropped
    assert facts["EntityCommonStockSharesOutstanding"][0].value == 14594180000
    assert facts["SomethingWeDoNotUse"] == []
    assert store.sync_times("320193") == ("2026-09-30T10:00:00+00:00", "2026-09-30T10:00:00+00:00")
    assert session.headers["User-Agent"] == "Test User test@example.org"


def test_known_company_skips_the_ticker_map(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    store.upsert_company("AAPL", "320193", "Apple Inc.", 3571, "Electronic Computers")
    sec, session, _ = source({"submissions/CIK0000320193": [FakeResponse(payload=SUBMISSIONS)],
                              "companyfacts/CIK0000320193": [FakeResponse(payload=FACTS)]})
    sec.sync_company("AAPL", store, now="2026-09-30T10:00:00+00:00")
    assert not any("company_tickers" in u for u in session.urls)


def test_unknown_ticker_raises(tmp_path):
    sec, _, _ = source({"company_tickers": [FakeResponse(payload=TICKERS)]})
    with pytest.raises(DataSourceError, match="ZZZZ"):
        sec.sync_company("ZZZZ", MarketStore(tmp_path / "m.db"), now="2026-09-30T10:00:00+00:00")


def test_throttles_and_retries_rate_limits():
    clock = FakeClock()
    sec, _, _ = source({"company_tickers": [FakeResponse(429, {}), FakeResponse(payload=TICKERS)]}, clock)
    assert sec.ticker_map()["BRK-B"] == "1067983"
    assert clock.sleeps and clock.sleeps[0] >= 1.0   # backed off after the 429


def test_predecessor_registrant_history_is_merged(tmp_path, monkeypatch):
    from hedge_fund.data import sec as sec_module
    monkeypatch.setitem(sec_module.PREDECESSORS, "AAPL", ["999"])
    old_facts = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
        {"start": "2020-01-01", "end": "2020-03-31", "val": 7, "accn": "old1", "fy": 2020, "fp": "Q1", "form": "10-Q", "filed": "2020-05-01"}]}}}}}
    old_subs = {"name": "Old Apple", "sic": "3571", "sicDescription": "x", "filings": {"recent": {
        "accessionNumber": ["o1"], "filingDate": ["2020-04-30"], "reportDate": ["2020-04-30"], "form": ["8-K"], "items": ["2.02"]}}}
    store = MarketStore(tmp_path / "m.db")
    store.upsert_company("AAPL", "320193", "Apple Inc.", 3571, "Electronic Computers")
    sec, _, _ = source({"submissions/CIK0000320193": [FakeResponse(payload=SUBMISSIONS)],
                        "companyfacts/CIK0000320193": [FakeResponse(payload=FACTS)],
                        "submissions/CIK0000000999": [FakeResponse(payload=old_subs)],
                        "companyfacts/CIK0000000999": [FakeResponse(payload=old_facts)]})
    sec.sync_company("AAPL", store, now="2026-09-30T10:00:00+00:00")
    values = [f.value for f in store.facts("320193", ["NetIncomeLoss"], "2026-12-31")["NetIncomeLoss"]]
    assert values == [7, 100]
    assert [f["accn"] for f in store.filings("320193", ["8-K"])] == ["o1", "a1"]
