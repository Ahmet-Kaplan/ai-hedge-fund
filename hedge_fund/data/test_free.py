"""FreeDataClient tests — fake sources that count calls, a real SQLite store."""

import logging
from datetime import datetime, timedelta, timezone

import pytest

from hedge_fund.data.errors import DataSourceError
from hedge_fund.data.free import PRICE_HISTORY_START, FreeDataClient
from hedge_fund.data.models import Price
from hedge_fund.data.protocol import DataClient
from hedge_fund.data.store import MarketStore
from hedge_fund.data.test_fundamentals import company


def bar(day, close):
    return Price(open=close, high=close, low=close, close=close, volume=1, time=f"{day}T00:00:00Z")


class FakePrices:
    def __init__(self, closes):
        self.closes = closes      # {ticker: {date: close}}
        self.calls = []

    def fetch(self, tickers, start, end):
        self.calls.append((tuple(tickers), start, end))
        return {t: [bar(d, c) for d, c in sorted(self.closes.get(t, {}).items()) if start <= d <= end]
                for t in tickers if self.closes.get(t)}


class FakeSec:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def sync_company(self, ticker, store, *, now):
        self.calls.append((ticker, now))
        if self.fail:
            raise DataSourceError("SEC unreachable")
        store.upsert_company(ticker, "42", "Test Co", 3571, "Electronic Computers")
        store.replace_facts("42", [f for facts in company().values() for f in facts])
        store.replace_filings("42", [])
        store.mark_synced("42", facts_at=now, submissions_at=now)
        return "42"


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


CLOSES = {"TEST": {"2026-07-31": 50.0, "2026-08-03": 51.0}, "SPY": {"2026-08-03": 500.0}}


def client(tmp_path, prices=None, sec=None, clock=None):
    return FreeDataClient(MarketStore(tmp_path / "m.db"), prices or FakePrices(CLOSES), sec or FakeSec(),
                          now=clock or Clock())


def test_satisfies_the_data_client_protocol(tmp_path):
    assert isinstance(client(tmp_path), DataClient)


def test_prices_download_history_once_then_read_locally(tmp_path):
    prices = FakePrices(CLOSES)
    c = client(tmp_path, prices=prices)
    assert [p.close for p in c.get_prices("TEST", "2026-07-01", "2026-08-03")] == [50.0, 51.0]
    assert prices.calls == [(("TEST",), PRICE_HISTORY_START, "2026-08-03")]
    c.get_prices("TEST", "2026-07-31", "2026-07-31")
    assert len(prices.calls) == 1                       # inside the stored range
    c.get_prices("TEST", "2026-08-01", "2026-08-10")
    assert prices.calls[-1] == (("TEST",), "2026-08-04", "2026-08-10")   # only the missing tail


def test_prefetch_batches_tickers(tmp_path):
    prices = FakePrices(CLOSES)
    c = client(tmp_path, prices=prices)
    c.prefetch_prices(["TEST", "SPY"], "2026-08-03")
    assert prices.calls == [(("SPY", "TEST"), PRICE_HISTORY_START, "2026-08-03")]
    c.get_prices("SPY", "2026-08-03", "2026-08-03")
    assert len(prices.calls) == 1


def test_fundamentals_from_sec_with_market_cap_from_stored_prices(tmp_path):
    rows = client(tmp_path).get_financial_metrics("TEST", "2026-08-15", limit=2)
    assert [r.report_period for r in rows] == ["2026-06-30", "2025-12-31"]
    assert rows[0].market_cap == pytest.approx(10 * 50.0)   # close on the 2026-08-01 filing date's last session
    assert client(tmp_path).get_financial_metrics("TEST", "2026-07-15", limit=1)[0].report_period == "2025-12-31"
    with pytest.raises(ValueError, match="ttm"):
        client(tmp_path).get_financial_metrics("TEST", "2026-08-15", period="quarterly")


def test_sec_synced_once_and_not_refetched_for_past_dates(tmp_path):
    sec, clock = FakeSec(), Clock()
    c = client(tmp_path, sec=sec, clock=clock)
    c.get_financial_metrics("TEST", "2026-09-29")
    c.get_company_facts("TEST")
    assert len(sec.calls) == 1
    clock.now += timedelta(hours=30)
    fresh = client(tmp_path, sec=sec, clock=clock)       # new process, same database
    fresh.get_financial_metrics("TEST", "2026-08-15")    # a backtest date before the last sync
    assert len(sec.calls) == 1
    fresh.get_financial_metrics("TEST", "2026-10-01")    # live date, stored copy is stale
    assert len(sec.calls) == 2


def test_offline_uses_stored_data_but_fails_without_it(tmp_path, caplog):
    client(tmp_path).get_financial_metrics("TEST", "2026-09-29")
    clock = Clock()
    clock.now += timedelta(days=2)
    offline = client(tmp_path, sec=FakeSec(fail=True), clock=clock)
    with caplog.at_level(logging.WARNING):
        assert offline.get_financial_metrics("TEST", "2026-10-01")
    assert "SEC unreachable" in caplog.text
    with pytest.raises(DataSourceError):
        client(tmp_path, sec=FakeSec(fail=True)).get_company_facts("OTHER")


def test_company_facts_earnings_and_unused_endpoints(tmp_path):
    c = client(tmp_path)
    facts = c.get_company_facts("TEST")
    assert (facts.name, facts.sector, facts.industry, facts.sic_code) == (
        "Test Co", "Manufacturing", "Electronic Computers", "3571")
    assert c.get_earnings_history("TEST") == []          # fixture has no EPS history
    assert c.get_news("TEST", "2026-09-29") == [] and c.get_insider_trades("TEST", "2026-09-29") == []
    assert c.get_earnings("TEST") is None
    assert c.get_market_cap("TEST", "2026-08-15") == pytest.approx(500.0)
