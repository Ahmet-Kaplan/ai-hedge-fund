"""make_data_client / CompositeDataClient — routing, configuration, and the
EDGAR <- Tiingo valuation path end to end (offline, synthetic prices)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from hedge_fund.data import tiingo as tiingo_mod
from hedge_fund.data.edgar import EdgarClient
from hedge_fund.data.factory import CompositeDataClient, make_data_client, required_data_env
from hedge_fund.data.protocol import DataClient
from hedge_fund.data.test_tiingo import AAPL, FakeTiingo, bar, weekdays
from hedge_fund.data.tiingo import TiingoClient

EDGAR_FIXTURES = Path(__file__).parent / "edgar" / "testdata"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("HEDGE_FUND_DATA_PROVIDER", "HEDGE_FUND_DATA_SUPPLEMENT"):
        monkeypatch.delenv(name, raising=False)
    tiingo_mod.clear_process_cache()
    yield
    tiingo_mod.clear_process_cache()


@pytest.fixture
def server(monkeypatch):
    fake = FakeTiingo({})
    monkeypatch.setattr(tiingo_mod.requests, "Session", lambda: fake)
    return fake


def _composite(tmp_path, server_histories=None, server=None):
    if server is not None and server_histories:
        server.histories.update(server_histories)
    edgar_dir = tmp_path / "edgar"
    shutil.copytree(EDGAR_FIXTURES, edgar_dir)
    tiingo = TiingoClient(api_key="fixture-key", cache_dir=tmp_path / "tiingo")
    edgar = EdgarClient(cache_dir=edgar_dir, offline=True, today="2026-09-30", price_source=tiingo)
    return make_data_client(tiingo=tiingo, edgar=edgar)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_default_provider_is_tiingo_plus_edgar(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    with make_data_client() as client:
        assert isinstance(client, CompositeDataClient)
        assert isinstance(client, DataClient)
        assert isinstance(client.prices, TiingoClient) and isinstance(client.fundamentals, EdgarClient)
        assert client.supplemental is None
        assert client.fundamentals._prices is client.prices  # EDGAR valuation uses Tiingo raw closes
        assert client.request_counts() == {"tiingo": 0, "sec": 0}
    assert required_data_env() == ["TIINGO_API_KEY", "SEC_USER_AGENT"]


def test_financial_datasets_only_when_selected(monkeypatch):
    from hedge_fund.data.client import FDClient
    monkeypatch.setenv("HEDGE_FUND_DATA_PROVIDER", "financial-datasets")
    assert required_data_env() == ["FINANCIAL_DATASETS_API_KEY"]
    with make_data_client() as client:
        assert isinstance(client._raw, FDClient) and hasattr(client, "get_prices")
    monkeypatch.setenv("HEDGE_FUND_DATA_PROVIDER", "yahoo")
    with pytest.raises(ValueError, match="HEDGE_FUND_DATA_PROVIDER"):
        make_data_client()


def test_supplement_is_opt_in(monkeypatch):
    monkeypatch.setenv("HEDGE_FUND_DATA_SUPPLEMENT", "financial-datasets")
    assert required_data_env() == ["TIINGO_API_KEY", "SEC_USER_AGENT", "FINANCIAL_DATASETS_API_KEY"]
    with make_data_client() as client:
        assert client.supplemental is not None
    monkeypatch.setenv("HEDGE_FUND_DATA_SUPPLEMENT", "other")
    with pytest.raises(ValueError):
        required_data_env()


def test_unexpected_factory_arguments_rejected():
    with pytest.raises(TypeError):
        make_data_client(tiingo=object(), edgar=object(), bogus=1)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class _Spy:
    def __init__(self, label):
        self.label, self.calls, self.requests, self.closed = label, [], 0, False

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append(name)
            return self.label
        return call

    def close(self):
        self.closed = True


def test_each_method_goes_to_its_owner():
    prices, fundamentals, supplement = _Spy("tiingo"), _Spy("edgar"), _Spy("fd")
    with CompositeDataClient(prices, fundamentals, supplement) as c:
        assert c.get_prices("A", "2020-01-01", "2020-02-01") == "tiingo"
        assert c.raw_closes("A", "x", "y") == c.split_events("A", "x", "y") == c.dividends("A", "x", "y") == "tiingo"
        assert c.get_financial_metrics("A", "2020-01-01") == c.get_company_facts("A") == "edgar"
        assert c.get_market_cap("A", "2020-01-01") == "edgar"
        assert c.get_news("A", "x") == c.get_insider_trades("A", "x") == c.get_earnings("A") == "fd"
        assert c.get_earnings_history("A") == "fd"
    assert prices.closed and fundamentals.closed and supplement.closed


def test_uncovered_data_raises_with_instructions():
    c = CompositeDataClient(_Spy("tiingo"), _Spy("edgar"))
    for call in (lambda: c.get_news("A", "x"), lambda: c.get_insider_trades("A", "x"),
                 lambda: c.get_earnings("A"), lambda: c.get_earnings_history("A")):
        with pytest.raises(NotImplementedError, match="HEDGE_FUND_DATA_SUPPLEMENT"):
            call()


# ---------------------------------------------------------------------------
# EDGAR valuation from Tiingo raw closes, end to end
# ---------------------------------------------------------------------------

def test_market_cap_and_pe_from_tiingo_raw_close_across_a_split(tmp_path, server):
    # AAPL 10-Q for 2020-06-27 filed 2020-07-31 (pre-split); split 4:1 on 2020-08-31
    history = [bar("2020-07-30", 384.76), bar("2020-07-31", 425.04)] + AAPL
    with _composite(tmp_path, {"AAPL": history}, server) as c:
        before = next(r for r in c.get_financial_metrics("AAPL", "2020-08-30", limit=3)
                      if r.report_period == "2020-06-27")
        after = next(r for r in c.get_financial_metrics("AAPL", "2020-09-30", limit=3)
                     if r.report_period == "2020-06-27")
        fv = next(v for k, v in c.fundamentals._values.items() if k[1] and v.filing.report_period == "2020-06-27")
        assert fv.price == 425.04 and fv.price_date == "2020-07-31"   # raw, not split-adjusted (106.26)
        assert before.market_cap == pytest.approx(fv.shares * 425.04)
        assert 1.6e12 < before.market_cap < 2.0e12                    # ~$1.8T on 2020-07-31
        # P/E = filing-date raw close / TTM diluted EPS as reported then
        assert before.price_to_earnings_ratio == pytest.approx(425.04 / before.earnings_per_share)
        assert after.market_cap == before.market_cap
        assert after.earnings_per_share == pytest.approx(before.earnings_per_share / 4)
        # the protocol price series is split-adjusted for the same day
        assert c.get_prices("AAPL", "2020-07-31", "2020-07-31")[0].close == pytest.approx(425.04 / 4)
        assert c.request_counts()["sec"] == 0 and c.request_counts()["tiingo"] == 1


def test_pe_is_price_over_diluted_eps(tmp_path, server):
    with _composite(tmp_path, {"KO": [bar("2020-02-24", 58.65)]}, server) as c:
        r = c.get_financial_metrics("KO", "2020-02-28", limit=1)[0]
        assert r.market_cap == pytest.approx(4_290_276_067 * 58.65)
        assert r.price_to_earnings_ratio == pytest.approx(58.65 / 2.07)


def test_ticker_without_prices_yields_no_valuation_not_an_error(tmp_path, server):
    with _composite(tmp_path, {}, server) as c:
        r = c.get_financial_metrics("KO", "2020-02-28", limit=1)[0]
        assert r.market_cap is None and r.net_margin is not None
        assert c.get_prices("KO", "2020-01-01", "2020-12-31") == []


def test_delisted_and_successor_names(tmp_path, server):
    histories = {
        "TWTR": [bar(d, 50.0) for d in weekdays("2022-10-03", "2022-10-27")],
        "FRCB": [bar("2023-04-27", 3.5), bar("2023-05-02", 0.3)],
        "SIVBQ": [bar("2023-03-28", 0.5)],
    }
    with _composite(tmp_path, histories, server) as c:
        assert c.get_prices("TWTR", "2022-10-01", "2022-12-31")[-1].time[:10] == "2022-10-27"
        assert c.get_financial_metrics("TWTR", "2022-12-31", limit=1)[0].report_period == "2022-06-30"
        assert len(c.get_prices("FRC", "2023-04-01", "2023-06-01")) == 2   # via FRCB
        assert c.get_financial_metrics("FRC", "2023-03-01") == []         # FDIC filer: no SEC data
        assert c.get_financial_metrics("SIVBQ", "2024-01-31", limit=1)[0].report_period == "2022-12-31"
        assert c.get_prices("ATVI", "2023-01-01", "2023-12-31") == []      # unknown to this fake: empty


# ---------------------------------------------------------------------------
# The pipeline keeps running when a name has no data
# ---------------------------------------------------------------------------

def test_backtest_skips_a_ticker_without_prices(tmp_path, server, monkeypatch):
    from hedge_fund.backtesting.fund import backtest_fund
    from hedge_fund.backtesting.test_fund import FakeAnalyst
    from hedge_fund.fund.spec import Fund, FundSpec
    from hedge_fund.signals import ALPHA_MODEL_REGISTRY

    monkeypatch.setitem(ALPHA_MODEL_REGISTRY, "a", FakeAnalyst)

    days = weekdays("2024-06-03", "2024-06-21")
    histories = {"SPY": [bar(d, 500.0) for d in days], "AAPL": [bar(d, 200.0) for d in days]}
    spec = FundSpec(schema_version=2, name="t", capital=100_000.0, rebalance="weekly",
                    strategies=[{"name": "solo", "models": [{"name": "a"}], "blend": {"mode": "long_short"}}],
                    risk={"max_position_pct": 0.5, "max_gross_exposure": 1.0})
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0, "ZZZZ": 1.0})]})
    with _composite(tmp_path, histories, server) as c:
        result = backtest_fund(fund, "2024-06-03", "2024-06-21", c, ["AAPL", "ZZZZ"])
    assert result.records and all("ZZZZ" in {s.ticker for s in r.skipped} for r in result.records)
    assert result.records[0].positions == {"AAPL": 250}
