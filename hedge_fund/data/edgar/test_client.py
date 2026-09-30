"""EdgarClient against committed SEC fixtures — offline, no key, no network.

Fixtures (testdata/) are real SEC companyfacts/submissions documents trimmed
by `trim_companyfacts`, laid out exactly like the on-disk cache. Prices are
synthetic (StaticRawPrices); no vendor price data is committed.
"""

import gzip
import json
import shutil
from pathlib import Path

import pytest

from hedge_fund.data.edgar import EdgarClient, EdgarClientError, StaticRawPrices
from hedge_fund.data.protocol import DataClient
from hedge_fund.features.snapshot import InsufficientData, build_snapshot

TESTDATA = Path(__file__).with_name("testdata")


@pytest.fixture
def cache(tmp_path):
    dst = tmp_path / "edgar"
    shutil.copytree(TESTDATA, dst)
    return dst


def _client(cache, **kwargs):
    return EdgarClient(cache_dir=cache, offline=True, today="2026-09-30", **kwargs)


def _row(rows, report_period):
    return next(r for r in rows if r.report_period == report_period)


# ---------------------------------------------------------------------------
# Protocol and graceful absence
# ---------------------------------------------------------------------------

def test_satisfies_data_client_protocol(cache):
    assert isinstance(_client(cache), DataClient)


def test_unsupported_methods_raise_and_prices_delegate(cache):
    c = _client(cache)
    for call in (lambda: c.get_news("KO", "2020-01-01"), lambda: c.get_insider_trades("KO", "2020-01-01"),
                 lambda: c.get_earnings("KO"), lambda: c.get_earnings_history("KO"),
                 lambda: c.get_prices("KO", "2020-01-01", "2020-01-31")):
        with pytest.raises(NotImplementedError):
            call()

    class Prices:
        def get_prices(self, ticker, start_date, end_date, **kwargs):
            return ["bar"]

    assert _client(cache, price_client=Prices()).get_prices("KO", "2020-01-01", "2020-01-31") == ["bar"]


@pytest.mark.parametrize("ticker", ["FRC", "FRCB", "ZZZZ"])
def test_no_sec_fundamentals_returns_empty_not_error(cache, ticker):
    c = _client(cache)
    assert c.get_financial_metrics(ticker, "2023-01-31") == []
    assert c.get_company_facts(ticker) is None
    assert c.get_market_cap(ticker, "2023-01-31") is None
    with pytest.raises(InsufficientData):  # the agent abstains; nothing crashes
        build_snapshot(ticker, "2023-01-31", c)


def test_unsupported_period_raises(cache):
    with pytest.raises(ValueError):
        _client(cache).get_financial_metrics("KO", "2020-01-01", period="quarterly")


def test_offline_cache_miss_raises_instead_of_fetching(cache):
    shutil.rmtree(cache / "companyfacts")
    with pytest.raises(EdgarClientError, match="offline"):
        _client(cache).get_financial_metrics("KO", "2020-01-01")


# ---------------------------------------------------------------------------
# Values against published figures
# ---------------------------------------------------------------------------

def test_coca_cola_fy2019(cache):
    rows = _client(cache).get_financial_metrics("KO", "2020-02-28", limit=5)
    r = rows[0]
    assert (r.report_period, r.filing_date, r.period) == ("2019-12-31", "2020-02-24", "ttm")
    assert r.net_margin == pytest.approx(8_920 / 37_266, rel=1e-6)
    assert r.earnings_per_share == pytest.approx(2.07)
    assert r.revenue_growth == pytest.approx(37_266 / 34_300 - 1, rel=1e-6)
    assert [x.filing_date for x in rows] == sorted((x.filing_date for x in rows), reverse=True)


def test_apple_fy2024(cache):
    r = _client(cache).get_financial_metrics("AAPL", "2024-12-31", limit=1)[0]
    assert r.report_period == "2024-09-28"
    assert r.net_margin == pytest.approx(93_736 / 391_035, rel=1e-6)
    assert r.earnings_per_share == pytest.approx(6.08)


def test_scale_error_in_weighted_shares_is_rejected(cache):
    # KO's Q1 2019 10-Q tagged diluted weighted shares as 4,306 (not 4.306bn)
    r = _row(_client(cache).get_financial_metrics("KO", "2019-06-01", limit=2), "2019-03-29")
    assert 1.0 < r.free_cash_flow_per_share < 2.0


def test_bank_revenue_and_debt(cache):
    jpm = _client(cache).get_financial_metrics("JPM", "2024-12-31", limit=1)[0]
    assert 0.25 < jpm.net_margin < 0.40           # net revenue, not gross interest income
    assert 1.0 < jpm.debt_to_equity < 1.6         # long-term debt incl. current + short-term borrowings
    svb = _client(cache).get_financial_metrics("SIVB", "2023-06-30", limit=1)[0]
    assert 0.1 < svb.net_margin < 0.5             # not the ASC 606 fee-only tag


def test_notes_only_debt(cache):
    twtr = _client(cache).get_financial_metrics("TWTR", "2022-12-31", limit=1)[0]
    assert 0.5 < twtr.debt_to_equity < 1.2        # convertible + senior notes


# ---------------------------------------------------------------------------
# TTM: year-to-date cash flow on real filings
# ---------------------------------------------------------------------------

def test_ytd_cash_flow_ttm_matches_quarter_sum(cache):
    c = _client(cache)
    ident = c.resolve("AAPL", "2024-08-02")
    store = c._store(ident)
    filing = next(f for f in store.filings() if f.report_period == "2024-06-29")
    view = store.view(filing.filed)
    via_ytd = view.ttm("operating_cash_flow", "2024-06-29")
    via_quarters = view._ttm_from_quarters("operating_cash_flow", "2024-06-29")
    assert via_ytd is not None and via_ytd == pytest.approx(via_quarters)
    # the 10-Q itself carries only a nine-month cash-flow value
    assert view.quarter("operating_cash_flow", "2024-06-29")[1] == "2024-03-31"


def test_ttm_at_fiscal_year_end_equals_annual(cache):
    c = _client(cache)
    store = c._store(c.resolve("AAPL", "2024-12-31"))
    view = store.view("2024-11-01")
    assert view.ttm("operating_cash_flow", "2024-09-28") == 118_254_000_000


# ---------------------------------------------------------------------------
# Point-in-time
# ---------------------------------------------------------------------------

SWEEP = ["2012-03-01", "2014-06-30", "2016-08-31", "2019-04-24", "2019-04-25", "2020-02-28",
         "2022-10-27", "2023-03-10", "2024-12-31"]


@pytest.mark.parametrize("ticker", ["KO", "JPM", "AAPL", "GOOGL", "BRK.B", "TWTR", "ATVI", "SIVB"])
def test_no_row_filed_after_end_date(cache, ticker):
    c = _client(cache)
    for as_of in SWEEP:
        for r in c.get_financial_metrics(ticker, as_of, limit=40):
            assert r.filing_date <= as_of


def _truncate(cache: Path, dst: Path, cutoff: str) -> Path:
    """A copy of the cache holding only facts filed on or before *cutoff*."""
    shutil.copytree(cache, dst)
    for path in (dst / "companyfacts").glob("*.json.gz"):
        with gzip.open(path, "rt") as fh:
            payload = json.load(fh)
        for ns in payload["data"]["facts"].values():
            for body in ns.values():
                for unit, rows in body["units"].items():
                    body["units"][unit] = [r for r in rows if r["filed"] <= cutoff]
        with gzip.open(path, "wt") as fh:
            json.dump(payload, fh)
    return dst


@pytest.mark.parametrize("ticker, as_of", [
    ("KO", "2019-04-24"), ("KO", "2020-02-28"), ("AAPL", "2020-09-30"), ("JPM", "2023-03-10"),
    ("GOOGL", "2016-06-30"), ("BRK.B", "2016-08-31"), ("TWTR", "2022-12-31"), ("SIVB", "2023-03-10"),
])
def test_results_do_not_depend_on_anything_filed_later(cache, tmp_path, ticker, as_of):
    """The look-ahead test: deleting every fact filed after *as_of* from the
    source data must not change a single value returned as of *as_of*."""
    prices = StaticRawPrices(closes={ticker: {"2000-01-01": 1.0}})
    full = _client(cache, price_source=prices).get_financial_metrics(ticker, as_of, limit=20)
    cut = _client(_truncate(cache, tmp_path / "cut", as_of), price_source=prices)
    assert full, "sweep point should have data"
    assert [r.model_dump() for r in full] == [r.model_dump() for r in cut.get_financial_metrics(ticker, as_of, limit=20)]


def test_filing_becomes_visible_on_its_filing_date(cache):
    c = _client(cache)
    assert c.get_financial_metrics("KO", "2020-02-23", limit=1)[0].report_period == "2019-09-27"
    assert c.get_financial_metrics("KO", "2020-02-24", limit=1)[0].report_period == "2019-12-31"


def test_rows_reflect_knowledge_at_their_own_filing(cache):
    """A later restatement does not rewrite an earlier row."""
    c = _client(cache)
    early = _row(c.get_financial_metrics("KO", "2019-06-01", limit=3), "2019-03-29")
    late = _row(c.get_financial_metrics("KO", "2021-06-01", limit=12), "2019-03-29")
    assert early.model_dump() == late.model_dump()


# ---------------------------------------------------------------------------
# Share classes and registrant lineage
# ---------------------------------------------------------------------------

def test_berkshire_class_b_per_share_and_market_cap(cache):
    prices = StaticRawPrices(closes={"BRK.B": {"2016-08-05": 145.0}, "BRK.A": {"2016-08-05": 217_500.0}})
    b = _client(cache, price_source=prices).get_financial_metrics("BRK.B", "2016-08-31", limit=1)[0]
    b_shares = 788_894 * 1500 + 1_282_442_561
    assert b.market_cap == pytest.approx(b_shares * 145.0)
    assert 100 < b.book_value_per_share < 115      # ~$160k per A share / 1500
    # EPS = NI / B-equivalent shares = ROE x book value per B share
    assert b.earnings_per_share == pytest.approx(b.return_on_equity * b.book_value_per_share)
    a = _client(cache, price_source=prices).get_financial_metrics("BRK.A", "2016-08-31", limit=1)[0]
    assert a.book_value_per_share == pytest.approx(b.book_value_per_share * 1500)
    assert a.market_cap == pytest.approx(b.market_cap)


def test_berkshire_single_class_dei_value_is_never_used(cache):
    # 2009-2011 dei facts count Class A only; without a cover page, no guess
    rows = _client(cache).get_financial_metrics("BRK.B", "2011-06-30", limit=4)
    assert rows and all(r.book_value_per_share is None and r.earnings_per_share is None for r in rows)
    assert all(r.return_on_equity is not None for r in rows)  # non-share metrics still present


def test_alphabet_joins_google_inc_history(cache):
    c = _client(cache)
    rows = c.get_financial_metrics("GOOGL", "2016-06-30", limit=8)
    assert [r.report_period for r in rows[:5]] == ["2016-03-31", "2015-12-31", "2015-09-30", "2015-06-30", "2015-03-31"]
    assert all(r.revenue_growth is not None and r.earnings_per_share is not None for r in rows[:5])
    before = c.get_financial_metrics("GOOGL", "2014-06-30", limit=8)  # before GOOGL/Alphabet existed
    assert len(before) >= 4 and before[0].report_period == "2014-03-31"


def test_predecessor_facts_stop_at_reorganization(cache):
    c = _client(cache)
    store = c._store(c.resolve("GOOGL", "2020-01-01"))
    assert all(f.filed <= "2015-10-01" for f in store.facts if f.cik == 1288776)


# ---------------------------------------------------------------------------
# Delisted names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ticker, as_of, last_period, active", [
    ("TWTR", "2023-06-30", "2022-06-30", False),
    ("ATVI", "2024-01-31", "2023-06-30", False),
    ("SIVB", "2023-06-30", "2022-12-31", False),
    ("SIVBQ", "2024-01-31", "2022-12-31", True),  # OTC successor: no end date recorded
])
def test_delisted_history_is_available(cache, ticker, as_of, last_period, active):
    c = _client(cache)
    rows = c.get_financial_metrics(ticker, as_of, limit=8)
    assert len(rows) >= 4 and rows[0].report_period == last_period
    snap = build_snapshot(ticker, as_of, c)
    assert snap.periods[0].report_period == last_period
    facts = c.get_company_facts(ticker)
    assert facts is not None and facts.is_active is active


def test_company_facts_for_listed_name(cache):
    f = _client(cache).get_company_facts("KO")
    assert f.cik == "21344" and f.is_active and f.sector == "Manufacturing" and f.industry


# ---------------------------------------------------------------------------
# Market cap, P/E and split basis
# ---------------------------------------------------------------------------

def test_market_cap_and_pe_use_filing_date_raw_close(cache):
    # synthetic close; filing 2020-02-24 has none, so the prior session is used
    prices = StaticRawPrices(closes={"KO": {"2020-02-21": 50.0, "2020-02-25": 999.0}})
    r = _client(cache, price_source=prices).get_financial_metrics("KO", "2020-02-28", limit=1)[0]
    shares = 4_290_276_067  # dei cover shares of this 10-K
    assert r.market_cap == pytest.approx(shares * 50.0)
    assert r.price_to_earnings_ratio == pytest.approx(shares * 50.0 / 8_920e6)
    assert r.price_to_book_ratio is not None and r.free_cash_flow_yield is not None
    assert _client(cache, price_source=prices).get_market_cap("KO", "2020-02-28") == pytest.approx(r.market_cap)


def test_stale_or_missing_price_gives_no_valuation(cache):
    prices = StaticRawPrices(closes={"KO": {"2020-02-10": 50.0}})  # > 7 days before filing
    r = _client(cache, price_source=prices).get_financial_metrics("KO", "2020-02-28", limit=1)[0]
    assert r.market_cap is None and r.price_to_earnings_ratio is None
    assert _client(cache).get_financial_metrics("KO", "2020-02-28", limit=1)[0].market_cap is None


def test_negative_earnings_give_no_pe(cache):
    prices = StaticRawPrices(closes={"TWTR": {"2022-07-26": 40.0}})
    r = _client(cache, price_source=prices).get_financial_metrics("TWTR", "2022-12-31", limit=1)[0]
    assert r.market_cap is not None and r.price_to_earnings_ratio is None


def test_per_share_restated_after_split_but_market_cap_is_not(cache):
    closes = {"AAPL": {"2020-07-31": 400.0}}
    before = StaticRawPrices(closes=closes)
    after = StaticRawPrices(closes=closes, splits={"AAPL": {"2020-08-31": 4.0}})
    pre = _row(_client(cache, price_source=before).get_financial_metrics("AAPL", "2020-08-30", limit=2), "2020-06-27")
    post = _row(_client(cache, price_source=after).get_financial_metrics("AAPL", "2020-09-30", limit=2), "2020-06-27")
    pre_same_date = _row(_client(cache, price_source=after).get_financial_metrics("AAPL", "2020-08-30", limit=2), "2020-06-27")
    assert pre_same_date.earnings_per_share == pre.earnings_per_share     # split not yet effective
    for name in ("earnings_per_share", "book_value_per_share", "free_cash_flow_per_share"):
        assert getattr(post, name) == pytest.approx(getattr(pre, name) / 4)
    assert post.market_cap == pre.market_cap and post.price_to_earnings_ratio == pre.price_to_earnings_ratio


# ---------------------------------------------------------------------------
# Snapshot integration
# ---------------------------------------------------------------------------

def test_buffett_snapshot_builds_from_edgar(cache):
    prices = StaticRawPrices(closes={"KO": {"2020-02-24": 58.0, "2019-10-24": 54.0}})
    snap = build_snapshot("KO", "2020-02-28", _client(cache, price_source=prices))
    assert len(snap.periods) == 20
    assert snap.market_cap_latest == pytest.approx(4_290_276_067 * 58.0)
    assert snap.roe_avg is not None and snap.bvps_cagr is not None
    text = snap.render(blind=True)
    assert "KO" not in text and "2019" not in text
