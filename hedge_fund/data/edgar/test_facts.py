"""TTM arithmetic and point-in-time resolution on synthetic facts."""

from hedge_fund.data.edgar.facts import Fact, FactStore


def _f(tag, start, end, val, filed, accn=None, form="10-Q", unit="USD"):
    return Fact(tag=tag, unit=unit, start=start, end=end, val=val,
                accn=accn or f"acc-{filed}", form=form, filed=filed, cik=1)


def _calendar_company(tag="NetCashProvidedByUsedInOperatingActivities"):
    """Cash-flow style: 10-Qs report only year-to-date values."""
    return [
        _f(tag, "2022-01-01", "2022-12-31", 400.0, "2023-02-15", form="10-K"),
        _f(tag, "2022-01-01", "2022-03-31", 90.0, "2022-04-28"),
        _f(tag, "2022-01-01", "2022-06-30", 190.0, "2022-07-28"),
        _f(tag, "2022-01-01", "2022-09-30", 295.0, "2022-10-27"),
        _f(tag, "2023-01-01", "2023-03-31", 110.0, "2023-04-27"),
        _f(tag, "2023-01-01", "2023-06-30", 230.0, "2023-07-27"),
    ]


def test_ttm_from_year_to_date_cash_flow():
    view = FactStore(_calendar_company()).view("2023-07-27")
    # YTD 2023H1 + FY2022 - YTD 2022H1 = 230 + 400 - 190
    assert view.ttm("operating_cash_flow", "2023-06-30") == 440.0


def test_ttm_at_fiscal_year_end_is_the_annual_value():
    view = FactStore(_calendar_company()).view("2023-02-15")
    assert view.ttm("operating_cash_flow", "2022-12-31") == 400.0


def test_derived_quarters_including_q4():
    view = FactStore(_calendar_company()).view("2023-07-27")
    assert view.quarter("operating_cash_flow", "2023-06-30") == (120.0, "2023-04-01")
    assert view.quarter("operating_cash_flow", "2022-12-31") == (105.0, "2022-10-01")  # FY - 9M


def test_ttm_from_four_quarters_when_prior_year_ytd_is_missing():
    tag = "Revenues"
    facts = [
        _f(tag, "2022-04-01", "2022-06-30", 10.0, "2022-07-28"),
        _f(tag, "2022-07-01", "2022-09-30", 11.0, "2022-10-27"),
        _f(tag, "2022-01-01", "2022-12-31", 50.0, "2023-02-15", form="10-K"),
        _f(tag, "2022-01-01", "2022-09-30", 33.0, "2023-02-15", form="10-K"),
        _f(tag, "2023-01-01", "2023-03-31", 14.0, "2023-04-27"),
    ]
    # No FY2022 nine-month-ago YTD for Q1 2023's prior year: falls back to
    # quarters: Q1'23 14 + Q4'22 (50-33=17) + Q3'22 11 + Q2'22 10
    view = FactStore(facts).view("2023-04-27")
    assert view.ttm("revenue", "2023-03-31") == 52.0


def test_facts_filed_later_are_invisible():
    view = FactStore(_calendar_company()).view("2023-07-26")  # day before the H1 10-Q
    assert view.ttm("operating_cash_flow", "2023-06-30") is None
    assert view.ttm("operating_cash_flow", "2023-03-31") == 420.0  # 110 + 400 - 90


def test_restatement_visible_only_from_its_filing_date():
    facts = _calendar_company() + [
        _f("NetCashProvidedByUsedInOperatingActivities", "2022-01-01", "2022-12-31", 380.0, "2023-08-01",
           form="10-K/A"),
    ]
    store = FactStore(facts)
    assert store.view("2023-07-31").ttm("operating_cash_flow", "2022-12-31") == 400.0
    assert store.view("2023-08-01").ttm("operating_cash_flow", "2022-12-31") == 380.0


def test_tag_fallback_is_per_period():
    facts = [
        _f("SalesRevenueGoodsNet", "2017-01-01", "2017-12-31", 35.0, "2018-02-20", form="10-K"),
        _f("Revenues", "2018-01-01", "2018-12-31", 32.0, "2019-02-20", form="10-K"),
        _f("SalesRevenueGoodsNet", "2018-01-01", "2018-12-31", 99.0, "2019-02-20", form="10-K"),
    ]
    view = FactStore(facts).view("2019-02-20")
    assert view.ttm("revenue", "2017-12-31") == 35.0  # only the old tag exists
    assert view.ttm("revenue", "2018-12-31") == 32.0  # higher-priority tag wins


def test_units_and_instants_do_not_mix():
    facts = [
        _f("StockholdersEquity", None, "2022-12-31", 7.0, "2023-02-15", form="10-K"),
        _f("EarningsPerShareDiluted", "2022-01-01", "2022-12-31", 2.5, "2023-02-15", form="10-K", unit="USD/shares"),
        _f("EarningsPerShareDiluted", "2022-01-01", "2022-12-31", 999.0, "2023-02-15", form="10-K", unit="USD"),
    ]
    view = FactStore(facts).view("2023-02-15")
    assert view.instant("equity", "2022-12-31") == 7.0
    assert view.ttm("equity", "2022-12-31") is None
    assert view.ttm("eps_diluted", "2022-12-31") == 2.5


def test_filings_one_row_per_period_earliest_wins_and_amendments_excluded():
    facts = [
        _f("Revenues", "2022-01-01", "2022-03-31", 1.0, "2022-04-28", accn="a"),
        _f("Revenues", "2022-01-01", "2022-03-31", 1.0, "2022-05-10", accn="b"),
        _f("Revenues", "2022-01-01", "2022-03-31", 2.0, "2022-06-01", accn="c", form="10-Q/A"),
        _f("Revenues", "2021-01-01", "2021-03-31", 1.0, "2022-04-28", accn="a"),  # comparative
    ]
    filings = FactStore(facts).filings()
    assert [(f.accn, f.report_period, f.filed) for f in filings] == [("a", "2022-03-31", "2022-04-28")]


def test_predecessor_cutoff_drops_later_filings():
    doc = {"cik": 5, "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2015-01-01", "end": "2015-03-31", "val": 1, "accn": "x", "form": "10-Q", "filed": "2015-04-20"},
        {"start": "2015-07-01", "end": "2015-09-30", "val": 1, "accn": "y", "form": "10-Q", "filed": "2015-11-01"},
    ]}}}}}
    store = FactStore.from_companyfacts(doc, filed_until="2015-10-01")
    assert [f.accn for f in store.facts] == ["x"]
