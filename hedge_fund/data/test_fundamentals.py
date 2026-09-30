"""SEC facts → point-in-time metrics, on a synthetic calendar-year company.

Hand-computed TTM at 2026-06-30 (YTD + prior FY − prior YTD):
  revenue 220 + 400 − 190 = 430   net income 44 + 80 − 38 = 86
  gross profit 110 + 200 − 95 = 215   operating income 55 + 100 − 47 = 108
  operating cash flow 70 + 120 − 55 = 135   capex 12 + 20 − 9 = 23
A year earlier (2025-06-30): revenue 190 + 360 − 170 = 380.
"""

from collections import defaultdict

import pytest

from hedge_fund.data.fundamentals import metrics_rows, quarter_ends, quarterly_values, ttm, latest_by_period
from hedge_fund.data.store import Fact


def f(concept, start, end, value, filed, form="10-Q", unit="USD"):
    return Fact(concept=concept, unit=unit, start=start, end=end, value=value, fy=None, fp=None,
                form=form, filed=filed, accn=f"{concept}-{end}-{filed}")


def company():
    rows = []
    for concept, fy24, h1_24, h1_25, fy25, h1_26 in [
        ("RevenueFromContractWithCustomerExcludingAssessedTax", 360, 170, 190, 400, 220),
        ("GrossProfit", None, None, 95, 200, 110),
        ("OperatingIncomeLoss", None, None, 47, 100, 55),
        ("NetIncomeLoss", None, None, 38, 80, 44),
        ("NetCashProvidedByUsedInOperatingActivities", None, None, 55, 120, 70),
        ("PaymentsToAcquirePropertyPlantAndEquipment", None, None, 9, 20, 12),
    ]:
        if fy24 is not None:
            rows += [f(concept, "2024-01-01", "2024-12-31", fy24, "2025-02-10", "10-K"),
                     f(concept, "2024-01-01", "2024-06-30", h1_24, "2024-08-01")]
        rows += [f(concept, "2025-01-01", "2025-06-30", h1_25, "2025-08-01"),
                 f(concept, "2025-01-01", "2025-12-31", fy25, "2026-02-10", "10-K"),
                 f(concept, "2026-01-01", "2026-06-30", h1_26, "2026-08-01")]
    rows += [
        f("RevenueFromContractWithCustomerExcludingAssessedTax", "2026-04-01", "2026-06-30", 115, "2026-08-01"),
        f("StockholdersEquity", None, "2026-06-30", 500, "2026-08-01"),
        f("AssetsCurrent", None, "2026-06-30", 300, "2026-08-01"),
        f("LiabilitiesCurrent", None, "2026-06-30", 150, "2026-08-01"),
        f("LongTermDebtNoncurrent", None, "2026-06-30", 90, "2026-08-01"),
        f("LongTermDebtCurrent", None, "2026-06-30", 10, "2026-08-01"),
        f("WeightedAverageNumberOfDilutedSharesOutstanding", "2026-04-01", "2026-06-30", 10, "2026-08-01", unit="shares"),
        f("EntityCommonStockSharesOutstanding", None, "2026-07-20", 10, "2026-08-01", unit="shares"),
    ]
    by_concept = defaultdict(list)
    for row in rows:
        by_concept[row.concept].append(row)
    return by_concept


def close_50(day):
    return 50.0


def test_ttm_uses_ytd_plus_prior_year_or_the_fiscal_year():
    periods = latest_by_period(company()["NetIncomeLoss"])
    assert ttm(periods, "2026-06-30") == pytest.approx(86)
    assert ttm(periods, "2025-12-31") == 80
    assert ttm(periods, "2026-03-31") is None    # no facts for that quarter


def test_quarter_ends_newest_first_with_first_filing_date():
    ends = quarter_ends(company())
    assert ends[:3] == [("2026-06-30", "2026-08-01"), ("2025-12-31", "2026-02-10"), ("2025-06-30", "2025-08-01")]


def test_metrics_row_matches_hand_computation():
    row = metrics_rows("TEST", company(), close_50, limit=10)[0]
    assert (row.report_period, row.filing_date, row.period) == ("2026-06-30", "2026-08-01", "ttm")
    assert row.market_cap == pytest.approx(500)
    assert row.price_to_earnings_ratio == pytest.approx(500 / 86)
    assert row.return_on_equity == pytest.approx(86 / 500)
    assert row.gross_margin == pytest.approx(215 / 430)
    assert row.operating_margin == pytest.approx(108 / 430)
    assert row.net_margin == pytest.approx(86 / 430)
    assert row.debt_to_equity == pytest.approx(100 / 500)
    assert row.current_ratio == pytest.approx(2.0)
    assert row.revenue_growth == pytest.approx(430 / 380 - 1)
    assert row.earnings_per_share == pytest.approx(8.6)
    assert row.book_value_per_share == pytest.approx(50)
    assert row.free_cash_flow_per_share == pytest.approx((135 - 23) / 10)


def test_rows_newest_first_and_limited():
    rows = metrics_rows("TEST", company(), close_50, limit=2)
    assert [r.report_period for r in rows] == ["2026-06-30", "2025-12-31"]
    assert rows[1].net_margin == pytest.approx(80 / 400)
    assert rows[1].market_cap is None          # no share count filed by then


def test_restatement_uses_the_latest_filed_value():
    facts = company()
    facts["NetIncomeLoss"].append(f("NetIncomeLoss", "2025-01-01", "2025-12-31", 70, "2026-09-01", "10-K/A"))
    row = next(r for r in metrics_rows("TEST", facts, close_50, limit=10) if r.report_period == "2025-12-31")
    assert row.net_margin == pytest.approx(70 / 400)


def test_missing_inputs_become_none_not_errors():
    facts = company()
    del facts["GrossProfit"]                       # e.g. a bank
    facts["NetIncomeLoss"] = [f("NetIncomeLoss", x.start, x.end, -abs(x.value), x.filed, x.form)
                              for x in facts["NetIncomeLoss"]]
    row = metrics_rows("TEST", facts, close_50, limit=1)[0]
    assert row.gross_margin is None
    assert row.price_to_earnings_ratio is None     # losses have no P/E
    assert row.net_margin == pytest.approx(-86 / 430)


def test_quarterly_values_derive_q4_and_ytd_differences():
    facts = [
        f("EarningsPerShareDiluted", "2025-01-01", "2025-03-31", 1.0, "2025-05-01"),
        f("EarningsPerShareDiluted", "2025-04-01", "2025-06-30", 1.1, "2025-08-01"),
        f("EarningsPerShareDiluted", "2025-01-01", "2025-09-30", 3.3, "2025-11-01"),   # 9-month YTD only
        f("EarningsPerShareDiluted", "2025-01-01", "2025-12-31", 4.6, "2026-02-10", "10-K"),
    ]
    q = quarterly_values(facts)
    assert q["2025-03-31"] == pytest.approx(1.0)
    assert q["2025-06-30"] == pytest.approx(1.1)
    assert "2025-09-30" not in q                  # no 6-month YTD to difference against
    assert q["2025-12-31"] == pytest.approx(4.6 - 3.3)


# ---------------------------------------------------------------------------
# Earnings events (SUE) for PEAD
# ---------------------------------------------------------------------------

QUARTERS = [("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31")]
EPS = {2023: [1.0, 1.0, 1.0, 1.0], 2024: [1.1, 1.2, 1.0, 1.1], 2025: [1.2, 1.3, 1.1, 1.2], 2026: [1.25, 2.0]}


def eps_facts(eps=EPS):
    rows = []
    for year, values in eps.items():
        for (start, end), value in zip(QUARTERS, values):
            end_date = f"{year}-{end}"
            rows.append(f("EarningsPerShareDiluted", f"{year}-{start}", end_date, value,
                          _plus(end_date, 35), unit="USD/shares"))
    return rows


def _plus(day, days):
    from datetime import date, timedelta
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


FILINGS = [
    {"accn": "k0", "form": "8-K", "filed": "2026-07-10", "report_date": "2026-07-10", "items": "5.02"},
    {"accn": "k1", "form": "8-K", "filed": "2026-07-25", "report_date": "2026-07-25", "items": "2.02,9.01"},
    {"accn": "q1", "form": "10-Q", "filed": "2026-05-05", "report_date": "2026-03-31", "items": ""},
]


def test_sue_events_classify_and_date_announcements():
    from hedge_fund.data.fundamentals import earnings_events
    events = earnings_events("TEST", eps_facts(), FILINGS, limit=12)
    assert [e.report_period for e in events] == [
        "2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"]
    latest, previous = events[0], events[1]
    # 2026Q2: +0.70 vs a year ago against a ~0.056 stdev of past changes → far above 1.0.
    assert (latest.source_type, latest.filing_date, latest.quarterly.eps_surprise) == ("8-K", "2026-07-25", "BEAT")
    assert latest.quarterly.earnings_per_share == pytest.approx(2.0)
    # 2026Q1: +0.05 against a ~0.053 stdev → 0.93, a MEET; no 8-K, so the 10-Q date.
    assert (previous.source_type, previous.filing_date, previous.quarterly.eps_surprise) == ("10-Q", "2026-05-05", "MEET")


def test_sue_miss_and_limit():
    from hedge_fund.data.fundamentals import earnings_events
    eps = {**EPS, 2026: [1.25, 0.5]}
    events = earnings_events("TEST", eps_facts(eps), FILINGS, limit=2)
    assert len(events) == 2 and events[0].quarterly.eps_surprise == "MISS"


def test_sue_events_drive_pead():
    from hedge_fund.data.fundamentals import earnings_events
    from hedge_fund.signals import PEADModel

    class Client:
        def get_earnings_history(self, ticker, limit=12):
            return earnings_events(ticker, eps_facts(), FILINGS, limit)

    assert PEADModel().predict("TEST", "2026-07-27", Client()).value == 1.0
