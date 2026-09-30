"""Point-in-time fundamentals from SEC XBRL facts — the free replacement for
Financial Datasets' /financial-metrics and /earnings.

Pure functions: facts in (already limited to what was filed by the as-of
date), rows out. Nothing here touches the network or the database.
"""

from __future__ import annotations

import statistics
from datetime import date, timedelta
from typing import Callable, Iterable

from hedge_fund.data.models import EarningsData, EarningsRecord, FinancialMetrics
from hedge_fund.data.store import Fact

# Candidates are tried in order, per period: companies switch concepts over
# time (Apple reported SalesRevenueNet until 2018) and industries differ
# (banks report net revenue, not gross profit).
REVENUE = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
           "RevenuesNetOfInterestExpense", "RevenueFromContractWithCustomerIncludingAssessedTax"]
EQUITY = ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
DEBT_PARTS = ["LongTermDebtNoncurrent", "LongTermDebtCurrent", "CommercialPaper", "ShortTermBorrowings"]
DEBT_TOTAL = "LongTermDebt"   # used when the noncurrent/current split is not reported

US_GAAP = [*REVENUE, "GrossProfit", "OperatingIncomeLoss", "NetIncomeLoss",
           "NetCashProvidedByUsedInOperatingActivities", "PaymentsToAcquirePropertyPlantAndEquipment",
           *EQUITY, "AssetsCurrent", "LiabilitiesCurrent", *DEBT_PARTS, DEBT_TOTAL,
           "WeightedAverageNumberOfDilutedSharesOutstanding", "EarningsPerShareDiluted"]
DEI = ["EntityCommonStockSharesOutstanding"]
CONCEPTS = US_GAAP + DEI

_QUARTER = (80, 100)       # days in a fiscal quarter (52/53-week calendars vary)
_YEAR = (350, 380)         # days in a fiscal year


def _d(day: str) -> date:
    return date.fromisoformat(day[:10])


def _days(start: str, end: str) -> int:
    return (_d(end) - _d(start)).days


def _near(a: str, b: str, tolerance: int) -> bool:
    return abs(_days(a, b)) <= tolerance


def _shift(day: str, days: int) -> str:
    return (_d(day) + timedelta(days=days)).isoformat()


def latest_by_period(facts: Iterable[Fact]) -> dict[tuple[str | None, str], Fact]:
    """One fact per (start, end): the latest filed — restatements win once public."""
    out: dict[tuple[str | None, str], Fact] = {}
    for fact in sorted(facts, key=lambda x: x.filed):
        out[(fact.start, fact.end)] = fact
    return out


def original_by_period(facts: Iterable[Fact]) -> dict[tuple[str | None, str], Fact]:
    """One fact per (start, end): the first filed — what the market saw at the time."""
    out: dict[tuple[str | None, str], Fact] = {}
    for fact in sorted(facts, key=lambda x: x.filed, reverse=True):
        out[(fact.start, fact.end)] = fact
    return out


def _duration_facts(periods: dict, end: str, tolerance: int) -> list[Fact]:
    return [x for (start, e), x in periods.items() if start and _near(e, end, tolerance)]


def ttm(periods: dict[tuple[str | None, str], Fact], end: str, tolerance: int = 3) -> float | None:
    """Trailing twelve months ending at `end` for a flow concept (income or cash flow).

    The fiscal-year value when `end` is a year end; otherwise year-to-date +
    prior fiscal year − prior year-to-date, which also works for cash-flow
    statements that 10-Qs report only year-to-date.
    """
    ending = _duration_facts(periods, end, tolerance)
    for fact in ending:
        if _YEAR[0] <= _days(fact.start, fact.end) <= _YEAR[1]:
            return fact.value
    ytd = [x for x in ending if _days(x.start, x.end) < _YEAR[0]]
    if not ytd:
        return None
    current = max(ytd, key=lambda x: _days(x.start, x.end))
    length = _days(current.start, current.end)
    prior_year_end = _shift(current.start, -1)
    prior_fy = next((x for x in _duration_facts(periods, prior_year_end, 7)
                     if _YEAR[0] <= _days(x.start, x.end) <= _YEAR[1]), None)
    prior_ytd = next((x for x in _duration_facts(periods, _shift(end, -365), 7)
                      if abs(_days(x.start, x.end) - length) <= 10), None)
    if prior_fy is None or prior_ytd is None:
        return None
    return current.value + prior_fy.value - prior_ytd.value


def instant(periods: dict[tuple[str | None, str], Fact], end: str, tolerance: int = 3) -> float | None:
    """A balance-sheet value at `end`."""
    matches = [x for (start, e), x in periods.items() if start is None and _near(e, end, tolerance)]
    return max(matches, key=lambda x: x.filed).value if matches else None


def quarterly_values(facts: Iterable[Fact]) -> dict[str, float]:
    """Single-quarter values per period end, from the ORIGINAL filings.

    Three-month facts directly; otherwise a year-to-date fact minus the
    year-to-date fact one quarter shorter with the same start (this is how
    Q4 = full year − nine months).
    """
    periods = original_by_period(facts)
    durations = [x for x in periods.values() if x.start]
    out: dict[str, float] = {}
    for fact in durations:
        if _QUARTER[0] <= _days(fact.start, fact.end) <= _QUARTER[1]:
            out[fact.end] = fact.value
    for fact in durations:
        if fact.end in out or _days(fact.start, fact.end) <= _QUARTER[1]:
            continue
        shorter = next((x for x in durations if x is not fact and _near(x.start, fact.start, 3)
                        and _QUARTER[0] <= _days(x.end, fact.end) <= _QUARTER[1]), None)
        if shorter is not None:
            out[fact.end] = fact.value - shorter.value
    return out


def quarter_ends(facts_by_concept: dict[str, list[Fact]]) -> list[tuple[str, str]]:
    """(period end, first filing date) for every reported fiscal period, newest first."""
    first_filed: dict[str, str] = {}
    for concept in [*REVENUE, "NetIncomeLoss"]:
        for fact in facts_by_concept.get(concept, []):
            if fact.start and _days(fact.start, fact.end) >= _QUARTER[0]:
                if fact.end not in first_filed or fact.filed < first_filed[fact.end]:
                    first_filed[fact.end] = fact.filed
    return sorted(first_filed.items(), reverse=True)


def metrics_rows(
    ticker: str, facts_by_concept: dict[str, list[Fact]],
    close_on_or_before: Callable[[str], float | None], limit: int,
) -> list[FinancialMetrics]:
    """Trailing-twelve-month metric rows, newest first, as Financial Datasets defines them."""
    periods = {c: latest_by_period(facts_by_concept.get(c, [])) for c in CONCEPTS}

    def flow(concepts: list[str], end: str, tolerance: int = 3) -> float | None:
        return next((v for c in concepts if (v := ttm(periods[c], end, tolerance)) is not None), None)

    def stock(concepts: list[str], end: str) -> float | None:
        return next((v for c in concepts if (v := instant(periods[c], end)) is not None), None)

    rows = []
    for end, filed in quarter_ends(facts_by_concept)[:limit]:
        revenue = flow(REVENUE, end)
        revenue_prior = flow(REVENUE, _shift(end, -365), tolerance=7)
        gross = flow(["GrossProfit"], end)
        operating = flow(["OperatingIncomeLoss"], end)
        net = flow(["NetIncomeLoss"], end)
        cash_ops = flow(["NetCashProvidedByUsedInOperatingActivities"], end)
        capex = flow(["PaymentsToAcquirePropertyPlantAndEquipment"], end)
        equity = stock(EQUITY, end)
        assets_current = stock(["AssetsCurrent"], end)
        liabilities_current = stock(["LiabilitiesCurrent"], end)
        debt_parts = [v for c in DEBT_PARTS if (v := stock([c], end)) is not None]
        debt = sum(debt_parts) if debt_parts else stock([DEBT_TOTAL], end)
        diluted = [x for x in _duration_facts(periods["WeightedAverageNumberOfDilutedSharesOutstanding"], end, 3)]
        diluted_shares = min(diluted, key=lambda x: _days(x.start, x.end)).value if diluted else None
        cover = [x for x in facts_by_concept.get("EntityCommonStockSharesOutstanding", []) if x.filed <= filed]
        shares = max(cover, key=lambda x: (x.filed, x.end)).value if cover else None
        close = close_on_or_before(filed)
        market_cap = shares * close if shares and close else None

        rows.append(FinancialMetrics(
            ticker=ticker, report_period=end, period="ttm", filing_date=filed,
            market_cap=market_cap,
            price_to_earnings_ratio=market_cap / net if market_cap and net and net > 0 else None,
            return_on_equity=net / equity if net is not None and equity else None,
            gross_margin=gross / revenue if gross is not None and revenue else None,
            operating_margin=operating / revenue if operating is not None and revenue else None,
            net_margin=net / revenue if net is not None and revenue else None,
            debt_to_equity=debt / equity if debt is not None and equity and equity > 0 else None,
            current_ratio=assets_current / liabilities_current if assets_current is not None and liabilities_current else None,
            revenue_growth=revenue / revenue_prior - 1 if revenue is not None and revenue_prior else None,
            earnings_per_share=net / diluted_shares if net is not None and diluted_shares else None,
            book_value_per_share=equity / shares if equity is not None and shares else None,
            free_cash_flow_per_share=(cash_ops - (capex or 0.0)) / shares if cash_ops is not None and shares else None,
        ))
    return rows


# ---------------------------------------------------------------------------
# Earnings events for PEAD
# ---------------------------------------------------------------------------

SUE_THRESHOLD = 1.0          # |SUE| at or above this is a BEAT / MISS
_SUE_HISTORY = 8             # year-over-year changes used for the scale
_SUE_MIN_HISTORY = 4
_ANNOUNCEMENT_WINDOW = 60    # days after quarter end an 8-K item 2.02 may land


def earnings_events(ticker: str, eps_facts: Iterable[Fact], filings: list[dict], limit: int) -> list[EarningsRecord]:
    """Earnings surprises as standardized unexpected earnings (Bernard & Thomas 1989).

    SUE = (EPS this quarter − EPS same quarter last year) / stdev of the
    previous 4–8 such changes, from the ORIGINAL filed figures. The event is
    dated by the earnings 8-K (item 2.02) when there is one, else by the
    10-Q/10-K in the filing index, else by the filing that first carried the
    EPS figure. EPS itself comes from the 10-Q/10-K XBRL, which can post a few
    days after the 8-K; the numbers are the ones the 8-K press release
    announced, so dating the event at the 8-K is not lookahead in substance.
    """
    eps_facts = list(eps_facts)
    quarters = quarterly_values(eps_facts)
    first_report: dict[str, tuple[str, str]] = {}
    for fact in sorted(eps_facts, key=lambda x: x.filed, reverse=True):
        first_report[fact.end] = (fact.form.split("/")[0], fact.filed)
    ends = sorted(quarters)
    changes: list[tuple[str, float]] = []
    for end in ends:
        prior = next((e for e in ends if 355 <= _days(e, end) <= 375), None)
        if prior is not None:
            changes.append((end, quarters[end] - quarters[prior]))

    records: list[EarningsRecord] = []
    for i, (end, change) in enumerate(changes):
        history = [c for _, c in changes[max(0, i - _SUE_HISTORY):i]]
        if len(history) < _SUE_MIN_HISTORY:
            continue
        scale = statistics.stdev(history)
        if scale == 0:
            continue
        sue = change / scale
        surprise = "BEAT" if sue >= SUE_THRESHOLD else "MISS" if sue <= -SUE_THRESHOLD else "MEET"
        form, filed = _announcement(end, filings) or first_report[end]
        records.append(EarningsRecord(
            ticker=ticker, report_period=end, source_type=form, filing_date=filed,
            quarterly=EarningsData(earnings_per_share=quarters[end], eps_surprise=surprise),
        ))
    records.sort(key=lambda r: r.report_period, reverse=True)
    return records[:limit]


def _announcement(end: str, filings: list[dict]) -> tuple[str, str] | None:
    """(form, date) the quarter's results became public."""
    releases = sorted(f["filed"] for f in filings
                      if f["form"].startswith("8-K") and "2.02" in (f.get("items") or "")
                      and 0 <= _days(end, f["filed"]) <= _ANNOUNCEMENT_WINDOW)
    if releases:
        return "8-K", releases[0]
    reports = sorted((f["filed"], f["form"]) for f in filings
                     if f["form"].startswith("10-") and f.get("report_date") and _near(f["report_date"], end, 3))
    if reports:
        return reports[0][1].split("/")[0], reports[0][0]
    return None
