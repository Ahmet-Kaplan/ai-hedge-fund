"""Point-in-time fundamentals from SEC XBRL facts — the free replacement for
Financial Datasets' /financial-metrics and /earnings.

Pure functions: facts in (already limited to what was filed by the as-of
date), rows out. Nothing here touches the network or the database.

Checked against 112 archived Financial Datasets rows (32 large caps, Sep 2025
to Jun 2026). Agreement: current ratio, gross margin, free cash flow/share
100%; book value/share, net margin 97-98%; operating margin, ROE, EPS,
revenue growth 91-95%; filing date 88% exact (the rest are one day later —
Financial Datasets dates by the earnings press release, this by the 10-Q, so
never earlier). Deliberate or known differences:
- debt/equity: Financial Datasets double-counts the current portion of
  long-term debt (Apple ~10% higher) and, in many 10-Q rows, omits long-term
  debt entirely (GE, KO, CVX, JPM ~0.1 vs ~1.1); Walmart's finance leases are
  not counted as debt here.
- market cap / P/E: valued at the filing date with the as-traded close;
  Financial Datasets mixes period-end and filing-date prices and undercounts
  multi-class issuers (Alphabet).
- banks' revenue growth and Disney's operating margin follow each filer's
  own tags and differ from Financial Datasets' normalization.
- Berkshire's per-share figures mix A/B share classes; ExxonMobil's pre-2026
  history sits under a predecessor registrant; SEC's company-facts feed can
  lag a new 10-Q by weeks (KO, NEE in Aug-Sep 2026), leaving the prior quarter.
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
COST_OF_REVENUE = ["CostOfRevenue", "CostOfGoodsAndServicesSold"]   # gross profit = revenue − this when untagged
CAPEX = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]
# Total debt, most complete label first:
#   1. the combined long- and short-term amount, when a filer tags it (GE);
#   2. else noncurrent long-term debt + DebtCurrent, when a filer reports all
#      current debt as one line (PG);
#   3. else long-term debt including its current portion (a total label, or
#      noncurrent + current) plus commercial paper, short-term borrowings and
#      notes payable.
# Financial Datasets adds the current portion of long-term debt twice, so its
# debt/equity runs ~5-15% higher for issuers like Apple; this is deliberate.
DEBT_COMBINED = "DebtLongtermAndShorttermCombinedAmount"
LONG_TERM_DEBT_TOTAL = ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities"]
LONG_TERM_DEBT_NONCURRENT = ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"]
LONG_TERM_DEBT_CURRENT = ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"]
SHORT_TERM_DEBT = ["CommercialPaper", "ShortTermBorrowings", "NotesAndLoansPayable"]
DEBT_CURRENT_TOTAL = "DebtCurrent"

US_GAAP = [*REVENUE, "GrossProfit", *COST_OF_REVENUE, "OperatingIncomeLoss", "NetIncomeLoss",
           "NetCashProvidedByUsedInOperatingActivities", *CAPEX,
           *EQUITY, "AssetsCurrent", "LiabilitiesCurrent", DEBT_COMBINED, *LONG_TERM_DEBT_TOTAL,
           *LONG_TERM_DEBT_NONCURRENT, *LONG_TERM_DEBT_CURRENT, *SHORT_TERM_DEBT, DEBT_CURRENT_TOTAL,
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
    derive_gross_profit: bool = True,
) -> list[FinancialMetrics]:
    """Trailing-twelve-month metric rows, newest first, as Financial Datasets defines them.

    `derive_gross_profit=False` (banks, insurers) stops gross profit being
    computed as revenue − cost of goods: those filers tag only a small product
    cost, and the result would be a meaningless ~90% "gross margin".
    """
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
        if gross is None and derive_gross_profit and revenue is not None and (cost := flow(COST_OF_REVENUE, end)) is not None:
            gross = revenue - cost
        operating = flow(["OperatingIncomeLoss"], end)
        net = flow(["NetIncomeLoss"], end)
        cash_ops = flow(["NetCashProvidedByUsedInOperatingActivities"], end)
        capex = flow(CAPEX, end)
        equity = stock(EQUITY, end)
        assets_current = stock(["AssetsCurrent"], end)
        liabilities_current = stock(["LiabilitiesCurrent"], end)
        debt = _total_debt(stock, end)
        diluted = [x for x in _duration_facts(periods["WeightedAverageNumberOfDilutedSharesOutstanding"], end, 3)]
        diluted_shares = min(diluted, key=lambda x: _days(x.start, x.end)).value if diluted else None
        cover_shares = _cover_shares(facts_by_concept.get("EntityCommonStockSharesOutstanding", []), end, filed)
        if diluted_shares and cover_shares and not 0.5 < diluted_shares / cover_shares < 2:
            diluted_shares = cover_shares   # mis-scaled tag (McDonald's reports diluted shares in millions)
        shares = cover_shares or diluted_shares
        # Valued when the numbers became public: cover-page shares × the close on the filing
        # date. (Financial Datasets' timing is inconsistent — some rows use the period end,
        # some the filing date — so it is not matched here.)
        close = close_on_or_before(filed)
        reported_eps = flow(["EarningsPerShareDiluted"], end)
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
            # Net income / diluted shares, as Financial Datasets computes it; it is also
            # split-proof, unlike summing reported EPS across a split (Netflix 2025).
            earnings_per_share=net / diluted_shares if net is not None and diluted_shares else reported_eps,
            book_value_per_share=equity / shares if equity is not None and shares else None,
            free_cash_flow_per_share=(cash_ops - capex) / shares if cash_ops is not None and capex is not None and shares else None,
        ))
    return rows


# ---------------------------------------------------------------------------
# Earnings events for PEAD
# ---------------------------------------------------------------------------

SUE_THRESHOLD = 1.0          # |SUE| at or above this is a BEAT / MISS
_SUE_HISTORY = 8             # year-over-year changes used for the scale
_SUE_MIN_HISTORY = 4
_ANNOUNCEMENT_WINDOW = 60    # days after quarter end an 8-K item 2.02 may land


def earnings_events(ticker: str, eps_facts: Iterable[Fact], filings: list[dict], limit: int,
                    per_share: bool = True) -> list[EarningsRecord]:
    """Earnings surprises as standardized unexpected earnings (Bernard & Thomas 1989).

    SUE = (EPS this quarter − EPS same quarter last year) / stdev of the
    previous 4–8 such changes, from the ORIGINAL filed figures. The event is
    dated by the earnings 8-K (item 2.02) when there is one, else by the
    10-Q/10-K in the filing index, else by the filing that first carried the
    EPS figure. EPS itself comes from the 10-Q/10-K XBRL, which can post a few
    days after the 8-K; the numbers are the ones the 8-K press release
    announced, so dating the event at the 8-K is not lookahead in substance.

    `per_share=False` runs the same test on quarterly net income, for filers
    that tag EPS only per share class (Berkshire, Visa); SUE is scale-free.
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
            quarterly=EarningsData(earnings_per_share=quarters[end] if per_share else None,
                                   net_income=None if per_share else quarters[end], eps_surprise=surprise),
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


def _total_debt(stock: Callable[[list[str], str], float | None], end: str) -> float | None:
    combined = stock([DEBT_COMBINED], end)
    if combined is not None:
        return combined
    current_total = stock([DEBT_CURRENT_TOTAL], end)
    noncurrent = stock(LONG_TERM_DEBT_NONCURRENT, end)
    if current_total is not None and noncurrent is not None:
        return noncurrent + current_total
    long_term = stock(LONG_TERM_DEBT_TOTAL, end)
    if long_term is None:
        split = [v for v in (noncurrent, stock(LONG_TERM_DEBT_CURRENT, end)) if v is not None]
        long_term = sum(split) if split else None
    short_term = [v for c in SHORT_TERM_DEBT if (v := stock([c], end)) is not None]
    if long_term is None and not short_term:
        return None
    return (long_term or 0.0) + sum(short_term)


def _cover_shares(cover: list[Fact], end: str, filed: str) -> float | None:
    """Shares outstanding from the cover page of the latest filing by `filed`.

    Multi-class filers (Visa, Alphabet) tag per-class counts only, which SEC's
    company-facts feed leaves out; what remains is an old undimensioned count
    (Visa's is from 2010). A count dated more than a month before the period
    end is therefore stale and ignored — the caller falls back to diluted shares.
    """
    public = [x for x in cover if x.filed <= filed and _days(end, x.end) >= -30]
    return max(public, key=lambda x: (x.filed, x.end)).value if public else None
