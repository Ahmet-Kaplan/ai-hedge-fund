"""Per-filing TTM fundamentals, and their FinancialMetrics projection.

A row describes one periodic report and is computed from the knowledge view
cut at that report's own filing date, so a row never changes after it is
filed and never contains anything filed later (a restatement shows up in
the rows of the filing that carried it, not retroactively).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hedge_fund.data.edgar.facts import Filing, KnowledgeView
from hedge_fund.data.models import FinancialMetrics

PER_SHARE_FIELDS = ("earnings_per_share", "book_value_per_share", "free_cash_flow_per_share")


@dataclass
class FilingValues:
    filing: Filing
    values: dict[str, float | None] = field(default_factory=dict)
    prior: dict[str, float | None] = field(default_factory=dict)
    shares: float | None = None          # outstanding, in units of the queried ticker's class
    shares_source: str | None = None
    price: float | None = None           # raw close on/before the filing date
    price_date: str | None = None


def _ratio(num: float | None, den: float | None, positive_den: bool = True) -> float | None:
    if num is None or den is None or den == 0 or (positive_den and den < 0):
        return None
    return num / den


def _growth(now: float | None, before: float | None) -> float | None:
    if now is None or before is None or before == 0:
        return None
    return (now - before) / abs(before)


def _price_to_earnings(price: float | None, eps: float | None, market_cap: float | None,
                       net_income: float | None) -> float | None:
    """Price over TTM diluted EPS, both as of the filing — so P/E x EPS is
    the filing-date close. Market cap / net income differs from it by
    preferred dividends and the diluted-vs-outstanding share gap (~5% at JPM
    and PG), and is only the fallback when no EPS exists. Non-positive
    earnings have no P/E."""
    if price is not None and eps is not None:
        return price / eps if eps > 0 else None
    return _ratio(market_cap, net_income)


def _plausible_weighted(weighted: float | None, outstanding: float | None) -> float | None:
    """Weighted-average shares, unless they disagree with shares outstanding
    by more than 2x — the signature of a scale error in the filing's tagging
    (Coca-Cola's Q1 2019 10-Q tagged 4,306 for 4.306 billion)."""
    if weighted is None or weighted <= 0:
        return outstanding
    if outstanding and not 0.5 <= weighted / outstanding <= 2.0:
        return outstanding
    return weighted


def _sum_or_none(*xs: float | None) -> float | None:
    present = [x for x in xs if x is not None]
    return sum(present) if present else None


def flows(view: KnowledgeView, end: str) -> dict[str, float | None]:
    """TTM flows and period-end balances at *end*, as known to *view*."""
    v: dict[str, float | None] = {}
    for concept in ("revenue", "net_income", "operating_income", "operating_cash_flow", "capital_expenditure",
                    "depreciation_amortization", "interest_expense", "dividends_paid"):
        v[concept] = view.ttm(concept, end)
    net_interest, noninterest = view.ttm("net_interest_income", end), view.ttm("noninterest_income", end)
    if net_interest is not None and noninterest is not None:  # a bank
        v["revenue"] = view.ttm("revenue_net_of_interest", end)
        if v["revenue"] is None:
            v["revenue"] = net_interest + noninterest
    if v["revenue"] is not None and v["net_income"] is not None and v["net_income"] > v["revenue"] > 0:
        v["revenue"] = None  # a partial revenue tag (net margin > 100%): better absent than wrong
    gross = view.ttm("gross_profit", end)
    if gross is None:
        cost = view.ttm("cost_of_revenue", end)
        gross = v["revenue"] - cost if v["revenue"] is not None and cost is not None else None
    v["gross_profit"] = gross
    ocf, capex = v["operating_cash_flow"], v["capital_expenditure"]
    v["free_cash_flow"] = ocf - capex if ocf is not None and capex is not None else None
    op, da = v["operating_income"], v["depreciation_amortization"]
    v["ebitda"] = op + da if op is not None and da is not None else None
    for concept in ("equity", "total_assets", "total_liabilities", "current_assets", "current_liabilities", "cash"):
        v[concept] = view.instant(concept, end)
    long_term = view.instant("long_term_debt_total", end)
    if long_term is None:
        long_term = _sum_or_none(view.instant("long_term_debt_noncurrent", end),
                                 view.instant("long_term_debt_current", end))
    if long_term is None:
        long_term = _sum_or_none(view.instant("convertible_notes", end), view.instant("senior_notes", end))
    short_term = view.instant("short_term_debt", end)
    v["total_debt"] = None if long_term is None and short_term is None else (long_term or 0.0) + (short_term or 0.0)
    v["eps_ttm_reported"] = view.ttm("eps_diluted", end)
    v["weighted_shares"] = view.latest_duration("weighted_shares_diluted", end)
    return v


def to_metrics(ticker: str, fv: FilingValues, classed: bool, split_factor: float = 1.0) -> FinancialMetrics:
    """Project filing values onto FinancialMetrics.

    *classed*: the registrant has share classes with different economics
    (Berkshire), so reported per-share figures are in another class's units
    and per-share values are derived from converted share counts instead.
    *split_factor*: cumulative splits between the filing and the query date;
    per-share fields are divided by it (market cap is basis-free).
    """
    v, p, shares = fv.values, fv.prior, fv.shares
    ni, rev, equity = v.get("net_income"), v.get("revenue"), v.get("equity")

    per_share_count = shares if classed else _plausible_weighted(v.get("weighted_shares"), shares)
    if classed:
        eps = _ratio(ni, shares)
    else:
        eps = v.get("eps_ttm_reported")
        if eps is None:
            eps = _ratio(ni, per_share_count)
    prior_eps = None if classed else p.get("eps_ttm_reported")

    market_cap = shares * fv.price if shares is not None and fv.price is not None else None
    debt, cash = v.get("total_debt"), v.get("cash")
    ev = market_cap + debt - cash if None not in (market_cap, debt, cash) else None
    ebitda = v.get("ebitda")

    row = dict(
        ticker=ticker, report_period=fv.filing.report_period, period="ttm", currency="USD",
        filing_date=fv.filing.filed, filing_datetime=None,
        market_cap=market_cap, enterprise_value=ev,
        price_to_earnings_ratio=_price_to_earnings(fv.price, eps, market_cap, ni),
        price_to_book_ratio=_ratio(market_cap, equity),
        price_to_sales_ratio=_ratio(market_cap, rev),
        enterprise_value_to_ebitda_ratio=_ratio(ev, ebitda),
        enterprise_value_to_revenue_ratio=_ratio(ev, rev),
        free_cash_flow_yield=_ratio(v.get("free_cash_flow"), market_cap),
        gross_margin=_ratio(v.get("gross_profit"), rev),
        operating_margin=_ratio(v.get("operating_income"), rev),
        net_margin=_ratio(ni, rev),
        return_on_equity=_ratio(ni, equity),
        return_on_assets=_ratio(ni, v.get("total_assets")),
        asset_turnover=_ratio(rev, v.get("total_assets")),
        current_ratio=_ratio(v.get("current_assets"), v.get("current_liabilities")),
        cash_ratio=_ratio(cash, v.get("current_liabilities")),
        operating_cash_flow_ratio=_ratio(v.get("operating_cash_flow"), v.get("current_liabilities")),
        debt_to_equity=_ratio(debt, equity),
        debt_to_assets=_ratio(debt, v.get("total_assets")),
        interest_coverage=_ratio(v.get("operating_income"), v.get("interest_expense")),
        revenue_growth=_growth(rev, p.get("revenue")),
        earnings_growth=_growth(ni, p.get("net_income")),
        book_value_growth=_growth(equity, p.get("equity")),
        earnings_per_share_growth=_growth(eps, prior_eps),
        free_cash_flow_growth=_growth(v.get("free_cash_flow"), p.get("free_cash_flow")),
        operating_income_growth=_growth(v.get("operating_income"), p.get("operating_income")),
        ebitda_growth=_growth(ebitda, p.get("ebitda")),
        payout_ratio=_ratio(v.get("dividends_paid"), ni),
        earnings_per_share=eps,
        book_value_per_share=_ratio(equity, shares),
        free_cash_flow_per_share=_ratio(v.get("free_cash_flow"), per_share_count),
    )
    if split_factor != 1.0:
        for name in PER_SHARE_FIELDS:
            if row[name] is not None:
                row[name] = row[name] / split_factor
    return FinancialMetrics(**row)
