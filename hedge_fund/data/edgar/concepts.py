"""XBRL concepts the EDGAR adapter reads, with ordered tag fallbacks.

Companies tag the same economic line item differently, and the same company
changes tags over time (Coca-Cola reports revenue as SalesRevenueGoodsNet
until 2018 and Revenues after). Each concept lists its us-gaap tags in
priority order; resolution happens per reporting period, so a series can
move from one tag to the next without a gap.

`trim_companyfacts` keeps only these tags and the periodic-report forms —
it is used both for the on-disk cache and for the committed test fixtures,
so the two can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Concept:
    name: str
    tags: tuple[str, ...]
    kind: Literal["duration", "instant"]
    unit: str = "USD"


CONCEPTS: dict[str, Concept] = {c.name: c for c in [
    # Income statement (durations: quarter, YTD, or fiscal year)
    Concept("revenue", (
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ), "duration"),
    # Banks: revenue is net interest income plus non-interest income. Their
    # ASC 606 "revenue from contracts" tag covers fee income only.
    Concept("revenue_net_of_interest", ("RevenuesNetOfInterestExpense",), "duration"),
    Concept("net_interest_income", ("InterestIncomeExpenseNet", "InterestIncomeExpenseAfterProvisionForLoanLoss"), "duration"),
    Concept("noninterest_income", ("NoninterestIncome",), "duration"),
    Concept("cost_of_revenue", (
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
        "CostOfServices",
    ), "duration"),
    Concept("gross_profit", ("GrossProfit",), "duration"),
    Concept("operating_income", ("OperatingIncomeLoss",), "duration"),
    Concept("net_income", (
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ), "duration"),
    Concept("depreciation_amortization", (
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "Depreciation",
    ), "duration"),
    Concept("interest_expense", (
        "InterestExpense",
        "InterestExpenseNonoperating",
        "InterestExpenseDebt",
    ), "duration"),
    Concept("eps_diluted", (
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
        "EarningsPerShareBasic",
    ), "duration", "USD/shares"),
    Concept("weighted_shares_diluted", (
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ), "duration", "shares"),
    # Cash flow statement (10-Q values are fiscal year-to-date)
    Concept("operating_cash_flow", (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ), "duration"),
    Concept("capital_expenditure", (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    ), "duration"),
    Concept("dividends_paid", (
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
        "PaymentsOfOrdinaryDividends",
    ), "duration"),
    # Balance sheet (instants at the period end)
    Concept("equity", (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ), "instant"),
    Concept("total_assets", ("Assets",), "instant"),
    Concept("total_liabilities", ("Liabilities",), "instant"),
    Concept("current_assets", ("AssetsCurrent",), "instant"),
    Concept("current_liabilities", ("LiabilitiesCurrent",), "instant"),
    Concept("cash", (
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash",
    ), "instant"),
    Concept("long_term_debt_total", (
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
    ), "instant"),
    Concept("long_term_debt_noncurrent", (
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
    ), "instant"),
    Concept("long_term_debt_current", (
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "DebtCurrent",
    ), "instant"),
    # Issuers that tag only note instruments (e.g. convertible + senior notes)
    Concept("convertible_notes", (
        "ConvertibleLongTermNotesPayable",
        "ConvertibleNotesPayable",
        "ConvertibleDebtNoncurrent",
    ), "instant"),
    Concept("senior_notes", ("SeniorLongTermNotes", "SeniorNotes"), "instant"),
    Concept("short_term_debt", (
        "ShortTermBorrowings",
        "CommercialPaper",
        "OtherShortTermBorrowings",
    ), "instant"),
    Concept("shares_outstanding_balance_sheet", ("CommonStockSharesOutstanding",), "instant", "shares"),
]}

# Cover-page share count, reported as of a date shortly before filing.
COVER_SHARES_TAG = "EntityCommonStockSharesOutstanding"

# Periodic reports whose facts we read. Amendments count as knowledge
# (restated values become visible on their own filing date) but do not
# create new reporting rows.
PERIODIC_FORMS = frozenset({"10-K", "10-Q", "10-KT", "10-QT"})
AMENDMENT_FORMS = frozenset({"10-K/A", "10-Q/A", "10-KT/A", "10-QT/A"})
KNOWLEDGE_FORMS = PERIODIC_FORMS | AMENDMENT_FORMS

_KEEP_FIELDS = ("start", "end", "val", "accn", "fy", "fp", "form", "filed")


def used_tags() -> set[str]:
    return {tag for c in CONCEPTS.values() for tag in c.tags}


def trim_companyfacts(raw: dict) -> dict:
    """Reduce an SEC companyfacts document to what the adapter reads.

    Keeps entity identity, the us-gaap tags in CONCEPTS, and dei cover-page
    shares — each only for periodic-report forms — typically ~5% of the
    original size.
    """
    facts = raw.get("facts") or {}
    wanted = used_tags()
    out: dict = {"cik": raw.get("cik"), "entityName": raw.get("entityName"), "facts": {"us-gaap": {}, "dei": {}}}
    for ns, tags in (("us-gaap", wanted), ("dei", {COVER_SHARES_TAG})):
        for tag, body in (facts.get(ns) or {}).items():
            if tag not in tags:
                continue
            units = {}
            for unit, rows in (body.get("units") or {}).items():
                kept = [{k: r[k] for k in _KEEP_FIELDS if k in r}
                        for r in rows if r.get("form") in KNOWLEDGE_FORMS]
                if kept:
                    units[unit] = kept
            if units:
                out["facts"][ns][tag] = {"units": units}
    return out


def trim_submissions(raw: dict) -> dict:
    """Keep the company-profile fields of an SEC submissions document."""
    keep = ("cik", "name", "sic", "sicDescription", "tickers", "exchanges",
            "stateOfIncorporation", "formerNames", "category", "entityType")
    out = {k: raw.get(k) for k in keep}
    business = ((raw.get("addresses") or {}).get("business") or {})
    out["location"] = ", ".join(x for x in (business.get("city"), business.get("stateOrCountry")) if x) or None
    return out
