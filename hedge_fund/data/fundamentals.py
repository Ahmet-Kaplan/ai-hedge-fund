"""Point-in-time fundamentals from SEC XBRL facts — the free replacement for
Financial Datasets' /financial-metrics and /earnings.

Pure functions: facts in (already limited to what was filed by the as-of
date), rows out. Nothing here touches the network or the database.
"""

from __future__ import annotations

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
