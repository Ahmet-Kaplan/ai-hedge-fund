"""Decision reporting — receipts rendered as per-name decisions, the selected
book, cash, rejections with reasons, and backtest results.

A read-only consumer of CycleRecord / PendingRunResult / FundBacktestResult;
``python -m hedge_fund.reporting receipt.json --out report.html``.
"""

from hedge_fund.reporting.decisions import (
    AnalystView,
    BacktestReport,
    DecisionReport,
    TickerDecision,
    backtest_report,
    cycle_report,
    load_receipt,
    pending_report,
    proposal_report,
)
from hedge_fund.reporting.render import to_html, to_markdown

__all__ = [
    "AnalystView",
    "BacktestReport",
    "DecisionReport",
    "TickerDecision",
    "backtest_report",
    "cycle_report",
    "load_receipt",
    "pending_report",
    "proposal_report",
    "to_html",
    "to_markdown",
]
