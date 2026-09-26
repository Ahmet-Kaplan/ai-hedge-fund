"""v2 backtesting — simulate a fund (or a single alpha model) over history."""

from hedge_fund.backtesting.engine import BacktestEngine
from hedge_fund.backtesting.fund import (
    backtest_fund,
    build_schedule,
    DailyValuation,
    FundBacktestMetrics,
    FundBacktestResult,
    performance_metrics,
    rebalance_grid,
    ReplaySchedule,
)
from hedge_fund.backtesting.ledger import PortfolioLedger
from hedge_fund.backtesting.models import (
    BacktestResult,
    PerformanceMetrics,
    PortfolioLedgerEntry,
    PositionSnapshot,
    Trade,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "DailyValuation",
    "FundBacktestMetrics",
    "FundBacktestResult",
    "PerformanceMetrics",
    "PortfolioLedger",
    "PortfolioLedgerEntry",
    "PositionSnapshot",
    "Trade",
    "ReplaySchedule",
    "backtest_fund",
    "build_schedule",
    "performance_metrics",
    "rebalance_grid",
]
