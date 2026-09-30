"""Does the fund add value? Backtest it, each strategy alone, and two yardsticks.

The yardsticks are the benchmark (SPY) and an equal-weight book of the same
universe run through the identical engine — same caps, costs and cadence. A
strategy "beats the benchmarks" only if it beats both on total return and on
Sharpe. Windows starting before the LLM's training cutoff are labelled: the
model may remember how those companies did.
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from hedge_fund.backtesting.fund import FundBacktestResult, backtest_fund, performance_metrics
from hedge_fund.data.protocol import DataClient
from hedge_fund.fund.spec import Fund, FundSpec

# After the default model's (claude-opus-5-5) stated June 2026 training cutoff.
POST_CUTOFF_START = "2026-07-01"


class BaselineRow(BaseModel):
    name: str
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_costs: float = 0.0
    beats_benchmarks: bool | None = None   # None for the yardsticks themselves


class BaselineReport(BaseModel):
    start: str
    end: str
    possibly_memorized: bool
    benchmark: str
    rows: list[BaselineRow]


def baseline_variants(spec: FundSpec) -> dict[str, FundSpec]:
    """The full fund, each strategy alone at full capital, and the equal-weight book."""
    base = spec.model_dump()
    variants = {"fund": spec}
    for strategy in spec.strategies:
        variants[f"only:{strategy.name}"] = FundSpec.model_validate({
            **base, "name": f"{spec.name}-{strategy.name}",
            "strategies": [{**strategy.model_dump(), "weight": 1.0}],
        })
    variants["equal-weight"] = FundSpec.model_validate({
        **base, "name": f"{spec.name}-equal-weight",
        "strategies": [{"name": "equal-weight", "models": [{"name": "equal_weight"}], "blend": {"mode": "long_only"}}],
    })
    return variants


def run_baseline(
    spec: FundSpec, universe: list[str], start: str, end: str, data_client: DataClient,
    *, backtest: Callable[..., FundBacktestResult] = backtest_fund,
) -> BaselineReport:
    results = {
        name: backtest(Fund(variant, blind=True), start, end, data_client, universe)
        for name, variant in baseline_variants(spec).items()
    }
    fund = results["fund"]
    spy = performance_metrics(fund.capital, fund.dates, fund.benchmark_nav, fund.benchmark_nav, [])
    ew = results["equal-weight"].metrics
    bar_return = max(spy.total_return_pct, ew.total_return_pct)
    bar_sharpe = max(spy.sharpe_ratio, ew.sharpe_ratio)

    rows = []
    for name, result in results.items():
        m = result.metrics
        beats = None if name == "equal-weight" else (m.total_return_pct > bar_return and m.sharpe_ratio > bar_sharpe)
        rows.append(BaselineRow(
            name=name, total_return_pct=m.total_return_pct, annualized_return_pct=m.annualized_return_pct,
            sharpe_ratio=m.sharpe_ratio, max_drawdown_pct=m.max_drawdown_pct,
            total_costs=m.total_costs, beats_benchmarks=beats,
        ))
    rows.append(BaselineRow(
        name=spec.benchmark.lower(), total_return_pct=spy.total_return_pct,
        annualized_return_pct=spy.annualized_return_pct, sharpe_ratio=spy.sharpe_ratio,
        max_drawdown_pct=spy.max_drawdown_pct,
    ))
    return BaselineReport(start=start, end=end, possibly_memorized=start < POST_CUTOFF_START,
                          benchmark=spec.benchmark, rows=rows)
