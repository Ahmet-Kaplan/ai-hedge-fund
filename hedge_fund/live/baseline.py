"""Does the fund add value? Backtest it, each strategy alone, and two yardsticks.

The yardsticks are the benchmark (SPY) and an equal-weight book of the same
universe run through the identical engine — same caps, costs and cadence. A
strategy "beats the benchmarks" only if it beats both on total return and on
Sharpe. Windows starting before the LLM's training cutoff are labelled: the
model may remember how those companies did.

With a MarketStore, each finished variant is saved under a key of its spec,
window, universe and LLM model: a re-run after a crash (say, LLM credit ran
out) reuses finished variants and computes only the rest, and a failing
variant is recorded without stopping the others.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Callable

from pydantic import BaseModel, Field

from hedge_fund.backtesting.fund import FundBacktestResult, backtest_fund, performance_metrics
from hedge_fund.data.protocol import DataClient
from hedge_fund.data.store import MarketStore
from hedge_fund.fund.spec import Fund, FundSpec

logger = logging.getLogger(__name__)

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
    failed: dict[str, str] = Field(default_factory=dict)   # variant → error


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


def run_key(variant: FundSpec, start: str, end: str, universe: list[str]) -> str:
    payload = json.dumps([variant.model_dump(mode="json"), start, end, universe,
                          os.environ.get("HEDGE_FUND_LLM_MODEL", "")], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def run_baseline(
    spec: FundSpec, universe: list[str], start: str, end: str, data_client: DataClient,
    *, backtest: Callable[..., FundBacktestResult] = backtest_fund,
    store: MarketStore | None = None, fresh: bool = False,
) -> BaselineReport:
    results: dict[str, FundBacktestResult] = {}
    failed: dict[str, str] = {}
    for name, variant in baseline_variants(spec).items():
        key = run_key(variant, start, end, universe)
        saved = store.get_run(key) if store is not None and not fresh else None
        if saved and saved["status"] == "done":
            results[name] = FundBacktestResult.model_validate_json(saved["result_json"])
            continue
        try:
            results[name] = backtest(Fund(variant, blind=True), start, end, data_client, universe)
        except Exception as exc:   # record and move on: finished variants must survive
            logger.exception("baseline variant %s failed", name)
            failed[name] = str(exc)
            if store is not None:
                store.put_run(key, name, "failed", None, error=str(exc))
            continue
        if store is not None:
            store.put_run(key, name, "done", results[name].model_dump_json())

    yardsticks = "fund" in results and "equal-weight" in results
    reference = results.get("fund") or next(iter(results.values()), None)
    spy = (performance_metrics(reference.capital, reference.dates, reference.benchmark_nav, reference.benchmark_nav, [])
           if reference is not None else None)
    if yardsticks:
        ew = results["equal-weight"].metrics
        bar_return = max(spy.total_return_pct, ew.total_return_pct)
        bar_sharpe = max(spy.sharpe_ratio, ew.sharpe_ratio)

    rows = []
    for name, result in results.items():
        m = result.metrics
        beats = None if name == "equal-weight" or not yardsticks else (
            m.total_return_pct > bar_return and m.sharpe_ratio > bar_sharpe)
        rows.append(BaselineRow(
            name=name, total_return_pct=m.total_return_pct, annualized_return_pct=m.annualized_return_pct,
            sharpe_ratio=m.sharpe_ratio, max_drawdown_pct=m.max_drawdown_pct,
            total_costs=m.total_costs, beats_benchmarks=beats,
        ))
    if spy is not None:
        rows.append(BaselineRow(
            name=spec.benchmark.lower(), total_return_pct=spy.total_return_pct,
            annualized_return_pct=spy.annualized_return_pct, sharpe_ratio=spy.sharpe_ratio,
            max_drawdown_pct=spy.max_drawdown_pct,
        ))
    return BaselineReport(start=start, end=end, possibly_memorized=start < POST_CUTOFF_START,
                          benchmark=spec.benchmark, rows=rows, failed=failed)
