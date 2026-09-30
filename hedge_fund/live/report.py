"""How the paper fund is doing — from its own ledger, with the backtest alongside."""

from __future__ import annotations

from pydantic import BaseModel, Field

from hedge_fund.backtesting.fund import FundBacktestResult, performance_metrics
from hedge_fund.data.protocol import DataClient
from hedge_fund.live.ledger import Ledger
from hedge_fund.live.runner import FillRecord
from hedge_fund.pipeline.run_cycle import exact_marks

MIN_REBALANCES_FOR_VERDICT = 12


class PaperReport(BaseModel):
    start: str
    end: str
    n_days: int
    total_return_pct: float
    benchmark_return_pct: float
    excess_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    n_rebalances: int
    avg_slippage_bps: float | None = None    # notional-weighted, positive = cost
    fill_rate: float | None = None           # filled / requested shares
    turnover: float = 0.0                    # filled notional / average equity
    backtest_return_pct: float | None = None # the backtest over the same dates, if supplied
    strategy_contribution: dict[str, float] = Field(default_factory=dict)


def build_report(ledger: Ledger, *, backtest: FundBacktestResult | None = None) -> PaperReport | None:
    """Metrics over every reconciled session; None until there are at least two."""
    rows = ledger.nav_rows()
    if len(rows) < 2:
        return None
    dates = [r.date for r in rows]
    nav = [r.equity for r in rows]
    bench = [nav[0] * r.benchmark_close / rows[0].benchmark_close for r in rows]
    m = performance_metrics(nav[0], dates, nav, bench, records=[])

    fills = [FillRecord.model_validate(f) for s in ledger.fill_sessions() for f in ledger.read_fills(s)["fills"]]
    filled = [(f, f.filled_qty * f.fill_price) for f in fills if f.filled_qty and f.fill_price]
    slipped = [(f.slippage_bps, notional) for f, notional in filled if f.slippage_bps is not None]
    requested = sum(f.quantity for f in fills)
    return PaperReport(
        start=dates[0], end=dates[-1], n_days=len(dates),
        total_return_pct=m.total_return_pct, benchmark_return_pct=m.benchmark_return_pct,
        excess_return_pct=m.excess_return_pct, sharpe_ratio=m.sharpe_ratio,
        max_drawdown_pct=m.max_drawdown_pct, n_rebalances=len(ledger.plan_sessions()),
        avg_slippage_bps=(sum(s * n for s, n in slipped) / sum(n for _, n in slipped)) if slipped else None,
        fill_rate=(sum(f.filled_qty for f in fills) / requested) if requested else None,
        turnover=sum(n for _, n in filled) / (sum(nav) / len(nav)),
        backtest_return_pct=_backtest_return(backtest, dates[0], dates[-1]) if backtest else None,
    )


def strategy_attribution(ledger: Ledger, data_client: DataClient) -> dict[str, float]:
    """Each strategy's return contribution: its final weights × each name's close-to-close
    return, from each rebalance to the next (or to the latest reconciled session)."""
    sessions = ledger.plan_sessions()
    rows = ledger.nav_rows()
    if not sessions or not rows:
        return {}
    bounds = sessions + [rows[-1].date]
    totals: dict[str, float] = {}
    for start, end in zip(bounds, bounds[1:]):
        if end <= start:
            continue
        decision = ledger.read_plan(start)["plan"].get("decision")
        if not decision:
            continue   # a flatten plan carries no strategy views
        contributions = {s["name"]: s["final_contribution"] for s in decision["strategies"]}
        if decision.get("equitization"):
            contributions["idle capital in benchmark"] = decision["equitization"]
        tickers = sorted({t for weights in contributions.values() for t, w in weights.items() if w})
        if not tickers:
            continue
        a = exact_marks(tickers, start, data_client)
        b = exact_marks(tickers, end, data_client)
        for name, weights in contributions.items():
            totals[name] = totals.get(name, 0.0) + sum(w * (b[t] / a[t] - 1) for t, w in weights.items() if w)
    return {name: round(value, 6) for name, value in totals.items()}


def _backtest_return(result: FundBacktestResult, start: str, end: str) -> float | None:
    points = [nav for day, nav in zip(result.dates, result.nav) if start <= day <= end]
    if len(points) < 2:
        return None
    return round(points[-1] / points[0] - 1, 6)
