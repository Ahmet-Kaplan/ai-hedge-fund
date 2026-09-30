import pytest

from hedge_fund.backtesting.fund import FundBacktestMetrics, FundBacktestResult
from hedge_fund.fund.spec import FundSpec
from hedge_fund.live.baseline import baseline_variants, run_baseline
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel


class Stub(QuantModel):
    investment_approach = "long_short"

    @property
    def name(self):
        return "stub"

    def predict(self, ticker, date, data_client):
        return Signal(model_name="stub", ticker=ticker, date=date, value=0.0)


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    from hedge_fund.signals import ALPHA_MODEL_REGISTRY
    monkeypatch.setitem(ALPHA_MODEL_REGISTRY, "stub", Stub)


def spec():
    return FundSpec(
        schema_version=2, name="pf",
        strategies=[
            {"name": "alpha", "weight": 0.7, "models": [{"name": "stub"}], "blend": {"mode": "long_short"}},
            {"name": "beta", "weight": 0.3, "models": [{"name": "stub"}], "blend": {"mode": "long_short"}},
        ],
        risk={"max_position_pct": 0.1, "max_gross_exposure": 1.0},
        costs={"commission_bps": 5},
    )


def test_variants():
    variants = baseline_variants(spec())
    assert list(variants) == ["fund", "only:alpha", "only:beta", "equal-weight"]
    assert [s.name for s in variants["only:alpha"].strategies] == ["alpha"]
    assert variants["only:alpha"].strategies[0].weight == 1.0
    ew = variants["equal-weight"]
    assert ew.strategies[0].models[0].name == "equal_weight"
    assert ew.strategies[0].blend.mode == "long_only"
    assert ew.costs.commission_bps == 5 and ew.risk == spec().risk


def result(total, sharpe):
    return FundBacktestResult(
        fund="pf", start="2026-07-01", end="2026-07-03", rebalance="weekly", benchmark="SPY",
        universe=["AAPL"], capital=100.0, dates=["2026-07-01", "2026-07-02", "2026-07-03"],
        nav=[100.0, 100.0, 100.0 * (1 + total)], benchmark_nav=[100.0, 102.0, 101.0],
        metrics=FundBacktestMetrics(total_return_pct=total, annualized_return_pct=total, sharpe_ratio=sharpe,
                                    max_drawdown_pct=0.0, benchmark_return_pct=0.01,
                                    excess_return_pct=total - 0.01, n_cycles=1, n_orders=1, total_costs=3.0),
        records=[],
    )


def test_run_baseline_flags_only_what_beats_both_benchmarks():
    outcomes = {"pf": (0.10, 2.0), "pf-alpha": (0.20, 10.0), "pf-beta": (0.03, 5.0), "pf-equal-weight": (0.05, 1.0)}
    seen = []

    def fake_backtest(fund, start, end, data_client, universe):
        seen.append((fund.spec.name, start, end, universe))
        return result(*outcomes[fund.spec.name])

    report = run_baseline(spec(), ["AAPL"], "2026-07-01", "2026-07-03", None, backtest=fake_backtest)
    rows = {r.name: r for r in report.rows}
    assert rows["only:alpha"].beats_benchmarks is True
    assert rows["only:beta"].beats_benchmarks is False       # loses to equal-weight on return
    assert rows["fund"].beats_benchmarks is False            # Sharpe below SPY's
    assert rows["equal-weight"].beats_benchmarks is None
    assert rows["spy"].total_return_pct == pytest.approx(0.01)
    assert report.possibly_memorized is False
    assert len(seen) == 4 and all(s[3] == ["AAPL"] for s in seen)


def test_windows_before_the_cutoff_are_labelled():
    report = run_baseline(spec(), ["AAPL"], "2025-01-01", "2026-07-03", None,
                          backtest=lambda *a: result(0.0, 0.0))
    assert report.possibly_memorized is True


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def test_finished_variants_are_reused_and_failures_recorded(tmp_path):
    from hedge_fund.data.store import MarketStore
    store = MarketStore(tmp_path / "m.db")
    calls = []

    def flaky(fund, start, end, data_client, universe):
        calls.append(fund.spec.name)
        if fund.spec.name == "pf-beta":
            raise RuntimeError("LLM credit exhausted")
        return result(0.05, 1.0)

    report = run_baseline(spec(), ["AAPL"], "2026-07-01", "2026-07-03", None, backtest=flaky, store=store)
    assert report.failed == {"only:beta": "LLM credit exhausted"}
    assert {r.name for r in report.rows} == {"fund", "only:alpha", "equal-weight", "spy"}
    assert len(calls) == 4

    calls.clear()
    rerun = run_baseline(spec(), ["AAPL"], "2026-07-01", "2026-07-03", None,
                         backtest=lambda *a: calls.append(a[0].spec.name) or result(0.05, 1.0), store=store)
    assert calls == ["pf-beta"]                       # only the failed variant is recomputed
    assert rerun.failed == {}

    calls.clear()
    run_baseline(spec(), ["AAPL"], "2026-07-01", "2026-07-03", None,
                 backtest=lambda *a: calls.append(a[0].spec.name) or result(0.05, 1.0), store=store, fresh=True)
    assert len(calls) == 4


def test_verdicts_need_both_yardsticks(tmp_path):
    def no_equal_weight(fund, *a):
        if fund.spec.name == "pf-equal-weight":
            raise RuntimeError("boom")
        return result(0.5, 50.0)
    report = run_baseline(spec(), ["AAPL"], "2026-07-01", "2026-07-03", None, backtest=no_equal_weight)
    assert all(r.beats_benchmarks is None for r in report.rows)
