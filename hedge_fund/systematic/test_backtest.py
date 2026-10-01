"""End-to-end systematic backtests on a synthetic market."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from hedge_fund.data.security_events import SecurityEvent
from hedge_fund.systematic import MarketPanel
from hedge_fund.systematic.backtest import BacktestConfig, SystematicBacktester
from hedge_fund.systematic.ensemble import EnsembleConfig, StrategyEvidence
from hedge_fund.systematic.execution import CostModel, FillTiming
from hedge_fund.systematic.regime import SimpleRegime
from hedge_fund.systematic.risk import RiskConfig
from hedge_fund.systematic.strategies import MeanReversion, StrategyConfig, SystematicStrategy, TimeSeriesMomentum
from hedge_fund.systematic.testing import RawBar, SyntheticMarket, weekdays

DAYS = weekdays("2021-01-04", "2023-12-29")
NAMES = ["SPY", "A", "B", "C", "D", "DIVY", "GONE"]


class FixedConfig(StrategyConfig):
    hold: tuple[str, ...] = ()


def market(days=DAYS) -> SyntheticMarket:
    m = SyntheticMarket()
    m.add_series("SPY", days, SyntheticMarket.random_walk(days, seed=0, drift=0.0003, vol=0.01), volume=50_000_000, gap=0.001)
    for i, s in enumerate("ABCD"):
        m.add_series(s, days, SyntheticMarket.random_walk(days, seed=10 + i, drift=0.0006 * (i - 1.5), vol=0.015),
                     volume=2_000_000, gap=0.002)
    # DIVY: steady riser paying $0.50 quarterly
    bars = []
    for i, d in enumerate(days):
        px = 50 * (1 + 0.0004) ** i
        bars.append(RawBar(d, px, px * 1.01, px * 0.99, px, 1_000_000, dividend=0.5 if i % 63 == 40 else 0.0))
    m.add("DIVY", bars)
    # GONE: acquired 2022-09-30, vendor keeps printing afterwards
    walk = SyntheticMarket.random_walk(days, seed=99, drift=0.001, vol=0.01)
    last = walk[max(i for i, d in enumerate(days) if d <= "2022-09-30")] if days[0] <= "2022-09-30" else walk[0]
    m.add("GONE", [RawBar(d, p, p * 1.01, p * 0.99, p, 1_000_000) if d <= "2022-09-30"
                   else RawBar(d, last, last, last, last, 3)          # vendor placeholder print
                   for d, p in zip(days, walk)])
    m.events["GONE"] = SecurityEvent(ticker="GONE", last_trading_day="2022-09-30", event_type="acquisition")
    return m


def build(m=None, end=DAYS[-1]):
    m = m or market()
    return MarketPanel.build(m, NAMES, DAYS[0], end)


def config(**kw) -> BacktestConfig:
    base = dict(start="2022-01-03", end=DAYS[-1], capital=100_000.0, rebalance="monthly",
                timing=FillTiming.NEXT_OPEN,
                costs=CostModel(commission_bps=1.0, half_spread_bps=2.0, impact_coef=0.1),
                risk=RiskConfig(max_position=0.3, vol_target=None, max_turnover=None, max_cluster_weight=1.0,
                                max_adv_participation=1.0, max_drawdown=0.9, max_daily_loss=0.9))
    return BacktestConfig(**{**base, **kw})


def tsmom():
    return TimeSeriesMomentum(lookback=126, skip=5, vol_lookback=40, exclude=("SPY",))


@pytest.fixture(scope="module")
def panel():
    return build()


@pytest.fixture(scope="module")
def result(panel):
    return SystematicBacktester(panel, [tsmom()], config()).run()


def test_runs_and_accounts_consistently(result):
    assert len(result.equity) == len(result.sessions) and result.equity.iloc[0] == pytest.approx(100_000.0)
    assert abs(result.reconciliation_error) < 1e-6
    assert result.trades and result.decisions
    assert result.metrics["transaction_costs"] > 0
    json.dumps(result.to_dict())                       # fully serializable receipt


def test_no_fill_on_or_before_its_decision_session(result):
    for t in result.trades:
        assert t["fill_session"] > t["decision_session"] == t["data_as_of"]


def test_every_trade_is_explainable(result):
    keys = {"decision_session", "data_as_of", "strategies", "ensemble", "risk_checks", "target_weight",
            "expected_price", "fill_price", "fill_reference_price", "commission", "spread_cost", "impact_cost"}
    for t in result.trades:
        assert keys <= set(t)
        assert t["strategies"]["tsmom"]["config_hash"] == tsmom().config_hash()
        side = 1 if t["side"] == "buy" else -1
        assert (t["fill_price"] - t["fill_reference_price"]) * side >= 0      # costs never help


def test_deterministic_repeat(panel, result):
    again = SystematicBacktester(panel, [tsmom()], config()).run()
    assert again.to_dict() == result.to_dict()


def test_costs_reduce_performance(panel, result):
    free = SystematicBacktester(panel, [tsmom()], config(costs=CostModel(half_spread_bps=0, impact_coef=0,
                                                                            max_participation=1.0))).run()
    assert free.equity.iloc[-1] > result.equity.iloc[-1]


def test_next_close_timing_also_respects_the_rule(panel):
    r = SystematicBacktester(panel, [tsmom()], config(timing=FillTiming.NEXT_CLOSE)).run()
    assert all(t["fill_session"] > t["decision_session"] for t in r.trades)


def test_dividends_are_credited_and_benchmark_is_total_return(panel):
    r = SystematicBacktester(panel, [TimeSeriesMomentum(lookback=60, skip=0, vol_lookback=20,
                                                        exclude=("SPY",))], config()).run()
    held_divy = any(t["symbol"] == "DIVY" for t in r.trades)
    assert held_divy and r.metrics["dividends"] > 0
    assert "benchmark_total_return" in r.metrics and len(r.benchmark_total_return) == len(r.sessions)


class Fixed(SystematicStrategy):
    """Always long the given names — a deterministic probe, not a strategy."""

    name, version, Config = "fixed", "test", FixedConfig

    def _generate(self, view, names):
        return [self._signal(view, t, 1.0 if t in self.config.hold else 0.0, {}) for t in names]


def test_delisted_holding_is_liquidated_at_last_tradable_close(panel):
    r = SystematicBacktester(panel, [Fixed(hold=("GONE", "A"))],
                             config(start="2022-03-01", end="2022-12-30")).run()
    bought = [t for t in r.trades if t["symbol"] == "GONE" and t["side"] == "buy"]
    assert bought and all(t["fill_session"] <= "2022-09-30" for t in bought)
    (liq,) = r.liquidations
    last_close = panel.as_of("2022-09-30").close("GONE")
    assert liq["symbol"] == "GONE" and liq["last_tradable_session"] == "2022-09-30"
    assert liq["price"] == pytest.approx(last_close)
    assert liq["session"] > "2022-09-30"
    # no fill of GONE after its last trading day (rebalance attempts are rejected)
    assert all(t["fill_session"] <= "2022-09-30" for t in r.trades if t["symbol"] == "GONE")
    assert abs(r.reconciliation_error) < 1e-6


def test_kill_switch_flattens_the_book(panel):
    r = SystematicBacktester(panel, [tsmom()], config(), kill_switch_on="2023-06-01").run()
    assert r.halted == "kill switch engaged"
    later = [d for d in r.decisions if d["session"] >= "2023-06-01"]
    assert later and all(d["risk"]["halted"] and d["risk"]["weights"] == {} for d in later)
    assert r.exposure.iloc[-1] == pytest.approx(0.0)


def test_ensemble_mode_requires_evidence_and_records_decisions(panel):
    strategies = [tsmom(), MeanReversion(exclude=("SPY",))]
    with pytest.raises(ValueError, match="evidence"):
        SystematicBacktester(panel, strategies, config())
    ev = {"tsmom": StrategyEvidence(name="tsmom", oos_sharpe=0.8, oos_observations=500, edge_bps=60),
          "meanrev": StrategyEvidence(name="meanrev", oos_sharpe=0.4, oos_observations=500, edge_bps=20)}
    r = SystematicBacktester(panel, strategies, config(ensemble=EnsembleConfig(round_trip_cost_bps=8)),
                             evidence=ev, regime=SimpleRegime()).run()
    d = r.decisions[-1]
    assert d["regime"] is not None and set(d["strategy_weights"]) == {"tsmom", "meanrev"}
    assert any(v["decision"] == "NO_TRADE" for dec in r.decisions for v in dec["ensemble"].values())


def test_equity_up_to_a_date_does_not_depend_on_later_data():
    """Same backtest on a vendor history truncated at X: identical equity through X."""
    x = "2023-03-31"
    full = SystematicBacktester(build(), [tsmom()], config(end=x)).run()
    m = market()
    m.bars = {k: [b for b in v if b.date <= x] for k, v in m.bars.items()}
    cut = SystematicBacktester(build(m, end=x), [tsmom()], config(end=x)).run()
    pd.testing.assert_series_equal(full.equity, cut.equity)
    assert full.trades == cut.trades


def test_attribution_sums_to_total_pnl(result):
    total = result.equity.iloc[-1] - result.equity.iloc[0]
    assert sum(result.pnl_by_symbol.values()) == pytest.approx(total, rel=1e-9, abs=1e-6)
    assert len(result.net_exposure) == len(result.sessions)
    assert (result.net_exposure <= result.exposure + 1e-12).all()
