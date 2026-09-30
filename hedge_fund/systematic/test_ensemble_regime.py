"""Regime labels, ensemble weighting and NO_TRADE, portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hedge_fund.models import Signal
from hedge_fund.systematic import MarketPanel
from hedge_fund.systematic.ensemble import Ensemble, EnsembleConfig, StrategyEvidence
from hedge_fund.systematic.portfolio import PortfolioConfig, target_weights
from hedge_fund.systematic.regime import RegimeState, SimpleRegime, SimpleRegimeConfig
from hedge_fund.systematic.testing import SyntheticMarket, weekdays

DAYS = weekdays("2021-01-04", "2023-12-29")


def panel_with(spy_path):
    m = SyntheticMarket()
    m.add_series("SPY", DAYS, spy_path, volume=10_000_000)
    m.add_series("LOWVOL", DAYS, SyntheticMarket.random_walk(DAYS, seed=2, vol=0.005))
    m.add_series("HIVOL", DAYS, SyntheticMarket.random_walk(DAYS, seed=3, vol=0.03))
    return MarketPanel.build(m, ["SPY", "LOWVOL", "HIVOL"], DAYS[0], DAYS[-1])


def test_regime_uptrend_calm_is_risk_on():
    n = len(DAYS)
    path = list(100 * np.exp(np.linspace(0, 0.5, n)) * (1 + 0.002 * np.sin(np.arange(n))))
    r = SimpleRegime().classify(panel_with(path).as_of(DAYS[-1]))
    assert (r.trend, r.risk) == ("up", "on") and r.features["trend_gap"] > 0.02


def test_regime_crash_is_down_high_vol_risk_off():
    rng = np.random.default_rng(1)
    n = len(DAYS)
    steps = np.r_[rng.normal(0.0003, 0.006, n - 60), rng.normal(-0.006, 0.03, 60)]
    path = list(100 * np.exp(np.cumsum(steps)))
    r = SimpleRegime().classify(panel_with(path).as_of(DAYS[-1]))
    assert r.trend == "down" and r.volatility == "high" and r.risk == "off"
    assert r.features["drawdown"] < -0.10


def test_regime_unknown_without_history():
    path = SyntheticMarket.random_walk(DAYS, seed=5)
    r = SimpleRegime(SimpleRegimeConfig()).classify(panel_with(path).as_of(DAYS[30]))
    assert (r.trend, r.volatility, r.risk) == ("unknown", "unknown", "unknown")


def sig(model, ticker, value, abstain=False):
    return Signal(model_name=model, ticker=ticker, date="2023-06-30", value=value,
                  metadata={"abstained": abstain})


EV = {
    "tsmom": StrategyEvidence(name="tsmom", oos_sharpe=1.0, oos_observations=500, edge_bps=40,
                              regimes=frozenset({"trend:up", "trend:down"})),
    "meanrev": StrategyEvidence(name="meanrev", oos_sharpe=0.5, oos_observations=500, edge_bps=30,
                                regimes=frozenset({"trend:range"})),
    "broken": StrategyEvidence(name="broken", oos_sharpe=-0.3, oos_observations=500, edge_bps=50),
}


def test_weights_follow_oos_evidence_and_regime():
    e = Ensemble(EnsembleConfig(max_strategy_share=1.0))
    up = RegimeState(session="x", trend="up", volatility="low", risk="on")
    w = e.strategy_weights(EV, up)
    assert w["broken"] == 0.0                         # negative OOS Sharpe gets nothing
    assert w["meanrev"] == 0.0 and w["tsmom"] == pytest.approx(1.0)   # off-regime
    rng = RegimeState(session="x", trend="range")
    assert e.strategy_weights(EV, rng)["meanrev"] == pytest.approx(1.0)


def test_confidence_and_correlation_penalty():
    ev = {"a": StrategyEvidence(name="a", oos_sharpe=1.0, oos_observations=100, edge_bps=10),
          "b": StrategyEvidence(name="b", oos_sharpe=1.0, oos_observations=500, edge_bps=10)}
    w = Ensemble(EnsembleConfig(max_strategy_share=1.0)).strategy_weights(ev)
    assert w["b"] == pytest.approx(5 * w["a"])
    ev3 = {**ev, "c": StrategyEvidence(name="c", oos_sharpe=1.0, oos_observations=500, edge_bps=10)}
    corr = pd.DataFrame(np.eye(3), index=list("abc"), columns=list("abc"))
    corr.loc["b", "c"] = corr.loc["c", "b"] = 0.9
    w3 = Ensemble(EnsembleConfig(max_strategy_share=1.0)).strategy_weights(ev3, correlation=corr)
    assert w3["b"] == pytest.approx(w3["c"]) and w3["b"] < w3["a"] * 5


def test_max_strategy_share_caps_concentration():
    w = Ensemble(EnsembleConfig(max_strategy_share=0.6)).strategy_weights(
        {"a": StrategyEvidence(name="a", oos_sharpe=3.0, oos_observations=500, edge_bps=1),
         "b": StrategyEvidence(name="b", oos_sharpe=1.0, oos_observations=500, edge_bps=1)})
    assert w["a"] == pytest.approx(0.6) and w["b"] == pytest.approx(0.4)


def test_combine_trades_only_when_edge_beats_costs():
    e = Ensemble(EnsembleConfig(round_trip_cost_bps=10, min_edge_after_cost_bps=5, max_strategy_share=1.0))
    signals = {"tsmom": [sig("tsmom", "A", 1.0), sig("tsmom", "B", 0.2)],
               "broken": [sig("broken", "A", -1.0)]}
    d = e.combine("2023-06-30", signals, EV)
    assert d.instruments["A"].decision == "TRADE" and d.instruments["A"].score == pytest.approx(1.0)
    assert d.instruments["A"].expected_edge_bps == pytest.approx(40.0)
    b = d.instruments["B"]
    assert b.decision == "NO_TRADE" and b.score == 0.0 and "minimum" in b.reason   # 8bp edge < 10+5
    assert d.scores() == {"A": pytest.approx(1.0)}


def test_abstentions_do_not_vote_and_unknown_strategy_is_refused():
    e = Ensemble(EnsembleConfig(max_strategy_share=1.0, round_trip_cost_bps=0, min_edge_after_cost_bps=0))
    d = e.combine("d", {"tsmom": [sig("tsmom", "A", 0.9, abstain=True)]}, EV)
    assert d.instruments == {}
    with pytest.raises(ValueError, match="evidence"):
        e.combine("d", {"mystery": [sig("mystery", "A", 1.0)]}, EV)


def test_inverse_vol_weights_long_only():
    view = panel_with(SyntheticMarket.random_walk(DAYS, seed=0)).as_of(DAYS[-1])
    w = target_weights({"LOWVOL": 0.5, "HIVOL": 0.5, "SPY": -0.5}, view, PortfolioConfig(gross_target=1.0))
    assert set(w) == {"LOWVOL", "HIVOL"} and sum(w.values()) == pytest.approx(1.0)
    assert w["LOWVOL"] > 3 * w["HIVOL"]
    ls = target_weights({"LOWVOL": 0.5, "SPY": -0.5}, view, PortfolioConfig(long_only=False))
    assert ls["SPY"] < 0 and sum(abs(v) for v in ls.values()) == pytest.approx(1.0)
    assert target_weights({"LOWVOL": 0.01}, view, PortfolioConfig()) == {}
