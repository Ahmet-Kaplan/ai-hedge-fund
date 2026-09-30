"""Risk engine: each limit alone, the halts, immutability and fractional Kelly."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from hedge_fund.systematic import MarketPanel
from hedge_fund.systematic.risk import KellyConfig, RiskConfig, RiskEngine, RiskState, kelly_fraction
from hedge_fund.systematic.testing import SyntheticMarket, weekdays

DAYS = weekdays("2023-01-02", "2023-06-30")


@pytest.fixture(scope="module")
def view():
    m = SyntheticMarket()
    base = SyntheticMarket.random_walk(DAYS, seed=1, vol=0.02)
    m.add_series("SPY", DAYS, SyntheticMarket.random_walk(DAYS, seed=0, vol=0.01), volume=10_000_000)
    m.add_series("A", DAYS, base, volume=1_000_000)
    m.add_series("A2", DAYS, [p * 1.5 for p in base], volume=1_000_000)          # perfectly correlated with A
    m.add_series("B", DAYS, SyntheticMarket.random_walk(DAYS, seed=7, vol=0.02), volume=1_000_000)
    m.add_series("THIN", DAYS, SyntheticMarket.random_walk(DAYS, seed=3, vol=0.01), volume=100)
    return MarketPanel.build(m, ["SPY", "A", "A2", "B", "THIN"], DAYS[0], DAYS[-1]).as_of(DAYS[-1])


def eng(**kw):
    base = dict(max_turnover=None, vol_target=None, max_cluster_weight=1.0, max_adv_participation=1.0)
    return RiskEngine(RiskConfig(**{**base, **kw}))


def state(eq=100_000.0, peak=None, day=None, kill=False):
    return RiskState(peak_equity=peak or eq, day_start_equity=day or eq, kill_switch=kill)


def run(e, targets, current=None, eq=100_000.0, st=None, view=None):
    return e.apply(targets, equity=eq, current=current or {}, state=st or state(eq), view=view)


def test_position_gross_net_caps():
    d = run(eng(max_position=0.2, max_gross=0.5), {"A": 0.4, "B": 0.3, "C": 0.2})
    assert d.weights["A"] == pytest.approx(0.2 * 0.5 / 0.6)
    assert sum(abs(v) for v in d.weights.values()) == pytest.approx(0.5)
    checks = [i["check"] for i in d.interventions]
    assert "max_position" in checks and "max_gross" in checks


def test_long_only_zeroes_shorts_and_net_limits():
    assert run(eng(), {"A": -0.1, "B": 0.1}).weights == {"B": pytest.approx(0.1)}
    e = eng(allow_short=True, min_net=-0.1, max_net=0.2, max_position=1.0, max_gross=2.0)
    d = run(e, {"A": 0.5, "B": 0.1})
    assert sum(d.weights.values()) == pytest.approx(0.2)
    d2 = run(e, {"A": -0.5})
    assert sum(d2.weights.values()) == pytest.approx(-0.1)


def test_turnover_limit_moves_part_way():
    d = run(eng(max_turnover=0.2), {"A": 0.1, "B": 0.1}, current={"C": 0.1})
    turn = sum(abs(d.weights.get(t, 0) - {"C": 0.1}.get(t, 0)) for t in ("A", "B", "C"))
    assert turn == pytest.approx(0.2)


def test_adv_participation_caps_trade_size(view):
    d = run(eng(max_adv_participation=0.01, max_position=1.0), {"THIN": 0.5}, view=view)
    assert 0 < d.weights["THIN"] < 0.001
    assert any(i["check"] == "adv_participation" for i in d.interventions)


def test_correlated_names_share_one_cluster_cap(view):
    d = run(eng(max_cluster_weight=0.15, max_position=1.0), {"A": 0.1, "A2": 0.1, "B": 0.1}, view=view)
    assert d.weights["A"] + d.weights["A2"] == pytest.approx(0.15)
    assert d.weights["B"] == pytest.approx(0.1)


def test_vol_target_scales_down_only(view):
    hot = run(eng(vol_target=0.05, max_position=1.0), {"A": 0.5, "B": 0.5}, view=view)
    assert sum(hot.weights.values()) < 1.0 and any(i["check"] == "vol_target" for i in hot.interventions)
    calm = run(eng(vol_target=5.0, max_position=1.0), {"A": 0.1}, view=view)
    assert calm.weights == {"A": pytest.approx(0.1)}


def test_kill_switch_and_drawdown_breaker_flatten():
    k = run(eng(), {"A": 0.1}, current={"A": 0.1}, st=state(kill=True))
    assert k.halted and k.weights == {} and "kill" in k.reason
    s = state(eq=79_000, peak=100_000)
    dd = run(eng(max_drawdown=0.2), {"A": 0.1}, eq=79_000, st=s)
    assert dd.halted and dd.weights == {} and s.halted_reason


def test_daily_loss_makes_the_book_reduce_only():
    s = state(eq=96_000, peak=100_000, day=100_000)
    d = run(eng(max_daily_loss=0.03), {"A": 0.2, "B": 0.05, "C": 0.1}, current={"A": 0.1, "B": 0.1}, eq=96_000, st=s)
    assert d.weights == {"A": pytest.approx(0.1), "B": pytest.approx(0.05)}      # no increase, no new name


def test_limits_are_immutable_and_hashed():
    c = RiskConfig()
    with pytest.raises(ValidationError):
        c.max_position = 0.9
    assert RiskConfig(max_position=0.2).config_hash() != c.config_hash()
    with pytest.raises(ValidationError):
        RiskConfig(min_net=-0.5)                    # shorts off
    with pytest.raises(ValidationError):
        KellyConfig(fraction=1.0)                   # full Kelly is never allowed


def test_non_finite_target_is_rejected():
    with pytest.raises(ValueError):
        run(eng(), {"A": float("nan")})


def test_fractional_kelly_needs_evidence():
    rng = np.random.default_rng(0)
    strong = rng.normal(0.002, 0.01, 1000)
    cfg = KellyConfig(enabled=True, fraction=0.25)
    k = kelly_fraction(strong, cfg)
    full = strong.mean() / strong.std(ddof=1) ** 2
    assert k == pytest.approx(min(0.25 * full, 1.0))
    assert kelly_fraction(strong[:100], cfg) is None                     # too short
    assert kelly_fraction(rng.normal(0.0, 0.01, 1000), cfg) is None      # not significant
    assert kelly_fraction(strong, KellyConfig()) is None                 # disabled by default
