"""Splits, overfitting statistics, registry chain, locked holdout and gates."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats as sps

from hedge_fund.validation import (
    ExperimentRegistry, GateConfig, HoldoutViolation, LockedHoldout, RegistryTampered, block_bootstrap,
    bootstrap_sharpe_ci, cpcv, deflated_sharpe, evaluate_gates, expected_max_sharpe, load_gates,
    monte_carlo_trades, pbo, probabilistic_sharpe, purged_kfold, walk_forward,
)


# -- splits -----------------------------------------------------------------

def test_walk_forward_tests_strictly_after_training():
    splits = walk_forward(100, train=40, test=20, gap=2)
    assert len(splits) == 2                          # tests 42-61, 62-81; 82-101 does not fit
    for s in splits:
        assert max(s.train) + 2 < min(s.test) and len(s.test) == 20
    exp = walk_forward(100, train=40, test=20, expanding=True)
    assert exp[-1].train[0] == 0 and len(exp[-1].train) > len(exp[0].train)


def test_purged_kfold_removes_overlap_and_embargo():
    folds = purged_kfold(100, 5, label_horizon=3, embargo=4)
    f = folds[2]                                     # test 40..59
    assert min(f.test) == 40 and max(f.test) == 59
    assert all(i + 3 < 40 or i > 59 + 4 for i in f.train)
    # label at i spans [i, i+3]: 37..59 overlap the test block, 60..63 are embargoed
    assert 37 not in f.train and 63 not in f.train and 36 in f.train and 64 in f.train


def test_cpcv_paths():
    splits = cpcv(60, 6, 2, embargo=1)
    assert len(splits) == math.comb(6, 2)
    for s in splits:
        assert not set(s.train) & set(s.test)


# -- statistics ---------------------------------------------------------------

def test_psr_matches_the_formula():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 500)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = sps.skew(r, bias=False), sps.kurtosis(r, fisher=False, bias=False)
    expect = sps.norm.cdf(sr * math.sqrt(499) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    assert probabilistic_sharpe(r) == pytest.approx(expect)


def test_deflated_sharpe_penalizes_more_trials():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0015, 0.01, 750)
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(1000, 0.01) > expected_max_sharpe(10, 0.01) > 0
    d1, d100, d10k = (deflated_sharpe(r, n) for n in (1, 100, 10_000))
    assert d1 == pytest.approx(probabilistic_sharpe(r))
    assert d1 > d100 > d10k


def test_pbo_low_for_a_real_edge_high_for_pure_noise():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, (400, 20))
    assert pbo(noise, n_splits=8) > 0.25
    skilled = noise.copy()
    skilled[:, 0] += 0.004                         # one config truly better everywhere
    assert pbo(skilled, n_splits=8) < 0.1


def test_bootstrap_and_monte_carlo_are_seeded():
    r = np.random.default_rng(3).normal(0.001, 0.01, 300)
    a, b = block_bootstrap(r, n_samples=50, block=10, seed=7), block_bootstrap(r, n_samples=50, block=10, seed=7)
    assert a.shape == (50, 300) and np.array_equal(a, b)
    lo, hi = bootstrap_sharpe_ci(r, n_samples=200, seed=1)
    assert lo < r.mean() / r.std(ddof=1) < hi
    mc = monte_carlo_trades([100, -50, 30, -80, 60] * 10, capital=1000, n_samples=200, seed=0)
    assert 0 <= mc["max_drawdown_median"] <= mc["max_drawdown_p95"] <= 1


# -- registry -------------------------------------------------------------------

def test_registry_is_append_only_and_tamper_evident(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.jsonl")
    reg.record(family="tsmom", spec={"lookback": 126}, stage="walk_forward", metrics={"sharpe": 0.4})
    reg.record(family="tsmom", spec={"lookback": 252}, stage="walk_forward", metrics={"sharpe": 0.6})
    reg.record(family="tsmom", spec={"lookback": 252}, stage="bootstrap", metrics={"lo": 0.1})
    reg.verify()
    assert reg.n_trials("tsmom") == 2 and reg.n_trials("other") == 0
    lines = (tmp_path / "exp.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    rec["metrics"]["sharpe"] = 9.9                   # quietly improve history
    (tmp_path / "exp.jsonl").write_text("\n".join([json.dumps(rec, sort_keys=True), *lines[1:]]) + "\n")
    with pytest.raises(RegistryTampered):
        reg.verify()


# -- locked holdout ---------------------------------------------------------------

def test_holdout_is_single_use_and_blocks_derived_tuning(tmp_path):
    reg = ExperimentRegistry(tmp_path / "exp.jsonl")
    hold = LockedHoldout("2025-01-01", "2025-12-31", reg, family="tsmom", embargo_days=10)
    calls = []

    def run(start, end):
        calls.append((start, end))
        return {"sharpe": -0.2}

    spec = {"strategy": "tsmom", "lookback": 252}
    out = hold.evaluate(spec, run, passed=lambda m: m["sharpe"] > 0.5)
    assert out["passed"] is False and calls == [("2025-01-01", "2025-12-31")]
    with pytest.raises(HoldoutViolation, match="already"):
        hold.evaluate(spec, run, passed=lambda m: True)
    with pytest.raises(HoldoutViolation, match="parent"):
        hold.evaluate({**spec, "lookback": 200}, run, passed=lambda m: True, parent_spec=spec)
    assert len(calls) == 1


def test_development_windows_cannot_touch_the_holdout(tmp_path):
    hold = LockedHoldout("2025-01-01", "2025-12-31", ExperimentRegistry(tmp_path / "e"), family="x",
                         embargo_days=10)
    hold.check_development_window("2016-01-01", "2024-12-20")
    for window in (("2016-01-01", "2024-12-25"), ("2016-01-01", "2025-03-01"), ("2025-02-01", "2025-06-01")):
        with pytest.raises(HoldoutViolation):
            hold.check_development_window(*window)


# -- gates ----------------------------------------------------------------------------

GOOD = {"n_trades": 120, "oos_sharpe_annual": 0.9, "deflated_sharpe": 0.97, "pbo": 0.2, "max_drawdown": 0.15,
        "walk_forward_positive_share": 0.75, "bootstrap_sharpe_lower": 0.05}


def test_gates_pass_and_fail_per_check():
    g = load_gates()
    assert evaluate_gates(GOOD, g).passed
    bad = evaluate_gates({**GOOD, "pbo": 0.7}, g)
    assert not bad.passed and bad.checks["pbo"] is False and sum(bad.checks.values()) == len(bad.checks) - 1
    assert not evaluate_gates({}, g).passed                 # missing evidence fails closed


def test_gate_thresholds_are_frozen_and_hashed():
    g = load_gates()
    with pytest.raises(ValidationError):
        g.max_pbo = 0.99
    with pytest.raises(ValidationError):
        GateConfig(unknown_knob=1)
    assert evaluate_gates(GOOD, g).gates_hash == g.config_hash()
    assert GateConfig(max_pbo=0.4).config_hash() != g.config_hash()
