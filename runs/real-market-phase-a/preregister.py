"""STEPS 3-5 — freeze configs and pre-register the validation plan.

Writes preregistration.json and strategy_configs.json and records one
'preregistered' entry per strategy in the experiment registry. Runs no
backtest and reads no returns. Must be committed before run_phase_a.py.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from hedge_fund.systematic.strategies import REGISTRY
from hedge_fund.validation import ExperimentRegistry, load_gates

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/real-market-phase-a")
FAMILY = "phase-a-reference"

strategies = {name: cls() for name, cls in REGISTRY.items()}          # code defaults, untouched
configs = {name: {"spec": s.spec(), "config_hash": s.config_hash()} for name, s in strategies.items()}

prereg = {
    "title": "Phase A: out-of-sample evidence for four frozen reference strategies on real PIT data",
    "family": FAMILY,
    "data": {
        "prices": "Tiingo daily OHLCV (split-adjusted, restated per as-of view), local cache, offline",
        "corporate_actions": "Tiingo splitFactor (incl. GE spin-offs encoded as factors) and divCash",
        "universe": "runs/buffett-baseline/universe_top10.json — survivorship-aware PIT top-10 by market cap, "
                    "annual reconstitution 2016-07-01..2025-07-01; membership = latest snapshot on or before the session",
        "panel_window": ["2015-06-01", "2026-06-30"],
        "warmup_note": "bars before 2016-07-01 are history only; no instrument is a member before the first snapshot",
        "benchmark": "SPY total return (dividends reinvested) — the portfolio is credited dividends",
        "risk_free_rate": "0 (no rate series in the cache); Sharpe is excess over zero",
        "financial_datasets": "not used (policy fail-closed, key absent)",
    },
    "strategies": {n: {"config_hash": c["config_hash"], "rebalance": r}
                   for (n, c), r in zip(configs.items(), ["monthly", "monthly", "weekly", "weekly"])},
    "rebalance_rationale": "momentum signals use 252-day lookbacks (monthly); mean reversion (20-day) and breakout "
                           "(55-day channel) are short-horizon (weekly). Fixed before any result was seen.",
    "backtest": {
        "capital_usd": 100_000.0,
        "execution": "next_open — orders decided on session D's close fill at D+1's open",
        "costs_baseline": {"commission_bps": 1.0, "commission_min": 0.0, "half_spread_bps": 2.0,
                           "impact_coef": 0.1, "max_participation": 0.05, "adv_lookback": 20, "vol_lookback": 20},
        "slippage_model": "half-spread + square-root impact (impact_coef * daily vol * sqrt(order/ADV notional))",
        "portfolio": {"long_only": True, "gross_target": 1.0, "vol_lookback": 60, "min_score": 0.05,
                      "weighting": "inverse volatility of positive scores"},
        "risk": {"max_position": 0.20, "max_gross": 1.0, "max_net": 1.0, "min_net": 0.0, "allow_short": False,
                 "max_turnover": None, "max_adv_participation": 0.05, "vol_target": None,
                 "max_cluster_weight": 1.0, "max_daily_loss": 1.0, "max_drawdown": 1.0},
        "risk_rationale": "measure the strategies' own behaviour; operational breakers (drawdown halt, daily loss, "
                          "vol target) are switched off so they cannot mask or create an edge",
    },
    "periods": {
        "development": ["2016-07-01", "2022-12-30"],
        "embargo_calendar_days": 30,
        "locked_holdout": ["2023-02-01", "2026-06-30"],
        "holdout_rule": "evaluated exactly once per strategy via LockedHoldout; fresh capital at holdout start; "
                        "no change to any config after a holdout result",
    },
    "validation": {
        "walk_forward": "frozen configs (nothing is fitted): consecutive calendar-year test windows 2017..2022 "
                        "on the development backtest; statistic = share of windows with positive net Sharpe",
        "purged_kfold": {"k": 6, "label_horizon_sessions": {"monthly": 21, "weekly": 5}, "embargo_sessions": 5},
        "cpcv": {"n_groups": 6, "k_test": 2},
        "psr_benchmark_sharpe": 0.0,
        "dsr": "n_trials = distinct configs in the registry family; SR variance across those trials",
        "pbo": {"configs": "the four strategies' daily development returns", "n_splits": 8},
        "bootstrap": {"block_sessions": 21, "samples": 2000, "seed": 20261001, "alpha": 0.05},
        "monte_carlo": {"what": "reshuffled round-trip P&L", "samples": 2000, "seed": 20261001, "ruin_level": 0.5},
        "sufficiency": {"min_daily_observations": 252, "min_round_trips": 30,
                        "otherwise": "INSUFFICIENT_EVIDENCE (no number reported)"},
        "gates_file": "configs/validation-gates.yaml",
        "gates_hash": load_gates().config_hash(),
    },
    "holdout_pass_criterion": {"annualized_sharpe_min": 0.5, "max_drawdown_max": 0.25, "total_return_min": 0.0},
    "cost_stress": {"multipliers": [1.0, 1.5, 2.0, 3.0], "applies_to": "commission, half-spread, impact coefficient",
                    "window": "development only (the holdout is not re-run)",
                    "fragile_if": "Sharpe at 2x costs <= 0, or below half of the baseline Sharpe"},
    "small_account": {
        "capital": 200.0, "currency_note": "EUR 200 modelled as 200 USD-equivalent units; FX not modelled",
        "window": "development only",
        "costs": {"commission_min_per_order": 1.0, "half_spread_bps": 5.0, "slippage_bps": 5.0,
                  "modelled_as": "commission_min=1.0, half_spread_bps=10 (spread+slippage), impact_coef 0.1"},
        "instruments": "fractional (1e-6 shares), min order notional 1.0",
        "variants": ["raw: every order the strategy produces",
                     "economics_filtered: drop orders whose expected edge (development round-trip expectancy in bp "
                     "from the 100k development run, never the holdout) does not clear round-trip cost + 20bp"],
    },
    "classification": {
        "REJECT": "development net Sharpe <= 0, or the locked holdout fails its criterion",
        "INSUFFICIENT_EVIDENCE": "a required statistic lacks sample, or development Sharpe > 0 but any gate fails",
        "EVIDENCE_SUPPORTS_FURTHER_RESEARCH": "all gates pass on development AND the holdout passes AND the strategy "
                                              "is not fragile at 2x costs",
        "note": "criteria fixed here; they are not changed after results",
    },
    "ensemble": "only if at least two strategies are EVIDENCE_SUPPORTS_FURTHER_RESEARCH; evidence from development only",
}
blob = json.dumps(prereg, sort_keys=True, separators=(",", ":"))
prereg["preregistration_hash"] = hashlib.sha256(blob.encode()).hexdigest()

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "preregistration.json").write_text(json.dumps(prereg, indent=1, sort_keys=True))
(OUT / "strategy_configs.json").write_text(json.dumps(configs, indent=1, sort_keys=True))
reg = ExperimentRegistry(OUT / "experiments.jsonl")
if not reg.records():
    for name, c in configs.items():
        reg.record(family=FAMILY, spec=c["spec"], stage="preregistered", metrics={},
                   notes=f"preregistration_hash={prereg['preregistration_hash']}")
reg.verify()
print(prereg["preregistration_hash"])
print(json.dumps({n: c["config_hash"] for n, c in configs.items()}))
