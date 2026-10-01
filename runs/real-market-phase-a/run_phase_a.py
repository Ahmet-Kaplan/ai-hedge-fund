"""STEPS 6-9, 11, 13 — real-data backtests, validation, cost stress, EUR 200, audit.

Follows preregistration.json exactly. Offline: Tiingo reads come from the
local store (offline=True). The locked holdout is evaluated once per strategy;
on a re-run its recorded result is read back from the registry, never re-run.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hedge_fund.core import FillEvent, InstrumentRegistry
from hedge_fund.data import client as fd_client
from hedge_fund.data.factory import CompositeDataClient
from hedge_fund.data.tiingo import TiingoClient
from hedge_fund.systematic import MarketPanel, metrics
from hedge_fund.systematic.backtest import BacktestConfig, SystematicBacktester
from hedge_fund.systematic.execution import CostModel, FillTiming
from hedge_fund.systematic.portfolio import PortfolioConfig
from hedge_fund.systematic.regime import SimpleRegime
from hedge_fund.systematic.risk import RiskConfig
from hedge_fund.systematic.small_account import (
    EconomicsFilter, SmallAccountConfig, probability_of_ruin, small_account_universe,
)
from hedge_fund.systematic.strategies import strategy_from_spec
from hedge_fund.universe.builder import load_schedule
from hedge_fund.validation import (
    ExperimentRegistry, LockedHoldout, bootstrap_sharpe_ci, cpcv, deflated_sharpe, evaluate_gates, load_gates,
    monte_carlo_trades, pbo, probabilistic_sharpe, purged_kfold, spec_hash,
)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/real-market-phase-a")
PRE = json.loads((OUT / "preregistration.json").read_text())
CFGS = json.loads((OUT / "strategy_configs.json").read_text())
FAMILY = PRE["family"]
DEV = PRE["periods"]["development"]
HOLD = PRE["periods"]["locked_holdout"]
B = PRE["backtest"]
PPY = 252

# -- guards: count any attempt to reach Financial Datasets ---------------------
FD_CALLS = {"constructed": 0}
_orig_init = fd_client.FDClient.__init__


def _trap(self, *a, **k):
    FD_CALLS["constructed"] += 1
    return _orig_init(self, *a, **k)


fd_client.FDClient.__init__ = _trap


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# -- data -----------------------------------------------------------------------
schedule = load_schedule("runs/buffett-baseline/universe_top10.json")
tickers = ["SPY", *schedule.all_tickers()]
tiingo = TiingoClient(cache_dir=Path.home() / ".hedge-fund" / "cache" / "tiingo", offline=True)
data = CompositeDataClient(tiingo, None)
panel = MarketPanel.build(data, tickers, PRE["data"]["panel_window"][0], PRE["data"]["panel_window"][1],
                          schedule=schedule)
log(f"panel {panel.info.sessions} sessions x {len(tickers)} instruments")


def cost_model(mult: float = 1.0, base: dict | None = None) -> CostModel:
    c = dict(base or B["costs_baseline"])
    for k in ("commission_bps", "commission_min", "half_spread_bps", "impact_coef"):
        c[k] = c.get(k, 0.0) * mult
    return CostModel(**c)


def bt_config(name, start, end, capital, costs) -> BacktestConfig:
    r = dict(B["risk"])
    return BacktestConfig(start=start, end=end, capital=capital, rebalance=PRE["strategies"][name]["rebalance"],
                          timing=FillTiming.NEXT_OPEN, costs=costs,
                          portfolio=PortfolioConfig(long_only=True, gross_target=1.0, vol_lookback=60, min_score=0.05),
                          risk=RiskConfig(**r), benchmark="SPY")


def strategy(name):
    s = strategy_from_spec(CFGS[name]["spec"])
    assert s.config_hash() == CFGS[name]["config_hash"] == PRE["strategies"][name]["config_hash"], name
    return s


def run(name, start, end, capital=B["capital_usd"], costs=None, instruments=None, order_filter=None):
    t0 = time.time()
    res = SystematicBacktester(panel, [strategy(name)], bt_config(name, start, end, capital, costs or cost_model()),
                               regime=SimpleRegime(), instruments=instruments, order_filter=order_filter).run()
    log(f"  {name} {start}..{end} cap={capital:g} -> {res.equity.iloc[-1]:,.2f} ({time.time() - t0:.0f}s)")
    return res


# -- analysis helpers --------------------------------------------------------------

def fills_of(res) -> list[FillEvent]:
    out = []
    for t in res.trades:
        out.append(FillEvent(client_order_id=t["client_order_id"], symbol=t["symbol"], side=t["side"],
                             quantity=t["quantity"], price=t["fill_price"], session=t["fill_session"],
                             mid_price=t["fill_reference_price"], commission=t["commission"],
                             spread_cost=t["spread_cost"], impact_cost=t["impact_cost"]))
    for liq in res.liquidations:
        if liq["price"] > 0:
            out.append(FillEvent(client_order_id="liq", symbol=liq["symbol"], side="sell", quantity=abs(liq["quantity"]),
                                 price=liq["price"], session=liq["session"], mid_price=liq["price"]))
    return sorted(out, key=lambda f: f.session)


def drawdown_periods(eq: pd.Series, n=3):
    dd = eq / eq.cummax() - 1
    out, i = [], 0
    vals, idx = dd.to_numpy(), list(dd.index)
    while i < len(vals):
        if vals[i] < 0:
            j = i
            while j < len(vals) and vals[j] < 0:
                j += 1
            seg = vals[i:j]
            k = i + int(np.argmin(seg))
            out.append({"start": idx[i - 1] if i else idx[i], "trough": idx[k], "depth": float(vals[k]),
                        "recovered": idx[j] if j < len(vals) else None})
            i = j
        else:
            i += 1
    return sorted(out, key=lambda d: d["depth"])[:n]


def regime_returns(res):
    rets = res.equity.pct_change().dropna()
    labels = pd.Series(index=res.equity.index, dtype=object)
    for d in res.decisions:
        r = d.get("regime") or {}
        labels.loc[d["session"]] = f"{r.get('trend')}/{r.get('volatility')}/risk-{r.get('risk')}"
    labels = labels.ffill().shift(1).reindex(rets.index)          # a day's return belongs to the prior decision
    out = {}
    for lab, grp in rets.groupby(labels):
        out[str(lab)] = {"days": int(len(grp)), "annualized_return": float(grp.mean() * PPY),
                         "sharpe": metrics.sharpe(grp) if len(grp) > 20 else None}
    return out


def per_year(res):
    eq, bench = res.equity, res.benchmark_total_return
    out = {}
    for y in sorted({d[:4] for d in eq.index}):
        e, b = eq[[d for d in eq.index if d[:4] == y]], bench[[d for d in bench.index if d[:4] == y]]
        prev_e = eq[eq.index < e.index[0]]
        prev_b = bench[bench.index < b.index[0]]
        e0 = prev_e.iloc[-1] if len(prev_e) else e.iloc[0]
        b0 = prev_b.iloc[-1] if len(prev_b) else b.iloc[0]
        out[y] = {"return": float(e.iloc[-1] / e0 - 1), "benchmark_tr": float(b.iloc[-1] / b0 - 1)}
    return out


def summarize(res) -> dict:
    fills = fills_of(res)
    trips = metrics.round_trips(fills)
    pos = {s: i for i, s in enumerate(res.sessions)}
    holds = [pos[t.exit_session] - pos[t.entry_session] for t in trips if t.entry_session in pos and t.exit_session in pos]
    m = dict(res.metrics)
    m.update({
        "round_trips": len(trips),
        "fills": len(fills),
        "average_holding_sessions": float(np.mean(holds)) if holds else None,
        "gross_exposure_mean": float(res.exposure.mean()),
        "net_exposure_mean": float(res.net_exposure.mean()),
        "final_equity": float(res.equity.iloc[-1]),
        "benchmark_total_return": float(res.benchmark_total_return.iloc[-1] / res.benchmark_total_return.iloc[0] - 1),
        "rejected_orders": len(res.rejected),
        "liquidations": len(res.liquidations),
        "reconciliation_error": float(res.reconciliation_error),
    })
    if len(trips) < PRE["validation"]["sufficiency"]["min_round_trips"]:
        for k in ("hit_rate", "average_win", "average_loss", "payoff_ratio", "profit_factor", "expectancy"):
            m[k] = "INSUFFICIENT_EVIDENCE"
    return m


def lookahead_violations(res) -> int:
    return sum(1 for t in res.trades if not (t["fill_session"] > t["decision_session"] == t["data_as_of"]))


# -- run -------------------------------------------------------------------------------

registry = ExperimentRegistry(OUT / "experiments.jsonl")
registry.verify()
holdout = LockedHoldout(HOLD[0], HOLD[1], registry, family=FAMILY,
                        embargo_days=PRE["periods"]["embargo_calendar_days"])
holdout.check_development_window(*DEV)

names = list(CFGS)
results, validation, cost_stress, small, audit = {}, {}, {}, {}, {"lookahead_violations": 0}
dev_runs = {}

log("STEP 6 development backtests")
for n in names:
    res = dev_runs[n] = run(n, *DEV)
    audit["lookahead_violations"] += lookahead_violations(res)
    s = summarize(res)
    results[n] = {"development": s, "per_year": per_year(res), "per_regime": regime_returns(res),
                  "drawdown_periods": drawdown_periods(res.equity),
                  "contribution_by_instrument": dict(sorted(res.pnl_by_symbol.items(), key=lambda kv: -kv[1]))}
    registry.record(family=FAMILY, spec=CFGS[n]["spec"], stage="development", window=tuple(DEV),
                    metrics={k: v for k, v in s.items() if isinstance(v, (int, float)) or v is None})

log("STEP 8 validation (development only)")
dev_returns = pd.DataFrame({n: dev_runs[n].equity.pct_change().dropna() for n in names}).dropna()
per_period_sr = {n: metrics.sharpe(dev_returns[n]) / math.sqrt(PPY) for n in names}
sr_var = float(np.var(list(per_period_sr.values()), ddof=1)) if len(names) > 1 else None
n_trials = registry.n_trials(FAMILY)
v = PRE["validation"]
pbo_value = pbo(dev_returns.to_numpy(), n_splits=v["pbo"]["n_splits"]) if len(dev_returns) >= 252 else "INSUFFICIENT_EVIDENCE"
gates = load_gates()
for n in names:
    r = dev_returns[n]
    res = dev_runs[n]
    enough = len(r) >= v["sufficiency"]["min_daily_observations"]
    years = sorted({d[:4] for d in r.index if "2017" <= d[:4] <= "2022"})
    wf = {y: metrics.sharpe(r[[d for d in r.index if d[:4] == y]]) for y in years}
    horizon = v["purged_kfold"]["label_horizon_sessions"][PRE["strategies"][n]["rebalance"]]
    folds = purged_kfold(len(r), v["purged_kfold"]["k"], label_horizon=horizon, embargo=v["purged_kfold"]["embargo_sessions"])
    fold_sr = [metrics.sharpe(r.iloc[list(f.test)]) for f in folds]
    paths = cpcv(len(r), v["cpcv"]["n_groups"], v["cpcv"]["k_test"], label_horizon=horizon,
                 embargo=v["purged_kfold"]["embargo_sessions"])
    cpcv_sr = [metrics.sharpe(r.iloc[list(p.test)]) for p in paths]
    lo, hi = bootstrap_sharpe_ci(r.to_numpy(), n_samples=v["bootstrap"]["samples"], block=v["bootstrap"]["block_sessions"],
                                 seed=v["bootstrap"]["seed"], alpha=v["bootstrap"]["alpha"])
    trips = metrics.round_trips(fills_of(res))
    mc = (monte_carlo_trades([t.pnl for t in trips], capital=B["capital_usd"], n_samples=v["monte_carlo"]["samples"],
                             seed=v["monte_carlo"]["seed"]) if len(trips) >= v["sufficiency"]["min_round_trips"]
          else "INSUFFICIENT_EVIDENCE")
    psr = probabilistic_sharpe(r.to_numpy()) if enough else "INSUFFICIENT_EVIDENCE"
    dsr = deflated_sharpe(r.to_numpy(), n_trials, sr_var) if enough and sr_var is not None else "INSUFFICIENT_EVIDENCE"
    dev = results[n]["development"]
    evidence = {
        "n_trades": len(trips), "oos_sharpe_annual": dev["sharpe"],
        "deflated_sharpe": dsr if isinstance(dsr, float) else 0.0,
        "pbo": pbo_value if isinstance(pbo_value, float) else 1.0,
        "max_drawdown": dev["max_drawdown"],
        "walk_forward_positive_share": float(np.mean([x > 0 for x in wf.values()])) if wf else 0.0,
        "bootstrap_sharpe_lower": lo,
    }
    gate = evaluate_gates(evidence, gates)
    validation[n] = {
        "observations": len(r), "walk_forward_sharpe_by_year": wf,
        "walk_forward_positive_share": evidence["walk_forward_positive_share"],
        "purged_kfold_sharpe": fold_sr, "cpcv_sharpe": {"paths": len(cpcv_sr), "mean": float(np.mean(cpcv_sr)),
                                                         "min": float(np.min(cpcv_sr)), "max": float(np.max(cpcv_sr)),
                                                         "share_positive": float(np.mean([x > 0 for x in cpcv_sr]))},
        "psr": psr, "dsr": dsr, "dsr_n_trials": n_trials, "dsr_sr_variance": sr_var, "pbo_family": pbo_value,
        "bootstrap_sharpe_ci_per_period": [lo, hi], "monte_carlo": mc,
        "gates": gate.model_dump(),
    }
    registry.record(family=FAMILY, spec=CFGS[n]["spec"], stage="validation", window=tuple(DEV),
                    metrics={"gates_passed": gate.passed, **{k: float(x) for k, x in evidence.items()}})

log("STEP 7 cost stress (development only)")
for n in names:
    cost_stress[n] = {}
    for mult in PRE["cost_stress"]["multipliers"]:
        res = dev_runs[n] if mult == 1.0 else run(n, *DEV, costs=cost_model(mult))
        s = res.metrics
        cost_stress[n][str(mult)] = {"sharpe": s["sharpe"], "cagr": s["cagr"], "total_return": s["total_return"],
                                     "transaction_costs": s["transaction_costs"], "max_drawdown": s["max_drawdown"]}
    base, two = cost_stress[n]["1.0"]["sharpe"], cost_stress[n]["2.0"]["sharpe"]
    cost_stress[n]["fragile"] = bool(two <= 0 or (base > 0 and two < 0.5 * base))

log("STEP 6b locked holdout (single evaluation per strategy)")
crit = PRE["holdout_pass_criterion"]
holdout_results = {}
prior = {r["spec_hash"]: r["metrics"] for r in registry.records() if r["stage"] == "locked_holdout"}
for n in names:
    h = spec_hash(CFGS[n]["spec"])
    if h in prior:
        holdout_results[n] = {**prior[h], "source": "registry (already evaluated once)"}
        continue

    def _run(start, end, _n=n):
        res = run(_n, start, end)
        audit["lookahead_violations"] += lookahead_violations(res)
        s = summarize(res)
        return {k: s[k] for k in ("total_return", "cagr", "sharpe", "sortino", "max_drawdown", "round_trips",
                                  "transaction_costs", "benchmark_total_return", "final_equity")}

    holdout_results[n] = holdout.evaluate(
        CFGS[n]["spec"], _run,
        passed=lambda m: m["sharpe"] >= crit["annualized_sharpe_min"] and m["max_drawdown"] <= crit["max_drawdown_max"]
        and m["total_return"] > crit["total_return_min"])

log("STEP 9 EUR 200 (development only)")
sa = SmallAccountConfig()
sa_costs = {"commission_bps": 0.0, "commission_min": 1.0, "half_spread_bps": 10.0, "impact_coef": 0.1,
            "max_participation": 0.05, "adv_lookback": 20, "vol_lookback": 20}
for n in names:
    trips = metrics.round_trips(fills_of(dev_runs[n]))
    edge_bps = (float(np.mean([t.pnl / (abs(t.quantity) * t.entry_price) for t in trips]) * 1e4) if trips else 0.0)
    small[n] = {"development_round_trip_expectancy_bps": edge_bps}
    for variant in ("raw", "economics_filtered"):
        filt = EconomicsFilter(sa, default_edge_bps=max(edge_bps, 0.0)) if variant == "economics_filtered" else None
        res = run(n, *DEV, capital=sa.capital, costs=CostModel(**sa_costs),
                  instruments=small_account_universe([], sa), order_filter=filt)
        audit["lookahead_violations"] += lookahead_violations(res)
        rets = res.equity.pct_change().dropna()
        invested = res.exposure * res.equity
        small[n][variant] = {
            "final_value": float(res.equity.iloc[-1]), "total_return": res.metrics["total_return"],
            "max_drawdown": res.metrics["max_drawdown"], "sharpe": res.metrics["sharpe"],
            "fills": len(res.trades), "orders_rejected_by_execution": len(res.rejected),
            "rejection_reasons": sorted({str(r["reason"]) for r in res.rejected})[:5],
            "orders_dropped_as_uneconomic": len(filt.dropped) if filt else 0,
            "average_invested_capital": float(invested.mean()),
            "average_cash_share": float(1 - res.exposure.mean()),
            "transaction_costs": res.metrics["transaction_costs"],
            "costs_pct_of_average_equity": float(res.metrics["transaction_costs"] / res.equity.mean()),
            "probability_of_ruin_1y": probability_of_ruin(float(rets.mean()), float(rets.std(ddof=1)), PPY,
                                                          ruin_level=sa.ruin_level, n_paths=5000, seed=1),
            "max_positions_held": int(max((len(d["risk"]["weights"]) for d in res.decisions), default=0)),
        }

log("STEP 10 classification")
status = {}
for n in names:
    dev = results[n]["development"]
    val = validation[n]
    insufficient = any(val[k] == "INSUFFICIENT_EVIDENCE" for k in ("psr", "dsr", "monte_carlo")) or \
        dev["round_trips"] < PRE["validation"]["sufficiency"]["min_round_trips"]
    hold_ok = bool(holdout_results[n].get("passed"))
    if dev["sharpe"] <= 0 or not hold_ok:
        status[n] = "REJECT"
    elif insufficient or not val["gates"]["passed"]:
        status[n] = "INSUFFICIENT_EVIDENCE"
    elif cost_stress[n]["fragile"]:
        status[n] = "INSUFFICIENT_EVIDENCE"
    else:
        status[n] = "EVIDENCE_SUPPORTS_FURTHER_RESEARCH"
    registry.record(family=FAMILY, spec=CFGS[n]["spec"], stage="classification", metrics={"status": status[n]})

ensemble = {"run": False, "reason": "fewer than two strategies are EVIDENCE_SUPPORTS_FURTHER_RESEARCH"}
if sum(s == "EVIDENCE_SUPPORTS_FURTHER_RESEARCH" for s in status.values()) >= 2:
    ensemble = {"run": False, "reason": "eligible — to be run as a separate, pre-registered experiment"}

audit.update({
    "financial_datasets_clients_constructed": FD_CALLS["constructed"],
    "tiingo_requests": tiingo.requests,
    "live_broker_modules_loaded": [m for m in sys.modules if m.endswith("brokers.adapters")],
    "registry_verified": (registry.verify() is None),
    "registry_trials_in_family": registry.n_trials(FAMILY),
    "config_hashes_match_preregistration": all(strategy(n).config_hash() == PRE["strategies"][n]["config_hash"]
                                               for n in names),
    "preregistration_hash": PRE["preregistration_hash"],
    "holdout_records": [{"spec_hash": r["spec_hash"], "recorded_at": r["recorded_at"]}
                        for r in registry.records() if r["stage"] == "locked_holdout"],
})

for name, obj in (("results", {"strategies": results, "holdout": holdout_results, "status": status,
                                "ensemble": ensemble}),
                  ("validation", validation), ("cost_stress", cost_stress), ("small_account_200", small),
                  ("audit", audit)):
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
log("done")
print(json.dumps(status, indent=1))
