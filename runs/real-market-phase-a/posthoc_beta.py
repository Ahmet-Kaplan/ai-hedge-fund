from __future__ import annotations
"""Post-hoc, descriptive only: beta/alpha vs SPY TR on the development window.\nNot pre-registered; not used for classification; the holdout is not touched."""
"""STEPS 6-9, 11, 13 — real-data backtests, validation, cost stress, EUR 200, audit.

Follows preregistration.json exactly. Offline: Tiingo reads come from the
local store (offline=True). The locked holdout is evaluated once per strategy;
on a re-run its recorded result is read back from the registry, never re-run.
"""

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



OUTP = OUT / "posthoc_beta.json"
spy_res = None
out = {}
for n in CFGS:
    res = run(n, *DEV)
    r = res.equity.pct_change().dropna()
    b = res.benchmark_total_return.pct_change().dropna().reindex(r.index)
    beta = float(np.cov(r, b, ddof=1)[0, 1] / np.var(b, ddof=1))
    alpha_daily = float(r.mean() - beta * b.mean())
    resid = r - beta * b
    out[n] = {"beta_vs_spy_tr": beta, "alpha_annualized": alpha_daily * PPY,
              "correlation_vs_spy_tr": float(np.corrcoef(r, b)[0, 1]),
              "alpha_tstat": float(resid.mean() / (resid.std(ddof=1) / np.sqrt(len(resid))))}
    spy_res = b
out["SPY_total_return_dev"] = {"sharpe": metrics.sharpe(spy_res), "max_drawdown": metrics.max_drawdown((1 + spy_res).cumprod()),
                               "cagr": metrics.cagr((1 + spy_res).cumprod())}
out["note"] = "post-hoc descriptive statistics; not pre-registered and not used for classification"
OUTP.write_text(json.dumps(out, indent=1, sort_keys=True))
print(json.dumps(out, indent=1))
