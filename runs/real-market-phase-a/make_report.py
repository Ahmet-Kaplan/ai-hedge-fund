"""Render report.md and report.html from the Phase A JSON outputs (no computation of new results)."""
from __future__ import annotations

import html
import json
from pathlib import Path

OUT = Path("runs/real-market-phase-a")
J = {n: json.loads((OUT / f"{n}.json").read_text()) for n in
     ("preregistration", "data_inventory", "strategy_configs", "results", "validation", "cost_stress",
      "small_account_200", "audit", "posthoc_beta")}
R, V, C, S, A, P, PB = (J["results"], J["validation"], J["cost_stress"], J["small_account_200"], J["audit"],
                        J["preregistration"], J["posthoc_beta"])
names = list(J["strategy_configs"])


def pct(x, d=1):
    return "—" if x is None or isinstance(x, str) else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return x if isinstance(x, str) else ("—" if x is None else f"{x:.{d}f}")


def money(x):
    return f"{x:,.2f}"


lines = []
w = lines.append
w("# Phase A — real-market evidence for four frozen reference strategies\n")
w(f"Pre-registration hash `{P['preregistration_hash'][:16]}…` (committed before any backtest). "
  "Research and simulation only; no live trading. Figures are simulated, net of modelled costs, and "
  "not a promise of future returns.\n")

w("## Summary table (development period unless stated)\n")
w("| Strategy | CAGR | Sharpe | Max DD | Trades | Costs | DSR | PBO | €200 Final (raw / filtered) | Holdout | Status |")
w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
for n in names:
    d, v, s = R["strategies"][n]["development"], V[n], S[n]
    raw = s["raw"]["final_value"]
    raw_txt = "0.00 (ruined)" if raw <= 0 else money(raw)
    h = R["holdout"][n]
    w(f"| {n} | {pct(d['cagr'])} | {num(d['sharpe'])} | {pct(d['max_drawdown'])} | {d['round_trips']} | "
      f"${d['transaction_costs']:,.0f} | {num(v['dsr'])} | {num(v['pbo_family'])} | {raw_txt} / "
      f"{money(s['economics_filtered']['final_value'])} | {'PASS' if h.get('passed') else 'FAIL'} | "
      f"**{R['status'][n]}** |")
spy = PB["SPY_total_return_dev"]
w(f"\nBenchmark SPY total return, development: CAGR {pct(spy['cagr'])}, Sharpe {num(spy['sharpe'])}, "
  f"max DD {pct(spy['max_drawdown'])}. Trades = closed round trips. DSR uses n_trials = "
  f"{V[names[0]]['dsr_n_trials']}; PBO is computed once for the family of four strategies.\n")

w("## Setup\n")
b = P["backtest"]
w(f"- Universe: {P['data']['universe']}.")
w(f"- Development {P['periods']['development'][0]} → {P['periods']['development'][1]}; embargo "
  f"{P['periods']['embargo_calendar_days']} days; locked holdout {P['periods']['locked_holdout'][0]} → "
  f"{P['periods']['locked_holdout'][1]} (evaluated once per strategy).")
w(f"- Benchmark: {P['data']['benchmark']}. Risk-free rate: {P['data']['risk_free_rate']}.")
w(f"- Execution: {b['execution']}. Costs: commission {b['costs_baseline']['commission_bps']} bp, half-spread "
  f"{b['costs_baseline']['half_spread_bps']} bp, square-root impact coef {b['costs_baseline']['impact_coef']}, "
  f"max {b['costs_baseline']['max_participation']:.0%} of ADV.")
w(f"- Portfolio: long-only, inverse-volatility, gross 100%, max 20% per name; operational breakers off "
  f"({b['risk_rationale']}).")
w("- Strategy configs: code defaults, frozen and hashed: " + ", ".join(
    f"{n} `{J['strategy_configs'][n]['config_hash']}` ({P['strategies'][n]['rebalance']})" for n in names) + ".\n")

w("## Data inventory\n")
inv = J["data_inventory"]
w(f"- {len(inv['instruments'])} instruments (SPY + {len(inv['universe']['union'])} universe names), "
  f"{inv['benchmark_sessions']} sessions {inv['benchmark_first']} → {inv['benchmark_last']}; "
  f"Tiingo requests during inventory: {inv['tiingo_requests']}.")
w("- No missing sessions, closes or volumes during any name's membership; every name is 100% complete.")
w("- Dividends present for all payers; BRK.B, AMZN, TSLA paid none. Splits covered "
  "(AAPL, GOOGL, AMZN, NVDA×2, TSLA×2, AVGO, T, GE). GE's 2023/2024 spin-offs are encoded by the vendor as "
  "split factors 1.281 and 1.253 (value-preserving restatement, not true splits).")
w("- Delisting coverage: no universe member stopped trading in the window, so delisting handling was not "
  "exercised by this universe (survivorship-aware membership, but megacaps did not delist).\n")

w("## Development results\n")
w("| Strategy | Total | CAGR | Vol | Sharpe | Sortino | Calmar | Max DD | Hit | Payoff | PF | Expectancy | "
  "Turnover/yr | Costs | Slippage | Hold (sessions) | Gross | Net |")
w("|---|" + "---:|" * 17)
for n in names:
    d = R["strategies"][n]["development"]
    w(f"| {n} | {pct(d['total_return'])} | {pct(d['cagr'])} | {pct(d['annualized_volatility'])} | "
      f"{num(d['sharpe'])} | {num(d['sortino'])} | {num(d['calmar'])} | {pct(d['max_drawdown'])} | "
      f"{pct(d['hit_rate']) if not isinstance(d['hit_rate'], str) else d['hit_rate']} | {num(d['payoff_ratio'])} | "
      f"{num(d['profit_factor'])} | {num(d['expectancy'], 0)} | {num(d['turnover'])} | "
      f"${d['transaction_costs']:,.0f} | ${d['slippage']:,.0f} | {num(d['average_holding_sessions'], 0)} | "
      f"{pct(d['gross_exposure_mean'], 0)} | {pct(d['net_exposure_mean'], 0)} |")
w(f"\nSPY total return over the same window: {pct(R['strategies'][names[0]]['development']['benchmark_total_return'])}.\n")

w("### Exposure to the market (post-hoc, descriptive — not used for classification)\n")
w("| Strategy | Beta vs SPY TR | Correlation | Alpha / yr | Alpha t-stat |")
w("|---|---:|---:|---:|---:|")
for n in names:
    p = PB[n]
    w(f"| {n} | {num(p['beta_vs_spy_tr'])} | {num(p['correlation_vs_spy_tr'])} | {pct(p['alpha_annualized'], 2)} | "
      f"{num(p['alpha_tstat'])} |")
w("\nThe three strategies with positive Sharpe are mostly long-only exposure to the same megacaps as the "
  "benchmark (beta 0.85–0.91); their alpha is indistinguishable from zero.\n")

w("### Per year (net return vs SPY total return)\n")
years = sorted(R["strategies"][names[0]]["per_year"])
w("| Year | " + " | ".join(names) + " | SPY TR |")
w("|---|" + "---:|" * (len(names) + 1))
for y in years:
    row = [pct(R["strategies"][n]["per_year"][y]["return"]) for n in names]
    w(f"| {y} | " + " | ".join(row) + f" | {pct(R['strategies'][names[0]]['per_year'][y]['benchmark_tr'])} |")

w("\n### Per regime (SimpleRegime at each decision; annualized mean daily return)\n")
for n in names:
    reg = R["strategies"][n]["per_regime"]
    w(f"- **{n}**: " + "; ".join(f"{k} ({v['days']}d): {pct(v['annualized_return'])}, Sharpe {num(v['sharpe'])}"
                                  for k, v in sorted(reg.items(), key=lambda kv: -kv[1]["days"])[:5]))

w("\n### Largest drawdowns\n")
for n in names:
    dd = R["strategies"][n]["drawdown_periods"]
    w(f"- **{n}**: " + "; ".join(f"{pct(x['depth'])} {x['start']}→{x['trough']} (recovered {x['recovered'] or 'not by end'})"
                                  for x in dd))

w("\n### Contribution by instrument (P&L in $, development)\n")
for n in names:
    c = R["strategies"][n]["contribution_by_instrument"]
    items = list(c.items())
    w(f"- **{n}**: top " + ", ".join(f"{k} {v:+,.0f}" for k, v in items[:4]) + "; bottom " +
      ", ".join(f"{k} {v:+,.0f}" for k, v in items[-3:]))

w("\n## Validation (development only)\n")
w("| Strategy | WF positive years | Purged k-fold Sharpe (min / mean) | CPCV share positive | PSR | DSR | PBO | "
  "Bootstrap Sharpe 95% CI (daily) | MC max DD p95 | Gates |")
w("|---|---:|---:|---:|---:|---:|---:|---|---:|---|")
for n in names:
    v = V[n]
    pk = v["purged_kfold_sharpe"]
    mc = v["monte_carlo"]
    failed = [k for k, ok in v["gates"]["checks"].items() if not ok]
    w(f"| {n} | {v['walk_forward_positive_share']:.0%} | {min(pk):.2f} / {sum(pk) / len(pk):.2f} | "
      f"{v['cpcv_sharpe']['share_positive']:.0%} | {num(v['psr'])} | {num(v['dsr'])} | {num(v['pbo_family'])} | "
      f"[{v['bootstrap_sharpe_ci_per_period'][0]:.4f}, {v['bootstrap_sharpe_ci_per_period'][1]:.4f}] | "
      f"{pct(mc['max_drawdown_p95']) if isinstance(mc, dict) else mc} | "
      f"{'PASS' if v['gates']['passed'] else 'FAIL: ' + ', '.join(failed)} |")
w("\nGates: `configs/validation-gates.yaml` (hash " + P["validation"]["gates_hash"] + "). No gate was changed.\n")

w("## Locked holdout (2023-02-01 → 2026-06-30, evaluated once)\n")
w("| Strategy | Total | CAGR | Sharpe | Max DD | Round trips | Costs | SPY TR | Pass |")
w("|---|---:|---:|---:|---:|---:|---:|---:|---|")
for n in names:
    h = R["holdout"][n]
    w(f"| {n} | {pct(h['total_return'])} | {pct(h['cagr'])} | {num(h['sharpe'])} | {pct(h['max_drawdown'])} | "
      f"{h['round_trips']} | ${h['transaction_costs']:,.0f} | {pct(h['benchmark_total_return'])} | "
      f"{'PASS' if h['passed'] else 'FAIL'} |")
w(f"\nCriterion (pre-registered): Sharpe ≥ {P['holdout_pass_criterion']['annualized_sharpe_min']}, max DD ≤ "
  f"{P['holdout_pass_criterion']['max_drawdown_max']:.0%}, total return > 0. It does not require beating the "
  "benchmark; none of the strategies clearly did.\n")

w("## Cost stress (development)\n")
w("| Strategy | Sharpe 1× | 1.5× | 2× | 3× | Total return 1× → 3× | Fragile |")
w("|---|---:|---:|---:|---:|---|---|")
for n in names:
    c = C[n]
    w(f"| {n} | {num(c['1.0']['sharpe'])} | {num(c['1.5']['sharpe'])} | {num(c['2.0']['sharpe'])} | "
      f"{num(c['3.0']['sharpe'])} | {pct(c['1.0']['total_return'])} → {pct(c['3.0']['total_return'])} | "
      f"{'yes' if c['fragile'] else 'no'} |")

w("\n## €200 account (development)\n")
w("| Strategy | Variant | Final | Max DD | Fills | Rejected (exec) | Dropped uneconomic | Avg invested | "
  "Cash share | Costs | Costs / avg equity | P(ruin, 1y) |")
w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for n in names:
    for var in ("raw", "economics_filtered"):
        x = S[n][var]
        fin = "0.00 (ruined)" if x["final_value"] <= 0 else money(x["final_value"])
        w(f"| {n} | {var} | {fin} | {pct(min(x['max_drawdown'], 1.0))} | {x['fills']} | "
          f"{x['orders_rejected_by_execution']} | {x['orders_dropped_as_uneconomic']} | {money(x['average_invested_capital'])} | "
          f"{pct(x['average_cash_share'], 0)} | {money(x['transaction_costs'])} | {pct(x['costs_pct_of_average_equity'], 0)} | "
          f"{pct(x['probability_of_ruin_1y'])} |")
w("\nThe €1 minimum commission per order dominates: rebalancing a 200-unit account into several names "
  "means orders of a few to a few dozen units, where €1 is 2–50% of the order. Unfiltered, every strategy "
  "lost the whole account. With the economics filter, orders whose estimated edge did not cover costs were "
  "skipped; breakout and meanrev then never traded, tsmom ended at 230 (+15% over 6.5 years) and "
  "xsmom at 193. The filter's edge estimate (development round-trip expectancy, bp) includes market beta, so "
  "it overstates true edge. Results were simulated at €200 directly, not scaled from $100k.\n")
w("Simulator finding: the raw small-account runs ended slightly below zero because commissions were still "
  "charged on tiny sells when cash was exhausted. The report treats these as ruined (0); the cash floor should "
  "be enforced in the ledger before any small-account paper trading.\n")

w("## Classification (pre-registered rules, unchanged)\n")
for n in names:
    v, h = V[n], R["holdout"][n]
    w(f"- **{n} — {R['status'][n]}**: development Sharpe {num(R['strategies'][n]['development']['sharpe'])}; "
      f"gates {'pass' if v['gates']['passed'] else 'fail (' + ', '.join(k for k, ok in v['gates']['checks'].items() if not ok) + ')'}; "
      f"holdout {'pass' if h['passed'] else 'fail'}; fragile to costs: {'yes' if C[n]['fragile'] else 'no'}.")
w(f"\nEnsemble: not run — {R['ensemble']['reason']}.\n")

w("## Audit\n")
w(f"- Financial Datasets clients constructed: {A['financial_datasets_clients_constructed']}; Tiingo requests: "
  f"{A['tiingo_requests']} (offline store); live broker modules loaded: {len(A['live_broker_modules_loaded'])}.")
w(f"- Look-ahead violations (fill session not after decision session): {A['lookahead_violations']}.")
w(f"- Pre-registration committed at 2026-10-01T09:20:11Z (commit aa05bc1); holdout evaluated at "
  f"{', '.join(r['recorded_at'][11:19] for r in A['holdout_records'])} UTC, once per strategy.")
w(f"- Config hashes match the pre-registration: {A['config_hashes_match_preregistration']}; no parameter was "
  "changed before or after the holdout.")
w(f"- Experiment registry verified (hash chain): {A['registry_verified']}; distinct trials in family: "
  f"{A['registry_trials_in_family']}.")
w("- No secrets printed or stored; `main` unchanged.\n")

w("## Caveats\n")
w("- Small universe (10 megacaps at a time); results do not generalise to broader or less liquid markets.")
w("- Rebalance cadence per strategy was a pre-registered choice; other cadences were not tried (by design).")
w("- Risk-free rate set to 0; Sharpe ratios are slightly overstated relative to excess-of-cash Sharpe.")
w("- PBO is a family-level statistic over these four strategies, not a per-strategy probability.")
w("- The holdout criterion does not include a benchmark comparison; holdout passes therefore do not imply "
  "outperformance.")

md = "\n".join(lines) + "\n"
(OUT / "report.md").write_text(md)


def md_to_html(text: str) -> str:
    out, in_table, in_list = [], False, False
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-:") for c in cells):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def inline(s: str) -> str:
    import re
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase A real-market research</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#d2d2d7;--head:#f5f5f7}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141416;--fg:#f2f2f2;--muted:#a1a1a6;--line:#3a3a3c;--head:#1f1f22}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px 16px;max-width:1200px;margin:auto}}
table{{border-collapse:collapse;display:block;overflow-x:auto;margin:12px 0;font-size:13px}}
th,td{{border:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}}
th{{background:var(--head)}} td:first-child,th:first-child{{text-align:left}}
code{{font-size:12px}} h2{{margin-top:32px;border-bottom:1px solid var(--line)}} p,li{{color:var(--fg)}}
</style></head><body>
{md_to_html(md)}
</body></html>"""
(OUT / "report.html").write_text(page)
print("wrote report.md, report.html")
