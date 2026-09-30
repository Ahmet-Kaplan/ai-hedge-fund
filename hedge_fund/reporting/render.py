"""Render decision and backtest reports as Markdown or self-contained HTML.

The HTML page has no external dependencies (inline CSS, inline SVG chart,
a few lines of inline script for the chart's hover readout), so a report
opens offline and can be attached or archived as a single file. Every piece
of recorded text — tickers, analyst reasoning — is escaped: LLM output is
data, never markup.
"""

from __future__ import annotations

import json
from html import escape

from hedge_fund.reporting.decisions import BacktestReport, DecisionReport, TickerDecision

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def to_markdown(report: DecisionReport | BacktestReport) -> str:
    if isinstance(report, BacktestReport):
        return _backtest_md(report)
    return "\n".join(_decision_md(report, heading="#"))


def _decision_md(r: DecisionReport, heading: str) -> list[str]:
    title = "Proposal (pending — not executed)" if r.status == "pending" else f"Rebalance executed {r.execution_as_of}"
    lines = [f"{heading} {r.fund} — {title}", "",
             f"Assessment cutoff **{r.as_of}** · analysts: {', '.join(r.analysts)} · max position {r.max_position_pct:.0%} · "
             f"{r.rebalance} rebalance vs {r.benchmark}", ""]
    if r.pending_reason:
        lines += [f"> {r.pending_reason}", ""]
    if r.equity is not None:
        lines += [f"NAV **${r.equity:,.2f}** · cash **${r.cash:,.2f}** ({r.cash_weight:.1%}) · invested {r.invested_weight:.1%}", ""]
    else:
        lines += [f"Proposed: invested {r.invested_weight:.1%} · cash {r.cash_weight:.1%} (of ${r.capital:,.0f} capital)", ""]
    if r.gross_clamp:
        lines += [f"Risk: {r.gross_clamp}", ""]

    lines += [f"{heading}# Selected portfolio", ""]
    if r.selected:
        lines += ["| Ticker | Action | Weight | Shares | Price | Signal | Confidence | Notes |", "|---|---|---:|---:|---:|---|---:|---|"]
        for d in r.selected:
            weight = d.weight if d.weight is not None else d.target_weight
            lines.append(f"| {d.ticker} | {_action_label(d)} | {weight:.2%} | {_n(d.shares_after)} | {_money(d.price)} | "
                         f"{_signals(d)} | {_confidences(d)} | {_md(' · '.join(d.notes))} |")
    else:
        lines.append("_No positions — the whole book is in cash._")
    lines += ["", f"**Cash:** {r.cash_weight:.1%}" + (f" (${r.cash:,.2f})" if r.cash is not None else ""), ""]

    lines += [f"{heading}# Rejected", ""]
    if r.rejected:
        lines += ["| Ticker | Action | Reason |", "|---|---|---|"]
        for d in r.rejected:
            lines.append(f"| {d.ticker} | {_action_label(d)} | {_md(d.rejection_reason or '')} |")
    else:
        lines.append("_None._")

    lines += ["", f"{heading}# Analyst reasoning", ""]
    for d in r.decisions:
        for v in d.views:
            conf = f", confidence {v.confidence:.0f}" if v.confidence is not None else ""
            lines += [f"- **{d.ticker}** — {v.analyst}: {v.signal}{conf}. {_md(v.reasoning or '')}"]
    return lines


def _backtest_md(b: BacktestReport) -> str:
    m = b.metrics
    lines = [f"# {b.fund} — backtest {b.start} → {b.end}", "",
             f"{b.rebalance} rebalance vs {b.benchmark} · ${b.capital:,.0f} starting capital · universe: {', '.join(b.universe)}", "",
             "| Metric | Fund | " + b.benchmark + " |", "|---|---:|---:|",
             f"| Total return | {m.total_return_pct:+.2%} | {m.benchmark_return_pct:+.2%} |",
             f"| Annualized return | {m.annualized_return_pct:+.2%} | — |",
             f"| Max drawdown | {m.max_drawdown_pct:.2%} | — |",
             f"| Sharpe (rf = 0) | {m.sharpe_ratio:.2f} | — |",
             f"| Excess return | {m.excess_return_pct:+.2%} | |",
             f"| Ending NAV | ${b.nav[-1]:,.2f} | ${b.benchmark_nav[-1]:,.2f} |", "",
             f"{m.n_cycles} executed rebalances · {m.n_orders} orders · {m.n_pending} pending proposal(s)", "",
             "## Rebalance history", "", "| Decided | Executed | NAV | Cash | Holdings | Bought | Sold |", "|---|---|---:|---:|---|---|---|"]
    for s in b.rebalances:
        holdings = ", ".join(f"{h.ticker} {h.weight:.1%}" for h in s.holdings) or "—"
        lines.append(f"| {s.as_of} | {s.executed or '—'} | ${s.nav:,.2f} | {s.cash_weight:.1%} | {holdings} | {', '.join(s.buys) or '—'} | {', '.join(s.sells) or '—'} |")
    if b.latest:
        lines += ["", *_decision_md(b.latest, heading="##")]
    if b.next_proposal:
        lines += ["", *_decision_md(b.next_proposal, heading="##")]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root{color-scheme:light;--surface-0:#f4f3f0;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#76756f;
--rule:#e4e2dc;--series-1:#2a78d6;--series-2:#eb6834;--buy:#0f7a3d;--sell:#b3261e;--chip:#ecebe6}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;
--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#9a998f;--rule:#2e2e2b;--series-1:#3987e5;--series-2:#d95926;--buy:#5cc98a;--sell:#f08a82;--chip:#262624}}
:root[data-theme="dark"]{color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#9a998f;
--rule:#2e2e2b;--series-1:#3987e5;--series-2:#d95926;--buy:#5cc98a;--sell:#f08a82;--chip:#262624}
*{box-sizing:border-box}body{margin:0;background:var(--surface-0);color:var(--text-primary);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px}
.sub{color:var(--text-secondary);margin:0 0 16px}.card{background:var(--surface-1);border:1px solid var(--rule);border-radius:10px;padding:16px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.tile .k{color:var(--text-secondary);font-size:13px}.tile .v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .s{color:var(--text-muted);font-size:12px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule);vertical-align:top}th{color:var(--text-secondary);font-weight:600;font-size:13px;white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.chip{display:inline-block;padding:1px 8px;border-radius:999px;background:var(--chip);font-size:12px;font-weight:600;white-space:nowrap}
.BUY{color:var(--buy)}.SELL{color:var(--sell)}.HOLD,.NONE{color:var(--text-secondary)}.muted{color:var(--text-muted)}.why{color:var(--text-secondary)}
details{margin:6px 0}summary{cursor:pointer;color:var(--text-secondary)}.reason{white-space:pre-wrap;margin:4px 0 8px}
.legend{display:flex;gap:16px;font-size:13px;color:var(--text-secondary);margin-bottom:8px}.legend i{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:6px}
.chart{position:relative;min-width:820px}.chart svg{display:block;width:100%;height:auto}.tiles .card{margin:0}.tip{position:absolute;pointer-events:none;background:var(--surface-1);border:1px solid var(--rule);
border-radius:8px;padding:6px 10px;font-size:13px;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;white-space:nowrap}.tip b{font-variant-numeric:tabular-nums}
.banner{border-left:3px solid var(--text-muted);padding:4px 10px;color:var(--text-secondary)}
"""


def to_html(report: DecisionReport | BacktestReport) -> str:
    if isinstance(report, BacktestReport):
        title = f"{report.fund} backtest"
        body = _backtest_html(report)
    else:
        title = f"{report.fund} decisions"
        body = _decision_html(report, top=True)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{escape(title)}</title><style>{_CSS}</style></head><body><main>{body}</main></body></html>\n")


def _decision_html(r: DecisionReport, top: bool) -> str:
    h = "h1" if top else "h2"
    when = "Pending proposal — not executed" if r.status == "pending" else f"Rebalance executed {escape(r.execution_as_of or '')}"
    out = [f"<{h}>{escape(r.fund)} · {when}</{h}>",
           f'<p class="sub">Assessment cutoff {escape(r.as_of)} · analysts: {escape(", ".join(r.analysts))} · max position {r.max_position_pct:.0%} · '
           f"{escape(r.rebalance)} rebalance vs {escape(r.benchmark)}</p>"]
    if r.pending_reason:
        out.append(f'<p class="banner">{escape(r.pending_reason)}</p>')
    tiles = []
    if r.equity is not None:
        tiles += [("NAV", f"${r.equity:,.0f}", ""), ("Cash", f"{r.cash_weight:.1%}", f"${r.cash:,.0f}")]
    else:
        tiles += [("Capital", f"${r.capital:,.0f}", ""), ("Cash (proposed)", f"{r.cash_weight:.1%}", "")]
    tiles += [("Invested", f"{r.invested_weight:.1%}", ""), ("Selected", str(len(r.selected)), f"of {len(r.decisions)} names")]
    out.append(_tiles(tiles))
    if r.gross_clamp:
        out.append(f'<p class="banner">Risk: {escape(r.gross_clamp)}</p>')

    rows = "".join(
        f"<tr><td><b>{escape(d.ticker)}</b></td><td>{_chip(d)}</td><td class=num>{(d.weight if d.weight is not None else d.target_weight):.2%}</td>"
        f"<td class=num>{_n(d.shares_after)}</td><td class=num>{_money(d.price)}</td><td>{escape(_signals(d))}</td>"
        f"<td class=num>{escape(_confidences(d))}</td><td class=why>{escape(' · '.join(d.notes))}</td></tr>" for d in r.selected)
    cash = f'<tr><td><b>Cash</b></td><td></td><td class=num>{r.cash_weight:.2%}</td><td></td><td class=num>{_money(r.cash)}</td><td colspan=3></td></tr>'
    out.append('<h2>Selected portfolio</h2><div class="card scroll"><table><thead><tr><th>Ticker</th><th>Action</th><th class=num>Weight</th>'
               '<th class=num>Shares</th><th class=num>Price</th><th>Signal</th><th class=num>Confidence</th><th>Notes</th></tr></thead>'
               f"<tbody>{rows}{cash}</tbody></table></div>")

    rej = "".join(f"<tr><td><b>{escape(d.ticker)}</b></td><td>{_chip(d)}</td><td class=why>{escape(d.rejection_reason or '')}</td></tr>" for d in r.rejected)
    out.append('<h2>Rejected</h2><div class="card scroll">' + (
        f"<table><thead><tr><th>Ticker</th><th>Action</th><th>Reason</th></tr></thead><tbody>{rej}</tbody></table>" if rej else '<p class="muted">None.</p>') + "</div>")

    reasons = []
    for d in r.decisions:
        for v in d.views:
            conf = f" · confidence {v.confidence:.0f}" if v.confidence is not None else ""
            reasons.append(f"<details><summary><b>{escape(d.ticker)}</b> — {escape(v.analyst)}: {escape(v.signal)}{conf}</summary>"
                           f'<p class="reason">{escape(v.reasoning or "")}</p></details>')
    out.append('<h2>Analyst reasoning</h2><div class="card">' + ("".join(reasons) or '<p class="muted">No analyst views recorded.</p>') + "</div>")
    return "".join(out)


def _backtest_html(b: BacktestReport) -> str:
    m = b.metrics
    out = [f"<h1>{escape(b.fund)} · backtest</h1>",
           f'<p class="sub">{escape(b.start)} → {escape(b.end)} · {escape(b.rebalance)} rebalance vs {escape(b.benchmark)} · '
           f"${b.capital:,.0f} starting capital · {len(b.universe)} names</p>",
           _tiles([("Total return", f"{m.total_return_pct:+.2%}", f"{escape(b.benchmark)} {m.benchmark_return_pct:+.2%}"),
                   ("Annualized", f"{m.annualized_return_pct:+.2%}", ""), ("Max drawdown", f"{m.max_drawdown_pct:.2%}", ""),
                   ("Sharpe", f"{m.sharpe_ratio:.2f}", "daily, rf = 0"), ("Excess return", f"{m.excess_return_pct:+.2%}", f"vs {escape(b.benchmark)}"),
                   ("Rebalances", str(m.n_cycles), f"{m.n_orders} orders")]),
           "<h2>Equity curve</h2>", f'<div class="card">{_chart(b)}</div>']

    rows = "".join(
        f"<tr><td>{escape(s.as_of)}</td><td>{escape(s.executed or '—')}</td><td class=num>${s.nav:,.0f}</td><td class=num>{s.cash_weight:.1%}</td>"
        f"<td>{escape(', '.join(f'{h.ticker} {h.weight:.1%}' for h in s.holdings) or '—')}</td>"
        f"<td class=BUY>{escape(', '.join(s.buys) or '—')}</td><td class=SELL>{escape(', '.join(s.sells) or '—')}</td></tr>" for s in b.rebalances)
    out.append('<h2>Rebalance history</h2><div class="card scroll"><table><thead><tr><th>Decided</th><th>Executed</th><th class=num>NAV</th>'
               f"<th class=num>Cash</th><th>Holdings</th><th>Bought</th><th>Sold</th></tr></thead><tbody>{rows}</tbody></table></div>")
    if b.latest:
        out.append(_decision_html(b.latest, top=False))
    if b.next_proposal:
        out.append(_decision_html(b.next_proposal, top=False))
    return "".join(out)


def _chart(b: BacktestReport) -> str:
    """Fund vs benchmark NAV on one axis (same dollars), crosshair readout on hover."""
    W, H, L, R, T, B = 960, 320, 64, 64, 12, 28
    lo, hi = min(b.nav + b.benchmark_nav), max(b.nav + b.benchmark_nav)
    pad = (hi - lo) * 0.08 or hi * 0.02 or 1.0
    lo, hi = lo - pad, hi + pad
    n = len(b.dates)
    x = lambda i: L + (W - L - R) * (i / (n - 1) if n > 1 else 0.5)  # noqa: E731
    y = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))  # noqa: E731
    path = lambda vs: "M" + "L".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vs))  # noqa: E731
    ticks = [lo + (hi - lo) * k / 4 for k in range(5)]
    grid = "".join(f'<line x1="{L}" x2="{W - R}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="var(--rule)" stroke-width="1"/>'
                   f'<text x="{L - 8}" y="{y(t) + 4:.1f}" text-anchor="end" font-size="12" fill="var(--text-muted)">${t / 1000:,.0f}k</text>' for t in ticks)
    anchor = lambda i: "start" if i == 0 else "end" if i == n - 1 else "middle"  # noqa: E731  (keeps edge dates inside the plot)
    xlabels = "".join(f'<text x="{x(i):.1f}" y="{H - 8}" text-anchor="{anchor(i)}" font-size="12" fill="var(--text-muted)">{escape(b.dates[i])}</text>'
                      for i in sorted({0, n // 2, n - 1}))
    # Direct labels at the line ends, nudged apart when the two ends nearly meet.
    yf, yb = y(b.nav[-1]), y(b.benchmark_nav[-1])
    if abs(yf - yb) < 14:
        mid = (yf + yb) / 2
        yf, yb = (mid - 7, mid + 7) if yf <= yb else (mid + 7, mid - 7)
    ends = (f'<text x="{x(n - 1) + 6:.1f}" y="{yf + 4:.1f}" font-size="12" fill="var(--text-secondary)">Fund</text>'
            f'<text x="{x(n - 1) + 6:.1f}" y="{yb + 4:.1f}" font-size="12" fill="var(--text-secondary)">{escape(b.benchmark)}</text>')
    data = json.dumps({"dates": b.dates, "fund": b.nav, "bench": b.benchmark_nav, "L": L, "R": R, "W": W, "bench_name": b.benchmark})
    script = _CHART_JS.replace("__DATA__", data.replace("</", "<\\/"))  # a "</script>" inside data must not close the tag
    return (f'<div class="legend"><span><i style="background:var(--series-1)"></i>Fund</span><span><i style="background:var(--series-2)"></i>{escape(b.benchmark)} (same capital)</span></div>'
            f'<div class="scroll"><div class="chart" id="eq"><svg viewBox="0 0 {W} {H}" role="img" aria-label="Fund NAV versus {escape(b.benchmark)}, {escape(b.start)} to {escape(b.end)}">{grid}{xlabels}{ends}'
            f'<path d="{path(b.benchmark_nav)}" fill="none" stroke="var(--series-2)" stroke-width="2" stroke-linejoin="round"/>'
            f'<path d="{path(b.nav)}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round"/>'
            f'<line id="eq-x" y1="{T}" y2="{H - B}" stroke="var(--text-muted)" stroke-width="1" visibility="hidden"/>'
            f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" fill="transparent" id="eq-hit"/></svg><div class="tip" id="eq-tip"></div></div></div>'
            f"<script>{script}</script>")


_CHART_JS = """(()=>{const d=__DATA__,root=document.getElementById('eq'),svg=root.querySelector('svg'),hit=document.getElementById('eq-hit'),
line=document.getElementById('eq-x'),tip=document.getElementById('eq-tip'),n=d.dates.length,f=v=>'$'+Math.round(v).toLocaleString();
function show(e){const r=svg.getBoundingClientRect(),vx=(e.clientX-r.left)*d.W/r.width,span=d.W-d.L-d.R;
const i=Math.max(0,Math.min(n-1,Math.round((vx-d.L)/span*(n-1)))),px=d.L+span*(n>1?i/(n-1):.5);
line.setAttribute('x1',px);line.setAttribute('x2',px);line.setAttribute('visibility','visible');tip.replaceChildren();
const head=document.createElement('div');head.textContent=d.dates[i];tip.append(head);
for(const [k,v] of [['Fund',d.fund[i]],[d.bench_name,d.bench[i]]]){const row=document.createElement('div'),b=document.createElement('b');
b.textContent=f(v);row.append(b,document.createTextNode(' '+k));tip.append(row)}
tip.style.display='block';const sc=root.parentElement,lo=sc.scrollLeft,hi=lo+sc.clientWidth,tw=tip.offsetWidth,at=px*r.width/d.W;
let left=at+12+tw<=hi?at+12:at-12-tw;tip.style.left=Math.max(lo,Math.min(left,hi-tw))+'px';tip.style.top='8px'}
hit.addEventListener('pointermove',show);hit.addEventListener('pointerleave',()=>{tip.style.display='none';line.setAttribute('visibility','hidden')})})();"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tiles(items: list[tuple[str, str, str]]) -> str:
    return '<div class="tiles">' + "".join(
        f'<div class="card tile"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div><div class="s">{escape(s)}</div></div>' for k, v, s in items) + "</div>"


def _chip(d: TickerDecision) -> str:
    detail = "" if d.action_detail in ("none", "hold") else f" · {d.action_detail}"
    trade = f" {d.trade_shares:+,}" if d.trade_shares else ""
    return f'<span class="chip {d.action}">{d.action}{escape(detail)}</span><span class="muted">{trade}</span>'


def _action_label(d: TickerDecision) -> str:
    detail = "" if d.action_detail in ("none", "hold") else f" ({d.action_detail}{f' {d.trade_shares:+,}' if d.trade_shares else ''})"
    return d.action + detail


def _signals(d: TickerDecision) -> str:
    return ", ".join(f"{v.analyst} {v.signal}" for v in d.views) or "—"


def _confidences(d: TickerDecision) -> str:
    return ", ".join(f"{v.confidence:.0f}" for v in d.views if v.confidence is not None) or "—"


def _money(v: float | None) -> str:
    return f"${v:,.2f}" if v is not None else "—"


def _n(v: int | None) -> str:
    return f"{v:,}" if v is not None else "—"


def _md(text: str) -> str:
    """Keep free text inside one Markdown table cell or list item, and inert:
    many Markdown viewers render inline HTML, and this text is LLM output."""
    return " ".join(text.split()).replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")
