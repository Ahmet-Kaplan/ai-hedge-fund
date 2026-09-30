"""Offline integrity + look-ahead audit of the Buffett baseline result.

Reads result.json, universe_top10.json and llm_cache/; re-queries fundamentals
only through the local SEC/Tiingo caches (request counts are reported).
Makes no LLM calls. Financial Datasets is removed from the environment.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

for var in ("FINANCIAL_DATASETS_API_KEY", "HEDGE_FUND_DATA_PROVIDER", "HEDGE_FUND_DATA_SUPPLEMENT"):
    os.environ.pop(var, None)

from hedge_fund.data import make_data_client  # noqa: E402
from hedge_fund.universe.builder import load_schedule  # noqa: E402
from hedge_fund.verification.checks import pit_violations  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/buffett-baseline")
res = json.loads((OUT / "result.json").read_text())
pit = load_schedule(OUT / "universe_top10.json")
MAX_POS, MAX_GROSS, EPS = 0.15, 1.0, 1e-9

integrity, lookahead, notes = [], [], []

# ---- data integrity -------------------------------------------------------
dates, nav, bnav = res["dates"], res["nav"], res["benchmark_nav"]
if not (len(dates) == len(nav) == len(bnav)):
    integrity.append("dates/nav/benchmark_nav lengths differ")
if dates != sorted(set(dates)):
    integrity.append("valuation dates not strictly increasing")
if any(not math.isfinite(x) or x <= 0 for x in nav + bnav):
    integrity.append("non-finite or non-positive NAV")
if res["capital"] != 100000:
    integrity.append(f"capital {res['capital']} != 100000")
if dates[0] < "2016-07-01" or dates[-1] > "2026-06-30":
    integrity.append(f"valuation window {dates[0]}..{dates[-1]} outside the approved period")
for rec in res["records"]:
    tag = f"{rec['as_of']}->{rec['execution_as_of']}"
    fw = rec["final_weights"]
    if any(w < -EPS for w in fw.values()):
        integrity.append(f"{tag}: negative (short) weight")
    if any(w > MAX_POS + EPS for w in fw.values()):
        integrity.append(f"{tag}: weight above 15%")
    if sum(abs(w) for w in fw.values()) > MAX_GROSS + EPS:
        integrity.append(f"{tag}: gross exposure above 100%")
    if rec["cash"] < -0.01:
        integrity.append(f"{tag}: negative cash (leverage)")
    if any(s < 0 for s in rec["positions"].values()):
        integrity.append(f"{tag}: short position")
    models = {sig["model_name"] for sr in rec["strategies"] for sig in sr["signals"]}
    if models - {"buffett"}:
        integrity.append(f"{tag}: non-Buffett signal sources {models}")
    for f in rec["fills"]:
        if not (f["price"] > 0 and math.isfinite(f["price"])):
            integrity.append(f"{tag}: bad fill price {f}")

# ---- look-ahead -----------------------------------------------------------
pairs = set()
decisions = []
for rec in res["records"]:
    decisions.append(rec)
    if rec.get("refreshed_assessment"):
        decisions.append(rec["refreshed_assessment"])
decisions += [p["proposal"] for p in res.get("pending", [])]
for rec in decisions:
    as_of = rec["as_of"]
    if rec.get("execution_as_of") and rec["execution_as_of"] <= as_of:
        lookahead.append(f"executed {rec['execution_as_of']} not after decision {as_of}")
    snap = max((s for s in pit.snapshots if s.as_of <= as_of), key=lambda s: s.as_of, default=None)
    allowed = set(snap.tickers) if snap else set()
    if set(rec["universe"]) - allowed:
        lookahead.append(f"{as_of}: universe {sorted(set(rec['universe']) - allowed)} not in snapshot {snap and snap.as_of}")
    for sr in rec["strategies"]:
        for sig in sr["signals"]:
            pairs.add((sig["ticker"], as_of))
            if sig["date"] > as_of:
                lookahead.append(f"{sig['ticker']}: signal dated {sig['date']} after decision {as_of}")

with make_data_client() as fd:
    for ticker, as_of in sorted(pairs):
        rows = [m.model_dump() for m in fd.get_financial_metrics(ticker, as_of, period="ttm", limit=12)]
        bad = pit_violations(rows, as_of)
        if bad:
            lookahead.append(f"{ticker}@{as_of}: {bad[:2]}")
    requests = fd.request_counts()

# Blind prompts: no ticker symbols, company names or calendar years.
cache_files = sorted((OUT / "llm_cache").glob("*.json"))
year_re = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
universe = set(res["universe"])
leaks = 0
for p in cache_files:
    r = json.loads(p.read_text())
    user = r.get("user", "")
    if "Company: (withheld)" not in user:
        leaks += 1
        lookahead.append(f"{p.name}: prompt not anonymised")
    elif year_re.search(user) or any(re.search(rf"\b{re.escape(t)}\b", user) for t in universe if len(t) > 1):
        leaks += 1
        lookahead.append(f"{p.name}: prompt contains a calendar year or ticker")

report = {
    "data_integrity": "PASS" if not integrity else "FAIL",
    "integrity_problems": integrity[:50],
    "look_ahead": "PASS" if not lookahead else "FAIL",
    "look_ahead_problems": lookahead[:50],
    "fundamental_queries_checked": len(pairs),
    "prompts_checked_for_anonymisation": len(cache_files),
    "audit_requests": requests,
}
(OUT / "audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
