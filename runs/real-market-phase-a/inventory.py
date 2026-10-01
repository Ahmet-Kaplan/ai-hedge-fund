"""STEP 2 — inventory of the real data already in the local Tiingo/SEC caches.

Offline only: TiingoClient(offline=True) never makes a request.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from hedge_fund.data import tiingo as tmod
from hedge_fund.data.security_events import load_events
from hedge_fund.data.tiingo import TiingoClient, tiingo_symbol
from hedge_fund.universe.builder import load_schedule

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/real-market-phase-a")
START, END = "2016-07-01", "2026-06-30"
schedule = load_schedule("runs/buffett-baseline/universe_top10.json")
tickers = schedule.all_tickers()
cache = Path.home() / ".hedge-fund" / "cache" / "tiingo"

with TiingoClient(cache_dir=cache, offline=True) as tc:
    spy = tc.get_prices("SPY", START, END)
    sessions = [b.time[:10] for b in spy]
    inv = {"window": [START, END], "benchmark": "SPY", "benchmark_sessions": len(sessions),
           "benchmark_first": sessions[0], "benchmark_last": sessions[-1],
           "cache_files": len(list(cache.glob("*.json.gz"))), "instruments": {}}
    events = load_events()
    for t in ["SPY", *tickers]:
        rec = {"tiingo_symbol": tiingo_symbol(t), "cached": tc.is_stored(t)}
        try:
            rows = [r for r in tc._rows(t) if START <= r["date"] <= END]
        except tmod.TiingoClientError as exc:
            rec["error"] = str(exc)
            inv["instruments"][t] = rec
            continue
        days = {r["date"] for r in rows}
        member_days = [d for d in sessions if t == "SPY" or t in schedule.members_on(d)]
        hist = tc.history_range(t)
        rec.update(
            history_range=list(hist) if hist else None,
            rows_in_window=len(rows),
            missing_sessions_in_window=len([d for d in sessions if d not in days and (hist and hist[0] <= d <= hist[1])]),
            member_sessions=len(member_days),
            missing_member_sessions=len([d for d in member_days if d not in days]),
            missing_close=sum(1 for r in rows if not r["close"] or r["close"] <= 0),
            zero_volume=sum(1 for r in rows if not r["volume"]),
            dividends=sum(1 for r in rows if r["divCash"]),
            splits={r["date"]: r["splitFactor"] for r in rows if r["splitFactor"] != 1.0},
            delisting_event=events.get(t.replace("-", ".")).model_dump() if events.get(t.replace("-", ".")) else None,
        )
        rec["completeness_member_sessions"] = (1 - rec["missing_member_sessions"] / len(member_days)) if member_days else None
        inv["instruments"][t] = rec
    inv["tiingo_requests"] = tc.requests

inv["universe"] = {"source": "runs/buffett-baseline/universe_top10.json", "top_n": schedule.config.top_n,
                   "snapshots": {s.as_of: s.tickers for s in schedule.snapshots}, "union": tickers}
inv["sec_cache_files"] = sum(1 for _ in (Path.home() / ".hedge-fund" / "cache" / "edgar").rglob("*") if _.is_file())
(OUT / "data_inventory.json").write_text(json.dumps(inv, indent=1, sort_keys=True))
bad = {t: r for t, r in inv["instruments"].items() if r.get("error") or (r.get("completeness_member_sessions") or 1) < 0.99}
print(json.dumps({"instruments": len(inv["instruments"]), "sessions": inv["benchmark_sessions"],
                  "range": [inv["benchmark_first"], inv["benchmark_last"]], "tiingo_requests": inv["tiingo_requests"],
                  "incomplete_or_error": {t: r.get("error") or r.get("completeness_member_sessions") for t, r in bad.items()}}, indent=1))
for t, r in inv["instruments"].items():
    if "error" not in r:
        print(f"{t:6} hist={r['history_range']} rows={r['rows_in_window']} miss_member={r['missing_member_sessions']} "
              f"zero_vol={r['zero_volume']} divs={r['dividends']} splits={r['splits']}")
