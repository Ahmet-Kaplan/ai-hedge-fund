"""aihf-crypto — the crypto sleeve's research commands.

    aihf-crypto backtest [--fee-bps 25] [--end YYYY-MM-DD]

Backtests buy-and-hold 60/30/10 BTC/ETH/SOL against each pre-registered trend
rule on free Alpaca daily data since 2021, after fees, and applies the
spec's pre-committed bar. No money and no LLM are involved.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone

from hedge_fund.crypto.backtest import passes_bar, passes_rebalance_bar, run_backtest, split_metrics
from hedge_fund.crypto.rules import RULES, WARMUP_DAYS
from hedge_fund.data.crypto_prices import CryptoPriceSource, crypto_bars
from hedge_fund.data.store import MarketStore
from hedge_fund.paths import CRYPTO_DIR, MARKET_DB_PATH

CORE_WEIGHTS = {"BTC/USD": 0.6, "ETH/USD": 0.3, "SOL/USD": 0.1}
DATA_START = "2021-01-01"
SPLIT = "2023-06-30"
BANDS = {"band10": 0.10, "band20": 0.20}   # spec §12, pre-registered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aihf-crypto", description="Crypto sleeve research.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("backtest", help="core vs pre-registered trend rules, after fees")
    p.add_argument("--fee-bps", type=float, default=25.0, help="per side; Alpaca level-1 taker is 25")
    p.add_argument("--end", default=None, help="last UTC day (default: yesterday)")
    return parser


def evaluate(closes: dict[str, dict[str, float]], *, fee_bps: float, end: str,
             highs: dict | None = None, lows: dict | None = None) -> list[dict]:
    first = min(min(c) for c in closes.values())
    start = (date.fromisoformat(first) + timedelta(days=WARMUP_DAYS)).isoformat()
    variants = {"core": None, **RULES}
    results = {name: run_backtest(closes, CORE_WEIGHTS, rule, fee_bps=fee_bps, start=start, end=end,
                                  highs=highs, lows=lows)
               for name, rule in variants.items()}
    results.update({name: run_backtest(closes, CORE_WEIGHTS, None, fee_bps=fee_bps, start=start, end=end, band=b)
                    for name, b in BANDS.items()})
    split = {name: split_metrics(r, SPLIT) for name, r in results.items()}
    rows = []
    for name, r in results.items():
        m = split[name]
        rows.append({
            "variant": name, "start": r.dates[0], "end": r.dates[-1],
            "total_return": m["full"].total_return, "annualized": m["full"].annualized_return,
            "sharpe": m["full"].sharpe, "sharpe_h1": m["h1"].sharpe, "sharpe_h2": m["h2"].sharpe,
            "max_drawdown": m["full"].max_drawdown, "fees_pct": r.fees / r.nav[0],
            "passes": None if name == "core" else (passes_rebalance_bar(m, split["core"]) if name in BANDS
                                                   else passes_bar(m, split["core"])),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    end = args.end or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    closes, highs, lows = crypto_bars(MarketStore(MARKET_DB_PATH), CryptoPriceSource(), list(CORE_WEIGHTS), DATA_START, end)
    table = evaluate(closes, fee_bps=args.fee_bps, end=end, highs=highs, lows=lows)
    print(f"crypto backtest {table[0]['start']} → {table[0]['end']} · 60/30/10 BTC/ETH/SOL · "
          f"{args.fee_bps:g} bps per side · halves split {SPLIT}")
    print(f"{'variant':10} {'total':>9} {'annual':>8} {'sharpe':>7} {'1st half':>9} {'2nd half':>9} {'max dd':>7} {'fees':>6}  passes bar?")
    for r in table:
        verdict = {None: "-", True: "YES", False: "no"}[r["passes"]]
        print(f"{r['variant']:10} {r['total_return']:>+9.0%} {r['annualized']:>+8.1%} {r['sharpe']:>7.2f} "
              f"{r['sharpe_h1']:>9.2f} {r['sharpe_h2']:>9.2f} {r['max_drawdown']:>7.1%} {r['fees_pct']:>6.1%}  {verdict}")
    out = CRYPTO_DIR / "backtests" / f"{end}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fee_bps": args.fee_bps, "split": SPLIT, "rows": table}, indent=2))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
