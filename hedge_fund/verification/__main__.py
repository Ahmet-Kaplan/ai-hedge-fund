"""Verify the data assumptions a backtest depends on, against the live API.

Usage::

    python -m hedge_fund.verification
    python -m hedge_fund.verification --universe-file configs/baseline-universe.yaml --out verification.json
    python -m hedge_fund.verification --preflight   # key check only, no requests

Reads the data provider's keys (make_data_client: TIINGO_API_KEY and
SEC_USER_AGENT by default) from the process environment only — no .env
files — and exits 2 naming whatever is missing, without making a request.
Prices are downloaded once per ticker and kept in the local Tiingo store, so
a first run costs about one Tiingo request per ticker plus ~2 SEC requests
per company, and a repeat run close to none.

Exit codes: 0 all checks pass/info, 1 any fail/error/inconclusive, 2 keys missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from hedge_fund.data.factory import make_data_client, required_data_env
from hedge_fund.verification.runner import missing_env, request_count, required_env, run_checks

_MARK = {"pass": "PASS", "fail": "FAIL", "inconclusive": "INCONCLUSIVE", "info": "INFO", "error": "ERROR", "skipped": "SKIPPED"}


def load_universe(path: str | Path) -> list[str]:
    data = yaml.safe_load(Path(path).read_text())
    tickers = data.get("tickers") if isinstance(data, dict) else None
    if not isinstance(tickers, list) or not tickers:
        raise ValueError(f"{path}: expected a mapping with a non-empty 'tickers' list")
    return [str(t).strip().upper() for t in tickers]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hedge_fund.verification", description=__doc__.split("\n\n")[0])
    parser.add_argument("--universe-file", help="YAML with a 'tickers' list to check coverage for")
    parser.add_argument("--start", default="2025-01-01", help="universe coverage window start")
    parser.add_argument("--end", default="2026-08-31", help="universe coverage window end")
    parser.add_argument("--model", help="LLM model id to check the key for (default: HEDGE_FUND_LLM_MODEL or claude-opus-5-5)")
    parser.add_argument("--preflight", action="store_true", help="only report which environment variables are missing")
    parser.add_argument("--out", help="write the full JSON report here")
    args = parser.parse_args(argv)

    needed = required_env(args.model)
    missing = missing_env(needed)
    print("Environment:", file=sys.stderr)
    for name in needed:
        print(f"  {name}: {'MISSING' if name in missing else 'set'}", file=sys.stderr)
    if args.preflight:
        return 2 if missing else 0
    data_missing = [name for name in required_data_env() if name in missing]
    if data_missing:
        print(f"\n{', '.join(data_missing)} not set — no data checks were run. "
              "Export it in your shell and rerun.", file=sys.stderr)
        return 2

    universe = load_universe(args.universe_file) if args.universe_file else None

    def show(result):
        print(f"[{_MARK[result.status]:>12}] {result.name}: {result.summary}", file=sys.stderr)

    with make_data_client() as client:
        results = run_checks(client, universe, (args.start, args.end), on_result=show)
        requests_made = request_count(client)
        by_source = client.request_counts() if hasattr(client, "request_counts") else {}

    report = {"requests_made": requests_made, "requests_by_source": by_source,
              "results": [r.model_dump() for r in results]}
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    print(f"\n{requests_made} HTTP requests made {by_source}.", file=sys.stderr)
    return 0 if all(r.status in ("pass", "info") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
