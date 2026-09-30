"""Render a saved receipt as a decision report.

Usage::

    aihf configs/buffett-baseline.yaml --tickers ... --backtest --out result.json
    python -m hedge_fund.reporting result.json --out report.html
    python -m hedge_fund.reporting result.json            # Markdown to stdout

Accepts any receipt the engine writes: a run's CycleRecord, a pending
proposal, or a backtest result. The format follows --out's extension
(.html/.htm, else Markdown). Reads only the receipt — no API keys, no data
requests, no LLM calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from hedge_fund.reporting.decisions import load_receipt
from hedge_fund.reporting.render import to_html, to_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hedge_fund.reporting", description=__doc__.split("\n\n")[0])
    parser.add_argument("receipt", help="a run or backtest JSON receipt (aihf ... --out FILE)")
    parser.add_argument("--out", help="write here; .html/.htm renders HTML, anything else Markdown (default: Markdown to stdout)")
    args = parser.parse_args(argv)

    try:
        report = load_receipt(json.loads(Path(args.receipt).read_text()))
    except (OSError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        print(f"{args.receipt}: cannot build a report: {exc}", file=sys.stderr)
        return 1

    html = bool(args.out) and Path(args.out).suffix.lower() in (".html", ".htm")
    text = to_html(report) if html else to_markdown(report)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
