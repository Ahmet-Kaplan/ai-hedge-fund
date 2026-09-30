"""aihf-paper — run the fund on an Alpaca paper account.

    aihf-paper status                  account, halts, last NAV, schedule
    aihf-paper submit [--dry-run] [--now]  plan (and send) today's MOC rebalance
    aihf-paper reconcile               record the previous session's fills + NAV
    aihf-paper report [--backtest F] [--attribution]
    aihf-paper baseline [--start D] [--end D]
    aihf-paper flatten --yes           halt and close every position at today's close
    aihf-paper resume                  clear a halt
    aihf-paper install-schedule | uninstall-schedule

Paper only: the Alpaca client refuses any endpoint but paper-api.alpaca.markets.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

from hedge_fund.backtesting.fund import FundBacktestResult
from hedge_fund.brokers.alpaca import AlpacaPaperClient
from hedge_fund.data import CachedDataClient, FDClient
from hedge_fund.data.sessions import NEW_YORK, completed_through
from hedge_fund.fund import Fund, FundSpec, load_spec, normalize_universe
from hedge_fund.live.baseline import POST_CUTOFF_START, run_baseline
from hedge_fund.live.launchd import install_schedule, notify, schedule_installed, uninstall_schedule
from hedge_fund.live.ledger import Ledger
from hedge_fund.live.report import MIN_REBALANCES_FOR_VERDICT, build_report, strategy_attribution
from hedge_fund.live.runner import reconcile, session_to_reconcile, submit
from hedge_fund.paths import KILL_PATH
from hedge_fund.tui.keys import apply_credentials

logger = logging.getLogger("aihf-paper")

PACKAGE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANDATE = PACKAGE_DIR / "fund" / "paper.yaml"
DEFAULT_UNIVERSE = PACKAGE_DIR / "fund" / "paper_universe.list"


def load_universe(path: str | Path) -> list[str]:
    """Tickers from a text file: whitespace-separated, `#` comments, duplicates dropped."""
    tickers: list[str] = []
    for line in Path(path).read_text().splitlines():
        tickers.extend(line.split("#", 1)[0].split())
    return normalize_universe(tickers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aihf-paper", description="Run the fund on an Alpaca paper account.")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help)
        p.add_argument("--mandate", default=str(DEFAULT_MANDATE), help="fund spec YAML")
        p.add_argument("--universe", default=str(DEFAULT_UNIVERSE), help="ticker list file")
        p.add_argument("--model", help="LLM for the investor agents, e.g. claude-opus-5-5")
        return p

    p = command("submit", "plan and send today's market-on-close rebalance")
    p.add_argument("--dry-run", action="store_true", help="plan and save, send nothing (works any day)")
    p.add_argument("--now", action="store_true",
                   help="rebalance today even if it is not the first session of the period (e.g. to start the fund)")
    command("reconcile", "record the previous session's fills and closing NAV")
    p = command("report", "paper performance from the ledger")
    p.add_argument("--backtest", help="a backtest result JSON to compare over the same dates")
    p.add_argument("--attribution", action="store_true", help="per-strategy contribution (fetches prices)")
    command("status", "account, halts, last NAV, schedule")
    command("flatten", "halt and close every position at today's close").add_argument("--yes", action="store_true")
    command("resume", "clear a halt so the fund trades again")
    p = command("baseline", "backtest the fund, each strategy, SPY and equal-weight")
    p.add_argument("--start", default=POST_CUTOFF_START)
    p.add_argument("--end", default=completed_through())
    p.add_argument("--out", help="where to write the JSON (default: the ledger's baseline/ dir)")
    command("install-schedule", "install the launchd jobs")
    command("uninstall-schedule", "remove the launchd jobs")
    return parser


def main(argv: list[str] | None = None) -> int:
    apply_credentials()
    args = build_parser().parse_args(argv)
    if args.model:
        os.environ["HEDGE_FUND_LLM_MODEL"] = args.model
    spec = load_spec(args.mandate)
    ledger = Ledger.for_fund(spec.name)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True,
        handlers=[logging.StreamHandler(), logging.FileHandler(ledger.log_path(date.today().isoformat(), args.command))],
    )
    try:
        return _COMMANDS[args.command](args, spec, ledger)
    except Exception as exc:
        logger.exception("%s failed", args.command)
        notify(f"aihf-paper {args.command} failed", str(exc))
        return 1


def _submit(args, spec: FundSpec, ledger: Ledger) -> int:
    fund = Fund(spec)
    with FDClient() as raw:
        result = submit(fund, load_universe(args.universe), AlpacaPaperClient(), CachedDataClient(raw),
                        ledger, now=datetime.now(NEW_YORK), dry_run=args.dry_run,
                        force_rebalance=getattr(args, "now", False))
    print(f"{result.session}: {result.status} {result.detail}")
    if result.plan:
        print(f"equity ${result.plan.equity:,.2f} · {len(result.plan.targets)} targets · {len(result.plan.orders)} orders")
        for o in result.plan.orders:
            print(f"  {o.side:4} {o.quantity:>6} {o.ticker:6} ref ${o.price:,.2f}  (${o.quantity * o.price:,.0f})")
    for r in result.orders:
        if r.status == "rejected":
            print(f"  REJECTED {r.ticker}: {r.reason}")
    return 0


def _reconcile(args, spec: FundSpec, ledger: Ledger) -> int:
    client = AlpacaPaperClient()
    session = session_to_reconcile(client, datetime.now(NEW_YORK))
    with FDClient() as raw:
        result = reconcile(spec, client, CachedDataClient(raw), ledger, session=session)
    print(f"{session}: equity ${result.nav.equity:,.2f} · gross {result.nav.gross:.0%} · {len(result.fills)} fills")
    for message in result.mismatches:
        print(f"  WARNING {message}")
    return 0


def _report(args, spec: FundSpec, ledger: Ledger) -> int:
    backtest = FundBacktestResult.model_validate_json(Path(args.backtest).read_text()) if args.backtest else None
    report = build_report(ledger, backtest=backtest)
    if report is None:
        print("not enough history yet: need at least two reconciled sessions")
        return 0
    if args.attribution:
        with FDClient() as raw:
            report.strategy_contribution = strategy_attribution(ledger, CachedDataClient(raw))
    print(report.model_dump_json(indent=2))
    if report.n_rebalances < MIN_REBALANCES_FOR_VERDICT:
        print(f"note: {report.n_rebalances} rebalances so far; no verdict before {MIN_REBALANCES_FOR_VERDICT}", file=sys.stderr)
    return 0


def _status(args, spec: FundSpec, ledger: Ledger) -> int:
    client = AlpacaPaperClient()
    account = client.account()
    positions = client.positions()
    rows = ledger.nav_rows()
    halted = ledger.halted_path.read_text().strip() if ledger.is_halted() else "no"
    print(f"fund {spec.name} · account {account.status} · equity ${account.equity:,.2f} · cash ${account.cash:,.2f}")
    print(f"positions {len(positions)} ({sum(s < 0 for s in positions.values())} short)")
    fractions = client.fractional_holdings()
    if fractions:
        print(f"WARNING fractional holdings the fund ignores (close them in Alpaca): "
              f"{', '.join(f'{t} {q:g}' for t, q in sorted(fractions.items()))}")
    print(f"kill switch {'ON' if KILL_PATH.exists() else 'off'} · halted: {halted}")
    print(f"last reconciled: {rows[-1].date} ${rows[-1].equity:,.2f}" if rows else "last reconciled: never")
    print(f"schedule {'installed' if schedule_installed() else 'not installed'} "
          "(local times; reinstall after a DST change; the Mac must be awake at 09:00 and 10:00 ET)")
    print(f"ledger {ledger.root}")
    return 0


def _flatten(args, spec: FundSpec, ledger: Ledger) -> int:
    if not args.yes:
        print("flatten halts the fund and closes every position at today's close; re-run with --yes")
        return 2
    ledger.halt("manual flatten via `aihf-paper flatten`. Run `aihf-paper resume` to trade again.")
    args.dry_run = False
    return _submit(args, spec, ledger)


def _resume(args, spec: FundSpec, ledger: Ledger) -> int:
    print("resumed" if ledger.resume() else "was not halted")
    return 0


def _baseline(args, spec: FundSpec, ledger: Ledger) -> int:
    with FDClient() as raw:
        report = run_baseline(spec, load_universe(args.universe), args.start, args.end, CachedDataClient(raw))
    out = Path(args.out) if args.out else ledger.root / "baseline" / f"{args.start}_{args.end}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2))
    label = " (POSSIBLY MEMORIZED: window starts before the LLM's training cutoff)" if report.possibly_memorized else ""
    print(f"baseline {report.start} → {report.end}{label}")
    print(f"{'variant':24} {'return':>8} {'annual':>8} {'sharpe':>7} {'max dd':>7} {'costs':>9}  beats both?")
    for r in report.rows:
        verdict = {True: "yes", False: "no", None: "-"}[r.beats_benchmarks]
        print(f"{r.name:24} {r.total_return_pct:>+8.1%} {r.annualized_return_pct:>+8.1%} {r.sharpe_ratio:>7.2f} "
              f"{r.max_drawdown_pct:>7.1%} {r.total_costs:>9,.0f}  {verdict}")
    print(f"saved {out}")
    return 0


def _install(args, spec: FundSpec, ledger: Ledger) -> int:
    paths = install_schedule(mandate=Path(args.mandate).resolve(), universe=Path(args.universe).resolve(),
                             log_dir=ledger.root / "logs")
    for path in paths:
        print(f"installed {path}")
    print("reconcile 09:00 ET and submit 10:00 ET on weekdays; the Mac must be awake then")
    return 0


def _uninstall(args, spec: FundSpec, ledger: Ledger) -> int:
    for path in uninstall_schedule():
        print(f"removed {path}")
    return 0


_COMMANDS = {
    "submit": _submit, "reconcile": _reconcile, "report": _report, "status": _status,
    "flatten": _flatten, "resume": _resume, "baseline": _baseline,
    "install-schedule": _install, "uninstall-schedule": _uninstall,
}


if __name__ == "__main__":
    sys.exit(main())
