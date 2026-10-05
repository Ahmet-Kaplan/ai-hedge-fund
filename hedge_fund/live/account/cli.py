"""aihf-live — the real-money account: S&P 500 core + a satellite mirroring the paper fund.

    aihf-live init                     write a safe ./live.yaml (confirm_live: false)
    aihf-live status                   account, settings, halts
    aihf-live run [--dry-run]          reconcile yesterday, then submit today
    aihf-live reconcile | submit [--dry-run]
    aihf-live report                   deposits, value, time-weighted return vs the core ETF
    aihf-live review [--apply]         propose (and apply) a new agent share

Real orders need live keys (ALPACA_LIVE_KEY_ID / ALPACA_LIVE_SECRET_KEY),
confirm_live: true, and one reviewed dry run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from hedge_fund.brokers.alpaca import AlpacaLiveClient
from hedge_fund.data.crypto_prices import CryptoPriceSource, crypto_closes
from hedge_fund.data.factory import open_data_client
from hedge_fund.data.sessions import NEW_YORK
from hedge_fund.data.store import MarketStore
from hedge_fund.live.account.ledger import LiveLedger
from hedge_fund.live.account.report import build_live_report
from hedge_fund.live.account.review import apply_review, propose_review
from hedge_fund.live.account.runner import reconcile_live, submit_live
from hedge_fund.live.account.settings import LiveSettings, load_settings, save_settings
from hedge_fund.live.launchd import notify
from hedge_fund.live.ledger import Ledger
from hedge_fund.live.runner import session_to_reconcile
from hedge_fund.paths import KILL_PATH, LIVE_SETTINGS_PATH, MARKET_DB_PATH
from hedge_fund.tui.keys import apply_credentials

logger = logging.getLogger("aihf-live")


def default_settings_path() -> Path:
    """live.yaml in the working directory (the repo), like .env; else ~/.hedge-fund/live.yaml.

    The local file wins when both exist. With neither, `init` creates the local one.
    """
    local = Path.cwd() / "live.yaml"
    if local.exists() or not LIVE_SETTINGS_PATH.exists():
        return local
    return LIVE_SETTINGS_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aihf-live", description="Run the real-money core + satellite account.")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help)
        p.add_argument("--settings", default=None, help="settings file (default: ./live.yaml, else ~/.hedge-fund/live.yaml)")
        return p

    command("init", "write a safe live.yaml (never overwrites)")
    command("status", "account, settings, halts")
    command("reconcile", "record the previous session's value and deposits")
    command("submit", "trade today").add_argument("--dry-run", action="store_true")
    command("run", "reconcile then submit").add_argument("--dry-run", action="store_true")
    command("report", "deposits, value, return vs the core ETF")
    command("review", "propose a new agent share").add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    apply_credentials()
    args = build_parser().parse_args(argv)
    path = Path(args.settings) if args.settings else default_settings_path()
    if args.command == "init":
        if path.exists():
            print(f"{path} already exists; not overwriting")
            return 1
        save_settings(LiveSettings(), path)
        print(f"wrote {path} (confirm_live: false, agent_share: 0). Review it, then set confirm_live: true.")
        return 0
    ledger = LiveLedger.default()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True,
        handlers=[logging.StreamHandler(), logging.FileHandler(ledger.log_path(date.today().isoformat(), args.command))],
    )
    try:
        settings = load_settings(path)
        return _COMMANDS[args.command](args, settings, path, ledger)
    except Exception as exc:
        logger.exception("%s failed", args.command)
        notify(f"aihf-live {args.command} failed", str(exc))
        return 1


def _crypto_marks(names: list[str], day: str) -> dict[str, float]:
    """Free Alpaca crypto closes for the UTC day `day`, cached in the market store."""
    closes = crypto_closes(MarketStore(MARKET_DB_PATH), CryptoPriceSource(), names, day, day)
    missing = [n for n in names if day not in closes[n]]
    if missing:
        raise ValueError(f"no crypto close on {day} for {', '.join(missing)}")
    return {n: closes[n][day] for n in names}


def _reconcile(args, settings, path, ledger) -> int:
    client = AlpacaLiveClient()
    session = session_to_reconcile(client, datetime.now(NEW_YORK))
    with open_data_client() as data:
        row = reconcile_live(settings, client, data, ledger, session=session, crypto_marks=_crypto_marks)
    print(f"{session}: equity ${row.equity:,.2f} · core ${row.core_value:,.2f} · satellite ${row.satellite_value:,.2f}"
          f" · crypto ${row.crypto_value:,.2f} · deposits ${row.net_flow:,.2f}")
    return 0


def _submit(args, settings, path, ledger) -> int:
    client = AlpacaLiveClient()
    with open_data_client() as data:
        result = submit_live(settings, client, data, ledger, Ledger.for_fund(settings.paper_fund),
                             now=datetime.now(NEW_YORK), dry_run=args.dry_run, kill_path=KILL_PATH,
                             crypto_marks=_crypto_marks)
    print(f"{result.session}: {result.status} {result.detail}")
    for t, w in sorted(result.target.items(), key=lambda kv: -abs(kv[1])):
        print(f"  target {t:6} {w:+.1%}")
    for o in result.orders:
        what = f"${o.dollars:,.2f}" if o.dollars is not None else (f"{o.qty:g} sh" if o.qty is not None else f"{o.shares} sh")
        print(f"  {o.side:4} {o.ticker:6} {what}")
    for r in result.results:
        if r.status == "rejected":
            print(f"  REJECTED {r.ticker}: {r.reason}")
    return 0


def _run(args, settings, path, ledger) -> int:
    try:
        _reconcile(args, settings, path, ledger)
    except Exception:
        logger.exception("reconcile failed; submitting anyway")
    return _submit(args, settings, path, ledger)


def _status(args, settings, path, ledger) -> int:
    account = AlpacaLiveClient().account()
    print(f"live account {account.status} · equity ${account.equity:,.2f} · cash ${account.cash:,.2f}")
    print(f"crypto half {settings.crypto_share:.0%} ({', '.join(f'{t} {w:.0%}' for t, w in settings.crypto_core.items())})")
    print(f"core {settings.core_ticker} · agent share {settings.agent_share:.0%} · shorts "
          f"{'on' if settings.shorts_enabled else 'off'} · confirm_live {settings.confirm_live}")
    print(f"kill switch {'ON' if KILL_PATH.exists() else 'off'} · satellite halted: {ledger.satellite_halted()}"
          f" · dry run done: {ledger.dry_run_done()}")
    return 0


def _report(args, settings, path, ledger) -> int:
    report = build_live_report(ledger, settings)
    print(report.model_dump_json(indent=2) if report else "no reconciled sessions yet")
    return 0


def _review(args, settings, path, ledger) -> int:
    result = propose_review(Ledger.for_fund(settings.paper_fund), ledger, settings)
    print(result.model_dump_json(indent=2))
    if args.apply:
        apply_review(result, settings, path, ledger, today=date.today().isoformat())
        print(f"agent share {result.old_share:.0%} → {result.new_share:.0%}")
    else:
        print("re-run with --apply to set this share")
    return 0


_COMMANDS = {"status": _status, "reconcile": _reconcile, "submit": _submit, "run": _run,
             "report": _report, "review": _review}


if __name__ == "__main__":
    sys.exit(main())
