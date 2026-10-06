"""Run the AI hedge fund.

Usage::

    aihf
        No arguments: the interactive app — a Textual TUI (the same app as
        `aihf` with no arguments). Build a fund — pick stocks, strategies, rebalance
        cadence — or backtest a saved fund and watch its equity curve draw
        against its benchmark.

    aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT [--paper]
        With a mandate: one live-clock paper cycle (PaperBroker fills at
        mark; no live venue). --paper is the explicit flag and is also the
        default without --backtest. Assessment uses --date; execution lands
        on the next completed session close, so a proposal with no completed
        session yet comes back as a pending record instead. If this mandate
        has a prior CycleRecord receipt, the broker opens that ending book so
        cash, positions, and NAV carry forward; otherwise it opens at the
        mandate's capital. A corrupt or incompatible receipt fails the run.
        The full record prints to stdout as JSON (pipe it anywhere); a short
        human summary goes to stderr. Add --out record.json to also write a
        copy to a file. Optional cycle observability (--heartbeat / --events,
        or HEDGE_FUND_* env vars) records start/end/error without needing a
        live venue.

    aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --backtest
        Backtest the mandate: run_cycle looped over history at the mandate's
        rebalance cadence against SimBroker, executed at the next completed
        close; the full result JSON prints to stdout. Mutually exclusive
        with --paper.

    --allocator {static,equal_weight}
        Override the mandate's CIO. static (default) keeps StrategySpec
        slices; equal_weight is a stub that splits capital evenly.

A mandate is the desk — strategies, staff, risk, capital, cadence — and never
names tickers; --tickers says what to point it at for this run.

Both paths run the same engine underneath. The interactive app is a thin
client: it only *composes a FundSpec* — the same machine-facing YAML this
CLI reads. Humans click, machines write, the engine reads one thing.
"""

from __future__ import annotations

import argparse
import os
from datetime import date as _date
from datetime import timedelta
from pathlib import Path

from rich.console import Console

from hedge_fund.backtesting import backtest_fund
from collections.abc import Callable

from hedge_fund.data import open_data_client
from hedge_fund.data.protocol import DataClient
from hedge_fund.fund import ALLOCATOR_NAMES, Fund, load_spec, normalize_universe
from hedge_fund.journal import FileOrderJournal, JournalledBroker, journal_summary
from hedge_fund.ledger import save_cycle_record
from hedge_fund.observability import CycleObserver, observe_cycle
from hedge_fund.paths import ensure_mandates_dir, journal_path
from hedge_fund.pipeline.models import PendingRunResult
from hedge_fund.tui.keys import apply_credentials
from hedge_fund.tui.shared import _BACKTEST_WEEKS
from hedge_fund.venue import open_venue, VENUES


def _pit_universe(
    data: DataClient,
    start: str,
    end: str,
    size: int,
) -> Callable[[str], list[str]]:
    """A survivorship-free universe: the most liquid S&P members per month.

    Membership is reconstructed from the dated change log and ranked by dollar
    volume from *before* each month begins, so a backtest is offered the names
    an investor could have chosen then. The candidate set is prefetched once —
    that is the slow part, and it is why this is opt-in rather than the default.
    """
    from hedge_fund.data.store import MarketStore
    from hedge_fund.data.universe import PointInTimeUniverse
    from hedge_fund.paths import MARKET_DB_PATH

    universe = PointInTimeUniverse(MarketStore(MARKET_DB_PATH), data, size=size)
    universe.prefetch(start, end)
    return universe


def main() -> None:
    apply_credentials()
    ensure_mandates_dir()
    parser = argparse.ArgumentParser(
        prog="aihf",
        description="Run the AI hedge fund. No arguments: launch the "
        "interactive app. With a mandate YAML: run one live-clock paper "
        "cycle (seeded from the newest receipt when one exists) and print "
        "the record, or --backtest over history.",
    )
    parser.add_argument("mandate", nargs="?",
                        help="path to a fund spec YAML, e.g. "
                        "~/.hedge-fund/mandates/example.yaml "
                        "(omit to launch the interactive app)")
    parser.add_argument(
        "--tickers",
        help="what to trade this run, comma or space separated, e.g. "
        "AAPL,MSFT,NVDA — required with a mandate (a fund carries no "
        "watchlist; the universe is a run-time input)",
    )
    parser.add_argument(
        "--date",
        default=_date.today().isoformat(),
        help="analysis date YYYY-MM-DD (default: today), capped at yesterday "
        "in New York; execution uses the next completed session close",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--paper", action="store_true",
        help="run one live-clock paper cycle: PaperBroker fills at mark "
        "(no live venue); seed from the newest CycleRecord when one exists. "
        "This is also the default without --backtest",
    )
    mode.add_argument(
        "--backtest", action="store_true",
        help="backtest the mandate instead of running one cycle: one run_cycle "
        "per rebalance date from --start to --date against SimBroker, with "
        "daily valuation and next-close execution; full result JSON on stdout",
    )
    parser.add_argument(
        "--start",
        help=f"backtest start date YYYY-MM-DD (default: {_BACKTEST_WEEKS} weeks "
        "before --date)",
    )
    parser.add_argument(
        "--broker",
        choices=VENUES,
        default="paper",
        help="which book to run against (default: paper). 'alpaca' reads the "
        "real account and reuses the newest receipt as the reconciliation "
        "reference; it stays read-only until ALPACA_TRADING_ENABLED=1",
    )
    parser.add_argument(
        "--universe",
        choices=("fixed", "pit"),
        default="fixed",
        help="fixed: the --tickers list for the whole backtest (default). "
        "pit: the most liquid S&P 500 members as of each assessment date, "
        "rebuilt from membership history and prior dollar volume — slower "
        "to start, and the only way a backtest avoids picking today's "
        "winners. Needs the free data sources.",
    )
    parser.add_argument(
        "--universe-size",
        type=int,
        default=25,
        help="names in each month's universe with --universe pit (default 25)",
    )
    parser.add_argument(
        "--allocator",
        choices=sorted(ALLOCATOR_NAMES),
        help="CIO capital-allocation policy (default: the mandate's "
        "allocator, else static). equal_weight is a stub that ignores "
        "mandate slices and splits capital evenly",
    )
    parser.add_argument(
        "--model",
        help="LLM the investor agents reason with, e.g. claude-opus-5-5 "
        "(default: HEDGE_FUND_LLM_MODEL env, else the built-in default); quant models "
        "ignore it",
    )
    parser.add_argument("--out", help="also write the record JSON to this file")
    parser.add_argument(
        "--heartbeat",
        nargs="?",
        const="default",
        metavar="PATH",
        help="write a cycle heartbeat file (default: "
        "~/.hedge-fund/observability/heartbeat.json). Also honored via "
        "HEDGE_FUND_HEARTBEAT_PATH or HEDGE_FUND_HEARTBEAT=1",
    )
    parser.add_argument(
        "--events",
        nargs="?",
        const="default",
        metavar="PATH",
        help="append cycle start/end/error events as JSONL (default: "
        "~/.hedge-fund/observability/events.jsonl). Also honored via "
        "HEDGE_FUND_EVENTS_PATH",
    )
    args = parser.parse_args()

    if args.model:
        os.environ["HEDGE_FUND_LLM_MODEL"] = args.model

    if args.mandate is None:
        # The interactive experience is the Textual app. Import it lazily so
        # the non-interactive path never pays to load Textual.
        from hedge_fund.tui.app import HedgeFundApp

        HedgeFundApp().run()
        return

    if not args.tickers and args.universe == "pit":
        # A point-in-time universe IS the watchlist; naming one as well would
        # be a second, contradictory answer to the same question.
        universe = []
    elif not args.tickers:
        parser.error("--tickers is required with a mandate, e.g. --tickers AAPL,MSFT")
    else:
        universe = normalize_universe(args.tickers.replace(",", " ").split())

    if args.universe == "pit" and not args.backtest:
        parser.error(
            "--universe pit is a backtest option: a live or paper cycle trades "
            "the tickers you name, because the universe it may pick from has to "
            "be decided before the run, not recomputed while it is trading."
        )

    console = Console(stderr=True)  # status + summary on stderr; stdout stays pure JSON
    try:
        spec = load_spec(args.mandate)
    except ValueError as exc:
        parser.error(str(exc))
    if args.allocator is not None:
        spec = spec.model_copy(update={"allocator": args.allocator})
    # A backtest blinds the investor agents' prompts (no ticker, industry or
    # calendar dates): the LLM may have been trained on what those companies
    # did over the window, and that memory would otherwise score as skill.
    fund = Fund(spec, blind=args.backtest)

    if args.backtest:
        start = args.start or (
            _date.fromisoformat(args.date) - timedelta(weeks=_BACKTEST_WEEKS)
        ).isoformat()
        with open_data_client() as fd:
            selection: list[str] | Callable[[str], list[str]] = universe
            if args.universe == "pit":
                selection = _pit_universe(fd, start, args.date, args.universe_size)
            label = (
                f"{args.universe_size} most liquid S&P members per month"
                if args.universe == "pit"
                else ", ".join(universe)
            )
            with console.status(
                f"[cyan]{spec.name}: backtesting {start} → {args.date} "
                f"({spec.rebalance} rebalance vs {spec.benchmark}) "
                f"over {label}…",
                spinner="dots",
            ):
                result = backtest_fund(fund, start, args.date, fd, selection)
        print(result.model_dump_json(indent=2))
        if args.out:
            Path(args.out).write_text(result.model_dump_json(indent=2))
        m = result.metrics
        console.print(
            f"[bold]{spec.name}[/] {result.start} → {result.end}  ·  "
            f"{m.n_cycles} executed cycles  ·  {m.n_pending} pending proposals  ·  return {m.total_return_pct:+.1%} "
            f"vs {spec.benchmark} {m.benchmark_return_pct:+.1%}  ·  "
            f"sharpe {m.sharpe_ratio:.2f}  ·  max drawdown {m.max_drawdown_pct:.1%}"
        )
        return

    receipts = ensure_mandates_dir()
    try:
        venue = open_venue(args.broker, fund_name=spec.name,
                           capital=spec.capital, receipts=receipts,
                           commission=spec.commission)
    except ValueError as exc:
        parser.error(str(exc))

    # A venue that cannot submit is a setup state, not a mid-cycle surprise:
    # refusing here beats discovering it when the first order is rejected.
    if venue.name == "alpaca" and not venue.trading_enabled:
        console.print(f"[yellow]{venue.note}[/]")
        parser.error(
            "the alpaca venue is read-only right now. To place orders set "
            "ALPACA_TRADING_ENABLED=1 (and ALPACA_LIVE_TRADING_CONFIRMED=1 as "
            "well if ALPACA_PAPER is false). Use --broker paper to rehearse "
            "without touching the account."
        )
    tone = "bold yellow" if venue.live else "dim"
    console.print(f"[{tone}]venue: {venue.label}  ·  {venue.note}[/]")
    if venue.reference is not None:
        console.print(
            f"[dim]ledger expects {venue.reference.source} · "
            f"cash ${venue.reference.cash:,.2f} · "
            f"{len(venue.reference.positions)} positions[/]"
        )

    # Journal every submission: the pipeline only writes a receipt once the
    # whole loop has filled, so without this a crash mid-execution would lose
    # the fills that already happened at the venue. `session` here is the
    # run's as-of date — the executed session lands in the receipt.
    # A real venue is priced from its newest completed close and traded *now*:
    # the next completed session is always in the future, so the backtest's
    # next-close rule would refuse to trade during every session. The in-process
    # books keep next-close, which is what "fill at the mark" already means.
    live = args.broker == "alpaca"

    journal = FileOrderJournal(journal_path(spec.name))
    broker = JournalledBroker(venue.broker, journal, fund=spec.name, session=args.date)

    with open_data_client() as fd:
        n_models = sum(len(staff) for _, staff in fund.strategies)
        with console.status(
            f"[cyan]{spec.name}: {'live' if live else 'paper'} cycle as of {args.date} — "
            f"{len(universe)} tickers x {n_models} models "
            f"across {len(fund.strategies)} strategies…",
            spinner="dots",
        ):
            observer = CycleObserver.from_env(
                events_path=args.events,
                heartbeat_path=args.heartbeat,
            )
            # The newest receipt is the ledger's claim about this account;
            # reconcile it against the venue so drift and any order still
            # working are visible before we trade on top of it.
            record = observe_cycle(
                fund, args.date, broker, fd, universe, observer=observer,
                reference=venue.reference, live=live,
            )

    receipt = save_cycle_record(record, receipts)
    print(record.model_dump_json(indent=2))
    if args.out:
        Path(args.out).write_text(record.model_dump_json(indent=2))

    if isinstance(record, PendingRunResult):
        console.print(f"[bold]{spec.name}[/] · Pending · analysis cutoff {record.as_of}")
        console.print(record.reason)
        for ticker, weight in record.proposal.final_weights.items():
            console.print(f"  {ticker}: proposed {weight:+.2%}")
        return

    for sr in record.strategies:
        abstained = sum(1 for s in sr.signals if s.metadata.get("abstained") is True)
        console.print(
            f"[dim]  {sr.name} ({sr.slice:.0%} of capital): "
            f"{len(sr.signals)} signals ({abstained} abstained)[/]"
        )
    n_signals = sum(len(sr.signals) for sr in record.strategies)
    console.print(
        f"[bold]{spec.name}[/] · initial {record.as_of} · refreshed "
        f"{record.refreshed_assessment.as_of} · executed {record.execution_as_of}  ·  "
        f"{len(record.strategies)} strategies  ·  {n_signals} signals  ·  "
        f"{len(record.clamps)} risk clamps  ·  "
        f"{len(record.orders)} orders  ·  NAV ${record.nav:,.2f}"
    )
    if record.reconciliation is not None:
        tone = "yellow" if not record.reconciliation.clean else "dim"
        console.print(f"[{tone}]reconciled: {record.reconciliation.summary}[/]")
    if record.skipped:
        console.print(f"[dim]skipped: {', '.join(s.ticker for s in record.skipped)}[/]")
    if record.dropped:
        console.print(
            "[dim]dropped: "
            + ", ".join(f"{d.model}@{d.ticker} ({d.reason})" for d in record.dropped)
            + "[/]"
        )
    if record.slippage:
        # The cost a backtest cannot see: what the venue charged against the
        # close the fund sized on. Called out even when it is small, because
        # the whole point of pricing live is to stop hiding it.
        total = sum(s.notional for s in record.slippage)
        worst = max(record.slippage, key=lambda s: abs(s.notional))
        tone = "yellow" if abs(total) > 0 else "dim"
        console.print(
            f"[{tone}]slippage vs the pricing close: ${total:+,.2f} over "
            f"{len(record.slippage)} fills · worst {worst.ticker} "
            f"${worst.notional:+,.2f} ({worst.per_share:+.4f}/share)[/]"
        )
    console.print(f"[dim]saved {receipt}[/]")
    counts = journal_summary(journal.entries())["by_event"]
    if counts:
        console.print(
            f"[dim]orders journaled: {counts.get('fill', 0)} filled, "
            f"{counts.get('error', 0)} errored  ·  {journal.path}[/]"
        )


if __name__ == "__main__":
    main()
