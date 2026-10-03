"""Persistent ledger — CycleRecord receipts, write and read.

Every live-clock paper run writes a receipt (cash, positions, NAV, every
thesis). The next run loads the newest CycleRecord for that mandate and
seeds PaperBroker from the ending book, so NAV is a track record instead
of a reset to the mandate's capital.

Backtests do not read the ledger. They open a fresh SimBroker at the
mandate's capital and carry the book only across ticks inside that run.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from hedge_fund.brokers.paper import PaperBroker
from hedge_fund.pipeline.models import CycleRecord, PendingRunResult
from hedge_fund.reconciliation import LedgerReference


#: A completed run writes `{fund}-run-*.json`; a pending proposal writes
#: `{fund}-proposal-*.json`. The split is load-bearing — see `save_cycle_record`.
RUN_KIND = "run"
PROPOSAL_KIND = "proposal"


def run_receipt_paths(fund_name: str, directory: Path) -> list[Path]:
    """This mandate's `{name}-run-*.json` receipts, newest first by mtime.

    A pending proposal is deliberately absent: it is not a book. It is written
    under `{name}-proposal-*.json`, because a proposal on this path becomes
    "the newest receipt", and the next paper run then fails to load it as a
    completed record. Anyone who ran on a Saturday and ran again would hit it.
    """
    paths = list(directory.glob(f"{fund_name}-{RUN_KIND}-*.json"))
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def latest_run_receipt(fund_name: str, directory: Path) -> Path | None:
    """Newest run receipt for *fund_name*, or None if it has never run."""
    paths = run_receipt_paths(fund_name, directory)
    return paths[0] if paths else None


def load_cycle_record(
    path: Path,
    *,
    expected_fund: str | None = None,
) -> CycleRecord:
    """Load a CycleRecord. Corrupt or incompatible receipts raise.

    A backtest result sitting on a run-receipt path is incompatible — it
    is not a CycleRecord. A fund-name mismatch against *expected_fund*
    is incompatible. JSON or schema failures are corrupt. The caller must
    not skip to an older file; the newest receipt is the book.
    """
    try:
        text = path.read_text()
    except OSError as exc:
        raise ValueError(f"corrupt receipt {path}: {exc}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupt receipt {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"corrupt receipt {path}: expected a JSON object")

    if "metrics" in raw and "records" in raw:
        raise ValueError(
            f"incompatible receipt {path}: expected a CycleRecord, "
            f"got a backtest result"
        )

    try:
        record = CycleRecord.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"corrupt receipt {path}: {exc}") from exc

    if expected_fund is not None and record.fund != expected_fund:
        raise ValueError(
            f"incompatible receipt {path}: fund {record.fund!r} "
            f"does not match mandate {expected_fund!r}"
        )
    return record


def save_cycle_record(record: CycleRecord | PendingRunResult, directory: Path) -> Path:
    """Write *record* as this mandate's newest run receipt, or — for a
    proposal waiting on a completed session — as a proposal beside it.

    The two are different things and get different names. A pending proposal
    describes an intent that has not been priced yet, so it is not a reference
    book: `latest_reference` and `broker_for_run` must not find it, or the next
    run tries to reconcile against a book that was never traded.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    kind = PROPOSAL_KIND if isinstance(record, PendingRunResult) else RUN_KIND
    path = directory / f"{record.fund}-{kind}-{stamp}.json"
    path.write_text(record.model_dump_json(indent=2))
    return path


def latest_reference(fund_name: str, directory: Path) -> LedgerReference | None:
    """The newest receipt's ending book, as a reconciliation reference.

    Distinct from `broker_for_run`, which *seeds* a paper book: a live venue
    must not be seeded from a file, only reconciled against one.
    """
    path = latest_run_receipt(fund_name, directory)
    if path is None:
        return None
    return LedgerReference.from_cycle_record(load_cycle_record(path, expected_fund=fund_name))


def broker_for_run(
    fund_name: str,
    capital: float,
    directory: Path,
) -> tuple[PaperBroker, CycleRecord | None]:
    """Open a PaperBroker for a live-clock paper run of *fund_name*.

    If a prior CycleRecord exists, the broker is seeded from that ending
    book (cash + signed shares). If not, it opens at *capital*. The newest
    receipt is the only candidate — a corrupt or incompatible file raises
    rather than falling back to capital or to an older receipt.
    """
    path = latest_run_receipt(fund_name, directory)
    if path is None:
        return PaperBroker(cash=capital), None
    record = load_cycle_record(path, expected_fund=fund_name)
    return PaperBroker(cash=record.cash, positions=record.positions), record
