"""Ticker -> SEC company identity, as of a date.

SEC keys everything by CIK; our universe is keyed by ticker, and tickers
are delisted, renamed, recycled and moved to successor symbols. Resolution:

1. `ticker_history.csv` (bundled, curated) — delisted names, successor
   symbols, registrants with a predecessor CIK (Alphabet <- Google Inc.),
   and names known to have no SEC fundamentals (FDIC filers). Among the
   rows for a ticker, the one with the latest start on or before the date
   wins; before any start, the earliest row is used.
2. SEC's current company_tickers.json — only listings trading today, so it
   is consulted after the history file.

A row with an empty CIK means "known, but no SEC fundamentals": callers
return no data instead of guessing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

HISTORY_PATH = Path(__file__).with_name("ticker_history.csv")


def normalize_ticker(ticker: str) -> str:
    """Upper-case, share-class separator as '.', so BRK-B == BRK.B."""
    return ticker.strip().upper().replace("-", ".").replace("/", ".")


@dataclass(frozen=True)
class Identity:
    ticker: str
    cik: int | None
    start_date: str | None = None
    end_date: str | None = None
    predecessor_cik: int | None = None
    predecessor_until: str | None = None
    note: str = ""
    source: str = "history"

    @property
    def has_fundamentals(self) -> bool:
        return self.cik is not None

    def is_active(self, today: str) -> bool:
        return self.end_date is None or self.end_date >= today

    def lineage(self) -> list[tuple[int, str | None]]:
        """(cik, filed_until) for this registrant and its predecessor."""
        out: list[tuple[int, str | None]] = []
        if self.cik is not None:
            out.append((self.cik, None))
        if self.predecessor_cik is not None:
            out.append((self.predecessor_cik, self.predecessor_until))
        return out


def _int(value: str) -> int | None:
    return int(value) if value and value.strip() else None


def _str(value: str) -> str | None:
    return value.strip() or None if value is not None else None


def load_history(path: Path = HISTORY_PATH) -> dict[str, list[Identity]]:
    history: dict[str, list[Identity]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            ident = Identity(
                ticker=normalize_ticker(row["ticker"]), cik=_int(row["cik"]),
                start_date=_str(row["start_date"]), end_date=_str(row["end_date"]),
                predecessor_cik=_int(row["predecessor_cik"]), predecessor_until=_str(row["predecessor_until"]),
                note=row.get("note", "").strip(),
            )
            history.setdefault(ident.ticker, []).append(ident)
    return history


class TickerResolver:
    """Resolve tickers against the curated history, then SEC's current map."""

    def __init__(self, current: dict[str, int] | None = None, history: dict[str, list[Identity]] | None = None) -> None:
        self._history = load_history() if history is None else history
        self._current = {normalize_ticker(t): int(c) for t, c in (current or {}).items()}

    def known_in_history(self, ticker: str) -> bool:
        return normalize_ticker(ticker) in self._history

    def resolve(self, ticker: str, as_of: str | None = None) -> Identity | None:
        t = normalize_ticker(ticker)
        rows = self._history.get(t)
        if rows:
            if as_of is None:
                return max(rows, key=lambda r: r.start_date or "")
            started = [r for r in rows if (r.start_date or "") <= as_of]
            if started:
                return max(started, key=lambda r: r.start_date or "")
            return min(rows, key=lambda r: r.start_date or "")
        cik = self._current.get(t)
        if cik is not None:
            return Identity(ticker=t, cik=cik, source="sec_current")
        return None


# ---------------------------------------------------------------------------
# Multi-class registrants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShareClassing:
    """Economic units per share of each class, by date, and which class a
    ticker trades. Cover-page counts are converted to the ticker's class:
    shares_in_ticker_units = sum(n_c * units_c) / units_ticker."""

    ticker_class: dict[str, str]
    # (effective_from, {class: units}); the last entry on or before a date applies
    schedule: tuple[tuple[str, dict[str, float]], ...]

    def units(self, on: str) -> dict[str, float]:
        applicable = [u for start, u in self.schedule if start <= on]
        return applicable[-1] if applicable else self.schedule[0][1]


SHARE_CLASSES: dict[int, ShareClassing] = {
    # Berkshire Hathaway: a Class A share converts into 30 Class B shares
    # until the 50-for-1 Class B split effective 2010-01-21, 1,500 after.
    1067983: ShareClassing(
        ticker_class={"BRK.A": "A", "BRK.B": "B"},
        schedule=(("1900-01-01", {"A": 30.0, "B": 1.0}), ("2010-01-21", {"A": 1500.0, "B": 1.0})),
    ),
}


def shares_in_ticker_units(cik: int | None, ticker: str, by_class: dict[str | None, float], on: str) -> float | None:
    """Convert per-class share counts into units of *ticker*'s class.

    Registrants without a ShareClassing entry are treated as economically
    equal classes (Alphabet A/B/C) and summed.
    """
    if not by_class:
        return None
    classing = SHARE_CLASSES.get(cik) if cik is not None else None
    if classing is None:
        return float(sum(by_class.values()))
    units = classing.units(on)
    mine = classing.ticker_class.get(normalize_ticker(ticker))
    if mine is None or None in by_class or any(c not in units for c in by_class):
        return None  # unlabelled or unknown class: refuse to guess
    return sum(n * units[c] for c, n in by_class.items()) / units[mine]
