"""Local market database — prices, SEC facts and filings, and finished runs.

One SQLite file (~/.hedge-fund/market.db). Everything fetched from a free
source lands here once and is read from here after, so a run never pays for
or waits on data it already has. No network code lives here; the sources in
alpaca_prices.py and sec.py fill it, and FreeDataClient reads it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from hedge_fund.data.models import Price

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER,
    raw_close REAL,   -- as traded that day (not split-adjusted), for market caps
    PRIMARY KEY (ticker, date));
CREATE TABLE IF NOT EXISTS price_sync (ticker TEXT PRIMARY KEY, first_date TEXT, synced_through TEXT);
CREATE TABLE IF NOT EXISTS sec_companies (
    ticker TEXT PRIMARY KEY, cik TEXT, name TEXT, sic INTEGER, sic_description TEXT);
CREATE TABLE IF NOT EXISTS sec_facts (
    cik TEXT, concept TEXT, unit TEXT, start TEXT, end TEXT, value REAL,
    fy INTEGER, fp TEXT, form TEXT, filed TEXT, accn TEXT,
    PRIMARY KEY (cik, concept, unit, start, end, accn));
CREATE INDEX IF NOT EXISTS sec_facts_lookup ON sec_facts (cik, concept, filed);
CREATE TABLE IF NOT EXISTS sec_filings (
    cik TEXT, accn TEXT PRIMARY KEY, form TEXT, filed TEXT, report_date TEXT, items TEXT);
CREATE INDEX IF NOT EXISTS sec_filings_cik ON sec_filings (cik, form);
CREATE TABLE IF NOT EXISTS sec_sync (cik TEXT PRIMARY KEY, facts_fetched_at TEXT, submissions_fetched_at TEXT);
CREATE TABLE IF NOT EXISTS sp500_members (ticker TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS sp500_changes (date TEXT, added TEXT, removed TEXT);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_key TEXT PRIMARY KEY, label TEXT, status TEXT, started_at TEXT, finished_at TEXT,
    result_json TEXT, error TEXT);
"""


@dataclass(frozen=True)
class Fact:
    """One XBRL fact from an SEC filing. `start` is None for balance-sheet (instant) facts."""

    concept: str
    unit: str
    start: str | None
    end: str
    value: float
    fy: int | None
    fp: str | None
    form: str
    filed: str
    accn: str


class MarketStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, timeout=30)   # TUI workers may write concurrently
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    # -- prices ----------------------------------------------------------------

    def upsert_prices(self, ticker: str, bars: list[Price], raw_closes: dict[str, float] | None = None) -> None:
        """Store adjusted bars; `raw_closes` (date → as-traded close) goes alongside."""
        raw_closes = raw_closes or {}
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?,?)",
                [(ticker, b.time[:10], b.open, b.high, b.low, b.close, b.volume, raw_closes.get(b.time[:10]))
                 for b in bars],
            )

    def delete_prices(self, ticker: str) -> None:
        with self._db:
            self._db.execute("DELETE FROM prices WHERE ticker=?", (ticker,))
            self._db.execute("DELETE FROM price_sync WHERE ticker=?", (ticker,))

    def raw_close(self, ticker: str, on_or_before: str, max_age_days: int = 10) -> float | None:
        """The as-traded close on the latest session at or before a date."""
        row = self._db.execute(
            "SELECT raw_close FROM prices WHERE ticker=? AND date <= ? AND date >= date(?, ?) "
            "AND raw_close IS NOT NULL ORDER BY date DESC LIMIT 1",
            (ticker, on_or_before, on_or_before, f"-{max_age_days} days"),
        ).fetchone()
        return row[0] if row else None

    def prices(self, ticker: str, start: str, end: str) -> list[Price]:
        rows = self._db.execute(
            "SELECT * FROM prices WHERE ticker=? AND date BETWEEN ? AND ? ORDER BY date",
            (ticker, start, end),
        ).fetchall()
        return [Price(open=r["open"], high=r["high"], low=r["low"], close=r["close"],
                      volume=r["volume"], time=f"{r['date']}T00:00:00Z") for r in rows]

    def price_range(self, ticker: str) -> tuple[str, str] | None:
        row = self._db.execute("SELECT first_date, synced_through FROM price_sync WHERE ticker=?", (ticker,)).fetchone()
        return (row[0], row[1]) if row else None

    def set_price_range(self, ticker: str, first: str, through: str) -> None:
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO price_sync VALUES (?,?,?)", (ticker, first, through))

    # -- SEC -------------------------------------------------------------------

    def upsert_company(self, ticker: str, cik: str, name: str | None, sic: int | None, sic_description: str | None) -> None:
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO sec_companies VALUES (?,?,?,?,?)",
                             (ticker, cik, name, sic, sic_description))

    def company(self, ticker: str) -> dict | None:
        row = self._db.execute("SELECT * FROM sec_companies WHERE ticker=?", (ticker,)).fetchone()
        return dict(row) if row else None

    def replace_facts(self, cik: str, facts: list[Fact]) -> None:
        with self._db:
            self._db.execute("DELETE FROM sec_facts WHERE cik=?", (cik,))
            self._db.executemany(
                "INSERT OR REPLACE INTO sec_facts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(cik, f.concept, f.unit, f.start or "", f.end, f.value, f.fy, f.fp, f.form, f.filed, f.accn)
                 for f in facts],
            )

    def facts(self, cik: str, concepts: list[str], filed_lte: str) -> dict[str, list[Fact]]:
        """Facts per concept that were public by `filed_lte`, oldest filing first."""
        out: dict[str, list[Fact]] = {c: [] for c in concepts}
        marks = ",".join("?" * len(concepts))
        rows = self._db.execute(
            f"SELECT * FROM sec_facts WHERE cik=? AND concept IN ({marks}) AND filed <= ? ORDER BY filed, end",
            (cik, *concepts, filed_lte),
        ).fetchall()
        for r in rows:
            out[r["concept"]].append(Fact(
                concept=r["concept"], unit=r["unit"], start=r["start"] or None, end=r["end"],
                value=r["value"], fy=r["fy"], fp=r["fp"], form=r["form"], filed=r["filed"], accn=r["accn"],
            ))
        return out

    def replace_filings(self, cik: str, filings: list[dict]) -> None:
        with self._db:
            self._db.execute("DELETE FROM sec_filings WHERE cik=?", (cik,))
            self._db.executemany(
                "INSERT OR REPLACE INTO sec_filings VALUES (?,?,?,?,?,?)",
                [(cik, f["accn"], f["form"], f["filed"], f.get("report_date"), f.get("items", "")) for f in filings],
            )

    def filings(self, cik: str, forms: list[str]) -> list[dict]:
        marks = ",".join("?" * len(forms))
        rows = self._db.execute(
            f"SELECT accn, form, filed, report_date, items FROM sec_filings WHERE cik=? AND form IN ({marks}) ORDER BY filed",
            (cik, *forms),
        ).fetchall()
        return [dict(r) for r in rows]

    def sync_times(self, cik: str) -> tuple[str | None, str | None]:
        row = self._db.execute("SELECT facts_fetched_at, submissions_fetched_at FROM sec_sync WHERE cik=?", (cik,)).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def mark_synced(self, cik: str, *, facts_at: str | None = None, submissions_at: str | None = None) -> None:
        current_facts, current_subs = self.sync_times(cik)
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO sec_sync VALUES (?,?,?)",
                             (cik, facts_at or current_facts, submissions_at or current_subs))

    # -- S&P 500 membership ----------------------------------------------------

    def replace_sp500(self, current: list[str], changes: list[tuple], fetched_at: str) -> None:
        with self._db:
            self._db.execute("DELETE FROM sp500_members")
            self._db.execute("DELETE FROM sp500_changes")
            self._db.executemany("INSERT OR IGNORE INTO sp500_members VALUES (?)", [(t,) for t in current])
            self._db.executemany("INSERT INTO sp500_changes VALUES (?,?,?)",
                                 [(d.isoformat(), added, removed) for d, added, removed in changes])
            self._db.execute("INSERT OR REPLACE INTO meta VALUES ('sp500_fetched_at', ?)", (fetched_at,))

    def sp500(self) -> tuple[list[str], list[tuple], str | None]:
        """(current members, [(date, added, removed)] newest first, fetched_at)."""
        current = [r[0] for r in self._db.execute("SELECT ticker FROM sp500_members ORDER BY ticker")]
        changes = [(date.fromisoformat(r[0]), r[1], r[2])
                   for r in self._db.execute("SELECT date, added, removed FROM sp500_changes ORDER BY date DESC")]
        row = self._db.execute("SELECT value FROM meta WHERE key='sp500_fetched_at'").fetchone()
        return current, changes, row[0] if row else None

    # -- finished runs ---------------------------------------------------------

    def get_run(self, run_key: str) -> dict | None:
        row = self._db.execute("SELECT * FROM backtest_runs WHERE run_key=?", (run_key,)).fetchone()
        return dict(row) if row else None

    def put_run(self, run_key: str, label: str, status: str, result_json: str | None, *, error: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        existing = self.get_run(run_key)
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO backtest_runs VALUES (?,?,?,?,?,?,?)",
                (run_key, label, status, existing["started_at"] if existing else now,
                 now if status in ("done", "failed") else None, result_json, error),
            )
