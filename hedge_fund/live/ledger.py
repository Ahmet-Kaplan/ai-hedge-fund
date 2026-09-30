"""The paper fund's books on disk — every plan, fill, and daily NAV.

One directory per fund under ~/.hedge-fund/paper/. Plain JSON and CSV so a
human can read the track record without the app. Alpaca, not this ledger, is
the source of truth for positions and cash; the ledger is the audit trail and
the NAV history the drawdown breaker and the report read.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import BaseModel

from hedge_fund.paths import PAPER_DIR


class NavRow(BaseModel):
    """The fund at one session's close."""

    date: str
    equity: float
    cash: float
    long_exposure: float
    short_exposure: float
    gross: float                      # (long + short) / equity
    benchmark_close: float


_NAV_FIELDS = list(NavRow.model_fields)


class Ledger:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @classmethod
    def for_fund(cls, fund_name: str) -> "Ledger":
        return cls(PAPER_DIR / fund_name)

    # -- halt flag ---------------------------------------------------------

    @property
    def halted_path(self) -> Path:
        return self.root / "HALTED"

    def is_halted(self) -> bool:
        return self.halted_path.exists()

    def halt(self, reason: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.halted_path.write_text(reason + "\n")

    def resume(self) -> bool:
        """Clear the halt flag; True if it was set."""
        if not self.is_halted():
            return False
        self.halted_path.unlink()
        return True

    # -- plans and fills -----------------------------------------------------

    def write_plan(self, session: str, payload: dict, *, dry_run: bool = False) -> Path:
        return self._write_json("plans", f"{session}.dryrun.json" if dry_run else f"{session}.json", payload)

    def read_plan(self, session: str) -> dict | None:
        return self._read_json("plans", f"{session}.json")

    def plan_sessions(self) -> list[str]:
        """Sessions with a real (not dry-run) plan, oldest first."""
        return self._sessions("plans")

    def write_fills(self, session: str, payload: dict) -> Path:
        return self._write_json("fills", f"{session}.json", payload)

    def read_fills(self, session: str) -> dict | None:
        return self._read_json("fills", f"{session}.json")

    def fill_sessions(self) -> list[str]:
        return self._sessions("fills")

    # -- NAV -------------------------------------------------------------------

    @property
    def nav_path(self) -> Path:
        return self.root / "nav.csv"

    def upsert_nav(self, row: NavRow) -> None:
        """Insert or replace the row for row.date; the file stays date-sorted."""
        rows = {r.date: r for r in self.nav_rows()}
        rows[row.date] = row
        self.root.mkdir(parents=True, exist_ok=True)
        with self.nav_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_NAV_FIELDS)
            writer.writeheader()
            for day in sorted(rows):
                writer.writerow(rows[day].model_dump())

    def nav_rows(self) -> list[NavRow]:
        if not self.nav_path.exists():
            return []
        with self.nav_path.open(newline="") as f:
            return [NavRow.model_validate(r) for r in csv.DictReader(f)]

    def peak_equity(self) -> float | None:
        rows = self.nav_rows()
        return max(r.equity for r in rows) if rows else None

    def latest_equity(self) -> float | None:
        rows = self.nav_rows()
        return rows[-1].equity if rows else None

    # -- logs ------------------------------------------------------------------

    def log_path(self, day: str, step: str) -> Path:
        logs = self.root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        return logs / f"{day}-{step}.log"

    # -- helpers ---------------------------------------------------------------

    def _write_json(self, kind: str, name: str, payload: dict) -> Path:
        directory = self.root / kind
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_text(json.dumps(payload, indent=2))
        return path

    def _read_json(self, kind: str, name: str) -> dict | None:
        path = self.root / kind / name
        return json.loads(path.read_text()) if path.exists() else None

    def _sessions(self, kind: str) -> list[str]:
        return sorted(p.stem for p in (self.root / kind).glob("*.json") if "." not in p.stem)
