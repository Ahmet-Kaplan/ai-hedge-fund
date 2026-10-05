"""The live account's books: daily NAV with deposits separated, cash flows, holdings, reviews."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import BaseModel

from hedge_fund.brokers.alpaca import CashFlow
from hedge_fund.live.ledger import Ledger
from hedge_fund.paths import LIVE_DIR

FLOW_KINDS = frozenset({"CSD", "CSW", "JNLC"})   # money in or out; dividends and fees are returns


class LiveNavRow(BaseModel):
    date: str
    equity: float
    cash: float
    net_flow: float                     # deposits − withdrawals since the previous row
    core_value: float
    satellite_value: float
    crypto_value: float = 0.0                # the crypto half (never part of the satellite)
    core_close: float
    satellite_return: float | None = None   # satellite holdings' return since the previous row


class LiveLedger(Ledger):
    """Plans, logs and JSON helpers come from the paper Ledger; NAV, flows and reviews are live-specific."""

    @classmethod
    def default(cls) -> "LiveLedger":
        return cls(LIVE_DIR / "main")

    # -- NAV ---------------------------------------------------------------
    def upsert_live_nav(self, row: LiveNavRow) -> None:
        rows = {r.date: r for r in self.live_nav_rows()}
        rows[row.date] = row
        _write_csv(self.root / "nav.csv", list(LiveNavRow.model_fields), [rows[d].model_dump() for d in sorted(rows)])

    def live_nav_rows(self) -> list[LiveNavRow]:
        return [LiveNavRow.model_validate({k: (v if v != "" else None) for k, v in r.items()})
                for r in _read_csv(self.root / "nav.csv")]

    # -- flows ---------------------------------------------------------------
    def add_flows(self, flows: list[CashFlow]) -> None:
        known = {f.id: f for f in self.flows()}
        known.update({f.id: f for f in flows})
        rows = sorted(known.values(), key=lambda f: (f.date, f.id))
        _write_csv(self.root / "flows.csv", list(CashFlow.model_fields), [f.model_dump() for f in rows])

    def flows(self) -> list[CashFlow]:
        return [CashFlow.model_validate(r) for r in _read_csv(self.root / "flows.csv")]

    def net_flow(self, after: str | None, through: str) -> float:
        return sum(f.amount for f in self.flows()
                   if f.kind in FLOW_KINDS and (after is None or f.date > after) and f.date <= through)

    def total_deposited(self) -> float:
        return sum(f.amount for f in self.flows() if f.kind in FLOW_KINDS)

    # -- holdings snapshots ----------------------------------------------------
    def save_holdings(self, session: str, holdings: dict[str, float]) -> None:
        self._write_json("holdings", f"{session}.json", holdings)

    def holdings_before(self, session: str) -> tuple[str, dict[str, float]] | None:
        earlier = [s for s in self._sessions("holdings") if s < session]
        return (earlier[-1], self._read_json("holdings", f"{earlier[-1]}.json")) if earlier else None

    # -- markers -----------------------------------------------------------------
    def dry_run_done(self) -> bool:
        return (self.root / "DRY_RUN_DONE").exists()

    def mark_dry_run_done(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "DRY_RUN_DONE").touch()

    @property
    def satellite_halt_path(self) -> Path:
        return self.root / "SATELLITE_HALTED"

    def satellite_halted(self) -> bool:
        return self.satellite_halt_path.exists()

    def halt_satellite(self, reason: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.satellite_halt_path.write_text(reason + "\n")

    def clear_satellite_halt(self) -> None:
        self.satellite_halt_path.unlink(missing_ok=True)

    # -- reviews -------------------------------------------------------------------
    def append_review(self, day: str, old_share: float, new_share: float, *,
                      paper_excess: float, live_excess: float | None, passed: bool) -> None:
        rows = _read_csv(self.root / "reviews.csv")
        rows.append({"date": day, "old_share": old_share, "new_share": new_share, "paper_excess": paper_excess,
                     "live_excess": "" if live_excess is None else live_excess, "passed": passed})
        _write_csv(self.root / "reviews.csv", ["date", "old_share", "new_share", "paper_excess", "live_excess", "passed"], rows)

    def last_review_date(self) -> str | None:
        rows = _read_csv(self.root / "reviews.csv")
        return rows[-1]["date"] if rows else None

    def satellite_excess(self, since: str) -> float | None:
        """Satellite's compounded return minus the core's, over rows after `since` that held a satellite."""
        rows = [r for r in self.live_nav_rows() if r.date >= since]
        sat, core, used = 1.0, 1.0, False
        for prev, cur in zip(rows, rows[1:]):
            if cur.satellite_return is None:
                continue
            sat *= 1 + cur.satellite_return
            core *= cur.core_close / prev.core_close
            used = True
        return sat - core if used else None

    # -- crypto limit orders -------------------------------------------------
    def crypto_limits(self) -> dict[str, dict]:
        """{pair: {"side", "tries"}} — consecutive limit attempts still waiting to fill."""
        path = self.root / "crypto_limits.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def save_crypto_limits(self, limits: dict[str, dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "crypto_limits.json").write_text(json.dumps(limits, indent=2, sort_keys=True))

    def satellite_plans(self) -> list[tuple[str, dict]]:
        return [(s, self.read_plan(s) or {}) for s in self.plan_sessions()]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
