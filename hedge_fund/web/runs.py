"""Simulation runs: start one, watch it, read the result.

A backtest is minutes of blocking work and hundreds of LLM calls, so it
cannot happen inside a request. Runs go to a single background worker and the
browser polls for progress.

The worker pool is deliberately one thread wide. Two concurrent backtests
would race on the shared data cache and double the inference bill for no
benefit, and the single-slot design is what lets the container run at exactly
one replica — which is in turn what keeps the paper funds' hash-chained
ledgers safe from concurrent writers.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from hedge_fund.backtesting import backtest_fund
from hedge_fund.data.cached import CachedDataClient
from hedge_fund.data.client import FDClient
from hedge_fund.fund import Fund, FundSpec
from hedge_fund.paths import RESEARCH_DIR

log = logging.getLogger(__name__)

QUEUED, RUNNING, SUCCEEDED, FAILED = "queued", "running", "succeeded", "failed"
TERMINAL = (SUCCEEDED, FAILED)


@dataclass
class Run:
    """One simulation, from submission to result."""

    id: str
    fund: str
    universe: list[str]
    start: str
    end: str
    capital: float
    rebalance: str
    requested_by: str
    status: str = QUEUED
    submitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    sessions_done: int = 0
    sessions_total: int = 0
    cycles_done: int = 0
    last_nav: float | None = None
    error: str | None = None
    result_file: str | None = None

    @property
    def percent(self) -> int:
        if self.status in TERMINAL:
            return 100
        if not self.sessions_total:
            return 0
        return min(100, round(100 * self.sessions_done / self.sessions_total))

    def as_dict(self) -> dict:
        return {
            "id": self.id, "fund": self.fund, "status": self.status,
            "percent": self.percent, "sessions_done": self.sessions_done,
            "sessions_total": self.sessions_total, "cycles_done": self.cycles_done,
            "last_nav": self.last_nav, "error": self.error,
            "terminal": self.status in TERMINAL,
        }


class Runs:
    """In-process registry of runs, backed by result files on disk.

    Only the live progress is in memory; a finished run's result is a JSON
    file under research/, so a container restart loses the progress bars but
    never a completed simulation.
    """

    def __init__(self, research_dir: Path = RESEARCH_DIR) -> None:
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")
        self.research_dir = Path(research_dir)

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def recent(self, limit: int = 25) -> list[Run]:
        with self._lock:
            runs = sorted(self._runs.values(), key=lambda r: r.submitted_at, reverse=True)
        return runs[:limit]

    def active(self) -> int:
        with self._lock:
            return sum(1 for r in self._runs.values() if r.status in (QUEUED, RUNNING))

    def submit(self, spec: FundSpec, universe: list[str], start: str, end: str, requested_by: str) -> Run:
        run = Run(
            id=uuid.uuid4().hex[:12], fund=spec.name, universe=list(universe),
            start=start, end=end, capital=spec.capital, rebalance=spec.rebalance,
            requested_by=requested_by,
        )
        with self._lock:
            self._runs[run.id] = run
        self._pool.submit(self._execute, run, spec, universe, start, end)
        return run

    def _execute(self, run: Run, spec: FundSpec, universe: list[str], start: str, end: str) -> None:
        run.status = RUNNING
        try:
            # blind=True for the same reason the CLI does it: the agents must
            # not score on remembering what these companies actually did.
            fund = Fund(spec, blind=True)
            with FDClient() as raw:
                data = CachedDataClient(raw)
                result = backtest_fund(
                    fund, start, end, data, universe,
                    on_cycle=lambda i, n, _c: setattr(run, "cycles_done", i + 1),
                    on_valuation=lambda i, n, v: self._progress(run, i, n, v),
                )
            self.research_dir.mkdir(parents=True, exist_ok=True)
            path = self.research_dir / f"{spec.name}-{result.start}-{result.end}-{run.id}.json"
            path.write_text(result.model_dump_json(indent=2))
            run.result_file = path.name
            run.status = SUCCEEDED
        except Exception:
            # The detail goes to the logs under the run id; the browser gets
            # the id alone. Upstream errors routinely quote account state and
            # request parameters, which is not something to render in a page.
            log.exception("simulation %s failed", run.id)
            run.error = run.id
            run.status = FAILED

    @staticmethod
    def _progress(run: Run, i: int, n: int, valuation) -> None:
        run.sessions_done = i + 1
        run.sessions_total = n
        run.last_nav = valuation.nav

    def result_path(self, run: Run) -> Path | None:
        if not run.result_file:
            return None
        path = self.research_dir / run.result_file
        return path if path.is_file() else None
