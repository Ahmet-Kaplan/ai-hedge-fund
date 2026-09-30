"""Locked holdout: evaluated once per candidate, never used for tuning.

The holdout window is declared up front. Research code gets the
development window (everything before the holdout, minus an embargo) and
`check_development_window` refuses any window that touches the holdout.
`evaluate` runs a candidate on the holdout at most once, records the
outcome in the experiment registry, and refuses:

  - a second evaluation of the same candidate (spec hash), and
  - any candidate whose declared parent failed the holdout — iterating on a
    failed configuration against the same holdout is tuning on the holdout.
"""

from __future__ import annotations

from typing import Callable

from hedge_fund.validation.registry import ExperimentRegistry, spec_hash

STAGE = "locked_holdout"


class HoldoutViolation(RuntimeError):
    pass


class LockedHoldout:
    def __init__(self, start: str, end: str, registry: ExperimentRegistry, *, family: str,
                 embargo_days: int = 0) -> None:
        if start > end:
            raise ValueError("holdout start after end")
        self.start, self.end, self.registry, self.family = start, end, registry, family
        self.embargo_days = embargo_days

    def check_development_window(self, start: str, end: str) -> None:
        from datetime import date, timedelta
        limit = (date.fromisoformat(self.start) - timedelta(days=self.embargo_days)).isoformat()
        if end >= limit or start >= self.start:
            raise HoldoutViolation(f"window {start}..{end} reaches the locked holdout ({self.start}..{self.end}, "
                                   f"embargo {self.embargo_days}d)")

    def _outcomes(self) -> dict[str, bool]:
        return {r["spec_hash"]: bool(r["metrics"].get("passed"))
                for r in self.registry.records() if r["family"] == self.family and r["stage"] == STAGE}

    def evaluate(self, spec: dict, run: Callable[[str, str], dict], *, passed: Callable[[dict], bool],
                 parent_spec: dict | None = None, code_commit: str = "", data_hash: str = "") -> dict:
        outcomes = self._outcomes()
        h = spec_hash(spec)
        if h in outcomes:
            raise HoldoutViolation(f"candidate {h} was already evaluated on the locked holdout")
        if parent_spec is not None and outcomes.get(spec_hash(parent_spec)) is False:
            raise HoldoutViolation("parent failed the locked holdout; derived candidates need a new holdout")
        result = dict(run(self.start, self.end))
        result["passed"] = bool(passed(result))
        self.registry.record(family=self.family, spec=spec, stage=STAGE, metrics=result,
                             code_commit=code_commit, data_hash=data_hash, window=(self.start, self.end),
                             notes=f"parent={spec_hash(parent_spec) if parent_spec else ''}")
        return result
