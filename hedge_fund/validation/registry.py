"""Append-only experiment registry with a hash chain.

Every candidate evaluation is one JSON line: what was tested (spec hash),
on which code (git commit) and data (data hash), where (window/stage), and
the result. Each record carries the hash of the previous one, so edits or
deletions are detectable (`verify()`). The registry is also the trial
counter for multiple-testing corrections: `n_trials(family)` counts every
configuration ever evaluated in a family, successful or not.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]


class RegistryTampered(RuntimeError):
    pass


class ExperimentRegistry:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def _last_hash(self) -> str:
        recs = self.records()
        return recs[-1]["record_hash"] if recs else GENESIS

    def record(self, *, family: str, spec: dict, stage: str, metrics: dict, code_commit: str = "",
               data_hash: str = "", window: tuple[str, str] | None = None, notes: str = "") -> dict:
        body = {
            "family": family, "spec": spec, "spec_hash": spec_hash(spec), "stage": stage,
            "metrics": metrics, "code_commit": code_commit, "data_hash": data_hash,
            "window": list(window) if window else None, "notes": notes,
            "recorded_at": datetime.now(timezone.utc).isoformat(), "prev_hash": self._last_hash(),
        }
        body["record_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(body, sort_keys=True, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return body

    def verify(self) -> None:
        prev = GENESIS
        for i, rec in enumerate(self.records()):
            h = rec.get("record_hash")
            body = {k: v for k, v in rec.items() if k != "record_hash"}
            if rec.get("prev_hash") != prev or hashlib.sha256(
                    json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() != h:
                raise RegistryTampered(f"record {i} does not match the chain")
            prev = h

    def n_trials(self, family: str) -> int:
        return len({r["spec_hash"] for r in self.records() if r["family"] == family})

    def history(self, spec_hash_: str) -> list[dict]:
        return [r for r in self.records() if r["spec_hash"] == spec_hash_]
