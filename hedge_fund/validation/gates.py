"""Validation gates — fixed, human-owned thresholds a candidate must pass.

Thresholds live in `configs/validation-gates.yaml` (reviewed by a human in
git). `GateConfig` is frozen; `load_gates` returns it with its hash so every
decision records which thresholds it was judged against. Nothing in the
research loop can edit them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "validation-gates.yaml"


class GateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_trades: int = Field(30, ge=1)
    min_oos_sharpe_annual: float = 0.5
    min_deflated_sharpe: float = Field(0.95, gt=0, lt=1)
    max_pbo: float = Field(0.5, gt=0, le=1)
    max_drawdown: float = Field(0.25, gt=0, le=1)
    min_walk_forward_positive_share: float = Field(0.6, ge=0, le=1)
    require_positive_bootstrap_sharpe_lower: bool = True

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:16]


class GateResult(BaseModel):
    passed: bool
    checks: dict[str, bool]
    values: dict[str, float]
    gates_hash: str


def load_gates(path: Path | str = DEFAULT_PATH) -> GateConfig:
    return GateConfig(**(yaml.safe_load(Path(path).read_text()) or {}))


def evaluate_gates(evidence: dict[str, float], gates: GateConfig) -> GateResult:
    """evidence keys: n_trades, oos_sharpe_annual, deflated_sharpe, pbo, max_drawdown,
    walk_forward_positive_share, bootstrap_sharpe_lower."""
    e = {k: float(v) for k, v in evidence.items()}
    checks = {
        "min_trades": e.get("n_trades", 0) >= gates.min_trades,
        "oos_sharpe": e.get("oos_sharpe_annual", float("-inf")) >= gates.min_oos_sharpe_annual,
        "deflated_sharpe": e.get("deflated_sharpe", 0.0) >= gates.min_deflated_sharpe,
        "pbo": e.get("pbo", 1.0) <= gates.max_pbo,
        "max_drawdown": e.get("max_drawdown", 1.0) <= gates.max_drawdown,
        "walk_forward": e.get("walk_forward_positive_share", 0.0) >= gates.min_walk_forward_positive_share,
    }
    if gates.require_positive_bootstrap_sharpe_lower:
        checks["bootstrap"] = e.get("bootstrap_sharpe_lower", float("-inf")) > 0
    return GateResult(passed=all(checks.values()), checks=checks, values=e, gates_hash=gates.config_hash())
