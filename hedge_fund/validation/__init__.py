"""Validation framework: time-series splits, locked holdout, experiment registry,
overfitting statistics (PSR, Deflated Sharpe, PBO, bootstrap) and fixed gates."""

from hedge_fund.validation.gates import GateConfig, GateResult, evaluate_gates, load_gates
from hedge_fund.validation.holdout import HoldoutViolation, LockedHoldout
from hedge_fund.validation.registry import ExperimentRegistry, RegistryTampered, spec_hash
from hedge_fund.validation.splits import Split, cpcv, purged_kfold, walk_forward
from hedge_fund.validation.stats import (
    block_bootstrap, bootstrap_sharpe_ci, deflated_sharpe, expected_max_sharpe, monte_carlo_trades, pbo,
    probabilistic_sharpe,
)

__all__ = [
    "ExperimentRegistry", "GateConfig", "GateResult", "HoldoutViolation", "LockedHoldout", "RegistryTampered",
    "Split", "block_bootstrap", "bootstrap_sharpe_ci", "cpcv", "deflated_sharpe", "evaluate_gates",
    "expected_max_sharpe", "load_gates", "monte_carlo_trades", "pbo", "probabilistic_sharpe", "purged_kfold",
    "spec_hash", "walk_forward",
]
