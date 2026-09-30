"""Overfitting-aware statistics.

probabilistic_sharpe   P(true SR > benchmark SR) given sample length, skew, kurtosis
                       (Bailey & Lopez de Prado, 2012)
expected_max_sharpe    expected maximum SR among N independent trials of zero skill
deflated_sharpe        PSR against that expected maximum — penalizes multiple testing
                       (Bailey & Lopez de Prado, 2014)
pbo                    probability of backtest overfitting via combinatorially
                       symmetric cross-validation (Bailey et al., 2016)
block_bootstrap        stationary-ish moving-block bootstrap of a return series
monte_carlo_trades     reshuffled trade sequences -> drawdown distribution

Sharpe ratios here are per-period (not annualized) unless stated.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy import stats as sps

EULER = 0.5772156649015329


def sharpe(returns) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe(returns, benchmark_sr: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 3:
        return 0.0
    sr = sharpe(r)
    g3 = float(sps.skew(r, bias=False))
    g4 = float(sps.kurtosis(r, fisher=False, bias=False))
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(sps.norm.cdf((sr - benchmark_sr) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] of n_trials unskilled strategies whose SR estimates have this variance."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    z1 = sps.norm.ppf(1 - 1 / n_trials)
    z2 = sps.norm.ppf(1 - 1 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * ((1 - EULER) * z1 + EULER * z2))


def deflated_sharpe(returns, n_trials: int, sr_variance: float | None = None) -> float:
    """PSR against the expected max SR of `n_trials` trials.

    `sr_variance` is the variance of the SR estimates across all trials that
    were run; when unknown, the sampling variance of one SR (1/(n-1)) is used.
    """
    r = np.asarray(returns, dtype=float)
    var = sr_variance if sr_variance is not None else 1.0 / max(len(r) - 1, 1)
    return probabilistic_sharpe(r, expected_max_sharpe(n_trials, var))


def pbo(performance: np.ndarray, n_splits: int = 8) -> float:
    """Probability of backtest overfitting.

    performance: T x N matrix — per-period returns of N candidate configs.
    Rows are cut into n_splits blocks; for every half/half combination the
    best in-sample config's out-of-sample rank is recorded. PBO is the share
    of combinations where that rank falls below the median.
    """
    m = np.asarray(performance, dtype=float)
    t, n = m.shape
    if n < 2 or n_splits % 2 or n_splits < 2 or t < n_splits:
        raise ValueError("need >= 2 configs, an even n_splits >= 2 and at least n_splits rows")
    blocks = np.array_split(np.arange(t), n_splits)
    logits = []
    for combo in combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        is_sr = np.array([sharpe(m[is_idx, j]) for j in range(n)])
        oos_sr = np.array([sharpe(m[oos_idx, j]) for j in range(n)])
        best = int(np.argmax(is_sr))
        rank = (sps.rankdata(oos_sr)[best]) / (n + 1)
        logits.append(math.log(rank / (1 - rank)))
    return float(np.mean(np.array(logits) <= 0))


def block_bootstrap(returns, *, n_samples: int = 1000, block: int = 20, seed: int = 0) -> np.ndarray:
    """Resampled return paths (n_samples x len(returns)) from moving blocks."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n == 0:
        raise ValueError("empty return series")
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_samples, math.ceil(n / block)))
    paths = np.stack([np.concatenate([r[s:s + block] for s in row])[:n] for row in starts])
    return paths


def bootstrap_sharpe_ci(returns, *, alpha: float = 0.05, **kw) -> tuple[float, float]:
    paths = block_bootstrap(returns, **kw)
    srs = np.array([sharpe(p) for p in paths])
    return float(np.quantile(srs, alpha / 2)), float(np.quantile(srs, 1 - alpha / 2))


def monte_carlo_trades(trade_pnls, *, capital: float, n_samples: int = 1000, seed: int = 0) -> dict[str, float]:
    """Reshuffle trade order; report the distribution of max drawdown."""
    pnl = np.asarray(trade_pnls, dtype=float)
    rng = np.random.default_rng(seed)
    dds = []
    for _ in range(n_samples):
        path = capital + np.cumsum(rng.permutation(pnl))
        path = np.r_[capital, path]
        peak = np.maximum.accumulate(path)
        dds.append(float(np.max(1 - path / peak)))
    dds = np.array(dds)
    return {"max_drawdown_median": float(np.median(dds)), "max_drawdown_p95": float(np.quantile(dds, 0.95)),
            "probability_of_ruin": float(np.mean(dds >= 0.5))}
