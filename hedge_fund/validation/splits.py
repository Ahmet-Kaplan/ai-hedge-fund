"""Time-series splits that never train on the future.

walk_forward       rolling or expanding train window, test strictly after it
purged_kfold       k contiguous test folds; training drops observations whose
                   label window overlaps the test fold (purge) and an embargo
                   after it
cpcv               combinatorial purged CV: choose k of n contiguous groups as
                   test, purge/embargo around each, yielding many backtest paths
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Split:
    train: tuple[int, ...]
    test: tuple[int, ...]


def walk_forward(n: int, *, train: int, test: int, step: int | None = None, expanding: bool = False,
                 gap: int = 0) -> list[Split]:
    """Indices over n ordered observations. `gap` sessions separate train from test."""
    if min(n, train, test) <= 0:
        raise ValueError("n, train and test must be positive")
    step = step or test
    out, start = [], 0
    while True:
        tr_end = start + train
        te_start, te_end = tr_end + gap, tr_end + gap + test
        if te_end > n:
            break
        tr_start = 0 if expanding else start
        out.append(Split(tuple(range(tr_start, tr_end)), tuple(range(te_start, te_end))))
        start += step
    return out


def _purge(train: set[int], test: list[int], label_horizon: int, embargo: int, n: int) -> set[int]:
    lo, hi = min(test), max(test)
    # a training label at i covers [i, i + label_horizon]; drop if it reaches the test block
    purged = {i for i in train if not (i + label_horizon < lo or i > hi + embargo)}
    return train - purged


def purged_kfold(n: int, k: int, *, label_horizon: int = 0, embargo: int = 0) -> list[Split]:
    if k < 2 or k > n:
        raise ValueError("need 2 <= k <= n")
    bounds = [round(i * n / k) for i in range(k + 1)]
    out = []
    for f in range(k):
        test = list(range(bounds[f], bounds[f + 1]))
        train = _purge(set(range(n)) - set(test), test, label_horizon, embargo, n)
        out.append(Split(tuple(sorted(train)), tuple(test)))
    return out


def cpcv(n: int, n_groups: int, k_test: int, *, label_horizon: int = 0, embargo: int = 0) -> list[Split]:
    if not 1 <= k_test < n_groups <= n:
        raise ValueError("need 1 <= k_test < n_groups <= n")
    bounds = [round(i * n / n_groups) for i in range(n_groups + 1)]
    groups = [list(range(bounds[g], bounds[g + 1])) for g in range(n_groups)]
    out = []
    for combo in combinations(range(n_groups), k_test):
        test = sorted(i for g in combo for i in groups[g])
        train = set(range(n)) - set(test)
        for g in combo:                     # purge/embargo around each test group separately
            train = _purge(train, groups[g], label_horizon, embargo, n)
        out.append(Split(tuple(sorted(train)), tuple(test)))
    return out
