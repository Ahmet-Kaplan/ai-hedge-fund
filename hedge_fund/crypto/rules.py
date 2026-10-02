"""Pre-registered trend rules (spec §2). Fixed textbook parameters, no tuning.

Each rule takes a coin's closes, oldest first, ending on the decision day,
and returns 1 (hold) or 0 (cash). Too little history means 0.
"""

from __future__ import annotations

from typing import Callable

Rule = Callable[[list[float]], int]


def _sma(closes: list[float], n: int) -> float:
    return sum(closes[-n:]) / n


def ma100(closes: list[float]) -> int:
    """Close above its 100-day simple moving average."""
    return int(len(closes) >= 100 and closes[-1] > _sma(closes, 100))


def mom12w(closes: list[float]) -> int:
    """Positive 84-day (12-week) return."""
    return int(len(closes) >= 85 and closes[-1] > closes[-85])


def ma_cross(closes: list[float]) -> int:
    """50-day average above the 200-day average."""
    return int(len(closes) >= 200 and _sma(closes, 50) > _sma(closes, 200))


RULES: dict[str, Rule] = {"ma100": ma100, "mom12w": mom12w, "ma_cross": ma_cross}
WARMUP_DAYS = 200   # the longest rule's history need; every variant starts after it
