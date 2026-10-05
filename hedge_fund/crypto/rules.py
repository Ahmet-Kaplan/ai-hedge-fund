"""Pre-registered trend rules (spec §2). Fixed textbook parameters, no tuning.

Each rule takes a coin's closes, oldest first, ending on the decision day,
and returns its exposure: 1 (hold), 0 (cash), or a fraction (td2).
Too little history means 0. Rules with a series() method compute every day in
one pass (the backtest uses it); td1 also needs daily highs and lows.
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


class _Breakout:
    """td1 — trader.dev "A2 Price Action Breakout, lookback 20" (spec §13), long-only.

    Path-dependent: series() walks the whole history once. Needs highs and lows.
    """

    needs_ohlc = True
    lookback = 20

    def series(self, closes: list[float], highs: list[float], lows: list[float]) -> list[float]:
        out: list[float] = []
        level: float | None = None
        direction = 0
        n = self.lookback
        for t, close in enumerate(closes):
            if t >= n:
                prior_high, prior_low = max(highs[t - n:t]), min(lows[t - n:t])
                if level is None:
                    level, direction = prior_low, 1
                if direction == 1:
                    level = max(level, prior_low)
                    if close < level:
                        direction, level = -1, prior_high
                else:
                    level = min(level, prior_high)
                    if close > level:
                        direction, level = 1, prior_low
            out.append(1.0 if direction == 1 else 0.0)
        return out

    def __call__(self, closes, highs, lows) -> float:
        return self.series(closes, highs, lows)[-1] if closes else 0.0


class _TsmomVote:
    """td2 — trader.dev "S1 TSMOM vote vol-target" (spec §13), adapted long-only.

    Long when ≥ 2 of the 20/60/120-day returns are positive; size =
    min(1, 0.5 / annualized 30-day volatility), fixed at entry until exit.
    """

    needs_ohlc = False
    lookbacks = (20, 60, 120)
    target_vol, vol_len = 0.5, 30

    def series(self, closes: list[float]) -> list[float]:
        import math
        import statistics
        out: list[float] = []
        size = 0.0
        for t, close in enumerate(closes):
            if t < max(self.lookbacks) or t < self.vol_len:
                out.append(0.0)
                continue
            ups = sum(close > closes[t - n] for n in self.lookbacks)
            if ups >= 2:
                if size == 0.0:   # entry: size fixed until exit
                    rets = [math.log(closes[k] / closes[k - 1]) for k in range(t - self.vol_len + 1, t + 1)]
                    vol = statistics.pstdev(rets) * math.sqrt(365)
                    size = min(1.0, self.target_vol / max(vol, 0.01))
            else:
                size = 0.0
            out.append(size)
        return out

    def __call__(self, closes) -> float:
        return self.series(closes)[-1] if closes else 0.0


td1 = _Breakout()
td2 = _TsmomVote()

RULES: dict[str, Rule] = {"ma100": ma100, "mom12w": mom12w, "ma_cross": ma_cross, "td1": td1, "td2": td2}
WARMUP_DAYS = 200   # the longest rule's history need; every variant starts after it
