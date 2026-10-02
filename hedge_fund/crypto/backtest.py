"""Daily crypto backtest: a weighted basket, optionally gated by a trend rule.

Prices are UTC daily closes. Trading happens only on weekdays (the live run
does), at that day's close, using the rule's decision from the previous
day's close, so a weekend signal executes on Monday and nothing looks ahead.
The basket rebalances to target on each Monday and whenever a rule decision
changes. Fees are charged per side on traded notional.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Callable

from pydantic import BaseModel

Rule = Callable[[list[float]], int]


class Metrics(BaseModel):
    total_return: float
    annualized_return: float
    sharpe: float          # daily, × √365
    max_drawdown: float


class CryptoBacktest(BaseModel):
    dates: list[str]
    nav: list[float]
    fees: float
    traded: dict[str, float]          # date → traded notional

    def trades_on(self, day: str) -> float:
        return self.traded.get(day, 0.0)


def run_backtest(
    closes: dict[str, dict[str, float]], weights: dict[str, float], rule: Rule | None, *,
    fee_bps: float, start: str, end: str, capital: float = 10_000.0,
) -> CryptoBacktest:
    symbols = sorted(weights)
    all_days = sorted(set.intersection(*(set(closes[s]) for s in symbols)))
    days = [d for d in all_days if start <= d <= end]
    history = {s: [closes[s][d] for d in all_days] for s in symbols}
    index = {d: i for i, d in enumerate(all_days)}
    fee = fee_bps / 10_000
    units = dict.fromkeys(symbols, 0.0)
    cash = capital
    held_exposure: dict[str, int] | None = None
    nav: list[float] = []
    traded: dict[str, float] = {}
    total_fees = 0.0
    for day in days:
        i = index[day]
        if date.fromisoformat(day).weekday() < 5:          # weekdays only, like the live run
            exposure = {s: (1 if rule is None else (rule(history[s][:i]) if i > 0 else 0)) for s in symbols}
            if held_exposure is None or exposure != held_exposure or date.fromisoformat(day).weekday() == 0:
                value = cash + sum(units[s] * history[s][i] for s in symbols)
                notional = 0.0
                for s in symbols:
                    target = weights[s] * exposure[s] * value / history[s][i]
                    notional += abs(target - units[s]) * history[s][i]
                    cash -= (target - units[s]) * history[s][i]
                    units[s] = target
                cost = notional * fee
                cash -= cost
                total_fees += cost
                if notional:
                    traded[day] = notional
                held_exposure = exposure
        nav.append(cash + sum(units[s] * history[s][i] for s in symbols))
    return CryptoBacktest(dates=days, nav=nav, fees=round(total_fees, 6), traded=traded)


def metrics(dates: list[str], nav: list[float]) -> Metrics:
    total = nav[-1] / nav[0] - 1
    years = (date.fromisoformat(dates[-1]) - date.fromisoformat(dates[0])).days / 365.25
    annual = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else 0.0
    rets = [b / a - 1 for a, b in zip(nav, nav[1:])]
    mean = sum(rets) / len(rets) if rets else 0.0
    std = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)) if len(rets) > 1 else 0.0
    peak, dd = nav[0], 0.0
    for v in nav:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak)
    return Metrics(total_return=total, annualized_return=annual,
                   sharpe=(mean / std * math.sqrt(365)) if std > 0 else 0.0, max_drawdown=dd)


def split_metrics(result: CryptoBacktest, split: str) -> dict[str, Metrics]:
    """Full-window metrics plus each half, split at `split` (first half ends on it)."""
    h1 = [i for i, d in enumerate(result.dates) if d <= split]
    h2 = [i for i, d in enumerate(result.dates) if d > split]
    pick = lambda idx: ([result.dates[i] for i in idx], [result.nav[i] for i in idx])
    return {"full": metrics(result.dates, result.nav), "h1": metrics(*pick(h1)), "h2": metrics(*pick(h2))}


def passes_bar(candidate: dict[str, Metrics], core: dict[str, Metrics]) -> bool:
    """Spec §3: Sharpe beats the core in both halves, and full-window drawdown ≥ 25% smaller."""
    return (candidate["h1"].sharpe > core["h1"].sharpe and candidate["h2"].sharpe > core["h2"].sharpe
            and candidate["full"].max_drawdown <= 0.75 * core["full"].max_drawdown)
