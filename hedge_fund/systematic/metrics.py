"""Performance metrics — plain formulas, each pinned by a hand-computed test.

Conventions: `returns` are simple per-period returns of the equity curve;
annualization uses `periods_per_year` (252 trading days); the risk-free rate
is an annual rate converted per period. Round-trip trade statistics come
from FIFO matching of fills, with commissions allocated to the lots they
opened and closed.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def to_returns(equity: pd.Series | list[float]) -> pd.Series:
    s = pd.Series(equity, dtype=float)
    return s.pct_change().dropna()


def total_return(equity) -> float:
    s = pd.Series(equity, dtype=float)
    return float(s.iloc[-1] / s.iloc[0] - 1)


def cagr(equity, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    s = pd.Series(equity, dtype=float)
    years = (len(s) - 1) / periods_per_year
    if years <= 0 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return 0.0
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1)


def annualized_volatility(returns, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    r = pd.Series(returns, dtype=float)
    return float(r.std(ddof=1) * math.sqrt(periods_per_year)) if len(r) > 1 else 0.0


def sharpe(returns, rf_annual: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    r = pd.Series(returns, dtype=float) - rf_annual / periods_per_year
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0


def sortino(returns, rf_annual: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Mean excess return over downside deviation (target = risk-free)."""
    r = pd.Series(returns, dtype=float) - rf_annual / periods_per_year
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)) if len(r) else 0.0
    return float(r.mean() / downside * math.sqrt(periods_per_year)) if downside > 0 else 0.0


def drawdown_series(equity) -> pd.Series:
    s = pd.Series(equity, dtype=float)
    return s / s.cummax() - 1.0


def max_drawdown(equity) -> float:
    """Largest peak-to-trough loss as a positive fraction."""
    return float(-drawdown_series(equity).min()) if len(equity) else 0.0


def calmar(equity, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    mdd = max_drawdown(equity)
    return float(cagr(equity, periods_per_year) / mdd) if mdd > 0 else 0.0


def cvar(returns, alpha: float = 0.05) -> float:
    """Expected shortfall: mean of the worst `alpha` share of returns (positive = loss)."""
    r = np.sort(pd.Series(returns, dtype=float).to_numpy())
    if not len(r):
        return 0.0
    k = max(1, int(math.floor(alpha * len(r))))
    return float(-r[:k].mean())


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundTrip:
    symbol: str
    quantity: float
    entry_session: str
    exit_session: str
    entry_price: float
    exit_price: float
    pnl: float                 # after allocated commissions


def round_trips(fills) -> list[RoundTrip]:
    """FIFO-match fills per symbol (long and short) into closed round trips."""
    lots: dict[str, deque] = {}
    out: list[RoundTrip] = []
    for f in fills:
        q = f.signed_quantity
        per_unit_fee = f.commission / f.quantity if f.quantity else 0.0
        book = lots.setdefault(f.symbol, deque())
        while q and book and (book[0][0] > 0) != (q > 0):
            open_q, open_px, open_fee, open_session = book[0]
            matched = min(abs(q), abs(open_q))
            direction = 1.0 if open_q > 0 else -1.0
            pnl = direction * matched * (f.price - open_px) - matched * (open_fee + per_unit_fee)
            out.append(RoundTrip(f.symbol, direction * matched, open_session, f.session, open_px, f.price, pnl))
            remaining = open_q - direction * matched
            if abs(remaining) < 1e-12:
                book.popleft()
            else:
                book[0] = (remaining, open_px, open_fee, open_session)
            q += direction * matched
            if abs(q) < 1e-12:
                q = 0.0
        if q:
            book.append((q, f.price, per_unit_fee, f.session))
    return out


def trade_stats(trips: list[RoundTrip]) -> dict[str, float]:
    pnl = np.array([t.pnl for t in trips], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    n = len(pnl)
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    return {
        "n_trades": float(n),
        "hit_rate": float(len(wins) / n) if n else 0.0,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "payoff_ratio": float(avg_win / abs(avg_loss)) if avg_loss else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else (math.inf if len(wins) else 0.0),
        "expectancy": float(pnl.mean()) if n else 0.0,
    }


def turnover(traded_notional: float, average_equity: float, years: float) -> float:
    """Annual one-way turnover: traded notional / average equity / years."""
    if average_equity <= 0 or years <= 0:
        return 0.0
    return float(traded_notional / average_equity / years)


def summarize(equity: pd.Series, *, fills=(), exposure: pd.Series | None = None, rf_annual: float = 0.0,
              periods_per_year: int = PERIODS_PER_YEAR) -> dict[str, float]:
    rets = to_returns(equity)
    fills = list(fills)
    traded = sum(f.quantity * f.price for f in fills)
    years = max((len(equity) - 1) / periods_per_year, 1e-9)
    out = {
        "total_return": total_return(equity),
        "cagr": cagr(equity, periods_per_year),
        "annualized_volatility": annualized_volatility(rets, periods_per_year),
        "sharpe": sharpe(rets, rf_annual, periods_per_year),
        "sortino": sortino(rets, rf_annual, periods_per_year),
        "calmar": calmar(equity, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "cvar_5": cvar(rets, 0.05),
        "turnover": turnover(traded, float(pd.Series(equity).mean()), years),
        "commissions": float(sum(f.commission for f in fills)),
        "slippage": float(sum(f.spread_cost + f.impact_cost for f in fills)),
        "transaction_costs": float(sum(f.total_cost for f in fills)),
        "exposure": float(exposure.mean()) if exposure is not None and len(exposure) else 0.0,
    }
    out.update(trade_stats(round_trips(fills)))
    return out
