"""Every metric against numbers computed by hand."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from hedge_fund.core import FillEvent
from hedge_fund.systematic import metrics as m

EQ = [100.0, 110.0, 99.0, 108.9, 120.0]            # returns: +10%, -10%, +10%, +10.19%
R = pd.Series(EQ).pct_change().dropna()


def test_total_return_and_cagr():
    assert m.total_return(EQ) == pytest.approx(0.20)
    # 4 periods at 252/yr
    assert m.cagr(EQ) == pytest.approx(1.2 ** (252 / 4) - 1)
    assert m.cagr([100, 121], periods_per_year=1) == pytest.approx(0.21)


def test_volatility_and_sharpe():
    sd = np.std(R, ddof=1)
    assert m.annualized_volatility(R) == pytest.approx(sd * math.sqrt(252))
    assert m.sharpe(R) == pytest.approx(R.mean() / sd * math.sqrt(252))
    rf = 0.0252
    ex = R - rf / 252
    assert m.sharpe(R, rf_annual=rf) == pytest.approx(ex.mean() / np.std(ex, ddof=1) * math.sqrt(252))
    assert m.sharpe([0.01, 0.01, 0.01]) == 0.0                 # zero variance


def test_sortino_uses_downside_deviation():
    down = math.sqrt(((-0.1) ** 2) / 4)
    assert m.sortino(R) == pytest.approx(R.mean() / down * math.sqrt(252))
    assert m.sortino([0.01, 0.02]) == 0.0                      # no downside


def test_drawdown_and_calmar():
    assert m.max_drawdown(EQ) == pytest.approx(0.10)
    assert m.max_drawdown([100, 80, 120, 60]) == pytest.approx(0.5)
    assert m.calmar(EQ) == pytest.approx(m.cagr(EQ) / 0.10)


def test_cvar_is_mean_of_worst_tail():
    r = [-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05]
    assert m.cvar(r, alpha=0.2) == pytest.approx(0.04)          # mean(-0.05, -0.03)
    assert m.cvar(r, alpha=0.05) == pytest.approx(0.05)         # at least one observation


def fill(symbol, side, q, px, session, fee=0.0, spread=0.0, impact=0.0):
    return FillEvent(client_order_id="x", symbol=symbol, side=side, quantity=q, price=px, session=session,
                     mid_price=px, commission=fee, spread_cost=spread, impact_cost=impact)


def test_round_trips_fifo_long_and_short():
    fills = [
        fill("A", "buy", 10, 100, "d1", fee=1.0),
        fill("A", "buy", 10, 110, "d2", fee=1.0),
        fill("A", "sell", 15, 120, "d3", fee=1.5),     # closes 10@100 and 5@110
        fill("B", "sell", 5, 50, "d1"),                 # short
        fill("B", "buy", 5, 40, "d4"),                  # cover: +50
    ]
    trips = m.round_trips(fills)
    a1, a2, b = trips[0], trips[1], trips[2]
    assert (a1.quantity, a1.entry_price, a1.exit_price) == (10, 100, 120)
    assert a1.pnl == pytest.approx(10 * 20 - 10 * (0.1 + 0.1))
    assert (a2.quantity, a2.entry_price) == (5, 110)
    assert a2.pnl == pytest.approx(5 * 10 - 5 * (0.1 + 0.1))
    assert b.quantity == -5 and b.pnl == pytest.approx(50)
    assert len(trips) == 3                              # 5 of A still open


def test_trade_stats():
    trips = [m.RoundTrip("A", 1, "a", "b", 1, 1, p) for p in (10.0, -5.0, 20.0, -5.0)]
    s = m.trade_stats(trips)
    assert s["hit_rate"] == 0.5
    assert s["average_win"] == 15.0 and s["average_loss"] == -5.0
    assert s["payoff_ratio"] == 3.0
    assert s["profit_factor"] == pytest.approx(30 / 10)
    assert s["expectancy"] == pytest.approx(5.0)
    assert m.trade_stats([])["n_trades"] == 0


def test_turnover_costs_and_summary():
    eq = pd.Series([1000.0] * 253)
    fills = [fill("A", "buy", 10, 100, "d1", fee=1, spread=0.5, impact=0.25),
             fill("A", "sell", 10, 100, "d2", fee=1, spread=0.5, impact=0.25)]
    out = m.summarize(eq, fills=fills, exposure=pd.Series([0.5, 1.0]))
    assert out["turnover"] == pytest.approx(2000 / 1000 / 1.0)
    assert out["commissions"] == 2 and out["slippage"] == pytest.approx(1.5)
    assert out["transaction_costs"] == pytest.approx(3.5)
    assert out["exposure"] == pytest.approx(0.75)
    assert out["n_trades"] == 1 and out["expectancy"] == pytest.approx(-2.0)
