from datetime import date, timedelta

import pytest

from hedge_fund.crypto.backtest import Metrics, metrics, passes_bar, run_backtest

WEIGHTS = {"A/USD": 0.5, "B/USD": 0.5}


def days(start, n):
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


def flat(n=30, start="2024-06-03"):           # 2024-06-03 is a Monday
    return {s: {d: 100.0 for d in days(start, n)} for s in WEIGHTS}


def test_core_on_flat_prices_only_loses_the_first_fee():
    r = run_backtest(flat(), WEIGHTS, None, fee_bps=25, start="2024-06-03", end="2024-07-02", capital=1000.0)
    assert r.nav[0] == pytest.approx(1000.0 - 1000.0 * 0.0025)
    # The first fee leaves cash a little negative, so Mondays trim a few cents: nearly flat after.
    assert r.nav[-1] == pytest.approx(r.nav[0], rel=1e-4)
    assert r.fees == pytest.approx(2.5, rel=0.01)


def test_rule_off_holds_cash_and_is_never_charged():
    r = run_backtest(flat(), WEIGHTS, lambda closes: 0, fee_bps=25, start="2024-06-03", end="2024-07-02", capital=1000.0)
    assert r.nav == [1000.0] * len(r.nav) and r.fees == 0


def test_weekend_signal_executes_on_monday():
    closes = flat(14)
    for s in WEIGHTS:
        closes[s].update({d: 200.0 for d in days("2024-06-08", 9)})   # jumps Saturday 06-08
    seen = []

    def rule(c):
        seen.append(len(c))
        return int(c[-1] > 150)

    r = run_backtest(closes, WEIGHTS, rule, fee_bps=0, start="2024-06-03", end="2024-06-16", capital=1000.0)
    # Out until Mon 06-10 (decision from Sun 06-09's close), then in at 200: flat after.
    assert r.nav[r.dates.index("2024-06-10")] == pytest.approx(1000.0)
    assert r.trades_on("2024-06-08") == 0 and r.trades_on("2024-06-10") > 0


def test_metrics_and_bar():
    m = metrics(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 110.0, 99.0])
    assert m.total_return == pytest.approx(-0.01)
    assert m.max_drawdown == pytest.approx(0.1)

    def m(sharpe, dd):
        return Metrics(total_return=0.0, annualized_return=0.0, sharpe=sharpe, max_drawdown=dd)

    core = {"h1": m(1.0, 0.5), "h2": m(1.0, 0.5), "full": m(1.0, 0.6)}
    good = {"h1": m(1.2, 0.3), "h2": m(1.1, 0.3), "full": m(1.15, 0.4)}    # 0.4 ≤ 0.75 × 0.6
    assert passes_bar(good, core) is True
    assert passes_bar({**good, "h2": m(0.9, 0.3)}, core) is False         # loses one half
    assert passes_bar({**good, "full": m(1.15, 0.5)}, core) is False      # drawdown only 17% smaller
