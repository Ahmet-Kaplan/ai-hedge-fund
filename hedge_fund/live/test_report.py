import pytest

from hedge_fund.backtesting.fund import FundBacktestMetrics, FundBacktestResult
from hedge_fund.live.ledger import Ledger, NavRow
from hedge_fund.live.report import build_report, strategy_attribution
from hedge_fund.live.test_plan import FakeDataClient


def nav(day, equity, bench):
    return NavRow(date=day, equity=equity, cash=0.0, long_exposure=0.0, short_exposure=0.0,
                  gross=0.0, benchmark_close=bench)


def seeded(tmp_path):
    ledger = Ledger(tmp_path)
    for day, equity, bench in [("2024-06-10", 100_000.0, 500.0), ("2024-06-11", 101_000.0, 505.0),
                               ("2024-06-12", 102_000.0, 500.0)]:
        ledger.upsert_nav(nav(day, equity, bench))
    ledger.write_plan("2024-06-10", {"plan": {"decision": {"strategies": [
        {"name": "solo", "final_contribution": {"AAPL": 0.5}}]}}})
    ledger.write_fills("2024-06-10", {"fills": [
        {"client_order_id": "x-AAPL", "ticker": "AAPL", "side": "buy", "quantity": 500, "filled_qty": 500,
         "status": "filled", "fill_price": 101.0, "reference_price": 100.0, "slippage_bps": 100.0},
        {"client_order_id": "x-MSFT", "ticker": "MSFT", "side": "sell", "quantity": 100, "filled_qty": 50,
         "status": "partially_filled", "fill_price": 99.0, "reference_price": 100.0, "slippage_bps": 100.0},
    ]})
    return ledger


def test_report_metrics(tmp_path):
    report = build_report(seeded(tmp_path))
    assert (report.start, report.end, report.n_days) == ("2024-06-10", "2024-06-12", 3)
    assert report.total_return_pct == pytest.approx(0.02)
    assert report.benchmark_return_pct == pytest.approx(0.0)
    assert report.excess_return_pct == pytest.approx(0.02)
    assert report.n_rebalances == 1
    assert report.fill_rate == pytest.approx(550 / 600)
    assert report.avg_slippage_bps == pytest.approx(100.0)
    assert report.turnover == pytest.approx((500 * 101.0 + 50 * 99.0) / 101_000.0)


def test_report_needs_two_sessions(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.upsert_nav(nav("2024-06-10", 100_000.0, 500.0))
    assert build_report(ledger) is None


def test_report_compares_same_dates_backtest(tmp_path):
    result = FundBacktestResult(
        fund="f", start="2024-06-07", end="2024-06-12", rebalance="weekly", benchmark="SPY",
        universe=["AAPL"], capital=100.0,
        dates=["2024-06-07", "2024-06-10", "2024-06-11", "2024-06-12"],
        nav=[100.0, 100.0, 103.0, 105.0], benchmark_nav=[100.0] * 4,
        metrics=FundBacktestMetrics(total_return_pct=0.05, annualized_return_pct=0.0, sharpe_ratio=0.0,
                                    max_drawdown_pct=0.0, benchmark_return_pct=0.0, excess_return_pct=0.05,
                                    n_cycles=1, n_orders=1),
        records=[],
    )
    report = build_report(seeded(tmp_path), backtest=result)
    assert report.backtest_return_pct == pytest.approx(0.05)


def test_strategy_attribution(tmp_path):
    data = FakeDataClient({"AAPL": {"2024-06-10": 100.0, "2024-06-12": 110.0}})
    assert strategy_attribution(seeded(tmp_path), data) == {"solo": pytest.approx(0.05)}
