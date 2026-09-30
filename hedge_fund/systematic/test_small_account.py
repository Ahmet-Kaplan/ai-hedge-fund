"""Small-account economics, ruin probability and a EUR 200 backtest."""

from __future__ import annotations

import pytest

from hedge_fund.systematic import MarketPanel
from hedge_fund.systematic.backtest import BacktestConfig, SystematicBacktester
from hedge_fund.systematic.execution import FillTiming
from hedge_fund.systematic.portfolio import PortfolioConfig
from hedge_fund.systematic.risk import RiskConfig
from hedge_fund.systematic.small_account import (
    EconomicsFilter, SmallAccountConfig, load_small_account, probability_of_ruin, required_growth,
    round_trip_cost_bps, small_account_universe, trade_economics,
)
from hedge_fund.systematic.strategies import TimeSeriesMomentum
from hedge_fund.systematic.testing import SyntheticMarket, weekdays

CFG = SmallAccountConfig()


def test_fixed_commission_dominates_small_orders():
    assert round_trip_cost_bps(20.0, CFG) == pytest.approx(2 * (1.0 / 20 * 1e4 + 10))       # 1,020 bp
    assert round_trip_cost_bps(200.0, CFG) == pytest.approx(2 * (50 + 10))
    assert not trade_economics(20.0, 300, CFG)["economic"]
    assert trade_economics(200.0, 150, CFG)["economic"]                                       # 120 + 20 margin
    assert "minimum" in trade_economics(0.5, 10_000, CFG)["reason"]


def test_config_file_loads():
    cfg = load_small_account()
    assert cfg.capital == 200.0 and cfg.fractional and cfg.cost_model().commission_min == 1.0


def test_ruin_probability_rises_with_risk():
    calm = probability_of_ruin(0.0005, 0.01, 250, seed=1)
    wild = probability_of_ruin(0.0005, 0.06, 250, seed=1)
    assert calm < 0.01 < wild


def test_unrealistic_targets_are_spelled_out():
    assert required_growth(200, 5_000, 1) == pytest.approx(24.0)            # +2,400% in one month
    assert required_growth(200, 10_000, 1) == pytest.approx(49.0)
    assert required_growth(200, 5_000, 12) > 0.30                           # >30% per month for a year


def test_fractional_instruments_allow_tiny_positions():
    reg = small_account_universe(["SPY"], CFG)
    assert reg.get("SPY").quantity_for_notional(50.0, 450.0) == pytest.approx(0.111111)


def test_eur200_backtest_skips_uneconomic_trades():
    days = weekdays("2021-01-04", "2023-12-29")
    m = SyntheticMarket()
    m.add_series("SPY", days, SyntheticMarket.random_walk(days, seed=0, drift=0.0004, vol=0.01), volume=50_000_000)
    for i, s in enumerate(("AAA", "BBB", "CCC", "DDD")):
        m.add_series(s, days, SyntheticMarket.random_walk(days, seed=i + 5, drift=0.0005, vol=0.015),
                     volume=5_000_000)
    panel = MarketPanel.build(m, ["SPY", "AAA", "BBB", "CCC", "DDD"], days[0], days[-1])
    cfg = BacktestConfig(start="2022-01-03", end=days[-1], capital=CFG.capital, timing=FillTiming.NEXT_OPEN,
                         costs=CFG.cost_model(), portfolio=PortfolioConfig(top_n=CFG.max_positions),
                         risk=RiskConfig(max_position=0.5, vol_target=None, max_turnover=None,
                                         max_cluster_weight=1.0, max_adv_participation=1.0))
    strat = [TimeSeriesMomentum(lookback=126, skip=5, vol_lookback=40, exclude=("SPY",))]
    unfiltered = SystematicBacktester(panel, strat, cfg, instruments=small_account_universe([], CFG)).run()
    filt = EconomicsFilter(CFG, default_edge_bps=150.0)
    filtered = SystematicBacktester(panel, strat, cfg, instruments=small_account_universe([], CFG),
                                    order_filter=filt).run()
    assert unfiltered.trades and any(q["quantity"] < 1 for q in unfiltered.trades)     # fractional shares
    assert filt.dropped and all(not d["economic"] for d in filt.dropped)
    assert len(filtered.trades) < len(unfiltered.trades)
    assert filtered.metrics["commissions"] < unfiltered.metrics["commissions"]
    assert any(d["filtered_orders"] for d in filtered.decisions)
