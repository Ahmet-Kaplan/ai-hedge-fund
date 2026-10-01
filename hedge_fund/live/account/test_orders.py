import pytest

from hedge_fund.live.account.orders import PlannedOrder, check_orders, size_orders

MARKS = {"SPY": 500.0, "NVDA": 100.0, "WMT": 100.0}
KW = dict(min_order_usd=1.0, min_trade_pct=0.005, cash_buffer_pct=0.0)


def plan(holdings, cash, target, rebalance=True, **kw):
    return size_orders(holdings, cash, MARKS, target, rebalance=rebalance, **{**KW, **kw})


def test_first_deposit_buys_the_core_by_dollars():
    assert plan({}, 100.0, {"SPY": 1.0}) == [PlannedOrder(ticker="SPY", side="buy", dollars=100.0)]


def test_cash_buffer_is_left_unspent():
    assert plan({}, 100.0, {"SPY": 1.0}, cash_buffer_pct=0.01)[0].dollars == 99.0


def test_deposit_day_only_buys_with_cash():
    # Holding 0.2 SPY ($100) + $50 new cash; target 80/20 SPY/NVDA. No sells on a cash day.
    orders = plan({"SPY": 0.2}, 50.0, {"SPY": 0.8, "NVDA": 0.2}, rebalance=False)
    assert all(o.side == "buy" for o in orders)
    assert sum(o.dollars for o in orders) == pytest.approx(50.0)
    assert {o.ticker for o in orders} == {"SPY", "NVDA"}


def test_rebalance_sells_fractions_and_closes_unwanted_names():
    orders = plan({"SPY": 1.0, "WMT": 0.004}, 0.0, {"SPY": 1.0})
    assert orders == [PlannedOrder(ticker="WMT", side="sell", qty=0.004)]   # $0.40 close is never skipped


def test_dust_trades_skipped_on_rebalance():
    # $1,000 account ($994 SPY + $6 cash): SPY +$2 and NVDA +$4 are both under the 0.5% floor ($5).
    orders = plan({"SPY": 1.988}, 6.0, {"SPY": 0.996, "NVDA": 0.004}, rebalance=True)
    assert orders == []


def test_buys_scaled_to_available_cash_not_sale_proceeds():
    orders = plan({"SPY": 1.0}, 10.0, {"SPY": 0.5, "NVDA": 0.5})
    sells = [o for o in orders if o.side == "sell"]
    buys = [o for o in orders if o.side == "buy"]
    assert [(o.ticker, o.qty) for o in sells] == [("SPY", pytest.approx(0.49))]
    assert buys == [PlannedOrder(ticker="NVDA", side="buy", dollars=10.0)]


def test_shorts_use_whole_shares():
    orders = plan({"SPY": 4.0}, 2_000.0, {"SPY": 0.9, "WMT": -0.1})   # equity 4,000 → short $400 = 4 sh
    assert PlannedOrder(ticker="WMT", side="sell", shares=4) in orders


def test_check_blocks_overspend_shorts_and_oversized_names():
    check_orders([PlannedOrder(ticker="SPY", side="buy", dollars=100.0)], {}, 100.0, MARKS, "SPY",
                 shorts_ok=False, max_name=0.1)
    with pytest.raises(ValueError, match="cash"):
        check_orders([PlannedOrder(ticker="SPY", side="buy", dollars=101.0)], {}, 100.0, MARKS, "SPY",
                     shorts_ok=False, max_name=0.1)
    with pytest.raises(ValueError, match="short"):
        check_orders([PlannedOrder(ticker="WMT", side="sell", shares=1)], {}, 1_000.0, MARKS, "SPY",
                     shorts_ok=False, max_name=0.5)
    with pytest.raises(ValueError, match="NVDA"):
        check_orders([PlannedOrder(ticker="NVDA", side="buy", dollars=50.0)], {}, 100.0, MARKS, "SPY",
                     shorts_ok=False, max_name=0.1)
