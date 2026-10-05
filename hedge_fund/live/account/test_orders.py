import pytest

from hedge_fund.live.account.orders import PlannedOrder, check_orders, size_orders, to_limits

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
    check_orders([PlannedOrder(ticker="SPY", side="buy", dollars=100.0)], {}, 100.0, MARKS, {"SPY"},
                 shorts_ok=False, max_name=0.1)
    with pytest.raises(ValueError, match="cash"):
        check_orders([PlannedOrder(ticker="SPY", side="buy", dollars=101.0)], {}, 100.0, MARKS, {"SPY"},
                     shorts_ok=False, max_name=0.1)
    with pytest.raises(ValueError, match="short"):
        check_orders([PlannedOrder(ticker="WMT", side="sell", shares=1)], {}, 1_000.0, MARKS, {"SPY"},
                     shorts_ok=False, max_name=0.5)
    with pytest.raises(ValueError, match="NVDA"):
        check_orders([PlannedOrder(ticker="NVDA", side="buy", dollars=50.0)], {}, 100.0, MARKS, {"SPY"},
                     shorts_ok=False, max_name=0.1)


def test_cap_only_blocks_orders_that_grow_a_name():
    # NVDA drifted to 12% of a $1,000 account; investing a deposit in SPY must still be allowed.
    holdings = {"SPY": 1.76, "NVDA": 1.2}        # $880 + $120
    check_orders([PlannedOrder(ticker="SPY", side="buy", dollars=50.0)], holdings, 50.0, MARKS, {"SPY"},
                 shorts_ok=False, max_name=0.1)
    with pytest.raises(ValueError, match="NVDA"):
        check_orders([PlannedOrder(ticker="NVDA", side="buy", dollars=5.0)], holdings, 50.0, MARKS, {"SPY"},
                     shorts_ok=False, max_name=0.1)


def test_crypto_core_names_are_exempt_from_the_satellite_cap():
    marks = {**MARKS, "BTC/USD": 60_000.0}
    check_orders([PlannedOrder(ticker="BTC/USD", side="buy", dollars=30.0)], {}, 100.0, marks,
                 {"SPY", "BTC/USD"}, shorts_ok=False, max_name=0.1)


CRYPTO_MARKS = {**MARKS, "BTC/USD": 60_000.0, "ETH/USD": 3_000.0}


def test_also_rebalance_sells_an_out_coin_on_a_cash_day():
    orders = size_orders({"SPY": 0.2, "ETH/USD": 0.01}, 0.0, CRYPTO_MARKS, {"SPY": 0.5, "ETH/USD": 0.0},
                         rebalance=False, also_rebalance={"ETH/USD"}, **KW)
    assert orders == [PlannedOrder(ticker="ETH/USD", side="sell", qty=0.01)]
    assert size_orders({"SPY": 0.2, "ETH/USD": 0.01}, 0.0, CRYPTO_MARKS, {"SPY": 0.5, "ETH/USD": 0.0},
                       rebalance=False, **KW) == []


def test_crypto_orders_become_limits_at_the_touch():
    orders = [PlannedOrder(ticker="SPY", side="buy", dollars=50.0),
              PlannedOrder(ticker="BTC/USD", side="buy", dollars=30.0),
              PlannedOrder(ticker="ETH/USD", side="sell", qty=0.01)]
    quotes = {"BTC/USD": (60_000.0, 60_010.0), "ETH/USD": (2_999.0, 3_001.0)}
    spy, btc, eth = to_limits(orders, quotes, market=set())
    assert spy == orders[0]
    assert (btc.side, btc.qty, btc.limit_price, btc.dollars) == ("buy", 0.0005, 60_000.0, None)
    assert (eth.side, eth.qty, eth.limit_price) == ("sell", 0.01, 3_001.0)


def test_limit_buy_size_rounds_down_and_market_names_stay_market():
    orders = [PlannedOrder(ticker="BTC/USD", side="buy", dollars=10.0),
              PlannedOrder(ticker="ETH/USD", side="buy", dollars=10.0)]
    btc, eth = to_limits(orders, {"BTC/USD": (30_000.0, 30_001.0)}, market={"ETH/USD"})
    assert btc.qty == 0.000333333 and btc.qty * btc.limit_price <= 10.0
    assert eth == orders[1]


def test_check_counts_limit_buys_as_spend_and_growth():
    buy = PlannedOrder(ticker="BTC/USD", side="buy", qty=0.001, limit_price=60_000.0)
    check_orders([buy], {}, 60.0, CRYPTO_MARKS, {"BTC/USD"}, shorts_ok=False, max_name=0.1)
    with pytest.raises(ValueError, match="spend"):
        check_orders([buy], {}, 59.0, CRYPTO_MARKS, {"BTC/USD"}, shorts_ok=False, max_name=0.1)
    with pytest.raises(ValueError, match="BTC/USD"):   # a limit buy grows the name, so the cap applies
        check_orders([buy], {}, 600.0, CRYPTO_MARKS, set(), shorts_ok=False, max_name=0.05)
