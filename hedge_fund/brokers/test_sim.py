"""SimBroker tests — deterministic fills and bookkeeping."""

import pytest

from hedge_fund.brokers.models import Order
from hedge_fund.brokers.sim import SimBroker


def test_buy_updates_cash_and_position():
    broker = SimBroker(cash=10_000.0)
    fill = broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    assert broker.cash() == pytest.approx(9_000.0)
    assert broker.positions()["AAPL"].shares == 10
    assert fill.quantity == 10
    assert fill.price == 100.0


def test_sell_updates_cash_and_position():
    broker = SimBroker(cash=0.0)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    broker.place_order(Order(ticker="AAPL", side="sell", quantity=4, price=110.0))
    assert broker.positions()["AAPL"].shares == 6
    assert broker.cash() == pytest.approx(-1_000.0 + 440.0)


def test_position_removed_at_zero():
    broker = SimBroker(cash=1_000.0)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=5, price=100.0))
    broker.place_order(Order(ticker="AAPL", side="sell", quantity=5, price=100.0))
    assert broker.positions() == {}


def test_sell_past_zero_creates_short():
    broker = SimBroker(cash=0.0)
    broker.place_order(Order(ticker="AAPL", side="sell", quantity=3, price=100.0))
    assert broker.positions()["AAPL"].shares == -3
    assert broker.cash() == pytest.approx(300.0)


def test_nonpositive_price_raises():
    broker = SimBroker(cash=1_000.0)
    with pytest.raises(ValueError):
        broker.place_order(Order(ticker="AAPL", side="buy", quantity=1, price=0.0))


@pytest.mark.parametrize("price", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_price_raises_without_mutating_book(price):
    broker = SimBroker(cash=1_000.0)
    with pytest.raises(ValueError):
        broker.place_order(Order(ticker="AAPL", side="buy", quantity=1, price=price))
    assert broker.cash() == 1_000.0
    assert broker.positions() == {}


def test_positions_returns_a_copy():
    broker = SimBroker(cash=1_000.0)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=5, price=100.0))
    broker.positions().clear()
    assert broker.positions()["AAPL"].shares == 5


def test_commission_charged_on_every_fill():
    broker = SimBroker(cash=10_000.0, commission_bps=10)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    assert broker.cash() == pytest.approx(10_000.0 - 1_000.0 - 1.0)
    broker.place_order(Order(ticker="AAPL", side="sell", quantity=10, price=100.0))
    assert broker.cash() == pytest.approx(10_000.0 - 2.0)
    assert broker.costs() == pytest.approx(2.0)


def test_default_is_cost_free():
    broker = SimBroker(cash=10_000.0)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    assert broker.costs() == 0.0


def test_borrow_accrues_on_shorts_only():
    broker = SimBroker(cash=0.0)
    broker.place_order(Order(ticker="AAPL", side="sell", quantity=10, price=100.0))
    broker.place_order(Order(ticker="MSFT", side="buy", quantity=10, price=100.0))
    fee = broker.accrue_borrow({"AAPL": 100.0, "MSFT": 100.0}, days=365, borrow_bps_annual=100)
    assert fee == pytest.approx(10.0)          # 1% of $1,000 short notional
    assert broker.cash() == pytest.approx(-10.0)
    assert broker.costs() == pytest.approx(10.0)
