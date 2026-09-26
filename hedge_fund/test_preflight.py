"""Pre-trade account checks — buying power, short permission, day trades.

These are the facts only the venue has. The portfolio layer (`risk/limits.py`)
clamps sizes and gross exposure without knowing anything about the account.
"""

from __future__ import annotations

import pytest

from hedge_fund.brokers.account import (
    AccountSnapshot,
    PDT_DAY_TRADE_LIMIT,
    account_snapshot,
    preflight,
)
from hedge_fund.brokers.models import Fill, Order, Position
from hedge_fund.brokers.paper import PaperBroker
from hedge_fund.brokers.sim import SimBroker
from hedge_fund.fund import Fund, FundSpec
from hedge_fund.pipeline import run_cycle
from hedge_fund.pipeline.test_run_cycle import CLOSES, FakeAnalyst, FakeDataClient, UNIVERSE


def _account(**over):
    base = dict(venue="test-venue", cash=50_000.0, buying_power=50_000.0,
                equity=50_000.0, shorting_enabled=True)
    base.update(over)
    return AccountSnapshot(**base)


def _orders(*specs):
    return [Order(ticker=t, side=s, quantity=q, price=p) for t, s, q, p in specs]


# ---------------------------------------------------------------------------
# Buying power
# ---------------------------------------------------------------------------

def test_orders_within_buying_power_pass():
    report = preflight(_account(buying_power=10_000.0),
                       _orders(("AAPL", "buy", 10, 100.0)), {})

    assert report.ok
    assert report.required_cash == pytest.approx(1_000.0)
    assert "buying power" in report.summary


def test_orders_beyond_buying_power_are_refused():
    """cash() is not spendable cash; buying_power is what governs."""
    report = preflight(_account(buying_power=5_000.0),
                       _orders(("AAPL", "buy", 100, 100.0)), {})

    assert not report.ok
    assert "10,000.00" in report.summary and "5,000.00" in report.summary


def test_a_closing_sale_frees_cash_for_the_buys():
    positions = {"MSFT": Position(ticker="MSFT", shares=50)}
    report = preflight(
        _account(buying_power=1_000.0),
        _orders(("AAPL", "buy", 20, 100.0), ("MSFT", "sell", 50, 100.0)),
        positions,
    )

    assert report.closing_notional == pytest.approx(5_000.0)
    assert report.required_cash == 0.0  # 2000 - 5000, floored
    assert report.ok


def test_selling_past_zero_opens_a_short_and_does_not_free_cash():
    """Only the closing part of a sell frees cash; the rest consumes it."""
    positions = {"MSFT": Position(ticker="MSFT", shares=10)}
    report = preflight(
        _account(buying_power=1_000.0, shorting_enabled=True),
        _orders(("MSFT", "sell", 40, 100.0)),
        positions,
    )

    assert report.closing_notional == pytest.approx(1_000.0)
    assert report.opening_short_notional == pytest.approx(3_000.0)
    assert report.required_cash == 0.0  # nothing is being bought


def test_a_short_without_permission_is_refused():
    report = preflight(_account(shorting_enabled=False),
                       _orders(("AAPL", "sell", 10, 100.0)), {})

    assert not report.ok
    assert "does not permit short selling" in report.summary


def test_a_plain_sale_of_held_shares_is_not_a_short():
    positions = {"AAPL": Position(ticker="AAPL", shares=10)}
    report = preflight(_account(shorting_enabled=False),
                       _orders(("AAPL", "sell", 10, 100.0)), positions)

    assert report.opening_short_notional == 0.0
    assert report.ok


# ---------------------------------------------------------------------------
# Account state
# ---------------------------------------------------------------------------

def test_a_blocked_account_is_refused():
    report = preflight(_account(trading_blocked=True), [], {})

    assert not report.ok
    assert "blocked" in report.summary


def test_a_non_finite_balance_is_refused():
    report = preflight(_account(cash=float("nan")), [], {})

    assert not report.ok
    assert "non-finite" in report.summary


def test_the_day_trade_window_blocks_a_flagged_account():
    report = preflight(_account(pattern_day_trader=True, daytrade_count=PDT_DAY_TRADE_LIMIT),
                       [], {})

    assert not report.ok
    assert "pattern-day-trader" in report.summary


def test_a_small_account_below_the_floor_is_warned_not_blocked():
    report = preflight(_account(equity=5_000.0, daytrade_count=PDT_DAY_TRADE_LIMIT), [], {})

    assert report.ok
    assert any("pattern-day-trader floor" in w for w in report.warnings)


def test_a_cushion_leaves_room_for_price_drift():
    """A correctly-sized order must not be rejected because the price moved."""
    orders = _orders(("AAPL", "buy", 100, 100.0))  # exactly 10,000

    # Spending every last cent of buying power is refused: prices move between
    # sizing and submission.
    assert not preflight(_account(buying_power=10_000.0), orders, {}).ok
    # 1% of headroom is enough (10,000 / 0.99 = 10,101.02).
    assert preflight(_account(buying_power=10_200.0), orders, {}).ok


# ---------------------------------------------------------------------------
# Opting in
# ---------------------------------------------------------------------------

def test_brokers_without_an_account_skip_the_layer():
    assert account_snapshot(SimBroker(cash=1.0)) is None
    assert account_snapshot(PaperBroker(cash=1.0)) is None


def test_a_broker_with_an_account_is_used():
    class Brokered(SimBroker):
        def account(self):
            return _account(venue="brokered")

    assert account_snapshot(Brokered(cash=1.0)).venue == "brokered"


# ---------------------------------------------------------------------------
# Through the pipeline
# ---------------------------------------------------------------------------

class AccountBroker:
    """A broker that reports an account, so the preflight layer runs."""

    venue = "accounted"

    def __init__(self, inner, account) -> None:
        self._inner = inner
        self._account = account
        self.submitted: list[Order] = []

    def positions(self):
        return self._inner.positions()

    def cash(self):
        return self._inner.cash()

    def account(self):
        return self._account

    def place_order(self, order: Order) -> Fill:
        self.submitted.append(order)
        return self._inner.place_order(order)


def _fund() -> Fund:
    spec = FundSpec(
        schema_version=2,
        name="test-fund",
        strategies=[{"name": "solo", "models": [{"name": "pead"}],
                     "blend": {"mode": "long_short"}}],
        risk={"max_position_pct": 0.25, "max_gross_exposure": 1.0},
        capital=100_000.0,
    )
    return Fund(spec, models={"solo": [FakeAnalyst("pead", {"AAPL": 0.8, "MSFT": -0.6})]})


def test_the_cycle_refuses_when_buying_power_is_short():
    broker = AccountBroker(SimBroker(cash=100_000.0),
                           _account(buying_power=1.0, cash=100_000.0))

    with pytest.raises(ValueError, match="refusing to submit orders"):
        run_cycle(_fund(), "2024-06-03", broker, FakeDataClient(CLOSES), UNIVERSE)

    assert broker.submitted == []  # nothing reached the venue


def test_the_cycle_refuses_shorts_the_account_cannot_hold():
    broker = AccountBroker(SimBroker(cash=100_000.0),
                           _account(buying_power=1e9, shorting_enabled=False))

    with pytest.raises(ValueError, match="short selling"):
        run_cycle(_fund(), "2024-06-03", broker, FakeDataClient(CLOSES), UNIVERSE)

    assert broker.submitted == []


def test_a_healthy_account_lets_the_cycle_through_and_records_it():
    broker = AccountBroker(SimBroker(cash=100_000.0),
                           _account(venue="accounted", buying_power=1e9))

    record = run_cycle(_fund(), "2024-06-03", broker, FakeDataClient(CLOSES), UNIVERSE)

    assert record.preflight is not None
    assert record.preflight.ok
    assert record.preflight.venue == "accounted"
    assert broker.submitted  # the orders did go out
