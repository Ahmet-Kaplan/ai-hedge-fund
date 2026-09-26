"""Desk snapshot tests — the read model both the TUI and the web dashboard use.

Nothing here can submit an order; these pin what an operator is shown, and in
particular the labels they must not misread (PAPER versus LIVE, read-only).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hedge_fund.brokers.account import AccountSnapshot, PositionDetail
from hedge_fund.brokers.models import Fill, Order, Position
from hedge_fund.brokers.paper import PaperBroker
from hedge_fund.brokers.sim import SimBroker
from hedge_fund.desk import desk_snapshot
from hedge_fund.reconciliation import LedgerReference
from hedge_fund.venue import OpenVenue


class RichBroker:
    """A venue that reports an account and valued holdings."""

    venue = "rich"

    def __init__(self, *, cash=10_000.0, positions=None, account=None,
                 details=None, orders=None) -> None:
        self._cash = cash
        self._positions = positions or {}
        self._account = account
        self._details = details
        self._orders = orders or {}

    def positions(self):
        return {t: Position(ticker=t, shares=s) for t, s in self._positions.items()}

    def cash(self):
        return self._cash

    def account(self):
        return self._account

    def position_details(self):
        return self._details if self._details is not None else []

    def open_orders(self):
        return self._orders

    def place_order(self, order: Order) -> Fill:
        raise AssertionError("the desk must never submit")


def _opened(broker, *, label="rich", live=False, note="", reference=None,
            settings=None) -> OpenVenue:
    return OpenVenue(name="test", broker=broker, reference=reference, label=label,
                     live=live, note=note, settings=settings)


def _account(**over):
    base = dict(venue="rich", cash=10_000.0, buying_power=40_000.0, equity=12_000.0,
                shorting_enabled=True)
    base.update(over)
    return AccountSnapshot(**base)


# ---------------------------------------------------------------------------
# Plain venues
# ---------------------------------------------------------------------------

def test_paper_venue_reads_as_paper_and_not_read_only():
    opened = _opened(PaperBroker(cash=5_000.0, positions={"AAPL": 3}),
                     label="paper", note="paper venue · fills at mark")

    desk = desk_snapshot(opened)

    assert desk.badge == "PAPER"
    assert desk.live is False
    assert desk.cash == pytest.approx(5_000.0)
    assert [p.ticker for p in desk.positions] == ["AAPL"]
    assert desk.positions[0].shares == 3
    # No venue valuation, so no invented P&L.
    assert desk.has_valuation is False
    assert desk.unrealized_pnl is None
    assert desk.buying_power is None
    assert desk.warnings == []


def test_a_venue_without_valuation_still_lists_holdings():
    opened = _opened(SimBroker(cash=1_000.0, positions={"MSFT": -2, "AAPL": 1}),
                     label="sim")

    desk = desk_snapshot(opened)

    assert [(p.ticker, p.shares) for p in desk.positions] == [("AAPL", 1), ("MSFT", -2)]
    assert desk.positions[1].side == "short"


# ---------------------------------------------------------------------------
# A venue that reports more
# ---------------------------------------------------------------------------

def test_buying_power_and_valuation_come_from_the_venue():
    details = [
        PositionDetail(ticker="AAPL", shares=10, avg_entry_price=100.0,
                       current_price=110.0, market_value=1_100.0,
                       unrealized_pnl=100.0, unrealized_pnl_pct=0.1),
        PositionDetail(ticker="MSFT", shares=-4, avg_entry_price=50.0,
                       current_price=45.0, market_value=-180.0,
                       unrealized_pnl=20.0, unrealized_pnl_pct=0.1),
    ]
    broker = RichBroker(cash=9_000.0, account=_account(), details=details)

    desk = desk_snapshot(_opened(broker))

    assert desk.has_valuation is True
    assert desk.buying_power == pytest.approx(40_000.0)
    assert desk.equity == pytest.approx(12_000.0)
    assert desk.gross_exposure == pytest.approx(1_280.0)
    assert desk.net_exposure == pytest.approx(920.0)
    assert desk.unrealized_pnl == pytest.approx(120.0)


def test_a_blocked_account_is_warned_about():
    broker = RichBroker(cash=1.0, account=_account(trading_blocked=True))

    desk = desk_snapshot(_opened(broker))

    assert desk.blocked is True
    assert any("blocked" in w for w in desk.warnings)


def test_working_orders_are_warned_about():
    broker = RichBroker(cash=1.0, account=_account(),
                        orders={"a": Order(ticker="AAPL", side="buy", quantity=1, price=1.0)})

    desk = desk_snapshot(_opened(broker))

    assert desk.working_orders == 1
    assert any("still working" in w for w in desk.warnings)


def test_a_read_only_venue_says_so():
    # Read-only is an Alpaca state: its settings decide, and they are closed.
    from hedge_fund.brokers.alpaca import AlpacaSettings

    gated = AlpacaSettings(api_key="k", secret_key="s", paper=True,
                           trading_enabled=False)
    desk = desk_snapshot(_opened(RichBroker(cash=1.0, account=_account()),
                                 label="alpaca-paper", settings=gated,
                                 note="READ-ONLY (set ALPACA_TRADING_ENABLED=1)"))

    assert any("read-only" in w.lower() for w in desk.warnings)


def test_a_live_venue_is_labelled_live():
    desk = desk_snapshot(_opened(RichBroker(cash=1.0, account=_account()),
                                 label="alpaca-live", live=True,
                                 note="LIVE venue alpaca-live · orders WILL be submitted for real"))

    assert desk.badge == "LIVE"
    assert desk.live is True


# ---------------------------------------------------------------------------
# Reconciliation is part of the view
# ---------------------------------------------------------------------------

def test_reconciliation_drift_surfaces_as_a_warning():
    reference = LedgerReference(positions={"AAPL": 5}, cash=10_000.0, source="desk 2024-06-03")
    broker = RichBroker(cash=10_000.0, account=_account())

    desk = desk_snapshot(_opened(broker, reference=reference))

    assert desk.reconciliation is not None
    assert desk.reconciliation_reference == "desk 2024-06-03"
    assert not desk.reconciliation.clean
    assert any("positions differ" in w for w in desk.warnings)


def test_a_matching_book_produces_no_drift_warning():
    reference = LedgerReference(positions={"AAPL": 5}, cash=10_000.0, source="desk")
    broker = RichBroker(cash=10_000.0, positions={"AAPL": 5}, account=_account())

    desk = desk_snapshot(_opened(broker, reference=reference))

    assert desk.reconciliation.clean
    assert not any("positions differ" in w for w in desk.warnings)
