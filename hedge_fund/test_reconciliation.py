"""Reconciliation tests — broker versus ledger.

Only one of these differences blocks trading (an order still working at the
venue); the rest are reported. Both halves are pinned here.
"""

from __future__ import annotations

import pytest

from hedge_fund.brokers.models import Fill, Order, Position
from hedge_fund.brokers.paper import PaperBroker
from hedge_fund.brokers.sim import SimBroker
from hedge_fund.fund import Fund, FundSpec
from hedge_fund.pipeline import run_cycle
from hedge_fund.pipeline.test_run_cycle import CLOSES, FakeAnalyst, FakeDataClient, UNIVERSE
from hedge_fund.reconciliation import (
    LedgerReference,
    reconcile,
    require_settled,
)


class WorkingOrderBroker:
    """A broker with an order still in flight at the venue."""

    venue = "working"

    def __init__(self, inner, orders) -> None:
        self._inner = inner
        self._orders = orders

    def positions(self):
        return self._inner.positions()

    def cash(self):
        return self._inner.cash()

    def open_orders(self):
        return self._orders

    def place_order(self, order: Order) -> Fill:
        return self._inner.place_order(order)


class UnreadableOrderBook:
    venue = "unreadable"

    def __init__(self, inner) -> None:
        self._inner = inner

    def positions(self):
        return self._inner.positions()

    def cash(self):
        return self._inner.cash()

    def open_orders(self):
        raise RuntimeError("venue order book unavailable")

    def place_order(self, order: Order) -> Fill:
        return self._inner.place_order(order)


def _ref(positions=None, cash=1_000.0, source="receipt"):
    return LedgerReference(positions=positions or {}, cash=cash, source=source)


# ---------------------------------------------------------------------------
# Position and cash drift
# ---------------------------------------------------------------------------

def test_no_reference_reports_the_broker_as_truth():
    report = reconcile(SimBroker(cash=500.0, positions={"AAPL": 3}))

    assert report.reference is None
    assert report.cash_actual == 500.0
    assert report.cash_delta is None
    assert not report.has_position_drift
    assert report.clean
    assert "source of truth" in report.summary


def test_matching_book_is_clean():
    broker = SimBroker(cash=1_000.0, positions={"AAPL": 5, "MSFT": -2})
    report = reconcile(broker, _ref({"AAPL": 5, "MSFT": -2}))

    assert report.clean
    assert report.matched == {"AAPL": 5, "MSFT": -2}
    assert report.summary == "matches receipt"


def test_share_count_difference_is_reported_with_a_delta():
    broker = SimBroker(cash=1_000.0, positions={"AAPL": 7})
    report = reconcile(broker, _ref({"AAPL": 5}))

    assert not report.clean
    assert report.has_position_drift
    assert [(d.ticker, d.expected, d.actual, d.delta) for d in report.changed] == [
        ("AAPL", 5, 7, 2)
    ]
    assert "positions differ" in report.summary


def test_positions_present_on_only_one_side():
    broker = SimBroker(cash=1_000.0, positions={"NVDA": 1})
    report = reconcile(broker, _ref({"AAPL": 5}))

    assert report.only_at_broker == {"NVDA": 1}
    assert report.only_in_ledger == {"AAPL": 5}


def test_flat_rows_do_not_count_as_drift():
    """A position closed to zero is absent, not a mismatch."""
    broker = SimBroker(cash=1_000.0)
    report = reconcile(broker, _ref({"AAPL": 0}))

    assert report.clean


def test_cash_within_tolerance_is_not_drift():
    broker = SimBroker(cash=999.50)
    report = reconcile(broker, _ref(cash=1_000.0), cash_tolerance=1.0)

    assert report.cash_delta == 0.0
    assert report.clean


def test_cash_beyond_tolerance_is_drift():
    broker = SimBroker(cash=950.0)
    report = reconcile(broker, _ref(cash=1_000.0), cash_tolerance=1.0)

    assert report.cash_delta == pytest.approx(-50.0)
    assert not report.clean
    assert "cash" in report.summary


def test_ledger_reference_from_a_cycle_record():
    record = _run_cycle_once()
    reference = LedgerReference.from_cycle_record(record)

    assert reference.positions == record.positions
    assert reference.cash == record.cash
    assert reference.source.startswith("test-fund")


# ---------------------------------------------------------------------------
# Working orders — the case that actually blocks
# ---------------------------------------------------------------------------

def test_working_orders_are_flagged_and_block():
    broker = WorkingOrderBroker(SimBroker(cash=1_000.0), {"t1": Order(
        ticker="AAPL", side="buy", quantity=1, price=10.0)})
    report = reconcile(broker, _ref())

    assert report.working_orders == 1
    assert not report.clean
    assert any("still holding orders" in n for n in report.notes)
    with pytest.raises(ValueError, match="working order"):
        require_settled(report)


def test_drift_alone_does_not_block():
    """The broker is authoritative for sizing; drift is a reporting concern."""
    broker = SimBroker(cash=1_000.0, positions={"AAPL": 9})
    report = reconcile(broker, _ref({"AAPL": 5}))

    require_settled(report)  # must not raise
    assert not report.clean


def test_broker_without_open_orders_reports_none_working():
    assert reconcile(SimBroker(cash=1.0)).working_orders == 0
    assert reconcile(PaperBroker(cash=1.0)).working_orders == 0


def test_an_unreadable_order_book_is_treated_as_working():
    """'Cannot tell' must not read as 'nothing in flight'."""
    broker = UnreadableOrderBook(SimBroker(cash=1_000.0))
    report = reconcile(broker, _ref())

    assert report.working_orders == 1
    with pytest.raises(ValueError, match="working order"):
        require_settled(report)


# ---------------------------------------------------------------------------
# Through the pipeline
# ---------------------------------------------------------------------------

def test_cycle_records_the_reconciliation():
    """A receipt claiming a book the venue does not hold is visible on the record."""
    record = _run_cycle_once(reference=_ref({"AAPL": 5}, cash=100_000.0))

    assert record.reconciliation is not None
    assert record.reconciliation.reference == "receipt"
    assert record.reconciliation.cash_actual == pytest.approx(record.cash_before)
    # The venue is flat but the receipt claims 5 AAPL, and AAPL was traded.
    drift = record.reconciliation
    assert drift.only_in_ledger == {"AAPL": 5}
    assert not drift.clean
    assert any("authoritative" in note for note in drift.notes)


def test_cycle_refuses_to_trade_over_a_working_order():
    fund = _fund()
    broker = WorkingOrderBroker(
        SimBroker(cash=100_000.0),
        {"t1": Order(ticker="AAPL", side="buy", quantity=1, price=10.0)},
    )

    with pytest.raises(ValueError, match="working order"):
        run_cycle(fund, "2024-06-03", broker, FakeDataClient(CLOSES), UNIVERSE)

    assert isinstance(broker._inner, SimBroker)
    # Nothing was submitted: the book is untouched.
    assert broker._inner.positions() == {}


def test_cycle_can_be_told_to_run_anyway():
    fund = _fund()
    broker = WorkingOrderBroker(
        SimBroker(cash=100_000.0),
        {"t1": Order(ticker="AAPL", side="buy", quantity=1, price=10.0)},
    )

    record = run_cycle(fund, "2024-06-03", broker, FakeDataClient(CLOSES), UNIVERSE,
                       require_settled_broker=False)

    assert record.reconciliation.working_orders == 1


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


def _run_cycle_once(reference=None, broker=None):
    return run_cycle(_fund(), "2024-06-03", broker or SimBroker(cash=100_000.0),
                     FakeDataClient(CLOSES), UNIVERSE, reference=reference)
