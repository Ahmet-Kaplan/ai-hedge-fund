"""Order journal tests — the crash-safe half of execution.

The scenario that matters: `execute_decision` submits orders in a loop and
writes nothing until the last one fills. If a later order raises, the earlier
fills happened at the venue and were never recorded. These tests pin that the
journal still has them.
"""

from __future__ import annotations

import json

import pytest

from hedge_fund.brokers.models import Fill, Order, Position
from hedge_fund.brokers.sim import SimBroker
from hedge_fund.journal import (
    FileOrderJournal,
    JournalledBroker,
    MemoryJournal,
    OrderEvent,
    journal_summary,
    unfilled_intents,
)
from hedge_fund.pipeline import run_cycle
from hedge_fund.fund import Fund, FundSpec
from hedge_fund.pipeline.test_run_cycle import (
    CLOSES,
    FakeAnalyst,
    FakeDataClient,
    UNIVERSE,
)


def _order(ticker="AAPL", side="buy", quantity=5, price=100.0, cid="cid-1"):
    return Order(ticker=ticker, side=side, quantity=quantity, price=price,
                 client_order_id=cid)


class FailingBroker:
    """Fails on the nth submission, after the earlier ones have gone through."""

    venue = "failing"

    def __init__(self, inner, fail_on: int) -> None:
        self._inner = inner
        self._fail_on = fail_on
        self.calls = 0

    def positions(self) -> dict[str, Position]:
        return self._inner.positions()

    def cash(self) -> float:
        return self._inner.cash()

    def place_order(self, order: Order) -> Fill:
        self.calls += 1
        if self.calls == self._fail_on:
            raise RuntimeError("venue went away mid-loop")
        return self._inner.place_order(order)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def test_file_journal_appends_jsonl_and_reads_back(tmp_path):
    path = tmp_path / "desk.jsonl"
    journal = FileOrderJournal(path)
    broker = JournalledBroker(SimBroker(cash=10_000.0), journal,
                              fund="desk", session="2025-01-10")

    broker.place_order(_order())

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2  # intent, then fill
    assert [json.loads(l)["event"] for l in lines] == ["intent", "fill"]
    events = journal.entries()
    assert events[0].fund == "desk" and events[0].session == "2025-01-10"
    assert events[1].filled_quantity == 5
    assert events[1].venue == "sim"


def test_intent_is_written_before_the_order_is_sent(tmp_path):
    """Order of operations is the whole point: intent first, outcome after."""
    events: list[str] = []

    class Recording(FileOrderJournal):
        def record(self, event):
            events.append(event.event)
            super().record(event)

    broker = JournalledBroker(SimBroker(cash=10_000.0),
                              Recording(tmp_path / "j.jsonl"),
                              fund="d", session="s")
    broker.place_order(_order())

    assert events == ["intent", "fill"]


def test_a_failed_submission_records_the_error_and_reraises(tmp_path):
    journal = FileOrderJournal(tmp_path / "j.jsonl")
    broker = JournalledBroker(FailingBroker(SimBroker(cash=10_000.0), fail_on=1),
                              journal, fund="d", session="s")

    with pytest.raises(RuntimeError, match="went away"):
        broker.place_order(_order())

    assert [e.event for e in journal.entries()] == ["intent", "error"]
    assert "went away" in journal.entries()[1].error


def test_reads_pass_through_and_are_not_journalled(tmp_path):
    journal = FileOrderJournal(tmp_path / "j.jsonl")
    inner = SimBroker(cash=1_000.0, positions={"AAPL": 3})
    broker = JournalledBroker(inner, journal, fund="d", session="s")

    assert broker.cash() == 1_000.0
    assert broker.positions()["AAPL"].shares == 3
    assert broker.venue == "sim"
    assert journal.entries() == []


def test_journal_does_not_change_broker_behaviour(tmp_path):
    inner = SimBroker(cash=1_000.0)
    broker = JournalledBroker(inner, FileOrderJournal(tmp_path / "j.jsonl"),
                              fund="d", session="s")

    fill = broker.place_order(_order(quantity=2, price=50.0))

    assert fill.quantity == 2
    assert inner.cash() == 900.0
    assert inner.positions()["AAPL"].shares == 2


# ---------------------------------------------------------------------------
# Accounting for what happened
# ---------------------------------------------------------------------------

def test_unfilled_intents_finds_orders_with_no_outcome():
    events = [
        OrderEvent(ts="t", event="intent", fund="f", session="s",
                   client_order_id="a", ticker="AAPL", side="buy",
                   quantity=1, price=10.0),
        OrderEvent(ts="t", event="fill", fund="f", session="s",
                   client_order_id="a", ticker="AAPL", side="buy",
                   quantity=1, price=10.0, filled_quantity=1, filled_price=10.0),
        OrderEvent(ts="t", event="intent", fund="f", session="s",
                   client_order_id="b", ticker="MSFT", side="buy",
                   quantity=1, price=10.0),
    ]

    orphaned = unfilled_intents(events)

    assert [e.client_order_id for e in orphaned] == ["b"]


def test_unfilled_intents_matches_without_client_ids():
    """A caller may journal id-less orders; identity falls back to the ask."""
    events = [
        OrderEvent(ts="t", event="intent", fund="f", session="s",
                   ticker="AAPL", side="buy", quantity=1, price=10.0),
        OrderEvent(ts="t", event="fill", fund="f", session="s",
                   ticker="AAPL", side="buy", quantity=1, price=10.0,
                   filled_quantity=1, filled_price=10.0),
    ]

    assert unfilled_intents(events) == []


def test_an_interleaved_fill_does_not_resolve_the_wrong_intent():
    events = [
        OrderEvent(ts="t", event="intent", fund="f", session="s",
                   client_order_id="a", ticker="AAPL", side="buy",
                   quantity=1, price=10.0),
        OrderEvent(ts="t", event="intent", fund="f", session="s",
                   client_order_id="b", ticker="MSFT", side="buy",
                   quantity=1, price=10.0),
        OrderEvent(ts="t", event="fill", fund="f", session="s",
                   client_order_id="b", ticker="MSFT", side="buy",
                   quantity=1, price=10.0, filled_quantity=1, filled_price=10.0),
    ]

    assert [e.client_order_id for e in unfilled_intents(events)] == ["a"]


def test_summary_counts_by_event():
    events = [
        OrderEvent(ts="t", event="intent", fund="f", session="s", ticker="A",
                   side="buy", quantity=1, price=1.0),
        OrderEvent(ts="t", event="fill", fund="f", session="s", ticker="A",
                   side="buy", quantity=1, price=1.0, filled_quantity=1,
                   filled_price=1.0),
        OrderEvent(ts="t", event="error", fund="f", session="s", ticker="B",
                   side="buy", quantity=1, price=1.0, error="nope"),
    ]

    summary = journal_summary(events)

    assert summary["by_event"] == {"intent": 1, "fill": 1, "error": 1}
    assert summary["filled_quantity"] == 1
    assert summary["unfilled_intents"] == 0


# ---------------------------------------------------------------------------
# The acceptance case: a crash mid-execution still leaves a trail
# ---------------------------------------------------------------------------

def test_crash_mid_execution_keeps_the_fills_that_happened(tmp_path):
    """No CycleRecord is written, but the fills that already executed survive."""
    journal = FileOrderJournal(tmp_path / "j.jsonl")
    fund = _fund()
    broker = JournalledBroker(
        FailingBroker(SimBroker(cash=100_000.0), fail_on=2),
        journal, fund="test-fund", session="2024-06-10",
    )

    with pytest.raises(RuntimeError, match="mid-loop"):
        run_cycle(fund, "2024-06-03", broker, FakeDataClient(CLOSES), UNIVERSE)

    events = journal.entries()
    summary = journal_summary(events)
    assert summary["by_event"]["intent"] == 2   # both were sent
    assert summary["by_event"]["fill"] == 1     # one really executed
    assert summary["by_event"]["error"] == 1    # the second did not
    # The executed trade is on disk with its price — the whole point.
    filled = [e for e in events if e.event == "fill"][0]
    assert filled.filled_quantity > 0 and filled.filled_price > 0
    assert filled.client_order_id is not None


def test_every_order_in_a_clean_cycle_is_journalled(tmp_path):
    journal = FileOrderJournal(tmp_path / "j.jsonl")
    broker = JournalledBroker(SimBroker(cash=100_000.0), journal,
                              fund="test-fund", session="2024-06-10")
    records = run_cycle(_fund(), "2024-06-03", broker,
                        FakeDataClient(CLOSES), UNIVERSE)

    events = journal.entries()
    # One intent and one fill per order the pipeline decided to send.
    assert journal_summary(events)["by_event"] == {"intent": len(records.orders), "fill": len(records.orders)}
    assert unfilled_intents(events) == []
    assert {e.client_order_id for e in events if e.event == "intent"} == {
        o.client_order_id for o in records.orders
    }


def _fund(views=None):
    """A one-strategy fund staffed with a fake under a real registry key.

    `pipeline.test_run_cycle._spec()` leans on an autouse fixture that stubs
    "a"/"b" into the registry, which is scoped to that module — so build the
    spec here against "pead" and inject the fake through Fund(models=...).
    """
    spec = FundSpec(
        schema_version=2,
        name="test-fund",
        strategies=[{"name": "solo", "models": [{"name": "pead"}],
                     "blend": {"mode": "long_short"}}],
        risk={"max_position_pct": 0.25, "max_gross_exposure": 1.0},
        capital=100_000.0,
    )
    staff = FakeAnalyst("pead", views if views is not None else {"AAPL": 0.8, "MSFT": -0.6})
    return Fund(spec, models={"solo": [staff]})
