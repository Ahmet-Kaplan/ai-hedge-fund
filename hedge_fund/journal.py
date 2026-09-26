"""Append-only order journal — the crash-safe half of execution.

`execute_decision` submits orders in a loop and only writes a `CycleRecord`
once every one of them has filled:

    fills = [broker.place_order(order) for order in orders]

If order 3 of 5 raises, orders 1 and 2 are already **live at the venue** but
nothing was recorded. Positions self-heal (the next cycle reads them back from
the broker), the audit trail does not: the fills, their prices, and the
reasoning behind them are gone.

The journal closes that gap without touching the pipeline. `JournalledBroker`
wraps any broker and writes one line *before* each submission (the intent) and
one line *after* it (the fill, or the error). Because it is a decorator it
works for every venue — sim, paper, Alpaca — and the pipeline stays
broker-agnostic and unaware.

Format: JSONL, one event per line, appended and fsynced, so a killed process
loses at most the line it was writing. `entries()` reads it back;
`unfilled_intents()` answers the operational question that matters after a
crash: which orders did we send that we have no fill for?
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from hedge_fund.brokers.models import Fill, Order, Position

EventName = Literal["intent", "fill", "error"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OrderEvent(BaseModel):
    """One step in an order's life. Appended, never rewritten."""

    ts: str
    event: EventName
    fund: str
    session: str
    client_order_id: str | None = None
    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    price: float
    filled_quantity: int | None = None
    filled_price: float | None = None
    error: str | None = None
    venue: str | None = None


class OrderJournal(Protocol):
    """Where order events go. `NullJournal` drops them; the file one keeps them."""

    def record(self, event: OrderEvent) -> None: ...

    def entries(self) -> list[OrderEvent]: ...


class NullJournal:
    """No journal configured — the pre-journal behaviour."""

    def record(self, event: OrderEvent) -> None:
        return

    def entries(self) -> list[OrderEvent]:
        return []


class MemoryJournal:
    """In-memory journal for tests and dry runs."""

    def __init__(self) -> None:
        self._events: list[OrderEvent] = []

    def record(self, event: OrderEvent) -> None:
        self._events.append(event)

    def entries(self) -> list[OrderEvent]:
        return list(self._events)


class FileOrderJournal:
    """Append-only JSONL on disk, flushed and fsynced per event."""

    def __init__(self, path: Path | str, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self._fsync = fsync

    def record(self, event: OrderEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = event.model_dump_json() + "\n"
        # One open/close per event: an interrupted process must not be able to
        # leave a half-written line buffered in memory.
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            if self._fsync:
                os.fsync(handle.fileno())

    def entries(self) -> list[OrderEvent]:
        if not self.path.exists():
            return []
        events: list[OrderEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(OrderEvent.model_validate_json(line))
        return events


class JournalledBroker:
    """A broker that journals every submission. Same three methods.

    Reads pass straight through. `place_order` writes the intent first, so an
    order that reaches the venue is always on disk even if the process dies
    before the fill comes back.
    """

    def __init__(
        self,
        inner: Any,
        journal: OrderJournal,
        *,
        fund: str,
        session: str,
        clock: Any = None,
    ) -> None:
        self._inner = inner
        self._journal = journal
        self._fund = fund
        self._session = session
        self._clock = clock or _utc_now

    @property
    def venue(self) -> str:
        return getattr(self._inner, "venue", type(self._inner).__name__)

    @property
    def inner(self) -> Any:
        return self._inner

    def positions(self) -> dict[str, Position]:
        return self._inner.positions()

    def cash(self) -> float:
        return self._inner.cash()

    def place_order(self, order: Order) -> Fill:
        self._journal.record(self._event("intent", order))
        try:
            fill = self._inner.place_order(order)
        except BaseException as exc:
            # Record the failure before it propagates: an order may have
            # reached the venue even though we never saw a fill.
            self._journal.record(
                self._event("error", order, error=f"{type(exc).__name__}: {exc}")
            )
            raise
        self._journal.record(
            self._event(
                "fill", order,
                filled_quantity=fill.quantity, filled_price=fill.price,
            )
        )
        return fill

    def _event(self, event: EventName, order: Order, **extra: Any) -> OrderEvent:
        return OrderEvent(
            ts=self._clock(),
            event=event,
            fund=self._fund,
            session=self._session,
            client_order_id=order.client_order_id,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            venue=self.venue,
            **extra,
        )


def unfilled_intents(events: list[OrderEvent]) -> list[OrderEvent]:
    """Intents with no fill and no error — orders we cannot account for.

    The set an operator must reconcile by hand after a crash: the venue may
    hold them, and the fund's books do not know about them.

    Events are folded in order, so a fill resolves against the intent it
    belongs to. `client_order_id` is the key when both sides carry one; the
    pipeline always stamps them, but a caller using `JournalledBroker`
    directly may not, so identities fall back to the instruction itself
    (ticker, side, quantity, price) consumed oldest-first.
    """
    pending: list[OrderEvent] = []
    for event in events:
        if event.event == "intent":
            pending.append(event)
            continue
        for index, intent in enumerate(pending):
            if _same_instruction(intent, event):
                del pending[index]
                break
    return pending


def _same_instruction(intent: OrderEvent, outcome: OrderEvent) -> bool:
    if intent.client_order_id is not None and outcome.client_order_id is not None:
        return intent.client_order_id == outcome.client_order_id
    return (
        intent.ticker == outcome.ticker
        and intent.side == outcome.side
        and intent.quantity == outcome.quantity
        and intent.price == outcome.price
    )


def journal_summary(events: list[OrderEvent]) -> dict[str, Any]:
    """Counts by event, for a quick operational read."""
    by_event: dict[str, int] = {}
    for event in events:
        by_event[event.event] = by_event.get(event.event, 0) + 1
    return {
        "total": len(events),
        "by_event": by_event,
        "unfilled_intents": len(unfilled_intents(events)),
        "filled_quantity": sum(e.filled_quantity or 0 for e in events if e.event == "fill"),
    }
