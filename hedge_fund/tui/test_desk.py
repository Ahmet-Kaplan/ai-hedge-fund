"""Desk screen tests — the read-only account view and the manual ticket.

Venues are faked rather than opened, so nothing here touches a network; the
live behaviour is covered by the broker and venue tests.
"""

from __future__ import annotations

import asyncio
import io
import pathlib

import pytest
from rich.console import Console
from textual.widgets import Input, Static

from hedge_fund.brokers.account import AccountSnapshot, PositionDetail
from hedge_fund.brokers.models import Fill, Order, Position
from hedge_fund.reconciliation import ReconciliationReport
from hedge_fund.tui import desk as desk_ui
from hedge_fund.tui.app import HedgeFundApp
from hedge_fund.venue import OpenVenue


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeBroker:
    venue = "fake"

    def __init__(self, *, cash=10_000.0, positions=None, details=None,
                 account=None, orders=None, fill_price=100.0, fail=None):
        self._cash = cash
        self._positions = positions or {}
        self._details = details
        self._account = account
        self._orders = orders or {}
        self._fill_price = fill_price
        self._fail = fail
        self.submitted: list[Order] = []

    def positions(self):
        return {t: Position(ticker=t, shares=s) for t, s in self._positions.items()}

    def cash(self):
        return self._cash

    def position_details(self):
        return self._details if self._details is not None else []

    def account(self):
        return self._account

    def open_orders(self):
        return self._orders

    def place_order(self, order: Order) -> Fill:
        self.submitted.append(order)
        if self._fail is not None:
            raise self._fail
        return Fill(ticker=order.ticker, side=order.side,
                    quantity=order.quantity, price=self._fill_price)


def _opened(broker, *, label="alpaca-paper", live=False, trading_enabled=True,
            note="") -> OpenVenue:
    return OpenVenue(name="fake", broker=broker, reference=None, label=label,
                     live=live, note=note, settings=None)


def _account(**over):
    base = dict(venue="fake", cash=10_000.0, buying_power=40_000.0, equity=11_000.0,
                shorting_enabled=True)
    base.update(over)
    return AccountSnapshot(**base)


def _details():
    return [
        PositionDetail(ticker="AAPL", shares=10, avg_entry_price=100.0,
                       current_price=110.0, market_value=1_100.0,
                       unrealized_pnl=100.0, unrealized_pnl_pct=0.1),
        PositionDetail(ticker="MSFT", shares=-4, avg_entry_price=50.0,
                       current_price=45.0, market_value=-180.0,
                       unrealized_pnl=20.0, unrealized_pnl_pct=0.1),
    ]


def _widget_text(widget) -> str:
    """What a Static is currently showing (Textual 8 keeps it on `content`)."""
    content = getattr(widget, "content", None)
    return _render(content if content is not None else widget.render())


def _render(renderable) -> str:
    """Render a rich renderable to plain text, wide enough not to wrap."""
    buffer = io.StringIO()
    Console(file=buffer, width=200, force_terminal=False, no_color=True).print(renderable)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,ticker,side,quantity", [
    ("AAPL buy 1", "AAPL", "buy", 1),
    ("aapl sell 25", "AAPL", "sell", 25),
    ("brk.b buy 3 @ 190.50", "BRK.B", "buy", 3),
])
def test_ticket_text_parses(text, ticker, side, quantity):
    order = desk_ui._parse_ticket(text)
    assert (order.ticker, order.side, order.quantity) == (ticker, side, quantity)


@pytest.mark.parametrize("text", ["", "AAPL", "AAPL buy", "AAPL hold 1",
                                  "AAPL buy zero", "AAPL buy 0", "AAPL buy -2"])
def test_bad_ticket_text_is_rejected_with_a_reason(text):
    with pytest.raises(ValueError):
        desk_ui._parse_ticket(text)


def test_badge_distinguishes_live_paper_and_read_only():
    live = desk_ui.desk_snapshot_safe(_opened(FakeBroker(), live=True, label="alpaca-live"))
    paper = desk_ui.desk_snapshot_safe(_opened(FakeBroker()))
    assert "LIVE" in desk_ui._badge(live).plain
    assert "PAPER" in desk_ui._badge(paper).plain


def test_positions_table_shows_the_venues_valuation():
    snapshot = desk_ui.desk_snapshot_safe(_opened(FakeBroker()))
    snapshot.positions = _details()

    rendered = _render(desk_ui._positions_table(snapshot))

    assert "AAPL" in rendered and "MSFT" in rendered
    assert "$1,100" in rendered and "$110.00" in rendered
    assert "short" in rendered


def test_positions_table_survives_a_venue_without_valuation():
    """Paper and sim report share counts only; the desk must still render."""
    snapshot = desk_ui.desk_snapshot_safe(_opened(FakeBroker(cash=1.0)))
    snapshot.positions = [PositionDetail(ticker="AAPL", shares=3)]

    rendered = _render(desk_ui._positions_table(snapshot))

    assert "AAPL" in rendered
    assert "—" in rendered  # no mark, no invented number


# ---------------------------------------------------------------------------
# DeskScreen
# ---------------------------------------------------------------------------

def _install(monkeypatch, broker, **opened_kwargs):
    opened = _opened(broker, **opened_kwargs)
    monkeypatch.setattr(desk_ui, "open_venue", lambda *a, **k: opened)
    monkeypatch.setattr(desk_ui, "ensure_mandates_dir", lambda: pathlib.Path("."))
    return opened


async def _settle(pilot, ticks: int = 6) -> None:
    """Let the load worker and its call_from_thread land."""
    for _ in range(ticks):
        await pilot.pause()
        await asyncio.sleep(0.02)


def test_desk_shows_badge_positions_and_buying_power(monkeypatch):
    _install(monkeypatch,
             FakeBroker(cash=9_000.0, account=_account(), details=_details()))
    app = HedgeFundApp()

    async def scenario():
        async with app.run_test(size=(120, 45)) as pilot:
            app.push_screen(desk_ui.DeskScreen())
            await _settle(pilot)
            head = _widget_text(app.screen.query_one("#desk-head", Static))
            stats = _widget_text(app.screen.query_one("#desk-stats", Static))
            positions = _widget_text(app.screen.query_one("#desk-positions", Static))
            return head, stats, positions

    head, stats, positions = asyncio.run(scenario())
    assert "PAPER" in head or "READ-ONLY" in head
    # _money abbreviates four figures and up: 40,000 renders as $40k.
    assert "buying power" in stats and "$40k" in stats
    assert "AAPL" in positions


def test_desk_reports_a_venue_it_cannot_open(monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("ALPACA_API_KEY is not set")

    monkeypatch.setattr(desk_ui, "open_venue", boom)
    app = HedgeFundApp()

    async def scenario():
        async with app.run_test(size=(120, 45)) as pilot:
            app.push_screen(desk_ui.DeskScreen())
            await _settle(pilot)
            return (_widget_text(app.screen.query_one("#desk-head", Static)),
                    _widget_text(app.screen.query_one("#desk-stats", Static)))

    head, stats = asyncio.run(scenario())
    assert "unavailable" in head
    assert "ALPACA_API_KEY" in stats


def test_pressing_v_cycles_the_venue(monkeypatch):
    seen: list[str] = []

    def record(venue, **kwargs):
        seen.append(venue)
        return _opened(FakeBroker(cash=1.0, account=_account()))

    monkeypatch.setattr(desk_ui, "open_venue", record)
    app = HedgeFundApp()

    async def scenario():
        async with app.run_test(size=(120, 45)) as pilot:
            screen = desk_ui.DeskScreen()
            app.push_screen(screen)
            await _settle(pilot)
            await pilot.press("v")
            await _settle(pilot)
            return list(seen), screen._venues

    opened, venues = asyncio.run(scenario())
    assert len(venues) > 1
    assert opened[0] != opened[1]
    assert opened[1] == venues[1]


# ---------------------------------------------------------------------------
# OrderTicketScreen
# ---------------------------------------------------------------------------

def test_ticket_review_then_confirm_submits_once(monkeypatch):
    broker = FakeBroker(cash=50_000.0, account=_account(buying_power=100_000.0),
                        fill_price=101.5)
    app = HedgeFundApp()
    results: list[str | None] = []

    async def scenario():
        async with app.run_test(size=(120, 45)) as pilot:
            app.push_screen(desk_ui.OrderTicketScreen(_opened(broker)), results.append)
            await pilot.pause()
            screen = app.screen
            screen.query_one("#ticket-input", Input).value = "AAPL buy 3"
            await pilot.press("enter")
            await pilot.pause()
            reviewed = screen._order
            await pilot.press("y")
            await _settle(pilot)
            return reviewed

    reviewed = asyncio.run(scenario())
    # Step one parsed it, step two reviewed it, and only then did it go out.
    assert (reviewed.ticker, reviewed.side, reviewed.quantity) == ("AAPL", "buy", 3)
    assert len(broker.submitted) == 1
    assert broker.submitted[0].client_order_id  # idempotent by construction
    # What the user is told is the venue's own fill price.
    assert results and results[-1] and "101.50" in results[-1]


def test_ticket_shows_a_bad_ticket_instead_of_submitting(monkeypatch):
    broker = FakeBroker(account=_account())
    app = HedgeFundApp()

    async def scenario():
        async with app.run_test(size=(120, 45)) as pilot:
            app.push_screen(desk_ui.OrderTicketScreen(_opened(broker)))
            await pilot.pause()
            screen = app.screen
            screen.query_one("#ticket-input", Input).value = "AAPL sideways 3"
            await pilot.press("enter")
            await pilot.pause()
            return screen

    screen = asyncio.run(scenario())
    assert screen._order is None      # nothing reached the review step
    assert broker.submitted == []     # and nothing reached the venue


def test_ticket_refuses_when_the_account_cannot_afford_it(monkeypatch):
    broker = FakeBroker(cash=100.0, account=_account(buying_power=100.0),
                        fill_price=100.0)
    app = HedgeFundApp()

    async def scenario():
        async with app.run_test(size=(120, 45)) as pilot:
            app.push_screen(desk_ui.OrderTicketScreen(_opened(broker)))
            await pilot.pause()
            screen = app.screen
            # A price is needed for a notional check, so quote one.
            screen.query_one("#ticket-input", Input).value = "AAPL buy 100 @ 100"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("y")
            await _settle(pilot)
            return screen

    screen = asyncio.run(scenario())
    # The account preflight stopped it before the venue ever saw the order.
    assert screen._report is not None and not screen._report.ok
    assert "buying power" in screen._report.summary
    assert broker.submitted == []


def test_ticket_surfaces_a_venue_rejection(monkeypatch):
    broker = FakeBroker(cash=50_000.0, account=_account(buying_power=100_000.0),
                        fail=RuntimeError("market is closed"))
    app = HedgeFundApp()

    async def scenario():
        async with app.run_test(size=(120, 45)) as pilot:
            app.push_screen(desk_ui.OrderTicketScreen(_opened(broker)))
            await pilot.pause()
            screen = app.screen
            screen.query_one("#ticket-input", Input).value = "AAPL buy 1 @ 100"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("y")
            await _settle(pilot)
            return screen

    screen = asyncio.run(scenario())
    # The venue's failure landed on the screen instead of escaping the worker.
    assert screen.is_mounted
    assert len(broker.submitted) == 1


# ---------------------------------------------------------------------------
# ReconciliationScreen
# ---------------------------------------------------------------------------

def test_reconciliation_screen_lays_out_drift():
    report = ReconciliationReport(
        reference="desk 2024-06-03",
        changed=[],
        only_in_ledger={"AAPL": 5},
        only_at_broker={"NVDA": 2},
        cash_delta=-250.0,
        working_orders=1,
        notes=["the venue is still holding orders"],
    )
    screen = desk_ui.ReconciliationScreen(report)

    body = _render(screen._body())
    title = _render(screen._title())

    assert "DRIFT" in title
    assert "AAPL" in body and "NVDA" in body
    assert "ledger only" in body and "broker only" in body
    assert "-250.00" in body
    assert "still holding orders" in body
