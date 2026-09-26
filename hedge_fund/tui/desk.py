"""Desk screens — watching a venue, and placing one order by hand.

Three surfaces, all built on the read model in `hedge_fund.desk` so the TUI and
the web dashboard cannot disagree about what an account looks like:

`DeskScreen`
    Positions with the venue's own valuation, cash and buying power, working
    orders, and the reconciliation against the newest receipt. Read-only: it
    opens a venue and calls nothing that submits.

`OrderTicketScreen`
    A manual order in two steps — fill the ticket, then review it against the
    venue's own pre-trade checks. The confirm step states the venue and whether
    it is live, and the submit path is the same one the pipeline uses:
    `client_order_id` for idempotency, the order journal, the account
    preflight, and the venue's own gates.

`ReconciliationScreen`
    One receipt's claim versus the venue, laid out so drift is legible.

Venues are opened through `hedge_fund.venue`, so the paper/live/confirmation
gates apply here exactly as they do on the command line — the ticket cannot
become a way around them.
"""

from __future__ import annotations

import os
from datetime import date as _date

from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Input, OptionList, Static
from textual.widgets.option_list import Option

from hedge_fund.brokers.models import Order
from hedge_fund.desk import DeskSnapshot, desk_snapshot
from hedge_fund.journal import FileOrderJournal, JournalledBroker
from hedge_fund.paths import ensure_mandates_dir, journal_path
from hedge_fund.pipeline.execution import client_order_id
from hedge_fund.reconciliation import ReconciliationReport
from hedge_fund.tui.shared import BRIGHT, CYAN, GREEN, MUTED, RED, TEXT, _money
from hedge_fund.venue import VENUES, open_venue


def _badge(snapshot: DeskSnapshot) -> Text:
    """The label an operator must not misread."""
    if snapshot.live:
        return Text(" LIVE ", style=f"bold white on {RED}")
    if not snapshot.trading_enabled:
        return Text(" READ-ONLY ", style=f"bold {MUTED} on #22302a")
    return Text(" PAPER ", style=f"bold black on {GREEN}")


def _opt_money(value: float | None) -> str:
    """`_money` for a field the venue may not supply (paper/sim have no marks)."""
    return "—" if value is None else _money(value)


def _price(value: float | None) -> str:
    """A per-share figure to the cent.

    `_money` is built for portfolio totals and rounds: fine for a position's
    value, wrong for a price you are about to trade at.
    """
    return "—" if value is None else f"${value:,.2f}"


def _signed(value: float | None, *, money: bool = True) -> Text:
    if value is None:
        return Text("—", style=MUTED)
    tone = GREEN if value > 0 else (RED if value < 0 else MUTED)
    body = f"{value:+,.2f}" if money else f"{value:+g}"
    return Text(body, style=tone)


def _positions_table(snapshot: DeskSnapshot) -> Table:
    table = Table(box=box.SQUARE, header_style="bold", border_style="#1f2b25",
                  expand=True)
    table.add_column("Ticker", style=f"bold {CYAN}")
    table.add_column("Side", style=MUTED)
    table.add_column("Shares", justify="right")
    table.add_column("Avg entry", justify="right")
    table.add_column("Mark", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Unrealized", justify="right")
    table.add_column("%", justify="right")

    if not snapshot.positions:
        table.add_row(Text("flat — no open positions", style=MUTED),
                      "", "", "", "", "", "", "")
        return table

    for position in snapshot.positions:
        pct = position.unrealized_pnl_pct
        table.add_row(
            position.ticker,
            position.side,
            f"{position.shares:,}",
            _price(position.avg_entry_price),
            _price(position.current_price),
            _opt_money(position.market_value),
            _signed(position.unrealized_pnl),
            "—" if pct is None else Text(f"{pct * 100:+.2f}%",
                                         style=GREEN if pct > 0 else (RED if pct < 0 else MUTED)),
        )
    return table


def _stats(snapshot: DeskSnapshot) -> Table:
    table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
    table.add_column(style=MUTED)
    table.add_column(style=TEXT, justify="right")
    table.add_row("cash", _money(snapshot.cash))
    if snapshot.buying_power is not None:
        table.add_row("buying power", _money(snapshot.buying_power))
    if snapshot.equity is not None:
        table.add_row("equity", _money(snapshot.equity))
    if snapshot.gross_exposure is not None:
        table.add_row("gross exposure", _money(snapshot.gross_exposure))
    if snapshot.unrealized_pnl is not None:
        table.add_row("unrealized", _signed(snapshot.unrealized_pnl))
    if snapshot.shorting_enabled is not None:
        table.add_row("shorting", "allowed" if snapshot.shorting_enabled else "not permitted")
    if snapshot.pattern_day_trader is not None:
        table.add_row("day trades", f"{snapshot.daytrade_count}"
                      + ("  (pattern day trader)" if snapshot.pattern_day_trader else ""))
    table.add_row("working orders", str(snapshot.working_orders))
    return table


class ReconciliationScreen(ModalScreen[None]):
    """One receipt's claim versus the venue, laid out so drift is legible."""

    BINDINGS = [Binding("escape", "close", "close")]

    def __init__(self, report: ReconciliationReport) -> None:
        super().__init__()
        self._report = report

    def compose(self) -> ComposeResult:
        with Vertical(id="recon"):
            yield Static(self._title(), id="recon-title")
            yield Static(self._body(), id="recon-body")
            yield Static(Text("esc to close", style=MUTED), id="recon-foot")

    def _title(self) -> Text:
        r = self._report
        return Text.assemble(
            ("RECONCILIATION  ", f"bold {BRIGHT}"),
            (r.reference or "no reference receipt", MUTED),
            ("   ", MUTED),
            ("clean" if r.clean else "DRIFT", f"bold {GREEN if r.clean else RED}"),
        )

    def _body(self) -> Group:
        r = self._report
        rows = Table(box=box.SIMPLE, header_style="bold", border_style="#1f2b25")
        rows.add_column("Ticker", style=f"bold {CYAN}")
        rows.add_column("Ledger", justify="right")
        rows.add_column("Broker", justify="right")
        rows.add_column("Note", style=MUTED)
        for drill in r.changed:
            rows.add_row(drill.ticker, f"{drill.expected:,}", f"{drill.actual:,}",
                         f"delta {drill.delta:+,}")
        for ticker, shares in r.only_in_ledger.items():
            rows.add_row(ticker, f"{shares:,}", "—", "ledger only")
        for ticker, shares in r.only_at_broker.items():
            rows.add_row(ticker, "—", f"{shares:,}", "broker only")
        if not (r.changed or r.only_in_ledger or r.only_at_broker):
            rows.add_row("—", "", "", "positions agree")

        items = [rows, Text("")]
        if r.cash_delta is not None:
            items.append(Text.assemble(("cash difference  ", MUTED),
                                       _signed(r.cash_delta)))
        items.append(Text.assemble(("working orders  ", MUTED),
                                   (str(r.working_orders),
                                    RED if r.working_orders else TEXT)))
        for note in r.notes:
            items.append(Text.assemble(("• ", MUTED), (note, MUTED)))
        return Group(*items)

    def action_close(self) -> None:
        self.dismiss(None)


class OrderTicketScreen(ModalScreen[str | None]):
    """A manual order: fill the ticket, review it, then confirm.

    The review step is not decoration. It states the venue, whether that venue
    is live, the notional, and what the account's own pre-trade checks say —
    all the things that are cheap to look at and expensive to skip.
    """

    BINDINGS = [
        Binding("escape", "cancel", "cancel"),
        Binding("y", "confirm", "confirm", show=False),
        Binding("n", "cancel", "back", show=False),
    ]

    def __init__(self, opened) -> None:
        super().__init__()
        self._opened = opened
        self._order: Order | None = None
        self._report = None
        self._sequence = 0
        self._result: str | None = None

    # -- step 1: the ticket -------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="ticket"):
            yield Static(Text.assemble(
                ("NEW ORDER  ", f"bold {BRIGHT}"),
                (self._opened.label, MUTED),
                ("   ", MUTED),
                _badge(desk_snapshot_safe(self._opened)),
            ), id="ticket-title")
            yield Static(Text("ticker, side, quantity — one per line, e.g. "
                              "'AAPL buy 1'", MUTED), id="ticket-hint")
            yield Input(placeholder="AAPL buy 1", id="ticket-input")
            yield Static("", id="ticket-body")

    def on_mount(self) -> None:
        self.query_one("#ticket-input", Input).focus()

    @on(Input.Submitted, "#ticket-input")
    def _parse(self, event: Input.Submitted) -> None:
        try:
            self._order = _parse_ticket(event.value)
        except ValueError as exc:
            self.query_one("#ticket-body", Static).update(Text(str(exc), style=RED))
            return
        self._review()

    # -- step 2: the review -------------------------------------------------

    def _review(self) -> None:
        order = self._order
        assert order is not None
        broker = self._opened.broker
        marks = {order.ticker: order.price}

        from hedge_fund.brokers.account import account_snapshot, preflight

        account = account_snapshot(broker)
        self._report = None
        if account is not None:
            self._report = preflight(account, [order], broker.positions())

        lines: list[Text] = [
            Text.assemble(("review  ", f"bold {BRIGHT}"),
                          (f"{order.side.upper()} {order.quantity:,} {order.ticker}", f"bold {TEXT}")),
            Text.assemble(("venue   ", MUTED), (self._opened.label, TEXT),
                          ("   ", MUTED), _badge(desk_snapshot_safe(self._opened))),
            Text.assemble(("mark    ", MUTED), (_price(order.price), TEXT),
                          ("   notional  ", MUTED),
                          (_money(order.price * order.quantity), TEXT)),
        ]
        if self._report is not None:
            tone = GREEN if self._report.ok else RED
            lines.append(Text.assemble(("account ", MUTED),
                                       (self._report.summary, tone)))
        else:
            lines.append(Text.assemble(("account ", MUTED),
                                       ("this venue reports no account to check", MUTED)))
        lines.append(Text(""))
        if self._opened.live:
            lines.append(Text("this is a LIVE account — y submits for real",
                              style=f"bold {RED}"))
        else:
            lines.append(Text("y to submit   ·   n to cancel", style=MUTED))

        self.query_one("#ticket-input", Input).display = False
        self.query_one("#ticket-hint", Static).update(
            Text("prices come from the venue's own mark", MUTED)
        )
        self.query_one("#ticket-body", Static).update(Group(*lines))
        self.refresh_bindings()

    def action_confirm(self) -> None:
        if self._order is None:
            return
        if self._report is not None and not self._report.ok:
            self.query_one("#ticket-body", Static).update(
                Text(f"refused: {self._report.summary}", style=RED)
            )
            return
        self._sequence += 1
        self._submit()

    @work(thread=True, exclusive=True)
    def _submit(self) -> None:
        app = self.app
        order = self._order
        assert order is not None
        session = _date.today().isoformat()
        stamped = order.model_copy(update={
            "client_order_id": client_order_id("manual", session, self._sequence, order),
        })
        journal = FileOrderJournal(journal_path("manual"))
        broker = JournalledBroker(self._opened.broker, journal,
                                  fund="manual", session=session)
        try:
            fill = broker.place_order(stamped)
        except Exception as exc:
            # A ticket is a UI action: every venue failure (rejected, closed,
            # timed out, gate closed) belongs on the screen, not in a crash.
            app.call_from_thread(
                self._failed, f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            )
            return
        app.call_from_thread(
            self._filled,
            f"filled {fill.quantity:,} {fill.ticker} at {_price(fill.price)}",
        )

    def _filled(self, message: str) -> None:
        self._result = message
        self.dismiss(message)

    def _failed(self, message: str) -> None:
        self.query_one("#ticket-body", Static).update(Text(f"not filled: {message}", style=RED))

    def action_cancel(self) -> None:
        self.dismiss(None)


def _parse_ticket(text: str) -> Order:
    """'AAPL buy 1' or 'AAPL sell 2 @ 190.50' -> an Order."""
    parts = text.replace("@", " ").split()
    if len(parts) < 3:
        raise ValueError("expected '<ticker> <buy|sell> <quantity> [@ price]'")
    ticker, side, raw_quantity = parts[0].upper(), parts[1].lower(), parts[2]
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be buy or sell, got {side!r}")
    try:
        quantity = int(raw_quantity)
    except ValueError as exc:
        raise ValueError(f"quantity must be a whole number, got {raw_quantity!r}") from exc
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    price = 0.0
    if len(parts) >= 4:
        try:
            price = float(parts[3])
        except ValueError as exc:
            raise ValueError(f"price must be a number, got {parts[3]!r}") from exc
    return Order(ticker=ticker, side=side, quantity=quantity, price=price)


def desk_snapshot_safe(opened) -> DeskSnapshot:
    """A badge-only snapshot, for screens that must not hit the network.

    The ticket needs the PAPER/LIVE/READ-ONLY label before it has any account
    data; asking the venue again just for a colour would be a wasted call.
    """
    return DeskSnapshot(
        venue=opened.label,
        live=opened.live,
        trading_enabled=opened.trading_enabled,
        note=opened.note,
        cash=0.0,
    )


class DeskScreen(Screen):
    """What the account holds right now. Read-only."""

    BINDINGS = [
        Binding("escape", "back", "back"),
        Binding("r", "refresh", "refresh"),
        Binding("v", "cycle_venue", "venue"),
        Binding("o", "order_ticket", "new order"),
        Binding("c", "reconciliation", "reconcile"),
    ]

    def __init__(self, venue: str | None = None) -> None:
        super().__init__()
        self._venues = _available_venues()
        self._index = self._venues.index(venue) if venue in self._venues else 0
        self._opened = None
        self._snapshot: DeskSnapshot | None = None
        self._error: str | None = None

    @property
    def venue(self) -> str:
        return self._venues[self._index]

    def compose(self) -> ComposeResult:
        with Vertical(id="desk"):
            yield Static("", id="desk-head")
            yield Static("", id="desk-stats")
            yield Static("", id="desk-positions")
            yield Static("", id="desk-warnings")
        yield Footer()

    def on_mount(self) -> None:
        self._load()

    def action_refresh(self) -> None:
        self._load()

    def action_cycle_venue(self) -> None:
        self._index = (self._index + 1) % len(self._venues)
        self._load()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_order_ticket(self) -> None:
        if self._opened is None:
            self.notify("nothing loaded yet", severity="warning")
            return
        self.app.push_screen(OrderTicketScreen(self._opened), self._after_ticket)

    def _after_ticket(self, message: str | None) -> None:
        if message:
            self.notify(message, severity="information")
            self._load()

    def action_reconciliation(self) -> None:
        if self._snapshot is None or self._snapshot.reconciliation is None:
            self.notify("no receipt to reconcile against yet", severity="warning")
            return
        self.app.push_screen(ReconciliationScreen(self._snapshot.reconciliation))

    @work(thread=True, exclusive=True)
    def _load(self) -> None:
        app = self.app
        venue = self.venue
        app.call_from_thread(self._show_loading, venue)
        try:
            opened = open_venue(venue, fund_name="desk", capital=0.0,
                                receipts=ensure_mandates_dir())
            snapshot = desk_snapshot(opened)
        except Exception as exc:  # a monitoring screen must not take the app down
            app.call_from_thread(self._fail, venue, str(exc))
            return
        app.call_from_thread(self._show, opened, snapshot)

    def _show_loading(self, venue: str) -> None:
        self.query_one("#desk-head", Static).update(
            Text.assemble(("DESK  ", f"bold {BRIGHT}"),
                          (venue, TEXT), ("   loading…", MUTED))
        )

    def _fail(self, venue: str, message: str) -> None:
        self._error = message
        self._opened = None
        self._snapshot = None
        self.query_one("#desk-head", Static).update(
            Text.assemble(("DESK  ", f"bold {BRIGHT}"), (venue, TEXT),
                          ("   unavailable", f"bold {RED}"))
        )
        self.query_one("#desk-stats", Static).update(Text(message, style=MUTED))
        self.query_one("#desk-positions", Static).update("")
        self.query_one("#desk-warnings", Static).update(
            Text("v cycles venue · r retries", style=MUTED))

    def _show(self, opened, snapshot: DeskSnapshot) -> None:
        self._opened = opened
        self._snapshot = snapshot
        self._error = None
        self.query_one("#desk-head", Static).update(Text.assemble(
            ("DESK  ", f"bold {BRIGHT}"),
            (snapshot.venue, TEXT),
            ("   ", MUTED),
            _badge(snapshot),
            ("   ", MUTED),
            (f"v: {'/'.join(self._venues)}", MUTED),
        ))
        self.query_one("#desk-stats", Static).update(_stats(snapshot))
        self.query_one("#desk-positions", Static).update(_positions_table(snapshot))
        warnings = snapshot.warnings
        if warnings:
            self.query_one("#desk-warnings", Static).update(Group(*[
                Text.assemble(("! ", RED), (warning, MUTED)) for warning in warnings
            ]))
        else:
            self.query_one("#desk-warnings", Static).update(
                Text("no drift, no orders in flight", style=MUTED))


def _available_venues() -> list[str]:
    """Alpaca first when it is configured — that is the account worth watching.

    Ordering is by usefulness, not preference: if there are no Alpaca keys the
    venue would only fail, so it is left out entirely.
    """
    venues: list[str] = []
    if os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID"):
        venues.append("alpaca")
    venues.extend(v for v in VENUES if v != "alpaca")
    return venues
