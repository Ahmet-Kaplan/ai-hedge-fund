"""Venue registry — one place that opens a book, whoever the broker is.

`paper`, `sim` and `alpaca` are the three books a cycle can run against. They
differ in where orders go, but a caller should not have to know that: it asks
for a venue and gets back the broker plus the ledger's claim about that
account, which `run_cycle` reconciles before trading.

Kept out of `hedge_fund/brokers/` on purpose: `hedge_fund.ledger` already
imports the brokers, so a venue module under that package would close a cycle.

The Alpaca branch adds no new safety of its own — it constructs
`AlpacaSettings`, which is where the paper/live/confirmation gates live. This
module's job is to state plainly what was opened (`label`, `live`, `note`) so
a caller can show it and a test can assert it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from hedge_fund.brokers.alpaca import AlpacaBroker, AlpacaSettings
from hedge_fund.brokers.paper import PaperBroker
from hedge_fund.brokers.sim import SimBroker
from hedge_fund.ledger import broker_for_run, latest_reference
from hedge_fund.reconciliation import LedgerReference

VENUES = ("paper", "sim", "alpaca")
VenueName = Literal["paper", "sim", "alpaca"]


@dataclass(frozen=True)
class OpenVenue:
    """An opened book, and what to tell the operator about it."""

    name: str
    broker: Any
    reference: LedgerReference | None
    label: str
    live: bool
    note: str
    settings: AlpacaSettings | None = None

    @property
    def trading_enabled(self) -> bool:
        """Whether this venue would actually submit, rather than only read."""
        return bool(self.settings is not None and self.settings.refuse_reason() is None)


def open_venue(
    venue: str,
    *,
    fund_name: str,
    capital: float,
    receipts: Path,
) -> OpenVenue:
    """Open *venue* for *fund_name*.

    `paper`/`sim` are seeded from the newest receipt when one exists — the
    same book the previous run ended with. `alpaca` reads the real account
    instead; the receipt is only the ledger's claim, handed back as the
    reconciliation reference.
    """
    if venue not in VENUES:
        raise ValueError(
            f"unknown venue {venue!r}; available: {', '.join(VENUES)}"
        )

    if venue == "paper":
        broker, prior = broker_for_run(fund_name, capital, receipts)
        return OpenVenue(
            name=venue,
            broker=broker,
            reference=LedgerReference.from_cycle_record(prior) if prior else None,
            label="paper",
            live=False,
            note="paper venue · live clock · fills at mark · no live venue",
        )

    if venue == "sim":
        reference = latest_reference(fund_name, receipts)
        if reference is None:
            broker = SimBroker(cash=capital)
        else:
            broker = SimBroker(cash=reference.cash, positions=dict(reference.positions))
        return OpenVenue(
            name=venue,
            broker=broker,
            reference=reference,
            label="sim",
            live=False,
            note="sim venue · deterministic fills at the reference price",
        )

    # alpaca: reads work with no extra opt-in; the settings decide whether
    # orders may be submitted, and refuse_reason() says why not.
    settings = AlpacaSettings.from_env()
    broker = AlpacaBroker(settings)
    reference = latest_reference(fund_name, receipts)
    refusal = settings.refuse_reason()
    if refusal is None and not settings.paper:
        note = f"LIVE venue {settings.venue} · orders WILL be submitted for real"
    elif refusal is None:
        note = f"{settings.venue} · orders will be submitted to the paper account"
    else:
        note = f"{settings.venue} · READ-ONLY ({refusal})"
    return OpenVenue(
        name=venue,
        broker=broker,
        reference=reference,
        label=settings.venue,
        live=not settings.paper,
        note=note,
        settings=settings,
    )
