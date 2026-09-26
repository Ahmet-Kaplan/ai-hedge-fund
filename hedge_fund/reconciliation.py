"""Reconciliation — compare the broker's book with what the ledger expects.

The pipeline sizes from `broker.positions()`, so a drifted broker does not by
itself produce a wrong order. What it does produce is a wrong *story*: the
fund's receipts say one thing, the venue says another, and nothing notices.

Two distinct problems live here, and only one of them blocks:

**Working orders.** If the venue is still holding an unfilled order, its
positions do not include that trade yet. The fund re-assesses, sees the old
book, and issues the same instruction again — a double execution, and the one
failure mode that genuinely costs money. `require_settled` refuses to trade
while any order is working.

**Drift.** Positions or cash that differ from the last receipt. Usually
benign (a manual trade, a corporate action, a fee) but always worth recording:
it means the receipt is no longer a faithful description of the account.
Reported, not blocked — a live broker is authoritative and the pipeline uses
its book.

Brokers expose capabilities by duck typing, not by widening the `Broker`
protocol: `open_orders()` is used when present, ignored otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping

from pydantic import BaseModel, Field

# Cash moves on its own: fees, interest, dividends, rounding. A drift report
# is about "this is not the book we recorded", not "this is off by a cent".
DEFAULT_CASH_TOLERANCE = 1.0


@dataclass(frozen=True)
class LedgerReference:
    """The book a receipt claims the broker holds."""

    positions: Mapping[str, int]
    cash: float
    source: str = "receipt"

    @classmethod
    def from_cycle_record(cls, record: Any) -> "LedgerReference":
        """Build a reference from a saved receipt's ending book."""
        return cls(
            positions=dict(record.positions),
            cash=float(record.cash),
            source=f"{record.fund} {record.execution_as_of or record.as_of}",
        )


class PositionDrift(BaseModel):
    """One ticker whose share count disagrees."""

    ticker: str
    expected: int
    actual: int

    @property
    def delta(self) -> int:
        return self.actual - self.expected


class ReconciliationReport(BaseModel):
    """What the broker holds versus what the ledger recorded."""

    reference: str | None = None
    matched: dict[str, int] = Field(default_factory=dict)
    changed: list[PositionDrift] = Field(default_factory=list)
    only_at_broker: dict[str, int] = Field(default_factory=dict)
    only_in_ledger: dict[str, int] = Field(default_factory=dict)
    cash_expected: float | None = None
    cash_actual: float = 0.0
    cash_delta: float | None = None
    working_orders: int = 0
    notes: list[str] = Field(default_factory=list)

    @property
    def has_position_drift(self) -> bool:
        return bool(self.changed or self.only_at_broker or self.only_in_ledger)

    @property
    def clean(self) -> bool:
        """Nothing to reconcile: same book, same cash, no orders in flight."""
        return not self.has_position_drift and self.working_orders == 0 and (
            self.cash_delta is None or self.cash_delta == 0.0
        )

    @property
    def summary(self) -> str:
        if self.reference is None:
            return "no reference receipt; broker is the source of truth"
        parts: list[str] = []
        if self.working_orders:
            parts.append(f"{self.working_orders} working order(s)")
        if self.changed or self.only_at_broker or self.only_in_ledger:
            parts.append(
                f"positions differ from {self.reference} "
                f"({len(self.changed)} changed, {len(self.only_at_broker)} only at broker, "
                f"{len(self.only_in_ledger)} only in ledger)"
            )
        if self.cash_delta is not None and self.cash_delta != 0.0:
            parts.append(f"cash {self.cash_delta:+,.2f} vs {self.reference}")
        return "; ".join(parts) if parts else f"matches {self.reference}"


def reconcile(
    broker: Any,
    reference: LedgerReference | None = None,
    *,
    cash_tolerance: float = DEFAULT_CASH_TOLERANCE,
) -> ReconciliationReport:
    """Compare *broker* with *reference* and describe every difference."""
    actual = {t: p.shares for t, p in broker.positions().items() if p.shares != 0}
    cash_actual = float(broker.cash())
    if not isfinite(cash_actual):
        raise ValueError(f"broker cash is not finite: {cash_actual!r}")

    working = _working_orders(broker)
    report = ReconciliationReport(
        reference=reference.source if reference is not None else None,
        cash_actual=cash_actual,
        working_orders=working,
    )

    if reference is not None:
        expected = {t: s for t, s in reference.positions.items() if s != 0}
        for ticker in sorted(set(expected) | set(actual)):
            want, have = expected.get(ticker, 0), actual.get(ticker, 0)
            if want == have:
                if want:
                    report.matched[ticker] = want
            elif ticker not in expected:
                report.only_at_broker[ticker] = have
            elif ticker not in actual:
                report.only_in_ledger[ticker] = want
            else:
                report.changed.append(PositionDrift(ticker=ticker, expected=want, actual=have))

        delta = cash_actual - float(reference.cash)
        report.cash_delta = 0.0 if abs(delta) <= cash_tolerance else delta

    if working:
        report.notes.append(
            "the venue is still holding orders: its positions do not include them yet, "
            "so sizing from this book would re-issue the same trade"
        )
    if report.has_position_drift:
        report.notes.append(
            "broker positions disagree with the receipt; the broker is authoritative "
            "for sizing, the receipt no longer describes the account"
        )
    if report.cash_delta:
        report.notes.append("cash moved since the receipt (fees, interest, or a fill)")
    return report


def _working_orders(broker: Any) -> int:
    """Count orders in flight, when the venue can tell us.

    Duck-typed on purpose: `open_orders()` is an optional capability, so
    SimBroker (which never holds a ticket) simply has nothing to report.
    """
    probe = getattr(broker, "open_orders", None)
    if not callable(probe):
        return 0
    try:
        return len(probe())
    except Exception:
        # An unreadable order book is not "no orders". Assume the worst: the
        # caller that cares passes require_settled, and this makes it refuse.
        return 1


def require_settled(report: ReconciliationReport) -> None:
    """Refuse to trade while the venue holds unfilled orders.

    Raises ValueError; the caller decides whether to surface it as a hard stop
    or a warning. Position drift does NOT raise — that is a reporting concern.
    """
    if report.working_orders:
        raise ValueError(
            f"broker has {report.working_orders} working order(s){_ref(report)}; "
            "refusing to size a new cycle against a book that does not include them "
            "(the same trade would be re-issued). Reconcile or cancel first."
        )


def _ref(report: ReconciliationReport) -> str:
    return f" against {report.reference}" if report.reference else ""
