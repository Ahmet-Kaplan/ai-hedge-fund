"""Desk snapshot — everything a monitoring screen shows, in one read.

The TUI (`hedge_fund/tui/desk.py`) and the read-only web dashboard both want
the same thing: what the venue holds, what it is worth, what is still working,
and how it compares with the ledger. Assembling that in one place keeps the two
surfaces from drifting apart, and keeps the logic testable without either.

Read-only by construction — nothing here can submit an order. Brokers opt into
the richer views by exposing `account()` / `position_details()`; venues without
them still produce a snapshot, just a plainer one.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from hedge_fund.brokers.account import AccountSnapshot, PositionDetail, account_snapshot
from hedge_fund.reconciliation import ReconciliationReport, reconcile
from hedge_fund.venue import OpenVenue


class DeskSnapshot(BaseModel):
    """The state of one venue, as an operator would want to read it."""

    venue: str
    live: bool
    trading_enabled: bool
    note: str = ""
    reconciliation_reference: str | None = None

    cash: float
    buying_power: float | None = None
    equity: float | None = None
    shorting_enabled: bool | None = None
    daytrade_count: int | None = None
    pattern_day_trader: bool | None = None
    blocked: bool = False

    positions: list[PositionDetail] = Field(default_factory=list)
    gross_exposure: float | None = None
    net_exposure: float | None = None
    unrealized_pnl: float | None = None

    working_orders: int = 0
    reconciliation: ReconciliationReport | None = None

    @property
    def badge(self) -> str:
        """The one label an operator must not misread."""
        if self.live:
            return "LIVE"
        return "PAPER"

    @property
    def has_valuation(self) -> bool:
        """Whether the venue supplied prices, or only share counts."""
        return any(p.current_price is not None for p in self.positions)

    @property
    def warnings(self) -> list[str]:
        """Everything an operator should look at before trusting the screen."""
        out: list[str] = []
        if self.blocked:
            out.append("account is blocked from trading")
        if self.working_orders:
            out.append(f"{self.working_orders} order(s) still working — positions do not include them")
        if self.reconciliation is not None and not self.reconciliation.clean:
            out.append(self.reconciliation.summary)
        if not self.trading_enabled:
            out.append(f"read-only: {self.note}" if self.note else "read-only")
        return out


def desk_snapshot(opened: OpenVenue, *, cash_tolerance: float = 1.0) -> DeskSnapshot:
    """Assemble the desk view for an already-opened venue."""
    broker = opened.broker
    account: AccountSnapshot | None = account_snapshot(broker)
    details = _position_details(broker)
    report = reconcile(broker, opened.reference, cash_tolerance=cash_tolerance)

    market_values = [p.market_value for p in details if p.market_value is not None]
    pnls = [p.unrealized_pnl for p in details if p.unrealized_pnl is not None]
    return DeskSnapshot(
        venue=opened.label,
        live=opened.live,
        trading_enabled=opened.trading_enabled,
        note=opened.note,
        reconciliation_reference=(
            opened.reference.source if opened.reference is not None else None
        ),
        cash=broker.cash(),
        buying_power=account.buying_power if account else None,
        equity=account.equity if account else None,
        shorting_enabled=account.shorting_enabled if account else None,
        daytrade_count=account.daytrade_count if account else None,
        pattern_day_trader=account.pattern_day_trader if account else None,
        blocked=bool(account and account.trading_blocked),
        positions=details,
        gross_exposure=sum(abs(v) for v in market_values) if market_values else None,
        net_exposure=sum(market_values) if market_values else None,
        unrealized_pnl=sum(pnls) if pnls else None,
        working_orders=report.working_orders,
        reconciliation=report,
    )


def _position_details(broker: Any) -> list[PositionDetail]:
    """Rich holdings when the venue offers them, plain share counts otherwise."""
    probe = getattr(broker, "position_details", None)
    if callable(probe):
        # A venue that advertises the capability but hands back nothing is
        # "no rich view", not a crash on a monitoring screen.
        details = probe()
        if details is not None:
            return list(details)
    return [
        PositionDetail(ticker=ticker, shares=position.shares)
        for ticker, position in sorted(broker.positions().items())
    ]
