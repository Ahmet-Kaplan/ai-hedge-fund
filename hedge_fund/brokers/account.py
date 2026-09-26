"""Account-level pre-trade checks — what the venue knows and the fund does not.

`hedge_fund/risk/limits.py` clamps the *portfolio*: position sizes, gross
exposure, the mandate's own rules. It is deliberately broker-agnostic and
knows nothing about the account it is trading. This module is the other half:
the facts only the venue can supply, checked immediately before orders go out.

Four things can go wrong that the portfolio layer cannot see:

**Buying power.** `cash()` is not spendable cash. Unsettled sales, margin
treatment and open orders all sit between the two, and Alpaca's `buying_power`
is the number that actually governs. Sizing against cash can produce an order
the venue rejects.

**Short permission.** A sell that takes a position past zero is a short, and
some accounts cannot. The portfolio layer sees a negative target weight and has
no idea whether that is allowed.

**Pattern day trading.** A margin account under $25k that makes a fourth
day trade in five business days gets blocked. It is the venue's rule and only
the venue's count is authoritative.

**The calendar.** Holidays and half-days are not derivable from price bars
with any confidence. The venue publishes them.

Everything here is advisory-to-the-pipeline but blocking on violation: a
`PreflightReport` with a non-empty `violations` means the cycle must not send
orders, and `execute_decision` refuses. Brokers opt in by exposing `account()`
— SimBroker and PaperBroker have no account, so they simply skip this layer.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Iterable

from pydantic import BaseModel, Field

from hedge_fund.brokers.models import Order, Position

# Alpaca's pattern-day-trader threshold. Below this equity, a fourth day trade
# in five business days is rejected by the venue.
PDT_EQUITY_FLOOR = 25_000.0
PDT_DAY_TRADE_LIMIT = 3

# Fills are not instant and prices move between sizing and submission; a
# small cushion keeps a correctly-sized order from being rejected on drift.
DEFAULT_BUYING_POWER_CUSHION = 0.01


class AccountSnapshot(BaseModel):
    """Venue facts, normalised so the checks do not depend on the SDK."""

    venue: str
    cash: float
    buying_power: float
    equity: float
    shorting_enabled: bool = True
    trading_blocked: bool = False
    daytrade_count: int = 0
    pattern_day_trader: bool = False

    @property
    def settleable(self) -> bool:
        return deep_finite(self.cash) and deep_finite(self.buying_power) and deep_finite(self.equity)


def deep_finite(*values: float) -> bool:
    return all(isfinite(v) for v in values)


class PositionDetail(BaseModel):
    """One holding with the venue's own valuation.

    The `Broker` protocol reports signed share counts only, because sizing does
    not need more. A desk does: an operator wants to see what each position is
    worth and what it has made. Venues that can supply it expose
    `position_details()`; the field stays optional so the plain protocol is
    unaffected.
    """

    ticker: str
    shares: int
    avg_entry_price: float | None = None
    current_price: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    unrealized_pnl_pct: float | None = None

    @property
    def side(self) -> str:
        return "long" if self.shares >= 0 else "short"


class PreflightReport(BaseModel):
    """What the venue would say about this set of orders."""

    venue: str
    buying_power: float
    required_cash: float
    closing_notional: float = 0.0
    opening_short_notional: float = 0.0
    violations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def summary(self) -> str:
        if self.violations:
            return "; ".join(self.violations)
        if self.warnings:
            return "; ".join(self.warnings)
        return (
            f"needs {self.required_cash:,.2f} of {self.buying_power:,.2f} "
            "buying power"
        )


def account_snapshot(broker: Any) -> AccountSnapshot | None:
    """The broker's account, when it has one.

    Duck-typed: `account()` is an optional capability, so sim and paper venues
    — which have no account and nothing to check — return None and the layer is
    skipped entirely.
    """
    probe = getattr(broker, "account", None)
    if not callable(probe):
        return None
    return probe()


def preflight(
    account: AccountSnapshot,
    orders: Iterable[Order],
    positions: dict[str, Position],
    *,
    cushion: float = DEFAULT_BUYING_POWER_CUSHION,
) -> PreflightReport:
    """Check *orders* against *account* before anything is submitted.

    Sells are counted against buys the way the venue sees them: proceeds from a
    closing sale are spendable the same session, but the sale has to be a
    *closing* one — selling past zero opens a short, which consumes buying
    power rather than freeing it.
    """
    orders = list(orders)
    report = PreflightReport(
        venue=account.venue,
        buying_power=account.buying_power,
        required_cash=0.0,
    )

    if not account.settleable:
        report.violations.append(
            f"{account.venue} reported a non-finite account balance"
        )
        return report

    if account.trading_blocked:
        report.violations.append(f"{account.venue} reports the account is blocked from trading")

    buys = 0.0
    closes = 0.0
    opening_shorts = 0.0
    for order in orders:
        notional = order.quantity * order.price
        if order.side == "buy":
            buys += notional
            continue
        held = positions[order.ticker].shares if order.ticker in positions else 0
        closing = min(order.quantity, max(held, 0))
        closes += closing * order.price
        shorted = order.quantity - closing
        if shorted > 0:
            opening_shorts += shorted * order.price

    report.closing_notional = closes
    report.opening_short_notional = opening_shorts
    # Proceeds only help to the extent they close an existing long.
    report.required_cash = max(buys - closes, 0.0)

    budget = account.buying_power * (1.0 - cushion)
    if report.required_cash > budget:
        report.violations.append(
            f"orders need {report.required_cash:,.2f} but only "
            f"{account.buying_power:,.2f} buying power is available"
        )

    if opening_shorts > 0 and not account.shorting_enabled:
        report.violations.append(
            f"{account.venue} does not permit short selling, but these orders open "
            f"{opening_shorts:,.2f} of short exposure"
        )

    if account.pattern_day_trader and account.daytrade_count >= PDT_DAY_TRADE_LIMIT:
        report.violations.append(
            f"account is flagged pattern-day-trader and already has "
            f"{account.daytrade_count} day trades in the window; a fourth would be rejected"
        )
    elif account.equity < PDT_EQUITY_FLOOR and account.daytrade_count >= PDT_DAY_TRADE_LIMIT:
        report.warnings.append(
            f"equity {account.equity:,.2f} is under the {PDT_EQUITY_FLOOR:,.0f} "
            f"pattern-day-trader floor and {account.daytrade_count} day trades are used"
        )

    return report
