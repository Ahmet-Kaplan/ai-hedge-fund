"""Alpaca broker — a paper/live venue behind the :class:`Broker` protocol.

Optional dependency: this module needs ``alpaca-py``, which is deliberately
*not* a core dependency (the core fund runs on SimBroker/PaperBroker):

    pip install alpaca-py

Safety is layered so a half-remembered environment variable cannot move real
money:

``ALPACA_PAPER`` (default ``true``)
    Selects Alpaca's paper endpoint. Reads always work.
``ALPACA_TRADING_ENABLED`` (default ``false``)
    ``place_order`` refuses until this is explicitly truthy, whatever the venue.
``ALPACA_LIVE_TRADING_CONFIRMED`` (default ``false``)
    Submitting against a **live** account additionally requires this, so
    flipping ``ALPACA_PAPER=false`` alone is not enough to trade for real.

Contract note: ``Broker.place_order`` must fill *completely* or raise — no
partial fills, no silent drops, because the fund sizes its book from the
returned Fill. Alpaca accepts orders asynchronously, so this adapter submits a
market order and polls briefly for the execution report. Anything short of a
complete fill raises :class:`AlpacaOrderError` carrying the broker order id,
rather than inventing a Fill the venue never gave.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Callable

from hedge_fund.brokers.account import AccountSnapshot
from hedge_fund.brokers.models import Fill, Order, Position

logger = logging.getLogger(__name__)

# Alpaca's own SDK variables, then ours. ALPACA_API_SECRET is honored because
# it is the name several existing local setups (and Alpaca's own docs) use.
_API_KEY_VARS = ("ALPACA_API_KEY", "APCA_API_KEY_ID")
_SECRET_KEY_VARS = ("ALPACA_SECRET_KEY", "ALPACA_API_SECRET", "APCA_API_SECRET_KEY")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}

DEFAULT_FILL_TIMEOUT_SECONDS = 15.0
DEFAULT_POLL_INTERVAL_SECONDS = 0.25

# How long a market-open answer is reused. Orders arrive in bursts, and the
# answer cannot change mid-burst in any way that matters.
_CLOCK_TTL_SECONDS = 5.0

# Alpaca's error code for a client_order_id that is already in use.
_DUPLICATE_CLIENT_ORDER_ID = "40010001"


def _is_duplicate_client_order_id(exc: Exception) -> bool:
    text = f"{exc}"
    return _DUPLICATE_CLIENT_ORDER_ID in text or "client_order_id must be unique" in text


class AlpacaOrderError(RuntimeError):
    """An order did not reach a complete fill.

    The fund's books would desync if this were reported as a Fill, so it
    raises instead. ``order_id`` is the broker's id for manual reconciliation:
    the order may still be working at the venue.
    """

    def __init__(self, message: str, *, order_id: str | None = None) -> None:
        super().__init__(message)
        self.order_id = order_id


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"expected a boolean-ish value, got {raw!r}")


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


@dataclass(frozen=True)
class AlpacaSettings:
    """Everything the adapter needs, resolved once and inspectable."""

    api_key: str
    secret_key: str
    paper: bool = True
    trading_enabled: bool = False
    live_confirmed: bool = False
    allow_closed_market: bool = False
    fill_timeout_seconds: float = DEFAULT_FILL_TIMEOUT_SECONDS
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS

    @property
    def venue(self) -> str:
        return "alpaca-paper" if self.paper else "alpaca-live"

    @classmethod
    def from_env(cls, **overrides: Any) -> "AlpacaSettings":
        """Build settings from the environment; *overrides* win.

        Raises ValueError naming the variable to set when a key is missing.
        """
        api_key = _first_env(_API_KEY_VARS)
        secret_key = _first_env(_SECRET_KEY_VARS)
        if not api_key:
            raise ValueError(f"{_API_KEY_VARS[0]} is not set (accepted: {', '.join(_API_KEY_VARS)})")
        if not secret_key:
            raise ValueError(f"{_SECRET_KEY_VARS[0]} is not set (accepted: {', '.join(_SECRET_KEY_VARS)})")

        settings = cls(
            api_key=api_key,
            secret_key=secret_key,
            paper=_parse_bool(os.environ.get("ALPACA_PAPER"), default=True),
            trading_enabled=_parse_bool(os.environ.get("ALPACA_TRADING_ENABLED"), default=False),
            live_confirmed=_parse_bool(os.environ.get("ALPACA_LIVE_TRADING_CONFIRMED"), default=False),
            allow_closed_market=_parse_bool(
                os.environ.get("ALPACA_ALLOW_CLOSED_MARKET"), default=False
            ),
        )
        return replace(settings, **overrides) if overrides else settings

    def refuse_reason(self) -> str | None:
        """Why orders may not be submitted, or None when they may."""
        if not self.trading_enabled:
            return (
                "order submission is disabled: set ALPACA_TRADING_ENABLED=1 to "
                f"trade on the {self.venue} venue"
            )
        if not self.paper and not self.live_confirmed:
            return (
                "refusing to trade a LIVE account: ALPACA_PAPER is false, and "
                "ALPACA_LIVE_TRADING_CONFIRMED=1 is also required"
            )
        return None

    def check_can_trade(self) -> None:
        reason = self.refuse_reason()
        if reason is not None:
            raise AlpacaOrderError(reason)


def _int_shares(ticker: str, qty: Any) -> int:
    """Alpaca quantity as an integer share count, or fail loud.

    Broker.Position.shares is an int, and silently truncating would make the
    fund's books disagree with the broker's — a fractional or malformed
    position is a configuration error, not something to round away.
    """
    try:
        shares = float(qty)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{ticker}: unreadable position quantity {qty!r}") from exc
    if not isfinite(shares):
        raise ValueError(f"{ticker}: non-finite position quantity {qty!r}")
    if shares != int(shares):
        raise ValueError(
            f"{ticker}: fractional position of {shares} shares cannot be represented "
            "as an integer share count; close it or run without the fractional-size feature"
        )
    return int(shares)


class AlpacaBroker:
    """A :class:`~hedge_fund.brokers.protocol.Broker` backed by Alpaca.

    Reads (``positions``/``cash``/``clock``) always work. Writes go through
    the ``ALPACA_TRADING_ENABLED`` / ``ALPACA_LIVE_TRADING_CONFIRMED`` gates.
    """

    def __init__(
        self,
        settings: AlpacaSettings | None = None,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings if settings is not None else AlpacaSettings.from_env()
        self._client = client if client is not None else _make_trading_client(self._settings)
        self._sleep = sleep
        self._monotonic = monotonic
        self._clock_cache: bool | None = None
        self._clock_cached_at = float("-inf")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def settings(self) -> AlpacaSettings:
        return self._settings

    @property
    def venue(self) -> str:
        return self._settings.venue

    @property
    def client(self) -> Any:
        return self._client

    # ------------------------------------------------------------------
    # Broker protocol
    # ------------------------------------------------------------------

    def positions(self) -> dict[str, Position]:
        """Signed share counts keyed by ticker, with no zero-share rows."""
        held: dict[str, Position] = {}
        for raw in self._client.get_all_positions():
            ticker = str(raw.symbol).upper()
            shares = _int_shares(ticker, raw.qty)
            if shares:
                held[ticker] = Position(ticker=ticker, shares=shares)
        return held

    def cash(self) -> float:
        return float(self._client.get_account().cash)

    def place_order(self, order: Order) -> Fill:
        """Submit a market order and require a complete fill.

        Raises :class:`AlpacaOrderError` if the venue does not report a full
        fill within the timeout, or if trading is gated off.
        """
        self._settings.check_can_trade()
        self._refuse_outside_market_hours(order)

        account = self._client.get_account()
        if bool(getattr(account, "account_blocked", False)) or bool(getattr(account, "trading_blocked", False)):
            raise AlpacaOrderError("account is blocked from trading", order_id=None)

        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        # Pass the caller's deterministic id so the venue dedupes a retry:
        # resubmitting the same id returns the original order instead of
        # creating a second one.
        request = MarketOrderRequest(
            symbol=order.ticker.upper(),
            qty=order.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=order.client_order_id,
        )
        try:
            submitted = self._client.submit_order(request)
        except Exception as exc:
            adopted = self._adopt_existing_on_duplicate(order, exc)
            if adopted is None:
                raise AlpacaOrderError(
                    f"{self.venue}: {order.side} {order.quantity} {order.ticker} rejected: {exc}"
                ) from exc
            submitted = adopted

        order_id = str(getattr(submitted, "id", "") or "") or None
        filled = self._await_fill(order_id, order, submitted)
        return filled

    # ------------------------------------------------------------------
    # Venue helpers
    # ------------------------------------------------------------------

    def clock(self) -> dict[str, Any]:
        """Market clock: open flag and the next open/close."""
        raw = self._client.get_clock()
        return {
            "is_open": bool(raw.is_open),
            "next_open": raw.next_open.isoformat() if raw.next_open else None,
            "next_close": raw.next_close.isoformat() if raw.next_close else None,
        }

    def is_market_open(self) -> bool:
        return bool(self._client.get_clock().is_open)

    def account(self) -> AccountSnapshot:
        """Venue facts the pre-trade checks need, normalised away from the SDK.

        `shorting_enabled` defaults to False when the field is missing: an
        account whose short permission cannot be read must not be assumed to
        have it.
        """
        raw = self._client.get_account()
        return AccountSnapshot(
            venue=self.venue,
            cash=float(raw.cash),
            buying_power=float(raw.buying_power),
            equity=float(raw.equity),
            shorting_enabled=bool(getattr(raw, "shorting_enabled", False)),
            trading_blocked=bool(
                getattr(raw, "account_blocked", False)
                or getattr(raw, "trading_blocked", False)
            ),
            daytrade_count=int(getattr(raw, "daytrade_count", 0) or 0),
            pattern_day_trader=bool(getattr(raw, "pattern_day_trader", False)),
        )

    def calendar(self, start: str, end: str) -> list[dict[str, Any]]:
        """Exchange sessions between two dates, holidays and half-days included.

        Bars cannot tell you a half-day from a holiday; the venue can.
        """
        from alpaca.trading.requests import GetCalendarRequest

        days = self._client.get_calendar(GetCalendarRequest(start=start, end=end))
        return [
            {
                "date": _iso_date(day.date),
                "open": _hhmm(getattr(day, "open", None)),
                "close": _hhmm(getattr(day, "close", None)),
            }
            for day in days
        ]

    def is_trading_day(self, day: str) -> bool:
        return any(session["date"] == day for session in self.calendar(day, day))

    def open_orders(self) -> list[dict[str, Any]]:
        """Working orders, for reconciliation."""
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))
        return [
            {
                "order_id": str(o.id),
                "ticker": str(o.symbol).upper(),
                "side": str(o.side),
                "quantity": float(o.qty),
                "filled_quantity": float(o.filled_qty or 0),
                "status": str(o.status),
            }
            for o in orders
        ]

    def cancel_all_orders(self) -> None:
        """Cancel every working order. Gated like any other write."""
        self._settings.check_can_trade()
        self._client.cancel_orders()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _refuse_outside_market_hours(self, order: Order) -> None:
        """Do not queue orders for a session that has not opened.

        A market order submitted while the venue is closed is ACCEPTED and
        parked until the next open (verified against the paper API). The
        pipeline assumes a synchronous fill: it reads positions back straight
        away, so a parked order would leave the book unchanged and the next
        cycle would re-issue the same trade under a fresh session id — a
        double execution. Refusing is the only honest option.
        """
        if self._settings.allow_closed_market:
            return
        if not self._market_is_open_cached():
            clock = self.clock()
            raise AlpacaOrderError(
                f"{self.venue}: refusing to submit {order.side} {order.quantity} "
                f"{order.ticker} while the market is closed (next open {clock['next_open']}); "
                "a parked order would be re-issued by the next cycle. "
                "Set ALPACA_ALLOW_CLOSED_MARKET=1 to queue it deliberately."
            )

    def _market_is_open_cached(self) -> bool:
        """is_market_open with a short TTL: one clock call per burst of orders."""
        now = self._monotonic()
        if self._clock_cache is None or (now - self._clock_cached_at) > _CLOCK_TTL_SECONDS:
            self._clock_cache = self.is_market_open()
            self._clock_cached_at = now
        return self._clock_cache

    def _adopt_existing_on_duplicate(self, order: Order, exc: Exception) -> Any | None:
        """Resolve a duplicate client_order_id to the order it collided with.

        Alpaca does not hand back the original order on a duplicate submit —
        it rejects the second one with 40010001 "client_order_id must be
        unique". That is already safe (no second order is created), but a
        retry should *succeed* against the original instruction, so look it up
        and carry on polling it.
        """
        cid = order.client_order_id
        if not cid or not _is_duplicate_client_order_id(exc):
            return None
        try:
            existing = self._client.get_order_by_client_id(cid)
        except Exception:  # cannot resolve -> let the caller see the rejection
            return None
        logger.info(
            "%s: adopted existing order %s for client_order_id %s",
            self.venue, getattr(existing, "id", "?"), cid,
        )
        return existing

    def _await_fill(self, order_id: str | None, order: Order, submitted: Any) -> Fill:
        """Poll until the order is completely filled, or raise."""
        deadline = self._monotonic() + self._settings.fill_timeout_seconds
        latest = submitted
        while True:
            status = str(getattr(latest, "status", "")).lower()
            filled_qty = float(getattr(latest, "filled_qty", 0) or 0)
            avg_price = getattr(latest, "filled_avg_price", None)

            if status == "filled":
                if int(filled_qty) != order.quantity:
                    raise AlpacaOrderError(
                        f"{self.venue}: {order.ticker} reported filled but for "
                        f"{filled_qty:g} of {order.quantity} shares",
                        order_id=order_id,
                    )
                try:
                    price = float(avg_price)
                except (TypeError, ValueError):
                    price = float("nan")
                if not isfinite(price) or price <= 0:
                    raise AlpacaOrderError(
                        f"{self.venue}: {order.ticker} filled without a usable average price "
                        f"({avg_price!r})",
                        order_id=order_id,
                    )
                return Fill(
                    ticker=order.ticker.upper(),
                    side=order.side,
                    quantity=int(filled_qty),
                    price=price,
                )

            if status in {"canceled", "cancelled", "expired", "rejected", "replaced"}:
                raise AlpacaOrderError(
                    f"{self.venue}: {order.side} {order.quantity} {order.ticker} ended {status} "
                    f"({filled_qty:g} filled)",
                    order_id=order_id,
                )

            if self._monotonic() >= deadline:
                raise AlpacaOrderError(
                    f"{self.venue}: {order.side} {order.quantity} {order.ticker} was not completely "
                    f"filled within {self._settings.fill_timeout_seconds:g}s "
                    f"(status {status or 'unknown'}, {filled_qty:g}/{order.quantity} filled); "
                    "check the venue before assuming the book is flat",
                    order_id=order_id,
                )

            self._sleep(self._settings.poll_interval_seconds)
            if order_id is None:
                raise AlpacaOrderError(
                    f"{self.venue}: submission returned no order id; cannot confirm a fill for "
                    f"{order.side} {order.quantity} {order.ticker}",
                )
            latest = self._client.get_order_by_id(order_id)


def _iso_date(value: Any) -> str:
    """A session date as YYYY-MM-DD, whatever the SDK hands back."""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def _hhmm(value: Any) -> str:
    """A session time as HH:MM. Alpaca returns datetimes, not times, so the
    first five characters would be the year."""
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M")
    return str(value)[:5]


def _make_trading_client(settings: AlpacaSettings) -> Any:
    """Build the SDK client, naming the install command when it is missing."""
    try:
        from alpaca.trading.client import TradingClient
    except ImportError as exc:
        raise ImportError(
            "AlpacaBroker needs the optional alpaca-py package: pip install alpaca-py"
        ) from exc
    return TradingClient(
        api_key=settings.api_key,
        secret_key=settings.secret_key,
        paper=settings.paper,
    )


def broker_from_env(**overrides: Any) -> AlpacaBroker:
    """Convenience constructor: settings from the environment, then the client."""
    return AlpacaBroker(AlpacaSettings.from_env(**overrides))
