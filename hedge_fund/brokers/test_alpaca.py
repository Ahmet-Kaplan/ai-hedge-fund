"""AlpacaBroker tests — a fake TradingClient, no credentials, no network.

The SDK is an optional dependency, so only the tests that exercise the submit
path import it (`pytest.importorskip`). The safety gates are deliberately
tested *without* it: refusing to trade must not depend on the SDK being
installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import pytest

from hedge_fund.brokers.alpaca import (
    AlpacaBroker,
    AlpacaOrderError,
    AlpacaSettings,
    _int_shares,
)
from hedge_fund.brokers.models import Order, Position
from hedge_fund.brokers.protocol import Broker


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass
class _RawPosition:
    symbol: str
    qty: str


@dataclass
class _RawAccount:
    cash: str = "25000.50"
    account_blocked: bool = False
    trading_blocked: bool = False


@dataclass
class _RawClock:
    is_open: bool = True
    next_open: object = None
    next_close: object = None


@dataclass
class _RawOrder:
    id: str
    status: str
    filled_qty: str = "0"
    filled_avg_price: str | None = None


class FakeClient:
    """Stands in for alpaca-py's TradingClient."""

    def __init__(self, *, positions=None, account=None, order_script=None,
                 clock_open=True, by_client_id=None):
        self._positions = positions if positions is not None else []
        self._account = account or _RawAccount()
        self._script = list(order_script or [])
        self._clock_open = clock_open
        self._by_client_id = by_client_id
        self.submitted = []
        self.polled = []

    def get_all_positions(self):
        return list(self._positions)

    def get_account(self):
        return self._account

    def submit_order(self, request):
        self.submitted.append(request)
        if not self._script:
            raise AssertionError("submit_order called with no scripted result")
        return self._script.pop(0)

    def get_order_by_id(self, order_id):
        self.polled.append(order_id)
        if not self._script:
            raise AssertionError(f"no scripted poll result for {order_id}")
        return self._script.pop(0)

    def get_clock(self):
        return _RawClock(is_open=self._clock_open)

    def get_order_by_client_id(self, client_order_id):
        if self._by_client_id is None:
            raise AssertionError(f"no order registered for {client_order_id}")
        return self._by_client_id


def _settings(**over):
    base = dict(api_key="k", secret_key="s", paper=True, trading_enabled=True)
    base.update(over)
    return AlpacaSettings(**base)


def _broker(client, **over):
    return AlpacaBroker(_settings(**over), client=client, sleep=lambda _s: None)


# ---------------------------------------------------------------------------
# Settings: env resolution and the safety gates
# ---------------------------------------------------------------------------

def test_settings_read_env_and_accept_the_api_secret_alias(monkeypatch):
    """ALPACA_API_SECRET is what several existing local setups actually use."""
    monkeypatch.setenv("ALPACA_API_KEY", "key-1")
    monkeypatch.setenv("ALPACA_API_SECRET", "secret-1")
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    settings = AlpacaSettings.from_env()

    assert settings.api_key == "key-1"
    assert settings.secret_key == "secret-1"
    assert settings.paper is True          # default is paper
    assert settings.trading_enabled is False  # default is read-only
    assert settings.venue == "alpaca-paper"


def test_settings_prefer_our_secret_name_over_the_alias(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "canonical")
    monkeypatch.setenv("ALPACA_API_SECRET", "alias")
    assert AlpacaSettings.from_env().secret_key == "canonical"


def test_settings_name_the_missing_variable(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    with pytest.raises(ValueError, match="ALPACA_API_KEY"):
        AlpacaSettings.from_env()


def test_live_requires_an_explicit_confirmation():
    """Flipping ALPACA_PAPER=false alone must never be enough to trade live."""
    live = _settings(paper=False, trading_enabled=True, live_confirmed=False)
    assert live.refuse_reason() is not None
    with pytest.raises(AlpacaOrderError, match="LIVE"):
        live.check_can_trade()

    confirmed = _settings(paper=False, trading_enabled=True, live_confirmed=True)
    assert confirmed.refuse_reason() is None
    assert confirmed.venue == "alpaca-live"


def test_trading_disabled_refuses_before_anything_else():
    off = _settings(trading_enabled=False)
    with pytest.raises(AlpacaOrderError, match="ALPACA_TRADING_ENABLED"):
        off.check_can_trade()
    assert _settings(trading_enabled=True, paper=True).refuse_reason() is None


def test_place_order_refuses_without_submitting_when_gated_off():
    """No SDK import and no network: the gate fires first."""
    client = FakeClient()
    broker = _broker(client, trading_enabled=False)

    with pytest.raises(AlpacaOrderError, match="ALPACA_TRADING_ENABLED"):
        broker.place_order(Order(ticker="AAPL", side="buy", quantity=1, price=100.0))

    assert client.submitted == []


def test_blocked_account_is_refused_before_submitting():
    client = FakeClient(account=_RawAccount(account_blocked=True))
    broker = _broker(client)

    with pytest.raises(AlpacaOrderError, match="blocked"):
        broker.place_order(Order(ticker="AAPL", side="buy", quantity=1, price=100.0))

    assert client.submitted == []


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def test_positions_are_signed_and_drop_flat_rows():
    client = FakeClient(positions=[
        _RawPosition("AAPL", "10"),
        _RawPosition("MSFT", "-4"),
        _RawPosition("NVDA", "0"),
    ])

    held = _broker(client).positions()

    assert held == {"AAPL": Position(ticker="AAPL", shares=10),
                    "MSFT": Position(ticker="MSFT", shares=-4)}


def test_fractional_position_fails_loud_instead_of_truncating():
    """Position.shares is an int; rounding would desync the fund's books."""
    client = FakeClient(positions=[_RawPosition("AAPL", "1.5")])
    with pytest.raises(ValueError, match="fractional"):
        _broker(client).positions()


def test_int_shares_rejects_garbage():
    with pytest.raises(ValueError, match="unreadable"):
        _int_shares("AAPL", "not-a-number")


def test_cash_is_a_float():
    assert _broker(FakeClient()).cash() == pytest.approx(25000.50)


def test_satisfies_the_broker_protocol():
    broker = _broker(FakeClient())
    assert isinstance(broker, Broker)


# ---------------------------------------------------------------------------
# Submit path (needs the optional SDK for its enums)
# ---------------------------------------------------------------------------

def test_complete_fill_returns_a_fill_with_the_venue_price():
    pytest.importorskip("alpaca")
    client = FakeClient(order_script=[
        _RawOrder("ord-1", "accepted", "0", None),
        _RawOrder("ord-1", "filled", "3", "101.25"),
    ])

    fill = _broker(client).place_order(
        Order(ticker="aapl", side="buy", quantity=3, price=100.0)
    )

    assert fill.ticker == "AAPL"
    assert fill.side == "buy"
    assert fill.quantity == 3
    assert fill.price == pytest.approx(101.25)  # the venue's price, not the reference
    assert client.polled == ["ord-1"]


def test_incomplete_fill_raises_with_the_order_id():
    """Broker.place_order must fill completely or raise — never partial."""
    pytest.importorskip("alpaca")
    broker = _broker(FakeClient(order_script=[
        _RawOrder("ord-2", "accepted", "0", None),
    ]), fill_timeout_seconds=0.0)  # deadline already passed on the first poll

    with pytest.raises(AlpacaOrderError) as exc:
        broker.place_order(Order(ticker="AAPL", side="buy", quantity=5, price=100.0))

    assert exc.value.order_id == "ord-2"
    assert "not completely filled" in str(exc.value)


def test_rejected_order_raises_rather_than_reporting_a_fill():
    pytest.importorskip("alpaca")
    broker = _broker(FakeClient(order_script=[
        _RawOrder("ord-3", "rejected", "0", None),
    ]))

    with pytest.raises(AlpacaOrderError, match="rejected"):
        broker.place_order(Order(ticker="AAPL", side="sell", quantity=1, price=100.0))


def test_filled_but_short_of_requested_quantity_raises():
    """A partial fill reported as 'filled' must not pass as a complete one."""
    pytest.importorskip("alpaca")
    broker = _broker(FakeClient(order_script=[
        _RawOrder("ord-4", "accepted", "0", None),
        _RawOrder("ord-4", "filled", "4", "100.0"),
    ]))

    with pytest.raises(AlpacaOrderError, match="4 of 5"):
        broker.place_order(Order(ticker="AAPL", side="buy", quantity=5, price=100.0))


def test_submit_rejection_is_wrapped_with_context():
    pytest.importorskip("alpaca")

    class Rejecting(FakeClient):
        def submit_order(self, request):
            raise RuntimeError("market is closed")

    broker = _broker(Rejecting())
    with pytest.raises(AlpacaOrderError, match="market is closed"):
        broker.place_order(Order(ticker="AAPL", side="buy", quantity=1, price=100.0))


# ---------------------------------------------------------------------------
# Live paper account — opt-in, never run by default
# ---------------------------------------------------------------------------

def _live_settings():
    """Settings from the environment, or skip. Read-only unless opted in."""
    if not (os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")):
        pytest.skip("live: ALPACA_API_KEY is not set")
    if not (os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_API_SECRET")):
        pytest.skip("live: ALPACA_SECRET_KEY / ALPACA_API_SECRET is not set")
    return AlpacaSettings.from_env()


@pytest.mark.live
def test_paper_account_reads_are_live():
    """Cash, positions and the clock come back from the real paper venue."""
    broker = AlpacaBroker(_live_settings())

    assert isinstance(broker, Broker)
    assert broker.cash() >= 0.0
    held = broker.positions()
    assert all(p.shares != 0 for p in held.values())

    clock = broker.clock()
    assert isinstance(clock["is_open"], bool)
    # The venue is authoritative about what it holds — and the fund's books are
    # only as good as this round trip.
    assert isinstance(broker.open_orders(), list)


@pytest.mark.live
def test_live_and_paper_gates_hold_against_real_settings():
    """A live account must refuse to trade without the second confirmation."""
    live = replace(_live_settings(), paper=False, trading_enabled=True, live_confirmed=False)
    with pytest.raises(AlpacaOrderError, match="LIVE"):
        live.check_can_trade()


@pytest.mark.live
def test_live_order_round_trip_is_opt_in():
    """Place and flatten one share only when ALPACA_TEST_ORDER=1 (market open).

    Off by default: an unconditional order test would mutate a real account
    every time the suite ran.
    """
    if os.environ.get("ALPACA_TEST_ORDER") != "1":
        pytest.skip("live order test is opt-in: set ALPACA_TEST_ORDER=1")
    settings = replace(_live_settings(), trading_enabled=True)
    if not settings.paper:
        pytest.skip("live order test refuses a non-paper account")
    broker = AlpacaBroker(settings)
    if not broker.is_market_open():
        pytest.skip("live order test needs an open market")

    ticker = os.environ.get("ALPACA_TEST_TICKER", "AAPL")
    fill = broker.place_order(Order(ticker=ticker, side="buy", quantity=1, price=0.0))
    assert fill.quantity == 1 and fill.price > 0
    try:
        broker.place_order(Order(ticker=ticker, side="sell", quantity=1, price=0.0))
    finally:
        broker.cancel_all_orders()


def test_client_order_id_is_forwarded_to_the_venue():
    """The venue, not this adapter, is what makes a retry idempotent."""
    pytest.importorskip("alpaca")
    client = FakeClient(order_script=[
        _RawOrder("ord-9", "accepted", "0", None),
        _RawOrder("ord-9", "filled", "2", "50.0"),
    ])

    _broker(client).place_order(
        Order(ticker="AAPL", side="buy", quantity=2, price=50.0,
              client_order_id="desk-2025-01-10-000-b-deadbeef")
    )

    assert client.submitted[0].client_order_id == "desk-2025-01-10-000-b-deadbeef"


# ---------------------------------------------------------------------------
# Closed market: a parked order would be re-issued by the next cycle
# ---------------------------------------------------------------------------

def test_closed_market_refuses_instead_of_parking_an_order():
    """Verified against the paper API: a closed-market order is ACCEPTED and
    parked, while the pipeline reads positions back immediately — so the next
    cycle would re-issue the same trade under a new session id."""
    pytest.importorskip("alpaca")
    client = FakeClient(clock_open=False)

    with pytest.raises(AlpacaOrderError, match="market is closed"):
        _broker(client).place_order(Order(ticker="AAPL", side="buy", quantity=1, price=100.0))

    assert client.submitted == []


def test_closed_market_can_be_overridden_deliberately():
    pytest.importorskip("alpaca")
    client = FakeClient(clock_open=False, order_script=[
        _RawOrder("ord-c", "accepted", "0", None),
        _RawOrder("ord-c", "filled", "1", "100.0"),
    ])

    fill = _broker(client, allow_closed_market=True).place_order(
        Order(ticker="AAPL", side="buy", quantity=1, price=100.0)
    )

    assert fill.quantity == 1
    assert len(client.submitted) == 1


def test_duplicate_client_order_id_adopts_the_original_order():
    """Alpaca rejects a duplicate with 40010001 rather than returning the
    original; a retry must resolve to that order instead of failing."""
    pytest.importorskip("alpaca")

    class DuplicateRejecting(FakeClient):
        def submit_order(self, request):
            self.submitted.append(request)
            raise RuntimeError(
                '{"code":40010001,"message":"client_order_id must be unique"}'
            )

    adopted = _RawOrder("ord-orig", "accepted", "0", None)
    client = DuplicateRejecting(by_client_id=adopted, order_script=[
        _RawOrder("ord-orig", "filled", "2", "51.0"),
    ])

    fill = _broker(client).place_order(
        Order(ticker="AAPL", side="buy", quantity=2, price=50.0, client_order_id="cid-dup")
    )

    assert fill.ticker == "AAPL"
    assert fill.quantity == 2
    assert fill.price == pytest.approx(51.0)


def test_an_unrelated_submit_failure_is_still_an_error():
    """Only a duplicate-id rejection is recoverable."""
    pytest.importorskip("alpaca")

    class Other(FakeClient):
        def submit_order(self, request):
            raise RuntimeError('{"code":40310000,"message":"insufficient buying power"}')

    with pytest.raises(AlpacaOrderError, match="insufficient buying power"):
        _broker(Other()).place_order(Order(ticker="AAPL", side="buy", quantity=1, price=1.0))
