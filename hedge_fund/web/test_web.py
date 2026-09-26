"""Read-only dashboard tests.

The load-bearing claims are structural, so they are tested structurally: that
every registered route is a GET, and that the broker the view is built on
refuses to place an order. Auth, host gating and the desk payload are covered
alongside.

FastAPI is an optional dependency, so the whole module skips without it.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hedge_fund.brokers.account import AccountSnapshot, PositionDetail  # noqa: E402
from hedge_fund.brokers.models import Fill, Order, Position  # noqa: E402
from hedge_fund.desk import desk_snapshot  # noqa: E402
from hedge_fund.venue import OpenVenue  # noqa: E402
from hedge_fund.web import app as web  # noqa: E402

TOKEN = "test-token-please-ignore"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeBroker:
    venue = "fake"

    def __init__(self, *, cash=12_345.0, positions=None, details=None, account=None):
        self._cash = cash
        self._positions = positions or {}
        self._details = details
        self._account = account
        self.orders_placed: list[Order] = []

    def positions(self):
        return {t: Position(ticker=t, shares=s) for t, s in self._positions.items()}

    def cash(self):
        return self._cash

    def position_details(self):
        return self._details if self._details is not None else []

    def account(self):
        return self._account

    def open_orders(self):
        return {}

    def place_order(self, order: Order) -> Fill:
        self.orders_placed.append(order)
        return Fill(ticker=order.ticker, side=order.side,
                    quantity=order.quantity, price=order.price)


def _opened(broker, *, label="alpaca-paper", live=False, trading_enabled=True):
    return OpenVenue(name="fake", broker=broker, reference=None, label=label,
                     live=live, note="", settings=None)


def _snapshot(broker=None, **opened_kwargs):
    broker = broker or FakeBroker(
        account=AccountSnapshot(venue="fake", cash=12_345.0, buying_power=50_000.0,
                                equity=12_345.0, shorting_enabled=True),
        details=[PositionDetail(ticker="AAPL", shares=10, avg_entry_price=100.0,
                                current_price=110.0, market_value=1_100.0,
                                unrealized_pnl=100.0, unrealized_pnl_pct=0.1)],
    )
    return desk_snapshot(_opened(broker, **opened_kwargs))


def _client(**kwargs):
    kwargs.setdefault("token", TOKEN)
    kwargs.setdefault("snapshot_fn", lambda name: _snapshot())
    return TestClient(web.create_app(**kwargs))


AUTH = {"Authorization": f"Bearer {TOKEN}"}


# ---------------------------------------------------------------------------
# ReadOnlyBroker — the enforcement behind "never submits"
# ---------------------------------------------------------------------------

def test_read_only_broker_forwards_reads_and_refuses_writes():
    inner = FakeBroker(positions={"AAPL": 3}, details=[PositionDetail(ticker="AAPL", shares=3)])
    broker = web.ReadOnlyBroker(inner)

    assert broker.positions()["AAPL"].shares == 3
    assert broker.cash() == 12_345.0
    assert broker.position_details()[0].ticker == "AAPL"
    with pytest.raises(PermissionError, match="read-only"):
        broker.place_order(Order(ticker="AAPL", side="buy", quantity=1, price=1.0))
    assert inner.orders_placed == []   # nothing reached the real broker either


def test_read_only_broker_reports_no_capability_it_does_not_have():
    broker = web.ReadOnlyBroker(object())
    assert broker.account() is None
    assert broker.position_details() is None
    assert broker.open_orders() == {}
    assert broker.calendar("2024-01-01", "2024-01-02") == []
    assert broker.is_market_open() is False


def test_read_only_venue_preserves_everything_but_the_broker():
    from hedge_fund.brokers.alpaca import AlpacaSettings

    gated = AlpacaSettings(api_key="k", secret_key="s", paper=False,
                           trading_enabled=False)
    opened = OpenVenue(name="alpaca", broker=FakeBroker(), reference=None,
                       label="alpaca-live", live=True, note="", settings=gated)

    wrapped = web.read_only(opened)

    assert wrapped.label == "alpaca-live" and wrapped.live is True
    assert wrapped.trading_enabled is False   # the gate travels with the venue
    with pytest.raises(PermissionError):
        wrapped.broker.place_order(Order(ticker="A", side="buy", quantity=1, price=1.0))


# ---------------------------------------------------------------------------
# Tokens and host gating
# ---------------------------------------------------------------------------

def test_a_token_is_always_required(monkeypatch):
    monkeypatch.delenv(web.TOKEN_ENV, raising=False)
    first, second = web.resolve_token(), web.resolve_token()
    assert first and second and first != second  # generated per process, not a constant


def test_explicit_token_beats_the_environment(monkeypatch):
    monkeypatch.setenv(web.TOKEN_ENV, "from-env")
    assert web.resolve_token("explicit") == "explicit"
    assert web.resolve_token() == "from-env"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_are_allowed(host):
    web.check_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "10.0.0.1", "example.com"])
def test_non_loopback_hosts_are_refused(host):
    """0.0.0.0 is every interface, not loopback — publishing the account view."""
    with pytest.raises(ValueError, match="refusing to bind"):
        web.check_host(host)
    web.check_host(host, allow_remote=True)  # unless asked in so many words


def test_serve_refuses_a_remote_host_before_starting(monkeypatch):
    monkeypatch.setenv(web.TOKEN_ENV, TOKEN)
    with pytest.raises(ValueError, match="refusing to bind"):
        web.serve(host="0.0.0.0", port=0)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/", "/api/health", "/api/desk", "/api/stream"])
def test_every_route_needs_a_token(path):
    client = _client()
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(f"{path}?token=wrong").status_code == 401


def test_a_wrong_token_of_the_right_length_is_still_refused():
    client = _client()
    guess = "x" * len(TOKEN)
    assert client.get("/api/desk", headers={"Authorization": f"Bearer {guess}"}).status_code == 401


def test_the_token_works_as_a_header_and_as_a_query_parameter():
    """EventSource cannot set headers, so the stream needs the query form."""
    client = _client()
    assert client.get("/api/desk", headers=AUTH).status_code == 200
    assert client.get(f"/api/desk?token={TOKEN}").status_code == 200


# ---------------------------------------------------------------------------
# It cannot write — structurally
# ---------------------------------------------------------------------------

def test_only_get_routes_are_registered():
    """The strongest form of 'never submits orders': there is no other verb."""
    app = web.create_app(token=TOKEN, snapshot_fn=lambda name: _snapshot())

    methods = {method for route in app.routes for method in getattr(route, "methods", set())}

    assert methods <= {"GET", "HEAD"}
    assert not methods & {"POST", "PUT", "PATCH", "DELETE"}


def test_the_view_is_built_on_a_broker_that_cannot_trade(monkeypatch):
    """Even a future bug in a handler cannot submit through this broker."""
    calls: list[str] = []

    def open_fn(venue, **kwargs):
        calls.append(venue)
        return _opened(FakeBroker())

    app = web.create_app(token=TOKEN, open_fn=open_fn)
    client = TestClient(app)

    assert client.get("/api/desk?venue=paper", headers=AUTH).status_code == 200
    assert calls == ["paper"]
    # And the snapshot path itself hands back a read-only wrapper.
    opened = web.read_only(_opened(FakeBroker()))
    with pytest.raises(PermissionError):
        opened.broker.place_order(Order(ticker="A", side="buy", quantity=1, price=1.0))


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------

def test_desk_payload_carries_the_badge_and_the_holdings():
    client = _client()

    body = client.get("/api/desk", headers=AUTH).json()

    assert body["venue"] == "alpaca-paper"
    assert body["live"] is False
    assert body["cash"] == pytest.approx(12_345.0)
    assert body["buying_power"] == pytest.approx(50_000.0)
    assert [p["ticker"] for p in body["positions"]] == ["AAPL"]
    assert body["positions"][0]["unrealized_pnl"] == pytest.approx(100.0)
    # Computed fields must survive serialization, or the page shows nothing.
    assert body["badge"] == "PAPER"
    assert isinstance(body["warnings"], list)


def test_a_live_venue_is_reported_as_live():
    client = _client(snapshot_fn=lambda name: _snapshot(label="alpaca-live", live=True))

    body = client.get("/api/desk", headers=AUTH).json()

    assert body["live"] is True
    assert body["venue"] == "alpaca-live"


def test_a_venue_that_will_not_load_is_a_503_not_a_traceback():
    def broken(name):
        raise ValueError("ALPACA_API_KEY is not set")

    client = _client(snapshot_fn=broken)

    response = client.get("/api/desk", headers=AUTH)

    assert response.status_code == 503
    assert "ALPACA_API_KEY" in response.json()["detail"]


def test_health_says_read_only():
    body = _client().get("/api/health", headers=AUTH).json()
    assert body == {"ok": True, "venue": "paper", "read_only": True}


def test_index_serves_one_self_contained_page():
    """No static directory, so there is no path to traverse."""
    response = _client().get("/", headers=AUTH)

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "DESK" in response.text


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def test_the_stream_emits_a_snapshot_frame():
    # Bounded, so the response actually ends: the production stream never does.
    client = _client(refresh_seconds=0.01, max_stream_events=1)

    with client.stream("GET", f"/api/stream?token={TOKEN}") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                frames.append(line[len("data: "):])
                break

    assert frames, "the stream produced no data frame"
    assert json.loads(frames[0])["venue"] == "alpaca-paper"


def test_the_stream_reports_an_error_instead_of_dying():
    def broken(name):
        raise RuntimeError("venue unreachable")

    client = _client(refresh_seconds=0.01, max_stream_events=1, snapshot_fn=broken)

    with client.stream("GET", f"/api/stream?token={TOKEN}") as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
                break

    assert "venue unreachable" in payload["error"]
