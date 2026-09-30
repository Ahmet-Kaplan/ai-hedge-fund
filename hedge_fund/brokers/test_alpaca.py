"""AlpacaPaperClient tests — a fake requests session, no network."""

import json

import pytest

from hedge_fund.brokers.alpaca import PAPER_BASE_URL, AlpacaError, AlpacaPaperClient


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.headers = {}
        self.calls = []
        self._responses = list(responses)

    def request(self, method, url, params=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "params": params, "json": json})
        return self._responses.pop(0)


def _client(*responses):
    session = FakeSession(*responses)
    return AlpacaPaperClient("key", "secret", session=session), session


def test_refuses_any_non_paper_endpoint():
    with pytest.raises(ValueError, match="paper"):
        AlpacaPaperClient("key", "secret", base_url="https://api.alpaca.markets", session=FakeSession())


def test_requires_keys(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="APCA_API_KEY_ID"):
        AlpacaPaperClient(session=FakeSession())


def test_sends_auth_headers_to_paper_host():
    client, session = _client(FakeResponse(payload={"cash": "10.5", "equity": "20", "last_equity": "19", "status": "ACTIVE"}))
    account = client.account()
    assert (account.cash, account.equity, account.last_equity, account.status) == (10.5, 20.0, 19.0, "ACTIVE")
    assert session.headers == {"APCA-API-KEY-ID": "key", "APCA-API-SECRET-KEY": "secret"}
    assert session.calls[0]["url"] == f"{PAPER_BASE_URL}/v2/account"


def test_positions_are_signed():
    client, _ = _client(FakeResponse(payload=[
        {"symbol": "AAPL", "qty": "10", "side": "long"},
        {"symbol": "TSLA", "qty": "-4", "side": "short"},
    ]))
    assert client.positions() == {"AAPL": 10, "TSLA": -4}


def test_calendar_returns_sorted_session_dates():
    client, session = _client(FakeResponse(payload=[{"date": "2024-06-10"}, {"date": "2024-06-07"}]))
    assert client.calendar("2024-06-01", "2024-06-10") == ["2024-06-07", "2024-06-10"]
    assert session.calls[0]["params"] == {"start": "2024-06-01", "end": "2024-06-10"}


def test_submit_moc_sends_market_on_close():
    client, session = _client(FakeResponse(payload={
        "id": "o1", "client_order_id": "f-2024-06-10-AAPL", "symbol": "AAPL", "side": "buy",
        "qty": "5", "filled_qty": "0", "filled_avg_price": None, "status": "accepted",
    }))
    result = client.submit_moc("AAPL", "buy", 5, "f-2024-06-10-AAPL")
    assert session.calls[0]["json"] == {
        "symbol": "AAPL", "qty": "5", "side": "buy", "type": "market",
        "time_in_force": "cls", "client_order_id": "f-2024-06-10-AAPL",
    }
    assert (result.status, result.order_id, result.quantity, result.filled_qty) == ("accepted", "o1", 5, 0)


def test_refused_order_is_a_rejection_not_a_crash():
    client, _ = _client(FakeResponse(422, {"message": "asset not shortable"}))
    result = client.submit_moc("XYZ", "sell", 5, "f-2024-06-10-XYZ")
    assert result.status == "rejected"
    assert "not shortable" in result.reason


def test_server_error_raises():
    client, _ = _client(FakeResponse(500, {"message": "boom"}))
    with pytest.raises(AlpacaError) as exc:
        client.positions()
    assert exc.value.status_code == 500


def test_list_orders_maps_fills():
    client, session = _client(FakeResponse(payload=[{
        "id": "o1", "client_order_id": "c1", "symbol": "AAPL", "side": "sell",
        "qty": "5", "filled_qty": "5", "filled_avg_price": "101.25", "status": "filled",
    }]))
    [order] = client.list_orders(after="2024-06-10T00:00:00-04:00")
    assert (order.filled_qty, order.filled_avg_price, order.side) == (5, 101.25, "sell")
    assert session.calls[0]["params"]["status"] == "all"


def test_fractional_holdings_are_reported_not_hidden():
    rows = [
        {"symbol": "AAPL", "qty": "10.25", "side": "long", "market_value": "2050.5"},
        {"symbol": "SPY", "qty": "0.02", "side": "long", "market_value": "15.3"},
        {"symbol": "MSFT", "qty": "5", "side": "long", "market_value": "2000"},
    ]
    client, _ = _client(FakeResponse(payload=rows), FakeResponse(payload=rows))
    assert client.positions() == {"AAPL": 10, "MSFT": 5}
    assert client.fractional_holdings() == pytest.approx({"AAPL": 0.25, "SPY": 0.02})
