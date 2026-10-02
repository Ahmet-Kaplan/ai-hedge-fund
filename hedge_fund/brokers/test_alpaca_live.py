"""AlpacaLiveClient tests — fake session, no network, never a real order."""

import pytest

from hedge_fund.brokers.alpaca import LIVE_BASE_URL, AlpacaLiveClient, AlpacaPaperClient
from hedge_fund.brokers.test_alpaca import FakeResponse, FakeSession


def live(*responses):
    session = FakeSession(*responses)
    return AlpacaLiveClient("live-key", "live-secret", session=session), session


def test_uses_live_host_and_live_keys_only(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "paper-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "paper-secret")
    monkeypatch.delenv("ALPACA_LIVE_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_LIVE_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="ALPACA_LIVE_KEY_ID"):
        AlpacaLiveClient(session=FakeSession())          # paper keys are never picked up
    with pytest.raises(ValueError, match="paper"):
        AlpacaLiveClient("paper-key", "x", session=FakeSession())
    client, session = live(FakeResponse(payload={"cash": "5", "equity": "5", "last_equity": "5", "status": "ACTIVE"}))
    client.account()
    assert session.calls[0]["url"] == f"{LIVE_BASE_URL}/v2/account"


def test_paper_client_still_refuses_the_live_host():
    with pytest.raises(ValueError, match="paper"):
        AlpacaPaperClient("k", "s", base_url=LIVE_BASE_URL, session=FakeSession())


def test_holdings_are_signed_fractions():
    client, _ = live(FakeResponse(payload=[
        {"symbol": "SPY", "qty": "0.1308", "side": "long"},
        {"symbol": "XYZ", "qty": "-3", "side": "short"},
    ]))
    assert client.holdings() == {"SPY": pytest.approx(0.1308), "XYZ": -3.0}


def test_buy_notional_and_sell_fraction():
    order = {"id": "o1", "client_order_id": "c", "symbol": "SPY", "side": "buy", "qty": None,
             "notional": "100.00", "filled_qty": "0", "filled_avg_price": None, "status": "accepted"}
    client, session = live(FakeResponse(payload=order), FakeResponse(payload={**order, "side": "sell", "qty": "0.05", "notional": None}))
    bought = client.buy_notional("SPY", 100.0, "c")
    assert session.calls[0]["json"] == {"symbol": "SPY", "notional": "100.00", "side": "buy", "type": "market",
                                        "time_in_force": "day", "client_order_id": "c"}
    assert (bought.notional, bought.qty, bought.status) == (100.0, None, "accepted")
    client.sell_qty("SPY", 0.05, "c")
    assert session.calls[1]["json"]["qty"] == "0.05"


def test_rejection_is_recorded():
    client, _ = live(FakeResponse(403, {"message": "insufficient buying power"}))
    result = client.buy_notional("SPY", 100.0, "c")
    assert result.status == "rejected" and "buying power" in result.reason


def test_cash_flows_map_and_paginate():
    page1 = [{"id": f"a{i}", "activity_type": "CSD", "date": "2026-10-01", "net_amount": "50"} for i in range(100)]
    page2 = [{"id": "b", "activity_type": "DIV", "date": "2026-10-02", "net_amount": "0.12"}]
    client, session = live(FakeResponse(payload=page1), FakeResponse(payload=page2))
    flows = client.cash_flows(after="2026-09-30")
    assert len(flows) == 101 and flows[-1].kind == "DIV" and flows[-1].amount == pytest.approx(0.12)
    assert session.calls[1]["params"]["page_token"] == "a99"


def test_crypto_positions_use_pair_names():
    client, _ = live(FakeResponse(payload=[
        {"symbol": "BTCUSD", "qty": "0.0011", "side": "long", "asset_class": "crypto"},
        {"symbol": "SPY", "qty": "0.2", "side": "long", "asset_class": "us_equity"},
    ]))
    assert client.holdings() == {"BTC/USD": pytest.approx(0.0011), "SPY": pytest.approx(0.2)}


def test_crypto_orders_are_good_til_cancelled():
    order = {"id": "o", "client_order_id": "c", "symbol": "BTC/USD", "side": "buy", "qty": None,
             "notional": "30.00", "filled_qty": "0", "filled_avg_price": None, "status": "accepted"}
    client, session = live(FakeResponse(payload=order), FakeResponse(payload={**order, "symbol": "SPY"}))
    client.buy_notional("BTC/USD", 30.0, "c")
    client.buy_notional("SPY", 30.0, "c")
    assert session.calls[0]["json"]["time_in_force"] == "gtc"
    assert session.calls[1]["json"]["time_in_force"] == "day"
