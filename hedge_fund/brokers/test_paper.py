"""Paper broker lifecycle and locked real-broker adapters (no network)."""

from __future__ import annotations

import pytest

from hedge_fund.brokers import adapters
from hedge_fund.brokers.adapters import AlpacaAdapter, BrokerNotConfigured, IBKRAdapter, LiveTradingDisabled
from hedge_fund.brokers.paper import PaperBroker
from hedge_fund.core import OrderRequest, OrderStatus, TradingVenue
from hedge_fund.systematic import MarketPanel
from hedge_fund.systematic.execution import ZERO_COST, CostModel, FillTiming, SimulatedExecution
from hedge_fund.systematic.testing import SyntheticMarket, weekdays

DAYS = weekdays("2023-01-02", "2023-03-31")
D0, D1, D2 = "2023-02-01", "2023-02-02", "2023-02-03"


@pytest.fixture(scope="module")
def panel():
    m = SyntheticMarket()
    m.add_series("SPY", DAYS, [400.0] * len(DAYS), volume=10_000_000)
    m.add_series("AAA", DAYS, [100.0] * len(DAYS), volume=1_000_000)
    m.add_series("THIN", DAYS, [10.0] * len(DAYS), volume=1_000)
    return MarketPanel.build(m, ["SPY", "AAA", "THIN"], DAYS[0], DAYS[-1])


def broker(panel, cash=100_000.0, costs=ZERO_COST):
    return PaperBroker(cash, SimulatedExecution(panel, costs, FillTiming.NEXT_OPEN))


def order(symbol="AAA", side="buy", qty=10, tif="day", decided=D0):
    return OrderRequest(symbol=symbol, side=side, quantity=qty, decision_session=decided, reference_price=100.0,
                        time_in_force=tif, strategy="t")


def test_is_a_trading_venue(panel):
    assert isinstance(broker(panel), TradingVenue) and broker(panel).live is False


def test_accept_then_fill_on_a_later_session(panel):
    b = broker(panel)
    s = b.submit(order())
    assert s.status is OrderStatus.ACCEPTED
    assert b.process(D0) == [] and s.status is OrderStatus.ACCEPTED       # not on the decision session
    b.process(D1)
    assert s.status is OrderStatus.FILLED and b.positions() == {"AAA": 10}
    assert b.cash() == pytest.approx(100_000 - 1_000)


def test_duplicate_submission_is_idempotent(panel):
    b = broker(panel)
    first = b.submit(order())
    again = b.submit(order())
    assert again is first and len(b.orders) == 1
    b.process(D1)
    assert b.positions() == {"AAA": 10}


def test_rejections(panel):
    b = broker(panel, cash=500.0)
    assert b.submit(order(qty=100)).status is OrderStatus.REJECTED              # cash
    assert b.submit(order(side="sell")).status is OrderStatus.REJECTED          # short
    lim = OrderRequest(symbol="AAA", side="buy", quantity=1, decision_session=D0, reference_price=100.0,
                       order_type="limit", limit_price=99.0)
    assert b.submit(lim).status is OrderStatus.REJECTED


def test_partial_fill_day_expires_gtc_stays_open(panel):
    costs = CostModel(half_spread_bps=0, impact_coef=0, max_participation=0.1)
    day = broker(panel, costs=costs)
    s = day.submit(order(symbol="THIN", qty=250))
    day.process(D1)
    assert s.status is OrderStatus.EXPIRED and s.filled_quantity == 100
    gtc = broker(panel, costs=costs)
    g = gtc.submit(order(symbol="THIN", qty=250, tif="gtc"))
    gtc.process(D1)
    assert g.status is OrderStatus.PARTIALLY_FILLED and g.remaining == 150
    gtc.process(D2)
    gtc.process("2023-02-06")
    assert g.status is OrderStatus.FILLED and gtc.positions() == {"THIN": 250}


def test_cancel_and_reconcile(panel):
    b = broker(panel)
    s = b.submit(order())
    assert b.cancel(s.request.client_order_id).status is OrderStatus.CANCELLED
    b.process(D1)
    assert b.positions() == {}
    b.submit(order(qty=5, decided=D1))
    b.process(D2)
    assert b.reconcile({"AAA": 5}, b.cash()).ok
    bad = b.reconcile({"AAA": 4}, b.cash() + 10)
    assert not bad.ok and bad.position_differences == {"AAA": 1} and bad.cash_difference == pytest.approx(-10)


def test_alpaca_is_paper_only_and_needs_credentials(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(BrokerNotConfigured):
        AlpacaAdapter()
    monkeypatch.setenv("AIHF_ALLOW_LIVE_TRADING", "1")
    with pytest.raises(LiveTradingDisabled):
        AlpacaAdapter(live=True)                     # even with the env opt-in: hard lock
    assert adapters.LIVE_TRADING_ENABLED is False


def test_alpaca_payload_and_transport_stay_on_paper(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "k-not-real")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s-not-real")
    calls = []

    def transport(method, url, **kw):
        calls.append((method, url, kw))
        return {"status": "accepted"}

    a = AlpacaAdapter(transport=transport)
    s = a.submit(OrderRequest(symbol="AAA", side="buy", quantity=3, decision_session=D0, reference_price=100.0,
                              order_type="moo"))
    assert s.status is OrderStatus.ACCEPTED
    method, url, kw = calls[0]
    assert method == "POST" and url == "https://paper-api.alpaca.markets/v2/orders"
    assert kw["json"]["time_in_force"] == "opg" and kw["json"]["qty"] == "3.0"
    assert "k-not-real" not in str(kw)
    a.base_url = "https://api.alpaca.markets"           # live host
    with pytest.raises(LiveTradingDisabled):
        a.positions()


def test_ibkr_is_not_configured_and_refuses_live_ports():
    with pytest.raises(LiveTradingDisabled):
        IBKRAdapter(live=True)
    with pytest.raises(LiveTradingDisabled):
        IBKRAdapter(port=4001)                          # live gateway port
    with pytest.raises(BrokerNotConfigured):
        IBKRAdapter()
