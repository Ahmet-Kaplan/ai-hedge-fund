"""Core instruments, order lifecycle and protocol conformance."""

from __future__ import annotations

import pytest

from hedge_fund.core import (
    AssetClass, FillEvent, Instrument, InstrumentRegistry, InvalidTransition, MarketDataProvider,
    OrderRequest, OrderState, OrderStatus, equity,
)
from hedge_fund.data.factory import CompositeDataClient
from hedge_fund.data.tiingo import TiingoClient


def test_whole_share_rounding_never_overshoots():
    i = equity("aapl")
    assert i.symbol == "AAPL"
    assert i.round_quantity(10.99) == 10.0
    assert i.round_quantity(-3.7) == -3.0
    assert i.quantity_for_notional(1000.0, 333.0) == 3.0


def test_fractional_and_minimums():
    i = equity("SPY", fractional=True, min_notional=1.0)
    assert i.fractional
    assert i.round_quantity(0.1234567891) == pytest.approx(0.123456)
    lot = Instrument(symbol="EURUSD", asset_class=AssetClass.FX, quantity_step=1000, min_quantity=1000)
    assert lot.round_quantity(2500) == 2000
    assert lot.round_quantity(900) == 0.0


def test_futures_multiplier_in_notional():
    es = Instrument(symbol="ES", asset_class=AssetClass.FUTURE, multiplier=50, tick_size=0.25)
    assert es.notional(2, 5000.0) == 500_000.0
    assert es.quantity_for_notional(600_000.0, 5000.0) == 2.0


def test_registry_defaults_to_whole_share_equities():
    reg = InstrumentRegistry([Instrument(symbol="BTCUSD", asset_class=AssetClass.CRYPTO, quantity_step=1e-8)])
    assert reg.get("btcusd").asset_class == AssetClass.CRYPTO
    assert reg.get("msft").quantity_step == 1.0
    assert "MSFT" in reg


def _req(**kw):
    base = dict(symbol="AAPL", side="buy", quantity=10, decision_session="2024-01-02", reference_price=100.0)
    return OrderRequest(**{**base, **kw})


def _fill(q, cid="x"):
    return FillEvent(client_order_id=cid, symbol="AAPL", side="buy", quantity=q, price=100.1, session="2024-01-03",
                     mid_price=100.0)


def test_order_lifecycle_partial_then_filled():
    s = OrderState(request=_req())
    s.transition(OrderStatus.ACCEPTED)
    s.add_fill(_fill(4))
    assert s.status == OrderStatus.PARTIALLY_FILLED and s.remaining == 6
    s.add_fill(_fill(6))
    assert s.status == OrderStatus.FILLED and s.status.terminal


def test_illegal_transitions_raise():
    s = OrderState(request=_req())
    with pytest.raises(InvalidTransition):
        s.add_fill(_fill(1))                      # not accepted yet
    s.transition(OrderStatus.REJECTED, "no liquidity")
    with pytest.raises(InvalidTransition):
        s.transition(OrderStatus.ACCEPTED)        # terminal
    t = OrderState(request=_req())
    t.transition(OrderStatus.ACCEPTED)
    with pytest.raises(InvalidTransition):
        t.add_fill(_fill(11))                     # overfill


def test_idempotency_key_is_stable_and_decision_specific():
    a, b = _req(), _req()
    assert a.idempotency_key() == b.idempotency_key()
    assert _req(decision_session="2024-01-03").idempotency_key() != a.idempotency_key()
    assert a.with_client_id().client_order_id == a.idempotency_key()


def test_equity_stack_is_one_market_data_provider(tmp_path):
    tiingo = TiingoClient(api_key="k", cache_dir=tmp_path)
    assert isinstance(CompositeDataClient(tiingo, None), MarketDataProvider)
    assert isinstance(tiingo, MarketDataProvider)
