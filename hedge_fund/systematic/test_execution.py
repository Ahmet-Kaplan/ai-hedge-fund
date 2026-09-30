"""Execution costs, timing, liquidity, corporate actions and benchmark consistency."""

from __future__ import annotations

import math

import pytest

from hedge_fund.core import OrderRequest, OrderStatus
from hedge_fund.data.security_events import SecurityEvent
from hedge_fund.systematic import MarketPanel
from hedge_fund.systematic.benchmark import benchmark_curve
from hedge_fund.systematic.execution import (
    ZERO_COST, CostModel, FillTiming, LookAheadExecution, SimulatedExecution,
)
from hedge_fund.systematic.ledger import Ledger
from hedge_fund.systematic.testing import RawBar, SyntheticMarket, weekdays

DAYS = weekdays("2023-01-02", "2023-03-31")
D0, D1 = "2023-02-01", "2023-02-02"


def market() -> SyntheticMarket:
    m = SyntheticMarket()
    m.add_series("SPY", DAYS, [400.0 + i for i in range(len(DAYS))], volume=50_000_000, gap=0.001)
    # AAA: flat 100 close, open 1% above prior close; $1M/day volume ($100M ADV notional)
    m.add_series("AAA", DAYS, [100.0] * len(DAYS), volume=1_000_000, gap=0.01)
    # THIN: $10k/day traded
    m.add_series("THIN", DAYS, [10.0] * len(DAYS), volume=1_000)
    # ZERO: no volume at all until the execution day D1 (no ADV history)
    m.add("ZERO", [RawBar(d, 5, 5, 5, 5, 100 if d == D1 else 0) for d in DAYS])
    # DIV: pays $2 on 2023-03-01 (ex-date), splits 2:1 on 2023-03-15
    bars = []
    for d in DAYS:
        px = 50.0 if d >= "2023-03-15" else 100.0
        bars.append(RawBar(d, px, px, px, px, 100_000, dividend=2.0 if d == "2023-03-01" else 0.0,
                           split=2.0 if d == "2023-03-15" else 1.0))
    m.add("DIV", bars)
    # GONE: acquired, last trading day 2023-02-10; vendor prints placeholders after
    m.add("GONE", [RawBar(d, 30, 30, 30, 30, 10_000 if d <= "2023-02-10" else 5) for d in DAYS])
    m.events["GONE"] = SecurityEvent(ticker="GONE", last_trading_day="2023-02-10", event_type="acquisition")
    return m


@pytest.fixture(scope="module")
def panel():
    return MarketPanel.build(market(), ["SPY", "AAA", "THIN", "ZERO", "DIV", "GONE"], DAYS[0], DAYS[-1])


def order(symbol="AAA", side="buy", qty=100, decided=D0, ref=100.0):
    return OrderRequest(symbol=symbol, side=side, quantity=qty, decision_session=decided, reference_price=ref,
                        strategy="t")


def ex(panel, timing=FillTiming.NEXT_OPEN, **costs):
    return SimulatedExecution(panel, CostModel(**costs) if costs else ZERO_COST, timing)


def test_fill_on_decision_session_or_earlier_is_refused(panel):
    e = ex(panel)
    with pytest.raises(LookAheadExecution):
        e.execute(order(), D0)
    with pytest.raises(LookAheadExecution):
        e.execute(order(), "2023-01-31")


def test_timing_must_be_explicit(panel):
    with pytest.raises(TypeError):
        SimulatedExecution(panel, ZERO_COST, "next_open")


def test_next_open_fills_at_the_next_sessions_open(panel):
    s = ex(panel, FillTiming.NEXT_OPEN).execute(order(), D1)
    assert s.status is OrderStatus.FILLED
    assert s.fills[0].mid_price == pytest.approx(101.0)      # open = prior close * 1.01
    assert s.fills[0].session == D1


def test_next_close_fills_at_the_next_sessions_close(panel):
    s = ex(panel, FillTiming.NEXT_CLOSE).execute(order(), D1)
    assert s.fills[0].mid_price == pytest.approx(100.0)


def test_commission_forms(panel):
    e = ex(panel, commission_per_share=0.01, commission_min=1.0, half_spread_bps=0, impact_coef=0,
           max_participation=1.0)
    assert e.execute(order(qty=10), D1).fills[0].commission == pytest.approx(1.0)          # minimum binds
    assert e.execute(order(qty=1000), D1).fills[0].commission == pytest.approx(10.0)
    e2 = ex(panel, commission_bps=10, half_spread_bps=0, impact_coef=0, max_participation=1.0)
    f = e2.execute(order(qty=1000), D1).fills[0]
    assert f.commission == pytest.approx(1000 * 101.0 * 0.001)


def test_half_spread_moves_price_against_the_trader(panel):
    e = ex(panel, half_spread_bps=10, impact_coef=0, max_participation=1.0)
    buy = e.execute(order(side="buy"), D1).fills[0]
    sell = e.execute(order(side="sell"), D1).fills[0]
    assert buy.price == pytest.approx(101.0 * 1.001)
    assert sell.price == pytest.approx(101.0 * 0.999)
    assert buy.spread_cost == pytest.approx(100 * 101.0 * 0.001)


def test_impact_grows_with_size_relative_to_adv(panel):
    e = ex(panel, half_spread_bps=0, impact_coef=1.0, max_participation=1.0, default_daily_vol=0.02)
    small = e.execute(order(qty=100), D1).fills[0]
    big = e.execute(order(qty=100_000), D1).fills[0]
    assert big.price > small.price > 101.0
    # flat closes -> zero realised vol -> default vol used; sqrt law on notional/ADV
    adv, vol = e.liquidity("AAA", D1)
    assert adv == pytest.approx(100.0 * 1_000_000)
    assert big.impact_cost / (100_000 * 101.0) == pytest.approx(vol * math.sqrt(100_000 * 101.0 / adv))


def test_zero_adv_is_rejected(panel):
    s = ex(panel).execute(order(symbol="ZERO", ref=5.0), D1)
    assert s.status is OrderStatus.REJECTED and "liquidity" in s.reject_reason


def test_zero_volume_session_is_not_tradable(panel):
    s = ex(panel).execute(order(symbol="ZERO", ref=5.0, decided=D1), "2023-02-03")
    assert s.status is OrderStatus.REJECTED and "tradable" in s.reject_reason


def test_illiquid_order_is_capped_by_participation(panel):
    e = ex(panel, max_participation=0.10, half_spread_bps=0, impact_coef=0)
    s = e.execute(order(symbol="THIN", qty=1_000, ref=10.0), D1)
    assert s.status is OrderStatus.EXPIRED          # partial fill, remainder expired
    assert s.filled_quantity == 100                 # 10% of 1,000 shares/day
    assert s.fills[0].quantity == 100


def test_unavailable_price_after_delisting_is_rejected(panel):
    s = ex(panel).execute(order(symbol="GONE", ref=30.0, decided="2023-02-14"), "2023-02-15")
    assert s.status is OrderStatus.REJECTED and "tradable" in s.reject_reason
    ok = ex(panel).execute(order(symbol="GONE", ref=30.0, decided="2023-02-08"), "2023-02-09")
    assert ok.status is OrderStatus.FILLED


def test_rejection_when_notional_below_venue_minimum(panel):
    from hedge_fund.core import Instrument, InstrumentRegistry
    reg = InstrumentRegistry([Instrument(symbol="AAA", min_notional=1_000.0)])
    s = SimulatedExecution(panel, ZERO_COST, FillTiming.NEXT_OPEN, reg).execute(order(qty=5), D1)
    assert s.status is OrderStatus.REJECTED and "minimum" in s.reject_reason


def test_execution_is_deterministic(panel):
    e = ex(panel, commission_bps=1, half_spread_bps=3, impact_coef=0.2)
    a = [e.execute(order(qty=q), D1).model_dump() for q in (10, 500, 50_000)]
    b = [e.execute(order(qty=q), D1).model_dump() for q in (10, 500, 50_000)]
    assert a == b


def test_dividend_credited_on_ex_date_only_for_holders(panel):
    led = Ledger(cash=10_000.0)
    led.apply_fill(ex(panel, FillTiming.NEXT_CLOSE).execute(order(symbol="DIV", qty=10, decided="2023-02-27"),
                                                            "2023-02-28").fills[0])
    assert led.apply_corporate_actions(panel.as_of("2023-02-28")) == []
    flows = led.apply_corporate_actions(panel.as_of("2023-03-01"))
    assert [f.amount for f in flows] == [pytest.approx(20.0)]
    assert led.total("dividend") == pytest.approx(20.0)
    assert led.reconcile() == pytest.approx(0.0)


def test_split_multiplies_shares_and_preserves_value(panel):
    led = Ledger(cash=10_000.0)
    led.apply_fill(ex(panel, FillTiming.NEXT_CLOSE).execute(order(symbol="DIV", qty=10, decided="2023-03-08"),
                                                            "2023-03-09").fills[0])
    before = led.equity({"DIV": panel.as_of("2023-03-14").close("DIV")})
    led.apply_corporate_actions(panel.as_of("2023-03-15"))
    assert led.quantity("DIV") == pytest.approx(20)
    after = led.equity({"DIV": panel.as_of("2023-03-15").close("DIV")})
    assert after == pytest.approx(before)          # 20 x 50 == 10 x 100


def test_short_opening_fill_is_refused_by_default(panel):
    led = Ledger(cash=1_000.0)
    f = ex(panel).execute(order(side="sell", qty=1), D1).fills[0]
    with pytest.raises(ValueError, match="short"):
        led.apply_fill(f)


def test_benchmark_total_return_includes_dividends(panel):
    sessions = panel.sessions_through("2023-03-31")
    price = benchmark_curve(panel, "DIV", sessions, 1_000.0, total_return=False)
    tr = benchmark_curve(panel, "DIV", sessions, 1_000.0, total_return=True)
    assert price.iloc[-1] == pytest.approx(1_000.0)             # split-adjusted flat price
    assert tr.iloc[-1] == pytest.approx(1_000.0 * (1 + 2.0 / 100.0))
