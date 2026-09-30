"""Delisting handling: held names that stop trading never stop a backtest.

Synthetic prices only. Two levels: `resolve_delistings` on a SimBroker for
exact accounting, and full `backtest_fund` replays through the real
pipeline with a fixed-view analyst.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from hedge_fund.backtesting.delisting import DelistingPolicy, resolve_delistings
from hedge_fund.backtesting.fund import backtest_fund
from hedge_fund.backtesting.test_fund import FakeAnalyst
from hedge_fund.brokers.models import Order
from hedge_fund.brokers.sim import SimBroker
from hedge_fund.data.models import Price
from hedge_fund.data.security_events import SecurityEvent, load_events, security_event
from hedge_fund.fund.spec import Fund, FundSpec
from hedge_fund.reporting.decisions import backtest_report, load_receipt
from hedge_fund.reporting.render import to_html, to_markdown


def weekdays(start: str, end: str) -> list[str]:
    d, out = date.fromisoformat(start), []
    while d <= date.fromisoformat(end):
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DAYS = weekdays("2023-01-02", "2023-04-28")
LAST = "2023-01-20"          # last trading day in every scenario
AFTER = [d for d in DAYS if d > LAST]


class Data:
    """{ticker: {day: close}} plus optional security events; logs every
    get_prices call so tests can prove nothing past the session is read."""

    def __init__(self, series: dict[str, dict[str, float]], events: dict[str, SecurityEvent] | None = None):
        self.series = series
        self.events = events or {}
        self.calls: list[tuple[str, str, str]] = []

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        self.calls.append((ticker, start_date, end_date))
        return [Price(open=c, high=c, low=c, close=c, volume=1, time=f"{d}T00:00:00Z")
                for d, c in sorted(self.series.get(ticker, {}).items()) if start_date <= d <= end_date]

    def security_event(self, ticker):
        return self.events.get(ticker)


def listed(until: str = LAST, last_close: float = 90.0, before: float = 100.0) -> dict[str, float]:
    return {d: (last_close if d == until else before) for d in DAYS if d <= until}


def spy() -> dict[str, float]:
    return {d: 100.0 for d in DAYS}


def event(ticker, kind, successor=None, last=LAST):
    return SecurityEvent(ticker=ticker, last_trading_day=last, event_type=kind, successor=successor, note=f"{ticker} {kind}")


@pytest.fixture(autouse=True)
def analyst(monkeypatch):
    from hedge_fund.signals import ALPHA_MODEL_REGISTRY
    monkeypatch.setitem(ALPHA_MODEL_REGISTRY, "a", FakeAnalyst)


def run(data: Data, universe=("XYZ",), views=None, policy=None, end="2023-04-28"):
    spec = FundSpec(schema_version=2, name="t", capital=100_000.0, rebalance="weekly",
                    strategies=[{"name": "solo", "models": [{"name": "a"}], "blend": {"mode": "long_short"}}],
                    risk={"max_position_pct": 1.0, "max_gross_exposure": 1.0})
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views=views or {t: 1.0 for t in universe})]})
    kwargs = {"delisting_policy": policy} if policy else {}
    return backtest_fund(fund, DAYS[0], end, data, list(universe), **kwargs)


def nav_on(result, day):
    return result.nav[result.dates.index(day)]


def cash_ledger(result) -> float:
    """Capital, minus every buy, plus every sell and delisting proceeds."""
    cash = result.capital
    for r in result.records:
        for f in r.fills:
            cash += f.quantity * f.price * (1 if f.side == "sell" else -1)
    return cash + sum(e.proceeds for e in result.delistings)


# ---------------------------------------------------------------------------
# Scenarios through the full backtest
# ---------------------------------------------------------------------------

def test_normal_delisting_liquidates_after_grace_at_last_close():
    result = run(Data({"SPY": spy(), "XYZ": listed()}))
    [e] = result.delistings
    assert (e.kind, e.ticker, e.action, e.policy) == ("DELISTING", "XYZ", "liquidated", "no_close_within_grace_last_close")
    assert (e.delisting_date, e.price, e.price_date, e.shares) == (LAST, 90.0, LAST, 1000)
    assert e.session == AFTER[5] and e.event_type is None          # 6th session without a close
    assert e.proceeds == pytest.approx(90_000.0)
    # held at its last real close through the grace window; NAV never jumps
    assert all(nav_on(result, d) == pytest.approx(90_000.0) for d in DAYS if d >= LAST)
    assert result.records[-1].cash == pytest.approx(90_000.0) and result.records[-1].positions == {}
    assert result.metrics.n_delistings == 1


@pytest.mark.parametrize("kind", ["take_private", "acquisition", "bankruptcy"])
def test_known_event_liquidates_on_first_session_after_last_trading_day(kind):
    result = run(Data({"SPY": spy(), "XYZ": listed()}, {"XYZ": event("XYZ", kind)}))
    [e] = result.delistings
    assert e.session == AFTER[0] and e.policy == "known_delisting_last_close" and e.event_type == kind
    assert (e.price, e.price_date, e.proceeds) == (90.0, LAST, pytest.approx(90_000.0))
    assert nav_on(result, AFTER[0]) == pytest.approx(90_000.0)


def test_take_private_ignores_a_stray_bar_after_the_last_trading_day():
    # vendors sometimes print a bar on the suspension day; the reference
    # stays the last close on or before the listing's last trading day
    series = listed()
    series[AFTER[0]] = 95.0
    result = run(Data({"SPY": spy(), "XYZ": series}, {"XYZ": event("XYZ", "take_private")}))
    [e] = result.delistings
    assert e.session == AFTER[1] and (e.price, e.price_date) == (90.0, LAST)


def test_successor_symbol_converts_share_for_share_without_cash():
    successor_start = AFTER[9]
    data = Data({"SPY": spy(), "SIVB": listed(), "SIVBQ": {d: 0.5 for d in DAYS if d >= successor_start}},
                {"SIVB": event("SIVB", "bank_failure", successor="SIVBQ")})
    result = run(data, universe=("SIVB",))
    conversion = result.delistings[0]
    assert (conversion.action, conversion.policy, conversion.successor) == ("converted_to_successor", "successor_symbol", "SIVBQ")
    assert (conversion.session, conversion.shares, conversion.price, conversion.proceeds) == (successor_start, 1000, 0.5, 0.0)
    assert conversion.event_type == "bank_failure"
    # frozen at the last listed close until the successor trades, then marked at it
    assert nav_on(result, AFTER[8]) == pytest.approx(90_000.0)
    assert nav_on(result, successor_start) == pytest.approx(500.0)
    # the fund later exits the successor through a normal rebalance sell
    assert any(f.ticker == "SIVBQ" and f.side == "sell" for r in result.records for f in r.fills)
    assert cash_ledger(result) == pytest.approx(result.records[-1].cash)


def test_vendor_series_continuing_under_old_symbol_is_recorded_once():
    # Tiingo keeps SIVB's OTC trading under "SIVB" (and serves FRC from FRCB)
    series = listed()
    series.update({d: 0.4 for d in AFTER})
    data = Data({"SPY": spy(), "SIVB": series}, {"SIVB": event("SIVB", "bank_failure", successor="SIVBQ")})
    result = run(data, universe=("SIVB",), views={"SIVB": 0.0}, end="2023-02-10")  # no rebalancing noise
    assert result.delistings == []   # never held: nothing to record
    result = run(data, universe=("SIVB",), end="2023-02-10")
    [e] = result.delistings
    assert (e.action, e.policy, e.session, e.price, e.proceeds) == (
        "continued_as_successor", "successor_prices_under_original_symbol", AFTER[0], 0.4, 0.0)
    assert nav_on(result, AFTER[0]) == pytest.approx(400.0)            # marked at the real OTC price
    assert cash_ledger(result) == pytest.approx(result.records[-1].cash)
    assert "DELISTING: continues as SIVBQ" in to_markdown(backtest_report(result))


def test_failed_bank_whose_successor_never_trades_is_liquidated_after_wait():
    data = Data({"SPY": spy(), "FRC": listed()}, {"FRC": event("FRC", "bank_failure", successor="FRCB")})
    result = run(data, universe=("FRC",), policy=DelistingPolicy(grace_sessions=5, successor_wait_sessions=10))
    [e] = result.delistings
    assert e.session == AFTER[10] and e.policy == "successor_not_trading_last_close"
    assert (e.price, e.price_date, e.successor, e.event_type) == (90.0, LAST, "FRCB", "bank_failure")


def test_held_name_that_stops_on_an_execution_day_is_skipped_not_fatal():
    # views keep asking for XYZ; the Friday before the gap marks it tradeable,
    # so Monday's execution sees a frozen target and must skip it
    result = run(Data({"SPY": spy(), "XYZ": listed()}))
    first_after = next(r for r in result.records if r.execution_as_of == AFTER[0])
    assert first_after.frozen == {"XYZ": LAST}
    assert any(s.ticker == "XYZ" and "not trading" in s.reason for s in first_after.skipped)
    assert first_after.orders == [] and first_after.positions == {"XYZ": 1000}


def test_target_without_a_close_on_execution_day_is_skipped_not_fatal():
    # BBB prints one close on the Friday assessment, then never again: it is
    # targeted, and Monday's execution has no price to buy it at
    friday, monday = "2023-01-27", "2023-01-30"
    data = Data({"SPY": spy(), "AAA": {d: 50.0 for d in DAYS}, "BBB": {friday: 10.0}})
    result = run(data, universe=("AAA", "BBB"), views={"AAA": 1.0, "BBB": 1.0})
    record = next(r for r in result.records if r.execution_as_of == monday)
    assert any(s.ticker == "BBB" and "no close on execution session" in s.reason for s in record.skipped)
    assert all(o.ticker != "BBB" for o in record.orders) and "BBB" not in record.positions
    assert result.delistings == []


# ---------------------------------------------------------------------------
# Missing-price edge cases
# ---------------------------------------------------------------------------

def test_gap_within_grace_resumes_without_liquidation():
    series = {d: 100.0 for d in DAYS}
    for d in AFTER[:5]:          # exactly the grace window
        del series[d]
    result = run(Data({"SPY": spy(), "XYZ": series}))
    assert result.delistings == []
    assert result.records[-1].positions == {"XYZ": 1000}


def test_gap_one_session_longer_than_grace_liquidates_then_can_rebuy():
    series = {d: 100.0 for d in DAYS}
    for d in AFTER[:6]:
        del series[d]
    result = run(Data({"SPY": spy(), "XYZ": series}))
    [e] = result.delistings
    assert e.session == AFTER[5] and e.policy == "no_close_within_grace_last_close"
    assert result.records[-1].positions == {"XYZ": 1000}      # re-bought once it trades again


@pytest.mark.parametrize("bad", [0.0, -5.0, math.nan, math.inf])
def test_invalid_closes_count_as_missing(bad):
    # (the Tiingo adapter already drops such rows; the resolver never trusts them either)
    broker = SimBroker(cash=0.0)
    broker.place_order(Order(ticker="XYZ", side="buy", quantity=10, price=100.0))
    series = listed()
    for d in AFTER:
        series[d] = bad
    res = resolve_delistings(broker, AFTER[0], DAYS, Data({"XYZ": series}, {"XYZ": event("XYZ", "delisted")}), {})
    [e] = res.events
    assert (e.price, e.price_date, e.proceeds) == (90.0, LAST, 900.0)


def test_no_close_in_lookback_uses_the_backtests_own_fill_price():
    broker = SimBroker(cash=0.0)
    broker.place_order(Order(ticker="XYZ", side="buy", quantity=10, price=50.0))   # cash -500
    res = resolve_delistings(broker, "2023-03-01", ["2023-03-01"], Data({}), {"XYZ": 50.0},
                             DelistingPolicy(grace_sessions=0, lookback_days=10))
    [e] = res.events
    assert (e.policy, e.price, e.proceeds) == ("last_fill_price", 50.0, 500.0)
    assert broker.cash() == pytest.approx(0.0) and broker.positions() == {}


def test_unpriceable_position_is_a_loud_data_error():
    broker = SimBroker(cash=0.0)
    broker.place_order(Order(ticker="XYZ", side="buy", quantity=10, price=50.0))
    with pytest.raises(ValueError, match="never priced"):
        resolve_delistings(broker, "2023-03-01", ["2023-03-01"], Data({}), {}, DelistingPolicy(grace_sessions=0))


def test_short_position_is_covered_at_the_last_close():
    broker = SimBroker(cash=10_000.0)
    broker.place_order(Order(ticker="XYZ", side="sell", quantity=20, price=100.0))  # cash 12,000
    data = Data({"XYZ": listed()}, {"XYZ": event("XYZ", "acquisition")})
    res = resolve_delistings(broker, AFTER[0], DAYS, data, {})
    [e] = res.events
    assert (e.shares, e.price, e.proceeds) == (-20, 90.0, pytest.approx(-1_800.0))
    assert broker.cash() == pytest.approx(10_200.0) and broker.positions() == {}


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------

def test_resolver_never_reads_past_the_session():
    broker = SimBroker(cash=0.0)
    broker.place_order(Order(ticker="SIVB", side="buy", quantity=1, price=100.0))
    data = Data({"SIVB": listed(), "SIVBQ": {AFTER[3]: 0.5}}, {"SIVB": event("SIVB", "bank_failure", "SIVBQ")})
    for i, session in enumerate(AFTER[:5]):
        data.calls.clear()
        resolve_delistings(broker, session, DAYS[:DAYS.index(session) + 1], data, {})
        assert data.calls and all(end <= session for _, _, end in data.calls)
    assert "SIVBQ" in broker.positions()   # converted on AFTER[3], not before


def test_later_relisting_price_is_never_used():
    series = listed()
    series["2023-04-03"] = 500.0           # trades again much later, far higher
    result = run(Data({"SPY": spy(), "XYZ": series}), end="2023-03-31")
    [e] = result.delistings
    assert (e.price, e.price_date, e.session) == (90.0, LAST, AFTER[5])
    assert max(result.nav) <= 100_000.0


def test_known_event_is_not_applied_before_its_last_trading_day():
    # a halt before the recorded last trading day is a gap, not the event
    series = {d: 100.0 for d in DAYS}
    halt = [d for d in DAYS if "2023-01-09" < d <= "2023-01-12"]
    for d in halt:
        del series[d]
    result = run(Data({"SPY": spy(), "XYZ": series}, {"XYZ": event("XYZ", "take_private", last="2023-03-31")}),
                 end="2023-03-24")
    assert result.delistings == []
    assert result.records[-1].positions == {"XYZ": 1000}


def test_future_event_changes_nothing_before_it_happens():
    base = run(Data({"SPY": spy(), "XYZ": {d: 100.0 for d in DAYS}}), end="2023-02-24")
    with_event = run(Data({"SPY": spy(), "XYZ": {d: 100.0 for d in DAYS}},
                          {"XYZ": event("XYZ", "acquisition", last="2023-03-31")}), end="2023-02-24")
    assert base.model_dump() == with_event.model_dump()


# ---------------------------------------------------------------------------
# Cash and NAV conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("events", [{}, {"XYZ": event("XYZ", "take_private")}])
def test_liquidation_conserves_nav_and_cash(events):
    result = run(Data({"SPY": spy(), "XYZ": listed()}, events))
    [e] = result.delistings
    before = nav_on(result, DAYS[DAYS.index(e.session) - 1])
    assert nav_on(result, e.session) == pytest.approx(before)           # valued at the same close
    assert before == pytest.approx(e.shares * e.price)                  # fully invested before
    assert cash_ledger(result) == pytest.approx(result.records[-1].cash)
    assert result.nav[-1] == pytest.approx(result.records[-1].cash)


def test_multi_name_book_conserves_cash_through_two_delistings():
    data = Data({"SPY": spy(), "AAA": {d: 50.0 for d in DAYS}, "XYZ": listed(),
                 "TWTR": listed(until="2023-02-03", last_close=54.2, before=50.0)},
                {"TWTR": event("TWTR", "take_private", last="2023-02-03")})
    result = run(data, universe=("AAA", "XYZ", "TWTR"), views={"AAA": 1.0, "XYZ": 1.0, "TWTR": 1.0})
    assert {e.ticker for e in result.delistings} == {"XYZ", "TWTR"}
    final = result.records[-1]
    assert cash_ledger(result) == pytest.approx(final.cash)
    assert result.nav[-1] == pytest.approx(final.cash + sum(s * 50.0 for s in final.positions.values()))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_reports_show_ticker_date_price_proceeds_and_policy():
    result = run(Data({"SPY": spy(), "XYZ": listed()}, {"XYZ": event("XYZ", "take_private")}))
    report = load_receipt(__import__("json").loads(result.model_dump_json()))   # receipt round-trip
    assert report.delistings == backtest_report(result).delistings
    md, html = to_markdown(report), to_html(report)
    for text in (md, html):
        assert "Delistings" in text and "XYZ" in text and LAST in text
        assert "$90.0000" in text and "$90,000.00" in text
        assert "take_private" in text and "last close on or before the last trading day" in text
        assert "DELISTING" in text


def test_successor_report_names_the_successor():
    data = Data({"SPY": spy(), "SIVB": listed(), "SIVBQ": {d: 0.5 for d in DAYS if d >= AFTER[2]}},
                {"SIVB": event("SIVB", "bank_failure", successor="SIVBQ")})
    md = to_markdown(backtest_report(run(data, universe=("SIVB",))))
    assert "DELISTING: converted to SIVBQ" in md and "bank_failure" in md


def test_backtests_without_delistings_are_unchanged():
    result = run(Data({"SPY": spy(), "XYZ": {d: 100.0 for d in DAYS}}))
    assert result.delistings == [] and "Delistings" not in to_markdown(backtest_report(result))


# ---------------------------------------------------------------------------
# Curated events
# ---------------------------------------------------------------------------

def test_bundled_events_are_well_formed():
    events = load_events()
    assert {"TWTR", "ATVI", "SIVB", "FRC"} <= set(events)
    for e in events.values():
        date.fromisoformat(e.last_trading_day)
    assert security_event("sivb").successor == "SIVBQ" and security_event("FRC").successor == "FRCB"
    assert security_event("TWTR").event_type == "take_private" and security_event("AAPL") is None


def test_composite_client_exposes_events():
    from hedge_fund.data.factory import CompositeDataClient
    assert CompositeDataClient(object(), object()).security_event("ATVI").event_type == "acquisition"


# ---------------------------------------------------------------------------
# Vendor placeholder prints (data/tradability.py)
# ---------------------------------------------------------------------------

class VolumeData(Data):
    """Like Data, but {ticker: {day: (close, volume)}}."""

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        self.calls.append((ticker, start_date, end_date))
        return [Price(open=c, high=c, low=c, close=c, volume=v, time=f"{d}T00:00:00Z")
                for d, (c, v) in sorted(self.series.get(ticker, {}).items()) if start_date <= d <= end_date]


def test_tradable_closes_drop_zero_volume_and_post_delisting_carry_forwards():
    from hedge_fund.data.tradability import tradable_closes
    series = {"2023-01-18": (100.0, 10), "2023-01-19": (100.0, 0), "2023-01-20": (90.0, 10),
              "2023-01-23": (90.0, 5), "2023-01-24": (90.0, 0), "2023-01-25": (0.4, 1000)}
    plain = VolumeData({"X": series})
    assert tradable_closes(plain, "X", "2023-01-01", "2023-01-31") == {
        "2023-01-18": 100.0, "2023-01-20": 90.0, "2023-01-23": 90.0, "2023-01-25": 0.4}
    with_event = VolumeData({"X": series}, {"X": event("X", "bank_failure", "XQ")})
    assert tradable_closes(with_event, "X", "2023-01-01", "2023-01-31") == {
        "2023-01-18": 100.0, "2023-01-20": 90.0, "2023-01-25": 0.4}
    assert tradable_closes(with_event, "X", "2023-01-01", "2023-01-23") == {"2023-01-18": 100.0, "2023-01-20": 90.0}
    # the listed close is found even when it predates the requested window
    late = {**series, "2023-03-15": (90.0, 7), "2023-03-16": (0.5, 7)}
    assert tradable_closes(VolumeData({"X": late}, {"X": event("X", "bank_failure", "XQ")}), "X",
                           "2023-03-01", "2023-03-31") == {"2023-03-16": 0.5}


def test_take_private_placeholder_on_suspension_day_is_not_a_close():
    # Tiingo: TWTR 2022-10-28 printed 53.70 (= last close), volume 0
    series = {d: (100.0, 1) for d in DAYS if d < LAST}
    series[LAST] = (90.0, 1)
    series[AFTER[0]] = (90.0, 0)
    result = run(VolumeData({"SPY": {d: (100.0, 1) for d in DAYS}, "XYZ": series}, {"XYZ": event("XYZ", "take_private")}))
    [e] = result.delistings
    assert (e.session, e.price, e.price_date) == (AFTER[0], 90.0, LAST)


def test_halted_failed_bank_with_placeholder_prints_then_successor_trading():
    # Tiingo: SIVB 106.04 on 3/9, the same 106.04 (some with volume) through
    # the halt, then real OTC trades from 3/28 under the same symbol
    halt = AFTER[:12]
    series = {d: (100.0, 1) for d in DAYS if d < LAST}
    series[LAST] = (90.0, 1)
    series.update({d: (90.0, 2404 if i == 1 else 0) for i, d in enumerate(halt)})
    series.update({d: (0.4, 10_000) for d in AFTER[12:]})
    data = VolumeData({"SPY": {d: (100.0, 1) for d in DAYS}, "SIVB": series},
                      {"SIVB": event("SIVB", "bank_failure", successor="SIVBQ")})
    result = run(data, universe=("SIVB",), end="2023-02-28")
    [e] = result.delistings
    assert (e.action, e.session, e.price) == ("continued_as_successor", AFTER[12], 0.4)
    # during the halt: frozen at the last listed close, never traded at a placeholder
    assert all(nav_on(result, d) == pytest.approx(90_000.0) for d in halt)
    halted_executions = [r for r in result.records if r.execution_as_of in halt]
    assert halted_executions and all(r.orders == [] and r.frozen == {"SIVB": LAST} for r in halted_executions)
    assert nav_on(result, AFTER[12]) == pytest.approx(400.0)


def test_never_buys_at_a_placeholder_print():
    # not held; the vendor prints the dead listing's last close after the event
    series = {"2023-01-19": (100.0, 1), LAST: (90.0, 1)}
    series.update({d: (90.0, 3) for d in AFTER})
    data = VolumeData({"SPY": {d: (100.0, 1) for d in DAYS}, "XYZ": series}, {"XYZ": event("XYZ", "acquisition")})
    result = run(data, end="2023-03-31")
    assert all(f.ticker != "XYZ" or f.price != 90.0 or r.execution_as_of <= LAST for r in result.records for f in r.fills)
    assert all(r.positions == {} for r in result.records if r.execution_as_of > LAST)


def test_successor_placeholder_equal_to_last_listed_close_does_not_convert():
    # Tiingo's SIVBQ series repeats 106.04 through the halt, then trades at 0.40
    halt = AFTER[:12]
    data = VolumeData({
        "SPY": {d: (100.0, 1) for d in DAYS},
        "SIVB": {**{d: (100.0, 1) for d in DAYS if d < LAST}, LAST: (90.0, 1)},
        "SIVBQ": {**{d: (90.0, 5) for d in halt}, **{d: (0.4, 10_000) for d in AFTER[12:]}},
    }, {"SIVB": event("SIVB", "bank_failure", successor="SIVBQ")})
    result = run(data, universe=("SIVB",), end="2023-02-28")
    conversion = result.delistings[0]
    assert (conversion.action, conversion.session, conversion.price) == ("converted_to_successor", AFTER[12], 0.4)
    assert all(nav_on(result, d) == pytest.approx(90_000.0) for d in halt)
    assert cash_ledger(result) == pytest.approx(result.records[-1].cash)
