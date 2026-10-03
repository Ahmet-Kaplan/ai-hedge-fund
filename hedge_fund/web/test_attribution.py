"""Tests for analyst attribution.

The arithmetic here decides which analyst a user is told to follow, so the
cases that matter are the ones where a plausible-looking shortcut gives the
wrong answer: an analyst outvoted by their pod, an abstention mistaken for a
neutral view, and a short record mistaken for a track record.
"""

from __future__ import annotations

from datetime import date as _date
from datetime import timedelta

import pytest

from hedge_fund.backtesting.fund import FundBacktestMetrics, FundBacktestResult
from hedge_fund.brokers.models import Fill
from hedge_fund.fund.spec import BlendPolicy, FundSpec, ModelSpec, StrategySpec
from hedge_fund.models import Signal
from hedge_fund.pipeline.models import CycleRecord, DecisionRecord, StrategyRecord
from hedge_fund.pipeline.session import SessionRecord
from hedge_fund.risk.limits import RiskLimits
from hedge_fund.web import attribution


def spec(*models: tuple[str, float], name: str = "test-fund") -> FundSpec:
    return FundSpec(
        schema_version=2,
        name=name,
        strategies=[StrategySpec(
            name="pod",
            models=[ModelSpec(name=n, weight=w) for n, w in models],
            blend=BlendPolicy(mode="long_only"),
        )],
        risk=RiskLimits(max_position_pct=0.25, max_gross_exposure=1.0),
    )


def signal(name: str, ticker: str, value: float, *, abstained: bool = False) -> Signal:
    return Signal(
        model_name=name, ticker=ticker, date="2026-01-05", value=value,
        reasoning=f"{name} on {ticker}",
        metadata={"abstained": True} if abstained else {},
    )


def assessment(as_of: str, marks: dict, fund_spec: FundSpec,
               signals: list[Signal] | None = None,
               contribution: dict | None = None) -> DecisionRecord:
    """An assessment, which is where universe-wide prices actually live."""
    return DecisionRecord(
        fund=fund_spec.name, as_of=as_of, spec=fund_spec, universe=sorted(marks),
        marks=marks, skipped=[], target_weights={}, clamps=[], final_weights={},
        strategies=[StrategyRecord(
            name="pod", slice=1.0, signals=signals or [], convictions={}, weights={},
            final_contribution=contribution or {},
        )],
    )


def cycle(as_of: str, marks: dict, signals: list[Signal], contribution: dict,
          fund_spec: FundSpec, nav: float = 100_000.0) -> CycleRecord:
    """An executed rebalance that traded nothing — the common real case.

    `CycleRecord.marks` prices only the names the fund held or targeted, so a
    cycle that placed no orders carries none. The prices live on the
    assessment, and attribution has to read them from there.
    """
    decided = assessment(as_of, marks, fund_spec, signals, contribution)
    return CycleRecord(
        fund=fund_spec.name, as_of=as_of, spec=fund_spec, universe=sorted(marks),
        marks={}, skipped=[], target_weights={}, clamps=[], final_weights={},
        equity_before=nav, cash_before=nav, orders=[], fills=[], positions={},
        cash=0.0, nav=nav,
        strategies=[StrategyRecord(
            name="pod", slice=1.0, signals=signals, convictions={}, weights={},
            final_contribution=contribution,
        )],
        original_assessment=decided, refreshed_assessment=decided,
    )


def session(date: str, marks: dict, executed: CycleRecord | None,
            decided: DecisionRecord | None = None) -> SessionRecord:
    return SessionRecord(
        fund="test-fund", session=date, universe=sorted(marks), nav=100_000.0,
        cash=0.0, positions={}, marks=marks, benchmark="SPY", benchmark_close=500.0,
        rebalance=executed is not None, executed=executed, decision=decided,
        code_version="test", mandate_hash="test",
    )


def result_of(*sessions: SessionRecord) -> FundBacktestResult:
    dates = [s.session for s in sessions]
    return FundBacktestResult(
        fund="test-fund", start=dates[0], end=dates[-1], rebalance="weekly",
        benchmark="SPY", universe=["AAPL"], capital=100_000.0, dates=dates,
        nav=[100_000.0] * len(dates), benchmark_nav=[100_000.0] * len(dates),
        metrics=FundBacktestMetrics(
            total_return_pct=0.0, annualized_return_pct=0.0, sharpe_ratio=0.0,
            max_drawdown_pct=0.0, benchmark_return_pct=0.0, excess_return_pct=0.0,
            n_cycles=len(dates), n_orders=0,
        ),
        records=list(sessions),
    )


# ---- shares reproduce the blend exactly ------------------------------------

def test_shares_of_a_pod_sum_to_one() -> None:
    """Conviction is a weighted mean, so the parts must total the whole.

    If this drifts, attributed dollars silently stop adding up to the
    position the fund actually held.
    """
    fund_spec = spec(("graham", 2.0), ("buffett", 1.0))
    first = cycle("2026-01-05", {"AAPL": 100.0},
                  [signal("graham", "AAPL", 0.6), signal("buffett", "AAPL", 0.3)],
                  {"AAPL": 0.5}, fund_spec)
    second = cycle("2026-01-12", {"AAPL": 110.0}, [], {}, fund_spec)

    calls = attribution.calls_from_result(result_of(
        session("2026-01-05", {"AAPL": 100.0}, first),
        session("2026-01-12", {"AAPL": 110.0}, second),
    ))
    aapl = [c for c in calls if c.ticker == "AAPL" and c.date == "2026-01-05"]

    assert sum(c.share for c in aapl) == pytest.approx(1.0)
    # graham carries twice the weight and twice the conviction: 1.2 of 1.5.
    graham = next(c for c in aapl if c.analyst == "graham")
    assert graham.share == pytest.approx(0.8)
    assert graham.dollars == pytest.approx(0.8 * 0.5 * 100_000.0)


def test_an_analyst_who_argued_against_the_position_gets_a_negative_share() -> None:
    """Being outvoted is not the same as being absent, and must not read as it."""
    fund_spec = spec(("graham", 1.0), ("buffett", 1.0))
    first = cycle("2026-01-05", {"AAPL": 100.0},
                  [signal("graham", "AAPL", 0.9), signal("buffett", "AAPL", -0.3)],
                  {"AAPL": 0.4}, fund_spec)
    second = cycle("2026-01-12", {"AAPL": 90.0}, [], {}, fund_spec)

    calls = attribution.calls_from_result(result_of(
        session("2026-01-05", {"AAPL": 100.0}, first),
        session("2026-01-12", {"AAPL": 90.0}, second),
    ))
    buffett = next(c for c in calls if c.analyst == "buffett")

    assert buffett.share < 0
    assert buffett.dollars < 0
    # The name fell and he was bearish, so the call was right.
    assert buffett.correct is True
    assert buffett.edge > 0


# ---- abstention is not a neutral view --------------------------------------

def test_an_abstention_is_not_scored_as_a_neutral_call() -> None:
    """Counting abstentions as 0.0 conviction would manufacture a 50% hit rate
    for an analyst who never had the data to form a view."""
    fund_spec = spec(("graham", 1.0), ("buffett", 1.0))
    first = cycle("2026-01-05", {"AAPL": 100.0},
                  [signal("graham", "AAPL", 0.5),
                   signal("buffett", "AAPL", 0.0, abstained=True)],
                  {"AAPL": 0.3}, fund_spec)
    second = cycle("2026-01-12", {"AAPL": 110.0}, [], {}, fund_spec)

    calls = attribution.calls_from_result(result_of(
        session("2026-01-05", {"AAPL": 100.0}, first),
        session("2026-01-12", {"AAPL": 110.0}, second),
    ))
    scores = {s.analyst: s for s in attribution.score(calls)}

    assert scores["buffett"].calls == 0
    assert scores["buffett"].abstentions == 1
    assert scores["buffett"].hit_rate is None
    assert scores["buffett"].participation == 0.0
    # The abstainer is excluded from the blend, so graham owns all of it.
    assert next(c for c in calls if c.analyst == "graham").share == pytest.approx(1.0)


def test_an_abstainer_earns_no_dollars() -> None:
    fund_spec = spec(("graham", 1.0), ("buffett", 1.0))
    first = cycle("2026-01-05", {"AAPL": 100.0},
                  [signal("graham", "AAPL", 0.5),
                   signal("buffett", "AAPL", 0.0, abstained=True)],
                  {"AAPL": 0.9}, fund_spec)
    second = cycle("2026-01-12", {"AAPL": 200.0}, [], {}, fund_spec)

    calls = attribution.calls_from_result(result_of(
        session("2026-01-05", {"AAPL": 100.0}, first),
        session("2026-01-12", {"AAPL": 200.0}, second),
    ))
    buffett = next(c for c in calls if c.analyst == "buffett")

    assert buffett.dollars == 0.0
    assert buffett.edge is None


# ---- a live fund's own ledger is a source, not just backtests ---------------

def test_a_ledger_is_read_directly_without_a_backtest_wrapper() -> None:
    """Paper trading is the backtest one tick at a time, so the same records
    must score the same way whichever container wrote them. Reading only
    simulations is what let the page describe a universe the deployed fund
    had not used in months."""
    fund_spec = spec(("graham", 1.0))
    first = cycle("2026-01-05", {"AAPL": 100.0}, [signal("graham", "AAPL", 0.5)],
                  {"AAPL": 0.4}, fund_spec)
    ledger = [
        session("2026-01-05", {"AAPL": 100.0}, first),
        session("2026-01-12", {"AAPL": 110.0}, cycle("2026-01-12", {"AAPL": 110.0}, [], {}, fund_spec)),
    ]

    assert attribution.calls_from_records(ledger) == attribution.calls_from_result(result_of(*ledger))


def test_a_decision_awaiting_execution_is_already_a_call() -> None:
    """A fund that rebalances weekly assesses today and trades tomorrow. If
    only executed cycles counted, the newest views would be invisible for a
    full rebalance and a freshly deployed fund would contribute nothing at
    all — which is how the leaderboard came to lag the mandate."""
    fund_spec = spec(("graham", 1.0))
    pending = assessment("2026-01-05", {"KSS": 20.0}, fund_spec,
                         [signal("graham", "KSS", 0.7)], {"KSS": 0.25})

    calls = attribution.calls_from_records([session("2026-01-05", {}, None, pending)])

    assert [c.ticker for c in calls] == ["KSS"]
    assert calls[0].conviction == 0.7 and calls[0].price == 20.0
    # Nothing has happened to it yet, so it is recorded and left unscored
    # rather than credited with a return it has not earned.
    assert calls[0].forward_return is None and calls[0].edge is None
    assert calls[0].fill is None


def test_an_executed_decision_is_not_counted_twice() -> None:
    """The assessment at T rides on T's record and again inside T+1's cycle.
    Scoring both would double every analyst's evidence."""
    fund_spec = spec(("graham", 1.0))
    decided = assessment("2026-01-05", {"AAPL": 100.0}, fund_spec,
                         [signal("graham", "AAPL", 0.5)], {"AAPL": 0.4})
    executed = cycle("2026-01-05", {"AAPL": 100.0}, [signal("graham", "AAPL", 0.5)],
                     {"AAPL": 0.4}, fund_spec)

    calls = attribution.calls_from_records([
        session("2026-01-05", {}, None, decided),
        session("2026-01-06", {"AAPL": 100.0}, executed),
    ])

    assert len([c for c in calls if c.date == "2026-01-05"]) == 1


# ---- scoring ---------------------------------------------------------------

def test_edge_rewards_conviction_in_the_right_direction() -> None:
    fund_spec = spec(("graham", 1.0))
    first = cycle("2026-01-05", {"AAPL": 100.0}, [signal("graham", "AAPL", 0.5)],
                  {"AAPL": 1.0}, fund_spec)
    second = cycle("2026-01-12", {"AAPL": 120.0}, [], {}, fund_spec)

    calls = attribution.calls_from_result(result_of(
        session("2026-01-05", {"AAPL": 100.0}, first),
        session("2026-01-12", {"AAPL": 120.0}, second),
    ))
    graham = next(c for c in calls if c.analyst == "graham")

    assert graham.forward_return == pytest.approx(0.2)
    assert graham.edge == pytest.approx(0.1)   # 0.5 conviction × 20% move
    assert graham.correct is True


def test_the_final_call_is_priced_against_the_trailing_assessment() -> None:
    """A call at the last rebalance still has a measurable outcome.

    The backtest assesses once more at the closing session and never
    executes it; that unexecuted assessment is what prices the last call.
    """
    fund_spec = spec(("graham", 1.0))
    only = cycle("2026-01-05", {"AAPL": 100.0}, [signal("graham", "AAPL", 1.0)],
                 {"AAPL": 1.0}, fund_spec)

    calls = attribution.calls_from_result(result_of(
        session("2026-01-05", {"AAPL": 100.0}, only),
        session("2026-01-30", {}, None, assessment("2026-01-30", {"AAPL": 130.0}, fund_spec)),
    ))

    assert calls[0].forward_return == pytest.approx(0.3)


def test_a_name_the_fund_never_traded_is_still_scored() -> None:
    """The bug this guards against under-reported 252 calls down to 6.

    An analyst can have a view on a name the fund declined to buy. Those
    calls are absent from the cycle's own marks — pricing from there scores
    only the handful of names that were traded, which quietly turns the
    leaderboard into a measure of what the fund held rather than who was
    right.
    """
    fund_spec = spec(("graham", 1.0))
    traded_nothing = cycle("2026-01-05", {"AAPL": 100.0},
                           [signal("graham", "AAPL", 1.0)], {}, fund_spec)
    assert traded_nothing.marks == {}, "fixture must mirror a cycle that traded nothing"

    calls = attribution.calls_from_result(result_of(
        session("2026-01-05", {}, traded_nothing),
        session("2026-01-12", {}, None, assessment("2026-01-12", {"AAPL": 125.0}, fund_spec)),
    ))

    assert calls[0].forward_return == pytest.approx(0.25)
    assert calls[0].dollars == 0.0      # right call, no position behind it
    assert attribution.score(calls)[0].scored == 1


def test_a_call_with_no_closing_price_is_kept_but_unscored() -> None:
    """An opinion we cannot price is still an opinion the analyst voiced."""
    fund_spec = spec(("graham", 1.0))
    only = cycle("2026-01-05", {"AAPL": 100.0}, [signal("graham", "AAPL", 1.0)],
                 {"AAPL": 1.0}, fund_spec)

    calls = attribution.calls_from_result(result_of(
        session("2026-01-05", {"AAPL": 100.0}, only),
        session("2026-01-30", {}, None),
    ))

    assert len(calls) == 1
    assert calls[0].forward_return is None
    assert calls[0].edge is None
    assert attribution.score(calls)[0].scored == 0


# ---- overlapping backtests must not inflate a record -----------------------

def test_the_same_rebalance_replayed_twice_is_counted_once() -> None:
    """Rerunning a mandate over a longer window replays shared rebalances.

    Without this, an analyst's call count and attributed P&L grow every time
    someone presses Run, and the leaderboard measures simulation frequency
    rather than skill.
    """
    fund_spec = spec(("graham", 1.0))

    def one_window() -> list[attribution.Call]:
        first = cycle("2026-01-05", {"AAPL": 100.0}, [signal("graham", "AAPL", 1.0)],
                      {"AAPL": 1.0}, fund_spec)
        return attribution.calls_from_result(result_of(
            session("2026-01-05", {}, first),
            session("2026-01-12", {}, None, assessment("2026-01-12", {"AAPL": 110.0}, fund_spec)),
        ))

    overlapping = one_window() + one_window()
    assert len(overlapping) == 2

    merged = attribution.deduplicate(overlapping)

    assert len(merged) == 1
    assert attribution.score(merged)[0].pnl == pytest.approx(
        attribution.score(one_window())[0].pnl
    )


def test_two_mandates_making_the_same_call_both_count() -> None:
    """Deduplication keys on the mandate, so independent funds stay independent.

    Two different mandates reaching the same conclusion is corroboration,
    not duplication, and collapsing them would discard real evidence.
    """
    calls = []
    for name in ("fund-a", "fund-b"):
        fund_spec = spec(("graham", 1.0), name=name)
        only = cycle("2026-01-05", {"AAPL": 100.0}, [signal("graham", "AAPL", 1.0)],
                     {"AAPL": 1.0}, fund_spec)
        result = result_of(
            session("2026-01-05", {}, only),
            session("2026-01-12", {}, None, assessment("2026-01-12", {"AAPL": 110.0}, fund_spec)),
        )
        result.fund = name
        calls.extend(attribution.calls_from_result(result))

    assert len(attribution.deduplicate(calls)) == 2


def test_deduplication_keeps_the_first_occurrence() -> None:
    """Callers pass newest first, so a re-run supersedes the original."""
    fund_spec = spec(("graham", 1.0))

    def window(move: float) -> list[attribution.Call]:
        only = cycle("2026-01-05", {"AAPL": 100.0}, [signal("graham", "AAPL", 1.0)],
                     {"AAPL": 1.0}, fund_spec)
        return attribution.calls_from_result(result_of(
            session("2026-01-05", {}, only),
            session("2026-01-12", {}, None, assessment("2026-01-12", {"AAPL": move}, fund_spec)),
        ))

    merged = attribution.deduplicate(window(130.0) + window(110.0))

    assert len(merged) == 1
    assert merged[0].forward_return == pytest.approx(0.3)


# ---- the record only reaches back so far -----------------------------------

def window_calls(dates: list[str]) -> list[attribution.Call]:
    """One graham call on AAPL at each of *dates*, priced but unscored."""
    fund_spec = spec(("graham", 1.0))
    calls = []
    for date in dates:
        only = cycle(date, {"AAPL": 100.0}, [signal("graham", "AAPL", 1.0)],
                     {"AAPL": 1.0}, fund_spec)
        calls.extend(attribution.calls_from_result(result_of(session(date, {}, only))))
    return calls


def test_a_call_older_than_the_window_is_dropped() -> None:
    today = _date(2026, 7, 1)
    stale = (today - timedelta(weeks=attribution.WINDOW_WEEKS, days=1)).isoformat()

    assert attribution.within_window(window_calls([stale]), today=today) == []


def test_a_call_on_the_cutoff_itself_is_kept() -> None:
    """The boundary is inclusive, so a call exactly at the edge still counts."""
    today = _date(2026, 7, 1)
    edge = (today - timedelta(weeks=attribution.WINDOW_WEEKS)).isoformat()

    assert len(attribution.within_window(window_calls([edge]), today=today)) == 1


def test_the_window_keeps_recent_calls_and_drops_old_ones_together() -> None:
    """A long result spanning the cutoff is trimmed, not accepted or rejected whole."""
    today = _date(2026, 7, 1)
    kept = (today - timedelta(weeks=2)).isoformat()
    dropped = (today - timedelta(weeks=60)).isoformat()

    recent = attribution.within_window(window_calls([dropped, kept]), today=today)

    assert [call.date for call in recent] == [kept]


def test_an_aged_out_record_scores_as_nothing_rather_than_as_a_winner() -> None:
    """An analyst whose every call fell out of the window must not keep a rank.

    A stale leaderboard entry is worse than an empty one: it recommends on
    evidence the page is no longer willing to show.
    """
    today = _date(2026, 7, 1)
    old = [(today - timedelta(weeks=60 + n)).isoformat() for n in range(20)]

    recent = attribution.within_window(window_calls(old), today=today)

    assert attribution.score(recent) == []
    assert attribution.best(attribution.score(recent)) is None


# ---- the recommendation refuses to overclaim -------------------------------

def test_no_winner_is_named_on_a_thin_record() -> None:
    """One lucky call must not be presented as advice worth following."""
    fund_spec = spec(("graham", 1.0))
    first = cycle("2026-01-05", {"AAPL": 100.0}, [signal("graham", "AAPL", 1.0)],
                  {"AAPL": 1.0}, fund_spec)
    second = cycle("2026-01-12", {"AAPL": 180.0}, [], {}, fund_spec)

    calls = attribution.calls_from_result(result_of(
        session("2026-01-05", {"AAPL": 100.0}, first),
        session("2026-01-12", {"AAPL": 180.0}, second),
    ))
    scores = attribution.score(calls)

    assert scores[0].avg_edge > 0          # spectacular on paper
    assert scores[0].rankable is False
    assert attribution.best(scores) is None


def test_a_long_losing_record_is_never_recommended() -> None:
    fund_spec = spec(("graham", 1.0))
    sessions, previous = [], None
    for i in range(attribution.MIN_CALLS_TO_RANK + 1):
        date = f"2026-02-{i + 1:02d}"
        mark = 100.0 - i          # falls every week; graham stays bullish
        previous = cycle(date, {"AAPL": mark}, [signal("graham", "AAPL", 1.0)],
                         {"AAPL": 1.0}, fund_spec)
        sessions.append(session(date, {"AAPL": mark}, previous))

    scores = attribution.score(attribution.calls_from_result(result_of(*sessions)))

    assert scores[0].rankable is True
    assert scores[0].avg_edge < 0
    assert attribution.best(scores) is None


def test_a_sustained_winner_is_recommended() -> None:
    fund_spec = spec(("graham", 1.0))
    sessions = []
    for i in range(attribution.MIN_CALLS_TO_RANK + 1):
        date = f"2026-02-{i + 1:02d}"
        mark = 100.0 + i
        executed = cycle(date, {"AAPL": mark}, [signal("graham", "AAPL", 1.0)],
                         {"AAPL": 1.0}, fund_spec)
        sessions.append(session(date, {"AAPL": mark}, executed))

    scores = attribution.score(attribution.calls_from_result(result_of(*sessions)))
    winner = attribution.best(scores)

    assert winner is not None
    assert winner.analyst == "graham"
    assert winner.hit_rate == 1.0


# ---- prices and trades -----------------------------------------------------

def test_a_call_carries_the_prices_it_was_judged_on() -> None:
    """The move is a ratio; without both ends it cannot be checked by hand."""
    fund_spec = spec(("graham", 1.0))
    only = cycle("2026-01-05", {"AAPL": 100.0}, [signal("graham", "AAPL", 1.0)],
                 {"AAPL": 1.0}, fund_spec)

    call = attribution.calls_from_result(result_of(
        session("2026-01-05", {}, only),
        session("2026-01-12", {}, None, assessment("2026-01-12", {"AAPL": 125.0}, fund_spec)),
    ))[0]

    assert call.price == pytest.approx(100.0)
    assert call.exit_price == pytest.approx(125.0)
    assert call.forward_return == pytest.approx(0.25)


def test_the_funds_fill_is_attached_to_every_view_behind_it() -> None:
    """One order, several analysts. Each of their rows shows the same fill.

    The fill belongs to the pod, so showing it per analyst is only honest
    while it is labelled as the consequence of the view rather than the
    analyst's own trade — which is what the page says.
    """
    fund_spec = spec(("graham", 1.0), ("buffett", 1.0))
    traded = cycle("2026-01-05", {"AAPL": 100.0},
                   [signal("graham", "AAPL", 0.8), signal("buffett", "AAPL", 0.4)],
                   {"AAPL": 1.0}, fund_spec)
    traded.fills = [Fill(ticker="AAPL", side="buy", quantity=250, price=101.5)]

    calls = attribution.calls_from_result(result_of(
        session("2026-01-05", {}, traded),
        session("2026-01-12", {}, None, assessment("2026-01-12", {"AAPL": 110.0}, fund_spec)),
    ))

    assert {c.analyst for c in calls} == {"graham", "buffett"}
    for call in calls:
        assert call.fill is not None
        assert call.fill.quantity == 250
        assert call.fill.price == pytest.approx(101.5)


def test_a_name_with_no_order_has_no_fill() -> None:
    """A dash must mean "no trade", never "no data"."""
    fund_spec = spec(("graham", 1.0))
    untraded = cycle("2026-01-05", {"AAPL": 100.0, "MSFT": 50.0},
                     [signal("graham", "AAPL", 0.8), signal("graham", "MSFT", 0.2)],
                     {"AAPL": 1.0}, fund_spec)
    untraded.fills = [Fill(ticker="AAPL", side="buy", quantity=10, price=100.0)]

    calls = attribution.calls_from_result(result_of(
        session("2026-01-05", {}, untraded),
        session("2026-01-12", {}, None,
                assessment("2026-01-12", {"AAPL": 110.0, "MSFT": 55.0}, fund_spec)),
    ))
    by_ticker = {c.ticker: c for c in calls}

    assert by_ticker["AAPL"].fill is not None
    assert by_ticker["MSFT"].fill is None
    assert by_ticker["MSFT"].price == pytest.approx(50.0)   # priced, just not traded


# ---- the gate measures edge against its own noise --------------------------

def weekly_dates(n: int) -> list[str]:
    start = _date(2026, 1, 5)
    return [(start + timedelta(days=7 * i)).isoformat() for i in range(n)]


def run_of(marks: list[float], fund_spec: FundSpec, *,
           tickers: tuple[str, ...] = ("AAPL",)) -> FundBacktestResult:
    """A weekly backtest where every name follows the same price path.

    The last session only assesses, never executes, which is what prices the
    final call.
    """
    sessions = []
    for i, (date, mark) in enumerate(zip(weekly_dates(len(marks)), marks)):
        prices = {t: mark for t in tickers}
        if i == len(marks) - 1:
            sessions.append(session(date, {}, None, assessment(date, prices, fund_spec)))
            continue
        sessions.append(session(date, {}, cycle(
            date, prices, [signal("graham", t, 1.0) for t in tickers],
            {t: 1.0 / len(tickers) for t in tickers}, fund_spec,
        )))
    return result_of(*sessions)


def test_a_marginal_edge_swamped_by_noise_is_not_recommended() -> None:
    """The case that prompted this gate: a positive mean that is still noise.

    A record can be long, positive on average, and completely consistent
    with an analyst who guessed. Recommending it is the failure mode that
    matters, because the recommendation is what gets acted on.
    """
    fund_spec = spec(("graham", 1.0))
    marks = [100.0]
    for i in range(40):
        marks.append(marks[-1] * (1.10 if i % 2 == 0 else 0.92))

    scores = attribution.score(attribution.calls_from_result(run_of(marks, fund_spec)))

    assert scores[0].rankable is True       # plenty of calls over plenty of dates
    assert scores[0].avg_edge > 0           # and positive on average
    assert scores[0].t_stat < attribution.MIN_T_STAT
    assert attribution.best(scores) is None


def test_correlated_names_on_one_date_do_not_multiply_confidence() -> None:
    """Holding five names that move together is one observation, not five.

    Without clustering, a wider mandate would look more certain purely for
    being wider — the cheapest possible way to manufacture significance.
    """
    fund_spec = spec(("graham", 1.0))
    marks = [100.0]
    for i in range(30):
        marks.append(marks[-1] * (1.06 if i % 3 else 0.97))

    one = attribution.score(attribution.calls_from_result(
        run_of(marks, fund_spec, tickers=("AAPL",))))[0]
    five = attribution.score(attribution.calls_from_result(
        run_of(marks, fund_spec, tickers=("AAPL", "MSFT", "NVDA", "AMZN", "META"))))[0]

    assert five.scored == 5 * one.scored
    assert five.t_stat == pytest.approx(one.t_stat)


def test_many_names_on_a_few_dates_cannot_rank() -> None:
    """Ninety calls made on three days is three days of evidence."""
    fund_spec = spec(("graham", 1.0))
    wide = tuple(f"T{i:02d}" for i in range(30))

    scores = attribution.score(attribution.calls_from_result(
        run_of([100.0, 110.0, 121.0, 133.0], fund_spec, tickers=wide)))

    assert scores[0].scored == 90
    assert scores[0].rebalances == 3
    assert scores[0].rankable is False
    assert attribution.best(scores) is None


def test_ranking_puts_a_thin_record_below_a_proven_one() -> None:
    """A newcomer with one great call outranking a long record is the exact
    failure this page exists to avoid."""
    thin = attribution.AnalystScore(
        analyst="newcomer", scored=1, edge_total=0.9,
        edges_by_date={"2026-01-05": [0.9]},
    )
    proven = attribution.AnalystScore(
        analyst="veteran", scored=attribution.MIN_CALLS_TO_RANK, edge_total=0.2,
        edges_by_date={f"2026-01-{d + 1:02d}": [0.01] for d in range(attribution.MIN_CALLS_TO_RANK)},
    )

    ordered = sorted(
        [thin, proven],
        key=lambda s: (s.rankable, s.avg_edge if s.avg_edge is not None else float("-inf")),
        reverse=True,
    )

    assert ordered[0].analyst == "veteran"
