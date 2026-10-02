"""Tests for analyst attribution.

The arithmetic here decides which analyst a user is told to follow, so the
cases that matter are the ones where a plausible-looking shortcut gives the
wrong answer: an analyst outvoted by their pod, an abstention mistaken for a
neutral view, and a short record mistaken for a track record.
"""

from __future__ import annotations

import pytest

from hedge_fund.backtesting.fund import FundBacktestMetrics, FundBacktestResult
from hedge_fund.fund.spec import BlendPolicy, FundSpec, ModelSpec, StrategySpec
from hedge_fund.models import Signal
from hedge_fund.pipeline.models import CycleRecord, DecisionRecord, StrategyRecord
from hedge_fund.pipeline.session import SessionRecord
from hedge_fund.risk.limits import RiskLimits
from hedge_fund.web import attribution


def spec(*models: tuple[str, float]) -> FundSpec:
    return FundSpec(
        schema_version=2,
        name="test-fund",
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
        fund="test-fund", as_of=as_of, spec=fund_spec, universe=sorted(marks),
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
        fund="test-fund", as_of=as_of, spec=fund_spec, universe=sorted(marks),
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
    fund_spec = spec(("graham", 1.0))
    calls = []
    for name in ("fund-a", "fund-b"):
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


def test_ranking_puts_a_thin_record_below_a_proven_one() -> None:
    """A newcomer with one great call outranking a long record is the exact
    failure this page exists to avoid."""
    thin = attribution.AnalystScore(analyst="newcomer", scored=1, edge_total=0.9)
    proven = attribution.AnalystScore(
        analyst="veteran", scored=attribution.MIN_CALLS_TO_RANK, edge_total=0.2,
    )

    ordered = sorted(
        [thin, proven],
        key=lambda s: (s.rankable, s.avg_edge if s.avg_edge is not None else float("-inf")),
        reverse=True,
    )

    assert ordered[0].analyst == "veteran"
