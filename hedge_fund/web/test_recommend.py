"""Tests for the recommendations page.

The risk this page carries is overclaiming: it names trades, so every
plausible shortcut here ends with an unproven opinion printed as advice. The
cases below are the ones where that would happen quietly — an analyst with
no record, an analyst with a long but noisy one, a missing mark, and a
target too small for the fund to actually trade.
"""

from __future__ import annotations

from hedge_fund.fund.spec import BlendPolicy, FundSpec, ModelSpec, StrategySpec
from hedge_fund.models import Signal
from hedge_fund.pipeline.models import DecisionRecord, StrategyRecord
from hedge_fund.risk.limits import RiskLimits
from hedge_fund.web import recommend
from hedge_fund.web.attribution import AnalystScore


def spec() -> FundSpec:
    return FundSpec(
        schema_version=2,
        name="test-fund",
        strategies=[StrategySpec(
            name="pod",
            models=[ModelSpec(name="graham", weight=1.0)],
            blend=BlendPolicy(mode="long_only"),
        )],
        risk=RiskLimits(max_position_pct=0.25, max_gross_exposure=1.0),
    )


def signal(name: str, ticker: str, value: float, *, abstained: bool = False) -> Signal:
    return Signal(
        model_name=name, ticker=ticker, date="2026-06-01", value=value,
        reasoning=f"{name} likes {ticker}",
        metadata={"abstained": True} if abstained else {},
    )


def decision(marks: dict, weights: dict, signals: list[Signal]) -> DecisionRecord:
    return DecisionRecord(
        fund="test-fund", as_of="2026-06-01", spec=spec(), universe=sorted(marks),
        marks=marks, skipped=[], target_weights=weights, clamps=[],
        final_weights=weights,
        strategies=[StrategyRecord(
            name="pod", slice=1.0, signals=signals, convictions={}, weights=weights,
        )],
    )


def score(analyst: str, *, t_stat_clears: bool) -> AnalystScore:
    """A record sitting deliberately above or below the recommendation bar.

    Both have the same number of calls over the same number of rebalances, so
    the only thing separating them is how far the mean edge sits from zero
    relative to its own scatter — which is exactly what the bar measures.
    The edges vary per date on purpose: a constant series has zero variance
    and an infinite t-statistic, which would make every fixture here pass.
    """
    if t_stat_clears:
        edges = [0.020, 0.018][:] * 7          # mean well clear of its scatter
    else:
        edges = [0.010, -0.0098] * 7           # a coin flip with a faint tilt

    record = AnalystScore(analyst=analyst)
    record.calls = record.scored = len(edges) * 2
    record.edge_total = sum(edges) * 2
    record.edges_by_date = {
        f"2026-06-{day:02d}": [edge] for day, edge in enumerate(edges, start=1)
    }
    return record


def actions_for(dec: DecisionRecord, positions: dict, scores: dict) -> dict:
    found = recommend.actions_from_decision(
        dec, equity=100_000.0, positions=positions, scores=scores,
    )
    return {action.ticker: action for action in found}


# ---- direction mirrors what the fund can actually execute -------------------

def test_a_target_above_the_book_is_a_buy() -> None:
    actions = actions_for(
        decision({"AAPL": 100.0}, {"AAPL": 0.5}, [signal("graham", "AAPL", 1.0)]),
        positions={}, scores={},
    )

    assert actions["AAPL"].side == "buy"
    assert actions["AAPL"].shares == 500


def test_a_target_below_the_book_is_a_sell() -> None:
    actions = actions_for(
        decision({"AAPL": 100.0}, {"AAPL": 0.1}, [signal("graham", "AAPL", 1.0)]),
        positions={"AAPL": 400}, scores={},
    )

    assert actions["AAPL"].side == "sell"
    assert actions["AAPL"].shares == -300


def test_a_name_dropped_from_the_targets_is_sold_down() -> None:
    """A held name absent from final_weights has a target of zero, not no row."""
    actions = actions_for(
        decision({"AAPL": 100.0, "MSFT": 50.0}, {"AAPL": 0.5},
                 [signal("graham", "AAPL", 1.0)]),
        positions={"MSFT": 20}, scores={},
    )

    assert actions["MSFT"].side == "sell"
    assert actions["MSFT"].target == 0


def test_a_target_smaller_than_one_share_is_not_called_a_trade() -> None:
    """Execution truncates to whole shares, so a sub-share target is a hold.

    Calling it a buy would invent an action the fund cannot place.
    """
    actions = actions_for(
        decision({"BRK": 500_000.0}, {"BRK": 0.001}, [signal("graham", "BRK", 1.0)]),
        positions={}, scores={},
    )

    assert actions["BRK"].side == "hold"
    assert actions["BRK"].shares == 0


def test_an_unpriced_name_is_held_rather_than_liquidated() -> None:
    """A missing mark means the fund could not value the name.

    Reporting a sell would be a fabricated instruction to dump a position on
    the strength of absent data — the most expensive possible rounding.
    """
    actions = actions_for(
        decision({}, {"AAPL": 0.5}, [signal("graham", "AAPL", 1.0)]),
        positions={"AAPL": 300}, scores={},
    )

    assert actions["AAPL"].side == "hold"
    assert actions["AAPL"].target == 300
    assert actions["AAPL"].dollars is None


# ---- the page refuses to call an unproven view a recommendation ------------

def test_an_action_is_not_followable_without_a_proven_backer() -> None:
    actions = actions_for(
        decision({"AAPL": 100.0}, {"AAPL": 0.5}, [signal("graham", "AAPL", 1.0)]),
        positions={}, scores={"graham": score("graham", t_stat_clears=False)},
    )

    assert actions["AAPL"].backers[0].proven is False
    assert actions["AAPL"].followable is False


def test_an_analyst_with_no_record_at_all_is_not_proven() -> None:
    """A brand-new analyst is unmeasured, which is not the same as good."""
    actions = actions_for(
        decision({"AAPL": 100.0}, {"AAPL": 0.5}, [signal("graham", "AAPL", 1.0)]),
        positions={}, scores={},
    )

    assert actions["AAPL"].backers[0].score is None
    assert actions["AAPL"].followable is False


def test_a_long_record_that_is_statistically_flat_is_not_proven() -> None:
    """Volume of calls is not evidence; distance from zero is."""
    flat = score("graham", t_stat_clears=False)
    assert flat.rankable is True

    actions = actions_for(
        decision({"AAPL": 100.0}, {"AAPL": 0.5}, [signal("graham", "AAPL", 1.0)]),
        positions={}, scores={"graham": flat},
    )

    assert actions["AAPL"].followable is False


def test_a_proven_backer_makes_the_action_followable() -> None:
    actions = actions_for(
        decision({"AAPL": 100.0}, {"AAPL": 0.5}, [signal("graham", "AAPL", 1.0)]),
        positions={}, scores={"graham": score("graham", t_stat_clears=True)},
    )

    assert actions["AAPL"].followable is True


def test_one_proven_backer_is_enough_among_several() -> None:
    actions = actions_for(
        decision({"AAPL": 100.0}, {"AAPL": 0.5},
                 [signal("graham", "AAPL", 1.0), signal("pead", "AAPL", 0.4)]),
        positions={},
        scores={"graham": score("graham", t_stat_clears=False),
                "pead": score("pead", t_stat_clears=True)},
    )

    assert actions["AAPL"].followable is True
    # Proven views sort first so the evidence leads the row.
    assert actions["AAPL"].backers[0].analyst == "pead"


# ---- support is counted honestly -------------------------------------------

def test_an_abstention_is_not_listed_as_support() -> None:
    """An analyst with no data did not argue for the trade."""
    actions = actions_for(
        decision({"AAPL": 100.0}, {"AAPL": 0.5},
                 [signal("graham", "AAPL", 0.0, abstained=True)]),
        positions={}, scores={},
    )

    assert actions["AAPL"].backers == []


def test_a_zero_conviction_view_is_not_listed_as_support() -> None:
    actions = actions_for(
        decision({"AAPL": 100.0}, {"AAPL": 0.5}, [signal("graham", "AAPL", 0.0)]),
        positions={}, scores={},
    )

    assert actions["AAPL"].backers == []


def test_a_view_against_the_trade_is_still_shown() -> None:
    """The fund can buy a name an analyst argued against; hiding that would
    make the support look unanimous when it was not."""
    actions = actions_for(
        decision({"AAPL": 100.0}, {"AAPL": 0.5},
                 [signal("graham", "AAPL", 1.0), signal("pead", "AAPL", -0.8)]),
        positions={}, scores={},
    )

    assert {b.conviction for b in actions["AAPL"].backers} == {1.0, -0.8}


def test_the_largest_trade_is_listed_first() -> None:
    found = recommend.actions_from_decision(
        decision({"AAPL": 100.0, "MSFT": 100.0}, {"AAPL": 0.1, "MSFT": 0.6},
                 [signal("graham", "AAPL", 1.0), signal("graham", "MSFT", 1.0)]),
        equity=100_000.0, positions={}, scores={},
    )

    assert [a.ticker for a in found] == ["MSFT", "AAPL"]
