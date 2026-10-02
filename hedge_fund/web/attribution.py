"""Who said what, what it became, and whether it was right.

A backtest already records every analyst's view — `blend_signals` averages
them away into one conviction per ticker, but `StrategyRecord.signals` keeps
the originals. This module reads those back out and scores them.

Two things are measured here and they are not the same thing:

  * the *call* — conviction times the forward move of the name. This judges
    the analyst's opinion on its own terms, whether or not the fund acted.
  * the *attributed position* — the analyst's share of what the fund actually
    held in that name, in dollars. This judges the opinion's consequence.

An analyst can be right and own nothing (outvoted by their pod), or own a
lot and be wrong. Reporting one number would hide that, so both are kept.

The attribution share is exact, not a heuristic. Conviction is the
weight-mean of the pod's views, so analyst a's signed share of the numerator
is precisely the fraction of the pod's opinion that came from them.
Position *size* is a nonlinear function of conviction (clamped at zero,
normalized to a gross target, then risk-scaled), so dollars are apportioned
by that same share rather than recomputed — stated plainly because it is an
apportionment, not a replay of the sizing arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import copysign, fsum, inf, isfinite, sqrt

from hedge_fund.backtesting.fund import FundBacktestResult
from hedge_fund.brokers.models import Fill
from hedge_fund.pipeline.models import CycleRecord

# An analyst who abstains on every name still appears, with zero calls. That
# is the single most useful row on the page when the data feed is thin, so
# abstentions are counted rather than filtered away.
EPSILON = 1e-9

# Below this many scored calls a ranking is noise. The page still shows the
# numbers — it just refuses to crown anyone.
MIN_CALLS_TO_RANK = 20

# Calls made at the same rebalance are not independent evidence: the names
# move together with the market that day. Edges are therefore averaged
# within a date and the statistics run across dates, so a mandate holding
# thirty names does not look thirty times more certain than one holding one.
MIN_REBALANCES_TO_RANK = 12

# How many standard errors the mean edge must sit above zero.
#
# Two would be the textbook 95% bar. Three is deliberate, for two reasons
# this page cannot escape. It reports the *maximum* over every analyst, and
# the best of six candidates clears a nominal 95% bar by chance far more
# often than one in twenty. And backtested edges are measured on overlapping
# windows of correlated names, so even the clustered standard error flatters
# the result. Harvey, Liu and Zhu argue for exactly this threshold when a
# finding is selected from many candidates, which is what a leaderboard is.
#
# The cost of being wrong here is asymmetric: an unearned recommendation is
# acted on, a withheld one merely disappoints.
MIN_T_STAT = 3.0


@dataclass(frozen=True)
class Call:
    """One analyst's view on one ticker at one rebalance, and what came of it."""

    fund: str
    date: str
    analyst: str
    strategy: str
    ticker: str
    conviction: float
    reasoning: str | None
    abstained: bool
    # The close the view was formed against, and the one it is judged by.
    price: float | None
    exit_price: float | None
    # None at the final rebalance: nothing follows it to measure against.
    forward_return: float | None
    # Signed share of the pod's conviction that came from this analyst.
    share: float
    # That share applied to the fund's actual holding, in dollars.
    dollars: float
    # The fund's trade in this name at this rebalance, if it made one. It
    # belongs to the pod, not to this analyst — several views fed one order.
    fill: Fill | None = None

    @property
    def edge(self) -> float | None:
        """Conviction times the forward move: positive when the call paid."""
        if self.forward_return is None or self.abstained:
            return None
        return self.conviction * self.forward_return

    @property
    def correct(self) -> bool | None:
        """Did the name move the way the analyst leaned?"""
        if self.abstained or self.forward_return is None:
            return None
        if abs(self.conviction) < EPSILON or abs(self.forward_return) < EPSILON:
            return None
        return self.conviction * self.forward_return > 0


@dataclass
class AnalystScore:
    """An analyst's record across every call in scope."""

    analyst: str
    strategies: set[str] = field(default_factory=set)
    calls: int = 0
    abstentions: int = 0
    scored: int = 0           # calls with a measurable forward return
    decided: int = 0          # scored calls that took a side and moved
    hits: int = 0
    edge_total: float = 0.0
    pnl: float = 0.0
    gross_dollars: float = 0.0
    tickers: set[str] = field(default_factory=set)
    edges_by_date: dict[str, list[float]] = field(default_factory=dict)

    @property
    def participation(self) -> float:
        """Share of opportunities on which the analyst had a view at all."""
        offered = self.calls + self.abstentions
        return self.calls / offered if offered else 0.0

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.decided if self.decided else None

    @property
    def avg_edge(self) -> float | None:
        """Mean conviction-weighted forward return — the headline number."""
        return self.edge_total / self.scored if self.scored else None

    @property
    def session_edges(self) -> list[float]:
        """Mean edge per rebalance date — one independent-ish observation each."""
        return [fsum(edges) / len(edges) for edges in self.edges_by_date.values()]

    @property
    def rebalances(self) -> int:
        return len(self.edges_by_date)

    @property
    def t_stat(self) -> float | None:
        """Standard errors between the mean edge and zero, clustered by date.

        None when there is too little to measure. A record with no dispersion
        at all is not a failure of the test — a consistently positive edge is
        infinitely far from zero — so that case reports a signed infinity
        rather than discarding the strongest evidence there is.
        """
        observations = self.session_edges
        n = len(observations)
        if n < MIN_REBALANCES_TO_RANK:
            return None

        mean = fsum(observations) / n
        variance = fsum((x - mean) ** 2 for x in observations) / (n - 1)
        if variance <= 0:
            return None if abs(mean) < EPSILON else copysign(inf, mean)
        return mean / sqrt(variance / n)

    @property
    def rankable(self) -> bool:
        """Enough evidence to be placed on a leaderboard at all.

        This governs the "thin" label, not the recommendation — a record can
        be substantial enough to rank and still fall well short of being
        worth acting on.
        """
        return self.scored >= MIN_CALLS_TO_RANK and self.rebalances >= MIN_REBALANCES_TO_RANK


def calls_from_result(result: FundBacktestResult) -> list[Call]:
    """Every analyst view in *result*, priced against the following rebalance.

    Prices come from the *assessment* marks, not the cycle's own `marks`.
    That distinction is the difference between scoring every call and
    scoring almost none: a cycle only prices the names it held or targeted,
    so a rebalance that traded nothing carries no prices at all, while an
    assessment prices the whole universe it was asked to look at. Since an
    analyst can have a view on a name the fund never bought, the assessment
    series is the only one that can score the opinion.

    Both ends of a return therefore come from the same source, one rebalance
    apart. The last rebalance is priced against the trailing assessment the
    backtest makes at the final session and never executes. When no later
    price exists the call is kept unscored rather than dropped, because an
    unmeasurable opinion is still a thing the analyst said.
    """
    cycles = result.cycles
    if not cycles:
        return []

    weights_by_strategy = {
        strategy.name: strategy.model_weights for strategy in result.records[0].executed.spec.strategies
    } if result.records and result.records[0].executed else {}

    snapshots = _mark_snapshots(result)

    calls: list[Call] = []
    for cycle in cycles:
        now = _view_date(cycle)
        entry = _entry_marks(cycle, snapshots, now)
        later = _next_marks(snapshots, now)
        # A cycle carries its own audit copy of the spec, so a mandate edited
        # mid-flight cannot retro-reweight earlier attribution.
        spec_weights = {s.name: s.model_weights for s in cycle.spec.strategies} or weights_by_strategy
        # build_orders emits at most one order per name, so at most one fill.
        fills = {fill.ticker: fill for fill in cycle.fills}

        for strategy in cycle.strategies:
            model_weights = spec_weights.get(strategy.name, {})
            numerators = _numerators(strategy.signals, model_weights)

            for signal in strategy.signals:
                abstained = signal.metadata.get("abstained") is True
                weight = model_weights.get(signal.model_name, 1.0)
                total = numerators.get(signal.ticker, 0.0)
                share = 0.0
                if not abstained and abs(total) > EPSILON:
                    share = (weight * signal.value) / total

                equity_fraction = strategy.final_contribution.get(signal.ticker, 0.0)
                dollars = share * equity_fraction * cycle.nav
                price = entry.get(signal.ticker)
                exit_price = later.get(signal.ticker)

                calls.append(Call(
                    fund=result.fund,
                    date=cycle.as_of,
                    analyst=signal.model_name,
                    strategy=strategy.name,
                    ticker=signal.ticker,
                    conviction=signal.value,
                    reasoning=signal.reasoning,
                    abstained=abstained,
                    price=price,
                    exit_price=exit_price,
                    forward_return=_forward_return(price, exit_price),
                    share=share,
                    dollars=dollars,
                    fill=fills.get(signal.ticker),
                ))
    return calls


def deduplicate(calls: list[Call]) -> list[Call]:
    """Drop calls the same mandate already made on the same name and date.

    Backtest windows overlap all the time — rerun a mandate over a longer
    span and every rebalance the two windows share is replayed identically.
    Counting those twice does not add evidence, it just doubles the weight of
    whatever happened to be simulated more often, and an analyst's attributed
    P&L would grow with the number of times a user pressed Run.

    Identity is the mandate, the rebalance date, the pod, the analyst and the
    name: the same decision, however many files recorded it. Earlier callers
    win, so pass the newest results first and a re-run supersedes the
    original rather than piling on top of it.
    """
    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[Call] = []
    for call in calls:
        key = (call.fund, call.date, call.strategy, call.analyst, call.ticker)
        if key in seen:
            continue
        seen.add(key)
        unique.append(call)
    return unique


def score(calls: list[Call]) -> list[AnalystScore]:
    """Aggregate *calls* per analyst, best average edge first.

    Unrankable analysts sort last regardless of their numbers: a single lucky
    call must never outrank a long record.
    """
    scores: dict[str, AnalystScore] = {}
    for call in calls:
        entry = scores.setdefault(call.analyst, AnalystScore(analyst=call.analyst))
        entry.strategies.add(call.strategy)

        if call.abstained:
            entry.abstentions += 1
            continue

        entry.calls += 1
        entry.tickers.add(call.ticker)
        entry.gross_dollars += abs(call.dollars)

        if call.forward_return is not None:
            edge = call.conviction * call.forward_return
            entry.scored += 1
            entry.edge_total += edge
            entry.pnl += call.dollars * call.forward_return
            entry.edges_by_date.setdefault(call.date, []).append(edge)
        if call.correct is not None:
            entry.decided += 1
            if call.correct:
                entry.hits += 1

    return sorted(
        scores.values(),
        key=lambda s: (s.rankable, s.avg_edge if s.avg_edge is not None else float("-inf")),
        reverse=True,
    )


def best(scores: list[AnalystScore]) -> AnalystScore | None:
    """The analyst worth following, or None when the evidence is too thin.

    A positive average edge is not evidence of skill; it is the expected
    outcome for roughly half of any group of analysts who did nothing but
    guess. The bar is that the mean edge stands clear of its own noise —
    see MIN_T_STAT for why the threshold is as high as it is.

    Returning None is the common case and is the honest answer. A
    leaderboard topped by a coin flip with a name attached is worse than no
    leaderboard, because this one is read as advice.

    A positive mean is implied by clearing the threshold and is not
    re-checked.
    """
    ranked = [s for s in scores if s.rankable and s.t_stat is not None and s.t_stat >= MIN_T_STAT]
    return ranked[0] if ranked else None


def _numerators(signals, model_weights) -> dict[str, float]:
    """Per-ticker Σ(weight × conviction) over the analysts who had a view.

    This is the numerator of the blend's weighted mean, so dividing one
    analyst's term by it gives their exact share of the pod's opinion.
    """
    totals: dict[str, float] = {}
    for signal in signals:
        if signal.metadata.get("abstained") is True:
            continue
        weight = model_weights.get(signal.model_name, 1.0)
        totals[signal.ticker] = totals.get(signal.ticker, 0.0) + weight * signal.value
    return totals


def _mark_snapshots(result: FundBacktestResult) -> dict[str, dict[str, float]]:
    """Universe-wide closes by assessment date, oldest first.

    Every assessment in the backtest contributes one snapshot, including the
    trailing one the final session makes and never executes — that last
    snapshot is what gives the final rebalance something to be scored
    against.
    """
    snapshots: dict[str, dict[str, float]] = {}
    for record in result.records:
        if record.decision is not None and record.decision.marks:
            snapshots[record.decision.as_of] = record.decision.marks
    # Executions may re-assess on a cutoff date that is not itself a session,
    # so those views are priced on a day no SessionRecord covers.
    for cycle in result.cycles:
        for assessment in (cycle.refreshed_assessment, cycle.original_assessment):
            if assessment is not None and assessment.marks:
                snapshots.setdefault(assessment.as_of, assessment.marks)
    return dict(sorted(snapshots.items()))


def _view_date(cycle: CycleRecord) -> str:
    """The date the stored signals were actually formed.

    A cycle is stamped with the original assessment's date, but its signals
    come from the refreshed one when execution re-ran the analysts. Pricing
    against the wrong date would measure a move the analyst never saw.
    """
    refreshed = cycle.refreshed_assessment
    return refreshed.as_of if refreshed is not None else cycle.as_of


def _entry_marks(
    cycle: CycleRecord, snapshots: dict[str, dict[str, float]], now: str,
) -> dict[str, float]:
    return snapshots.get(now) or cycle.marks


def _next_marks(snapshots: dict[str, dict[str, float]], after: str) -> dict[str, float]:
    for date, marks in snapshots.items():
        if date > after:
            return marks
    return {}


def _forward_return(entry: float | None, exit_: float | None) -> float | None:
    if entry is None or exit_ is None or entry <= 0:
        return None
    move = exit_ / entry - 1.0
    return move if isfinite(move) else None
