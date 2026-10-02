"""What the fund is about to do, and whether anyone has earned the right to say it.

A deployed fund's most recent `SessionRecord.decision` is an assessment made
at that close and executed at the next one. It is therefore the only forward
-looking thing in the system: every other record describes a trade that has
already happened. This module turns that pending decision into a readable
list of actions, and attaches to each one the measured record of the
analysts whose views produced it.

The gate is the point. `hedge_fund.web.attribution` refuses to crown an
analyst whose edge sits inside its own noise, and a recommendations page
that ignored that would undo it — it would take the same unproven views,
print them as advice, and look more confident for having fewer caveats. So a
row is marked followable only when at least one analyst behind it has a
record that clears the bar on its own. Everything else is still shown,
because the book is a fact and hiding it would be its own distortion, but it
is shown as position reporting rather than as a recommendation.

Direction is computed in whole shares rather than from a weight threshold.
`execute_decision` sizes with int(target_weight * equity / mark), so a
target that differs from the book by less than one share is not a trade the
fund can place, and calling it a buy would invent an action. The arithmetic
here mirrors that truncation deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hedge_fund.pipeline.models import DecisionRecord
from hedge_fund.web.attribution import EPSILON, MIN_T_STAT, AnalystScore


@dataclass(frozen=True)
class Backer:
    """One analyst's view behind a recommended action, with their record."""

    analyst: str
    strategy: str
    conviction: float
    reasoning: str | None
    # None when this analyst has no scored record inside the attribution
    # window — a new analyst, not a bad one.
    score: AnalystScore | None

    @property
    def proven(self) -> bool:
        """Whether this analyst's own record clears the recommendation bar."""
        if self.score is None or not self.score.rankable:
            return False
        t_stat = self.score.t_stat
        return t_stat is not None and t_stat >= MIN_T_STAT


@dataclass(frozen=True)
class Action:
    """One name the fund intends to move, and who argued for it."""

    fund: str
    as_of: str
    ticker: str
    side: str                 # "buy" | "sell" | "hold"
    shares: int               # signed change in share count; 0 for a hold
    price: float | None       # the mark the sizing was computed against
    held: int                 # signed shares before the trade
    target: int               # signed shares after it
    target_weight: float
    backers: list[Backer] = field(default_factory=list)

    @property
    def followable(self) -> bool:
        """Whether any analyst behind this action has earned being followed."""
        return any(backer.proven for backer in self.backers)

    @property
    def dollars(self) -> float | None:
        return None if self.price is None else abs(self.shares) * self.price


def actions_from_decision(
    decision: DecisionRecord,
    *,
    equity: float,
    positions: dict[str, int],
    scores: dict[str, AnalystScore],
) -> list[Action]:
    """Turn a pending decision into the trades it implies, largest first.

    *positions* is the book as it stands and *equity* the NAV behind it, both
    taken from the same session record as *decision* so the three agree.

    Sizing uses the marks recorded with the decision. The fund will execute
    at the next close against marks nobody has yet, so the share counts here
    are an estimate of its intent, not a prediction of the fill — the same
    one-session gap the attribution page accounts for.
    """
    backers = _backers_by_ticker(decision, scores)
    names = set(decision.final_weights) | set(positions)

    actions = []
    for ticker in sorted(names):
        price = decision.marks.get(ticker)
        weight = decision.final_weights.get(ticker, 0.0)
        held = positions.get(ticker, 0)
        target = _target_shares(weight, equity, price, fallback=held)
        delta = target - held
        actions.append(Action(
            fund=decision.fund,
            as_of=decision.as_of,
            ticker=ticker,
            side="buy" if delta > 0 else "sell" if delta < 0 else "hold",
            shares=delta,
            price=price,
            held=held,
            target=target,
            target_weight=weight,
            backers=backers.get(ticker, []),
        ))

    # Largest intended trade first; holds sort last and alphabetically among
    # themselves, since nothing distinguishes one from another.
    actions.sort(key=lambda a: (-abs(a.dollars or 0.0), a.ticker))
    return actions


def _target_shares(weight: float, equity: float, price: float | None, *, fallback: int) -> int:
    """Shares the fund would hold at *weight*, truncated as execution does.

    An unpriced name cannot be sized. Returning the current holding rather
    than zero is deliberate: a missing mark means the fund could not value
    the name, not that it decided to sell it, and reporting a liquidation it
    never intended would be the more dangerous error.
    """
    if price is None or price <= 0:
        return fallback
    return int(weight * equity / price)


def _backers_by_ticker(
    decision: DecisionRecord, scores: dict[str, AnalystScore]
) -> dict[str, list[Backer]]:
    """Every non-abstaining view in the decision, grouped by name.

    Abstentions are dropped rather than recorded as neutral: an analyst with
    no data did not argue for anything, and listing them under a trade would
    pad it with support that was never given.
    """
    grouped: dict[str, list[Backer]] = {}
    for strategy in decision.strategies:
        for signal in strategy.signals:
            if signal.metadata.get("abstained") is True:
                continue
            if abs(signal.value) < EPSILON:
                continue
            grouped.setdefault(signal.ticker, []).append(Backer(
                analyst=signal.model_name,
                strategy=strategy.name,
                conviction=signal.value,
                reasoning=signal.reasoning,
                score=scores.get(signal.model_name),
            ))

    for views in grouped.values():
        views.sort(key=lambda b: (not b.proven, -abs(b.conviction), b.analyst))
    return grouped
