"""Deterministic portfolio targets from analyst opinions and permissions."""

from __future__ import annotations

from math import isfinite
from typing import Literal, Mapping, TYPE_CHECKING, TypeAlias

from pydantic import BaseModel, Field

from hedge_fund.models import Signal
from hedge_fund.signals.base import InvestmentApproach

if TYPE_CHECKING:
    from hedge_fund.fund.spec import PortfolioMode

FlatReason: TypeAlias = Literal["no_eligible_positions", "missing_long_side", "missing_short_side"]
WEIGHT_TOLERANCE = 1e-9


class BlendResult(BaseModel):
    """Raw opinions, permitted sizing scores, and strategy target weights."""

    convictions: dict[str, float]
    weights: dict[str, float]
    eligible_scores: dict[str, float] = Field(default_factory=dict)
    flat_reason: FlatReason | None = None


def blend_signals(
    signals: list[Signal],
    model_weights: Mapping[str, float],
    gross_target: float,
    *,
    mode: PortfolioMode,
    investment_approaches: Mapping[str, InvestmentApproach],
    max_name_weight: float | None = None,
) -> BlendResult:
    """Blend voting opinions, respecting each analyst's permission to short.

    Abstentions are excluded from both averages; neutral opinions still vote.
    All opinions contribute to ownership. Only short-capable analysts' signed
    opinions and long-only analysts' positive opinions contribute to the short
    assessment. Long-only bearishness can therefore reduce ownership without
    creating or strengthening short evidence.

    Eligible scores are normalized to gross_target. Dollar-neutral strategies
    allocate half to each side, or target zero when either side is missing.
    This relative sizing does not calibrate conviction into expected returns.

    With max_name_weight, no name exceeds that fraction of the sleeve: excess
    goes to the side's other names in proportion to their scores, and what
    the cap cannot place stays in cash. A dollar-neutral sleeve then shrinks
    its larger side to match the smaller, so it stays neutral.
    Invalid modes, profiles, weights, or signal values raise ValueError.
    """
    if mode not in ("long_only", "long_short", "dollar_neutral"):
        raise ValueError(f"unknown portfolio mode {mode!r}")
    if not isfinite(gross_target) or gross_target <= 0:
        raise ValueError("gross_target must be finite and positive")
    if max_name_weight is not None and (not isfinite(max_name_weight) or max_name_weight <= 0):
        raise ValueError("max_name_weight must be finite and positive")
    for name, weight in model_weights.items():
        if not isfinite(weight) or weight <= 0:
            raise ValueError(f"analyst {name!r}: blend weight must be finite and positive")
        if investment_approaches.get(name) not in ("long_only", "long_short"):
            raise ValueError(f"analyst {name!r}: missing or invalid investment approach")

    weighted_sum: dict[str, float] = {}
    short_sum: dict[str, float] = {}
    weight_total: dict[str, float] = {}
    for signal in signals:
        name, ticker, value = signal.model_name, signal.ticker, signal.value
        if name not in model_weights:
            raise ValueError(f"{ticker}: analyst {name!r} has no blend weight")
        if not isfinite(value) or not -1 <= value <= 1:
            raise ValueError(f"{ticker}: analyst {name!r} signal must be finite and within [-1, 1]")
        if signal.metadata.get("abstained") is True:
            continue
        weight = model_weights[name]
        short_value = value if investment_approaches[name] == "long_short" else max(value, 0.0)
        weighted_sum[ticker] = weighted_sum.get(ticker, 0.0) + weight * value
        short_sum[ticker] = short_sum.get(ticker, 0.0) + weight * short_value
        weight_total[ticker] = weight_total.get(ticker, 0.0) + weight
        if not all(isfinite(v) for v in (weighted_sum[ticker], short_sum[ticker], weight_total[ticker])):
            raise ValueError(f"{ticker}: blended signal totals must be finite")

    tickers = sorted({s.ticker for s in signals})
    convictions = {t: weighted_sum[t] / weight_total[t] if weight_total.get(t) else 0.0 for t in tickers}
    short_assessments = {t: short_sum[t] / weight_total[t] if weight_total.get(t) else 0.0 for t in tickers}
    scores = {t: max(convictions[t], 0.0) + (min(short_assessments[t], 0.0) if mode != "long_only" else 0.0) for t in tickers}
    weights = dict.fromkeys(tickers, 0.0)
    flat_reason: FlatReason | None = None
    if mode == "dollar_neutral":
        longs = sum(max(score, 0.0) for score in scores.values())
        shorts = sum(-min(score, 0.0) for score in scores.values())
        if longs < WEIGHT_TOLERANCE and shorts < WEIGHT_TOLERANCE:
            flat_reason = "no_eligible_positions"
        elif longs < WEIGHT_TOLERANCE:
            flat_reason = "missing_long_side"
        elif shorts < WEIGHT_TOLERANCE:
            flat_reason = "missing_short_side"
        elif max_name_weight is None:
            weights = {t: score / (longs if score > 0 else shorts) * (gross_target / 2) for t, score in scores.items()}
        else:
            long_side = _capped_allocation({t: s for t, s in scores.items() if s > 0}, gross_target / 2, max_name_weight)
            short_side = _capped_allocation({t: -s for t, s in scores.items() if s < 0}, gross_target / 2, max_name_weight)
            placed = min(sum(long_side.values()), sum(short_side.values()))
            long_scale = placed / sum(long_side.values())
            short_scale = placed / sum(short_side.values())
            weights = dict.fromkeys(tickers, 0.0)
            weights.update({t: w * long_scale for t, w in long_side.items()})
            weights.update({t: -w * short_scale for t, w in short_side.items()})
    else:
        gross = sum(abs(score) for score in scores.values())
        if gross < WEIGHT_TOLERANCE:
            flat_reason = "no_eligible_positions"
        elif max_name_weight is None:
            weights = {t: score / gross * gross_target for t, score in scores.items()}
        else:
            placed = _capped_allocation({t: abs(s) for t, s in scores.items() if s != 0}, gross_target, max_name_weight)
            weights = {t: placed.get(t, 0.0) * (1 if score > 0 else -1) for t, score in scores.items()}

    return BlendResult(convictions=convictions, eligible_scores=scores, weights=weights, flat_reason=flat_reason)


def _capped_allocation(scores: dict[str, float], budget: float, cap: float) -> dict[str, float]:
    """Split `budget` across positive `scores` proportionally, no name above `cap`.

    Water-filling: names whose share exceeds the cap are pinned at it and the
    rest re-split what remains. Budget the caps cannot place is left unallocated.
    """
    allocation: dict[str, float] = {}
    remaining = dict(scores)
    left = budget
    while remaining:
        total = sum(remaining.values())
        shares = {t: left * s / total for t, s in remaining.items()}
        over = [t for t, share in shares.items() if share > cap]
        if not over:
            allocation.update(shares)
            break
        for t in over:
            allocation[t] = cap
            left -= cap
            del remaining[t]
    return allocation
