"""Strategy ensemble: weigh strategies by evidence, trade only when the edge pays.

Not a vote. Each strategy's weight is

    w_s = reliability_s * confidence_s * regime_fit_s / (1 + sum_j max(corr_sj, 0))

    reliability   recent *out-of-sample* Sharpe, floored at 0 (a strategy that
                  has not worked out of sample gets no weight)
    confidence    grows with the number of OOS observations, saturating at
                  `full_confidence_obs`
    regime_fit    1 inside the regimes the strategy is declared for, `off_regime`
                  outside them (0 = switched off)
    correlation   strategies that mostly repeat each other share weight

Per instrument, the ensemble score is the weighted mean of non-abstaining
signals, and the expected edge is the weighted mean of value * edge_bps.
If |expected edge| minus the estimated round-trip cost is below
`min_edge_after_cost_bps`, the decision is NO_TRADE — an explicit,
recorded outcome, not a missing value.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class StrategyEvidence(BaseModel):
    """What validation has established about a strategy (never set by an AI)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    oos_sharpe: float
    oos_observations: int = Field(ge=0)
    edge_bps: float = Field(description="expected return per unit signal per holding period, bps, net of nothing")
    regimes: frozenset[str] = Field(default_factory=frozenset, description="e.g. {'trend:up','trend:down'}; empty = all")


class EnsembleConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    round_trip_cost_bps: float = Field(10.0, ge=0)
    min_edge_after_cost_bps: float = Field(5.0, ge=0)
    full_confidence_obs: int = Field(500, ge=1)
    off_regime: float = Field(0.0, ge=0, le=1)
    max_strategy_share: float = Field(0.6, gt=0, le=1)


class InstrumentDecision(BaseModel):
    ticker: str
    decision: Literal["TRADE", "NO_TRADE"]
    score: float
    expected_edge_bps: float
    contributions: dict[str, float]
    reason: str


class EnsembleDecision(BaseModel):
    session: str
    strategy_weights: dict[str, float]
    instruments: dict[str, InstrumentDecision]

    def scores(self) -> dict[str, float]:
        return {t: d.score for t, d in self.instruments.items() if d.decision == "TRADE"}


class Ensemble:
    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()

    def strategy_weights(self, evidence: dict[str, StrategyEvidence], regime=None,
                         correlation: pd.DataFrame | None = None) -> dict[str, float]:
        c = self.config
        labels = regime.labels() if regime is not None else set()
        raw = {}
        for name, ev in evidence.items():
            reliability = max(ev.oos_sharpe, 0.0)
            confidence = min(ev.oos_observations / c.full_confidence_obs, 1.0)
            fit = 1.0 if not ev.regimes or not labels or ev.regimes & labels else c.off_regime
            overlap = 0.0
            if correlation is not None and name in correlation.index:
                overlap = sum(max(float(correlation.loc[name, o]), 0.0) for o in evidence
                              if o != name and o in correlation.columns)
            raw[name] = reliability * confidence * fit / (1.0 + overlap)
        total = sum(raw.values())
        if total <= 0:
            return {n: 0.0 for n in raw}
        w = {n: v / total for n, v in raw.items()}
        # Cap concentration in one strategy, redistributing to the others.
        for _ in range(len(w)):
            over = {n: v for n, v in w.items() if v > c.max_strategy_share + 1e-12}
            if not over:
                break
            excess = sum(v - c.max_strategy_share for v in over.values())
            for n in over:
                w[n] = c.max_strategy_share
            rest = {n: v for n, v in w.items() if n not in over and v > 0}
            if not rest:
                break
            k = sum(rest.values())
            for n, v in rest.items():
                w[n] = v + excess * v / k
        return w

    def combine(self, session: str, signals: dict[str, list], evidence: dict[str, StrategyEvidence],
                regime=None, correlation: pd.DataFrame | None = None) -> EnsembleDecision:
        c = self.config
        unknown = set(signals) - set(evidence)
        if unknown:
            raise ValueError(f"no validation evidence for strategies: {sorted(unknown)}")
        weights = self.strategy_weights({n: evidence[n] for n in signals}, regime, correlation)
        by_ticker: dict[str, dict[str, float]] = {}
        for name, sigs in signals.items():
            for s in sigs:
                if s.metadata.get("abstained") or weights.get(name, 0.0) <= 0:
                    continue
                by_ticker.setdefault(s.ticker, {})[name] = float(s.value)
        out = {}
        for t, votes in sorted(by_ticker.items()):
            wsum = sum(weights[n] for n in votes)
            score = sum(weights[n] * v for n, v in votes.items()) / wsum
            edge = sum(weights[n] * v * evidence[n].edge_bps for n, v in votes.items()) / wsum
            net = abs(edge) - c.round_trip_cost_bps
            trade = net >= c.min_edge_after_cost_bps and score != 0
            out[t] = InstrumentDecision(
                ticker=t, decision="TRADE" if trade else "NO_TRADE", score=score if trade else 0.0,
                expected_edge_bps=edge, contributions={n: weights[n] * v / wsum for n, v in votes.items()},
                reason=(f"edge {edge:.1f}bp - cost {c.round_trip_cost_bps:.1f}bp = {net:.1f}bp"
                        + ("" if trade else f" < {c.min_edge_after_cost_bps:.1f}bp minimum")))
        return EnsembleDecision(session=session, strategy_weights=weights, instruments=out)
