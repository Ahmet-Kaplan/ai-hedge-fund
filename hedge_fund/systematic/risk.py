"""Risk engine — the last word on every target weight.

Targets arrive from the ensemble/portfolio stage as weights of equity. The
engine only ever *reduces* risk: it clips, scales or zeroes; it never adds
exposure a strategy did not ask for. Every intervention is recorded.

Order of checks (each sees the previous stage's output):

    1. kill switch            -> flatten, halt
    2. drawdown breaker       -> flatten, halt (equity below peak by max_drawdown)
    3. daily loss limit       -> reduce-only for the rest of the session
    4. shorting permission    -> negative targets to zero when shorts are off
    5. max position size      -> clip |w|
    6. ADV participation      -> clip the change |dw| * equity to a share of ADV
    7. correlation clusters   -> cap the gross weight of highly correlated groups
    8. volatility target      -> scale the book down to the target annual vol
    9. gross / net exposure   -> scale down to the caps
   10. max turnover           -> move only part of the way from current weights

`RiskConfig` is frozen and hashed: nothing downstream — including any AI
component — can change a limit at runtime; a new limit is a new config,
reviewed by a human.

Fractional Kelly is available only as an *upper bound* on a strategy's
allocation, never full Kelly, and it switches itself off (returns None)
unless the return sample is long and significant enough.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class KellyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    fraction: float = Field(0.25, gt=0, le=0.5, description="fraction of full Kelly; full Kelly is never allowed")
    min_observations: int = Field(250, ge=30)
    min_tstat: float = Field(2.0, gt=0)
    cap: float = Field(1.0, gt=0, description="maximum allocation the Kelly bound may return")


class RiskConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_position: float = Field(0.10, gt=0, le=1.0)
    max_gross: float = Field(1.0, gt=0)
    max_net: float = Field(1.0)
    min_net: float = Field(0.0, description="most negative net exposure allowed (0 = long-only book)")
    allow_short: bool = False
    max_turnover: float | None = Field(0.5, gt=0, description="max sum |dw| per rebalance")
    max_adv_participation: float = Field(0.05, gt=0, le=1.0)
    adv_lookback: int = Field(20, ge=1)
    vol_target: float | None = Field(0.15, gt=0, description="annualized; None disables")
    vol_lookback: int = Field(60, ge=10)
    corr_threshold: float = Field(0.8, gt=0, le=1.0)
    max_cluster_weight: float = Field(0.30, gt=0, le=1.0)
    max_daily_loss: float = Field(0.03, gt=0, le=1.0)
    max_drawdown: float = Field(0.20, gt=0, le=1.0)
    kelly: KellyConfig = KellyConfig()

    @model_validator(mode="after")
    def _consistent(self) -> RiskConfig:
        if self.min_net > self.max_net:
            raise ValueError("min_net must not exceed max_net")
        if not self.allow_short and self.min_net < 0:
            raise ValueError("min_net < 0 needs allow_short")
        return self

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class RiskState:
    """Mutable run state the engine reads (never limits)."""

    peak_equity: float
    day_start_equity: float
    kill_switch: bool = False
    halted_reason: str | None = None

    def update(self, equity: float, *, new_session: bool) -> None:
        if new_session:
            self.day_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)


@dataclass
class RiskDecision:
    weights: dict[str, float]
    interventions: list[dict] = field(default_factory=list)
    halted: bool = False
    reason: str | None = None

    def note(self, check: str, **detail) -> None:
        self.interventions.append({"check": check, **detail})


class RiskEngine:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------

    def apply(self, targets: dict[str, float], *, equity: float, current: dict[str, float],
              state: RiskState, view=None) -> RiskDecision:
        c = self.config
        names = sorted(set(targets) | set(current))
        w = {t: float(targets.get(t, 0.0)) for t in names}
        cur = {t: float(current.get(t, 0.0)) for t in names}
        for t, v in w.items():
            if not math.isfinite(v):
                raise ValueError(f"{t}: target weight must be finite")
        d = RiskDecision(weights=w)

        if state.kill_switch:
            return self._flatten(d, names, "kill switch engaged")
        if state.peak_equity > 0 and equity / state.peak_equity - 1 <= -c.max_drawdown:
            state.halted_reason = f"drawdown {equity / state.peak_equity - 1:.2%} breached {c.max_drawdown:.0%}"
            return self._flatten(d, names, state.halted_reason)
        reduce_only = state.day_start_equity > 0 and equity / state.day_start_equity - 1 <= -c.max_daily_loss
        if reduce_only:
            d.note("daily_loss", loss=equity / state.day_start_equity - 1, action="reduce-only")
            for t in names:
                if abs(w[t]) > abs(cur[t]) or w[t] * cur[t] < 0:
                    w[t] = cur[t] if w[t] * cur[t] >= 0 else 0.0

        if not c.allow_short:
            for t in names:
                if w[t] < 0:
                    d.note("no_short", ticker=t, before=w[t], after=0.0)
                    w[t] = 0.0

        for t in names:
            if abs(w[t]) > c.max_position:
                after = math.copysign(c.max_position, w[t])
                d.note("max_position", ticker=t, before=w[t], after=after)
                w[t] = after

        if view is not None and equity > 0:
            self._adv(w, cur, equity, view, d)
            self._clusters(w, view, d)
            self._vol_target(w, view, d)

        gross = sum(abs(v) for v in w.values())
        if gross > c.max_gross:
            self._scale(w, c.max_gross / gross, d, "max_gross", before=gross)
        net = sum(w.values())
        if net > c.max_net and net > 0:
            self._scale(w, c.max_net / net, d, "max_net", before=net)
        elif net < c.min_net and net < 0:
            self._scale(w, c.min_net / net, d, "min_net", before=net)

        if c.max_turnover is not None:
            turn = sum(abs(w[t] - cur[t]) for t in names)
            if turn > c.max_turnover:
                lam = c.max_turnover / turn
                for t in names:
                    w[t] = cur[t] + lam * (w[t] - cur[t])
                d.note("max_turnover", before=turn, after=c.max_turnover)
        d.weights = {t: v for t, v in w.items() if abs(v) > 1e-12}
        return d

    # ------------------------------------------------------------------

    def _flatten(self, d: RiskDecision, names, reason: str) -> RiskDecision:
        d.weights, d.halted, d.reason = {}, True, reason
        d.note("halt", reason=reason)
        return d

    @staticmethod
    def _scale(w: dict, k: float, d: RiskDecision, check: str, **detail) -> None:
        for t in w:
            w[t] *= k
        d.note(check, scale=k, **detail)

    def _returns(self, view, names, lookback) -> pd.DataFrame:
        px = view.bars("close", tickers=names, lookback=lookback + 1)
        return np.log(px).diff().iloc[1:]

    def _adv(self, w, cur, equity, view, d) -> None:
        names = [t for t in w if abs(w[t] - cur[t]) > 1e-12]
        if not names:
            return
        close = view.bars("close", tickers=names, lookback=self.config.adv_lookback)
        volume = view.bars("volume", tickers=names, lookback=self.config.adv_lookback)
        adv = (close * volume).mean()
        for t in names:
            limit = self.config.max_adv_participation * float(adv.get(t, 0.0) or 0.0) / equity
            limit = limit if math.isfinite(limit) else 0.0
            delta = w[t] - cur[t]
            if abs(delta) > limit:
                after = cur[t] + math.copysign(limit, delta)
                d.note("adv_participation", ticker=t, before=w[t], after=after)
                w[t] = after

    def _clusters(self, w, view, d) -> None:
        held = [t for t in w if abs(w[t]) > 1e-12]
        if len(held) < 2:
            return
        corr = self._returns(view, held, self.config.vol_lookback).corr().fillna(0.0)
        seen: set[str] = set()
        for t in held:
            if t in seen:
                continue
            group = [u for u in held if u not in seen and (u == t or corr.loc[t, u] >= self.config.corr_threshold)]
            seen.update(group)
            gross = sum(abs(w[u]) for u in group)
            if len(group) > 1 and gross > self.config.max_cluster_weight:
                k = self.config.max_cluster_weight / gross
                for u in group:
                    w[u] *= k
                d.note("correlation_cluster", members=group, before=gross, after=self.config.max_cluster_weight)

    def _vol_target(self, w, view, d) -> None:
        if self.config.vol_target is None:
            return
        held = [t for t in w if abs(w[t]) > 1e-12]
        if not held:
            return
        rets = self._returns(view, held, self.config.vol_lookback).dropna(how="all").fillna(0.0)
        if len(rets) < 10:
            return
        vec = np.array([w[t] for t in held])
        vol = float(math.sqrt(max(vec @ rets.cov().to_numpy() @ vec, 0.0) * 252))
        if vol > self.config.vol_target:
            self._scale(w, self.config.vol_target / vol, d, "vol_target", before=vol)


def kelly_fraction(returns, config: KellyConfig, periods_per_year: int = 252) -> float | None:
    """Fractional-Kelly allocation bound, or None when evidence is insufficient.

    f* = mu / sigma^2 per period (continuous approximation). Returns
    config.fraction * f* clipped to [0, cap]. Disabled unless enabled, the
    sample has at least `min_observations` returns, and the mean's t-stat
    reaches `min_tstat`.
    """
    if not config.enabled:
        return None
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < config.min_observations:
        return None
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    if sd <= 0:
        return None
    tstat = mu / (sd / math.sqrt(len(r)))
    if tstat < config.min_tstat:
        return None
    full = mu / sd ** 2
    return float(min(max(config.fraction * full, 0.0), config.cap))
