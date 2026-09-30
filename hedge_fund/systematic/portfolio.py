"""Portfolio construction: ensemble scores -> target weights (before risk).

Inverse-volatility scaling of each score, normalized to `gross_target`.
Long-only books keep only positive scores. Deterministic; the risk engine
applies every limit afterwards.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class PortfolioConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    long_only: bool = True
    gross_target: float = Field(1.0, gt=0)
    vol_lookback: int = Field(60, ge=10)
    min_score: float = Field(0.05, ge=0, description="scores below this in magnitude are ignored")
    top_n: int | None = Field(None, ge=1)


def target_weights(scores: dict[str, float], view, config: PortfolioConfig) -> dict[str, float]:
    picks = {t: s for t, s in scores.items() if abs(s) >= config.min_score and (s > 0 or not config.long_only)}
    if config.top_n is not None:
        picks = dict(sorted(picks.items(), key=lambda kv: -abs(kv[1]))[: config.top_n])
    if not picks:
        return {}
    px = view.bars("close", tickers=sorted(picks), lookback=config.vol_lookback + 1)
    vol = np.log(px).diff().std(ddof=1)
    raw = {}
    for t, s in picks.items():
        v = float(vol.get(t, float("nan")))
        if not math.isfinite(v) or v <= 0:
            continue
        raw[t] = s / v
    gross = sum(abs(x) for x in raw.values())
    return {t: x / gross * config.gross_target for t, x in raw.items()} if gross > 0 else {}
