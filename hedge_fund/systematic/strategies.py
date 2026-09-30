"""Strategy laboratory: interpretable reference strategies over an as-of view.

A strategy reads an `AsOfView` and returns `Signal`s — a score in [-1, +1]
per instrument, with the numbers behind it in `components`. It never sizes,
never sees a broker, never reads a clock: the view is all it knows, so it
cannot see the future. Same view + same config => same signals.

Configs are pydantic models: serializable, validated, hashed. The hash and
the strategy version travel in every signal's metadata, so a trade can be
traced to the exact rule that produced it.

These four are research baselines, not claims of edge:

    tsmom      time-series momentum: sign/strength of own trailing return
    xsmom      cross-sectional momentum: rank of trailing return in the universe
    meanrev    short-horizon mean reversion: fade a z-scored deviation
    breakout   Donchian channel breakout confirmed by volatility expansion
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from hedge_fund.models import Signal


class StrategyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    exclude: tuple[str, ...] = Field(default=(), description="symbols never scored (e.g. the benchmark)")
    use_membership: bool = Field(True, description="score only point-in-time universe members when known")


class SystematicStrategy(ABC):
    name: ClassVar[str]
    version: ClassVar[str]
    Config: ClassVar[type[StrategyConfig]]

    def __init__(self, config: StrategyConfig | None = None, **overrides) -> None:
        if config is not None and overrides:
            raise ValueError("pass a config or keyword overrides, not both")
        self.config = config if config is not None else self.Config(**overrides)
        if not isinstance(self.config, self.Config):
            raise TypeError(f"{self.name} needs a {self.Config.__name__}")

    # -- identity ------------------------------------------------------

    def spec(self) -> dict:
        return {"strategy": self.name, "version": self.version, "config": self.config.model_dump(mode="json")}

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.spec(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def from_spec(cls, spec: dict) -> SystematicStrategy:
        if spec.get("strategy") != cls.name:
            raise ValueError(f"spec is for {spec.get('strategy')!r}, not {cls.name!r}")
        return cls(cls.Config(**spec.get("config", {})))

    # -- universe and helpers -------------------------------------------

    def universe(self, view) -> list[str]:
        names = view.tickers
        if self.config.use_membership:
            try:
                names = view.members()
            except LookupError:
                pass
        tradable = view.tradable(lookback=1)
        return [t for t in names if t not in self.config.exclude and t in tradable.columns
                and bool(tradable[t].iloc[-1])]

    def _signal(self, view, ticker: str, value: float, components: dict, *, abstain: str | None = None) -> Signal:
        value = 0.0 if abstain else float(np.clip(value, -1.0, 1.0))
        meta = {"strategy_version": self.version, "config_hash": self.config_hash(),
                "data_as_of": view.session, "abstained": abstain is not None}
        if abstain:
            meta["abstain_reason"] = abstain
        clean = {k: float(v) for k, v in components.items() if v is not None and math.isfinite(float(v))}
        return Signal(model_name=self.name, ticker=ticker, date=view.session, value=value,
                      components=clean, metadata=meta)

    def generate(self, view) -> list[Signal]:
        names = self.universe(view)
        if not names:
            return []
        return self._generate(view, names)

    @abstractmethod
    def _generate(self, view, names: list[str]) -> list[Signal]: ...


def _closes(view, names: list[str], lookback: int) -> pd.DataFrame:
    return view.bars("close", tickers=names, lookback=lookback)


# ---------------------------------------------------------------------------


class TSMomentumConfig(StrategyConfig):
    lookback: int = Field(252, ge=20)
    skip: int = Field(21, ge=0)
    vol_lookback: int = Field(60, ge=10)
    scale: float = Field(1.0, gt=0, description="t-stat per unit of full conviction")


class TimeSeriesMomentum(SystematicStrategy):
    """Score = tanh(trailing log return / (daily vol * sqrt(horizon)) / scale)."""

    name, version, Config = "tsmom", "1.0.0", TSMomentumConfig

    def _generate(self, view, names):
        c = self.config
        px = _closes(view, names, c.lookback + 1)
        out = []
        for t in names:
            s = px[t].dropna()
            if len(s) < c.lookback + 1 or len(s) < c.vol_lookback + 1:
                out.append(self._signal(view, t, 0.0, {}, abstain="insufficient history"))
                continue
            end = s.iloc[-1 - c.skip]
            ret = math.log(end / s.iloc[0])
            vol = float(np.log(s).diff().dropna().tail(c.vol_lookback).std(ddof=1))
            horizon = c.lookback - c.skip
            tstat = ret / (vol * math.sqrt(horizon)) if vol > 0 else 0.0
            out.append(self._signal(view, t, math.tanh(tstat / c.scale),
                                    {"trailing_log_return": ret, "daily_vol": vol, "tstat": tstat}))
        return out


class XSMomentumConfig(StrategyConfig):
    lookback: int = Field(252, ge=20)
    skip: int = Field(21, ge=0)
    min_names: int = Field(3, ge=2)


class CrossSectionalMomentum(SystematicStrategy):
    """Score = 2 * percentile rank of trailing return - 1 across scored names."""

    name, version, Config = "xsmom", "1.0.0", XSMomentumConfig

    def _generate(self, view, names):
        c = self.config
        px = _closes(view, names, c.lookback + 1)
        rets = {}
        for t in names:
            s = px[t].dropna()
            if len(s) >= c.lookback + 1:
                rets[t] = math.log(s.iloc[-1 - c.skip] / s.iloc[0])
        out = []
        if len(rets) < c.min_names:
            return [self._signal(view, t, 0.0, {}, abstain="too few names with history") for t in names]
        ranked = pd.Series(rets).rank(method="average")
        n = len(ranked)
        for t in names:
            if t not in rets:
                out.append(self._signal(view, t, 0.0, {}, abstain="insufficient history"))
                continue
            pct = (ranked[t] - 1) / (n - 1)
            out.append(self._signal(view, t, 2 * pct - 1, {"trailing_log_return": rets[t], "rank_pct": pct}))
        return out


class MeanReversionConfig(StrategyConfig):
    window: int = Field(20, ge=5)
    entry_z: float = Field(1.0, gt=0, description="|z| at which conviction reaches tanh(1)")


class MeanReversion(SystematicStrategy):
    """Score = -tanh(z / entry_z), z = (close - SMA) / rolling std of closes."""

    name, version, Config = "meanrev", "1.0.0", MeanReversionConfig

    def _generate(self, view, names):
        c = self.config
        px = _closes(view, names, c.window)
        out = []
        for t in names:
            s = px[t].dropna()
            if len(s) < c.window:
                out.append(self._signal(view, t, 0.0, {}, abstain="insufficient history"))
                continue
            mean, sd = float(s.mean()), float(s.std(ddof=1))
            z = (float(s.iloc[-1]) - mean) / sd if sd > 0 else 0.0
            out.append(self._signal(view, t, -math.tanh(z / c.entry_z), {"zscore": z, "sma": mean}))
        return out


class BreakoutConfig(StrategyConfig):
    channel: int = Field(55, ge=5)
    atr_short: int = Field(20, ge=2)
    atr_long: int = Field(100, ge=5)
    min_expansion: float = Field(1.0, gt=0, description="ATR short/long ratio required to act")


class Breakout(SystematicStrategy):
    """+1 above the prior channel high, -1 below its low, only when volatility
    is expanding (ATR short / ATR long >= min_expansion); otherwise 0."""

    name, version, Config = "breakout", "1.0.0", BreakoutConfig

    def _generate(self, view, names):
        c = self.config
        n = max(c.channel, c.atr_long) + 1
        hi, lo, cl = (view.bars(f, tickers=names, lookback=n) for f in ("high", "low", "close"))
        out = []
        for t in names:
            frame = pd.DataFrame({"h": hi[t], "l": lo[t], "c": cl[t]}).dropna()
            if len(frame) < n:
                out.append(self._signal(view, t, 0.0, {}, abstain="insufficient history"))
                continue
            prev_c = frame["c"].shift(1)
            tr = pd.concat([frame["h"] - frame["l"], (frame["h"] - prev_c).abs(), (frame["l"] - prev_c).abs()],
                           axis=1).max(axis=1).dropna()
            ratio = float(tr.tail(c.atr_short).mean() / tr.tail(c.atr_long).mean()) if tr.tail(c.atr_long).mean() > 0 else 0.0
            prior = frame.iloc[-1 - c.channel:-1]
            last = float(frame["c"].iloc[-1])
            up, down = float(prior["h"].max()), float(prior["l"].min())
            value = 0.0
            if ratio >= c.min_expansion:
                value = 1.0 if last > up else -1.0 if last < down else 0.0
            out.append(self._signal(view, t, value, {"channel_high": up, "channel_low": down,
                                                     "atr_ratio": ratio, "close": last}))
        return out


REGISTRY: dict[str, type[SystematicStrategy]] = {
    cls.name: cls for cls in (TimeSeriesMomentum, CrossSectionalMomentum, MeanReversion, Breakout)
}


def strategy_from_spec(spec: dict) -> SystematicStrategy:
    try:
        return REGISTRY[spec["strategy"]].from_spec(spec)
    except KeyError as exc:
        raise ValueError(f"unknown strategy spec {spec!r}") from exc
