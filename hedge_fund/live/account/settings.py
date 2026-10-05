"""Settings for the real-money account (~/.hedge-fund/live.yaml).

Defaults are the safe state: not confirmed for live trading, no agent
share, no shorts. A review changes agent_share; the user changes the rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class LiveSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_live: bool = Field(default=False, description="must be true before any real order is sent")
    core_ticker: str = Field(default="SPY", description="the S&P 500 ETF the core holds")
    paper_fund: str = Field(default="paper-fund", description="the paper fund whose bets the satellite mirrors")
    rebalance: Literal["daily", "weekly", "monthly"] = "weekly"
    agent_share: float = Field(default=0.0, ge=0, le=0.5, description="fraction of the account the satellite gets")
    satellite_max_name_pct: float = Field(default=0.10, gt=0, le=1)
    satellite_min_position_usd: float = Field(default=5.0, ge=1.0, description="smallest AI pick; fewer picks in a small account")
    shorts_enabled: bool = Field(default=False, description="set true only after enabling margin at Alpaca")
    short_min_equity: float = Field(default=2000.0, ge=2000.0)
    satellite_halt_relative: float = Field(default=0.10, gt=0, lt=1)
    min_order_usd: float = Field(default=1.0, ge=1.0)
    min_trade_pct: float = Field(default=0.005, ge=0, lt=1)
    cash_buffer_pct: float = Field(default=0.01, ge=0, lt=0.5, description="cash left unspent for price moves and fees")
    stale_plan_days: int = Field(default=8, ge=1)
    crypto_share: float = Field(default=0.0, ge=0, le=0.5, description="fraction of the account in the crypto half")
    crypto_core: dict[str, float] = Field(
        default_factory=lambda: {"BTC/USD": 0.6, "ETH/USD": 0.3, "SOL/USD": 0.1},
        description="crypto half's buy-and-hold weights (Alpaca pairs)")

    @field_validator("crypto_core")
    @classmethod
    def _crypto_weights(cls, core: dict[str, float]) -> dict[str, float]:
        if any("/" not in t for t in core):
            raise ValueError("crypto_core keys must be Alpaca pairs like 'BTC/USD'")
        if any(w <= 0 for w in core.values()) or abs(sum(core.values()) - 1) > 1e-6:
            raise ValueError("crypto_core weights must be positive and sum to 1")
        return core


def load_settings(path: str | Path) -> LiveSettings:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{path} not found; run `aihf-live init` to create it")
    try:
        data = yaml.safe_load(path.read_text()) or {}
        return LiveSettings.model_validate(data)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def save_settings(settings: LiveSettings, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(settings.model_dump(), sort_keys=False))
