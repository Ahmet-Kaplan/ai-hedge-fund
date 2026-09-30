"""Small-account mode (e.g. EUR 200): costs first, then maybe a trade.

With a tiny account, fixed costs dominate: a EUR 1 commission on a EUR 20
order is 500 bp before spread and slippage. This module makes that explicit:

    trade_economics      round-trip cost of an order in bp vs its expected edge;
                         an order whose edge does not clear costs + margin is not placed
    EconomicsFilter      drops such orders inside the DecisionEngine (recorded)
    probability_of_ruin  Monte Carlo chance of losing `ruin_level` of the account
    required_growth      what a target implies (e.g. EUR 200 -> 5,000 in a month is
                         +2,400% — not a plan, a warning)

Configuration lives in configs/small_account.yaml. Nothing here predicts or
promises returns.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from hedge_fund.core.instruments import InstrumentRegistry, equity
from hedge_fund.core.orders import OrderRequest
from hedge_fund.systematic.execution import CostModel

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "small_account.yaml"


class SmallAccountConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    capital: float = Field(200.0, gt=0)
    currency: str = "EUR"
    fractional: bool = True
    min_order_notional: float = Field(1.0, ge=0)
    commission_per_order: float = Field(1.0, ge=0)
    commission_bps: float = Field(0.0, ge=0)
    half_spread_bps: float = Field(5.0, ge=0)
    slippage_bps: float = Field(5.0, ge=0)
    max_positions: int = Field(3, ge=1)
    min_edge_margin_bps: float = Field(20.0, ge=0, description="edge must exceed round-trip cost by this")
    max_annual_turnover: float = Field(4.0, gt=0)
    ruin_level: float = Field(0.5, gt=0, lt=1)

    def cost_model(self) -> CostModel:
        return CostModel(commission_bps=self.commission_bps, commission_min=self.commission_per_order,
                         half_spread_bps=self.half_spread_bps, impact_coef=0.0, max_participation=0.01)

    def instruments(self) -> InstrumentRegistry:
        return InstrumentRegistry(default_fractional=self.fractional)


def load_small_account(path: Path | str = DEFAULT_PATH) -> SmallAccountConfig:
    return SmallAccountConfig(**(yaml.safe_load(Path(path).read_text()) or {}))


def round_trip_cost_bps(notional: float, cfg: SmallAccountConfig) -> float:
    if notional <= 0:
        return math.inf
    one_way = (max(cfg.commission_per_order, cfg.commission_bps / 1e4 * notional) / notional * 1e4
               + cfg.half_spread_bps + cfg.slippage_bps)
    return 2 * one_way


def trade_economics(notional: float, expected_edge_bps: float, cfg: SmallAccountConfig) -> dict:
    cost = round_trip_cost_bps(notional, cfg)
    ok = notional >= cfg.min_order_notional and expected_edge_bps >= cost + cfg.min_edge_margin_bps
    if notional < cfg.min_order_notional:
        reason = f"notional {notional:.2f} below minimum {cfg.min_order_notional:.2f}"
    elif not ok:
        reason = f"edge {expected_edge_bps:.0f}bp < cost {cost:.0f}bp + margin {cfg.min_edge_margin_bps:.0f}bp"
    else:
        reason = "economic"
    return {"economic": ok, "round_trip_cost_bps": cost, "edge_bps": expected_edge_bps, "reason": reason}


class EconomicsFilter:
    """Order filter for DecisionEngine: drop orders whose edge cannot pay their costs."""

    def __init__(self, cfg: SmallAccountConfig, default_edge_bps: float = 0.0) -> None:
        self.cfg, self.default_edge_bps = cfg, default_edge_bps
        self.dropped: list[dict] = []

    def __call__(self, orders: list[OrderRequest]) -> list[OrderRequest]:
        kept = []
        for o in orders:
            ens = o.reason.get("ensemble") or {}
            edge = abs(float(ens.get("expected_edge_bps", self.default_edge_bps) or 0.0))
            exiting = o.reason.get("target_weight", 1.0) == 0.0
            verdict = trade_economics(o.quantity * o.reference_price, edge, self.cfg)
            if exiting or verdict["economic"]:        # exits are always allowed
                kept.append(o)
            else:
                self.dropped.append({"order": o.client_order_id, "symbol": o.symbol, **verdict})
        return kept


def probability_of_ruin(mean_return: float, std_return: float, n_periods: int, *, ruin_level: float = 0.5,
                        n_paths: int = 5000, seed: int = 0) -> float:
    """Share of simulated paths whose equity ever falls below (1 - ruin_level)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(mean_return, std_return, size=(n_paths, n_periods))
    paths = np.cumprod(1 + np.clip(r, -0.999, None), axis=1)
    return float(np.mean(paths.min(axis=1) <= 1 - ruin_level))


def required_growth(start: float, target: float, periods: int = 1) -> float:
    """Per-period return needed to turn start into target over `periods`."""
    return (target / start) ** (1 / periods) - 1


def small_account_universe(symbols: list[str], cfg: SmallAccountConfig) -> InstrumentRegistry:
    reg = cfg.instruments()
    for s in symbols:
        reg.add(equity(s, fractional=cfg.fractional, min_notional=cfg.min_order_notional))
    return reg
