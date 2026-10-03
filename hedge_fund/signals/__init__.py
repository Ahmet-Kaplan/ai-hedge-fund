"""Registered alpha models and their static investment approaches."""

from __future__ import annotations

from typing import cast

from hedge_fund.signals.ackman import AckmanAgent
from hedge_fund.signals.base import AlphaModel, InvestmentApproach, QuantModel
from hedge_fund.signals.buffett import BuffettAgent
from hedge_fund.signals.burry import BurryAgent
from hedge_fund.signals.damodaran import DamodaranAgent
from hedge_fund.signals.dpx_macro import DPXMacroStabilityModel
from hedge_fund.signals.druckenmiller import DruckenmillerAgent
from hedge_fund.signals.fisher import FisherAgent
from hedge_fund.signals.graham import GrahamAgent
from hedge_fund.signals.jhunjhunwala import JhunjhunwalaAgent
from hedge_fund.signals.llm_agent import LLMAgent
from hedge_fund.signals.lynch import LynchAgent
from hedge_fund.signals.mean_reversion import MeanReversionModel
from hedge_fund.signals.momentum import MomentumModel
from hedge_fund.signals.munger import MungerAgent
from hedge_fund.signals.news_analyst import NewsAnalystAgent
from hedge_fund.signals.news_sentiment import NewsSentimentModel
from hedge_fund.signals.pabrai import PabraiAgent
from hedge_fund.signals.pead import PEADModel
from hedge_fund.signals.ta_board import RATINGS_5_TIER, TradingAgentsBoard
from hedge_fund.signals.taleb import TalebAgent
from hedge_fund.signals.wood import WoodAgent

# Keys are last-name slugs (buffett, wood, damodaran) — short, stable ids
# for strategy YAML and Signal.model_name.
ALPHA_MODEL_REGISTRY: dict[str, type[AlphaModel]] = {
    # Quant models
    "pead": PEADModel,
    "news_sentiment": NewsSentimentModel,
    "momentum": MomentumModel,
    "mean_reversion": MeanReversionModel,
    # LLM investor agents
    "buffett": BuffettAgent,
    "munger": MungerAgent,
    "graham": GrahamAgent,
    "lynch": LynchAgent,
    "druckenmiller": DruckenmillerAgent,
    "news_analyst": NewsAnalystAgent,
    "wood": WoodAgent,
    "burry": BurryAgent,
    "ackman": AckmanAgent,
    "damodaran": DamodaranAgent,
    "fisher": FisherAgent,
    "pabrai": PabraiAgent,
    "taleb": TalebAgent,
    "jhunjhunwala": JhunjhunwalaAgent,
    # External research framework (optional dependency, no blind mode)
    "ta_board": TradingAgentsBoard,
    # Market-wide regime overlay (live-only; abstains on historical dates)
    "dpx_macro": DPXMacroStabilityModel,
}


def get_investment_approach(name: str) -> InvestmentApproach:
    """Read an analyst's declared approach without constructing a model or client.

    Raise ValueError for an unknown analyst or missing or invalid metadata.
    Profiles must be declared on the registered class, not inherited.
    """
    if name not in ALPHA_MODEL_REGISTRY:
        raise ValueError(f"unknown analyst {name!r}; available: {', '.join(sorted(ALPHA_MODEL_REGISTRY))}")
    approach = vars(ALPHA_MODEL_REGISTRY[name]).get("investment_approach")
    if approach not in ("long_only", "long_short"):
        raise ValueError(f"analyst {name!r} must declare investment_approach as long_only or long_short")
    return cast(InvestmentApproach, approach)


__all__ = [
    "AlphaModel",
    "InvestmentApproach",
    "QuantModel",
    "LLMAgent",
    "BuffettAgent",
    "MungerAgent",
    "GrahamAgent",
    "LynchAgent",
    "DruckenmillerAgent",
    "NewsAnalystAgent",
    "NewsSentimentModel",
    "PEADModel",
    "MomentumModel",
    "MeanReversionModel",
    "WoodAgent",
    "BurryAgent",
    "AckmanAgent",
    "DPXMacroStabilityModel",
    "DamodaranAgent",
    "FisherAgent",
    "PabraiAgent",
    "TalebAgent",
    "JhunjhunwalaAgent",
    "RATINGS_5_TIER",
    "TradingAgentsBoard",
    "ALPHA_MODEL_REGISTRY",
    "get_investment_approach",
]
