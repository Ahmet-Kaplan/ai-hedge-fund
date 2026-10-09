"""FundSpec — a fund's mandate as data, and the Fund that lives it.

The hierarchy mirrors a real shop (see VISION.md):

    FUND      = an allocator (CIO) over STRATEGIES  (master risk on the netted book)
    STRATEGY  = a blend policy over MODELS          (a "pod")
    MODEL     = an alpha model -> Signal

Models come in two kinds, and the strategy's character follows from its
staff: a strategy of LLM investor AGENTS (Buffett, Munger, ...) is a
discretionary pod — its identity is who's on the desk; a strategy powered
by quant models (PEAD, ...) is a systematic pod — its identity is the edge
it harvests. Same spec shape, same engine slot; the kind is derived, never
declared.

Specs are data (a Loop-2 ground rule): a mandate is a serializable YAML/JSON
config. The wizard, a chat LLM, and the strategy generator all emit this same
format — nothing downstream ever needs to know who authored a fund.

A `Fund` is the living counterpart: the spec plus its models instantiated
once. Models are stateful (LLM prompt caches, PEAD earnings caches), so
they must be constructed per fund — never per cycle — for caches to survive
across cycles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError

from hedge_fund.brokers.models import Commission
from hedge_fund.fund.allocator import ALLOCATOR_NAMES, Allocator, get_allocator
from hedge_fund.risk.limits import RiskLimits
from hedge_fund.signals import ALPHA_MODEL_REGISTRY, LLMAgent, get_investment_approach
from hedge_fund.signals.base import AlphaModel

PortfolioMode: TypeAlias = Literal["long_only", "long_short", "dollar_neutral"]


class ModelSpec(BaseModel):
    """One signal model in a strategy — an LLM agent or a quant model."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="key into ALPHA_MODEL_REGISTRY, e.g. 'buffett'")
    weight: float = Field(default=1.0, gt=0, description="blend weight")
    params: dict[str, Any] = Field(default_factory=dict, description="constructor kwargs for the model")

    @field_validator("name")
    @classmethod
    def _declared_approach(cls, name: str) -> str:
        get_investment_approach(name)
        return name


class BlendPolicy(BaseModel):
    """How a strategy's model views combine into one sleeve."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["conviction_weighted"] = "conviction_weighted"
    gross_target: float = Field(default=1.0, gt=0, description="desired sum of |weights| when views exist")
    mode: PortfolioMode
    max_name_weight: float | None = Field(
        default=None,
        gt=0,
        description="cap on any one name as a fraction of the sleeve; excess goes to the "
        "side's other names in proportion to their scores, and what the cap cannot place "
        "stays in cash. None leaves sizing unconstrained.",
    )
    sizing: Literal["conviction", "inverse_vol"] = Field(
        default="conviction",
        description="conviction: weight proportional to the blended view (today's "
        "behavior). inverse_vol: proportional to view / 60-day volatility, so each "
        "position carries similar risk.",
    )


class StrategySpec(BaseModel):
    """A strategy ("pod"): signal models plus the policy that blends them.

    `weight` is the fund's static capital slice for this strategy, relative
    to its siblings (normalized by StaticAllocator — 2/2 means the same as
    1/1). Other allocators may ignore it. In a library file
    (hedge_fund/strategies/) it stays at the default; slices are a
    fund-assembly decision, not a property of the strategy itself.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str | None = Field(default=None, description="human-facing name, e.g. 'Deep Value'")
    weight: float = Field(default=1.0, gt=0)
    models: list[ModelSpec] = Field(min_length=1)
    blend: BlendPolicy

    @property
    def title(self) -> str:
        """Display name, falling back to a title-cased slug."""
        return self.display_name or self.name.replace("-", " ").title()

    @property
    def model_weights(self) -> dict[str, float]:
        """model_name -> blend weight, as portfolio construction consumes it."""
        return {m.name: m.weight for m in self.models}


class FundSpec(BaseModel):
    """A mandate with fund-level risk limits and no fixed ticker universe.

    Unknown fields are rejected. Risk limits apply to the combined portfolio;
    the ticker universe is supplied separately for each run.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2]
    name: str
    strategies: list[StrategySpec] = Field(min_length=1)
    allocator: str = Field(
        default="static",
        description="CIO policy that turns strategy context into capital "
        "slices. 'static' (default) normalizes StrategySpec.weight — today's "
        "behavior. 'equal_weight' is a selectable stub that splits capital "
        "evenly.",
    )
    risk: RiskLimits
    capital: float = Field(default=100_000.0, gt=0)
    min_trade_pct: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="skip trades smaller than this fraction of equity, as long "
        "as they would grow a position rather than shrink it. Zero trades "
        "everything. Costs are dominated by ticket count at this book's size, "
        "so this is the cheapest cost lever there is.",
    )
    commission: Commission = Field(
        default_factory=Commission,
        description="what a fill costs to execute: a per-ticket charge plus a "
        "per-share rate. Defaults to zero, so a mandate that does not state a "
        "schedule backtests exactly as it did before costs existed.",
    )
    rebalance: Literal["daily", "weekly", "monthly"] = Field(
        default="weekly",
        description="how often the fund re-runs its cycle — a mandate choice, "
        "not an engine constant: a fundamentals fund trades weekly, a "
        "news-driven fund daily. The backtester and the scheduler daemon obey "
        "it; run_cycle itself never sees it.",
    )
    benchmark: str = Field(
        default="SPY",
        description="what the fund measures itself against; also the source " "of the backtest's trading-day grid",
    )

    @field_validator("benchmark")
    @classmethod
    def _uppercase_benchmark(cls, ticker: str) -> str:
        return ticker.upper()

    @field_validator("allocator")
    @classmethod
    def _known_allocator(cls, name: str) -> str:
        if name not in ALLOCATOR_NAMES:
            raise ValueError(
                f"unknown allocator {name!r}; available: {sorted(ALLOCATOR_NAMES)}"
            )
        return name

    @field_validator("strategies")
    @classmethod
    def _unique_strategy_names(cls, strategies: list[StrategySpec]) -> list[StrategySpec]:
        names = [s.name for s in strategies]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate strategy names: {sorted(duplicates)}")
        return strategies


def normalize_universe(tickers: list[str]) -> list[str]:
    """Strip, uppercase, and deduplicate tickers while preserving their order.

    Raise ValueError if no nonempty tickers remain.
    """
    universe: list[str] = []
    for ticker in tickers:
        upper = ticker.strip().upper()
        if upper and upper not in universe:
            universe.append(upper)
    if not universe:
        raise ValueError("universe is empty — a run needs at least one ticker")
    return universe


def load_spec(path: str | Path) -> FundSpec:
    """Load a YAML mandate, raising ValueError with its path and error details."""
    data = _read_yaml(path)
    if "schema_version" not in data:
        raise ValueError(f"{path}: schema_version: This fund uses an older format. " "Recreate it or update its configuration.")
    try:
        return FundSpec.model_validate(data)
    except ValidationError as exc:
        raise ValueError(_validation_message(path, exc)) from exc


def load_strategy(path: str | Path) -> StrategySpec:
    """Load one strategy (a library file under hedge_fund/strategies/) from YAML."""
    try:
        return StrategySpec.model_validate(_read_yaml(path))
    except ValidationError as exc:
        raise ValueError(_validation_message(path, exc)) from exc


def _validation_message(path: str | Path, exc: ValidationError) -> str:
    errors = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
    return f"{path}: {errors}"


def _read_yaml(path: str | Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(Path(path).read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: cannot read configuration: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: configuration must be a YAML mapping")
    return data


def custom_strategy(names: list[str]) -> StrategySpec:
    """Build a strategy with its mode derived from registered analyst profiles.

    Duplicate names are removed in selection order. Any analyst that permits
    shorting enables long/short mode; dollar neutrality is never inferred.
    Invalid or empty selections raise ValueError.
    """
    models = [ModelSpec(name=n) for n in dict.fromkeys(names)]
    mode: PortfolioMode = "long_short" if any(get_investment_approach(m.name) == "long_short" for m in models) else "long_only"
    return StrategySpec(name="custom", models=models, blend=BlendPolicy(mode=mode))


def _refuse_unblindeable(model: object, strategy: str, *, blind: bool) -> None:
    """A backtest must not staff a model that cannot be blinded.

    `blind=True` withholds the ticker, industry and dates so a model that
    remembers the company cannot recall the outcome it is being scored on.
    A model that needs the ticker handed to it cannot honour that, so the run
    stops here instead of producing a number that means nothing.
    """
    if not blind:
        return
    cls = model if isinstance(model, type) else type(model)
    if getattr(cls, "supports_blind", True):
        return
    raise ValueError(
        f"model {cls.__name__} in strategy {strategy!r} cannot be blinded, so it "
        "cannot run in a backtest: it is handed the ticker itself. Use it in a "
        "live run, or replace it in this mandate for backtesting."
    )


class Fund:
    """A validated mandate with persistent model instances for each strategy.

    Models are constructed once so their caches survive successive cycles.
    Callers may supply instances keyed by strategy name, including test doubles.

    `blind=True` is for backtests: the LLM agents render prompts without
    the ticker, industry or calendar dates, so the result can't lean on
    what the model remembers about the company. Quant models are unaffected.

    The `models` override (strategy name -> instances) exists for tests to
    inject fakes; production callers let the registry build the staff.
    The `allocator` override injects a CIO; production callers use the
    mandate's `allocator` field (default: static slices).
    """

    def __init__(
        self,
        spec: FundSpec,
        models: dict[str, list[AlphaModel]] | None = None,
        blind: bool = False,
        allocator: Allocator | None = None,
    ) -> None:
        self.spec = spec
        self.allocator: Allocator = (
            allocator if allocator is not None else get_allocator(spec.allocator)
        )
        self.strategies: list[tuple[StrategySpec, list[AlphaModel]]] = []
        for strategy in spec.strategies:
            if models is not None:
                for instance in models[strategy.name]:
                    _refuse_unblindeable(instance, strategy.name, blind=blind)
                self.strategies.append((strategy, models[strategy.name]))
                continue
            staff = []
            for m in strategy.models:
                if m.name not in ALPHA_MODEL_REGISTRY:
                    raise ValueError(f"unknown model {m.name!r} in strategy " f"{strategy.name!r}; available: {sorted(ALPHA_MODEL_REGISTRY)}")
                cls = ALPHA_MODEL_REGISTRY[m.name]
                _refuse_unblindeable(cls, strategy.name, blind=blind)
                params = {**m.params, "blind": True} if blind and issubclass(cls, LLMAgent) else m.params
                staff.append(cls(**params))
            self.strategies.append((strategy, staff))
