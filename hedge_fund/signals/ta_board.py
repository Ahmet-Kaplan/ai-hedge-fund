"""TradingAgentsBoard — TauricResearch's debate graph as an alpha model.

`TradingAgents <https://github.com/TauricResearch/TradingAgents>`_ (arXiv
2412.20138) runs a LangGraph of analysts, a bull/bear researcher debate, a
trader and a three-way risk debate, and emits a five-tier rating. This adapter
turns that rating into the `Signal` every other model in this pipeline emits, so
the board can sit in a strategy next to the personas and the quant models.

    .. code-block:: yaml

        strategies:
          - name: research
            models:
              - name: ta_board
                params: {llm_provider: anthropic, deep_think_llm: claude-opus-5-5}

Rating map (the framework's own vocabulary, ``agents/rating.py``)::

    Buy +1.0 · Overweight +0.5 · Hold 0.0 · Underweight -0.5 · Sell -1.0

Four things to know before using it, all of them deliberate:

**It cannot be blinded.** The framework is handed the ticker and fetches its own
news, social and fundamentals. `supports_blind = False`, and `Fund(blind=True)`
refuses to staff it rather than run a backtest whose prompts name the company —
see ``_refuse_unblindeable`` in ``hedge_fund/fund/spec.py``.

**It bypasses this project's data contract.** Every other model reads a
point-in-time snapshot built through `DataClient`, which filters on filing date
and drops undated articles. This one does its own fetching, the framework's
point-in-time story is its own, and its social/news sources reflect *now* even
for a historical trade date (its README says so). `Signal.metadata` records
`data_source: "tradingagents"` so the audit trail shows which numbers came from
outside the contract.

**It is expensive.** One `propagate` call is a multi-agent graph with debate
rounds — several LLM calls per ticker and date. Results are cached on disk by
(ticker, date, config fingerprint) so a replay is free; the framework itself is
non-deterministic by design.

**A decision nobody can parse abstains.** `REVIEW` (the framework's sentinel for
"no readable rating") and anything outside the five tiers become an abstention,
never a neutral 0.0 — "I could not read this" is not "the view is flat", the
same convention `LLMAgent` uses for a failed call.

Install: the framework is not on PyPI. ``pip install tradingagents`` fetches an
unrelated project that reuses the name (`Mai0313/tradingagents`), so this module
imports the real one and says so plainly when it finds the other.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Mapping

from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import AlphaModel

logger = logging.getLogger(__name__)

# The framework's five-tier scale, most bullish first. Verified against
# tradingagents/agents/rating.py (RATINGS_5_TIER), not guessed.
RATINGS_5_TIER: tuple[str, ...] = ("Buy", "Overweight", "Hold", "Underweight", "Sell")
DEFAULT_RATING_LEVELS: Mapping[str, float] = {
    "Buy": 1.0,
    "Overweight": 0.5,
    "Hold": 0.0,
    "Underweight": -0.5,
    "Sell": -1.0,
}
#: Sentinel the framework emits when a decision carries no readable rating.
RATING_REVIEW = "REVIEW"

INSTALL_HINT = (
    "TradingAgents is installed from its repository, not PyPI:\n"
    "    pip install git+https://github.com/TauricResearch/TradingAgents.git\n"
    "('pip install tradingagents' fetches an unrelated project of the same name.)"
)


def _import_framework() -> tuple[Any, Any, Any]:
    """Import the real framework, or explain exactly what is wrong.

    `pip install tradingagents` installs a different project that reuses the
    name, so "import succeeded" is not enough — the rating module has to look
    like TauricResearch's.
    """
    try:
        from tradingagents.agents.rating import RATINGS_5_TIER as _TIERS
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph
    except ImportError as exc:
        raise ImportError(f"TradingAgentsBoard needs its framework.\n{INSTALL_HINT}") from exc

    if tuple(_TIERS) != RATINGS_5_TIER:
        raise ImportError(
            "the installed 'tradingagents' package is not TauricResearch's "
            f"framework (its rating scale is {tuple(_TIERS)!r}, expected "
            f"{RATINGS_5_TIER!r}). A same-named project is published on PyPI.\n"
            f"{INSTALL_HINT}"
        )
    return TradingAgentsGraph, DEFAULT_CONFIG, _TIERS


class TradingAgentsBoard(AlphaModel):
    """One signal per ticker, from a full multi-agent research run."""

    investment_approach = "long_short"  # Buy..Sell is a two-sided call
    supports_blind: ClassVar[bool] = False

    def __init__(
        self,
        *,
        graph: Any | None = None,
        config: dict[str, Any] | None = None,
        selected_analysts: tuple[str, ...] | None = None,
        rating_levels: Mapping[str, float] | None = None,
        cache_dir: Path | str | None = None,
        propagate: Any | None = None,
    ) -> None:
        """*graph* (or *propagate*) is the seam tests inject; both are optional.

        Production builds the real graph lazily on first use, so constructing a
        fund costs nothing until a cycle actually asks this model for a view.
        """
        self._graph = graph
        self._propagate = propagate
        self._config = config
        self._selected_analysts = selected_analysts or ("market", "social", "news", "fundamentals")
        self._levels = dict(rating_levels or DEFAULT_RATING_LEVELS)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self._fingerprint: str | None = None

    # ------------------------------------------------------------------
    # AlphaModel
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "ta_board"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        """Run the board for *ticker* as of *date* and fold the rating.

        `data_client` is unused on purpose: the framework fetches its own data,
        which is exactly why this model declares `supports_blind = False`.
        """
        cached = self._cache_get(ticker, date)
        if cached is not None:
            return self._to_signal(ticker, date, cached, cached=True)

        try:
            rating = self._run(ticker, date)
        except Exception as exc:  # a failed run is no view, not a crash
            logger.warning("ta_board run failed for %s@%s: %s", ticker, date, exc)
            return self._abstain(ticker, date, f"run failed: {type(exc).__name__}: {exc}")

        self._cache_put(ticker, date, rating)
        return self._to_signal(ticker, date, rating, cached=False)

    # ------------------------------------------------------------------
    # The framework
    # ------------------------------------------------------------------

    def _run(self, ticker: str, date: str) -> str:
        if self._propagate is not None:
            result = self._propagate(ticker, date)
        else:
            result = self._graph_or_build().propagate(ticker, date)
        return _rating_from(result)

    def _graph_or_build(self) -> Any:
        if self._graph is None:
            graph_cls, default_config, _ = _import_framework()
            config = dict(default_config)
            config.update(self._config or {})
            # Keep every artefact under this project's data directory rather
            # than scattering another ~/.tradingagents next to ~/.hedge-fund.
            base = self._cache_dir.parent
            config.setdefault("data_cache_dir", str(base / "tradingagents" / "cache"))
            config.setdefault("results_dir", str(base / "tradingagents" / "logs"))
            config.setdefault(
                "memory_log_path",
                str(base / "tradingagents" / "memory" / "trading_memory.md"),
            )
            self._config = config
            self._graph = graph_cls(selected_analysts=self._selected_analysts, config=config)
        return self._graph

    def config_fingerprint(self) -> str:
        """What the cache key depends on: analysts plus the LLM settings.

        Change the models or the debate rounds and the cached ratings no longer
        describe this configuration, so they must not be served.
        """
        if self._fingerprint is None:
            payload = {
                "analysts": list(self._selected_analysts),
                "config": {k: v for k, v in sorted((self._config or {}).items())
                           if isinstance(v, (str, int, float, bool, type(None)))},
                "levels": dict(sorted(self._levels.items())),
            }
            blob = json.dumps(payload, sort_keys=True, default=str)
            self._fingerprint = hashlib.sha256(blob.encode()).hexdigest()[:12]
        return self._fingerprint

    # ------------------------------------------------------------------
    # Signal folding
    # ------------------------------------------------------------------

    def _to_signal(self, ticker: str, date: str, rating: str, *, cached: bool) -> Signal:
        level = self._levels.get(rating)
        if rating == RATING_REVIEW or level is None:
            return self._abstain(
                ticker, date,
                f"no readable rating ({rating!r}); the framework could not settle on a call",
                cached=cached,
            )
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=level,
            reasoning=f"TradingAgents board rating: {rating}",
            metadata={
                "rating": rating,
                "abstained": False,
                "cached": cached,
                "data_source": "tradingagents",
                "analysts": list(self._selected_analysts),
            },
        )

    def _abstain(self, ticker: str, date: str, why: str, *, cached: bool = False) -> Signal:
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=0.0,
            reasoning=f"abstained: {why}",
            metadata={
                "abstained": True,
                "abstain_reason": why,
                "cached": cached,
                "data_source": "tradingagents",
            },
        )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str, date: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in ticker.upper())
        return self._cache_dir / f"{safe}-{date}-{self.config_fingerprint()}.json"

    def _cache_get(self, ticker: str, date: str) -> str | None:
        path = self._cache_path(ticker, date)
        if not path.exists():
            return None
        try:
            return str(json.loads(path.read_text())["rating"])
        except (OSError, ValueError, KeyError, TypeError):
            return None  # an unreadable cache entry is a miss, not a failure

    def _cache_put(self, ticker: str, date: str, rating: str) -> None:
        path = self._cache_path(ticker, date)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"ticker": ticker, "date": date, "rating": rating}))
        except OSError as exc:  # a cache that cannot be written must not stop a run
            logger.warning("ta_board could not cache %s@%s: %s", ticker, date, exc)


def _rating_from(result: Any) -> str:
    """Pull the rating out of whatever `propagate` returned.

    `propagate` returns `(final_state, signal)`. The tuple is unwrapped, and a
    mapping is accepted too so a caller can inject `{"signal": "Buy"}`.
    """
    if isinstance(result, Mapping):
        for key in ("signal", "rating", "decision"):
            if key in result:
                return str(result[key]).strip()
        raise ValueError(f"no rating in mapping with keys {sorted(result)}")
    if isinstance(result, (tuple, list)):
        if not result:
            raise ValueError("propagate returned an empty result")
        return _rating_from(result[-1])
    return str(result).strip()


def _default_cache_dir() -> Path:
    from hedge_fund.paths import CACHE_DIR

    return CACHE_DIR / "ta_board"
