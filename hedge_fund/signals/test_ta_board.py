"""TradingAgentsBoard tests — the adapter, with the framework faked.

The framework itself is never imported here: it is a heavy optional dependency
that is not even on PyPI under this name. What is pinned is the adapter's own
contract — the rating map, the abstention rules, the cache, and the refusal to
run in a blind backtest.
"""

from __future__ import annotations

import sys
import types

import pytest

from hedge_fund.fund import Fund, FundSpec
from hedge_fund.models import Signal
from hedge_fund.signals import ALPHA_MODEL_REGISTRY, get_investment_approach
from hedge_fund.signals.ta_board import (
    DEFAULT_RATING_LEVELS,
    RATING_REVIEW,
    RATINGS_5_TIER,
    TradingAgentsBoard,
    _import_framework,
    _rating_from,
)


def _board(tmp_path, **kwargs):
    kwargs.setdefault("cache_dir", tmp_path / "cache")
    return TradingAgentsBoard(**kwargs)


class FakeGraph:
    def __init__(self, rating="Buy", *, error=None):
        self.rating = rating
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def propagate(self, ticker, date):
        self.calls.append((ticker, date))
        if self.error is not None:
            raise self.error
        return ({"messages": []}, self.rating)


# ---------------------------------------------------------------------------
# Registry and permissions
# ---------------------------------------------------------------------------

def test_registered_and_two_sided():
    assert ALPHA_MODEL_REGISTRY["ta_board"] is TradingAgentsBoard
    assert get_investment_approach("ta_board") == "long_short"


def test_the_rating_vocabulary_matches_the_framework():
    """Verified against tradingagents/agents/rating.py, not guessed."""
    assert RATINGS_5_TIER == ("Buy", "Overweight", "Hold", "Underweight", "Sell")
    assert set(DEFAULT_RATING_LEVELS) == set(RATINGS_5_TIER)


# ---------------------------------------------------------------------------
# Rating -> Signal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rating,value", [
    ("Buy", 1.0),
    ("Overweight", 0.5),
    ("Hold", 0.0),
    ("Underweight", -0.5),
    ("Sell", -1.0),
])
def test_every_tier_maps_to_a_signed_conviction(tmp_path, rating, value):
    signal = _board(tmp_path, propagate=lambda t, d: ("state", rating)).predict(
        "AAPL", "2025-01-10", None
    )

    assert signal.value == pytest.approx(value)
    assert signal.metadata["rating"] == rating
    assert signal.metadata["abstained"] is False
    assert signal.metadata["data_source"] == "tradingagents"


def test_a_bare_rating_string_is_accepted(tmp_path):
    signal = _board(tmp_path, propagate=lambda t, d: "Buy").predict("AAPL", "2025-01-10", None)
    assert signal.value == pytest.approx(1.0)


def test_a_mapping_result_is_accepted(tmp_path):
    signal = _board(tmp_path, propagate=lambda t, d: {"signal": "Sell", "x": 1}).predict(
        "AAPL", "2025-01-10", None
    )
    assert signal.value == pytest.approx(-1.0)


def test_the_rating_map_can_be_overridden(tmp_path):
    board = _board(tmp_path, propagate=lambda t, d: "Hold",
                   rating_levels={**DEFAULT_RATING_LEVELS, "Hold": 0.2})
    assert board.predict("AAPL", "2025-01-10", None).value == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Abstention, never a silent neutral
# ---------------------------------------------------------------------------

def test_review_abstains_rather_than_reading_as_neutral(tmp_path):
    """REVIEW is the framework's 'no readable decision' sentinel.

    Recording it as a Hold would put a call in the log that nobody made.
    """
    signal = _board(tmp_path, propagate=lambda t, d: ("state", RATING_REVIEW)).predict(
        "AAPL", "2025-01-10", None
    )

    assert signal.value == 0.0
    assert signal.metadata["abstained"] is True
    assert "no readable rating" in signal.metadata["abstain_reason"]


def test_an_unrecognised_rating_abstains(tmp_path):
    signal = _board(tmp_path, propagate=lambda t, d: ("state", "Strongly Maybe")).predict(
        "AAPL", "2025-01-10", None
    )

    assert signal.metadata["abstained"] is True
    assert "Strongly Maybe" in signal.metadata["abstain_reason"]


def test_a_failed_run_abstains_instead_of_raising(tmp_path):
    """A run failure is 'no view', the same contract LLMAgent applies to a
    failed call: the pipeline keeps going and the record says why."""
    board = _board(tmp_path, propagate=lambda t, d: (_ for _ in ()).throw(
        RuntimeError("rate limited")))

    signal = board.predict("AAPL", "2025-01-10", None)

    assert signal.value == 0.0
    assert signal.metadata["abstained"] is True
    assert "rate limited" in signal.metadata["abstain_reason"]


def test_no_rating_in_a_mapping_is_an_error(tmp_path):
    board = _board(tmp_path, propagate=lambda t, d: {"unrelated": 1})
    signal = board.predict("AAPL", "2025-01-10", None)

    assert signal.metadata["abstained"] is True
    assert "no rating in mapping" in signal.metadata["abstain_reason"]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_a_second_ask_for_the_same_cell_is_free(tmp_path):
    graph = FakeGraph("Buy")
    board = _board(tmp_path, graph=graph)

    first = board.predict("AAPL", "2025-01-10", None)
    second = board.predict("AAPL", "2025-01-10", None)

    assert len(graph.calls) == 1          # the graph ran once
    assert first.metadata["cached"] is False
    assert second.metadata["cached"] is True
    assert second.value == first.value


def test_a_different_date_is_a_different_cell(tmp_path):
    graph = FakeGraph("Buy")
    board = _board(tmp_path, graph=graph)

    board.predict("AAPL", "2025-01-10", None)
    board.predict("AAPL", "2025-01-11", None)

    assert len(graph.calls) == 2


def test_changing_the_models_does_not_serve_stale_ratings(tmp_path):
    """The cache key carries a config fingerprint, or a re-run with different
    models would silently reuse the previous configuration's answers."""
    graph = FakeGraph("Buy")
    cheap = _board(tmp_path, graph=graph, config={"deep_think_llm": "a"})
    assert cheap.predict("AAPL", "2025-01-10", None).value == pytest.approx(1.0)

    better = _board(tmp_path, graph=graph, config={"deep_think_llm": "b"})
    assert better.predict("AAPL", "2025-01-10", None).metadata["cached"] is False
    assert len(graph.calls) == 2


def test_an_unreadable_cache_entry_is_a_miss(tmp_path):
    graph = FakeGraph("Buy")
    board = _board(tmp_path, graph=graph)
    path = board._cache_path("AAPL", "2025-01-10")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    assert board.predict("AAPL", "2025-01-10", None).value == pytest.approx(1.0)
    assert len(graph.calls) == 1


def test_an_unwritable_cache_does_not_stop_a_run(tmp_path):
    """Losing the cache costs money, not the run."""
    blocked = tmp_path / "blocked"
    blocked.write_text("this is a file, not a directory")  # mkdir under it fails
    graph = FakeGraph("Buy")
    board = _board(tmp_path, graph=graph, cache_dir=blocked / "ta_board")

    signal = board.predict("AAPL", "2025-01-10", None)

    assert signal.value == pytest.approx(1.0)
    assert signal.metadata["cached"] is False
    assert len(graph.calls) == 1


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------

def test_rating_is_read_from_the_frameworks_return_shapes():
    assert _rating_from(({"messages": []}, "Buy")) == "Buy"
    assert _rating_from({"signal": "Sell"}) == "Sell"
    assert _rating_from("Overweight") == "Overweight"
    with pytest.raises(ValueError):
        _rating_from(())
    with pytest.raises(ValueError):
        _rating_from({"unrelated": 1})


# ---------------------------------------------------------------------------
# Blind backtests refuse it
# ---------------------------------------------------------------------------

def _spec(models=("ta_board",)):
    return FundSpec(
        schema_version=2,
        name="research-desk",
        strategies=[{"name": "board", "models": [{"name": m} for m in models],
                     "blend": {"mode": "long_short"}}],
        risk={"max_position_pct": 0.25, "max_gross_exposure": 1.0},
        capital=100_000.0,
    )


def test_model_declares_that_it_cannot_be_blinded():
    assert TradingAgentsBoard.supports_blind is False


def test_a_blind_backtest_refuses_to_staff_it(tmp_path, monkeypatch):
    """The whole point of blinds is that the model cannot see the ticker.
    This one is handed the ticker, so a blind run would score recall as skill."""
    monkeypatch.setattr(
        "hedge_fund.signals.ta_board._import_framework",
        lambda: (_FakeGraphClass, {"llm_provider": "openai"}, RATINGS_5_TIER),
    )
    with pytest.raises(ValueError, match="cannot be blinded"):
        Fund(_spec(), blind=True)


def test_a_live_run_is_allowed(tmp_path):
    board = _board(tmp_path, propagate=lambda t, d: ("state", "Buy"))
    fund = Fund(_spec(), models={"board": [board]})

    assert fund.strategies[0][1][0] is board


def test_blinding_a_blindeable_model_still_works(tmp_path):
    """The guard must not disturb every other model in the registry."""
    from hedge_fund.pipeline.test_run_cycle import FakeAnalyst
    fund = Fund(_spec(models=("pead",)),
                models={"board": [FakeAnalyst("pead", {"AAPL": 0.5})]})
    assert fund.strategies[0][1][0].name == "pead"


class _FakeGraphClass:
    def __init__(self, selected_analysts=None, config=None):
        self.selected_analysts = selected_analysts
        self.config = config


# ---------------------------------------------------------------------------
# Import guard — the PyPI name collision
# ---------------------------------------------------------------------------

def _install_fake_framework(monkeypatch, tiers):
    modules = {
        "tradingagents": types.ModuleType("tradingagents"),
        "tradingagents.agents": types.ModuleType("tradingagents.agents"),
        "tradingagents.agents.rating": types.ModuleType("tradingagents.agents.rating"),
        "tradingagents.default_config": types.ModuleType("tradingagents.default_config"),
        "tradingagents.graph": types.ModuleType("tradingagents.graph"),
        "tradingagents.graph.trading_graph": types.ModuleType("tradingagents.graph.trading_graph"),
    }
    modules["tradingagents.agents.rating"].RATINGS_5_TIER = tiers
    modules["tradingagents.agents.rating"].RATING_REVIEW = "REVIEW"
    modules["tradingagents.default_config"].DEFAULT_CONFIG = {"llm_provider": "openai"}
    modules["tradingagents.graph.trading_graph"].TradingAgentsGraph = _FakeGraphClass
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_the_real_framework_imports(monkeypatch):
    _install_fake_framework(monkeypatch, RATINGS_5_TIER)
    graph_cls, config, tiers = _import_framework()
    assert graph_cls is _FakeGraphClass
    assert config["llm_provider"] == "openai"
    assert tuple(tiers) == RATINGS_5_TIER


def test_a_missing_framework_explains_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "tradingagents", None)  # forces ImportError
    with pytest.raises(ImportError) as exc:
        _import_framework()

    message = str(exc.value)
    assert "github.com/TauricResearch/TradingAgents" in message
    assert "not PyPI" in message or "unrelated project" in message


def test_the_same_named_pypi_package_is_rejected_with_an_explanation(monkeypatch):
    """'pip install tradingagents' fetches a different project. Importing it
    and running it would produce ratings from code nobody reviewed here."""
    _install_fake_framework(monkeypatch, ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"))

    with pytest.raises(ImportError) as exc:
        _import_framework()

    message = str(exc.value)
    assert "not TauricResearch" in message
    assert "same-named project is published on PyPI" in message


def test_artefacts_stay_under_this_projects_cache(tmp_path, monkeypatch):
    """No second data directory: everything lands under the project's cache."""
    captured: dict = {}

    class Capturing(_FakeGraphClass):
        def __init__(self, selected_analysts=None, config=None):
            super().__init__(selected_analysts, config)
            captured.update(config)

    monkeypatch.setattr(
        "hedge_fund.signals.ta_board._import_framework",
        lambda: (Capturing, {"llm_provider": "openai"}, RATINGS_5_TIER),
    )
    board = _board(tmp_path, cache_dir=tmp_path / "hedge" / "ta_board")
    board.predict("AAPL", "2025-01-10", None)

    base = tmp_path / "hedge"
    assert captured["data_cache_dir"] == str(base / "tradingagents" / "cache")
    assert captured["results_dir"] == str(base / "tradingagents" / "logs")
    assert str(base) in captured["memory_log_path"]


# ---------------------------------------------------------------------------
# Inside the pipeline
# ---------------------------------------------------------------------------

def test_the_board_runs_inside_a_cycle(tmp_path):
    """Not just a Signal factory: it survives a real execution path."""
    import yaml

    from hedge_fund.brokers import SimBroker
    from hedge_fund.fund import load_spec
    from hedge_fund.pipeline import run_cycle
    from hedge_fund.pipeline.test_run_cycle import CLOSES, FakeDataClient, UNIVERSE

    mandate = tmp_path / "board.yaml"
    mandate.write_text(yaml.safe_dump({
        "schema_version": 2,
        "name": "research-desk",
        "strategies": [{"name": "board", "models": [{"name": "ta_board"}],
                        "blend": {"mode": "long_short", "gross_target": 1.0}}],
        "risk": {"max_position_pct": 1.0, "max_gross_exposure": 1.0},
        "capital": 100_000.0,
    }))
    spec = load_spec(mandate)          # the mandate validates against the registry
    fund = Fund(spec, models={"board": [
        _board(tmp_path, propagate=lambda t, d: ("state", "Underweight")),
    ]})

    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                       FakeDataClient(CLOSES), UNIVERSE)

    signals = record.strategies[0].signals
    assert signals, "the board produced no signals"
    assert {s.value for s in signals} == {-0.5}          # Underweight
    assert all(s.metadata["data_source"] == "tradingagents" for s in signals)
    # A two-sided call on a long/short sleeve can put on shorts.
    assert any(shares < 0 for shares in record.positions.values())


def test_a_board_run_is_cached_across_the_cycle(tmp_path):
    """The pipeline asks per ticker; a replay must not re-pay for the graph."""
    from hedge_fund.brokers import SimBroker
    from hedge_fund.pipeline import run_cycle
    from hedge_fund.pipeline.test_run_cycle import CLOSES, FakeDataClient, UNIVERSE

    graph = FakeGraph("Buy")
    fund = Fund(_spec(), models={"board": [_board(tmp_path, graph=graph)]})

    run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
              FakeDataClient(CLOSES), UNIVERSE)
    first = len(graph.calls)

    run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
              FakeDataClient(CLOSES), UNIVERSE)

    assert first == len(UNIVERSE)      # one run per ticker, not per cycle
    assert len(graph.calls) == first   # the second cycle was free
