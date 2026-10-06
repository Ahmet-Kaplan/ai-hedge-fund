"""run_cycle end-to-end tests — fake data client + fake analysts + real SimBroker."""

import json
from datetime import date as _date
from datetime import timedelta

import pytest

from hedge_fund.brokers.sim import SimBroker
from hedge_fund.data.client import FDClientError
from hedge_fund.data.models import FinancialMetrics, Price
from hedge_fund.fund.allocator import (
    AllocatorContext,
    EqualWeightAllocator,
    StaticAllocator,
)
from hedge_fund.fund.spec import Fund, FundSpec
from hedge_fund.llm import PromptCache
from hedge_fund.models import Signal
from hedge_fund.pipeline.models import CycleRecord, PendingRunResult
from hedge_fund.pipeline.run_cycle import assess_fund, run_cycle
from hedge_fund.signals import BuffettAgent, MungerAgent

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeDataClient:
    """Canned closes per ticker; a ticker absent from `closes` has no bars.

    ``price_error`` / ``news_error`` simulate infra failures (must raise,
    not look like empty data). ``news`` is the canned ``get_news`` payload.
    """

    def __init__(self, closes, news=None, price_error=None, news_error=None):
        self._closes = closes
        self._news = news if news is not None else []
        self._price_error = price_error
        self._news_error = news_error

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        if self._price_error is not None:
            raise self._price_error
        if ticker == "SPY":  # the benchmark grid: a bar on any requested session
            return [Price(open=100, close=100, high=100, low=100, volume=1000,
                          time=f"{start_date}T00:00:00Z")]
        close = self._closes.get(ticker)
        if close is None:
            return []
        return [Price(open=close, close=close, high=close, low=close,
                      volume=1000, time=f"{end_date}T00:00:00Z")]

    def get_news(self, ticker, end_date, start_date=None, limit=1000):
        if self._news_error is not None:
            raise self._news_error
        return list(self._news)


class FakeAnalyst:
    """Fixed conviction per ticker; counts predict calls."""

    investment_approach = "long_short"

    def __init__(self, name, views=None, abstain=False, error=None):
        self._name = name
        self._views = views or {}
        self._abstain = abstain
        self._error = error
        self.predict_calls = []

    @property
    def name(self):
        return self._name

    def predict(self, ticker, date, data_client):
        self.predict_calls.append(ticker)
        if self._error is not None:
            raise self._error
        metadata = {"abstained": True} if self._abstain else {}
        value = 0.0 if self._abstain else self._views.get(ticker, 0.0)
        return Signal(model_name=self._name, ticker=ticker, date=date,
                      value=value, metadata=metadata)


@pytest.fixture(autouse=True)
def registered_fakes(monkeypatch):
    from hedge_fund.signals import ALPHA_MODEL_REGISTRY
    for name in ("a", "b"):
        monkeypatch.setitem(ALPHA_MODEL_REGISTRY, name, FakeAnalyst)


def _spec(strategies=None, max_position_pct=0.25):
    if strategies is None:
        strategies = [{"name": "solo", "models": [{"name": "a"}]}]
    strategies = [{"blend": {"mode": "long_short"}, **s} for s in strategies]
    return FundSpec(
        schema_version=2,
        name="test-fund",
        strategies=strategies,
        risk={"max_position_pct": max_position_pct, "max_gross_exposure": 1.0},
        capital=100_000.0,
    )


CLOSES = {"AAPL": 200.0, "MSFT": 400.0, "NVDA": 100.0}
# What to trade is a run-time argument, not a mandate field.
UNIVERSE = ["AAPL", "MSFT", "NVDA"]


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_full_cycle_record_is_consistent():
    spec = _spec(strategies=[
        {"name": "long", "models": [{"name": "a"}]},
        {"name": "short", "models": [{"name": "b"}]},
    ])
    fund = Fund(spec, models={
        "long": [FakeAnalyst("a", views={"AAPL": 1.0, "NVDA": 0.5})],
        "short": [FakeAnalyst("b", views={"MSFT": -1.0})],
    })
    broker = SimBroker(cash=100_000.0)

    record = run_cycle(fund, "2024-06-03", broker, FakeDataClient(CLOSES),
                       UNIVERSE)

    assert record.fund == "test-fund"
    assert record.schema_version == 2
    assert record.equity_before == pytest.approx(100_000.0)
    assert len(record.strategies) == 2
    assert all(len(sr.signals) == 3 for sr in record.strategies)  # 3 tickers x 1 analyst
    # Weights respect the hard caps
    for w in record.final_weights.values():
        assert abs(w) <= 0.25 + 1e-12
    # Fills mirror orders one-to-one, and the books balance
    assert len(record.fills) == len(record.orders) > 0
    assert record.nav == pytest.approx(
        record.cash + sum(s * record.marks[t] for t, s in record.positions.items())
    )
    # The short strategy's bearish view -> short position
    assert record.positions["MSFT"] < 0


def test_netting_math_two_strategies_unequal_slices():
    """Two sleeves, overlapping ticker, 3:1 slices — hand-computed netting."""
    spec = _spec(strategies=[
        {"name": "s1", "weight": 3.0, "models": [{"name": "a"}]},
        {"name": "s2", "weight": 1.0, "models": [{"name": "b"}]},
    ], max_position_pct=1.0)
    fund = Fund(spec, models={
        # s1 sleeve: AAPL 0.5, MSFT 0.5 (equal convictions, gross 1.0)
        "s1": [FakeAnalyst("a", views={"AAPL": 1.0, "MSFT": 1.0})],
        # s2 sleeve: MSFT -1.0 (only conviction takes full gross)
        "s2": [FakeAnalyst("b", views={"MSFT": -1.0})],
    })

    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                       FakeDataClient(CLOSES), UNIVERSE)

    s1, s2 = record.strategies
    assert s1.slice == pytest.approx(0.75)
    assert s2.slice == pytest.approx(0.25)
    assert s1.weights == {"AAPL": pytest.approx(0.5), "MSFT": pytest.approx(0.5),
                          "NVDA": 0.0}
    assert s2.weights["MSFT"] == pytest.approx(-1.0)
    # Netted: AAPL = .75*.5 = .375 ; MSFT = .75*.5 + .25*(-1) = .125
    assert record.target_weights["AAPL"] == pytest.approx(0.375)
    assert record.target_weights["MSFT"] == pytest.approx(0.125)


def test_slices_normalize():
    """weights 2/2 must mean exactly the same as 1/1."""
    def run(w1, w2):
        spec = _spec(strategies=[
            {"name": "s1", "weight": w1, "models": [{"name": "a"}]},
            {"name": "s2", "weight": w2, "models": [{"name": "b"}]},
        ])
        fund = Fund(spec, models={
            "s1": [FakeAnalyst("a", views={"AAPL": 1.0})],
            "s2": [FakeAnalyst("b", views={"NVDA": -0.5})],
        })
        return run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                         FakeDataClient(CLOSES), UNIVERSE)

    assert run(2.0, 2.0).target_weights == run(1.0, 1.0).target_weights


def test_static_allocator_path_is_bit_identical_to_legacy_slices():
    """Default CIO path is exactly weight/sum(weight) — same floats, same book."""
    spec = _spec(strategies=[
        {"name": "s1", "weight": 3.0, "models": [{"name": "a"}]},
        {"name": "s2", "weight": 1.0, "models": [{"name": "b"}]},
    ], max_position_pct=1.0)
    models = {
        "s1": [FakeAnalyst("a", views={"AAPL": 1.0, "MSFT": 1.0})],
        "s2": [FakeAnalyst("b", views={"MSFT": -1.0})],
    }
    fund = Fund(spec, models=models)
    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                       FakeDataClient(CLOSES), UNIVERSE)

    total = 3.0 + 1.0
    legacy = {"s1": 3.0 / total, "s2": 1.0 / total}
    assert {sr.name: sr.slice for sr in record.strategies} == legacy
    assert record.strategies[0].slice == 3.0 / total
    assert record.strategies[1].slice == 1.0 / total
    # Same floats whether the CIO is implicit, explicit, or passed in.
    via_arg = run_cycle(
        Fund(spec, models=models), "2024-06-03", SimBroker(cash=100_000.0),
        FakeDataClient(CLOSES), UNIVERSE, allocator=StaticAllocator(),
    )
    via_spec = run_cycle(
        Fund(spec.model_copy(update={"allocator": "static"}), models=models),
        "2024-06-03", SimBroker(cash=100_000.0), FakeDataClient(CLOSES),
        UNIVERSE,
    )
    assert record.model_dump_json() == via_arg.model_dump_json() == via_spec.model_dump_json()


def test_equal_weight_allocator_is_selectable():
    """The stub CIO can be selected and ignores 3:1 mandate slices."""
    spec = _spec(strategies=[
        {"name": "s1", "weight": 3.0, "models": [{"name": "a"}]},
        {"name": "s2", "weight": 1.0, "models": [{"name": "b"}]},
    ], max_position_pct=1.0)
    models = {
        "s1": [FakeAnalyst("a", views={"AAPL": 1.0, "MSFT": 1.0})],
        "s2": [FakeAnalyst("b", views={"MSFT": -1.0})],
    }
    static = run_cycle(
        Fund(spec, models=models), "2024-06-03", SimBroker(cash=100_000.0),
        FakeDataClient(CLOSES), UNIVERSE,
    )
    equal = run_cycle(
        Fund(spec.model_copy(update={"allocator": "equal_weight"}), models=models),
        "2024-06-03", SimBroker(cash=100_000.0), FakeDataClient(CLOSES),
        UNIVERSE,
    )
    via_arg = run_cycle(
        Fund(spec, models=models), "2024-06-03", SimBroker(cash=100_000.0),
        FakeDataClient(CLOSES), UNIVERSE, allocator=EqualWeightAllocator(),
    )

    assert static.strategies[0].slice == 0.75
    assert static.strategies[1].slice == 0.25
    assert equal.strategies[0].slice == 0.5
    assert equal.strategies[1].slice == 0.5
    assert equal.spec.allocator == "equal_weight"
    assert via_arg.strategies[0].slice == 0.5
    # Hand-computed: each sleeve is 50%. s1 = AAPL 0.5, MSFT 0.5; s2 MSFT -1.
    assert equal.target_weights["AAPL"] == pytest.approx(0.25)
    assert equal.target_weights["MSFT"] == pytest.approx(-0.25)
    assert equal.target_weights != static.target_weights


def test_broken_allocator_fails_loud():
    """A CIO that drops a strategy must not silently rebalance the book."""

    class BrokenAllocator:
        def allocate(self, context: AllocatorContext):
            return {context.strategies[0].name: 1.0}

    spec = _spec(strategies=[
        {"name": "s1", "models": [{"name": "a"}]},
        {"name": "s2", "models": [{"name": "b"}]},
    ])
    fund = Fund(spec, models={
        "s1": [FakeAnalyst("a", views={"AAPL": 1.0})],
        "s2": [FakeAnalyst("b", views={"MSFT": -1.0})],
    }, allocator=BrokenAllocator())
    with pytest.raises(ValueError, match="expected"):
        run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                  FakeDataClient(CLOSES), UNIVERSE)


def test_deterministic_and_json_round_trips():
    def make():
        spec = _spec()
        fund = Fund(spec, models={
            "solo": [FakeAnalyst("a", views={"AAPL": 1.0, "MSFT": -0.5})],
        })
        return run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                         FakeDataClient(CLOSES), UNIVERSE)

    first, second = make(), make()
    assert first.model_dump_json() == second.model_dump_json()
    assert CycleRecord.model_validate_json(first.model_dump_json()) == first


def test_second_cycle_rebalances_not_restarts():
    analyst = FakeAnalyst("a", views={"AAPL": 1.0})
    fund = Fund(_spec(max_position_pct=1.0), models={"solo": [analyst]})
    broker = SimBroker(cash=100_000.0)
    data = FakeDataClient({"AAPL": 200.0})

    first = run_cycle(fund, "2024-06-03", broker, data, ["AAPL"])
    second = run_cycle(fund, "2024-06-04", broker, data, ["AAPL"])

    assert first.positions["AAPL"] == 500  # 100k at 200
    assert second.orders == []  # already at target; nothing to trade
    assert second.positions["AAPL"] == 500


# ---------------------------------------------------------------------------
# Abstain / flat behavior
# ---------------------------------------------------------------------------

def test_all_abstain_closes_the_book_to_flat():
    analyst = FakeAnalyst("a", views={"AAPL": 1.0})
    fund = Fund(_spec(max_position_pct=1.0), models={"solo": [analyst]})
    broker = SimBroker(cash=100_000.0)
    data = FakeDataClient({"AAPL": 200.0})
    run_cycle(fund, "2024-06-03", broker, data, ["AAPL"])
    assert broker.positions()["AAPL"].shares == 500

    analyst._abstain = True
    record = run_cycle(fund, "2024-06-04", broker, data, ["AAPL"])

    assert record.positions == {}  # book closed to flat
    assert record.nav == pytest.approx(100_000.0)  # flat closes at same price


# ---------------------------------------------------------------------------
# Pricing edge cases
# ---------------------------------------------------------------------------

def test_unpriced_unowned_ticker_skipped_and_analysts_never_called():
    analyst = FakeAnalyst("a", views={"AAPL": 1.0})
    fund = Fund(_spec(), models={"solo": [analyst]})
    closes = dict(CLOSES)
    del closes["NVDA"]

    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                       FakeDataClient(closes), UNIVERSE)

    assert [s.ticker for s in record.skipped] == ["NVDA"]
    assert "NVDA" not in analyst.predict_calls
    assert "NVDA" not in record.final_weights


def test_unpriced_held_ticker_raises():
    broker = SimBroker(cash=100_000.0)
    fund = Fund(_spec(), models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})
    run_cycle(fund, "2024-06-03", broker, FakeDataClient(CLOSES), UNIVERSE)
    assert broker.positions()  # something is held

    closes = {t: c for t, c in CLOSES.items() if t not in broker.positions()}
    with pytest.raises(ValueError, match="cannot value the book"):
        run_cycle(fund, "2024-06-04", broker, FakeDataClient(closes), UNIVERSE)


def test_universe_is_a_run_time_argument():
    """The same fund, pointed at different names, trades different names —
    and the record says what it was asked to trade."""
    fund = Fund(_spec(max_position_pct=1.0), models={
        "solo": [FakeAnalyst("a", views={"AAPL": 1.0, "MSFT": 1.0})],
    })

    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                       FakeDataClient(CLOSES), ["aapl", "AAPL"])

    assert record.universe == ["AAPL"]  # upper-cased and de-duped
    assert "MSFT" not in record.final_weights


def test_empty_universe_raises():
    fund = Fund(_spec(), models={"solo": [FakeAnalyst("a")]})
    with pytest.raises(ValueError, match="universe is empty"):
        run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                  FakeDataClient(CLOSES), [])


def test_analyst_error_propagates():
    """Fail loud: an infrastructure failure must not become a quiet no-trade."""
    fund = Fund(_spec(), models={
        "solo": [FakeAnalyst("a", error=ConnectionError("API down"))],
    })
    with pytest.raises(ConnectionError):
        run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                  FakeDataClient(CLOSES), UNIVERSE)


def test_changed_mode_is_enforced_on_next_cycle():
    fund = Fund(_spec(), models={"solo": [FakeAnalyst("a", views={"AAPL": -.8})]})
    fund.spec.strategies[0].blend.mode = "long_only"
    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000), FakeDataClient(CLOSES), UNIVERSE)
    assert record.orders == []
    assert record.final_weights == dict.fromkeys(UNIVERSE, 0)


def _rule_fund(mode, names=("buffett", "druckenmiller"), views=None, **risk):
    spec = FundSpec(
        schema_version=2, name="rules", strategies=[{
            "name": "team", "models": [{"name": name} for name in names],
            "blend": {"mode": mode},
        }], risk={"max_position_pct": .25, "max_gross_exposure": 1, **risk},
    )
    views = views or {name: {"A": .8, "B": -.6} for name in names}
    return Fund(spec, models={"team": [FakeAnalyst(name, views[name]) for name in names]})


@pytest.mark.parametrize("mode", ["long_only", "long_short", "dollar_neutral"])
def test_all_modes_execute_and_preserve_complete_records(mode):
    fund = _rule_fund(mode)
    record = run_cycle(fund, "2024-06-03", SimBroker(100_000), FakeDataClient({"A": 100, "B": 200}), ["A", "B"])
    strategy = record.strategies[0]
    assert strategy.convictions == pytest.approx({"A": .8, "B": -.6})
    assert strategy.eligible_scores == pytest.approx({"A": .8, "B": 0 if mode == "long_only" else -.3})
    if mode == "long_only":
        assert record.positions == {"A": 250}
    else:
        assert record.positions == {"A": 250, "B": -125}
    assert strategy.final_contribution == record.final_weights
    assert record.nav == 100_000
    assert CycleRecord.model_validate_json(record.model_dump_json()) == record
    old_fields = record.model_dump()
    old_fields.pop("risk_scale_factor")
    for sr in old_fields["strategies"]:
        for field in ("eligible_scores", "flat_reason", "final_contribution"):
            sr.pop(field)
    readable = CycleRecord.model_validate(old_fields)
    assert readable.strategies[0].eligible_scores == {}
    assert readable.risk_scale_factor is None


def test_neutral_flat_strategy_reserves_capital_and_closes_prior_exposure():
    spec = _spec([
        {"name": "neutral", "weight": 3, "models": [{"name": "a"}], "blend": {"mode": "dollar_neutral"}},
        {"name": "owner", "weight": 1, "models": [{"name": "b"}], "blend": {"mode": "long_only"}},
    ], max_position_pct=1)
    model = FakeAnalyst("a", {"A": 1, "B": -1})
    fund = Fund(spec, models={"neutral": [model], "owner": [FakeAnalyst("b", {"C": 1})]})
    broker = SimBroker(100_000)
    data = FakeDataClient({"A": 100, "B": 100, "C": 100})
    first = run_cycle(fund, "2024-06-03", broker, data, ["A", "B", "C"])
    assert first.positions == {"A": 375, "B": -375, "C": 250}
    model._views = {"A": 1, "B": .2}
    second = run_cycle(fund, "2024-06-04", broker, data, ["A", "B", "C"])
    assert second.strategies[0].flat_reason == "missing_short_side"
    assert second.positions == {"C": 250}
    assert second.cash == 75_000
    assert {o.ticker for o in second.orders} == {"A", "B"}


def test_neutral_fund_scaling_preserves_sleeves_with_overlap():
    spec = _spec([
        {"name": "neutral", "weight": 3, "models": [{"name": "a"}], "blend": {"mode": "dollar_neutral"}},
        {"name": "owner", "weight": 1, "models": [{"name": "b"}], "blend": {"mode": "long_only"}},
    ])
    fund = Fund(spec, models={"neutral": [FakeAnalyst("a", {"A": 1, "B": -1})],
                            "owner": [FakeAnalyst("b", {"A": 1})]})
    record = run_cycle(fund, "2024-06-03", SimBroker(100_000), FakeDataClient({"A": 100, "B": 100}), ["A", "B"])
    assert record.target_weights == {"A": .625, "B": -.375}
    assert record.risk_scale_factor == .4
    neutral, owner = record.strategies
    assert neutral.final_contribution == pytest.approx({"A": .15, "B": -.15})
    assert owner.final_contribution == pytest.approx({"A": .1, "B": 0})
    assert record.final_weights == pytest.approx({"A": .25, "B": -.15})
    assert sum(record.final_weights.values()) == pytest.approx(.1)  # only the neutral strategy must balance


def test_clipping_attribution_preserves_exactly_offset_contributions():
    spec = _spec([
        {"name": "s1", "models": [{"name": "a"}]},
        {"name": "s2", "models": [{"name": "b"}]},
    ], max_position_pct=.2)
    fund = Fund(spec, models={"s1": [FakeAnalyst("a", {"A": 1, "B": 1})],
                            "s2": [FakeAnalyst("b", {"A": -1, "B": 1})]})
    record = run_cycle(fund, "2024-06-03", SimBroker(100_000), FakeDataClient({"A": 100, "B": 100}), ["A", "B"])
    assert record.risk_scale_factor is None
    assert record.final_weights == {"A": 0, "B": .2}
    assert record.strategies[0].final_contribution == {"A": .25, "B": .1}
    assert record.strategies[1].final_contribution == {"A": -.25, "B": .1}


def test_whole_share_rounding_keeps_target_neutral_without_changing_orders():
    fund = _rule_fund("dollar_neutral", max_position_pct=1)
    record = run_cycle(fund, "2024-06-03", SimBroker(10_000), FakeDataClient({"A": 300, "B": 700}), ["A", "B"])
    assert record.final_weights == {"A": .5, "B": -.5}
    assert [(o.ticker, o.side, o.quantity) for o in record.orders] == [("B", "sell", 7), ("A", "buy", 16)]
    assert sum(record.positions[t] * record.marks[t] for t in record.positions) == -100


@pytest.mark.parametrize("case,expected", [
    ("nonfinite", "finite"), ("neutrality", "dollar-neutral"),
    ("position", "max_position_pct"), ("gross", "max_gross_exposure"),
    ("contributions", "contributions do not sum"),
])
def test_invalid_risk_output_never_reaches_broker(monkeypatch, case, expected):
    from importlib import import_module
    from unittest.mock import Mock

    from hedge_fund.risk.limits import RiskResult
    pipeline = import_module("hedge_fund.pipeline.run_cycle")
    mode = "dollar_neutral" if case in ("neutrality", "contributions") else "long_short"
    fund = _rule_fund(mode, max_gross_exposure=.3 if case == "gross" else 1)
    targets = {"A": .25, "B": -.25}
    factor = None
    if case == "nonfinite":
        targets["A"] = float("nan")
    elif case == "neutrality":
        targets["B"] = -.1
    elif case == "position":
        targets["A"] = .4
    elif case == "contributions":
        factor = .4  # recorded contributions will disagree with final weights
    monkeypatch.setattr(pipeline, "apply_limits", lambda *args, **kwargs: RiskResult(weights=targets, clamps=[], scale_factor=factor))
    broker = SimBroker(100_000)
    broker.place_order = Mock(side_effect=AssertionError("invalid orders reached broker"))
    with pytest.raises(ValueError, match=expected):
        run_cycle(fund, "2024-06-03", broker, FakeDataClient({"A": 100, "B": 100}), ["A", "B"])
    broker.place_order.assert_not_called()


@pytest.mark.parametrize("mode,expected", [("long_only", "long-only"), ("long_short", "short evidence")])
def test_invalid_short_targets_never_reach_broker(monkeypatch, mode, expected):
    from importlib import import_module
    from unittest.mock import Mock
    pipeline = import_module("hedge_fund.pipeline.run_cycle")
    original = pipeline.blend_signals
    def invalid_blend(*args, **kwargs):
        result = original(*args, **kwargs)
        result.weights["A"] = -1
        return result
    monkeypatch.setattr(pipeline, "blend_signals", invalid_blend)
    fund = _rule_fund(mode, names=("buffett",), views={"buffett": {"A": -.8}})
    broker = SimBroker(100_000)
    broker.place_order = Mock(side_effect=AssertionError("invalid orders reached broker"))
    with pytest.raises(ValueError, match=expected):
        run_cycle(fund, "2024-06-03", broker, FakeDataClient({"A": 100}), ["A"])
    broker.place_order.assert_not_called()


def test_strategy_gross_violation_is_rejected_even_when_fund_risk_clips_it(monkeypatch):
    from importlib import import_module
    from unittest.mock import Mock
    pipeline = import_module("hedge_fund.pipeline.run_cycle")
    original = pipeline.blend_signals
    def invalid_blend(*args, **kwargs):
        result = original(*args, **kwargs)
        result.weights = {"A": 1, "B": -1}
        return result
    monkeypatch.setattr(pipeline, "blend_signals", invalid_blend)
    fund = _rule_fund("long_short")
    broker = SimBroker(100_000)
    broker.place_order = Mock(side_effect=AssertionError("invalid orders reached broker"))
    with pytest.raises(ValueError, match="team.*strategy gross target"):
        run_cycle(fund, "2024-06-03", broker, FakeDataClient({"A": 100, "B": 100}), ["A", "B"])
    broker.place_order.assert_not_called()
def test_price_infra_failure_fails_the_cycle():
    """A data-client transport/auth error is not 'no close' / a skipped ticker."""
    fund = Fund(_spec(), models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})
    data = FakeDataClient(
        CLOSES,
        price_error=FDClientError("unauthorized", status_code=401, path="/prices/"),
    )
    with pytest.raises(FDClientError) as exc_info:
        run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0), data, UNIVERSE)
    assert exc_info.value.status_code == 401
    assert exc_info.value.path == "/prices/"


class NewsAnalyst:
    """Reads get_news: empty news → abstain; articles → a view.

    Mirrors the documented news/sentiment contract: empty list is no
    narrative input (a legitimate zero), not an infra failure.
    """

    name = "a"  # the key _spec() staffs it under

    def predict(self, ticker, date, data_client):
        news = data_client.get_news(ticker, date)
        if not news:
            return Signal(
                model_name=self.name, ticker=ticker, date=date, value=0.0,
                metadata={"abstained": True, "abstain_reason": "no news"},
            )
        return Signal(model_name=self.name, ticker=ticker, date=date, value=0.5)


def test_empty_news_is_no_narrative_not_a_failed_cycle():
    """Genuine empty news may abstain; the cycle still completes."""
    fund = Fund(_spec(), models={"solo": [NewsAnalyst()]})
    record = run_cycle(
        fund, "2024-06-03", SimBroker(cash=100_000.0),
        FakeDataClient(CLOSES, news=[]), UNIVERSE,
    )
    signals = record.strategies[0].signals
    assert signals
    assert all(s.value == 0.0 and s.metadata.get("abstained") for s in signals)
    # Abstentions are also listed on the record — not a silent omit.
    assert {d.ticker for d in record.dropped} == set(UNIVERSE)
    assert all(d.reason == "no news" for d in record.dropped)


def test_news_infra_failure_fails_the_cycle():
    """Auth/quota on get_news must not become a neutral Signal / empty cycle."""
    fund = Fund(_spec(), models={"solo": [NewsAnalyst()]})
    data = FakeDataClient(
        CLOSES,
        news_error=FDClientError(
            "unauthorized", status_code=401, path="/news/",
        ),
    )
    with pytest.raises(FDClientError) as exc_info:
        run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0), data, UNIVERSE)
    assert exc_info.value.status_code == 401
    assert exc_info.value.path == "/news/"


# ---------------------------------------------------------------------------
# Shared PIT snapshot + dropped views
# ---------------------------------------------------------------------------

BULLISH = json.dumps({
    "signal": "bullish", "confidence": 80, "reasoning": "Wonderful business.",
})


class RecordingLLM:
    """Canned LLM that keeps every user prompt it was shown."""

    model = "fake-model"

    def __init__(self, response=BULLISH):
        self._response = response
        self.users: list[str] = []

    def complete(self, system, user):
        self.users.append(user)
        return self._response


class DriftCycleClient:
    """Prices are stable; each fundamentals fetch mutates D/E and ROE."""

    def __init__(self, close=200.0):
        self._close = close
        self.metrics_calls = 0

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        return [Price(open=self._close, close=self._close, high=self._close,
                      low=self._close, volume=1000,
                      time=f"{end_date}T00:00:00Z")]

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        self.metrics_calls += 1
        quarters = [
            "2024-12-31", "2024-09-30", "2024-06-30", "2024-03-31",
            "2023-12-31", "2023-09-30", "2023-06-30", "2023-03-31",
        ]
        rows = []
        for i, q in enumerate(quarters):
            de = 0.5 * self.metrics_calls if i == 0 else 0.5
            roe = 0.20 * self.metrics_calls if i == 0 else 0.20
            rows.append(FinancialMetrics(
                ticker=ticker, report_period=q, period="ttm", filing_date=q,
                return_on_equity=roe, debt_to_equity=de,
                gross_margin=0.40, book_value_per_share=10.0, market_cap=1e9,
            ))
        return rows

    def get_company_facts(self, ticker):
        return None


def test_same_ticker_same_cycle_identical_fundamentals(tmp_path):
    """Two personas, one name, one tick — identical D/E, ROE, prompt, hash.

    The data client drifts on every fetch. If each persona built its own
    snapshot they would disagree; the cycle cache freezes the first fetch.
    """
    buffett_llm = RecordingLLM()
    munger_llm = RecordingLLM()
    data = DriftCycleClient()
    fund = Fund(_spec(strategies=[
        {"name": "value", "models": [{"name": "buffett"}]},
        {"name": "quality", "models": [{"name": "munger"}]},
    ]), models={
        "value": [BuffettAgent(llm=buffett_llm, cache=PromptCache(tmp_path / "b"))],
        "quality": [MungerAgent(llm=munger_llm, cache=PromptCache(tmp_path / "m"))],
    })

    record = assess_fund(fund, "2025-01-15", data, ["TEST"])

    hashes = [s.metadata["snapshot_hash"]
              for sr in record.strategies for s in sr.signals]
    assert len(hashes) == 2
    assert hashes[0] == hashes[1]
    assert buffett_llm.users == munger_llm.users
    assert len(buffett_llm.users) == 1
    prompt = buffett_llm.users[0]
    # First-fetch D/E 0.50 and ROE 0.20 — not the drifted 1.00 / 0.40.
    assert "| 0.50 |" in prompt  # latest D/E column
    assert "0.20" in prompt      # ROE avg / latest ROE
    assert "| 1.00 |" not in prompt
    assert data.metrics_calls == 1
    assert record.dropped == []


def test_abstained_views_are_listed_on_the_record():
    """A view blend ignores must appear on CycleRecord.dropped."""
    fund = Fund(_spec(), models={
        "solo": [FakeAnalyst("a", abstain=True)],
    })
    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                       FakeDataClient(CLOSES), UNIVERSE)

    assert {d.ticker for d in record.dropped} == set(UNIVERSE)
    assert all(d.model == "a" for d in record.dropped)
    assert all(d.strategy == "solo" for d in record.dropped)
    assert all(d.reason == "abstained" for d in record.dropped)
    # The view is still on the sleeve — dropped is the explicit exclude list.
    assert all(s.metadata.get("abstained") is True
               for sr in record.strategies for s in sr.signals)


def test_voting_views_are_not_dropped():
    fund = Fund(_spec(), models={
        "solo": [FakeAnalyst("a", views={"AAPL": 1.0})],
    })
    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                       FakeDataClient(CLOSES), UNIVERSE)
    assert record.dropped == []


def test_insufficient_history_is_dropped_with_reason(tmp_path):
    """Short history → abstain, and CycleRecord names the drop."""
    class ThinClient(DriftCycleClient):
        def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
            self.metrics_calls += 1
            return [
                FinancialMetrics(
                    ticker=ticker, report_period="2024-12-31", period="ttm",
                    filing_date="2024-12-31", return_on_equity=0.2,
                    debt_to_equity=0.5, market_cap=1e9,
                ),
            ]

    fund = Fund(_spec(strategies=[
        {"name": "value", "models": [{"name": "buffett"}]},
    ]), models={
        "value": [BuffettAgent(llm=RecordingLLM(),
                               cache=PromptCache(tmp_path / "b"))],
    })
    record = run_cycle(fund, "2025-01-15", SimBroker(cash=100_000.0),
                       ThinClient(), ["TEST"])

    assert len(record.dropped) == 1
    drop = record.dropped[0]
    assert drop.ticker == "TEST"
    assert drop.model == "buffett"
    assert drop.strategy == "value"
    assert "insufficient data" in drop.reason
    assert record.strategies[0].signals[0].metadata["abstained"] is True


# ---------------------------------------------------------------------------
# Sizing: risk-scaled weights and the per-name cap
# ---------------------------------------------------------------------------

def _series_client(series):
    from hedge_fund.backtesting.test_fund import FakeDataClient as SeriesClient

    return SeriesClient(series)


def _alt_days(n: int, start=_date(2024, 3, 1)):
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def test_inverse_vol_sizing_reads_price_history():
    """A calmer name carries more weight for the same view.

    Four times the volatility should be a quarter of the position, which is the
    whole point of sizing by risk rather than by conviction alone.
    """
    days = _alt_days(95)
    calm = {day: 100 * (1.005 if i % 2 else 0.995) for i, day in enumerate(days)}
    wild = {day: 100 * (1.02 if i % 2 else 0.98) for i, day in enumerate(days)}
    spec = _spec(strategies=[{"name": "solo", "models": [{"name": "a"}],
                              "blend": {"mode": "long_short", "sizing": "inverse_vol"}}],
                 max_position_pct=1.0)
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"CALM": 1.0, "WILD": 1.0})]})

    decision = assess_fund(fund, days[-1], _series_client(
        {"CALM": calm, "WILD": wild, "SPY": calm}), ["CALM", "WILD"])

    assert decision.final_weights["CALM"] > decision.final_weights["WILD"]
    assert decision.final_weights["CALM"] == pytest.approx(0.8, abs=0.02)


def test_conviction_sizing_is_the_default_and_ignores_volatility():
    """A mandate that does not ask for risk scaling must price as it always did."""
    days = _alt_days(95)
    calm = {day: 100 * (1.005 if i % 2 else 0.995) for i, day in enumerate(days)}
    wild = {day: 100 * (1.02 if i % 2 else 0.98) for i, day in enumerate(days)}
    spec = _spec(strategies=[{"name": "solo", "models": [{"name": "a"}]}], max_position_pct=1.0)
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"CALM": 1.0, "WILD": 1.0})]})

    decision = assess_fund(fund, days[-1], _series_client(
        {"CALM": calm, "WILD": wild, "SPY": calm}), ["CALM", "WILD"])

    assert decision.final_weights["CALM"] == pytest.approx(0.5, abs=0.001)


def test_volatility_reads_only_closes_on_or_before_the_as_of_date():
    """The window is point-in-time, like the marks the cycle trades at."""
    from hedge_fund.pipeline.run_cycle import _volatilities

    days = _alt_days(80)
    series = {day: 100 * (1.01 if i % 2 else 0.99) for i, day in enumerate(days)}
    series[days[-1]] = 5.0            # a violent move *after* the as-of date

    as_of = days[-2]
    window = _volatilities(["TEST"], as_of, _series_client({"TEST": series}))
    with_spike = _volatilities(["TEST"], days[-1], _series_client({"TEST": series}))

    assert window and with_spike
    assert window["TEST"] < with_spike["TEST"]     # the later move did not leak back


def test_a_name_without_enough_history_gets_no_volatility():
    """Left out rather than guessed; the blender gives it the median."""
    from hedge_fund.pipeline.run_cycle import _volatilities

    days = _alt_days(10)
    series = {day: 100.0 + i for i, day in enumerate(days)}

    assert _volatilities(["THIN"], days[-1], _series_client({"THIN": series})) == {}


def test_a_mandate_can_cap_a_name():
    """The cap rides the blend policy, so it is data, not code."""
    spec = _spec(strategies=[{"name": "solo", "models": [{"name": "a"}],
                              "blend": {"mode": "long_short", "max_name_weight": 0.3}}])
    assert spec.strategies[0].blend.max_name_weight == pytest.approx(0.3)


def test_the_cap_and_sizing_round_trip_through_the_mandate():
    spec = _spec(strategies=[{"name": "solo", "models": [{"name": "a"}],
                              "blend": {"mode": "long_short", "sizing": "inverse_vol",
                                        "max_name_weight": 0.4}}])

    reloaded = FundSpec.model_validate_json(spec.model_dump_json())

    assert reloaded.strategies[0].blend.sizing == "inverse_vol"
    assert reloaded.strategies[0].blend.max_name_weight == pytest.approx(0.4)


def test_an_unknown_sizing_mode_is_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _spec(strategies=[{"name": "solo", "models": [{"name": "a"}],
                           "blend": {"mode": "long_short", "sizing": "risk_parity"}}])


# ---------------------------------------------------------------------------
# Live execution: priced from the newest close, sent now
# ---------------------------------------------------------------------------

class SlippingBroker:
    """A venue that charges a different price than the one the fund sized on.

    Which is every real venue. The in-process books fill at the reference price
    by construction, so they cannot exercise the path that reports the gap.
    """

    def __init__(self, cash: float, *, drift: float) -> None:
        from hedge_fund.brokers.sim import SimBroker

        self._inner = SimBroker(cash=cash)
        self._drift = drift

    def positions(self):
        return self._inner.positions()

    def cash(self) -> float:
        return self._inner.cash()

    def place_order(self, order):
        filled = self._inner.place_order(
            order.model_copy(update={"price": order.price * (1 + self._drift)})
        )
        return filled.model_copy(update={"price": filled.price / (1 + self._drift) * (1 + self._drift)})


def _live_fund(views=None):
    return Fund(_spec(), models={"solo": [FakeAnalyst("a", views=views or {"AAPL": 1.0})]})


def test_a_live_run_trades_now_instead_of_waiting_for_a_close():
    """The next completed session is always in the future, so the backtest's
    next-close rule refuses to trade during every session. A live run cannot
    use it."""
    from hedge_fund.data.sessions import completed_through

    today = _date.today().isoformat()
    record = run_cycle(_live_fund(), today, SimBroker(cash=100_000.0),
                       FakeDataClient(CLOSES), UNIVERSE, live=True)

    assert not isinstance(record, PendingRunResult)
    assert record.execution_policy == "live_now"
    # Priced at the newest *completed* close — yesterday, not today. Today's own
    # close does not exist yet, which is exactly why the next-close rule could
    # never fire during a session.
    assert record.execution_as_of == completed_through()
    assert record.execution_as_of < today


def test_a_live_run_prices_from_its_own_assessment_close():
    """`session` must be the assessment date: a live order is sized on the
    newest completed close, not on a session that has not happened."""
    from hedge_fund.pipeline.run_cycle import execute_decision

    as_of = _date.today().isoformat()
    record = run_cycle(_live_fund(), as_of, SimBroker(cash=100_000.0),
                       FakeDataClient(CLOSES), UNIVERSE, live=True)

    with pytest.raises(ValueError, match="prices from its own assessment close"):
        execute_decision(_live_fund(), record.original_assessment, "1999-01-01",
                         SimBroker(cash=100_000.0), FakeDataClient(CLOSES), live=True)


def test_the_receipt_records_what_the_venue_charged_against_the_close():
    as_of = _date.today().isoformat()
    record = run_cycle(_live_fund(), as_of, SlippingBroker(cash=100_000.0, drift=0.01),
                       FakeDataClient(CLOSES), UNIVERSE, live=True)

    assert record.orders and record.slippage
    for slip in record.slippage:
        assert slip.fill_price == pytest.approx(slip.reference_price * 1.01)
        assert slip.per_share > 0                      # a buy paid more
        assert slip.notional == pytest.approx(slip.per_share * slip.quantity)


def test_slippage_is_signed_so_positive_is_always_worse():
    from hedge_fund.brokers.models import Fill, Order
    from hedge_fund.pipeline.run_cycle import _slippage

    worse_buy, worse_sell = (
        _slippage([Order(ticker="A", side="buy", quantity=10, price=100.0)],
                  [Fill(ticker="A", side="buy", quantity=10, price=101.0)]),
        _slippage([Order(ticker="A", side="sell", quantity=10, price=100.0)],
                  [Fill(ticker="A", side="sell", quantity=10, price=99.0)]),
    )

    assert worse_buy[0].per_share == pytest.approx(1.0)      # paid more
    assert worse_sell[0].per_share == pytest.approx(1.0)     # received less


def test_the_in_process_books_report_no_slippage():
    """Which is the cost a backtest cannot see — and the reason a live run has
    to report it rather than assume it away."""
    as_of = _date.today().isoformat()

    record = run_cycle(_live_fund(), as_of, SimBroker(cash=100_000.0),
                       FakeDataClient(CLOSES), UNIVERSE, live=True)

    assert record.slippage
    assert all(s.per_share == 0 and s.notional == 0 for s in record.slippage)
