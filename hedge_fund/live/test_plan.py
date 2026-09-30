"""plan_rebalance tests — fake data + fake analyst, real assessment and sizing."""

import pytest

from hedge_fund.brokers.models import Order
from hedge_fund.data.models import Price
from hedge_fund.fund.spec import Fund, FundSpec
from hedge_fund.live.plan import plan_rebalance
from hedge_fund.models import Signal

FRIDAY, MONDAY = "2024-06-07", "2024-06-10"
SERIES = {"AAPL": {FRIDAY: 100.0}, "MSFT": {FRIDAY: 50.0}, "SPY": {FRIDAY: 500.0}}


class FakeDataClient:
    """Closes per ticker per date: {ticker: {date: close}}."""

    def __init__(self, series):
        self._series = series

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        return [
            Price(open=c, close=c, high=c, low=c, volume=1000, time=f"{d}T00:00:00Z")
            for d, c in sorted(self._series.get(ticker, {}).items())
            if start_date <= d <= end_date
        ]


class FakeAnalyst:
    investment_approach = "long_short"

    def __init__(self, name, views=None):
        self._name = name
        self._views = views or {}
        self.calls = []

    @property
    def name(self):
        return self._name

    def predict(self, ticker, date, data_client):
        self.calls.append((ticker, date))
        return Signal(model_name=self._name, ticker=ticker, date=date, value=self._views.get(ticker, 0.0))


@pytest.fixture(autouse=True)
def registered_fakes(monkeypatch):
    from hedge_fund.signals import ALPHA_MODEL_REGISTRY
    monkeypatch.setitem(ALPHA_MODEL_REGISTRY, "a", FakeAnalyst)


def make_fund(views, rebalance="weekly"):
    spec = FundSpec(
        schema_version=2, name="paper-test",
        strategies=[{"name": "solo", "models": [{"name": "a"}], "blend": {"mode": "long_short"}}],
        risk={"max_position_pct": 0.5, "max_gross_exposure": 1.0},
        rebalance=rebalance,
    )
    analyst = FakeAnalyst("a", views)
    return Fund(spec, models={"solo": [analyst]}), analyst


def test_buys_toward_the_capped_target_with_data_through_yesterday():
    fund, analyst = make_fund({"AAPL": 1.0})
    plan = plan_rebalance(fund, ["AAPL"], {}, 100_000.0, FakeDataClient(SERIES), MONDAY)
    assert plan.cutoff == "2024-06-09"
    assert analyst.calls == [("AAPL", "2024-06-09")]
    assert plan.marks == {"AAPL": 100.0}
    assert plan.equity == 100_000.0
    assert plan.orders == [Order(ticker="AAPL", side="buy", quantity=500, price=100.0)]


def test_closes_held_names_outside_the_targets_sells_first():
    fund, _ = make_fund({"AAPL": 1.0})
    plan = plan_rebalance(fund, ["AAPL"], {"MSFT": 10}, 99_500.0, FakeDataClient(SERIES), MONDAY)
    assert plan.equity == 100_000.0
    assert [(o.side, o.ticker, o.quantity) for o in plan.orders] == [("sell", "MSFT", 10), ("buy", "AAPL", 500)]


def test_short_positions_count_against_equity():
    fund, _ = make_fund({"AAPL": 1.0})
    plan = plan_rebalance(fund, ["AAPL"], {"AAPL": -100}, 110_000.0, FakeDataClient(SERIES), MONDAY)
    assert plan.equity == 100_000.0
    assert plan.orders == [Order(ticker="AAPL", side="buy", quantity=600, price=100.0)]


def test_flatten_closes_everything_without_asking_analysts():
    fund, analyst = make_fund({"AAPL": 1.0})
    plan = plan_rebalance(fund, ["AAPL"], {"AAPL": 10, "MSFT": -4}, 1_000.0, FakeDataClient(SERIES), MONDAY, flatten=True)
    assert plan.flatten and plan.decision is None
    assert analyst.calls == []
    assert [(o.side, o.ticker, o.quantity) for o in plan.orders] == [("sell", "AAPL", 10), ("buy", "MSFT", 4)]


def test_held_name_without_a_price_raises():
    fund, _ = make_fund({"AAPL": 1.0})
    with pytest.raises(ValueError, match="ZZZ"):
        plan_rebalance(fund, ["AAPL"], {"ZZZ": 5}, 100_000.0, FakeDataClient(SERIES), MONDAY)


def test_nonpositive_equity_raises():
    fund, _ = make_fund({"AAPL": 1.0})
    with pytest.raises(ValueError, match="equity"):
        plan_rebalance(fund, ["AAPL"], {}, 0.0, FakeDataClient(SERIES), MONDAY)


def test_idle_capital_goes_to_the_benchmark_exempt_from_the_name_cap():
    fund, _ = make_fund({"AAPL": 1.0})
    fund.spec = fund.spec.model_copy(update={"equitize_idle": True})
    plan = plan_rebalance(fund, ["AAPL"], {}, 100_000.0, FakeDataClient(SERIES), MONDAY)
    assert plan.targets == {"AAPL": pytest.approx(0.5), "SPY": pytest.approx(0.5)}
    assert [(o.side, o.ticker, o.quantity) for o in plan.orders] == [("buy", "AAPL", 500), ("buy", "SPY", 100)]
