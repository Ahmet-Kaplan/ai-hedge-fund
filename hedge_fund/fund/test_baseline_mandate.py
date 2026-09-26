"""The Buffett baseline mandate (configs/buffett-baseline.yaml).

Validates the file against the real schema, pins every target-setup
invariant, and replays it through the real backtester with synthetic prices
and a stub standing in for the LLM — mechanics only, never a performance
claim.
"""

from datetime import date, timedelta
from pathlib import Path

import pytest

from hedge_fund.backtesting.fund import backtest_fund
from hedge_fund.data.models import Price
from hedge_fund.fund.spec import Fund, load_spec
from hedge_fund.models import Signal
from hedge_fund.signals import ALPHA_MODEL_REGISTRY, BuffettAgent, get_investment_approach
from hedge_fund.verification.__main__ import load_universe

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
MANDATE = CONFIGS / "buffett-baseline.yaml"
UNIVERSE = CONFIGS / "baseline-universe.yaml"

EXPECTED_UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BRK.B", "JPM", "KO", "PG", "JNJ", "WMT", "XOM", "CVX", "V", "MA"]


@pytest.fixture(scope="module")
def spec():
    return load_spec(MANDATE)


def test_mandate_matches_target_setup(spec):
    assert spec.name == "buffett-baseline"
    assert spec.capital == 100_000
    assert spec.rebalance == "monthly"
    assert spec.benchmark == "SPY"
    assert spec.risk.max_position_pct == 0.15
    assert spec.risk.max_gross_exposure == 1.0  # no leverage


def test_mandate_is_buffett_only_and_long_only(spec):
    assert len(spec.strategies) == 1
    strategy = spec.strategies[0]
    assert [m.name for m in strategy.models] == ["buffett"]
    assert strategy.models[0].params == {}
    assert strategy.blend.mode == "long_only"
    assert strategy.blend.gross_target <= 1.0
    assert ALPHA_MODEL_REGISTRY["buffett"] is BuffettAgent
    assert get_investment_approach("buffett") == "long_only"


def test_baseline_universe_file():
    assert load_universe(UNIVERSE) == EXPECTED_UNIVERSE


# ---------------------------------------------------------------------------
# Synthetic replay through the real pipeline
# ---------------------------------------------------------------------------

def _weekdays(start: str, end: str) -> list[str]:
    d, out = date.fromisoformat(start), []
    while d <= date.fromisoformat(end):
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DAYS = _weekdays("2024-01-02", "2024-04-30")


class FakeData:
    """Every name trades every weekday; prices drift so NAV moves."""

    def get_prices(self, ticker, start_date, end_date, **kw):
        base = 50.0 + 10 * (sum(map(ord, ticker)) % 20)
        return [Price(open=p, close=p, high=p, low=p, volume=1, time=f"{d}T00:00:00Z")
                for i, d in enumerate(DAYS) if start_date <= d <= end_date for p in [base * (1 + 0.001 * i)]]


class StubBuffett:
    """Stands in for the LLM: bullish on three names, bearish on one, neutral elsewhere."""

    investment_approach = "long_only"
    name = "buffett"
    VIEWS = {"KO": 0.9, "JNJ": 0.8, "AAPL": 0.7, "XOM": -0.8}

    def predict(self, ticker, date, data_client):
        return Signal(model_name="buffett", ticker=ticker, date=date, value=self.VIEWS.get(ticker, 0.0),
                      reasoning="stub", metadata={"abstained": False})


@pytest.fixture(scope="module")
def replay(spec):
    fund = Fund(spec, models={"buffett-value": [StubBuffett()]})
    return backtest_fund(fund, DAYS[0], DAYS[-1], FakeData(), EXPECTED_UNIVERSE)


def test_replay_rebalances_monthly_at_next_close(replay):
    # Month-ends Jan/Feb/Mar execute on the next session; April's is pending.
    assert [r.as_of for r in replay.records] == ["2024-01-31", "2024-02-29", "2024-03-29"]
    assert [r.execution_as_of for r in replay.records] == ["2024-02-01", "2024-03-01", "2024-04-01"]
    assert [p.as_of for p in replay.pending] == ["2024-04-30"]


def test_replay_caps_positions_and_keeps_the_rest_in_cash(replay):
    first = replay.records[0]
    assert set(first.final_weights) == set(EXPECTED_UNIVERSE)
    held = {t: w for t, w in first.final_weights.items() if w}
    # 3 bullish names would get ~33% each; the 15% cap clamps all of them.
    assert held == pytest.approx({"KO": 0.15, "JNJ": 0.15, "AAPL": 0.15})
    assert {c.ticker for c in first.clamps} == {"KO", "JNJ", "AAPL"}
    assert first.cash / first.nav == pytest.approx(0.55, abs=0.01)


def test_replay_never_shorts_levers_or_breaches_the_cap(replay):
    for record in replay.records:
        assert all(shares >= 0 for shares in record.positions.values())
        assert all(o.side == "buy" or record.positions.get(o.ticker, 0) >= 0 for o in record.orders)
        assert record.cash >= 0
        invested = sum(shares * record.marks[t] for t, shares in record.positions.items())
        assert invested <= record.nav + 1e-6
        for ticker, shares in record.positions.items():
            assert shares * record.marks[ticker] / record.nav <= 0.15 + 1e-9
    assert "XOM" not in replay.records[-1].positions  # bearish long-only view means no position, not a short


def test_replay_reports_the_target_metrics(replay):
    m = replay.metrics
    assert len(replay.nav) == len(replay.dates) == len(replay.benchmark_nav) == len(DAYS)
    assert replay.nav[0] == replay.benchmark_nav[0] == 100_000
    for field in ("total_return_pct", "annualized_return_pct", "max_drawdown_pct", "sharpe_ratio", "benchmark_return_pct", "excess_return_pct"):
        assert isinstance(getattr(m, field), float)
