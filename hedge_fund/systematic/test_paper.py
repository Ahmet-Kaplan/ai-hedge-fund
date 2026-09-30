"""Paper trading loop: idempotency, clock, stale data, retries, kill switch, audit."""

from __future__ import annotations

import json

import pytest

from hedge_fund.brokers.paper import BrokerUnavailable, PaperBroker
from hedge_fund.systematic import MarketPanel
from hedge_fund.systematic.backtest import BacktestConfig
from hedge_fund.systematic.decision import DecisionEngine
from hedge_fund.systematic.execution import CostModel, FillTiming, SimulatedExecution
from hedge_fund.systematic.paper import KILL_SWITCH, PaperTradingEngine, PaperTradingError
from hedge_fund.systematic.risk import RiskConfig
from hedge_fund.systematic.strategies import StrategyConfig, SystematicStrategy
from hedge_fund.systematic.testing import SyntheticMarket, weekdays

DAYS = weekdays("2023-01-02", "2023-04-28")


class HoldConfig(StrategyConfig):
    hold: tuple[str, ...] = ("AAA", "BBB")


class Hold(SystematicStrategy):
    name, version, Config = "hold", "test", HoldConfig

    def _generate(self, view, names):
        return [self._signal(view, t, 1.0 if t in self.config.hold else 0.0, {}) for t in names]


@pytest.fixture(scope="module")
def panel():
    m = SyntheticMarket()
    m.add_series("SPY", DAYS, SyntheticMarket.random_walk(DAYS, seed=0), volume=10_000_000)
    m.add_series("AAA", DAYS, SyntheticMarket.random_walk(DAYS, seed=1), volume=1_000_000)
    m.add_series("BBB", DAYS, SyntheticMarket.random_walk(DAYS, seed=2), volume=1_000_000)
    return MarketPanel.build(m, ["SPY", "AAA", "BBB"], DAYS[0], DAYS[-1])


def engine(panel, tmp_path, broker=None, **kw):
    cfg = BacktestConfig(start=DAYS[0], end=DAYS[-1], timing=FillTiming.NEXT_OPEN,
                         risk=RiskConfig(max_position=0.4, vol_target=None, max_turnover=None,
                                         max_adv_participation=1.0, max_cluster_weight=1.0))
    decider = DecisionEngine([Hold(exclude=("SPY",))], cfg)
    broker = broker or PaperBroker(100_000.0, SimulatedExecution(panel, CostModel(), FillTiming.NEXT_OPEN))
    return PaperTradingEngine(panel, decider, broker, tmp_path, rebalance="monthly", backoff=0,
                              clock=lambda: DAYS[-1], sleep=lambda s: None, **kw)


def run_through(e, end):
    return [e.run_session(d) for d in DAYS if d <= end]


def test_orders_submitted_after_close_fill_next_session(panel, tmp_path):
    e = engine(panel, tmp_path)
    reports = run_through(e, "2023-02-03")
    jan_end = next(r for r in reports if r["session"] == "2023-01-31")
    assert {o["symbol"] for o in jan_end["orders"]} == {"AAA", "BBB"}
    feb1 = next(r for r in reports if r["session"] == "2023-02-01")
    assert feb1["fills"] == 2 and feb1["reconciled"]
    assert set(e.broker.positions()) == {"AAA", "BBB"}


def test_rerunning_a_session_never_duplicates_orders(panel, tmp_path):
    e = engine(panel, tmp_path)
    run_through(e, "2023-01-31")
    n = len(e.broker.orders)
    assert e.run_session("2023-01-31")["status"] == "already_processed"
    # a fresh process with the same state dir resumes instead of resubmitting
    e2 = engine(panel, tmp_path, broker=e.broker)
    assert e2.run_session("2023-01-31")["status"] == "already_processed"
    assert len(e.broker.orders) == n


def test_clock_and_calendar(panel, tmp_path):
    e = engine(panel, tmp_path)
    e.clock = lambda: "2023-01-10"
    assert e.run_session("2023-01-11")["status"] == "refused_future_session"
    e.clock = lambda: DAYS[-1]
    assert e.run_session("2023-01-07")["status"] == "market_closed"          # Saturday


def test_stale_data_blocks_trading(panel, tmp_path):
    e = engine(panel, tmp_path)
    e.clock = lambda: "2023-06-30"
    r = e.run_session("2023-05-31")
    assert r["status"] == "stale_data" and r["orders"] == []


class Flaky(PaperBroker):
    def __init__(self, *a, failures=2, **kw):
        super().__init__(*a, **kw)
        self.failures = failures

    def submit(self, order):
        if self.failures:
            self.failures -= 1
            raise BrokerUnavailable("gateway reconnecting")
        return super().submit(order)


def test_transient_broker_failures_are_retried(panel, tmp_path):
    b = Flaky(100_000.0, SimulatedExecution(panel, CostModel(), FillTiming.NEXT_OPEN), failures=2)
    e = engine(panel, tmp_path, broker=b)
    r = run_through(e, "2023-01-31")[-1]
    assert len(r["orders"]) == 2
    events = [json.loads(x)["event"] for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert events.count("broker_retry") == 2


def test_persistent_broker_failure_stops_the_cycle(panel, tmp_path):
    b = Flaky(100_000.0, SimulatedExecution(panel, CostModel(), FillTiming.NEXT_OPEN), failures=99)
    e = engine(panel, tmp_path, broker=b)
    run_through(e, "2023-01-30")
    with pytest.raises(PaperTradingError):
        e.run_session("2023-01-31")
    assert e.load_state()["last_session"] == "2023-01-30"      # failed session not marked done


def test_kill_switch_flattens_and_halts(panel, tmp_path):
    e = engine(panel, tmp_path)
    run_through(e, "2023-02-10")
    assert e.broker.positions()
    (tmp_path / KILL_SWITCH).write_text("manual stop")
    r = e.run_session("2023-02-13")
    assert r["status"] == "halted" and {o["side"] for o in r["orders"]} == {"sell"}
    e.run_session("2023-02-14")
    assert e.broker.positions() == {}
    (tmp_path / KILL_SWITCH).unlink()
    for d in [d for d in DAYS if "2023-02-15" <= d <= "2023-03-31"]:
        assert e.run_session(d)["status"] == "halted"          # stays halted until a human resets state
    assert e.broker.positions() == {}


def test_refuses_a_live_venue(panel, tmp_path):
    class Live(PaperBroker):
        live = True
    with pytest.raises(PaperTradingError):
        engine(panel, tmp_path, broker=Live(1.0, SimulatedExecution(panel, CostModel(), FillTiming.NEXT_OPEN)))
