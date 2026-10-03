"""Ledger read-back — seed PaperBroker from fixture CycleRecord receipts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from hedge_fund.backtesting.fund import backtest_fund
from hedge_fund.brokers.paper import PaperBroker
from hedge_fund.data.models import Price
from hedge_fund.fund.spec import Fund, FundSpec
from hedge_fund.ledger import (
    broker_for_run,
    latest_reference,
    latest_run_receipt,
    load_cycle_record,
    run_receipt_paths,
    save_cycle_record,
)
from hedge_fund.models import Signal
from hedge_fund.pipeline.run_cycle import run_cycle

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ledger"
VALID = FIXTURES / "valid_cycle.json"


def _place(tmp_path: Path, fixture: Path, name: str = "alpha-one-run-2024-06-03-120000.json") -> Path:
    dest = tmp_path / name
    shutil.copy(fixture, dest)
    return dest


# ---------------------------------------------------------------------------
# Loading fixture receipts
# ---------------------------------------------------------------------------

def test_load_valid_fixture_round_trips_the_ending_book():
    record = load_cycle_record(VALID, expected_fund="alpha-one")
    assert record.fund == "alpha-one"
    assert record.as_of == "2024-06-03"
    assert record.cash == pytest.approx(90_000.0)
    assert record.positions == {"AAPL": 100, "MSFT": -25}
    assert record.nav == pytest.approx(100_000.0)


def test_broker_for_run_seeds_from_newest_fixture(tmp_path):
    _place(tmp_path, VALID)
    broker, prior = broker_for_run("alpha-one", 50_000.0, tmp_path)

    assert isinstance(broker, PaperBroker)
    assert broker.venue == "paper"
    assert prior is not None
    assert prior.as_of == "2024-06-03"
    assert broker.cash() == pytest.approx(90_000.0)
    assert {t: p.shares for t, p in broker.positions().items()} == {
        "AAPL": 100, "MSFT": -25,
    }


def test_broker_for_run_opens_at_capital_when_no_receipt(tmp_path):
    broker, prior = broker_for_run("alpha-one", 75_000.0, tmp_path)
    assert isinstance(broker, PaperBroker)
    assert prior is None
    assert broker.cash() == pytest.approx(75_000.0)
    assert broker.positions() == {}


def test_newest_receipt_wins_by_mtime(tmp_path):
    older = _place(tmp_path, VALID, "alpha-one-run-2024-01-01-000000.json")
    newer = tmp_path / "alpha-one-run-2024-06-03-120000.json"
    newer.write_text(
        VALID.read_text().replace('"cash": 90000.0', '"cash": 88000.0')
        .replace('"nav": 100000.0', '"nav": 98000.0')
    )
    older.touch()
    # Force mtimes so the edited file is strictly newer regardless of FS resolution.
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    path = latest_run_receipt("alpha-one", tmp_path)
    assert path == newer
    broker, prior = broker_for_run("alpha-one", 100_000.0, tmp_path)
    assert prior is not None
    assert broker.cash() == pytest.approx(88_000.0)


def test_backtest_receipts_are_not_run_receipts(tmp_path):
    shutil.copy(FIXTURES / "backtest_result.json",
                tmp_path / "alpha-one-backtest-2024-06-03-120000.json")
    broker, prior = broker_for_run("alpha-one", 100_000.0, tmp_path)
    assert prior is None
    assert broker.cash() == pytest.approx(100_000.0)


# ---------------------------------------------------------------------------
# Fail loud
# ---------------------------------------------------------------------------

def test_corrupt_json_raises(tmp_path):
    path = _place(tmp_path, FIXTURES / "corrupt.json")
    with pytest.raises(ValueError, match="corrupt receipt"):
        load_cycle_record(path, expected_fund="alpha-one")
    with pytest.raises(ValueError, match="corrupt receipt"):
        broker_for_run("alpha-one", 100_000.0, tmp_path)


def test_missing_fields_raises(tmp_path):
    _place(tmp_path, FIXTURES / "missing_fields.json")
    with pytest.raises(ValueError, match="corrupt receipt"):
        broker_for_run("alpha-one", 100_000.0, tmp_path)


def test_backtest_shaped_run_file_is_incompatible(tmp_path):
    _place(tmp_path, FIXTURES / "backtest_result.json")
    with pytest.raises(ValueError, match="incompatible receipt"):
        broker_for_run("alpha-one", 100_000.0, tmp_path)


def test_fund_name_mismatch_is_incompatible(tmp_path):
    _place(tmp_path, FIXTURES / "wrong_fund.json")
    with pytest.raises(ValueError, match="does not match mandate"):
        broker_for_run("alpha-one", 100_000.0, tmp_path)


def test_newest_corrupt_receipt_does_not_fall_back(tmp_path):
    """A bad newest file fails the run — an older valid book is not used."""
    good = _place(tmp_path, VALID, "alpha-one-run-2024-01-01-000000.json")
    bad = _place(tmp_path, FIXTURES / "corrupt.json",
                 "alpha-one-run-2024-06-03-120000.json")
    os.utime(good, (1_000_000, 1_000_000))
    os.utime(bad, (2_000_000, 2_000_000))
    with pytest.raises(ValueError, match="corrupt receipt"):
        broker_for_run("alpha-one", 100_000.0, tmp_path)


# ---------------------------------------------------------------------------
# Two sequential live-clock runs
# ---------------------------------------------------------------------------

class FakeDataClient:
    def __init__(self, closes):
        self._closes = closes

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        close = self._closes.get(ticker)
        if close is None:
            return []
        return [Price(open=close, close=close, high=close, low=close,
                      volume=1000, time=f"{end_date}T00:00:00Z")]


class FakeAnalyst:
    def __init__(self, name, views=None):
        self._name = name
        self._views = views or {}

    @property
    def name(self):
        return self._name

    def predict(self, ticker, date, data_client):
        return Signal(model_name=self._name, ticker=ticker, date=date,
                      value=self._views.get(ticker, 0.0))


def _fund(capital=100_000.0):
    spec = FundSpec(
        schema_version=2,
        name="alpha-one",
        strategies=[{"name": "solo", "models": [{"name": "pead"}],
                     "blend": {"mode": "long_short"}}],
        risk={"max_position_pct": 1.0, "max_gross_exposure": 1.0},
        capital=capital,
    )
    return Fund(spec, models={"solo": [FakeAnalyst("pead", views={"AAPL": 1.0})]})


def test_second_run_starts_from_first_ending_book(tmp_path):
    """Acceptance: two sequential runs against the same mandate.

    The second process opens a new PaperBroker seeded from the first run's
    receipt — cash_before / positions / equity_before match the first
    ending book, not the mandate's capital.
    """
    fund = _fund()
    data = FakeDataClient({"AAPL": 200.0, "SPY": 100.0})

    first_broker = PaperBroker(cash=fund.spec.capital)
    first = run_cycle(fund, "2024-06-03", first_broker, data, ["AAPL"])
    save_cycle_record(first, tmp_path)

    assert first.positions == {"AAPL": 500}
    assert first.cash == pytest.approx(0.0)
    assert first.nav == pytest.approx(100_000.0)

    second_broker, prior = broker_for_run(fund.spec.name, fund.spec.capital, tmp_path)
    assert isinstance(second_broker, PaperBroker)
    assert prior is not None
    assert prior.cash == pytest.approx(first.cash)
    assert prior.positions == first.positions
    assert {t: p.shares for t, p in second_broker.positions().items()} == first.positions
    assert second_broker.cash() == pytest.approx(first.cash)

    second = run_cycle(fund, "2024-06-04", second_broker, data, ["AAPL"])
    assert second.cash_before == pytest.approx(first.cash)
    assert second.equity_before == pytest.approx(first.nav)
    assert second.positions == first.positions
    assert second.orders == []  # already at target; rebalance, not restart


def test_fixture_receipt_seeds_a_live_cycle(tmp_path):
    """A canned receipt is enough to open the next cycle on that book."""
    _place(tmp_path, VALID)
    fund = _fund()
    broker, prior = broker_for_run("alpha-one", fund.spec.capital, tmp_path)
    assert prior is not None

    record = run_cycle(
        fund, "2024-06-04", broker,
        FakeDataClient({"AAPL": 200.0, "MSFT": 400.0, "SPY": 100.0}),
        ["AAPL", "MSFT"],
    )
    assert record.cash_before == pytest.approx(90_000.0)
    # 100 AAPL @ 200 + short 25 MSFT @ 400 + 90k cash = 100k
    assert record.equity_before == pytest.approx(100_000.0)
    assert record.positions.get("AAPL") == 500  # full-conviction long, 100% cap
    assert "MSFT" not in record.positions  # the short from the receipt is closed


# ---------------------------------------------------------------------------
# Backtest path is untouched
# ---------------------------------------------------------------------------

def test_backtest_starts_from_mandate_capital_despite_receipt(tmp_path):
    """A leftover live-clock receipt must not seed backtest_fund."""
    _place(tmp_path, VALID)
    # Even if the process has a receipt on disk, backtest_fund never
    # consults the ledger — it opens a fresh SimBroker at spec.capital.
    spec = FundSpec(
        schema_version=2,
        name="alpha-one",
        strategies=[{"name": "solo", "models": [{"name": "pead"}],
                     "blend": {"mode": "long_short"}}],
        risk={"max_position_pct": 1.0, "max_gross_exposure": 1.0},
        capital=100_000.0,
        rebalance="weekly",
        benchmark="SPY",
    )
    fund = Fund(spec, models={"solo": [FakeAnalyst("pead", views={"AAPL": 1.0})]})
    series = {
        "SPY": {"2024-06-07": 100.0, "2024-06-14": 102.0},
        "AAPL": {"2024-06-07": 200.0, "2024-06-14": 210.0},
    }

    class SeriesClient:
        def __init__(self, days):
            self._days = days

        def get_prices(self, ticker, start_date, end_date, **kwargs):
            return [
                Price(open=c, close=c, high=c, low=c, volume=1000,
                      time=f"{day}T00:00:00Z")
                for day, c in sorted(self._days.get(ticker, {}).items())
                if start_date <= day <= end_date
            ]

    result = backtest_fund(fund, "2024-06-03", "2024-06-14",
                           SeriesClient(series), ["AAPL"])
    first = result.records[0]
    assert first.cash_before == pytest.approx(100_000.0)
    assert first.equity_before == pytest.approx(100_000.0)
    # Assessed at 2024-06-07, sized at the NEXT completed close (2024-06-14,
    # AAPL 210) — so 100k / 210 floors to 476 shares, not 500 at the 200 mark.
    assert first.execution_as_of == "2024-06-14"
    assert first.marks == {"AAPL": 210.0}
    assert first.positions == {"AAPL": 476}
    # The fixture book (100 AAPL / -25 MSFT / 90k cash) was not used.
    assert first.positions != {"AAPL": 100, "MSFT": -25}


# ---------------------------------------------------------------------------
# A pending proposal is not a receipt
# ---------------------------------------------------------------------------

class TestPendingProposalsDoNotShadowTheBook:
    """Found by running two paper cycles on a weekend.

    The first run had no completed session to execute against, so it wrote a
    proposal. The second then picked that proposal as "the newest receipt" and
    failed to load it as a completed record — `orders`, `fills`, `positions`,
    `cash` and `nav` are all absent from a proposal, because nothing was
    traded. Two runs in a row is not an exotic sequence.
    """

    def _proposal(self):
        """A real proposal shape, derived from the valid receipt.

        Built rather than hand-written so a field added to `DecisionRecord`
        cannot quietly make this test construct something the pipeline never
        would.
        """
        from hedge_fund.pipeline.models import DecisionRecord, PendingRunResult

        cycle = load_cycle_record(VALID, expected_fund="alpha-one")
        decision = DecisionRecord(
            fund=cycle.fund, as_of=cycle.as_of, spec=cycle.spec,
            universe=cycle.universe, marks=cycle.marks, skipped=cycle.skipped,
            strategies=cycle.strategies, target_weights=cycle.target_weights,
            clamps=cycle.clamps, final_weights=cycle.final_weights,
        )
        return PendingRunResult(
            fund=cycle.fund, as_of=cycle.as_of, proposal=decision,
            reason="No subsequent completed benchmark session is available.",
        )

    def test_a_proposal_is_not_written_on_a_receipt_path(self, tmp_path):
        path = save_cycle_record(self._proposal(), tmp_path)

        assert "proposal" in path.name
        assert latest_run_receipt("alpha-one", tmp_path) is None
        assert run_receipt_paths("alpha-one", tmp_path) == []

    def test_a_proposal_does_not_hide_an_earlier_receipt(self, tmp_path):
        receipt = _place(tmp_path, VALID)                # a real completed run
        save_cycle_record(self._proposal(), tmp_path)     # then a weekend run

        assert latest_run_receipt("alpha-one", tmp_path) == receipt
        reference = latest_reference("alpha-one", tmp_path)
        assert reference is not None and reference.source  # loadable, not the proposal

    def test_the_proposal_is_still_kept_for_the_operator(self, tmp_path):
        """It must not be dropped — it is the intent, and it is what the
        receipt cannot yet say."""
        path = save_cycle_record(self._proposal(), tmp_path)

        assert path.exists()
        import json

        assert json.loads(path.read_text())["status"] == "pending"

    def test_the_next_paper_run_still_seeds_from_the_real_receipt(self, tmp_path):
        _place(tmp_path, VALID)
        save_cycle_record(self._proposal(), tmp_path)

        broker, record = broker_for_run("alpha-one", capital=1.0, directory=tmp_path)

        assert record is not None                     # seeded, not opened at capital
        assert broker.cash() == pytest.approx(record.cash)
