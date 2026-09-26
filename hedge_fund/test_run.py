"""CLI paths — config preflight, the two-phase execution modes, and the paper/ledger path.

Every client here is a fake: no live APIs are touched.
"""

from __future__ import annotations

import sys
from unittest.mock import Mock

import pytest
import yaml

from hedge_fund import run
from hedge_fund.backtesting.fund import backtest_fund
from hedge_fund.brokers.paper import PaperBroker
from hedge_fund.brokers.sim import SimBroker
from hedge_fund.data.models import Price
from hedge_fund.fund import custom_strategy, FundSpec
from hedge_fund.fund.spec import Fund
from hedge_fund.ledger import latest_run_receipt
from hedge_fund.models import Signal
from hedge_fund.pipeline.models import CycleRecord


@pytest.mark.parametrize("backtest", [False, True])
@pytest.mark.parametrize("unversioned", [False, True])
def test_cli_rejects_invalid_configuration_before_clients(tmp_path, monkeypatch, capsys, backtest, unversioned):
    spec = FundSpec(schema_version=2, name="mixed", strategies=[custom_strategy(["buffett", "druckenmiller"])],
                    risk={"max_position_pct": .25, "max_gross_exposure": 1})
    data = spec.model_dump()
    if unversioned:
        data.pop("schema_version")
    else:
        data["strategies"][0]["blend"]["mode"] = "invalid"
    path = tmp_path / "fund.yaml"
    path.write_text(yaml.safe_dump(data))
    original = path.read_bytes()
    forbidden = Mock(side_effect=AssertionError("external activity"))
    monkeypatch.setattr(run, "apply_credentials", lambda: None)
    monkeypatch.setattr(run, "ensure_mandates_dir", lambda: tmp_path)
    monkeypatch.setattr(run, "Fund", forbidden)
    monkeypatch.setattr(run, "FDClient", forbidden)
    monkeypatch.setattr(run, "broker_for_run", forbidden)
    monkeypatch.setattr(sys, "argv", ["aihf", str(path), "--tickers", "AAPL"] + (["--backtest"] if backtest else []))
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2
    assert forbidden.call_count == 0
    output = capsys.readouterr()
    assert output.out == ""
    assert ("older format" if unversioned else "blend.mode") in output.err
    assert path.read_bytes() == original


@pytest.mark.parametrize("backtest", [False, True])
@pytest.mark.parametrize("mode", ["long_only", "long_short", "dollar_neutral"])
@pytest.mark.parametrize("as_of", ["2025-01-10", "2025-01-13"])
def test_cli_executes_all_modes_with_offline_clients(tmp_path, monkeypatch, capsys, mode, backtest, as_of):
    import json

    from hedge_fund.backtesting.test_fund import FakeAnalyst, FakeDataClient
    from hedge_fund.fund import Fund
    spec = FundSpec(schema_version=2, name="mixed", strategies=[custom_strategy(["buffett", "druckenmiller"])],
                    risk={"max_position_pct": .25, "max_gross_exposure": 1})
    spec.strategies[0].blend.mode = mode
    path = tmp_path / "fund.yaml"
    path.write_text(yaml.safe_dump(spec.model_dump()))
    class OfflineClient(FakeDataClient):
        def __init__(self):
            super().__init__({ticker: {"2025-01-03": 100, "2025-01-10": 100, "2025-01-13": 100} for ticker in ("AAPL", "MSFT", "SPY")})
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    def build_fund(spec, blind):
        assert blind is backtest  # backtests blind the agents' prompts; live runs don't
        return Fund(spec, models={"custom": [FakeAnalyst(name, {"AAPL": .8, "MSFT": -.6}) for name in ("buffett", "druckenmiller")]})
    monkeypatch.setattr(run, "apply_credentials", lambda: None)
    monkeypatch.setattr(run, "ensure_mandates_dir", lambda: tmp_path)
    monkeypatch.setattr(run, "Fund", build_fund)
    monkeypatch.setattr(run, "FDClient", OfflineClient)
    monkeypatch.setattr(run, "CachedDataClient", lambda client: client)
    monkeypatch.setattr(sys, "argv", ["aihf", str(path), "--tickers", "AAPL,MSFT", "--date", as_of] +
                        (["--backtest", "--start", "2025-01-03"] if backtest else []))
    run.main()
    output = capsys.readouterr()
    result = json.loads(output.out)
    if not backtest and as_of == "2025-01-13":
        assert result["status"] == "pending"
        assert result["proposal"]["final_weights"]["AAPL"] > 0
        assert "Pending" in output.err
        assert "NAV" not in output.err
        return
    assert "executed cycles" in output.err if backtest else "refreshed" in output.err
    records = result["records"] if backtest else [result]
    for record in records:
        assert record["positions"]["AAPL"] > 0
        assert (record["positions"].get("MSFT", 0) < 0) == (mode != "long_only")
        if mode == "dollar_neutral":
            assert sum(record["final_weights"].values()) == pytest.approx(0)


class FakeDataClient:
    def __init__(self, closes):
        self._closes = closes

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

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


def _mandate(path, name="paper-desk"):
    path.write_text(
        f"""schema_version: 2
name: {name}
strategies:
  - name: solo
    models:
      - name: pead
    blend:
      mode: long_short
risk:
  max_position_pct: 1.0
  max_gross_exposure: 1.0
capital: 100000
"""
    )
    return path


def _fake_fund(spec, blind=False):
    # run.py passes blind= for backtests; the fake ignores it.
    assert blind is False
    return Fund(spec, models={"solo": [FakeAnalyst("pead", views={"AAPL": 1.0})]})


def _patch_cli(monkeypatch, tmp_path, closes):
    from hedge_fund import run
    from hedge_fund.tui import keys

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(keys, "ENV_PATH", tmp_path / "saved.env")
    monkeypatch.setattr(run, "ensure_mandates_dir", lambda: tmp_path)
    monkeypatch.setattr(run, "FDClient", lambda: FakeDataClient(closes))
    monkeypatch.setattr(run, "CachedDataClient", lambda raw: raw)
    monkeypatch.setattr(run, "Fund", _fake_fund)
    return run


def test_paper_flag_writes_receipt_and_next_run_seeds(tmp_path, monkeypatch, capsys):
    """Acceptance: --paper writes a CycleRecord; the next process seeds from it."""
    run = _patch_cli(monkeypatch, tmp_path, {"AAPL": 200.0, "SPY": 100.0})
    mandate = _mandate(tmp_path / "fund.yaml")
    output = tmp_path / "record.json"
    argv = [
        "aihf", str(mandate), "--tickers", "AAPL", "--paper",
        "--date", "2024-06-03", "--out", str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run.main()
    first = CycleRecord.model_validate_json(output.read_text())
    printed = CycleRecord.model_validate_json(capsys.readouterr().out)
    assert printed == first
    assert first.positions == {"AAPL": 500}
    assert first.cash == pytest.approx(0.0)
    assert first.nav == pytest.approx(100_000.0)
    receipts = list(tmp_path.glob("paper-desk-run-*.json"))
    assert len(receipts) == 1

    run.main()
    second = CycleRecord.model_validate_json(output.read_text())
    err = capsys.readouterr().err
    assert "carrying book from 2024-06-03" in err
    assert "paper venue" in err
    assert second.cash_before == pytest.approx(first.cash)
    assert second.equity_before == pytest.approx(first.nav)
    assert second.positions == first.positions
    assert second.orders == []
    # Same-second reruns may overwrite the stamp; the newest receipt is the book.
    assert latest_run_receipt("paper-desk", tmp_path) is not None


def test_default_live_clock_path_is_paper(tmp_path, monkeypatch, capsys):
    """One cycle without --backtest is the paper path (live clock + PaperBroker)."""
    run = _patch_cli(monkeypatch, tmp_path, {"AAPL": 200.0, "SPY": 100.0})
    seen: list[object] = []
    real = run.broker_for_run

    def wrapped(*args, **kwargs):
        broker, prior = real(*args, **kwargs)
        seen.append(broker)
        return broker, prior

    monkeypatch.setattr(run, "broker_for_run", wrapped)
    mandate = _mandate(tmp_path / "fund.yaml")
    monkeypatch.setattr(
        sys, "argv",
        ["aihf", str(mandate), "--tickers", "AAPL", "--date", "2024-06-03"],
    )
    run.main()
    assert len(seen) == 1
    assert isinstance(seen[0], PaperBroker)
    assert seen[0].venue == "paper"
    assert "paper venue" in capsys.readouterr().err


def test_paper_and_backtest_are_exclusive(tmp_path, monkeypatch):
    from hedge_fund import run

    mandate = _mandate(tmp_path / "fund.yaml")
    monkeypatch.setattr(
        sys, "argv",
        ["aihf", str(mandate), "--tickers", "AAPL", "--paper", "--backtest"],
    )
    with pytest.raises(SystemExit):
        run.main()


def test_allocator_flag_selects_equal_weight(tmp_path, monkeypatch, capsys):
    """--allocator equal_weight is selectable and recorded on the receipt."""
    run = _patch_cli(monkeypatch, tmp_path, {"AAPL": 200.0, "SPY": 100.0})
    mandate = tmp_path / "fund.yaml"
    mandate.write_text(
        """schema_version: 2
name: desk
strategies:
  - name: s1
    weight: 3.0
    models:
      - name: pead
    blend:
      mode: long_short
  - name: s2
    weight: 1.0
    models:
      - name: pead
    blend:
      mode: long_short
risk:
  max_position_pct: 1.0
  max_gross_exposure: 1.0
capital: 100000
"""
    )

    def two_sleeves(spec, blind=False):
        assert blind is False
        return Fund(spec, models={
            "s1": [FakeAnalyst("pead", views={"AAPL": 1.0})],
            "s2": [FakeAnalyst("pead", views={"AAPL": 1.0})],
        })

    monkeypatch.setattr(run, "Fund", two_sleeves)
    output = tmp_path / "record.json"
    monkeypatch.setattr(
        sys, "argv",
        ["aihf", str(mandate), "--tickers", "AAPL", "--date", "2024-06-03",
         "--allocator", "equal_weight", "--out", str(output)],
    )
    run.main()
    record = CycleRecord.model_validate_json(output.read_text())
    assert record.spec.allocator == "equal_weight"
    assert {sr.name: sr.slice for sr in record.strategies} == {
        "s1": 0.5, "s2": 0.5,
    }

    monkeypatch.setattr(
        sys, "argv",
        ["aihf", str(mandate), "--tickers", "AAPL", "--date", "2024-06-03",
         "--out", str(output)],
    )
    run.main()
    static = CycleRecord.model_validate_json(output.read_text())
    assert static.spec.allocator == "static"
    assert {sr.name: sr.slice for sr in static.strategies} == {
        "s1": 0.75, "s2": 0.25,
    }


def test_unknown_allocator_flag_is_rejected(tmp_path, monkeypatch):
    from hedge_fund import run

    mandate = _mandate(tmp_path / "fund.yaml")
    monkeypatch.setattr(
        sys, "argv",
        ["aihf", str(mandate), "--tickers", "AAPL", "--allocator", "risk_parity"],
    )
    with pytest.raises(SystemExit):
        run.main()


def test_backtest_still_constructs_sim_broker(monkeypatch):
    constructed: list[object] = []
    real = SimBroker

    def wrap(cash, positions=None):
        broker = real(cash=cash, positions=positions)
        constructed.append(broker)
        return broker

    monkeypatch.setattr("hedge_fund.backtesting.fund.SimBroker", wrap)
    spec = FundSpec(
        schema_version=2,
        name="bt",
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
        def get_prices(self, ticker, start_date, end_date, **kwargs):
            return [
                Price(open=c, close=c, high=c, low=c, volume=1000,
                      time=f"{day}T00:00:00Z")
                for day, c in sorted(series.get(ticker, {}).items())
                if start_date <= day <= end_date
            ]

    result = backtest_fund(fund, "2024-06-03", "2024-06-14",
                           SeriesClient(), ["AAPL"])
    assert constructed
    assert all(isinstance(b, SimBroker) for b in constructed)
    assert all(getattr(b, "venue", "sim") == "sim" for b in constructed)
    assert result.records[0].cash_before == pytest.approx(100_000.0)
