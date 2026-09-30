from hedge_fund.fund.spec import load_spec
from hedge_fund.live.cli import DEFAULT_MANDATE, DEFAULT_UNIVERSE, build_parser, load_universe
from hedge_fund.signals import PEADModel


def test_packaged_universe():
    universe = load_universe(DEFAULT_UNIVERSE)
    assert len(universe) == 32
    assert universe[:2] == ["AAPL", "MSFT"] and "BRK.B" in universe


def test_universe_parsing(tmp_path):
    path = tmp_path / "u.txt"
    path.write_text("# header\naapl msft  # comment\n\nNVDA\naapl\n")
    assert load_universe(path) == ["AAPL", "MSFT", "NVDA"]


def test_packaged_paper_mandate_is_valid():
    spec = load_spec(DEFAULT_MANDATE)
    assert spec.name == "paper-fund"
    assert [s.name for s in spec.strategies] == ["fundamental-ls", "deep-value", "inflections", "earnings-drift"]
    assert spec.risk.max_position_pct == 0.10 and spec.costs.commission_bps == 5
    assert all(s.blend.max_name_weight == 0.10 for s in spec.strategies)
    assert spec.costs.min_trade_pct == 0.005
    assert spec.equitize_idle is True
    pead = spec.strategies[-1].models[0]
    assert PEADModel(**pead.params)._decay is True


def test_parser_defaults():
    args = build_parser().parse_args(["submit", "--dry-run"])
    assert args.command == "submit" and args.dry_run is True
    assert args.mandate == str(DEFAULT_MANDATE)


def test_data_sync_prints_coverage(monkeypatch, capsys, tmp_path):
    from contextlib import contextmanager

    from hedge_fund.data.free import Coverage, FreeDataClient
    from hedge_fund.live import cli

    class Fake(FreeDataClient):
        def __init__(self):
            pass

        def coverage(self, tickers, as_of, prices_only=frozenset(), refresh=False):
            return [Coverage("AAPL", "2016-01-04", "2026-09-29", 12, "2026-06-27", 8),
                    Coverage("XOM", "2016-01-04", "2026-09-29", 1, "2026-06-30", 0, warning="lagging"),
                    Coverage("SPY", "2016-01-04", "2026-09-29")]

    @contextmanager
    def fake_open():
        yield Fake()

    universe = tmp_path / "u.list"
    universe.write_text("AAPL XOM\n")
    monkeypatch.setattr(cli, "open_data_client", fake_open)
    monkeypatch.setattr(cli, "apply_credentials", lambda: None)
    monkeypatch.setattr(cli.Ledger, "for_fund", classmethod(lambda cls, name: cli.Ledger(tmp_path / "ledger")))
    assert cli.main(["data-sync", "--universe", str(universe)]) == 0
    out = capsys.readouterr().out
    assert "AAPL" in out and "note: lagging" in out
    assert "agents will abstain): XOM" in out
