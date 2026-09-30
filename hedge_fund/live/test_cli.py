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
    pead = spec.strategies[-1].models[0]
    assert PEADModel(**pead.params)._decay is True


def test_parser_defaults():
    args = build_parser().parse_args(["submit", "--dry-run"])
    assert args.command == "submit" and args.dry_run is True
    assert args.mandate == str(DEFAULT_MANDATE)
