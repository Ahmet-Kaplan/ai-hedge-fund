from hedge_fund.live.account.cli import build_parser, main
from hedge_fund.live.account.settings import load_settings


def test_parser():
    args = build_parser().parse_args(["submit", "--dry-run"])
    assert args.command == "submit" and args.dry_run


def test_init_writes_safe_settings_once(tmp_path, capsys):
    path = tmp_path / "live.yaml"
    assert main(["init", "--settings", str(path)]) == 0
    s = load_settings(path)
    assert s.confirm_live is False and s.agent_share == 0.0
    assert main(["init", "--settings", str(path)]) == 1          # never overwrites


def test_settings_live_in_the_working_directory_like_env(tmp_path, monkeypatch):
    import hedge_fund.live.account.cli as cli
    home = tmp_path / "home" / "live.yaml"
    monkeypatch.setattr(cli, "LIVE_SETTINGS_PATH", home)
    monkeypatch.chdir(tmp_path)
    assert cli.default_settings_path() == tmp_path / "live.yaml"         # nothing yet: init writes here
    home.parent.mkdir()
    home.write_text("confirm_live: false\n")
    assert cli.default_settings_path() == home                            # only the old location exists
    (tmp_path / "live.yaml").write_text("confirm_live: false\n")
    assert cli.default_settings_path() == tmp_path / "live.yaml"         # the repo copy wins
    assert main(["init"]) == 1                                             # never overwrites


def test_submit_lines_end_with_the_crypto_trend():
    from hedge_fund.live.account.cli import submit_lines
    from hedge_fund.live.account.orders import PlannedOrder
    from hedge_fund.live.account.runner import LiveSubmitResult

    shadow = LiveSubmitResult(status="nothing_to_do", session="2026-10-06", target={"SPY": 0.5}, trend="shadow",
                              exposure={"BTC/USD": 1, "ETH/USD": 0},
                              shadow=[PlannedOrder(ticker="ETH/USD", side="sell", qty=0.01)])
    assert submit_lines(shadow)[-2:] == ["crypto ma100 (shadow, not trading): BTC in · ETH OUT",
                                         "  would sell ETH/USD 0.01"]
    live = LiveSubmitResult(status="submitted", session="2026-10-06", trend="ma100", exposure={"BTC/USD": 1},
                            orders=[PlannedOrder(ticker="BTC/USD", side="buy", qty=0.0005, limit_price=60_000.0)])
    lines = submit_lines(live)
    assert "  buy  BTC/USD 0.0005 limit @ 60,000.00" in lines and lines[-1] == "crypto ma100: BTC in"
    assert not any("ma100" in line for line in submit_lines(LiveSubmitResult(status="dry_run", session="x")))
