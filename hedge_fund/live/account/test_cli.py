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
