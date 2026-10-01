import pytest

from hedge_fund.live.account.ledger import LiveLedger
from hedge_fund.live.account.review import apply_review, propose_review
from hedge_fund.live.account.settings import LiveSettings, load_settings, save_settings
from hedge_fund.live.ledger import Ledger, NavRow


def paper_with(tmp_path, n_rebalances, fund_end, spy_end):
    paper = Ledger(tmp_path / "paper")
    days = [f"2026-{10 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n_rebalances)]
    for d in days:
        paper.write_plan(d, {"plan": {}})
    paper.upsert_nav(NavRow(date=days[0], equity=100.0, cash=0, long_exposure=0, short_exposure=0, gross=0, benchmark_close=100.0))
    paper.upsert_nav(NavRow(date=days[-1], equity=fund_end, cash=0, long_exposure=0, short_exposure=0, gross=0, benchmark_close=spy_end))
    return paper


def test_too_early(tmp_path):
    with pytest.raises(ValueError, match="11 of 12"):
        propose_review(paper_with(tmp_path, 11, 105, 100), LiveLedger(tmp_path / "live"), LiveSettings())


def test_first_pass_grants_twenty_percent(tmp_path):
    r = propose_review(paper_with(tmp_path, 12, 105, 102), LiveLedger(tmp_path / "live"), LiveSettings())
    assert r.passed and r.paper_excess == pytest.approx(0.03) and (r.old_share, r.new_share) == (0.0, 0.2)


def test_pass_steps_up_to_cap_and_fail_steps_down(tmp_path):
    paper = paper_with(tmp_path, 12, 105, 102)
    live = LiveLedger(tmp_path / "live")
    assert propose_review(paper, live, LiveSettings(agent_share=0.5)).new_share == 0.5
    assert propose_review(paper, live, LiveSettings(agent_share=0.3)).new_share == pytest.approx(0.4)
    losing = paper_with(tmp_path / "b", 12, 101, 102)
    assert propose_review(losing, live, LiveSettings(agent_share=0.3)).new_share == pytest.approx(0.2)
    assert propose_review(losing, live, LiveSettings(agent_share=0.0)).new_share == 0.0


def test_apply_writes_settings_logs_and_clears_halt(tmp_path):
    path = tmp_path / "live.yaml"
    save_settings(LiveSettings(confirm_live=True), path)
    live = LiveLedger(tmp_path / "live")
    live.halt_satellite("x")
    r = propose_review(paper_with(tmp_path, 12, 105, 102), live, load_settings(path))
    apply_review(r, load_settings(path), path, live, today="2026-12-20")
    assert load_settings(path).agent_share == 0.2
    assert live.last_review_date() == "2026-12-20"
    assert not live.satellite_halted()
