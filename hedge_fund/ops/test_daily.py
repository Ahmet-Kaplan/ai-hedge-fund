from datetime import date

import pytest

from hedge_fund.ops import daily
from hedge_fund.ops.daily import MODES, run_steps, summarize


def step(name, code=0, out="", raise_=None):
    def fn(argv):
        print(out)
        if raise_:
            raise raise_
        return code
    return (name, fn, [])


def test_modes():
    assert [s[0] for s in MODES["paper"]] == ["paper reconcile", "paper submit"]
    assert [s[0] for s in MODES["paper+live-dry"]][-1] == "live run (dry run)"
    assert MODES["paper+live-dry"][-1][2] == ["run", "--dry-run"]
    assert MODES["paper+live"][-1][2] == ["run"]
    assert MODES["off"] == []


def test_every_step_runs_after_a_failure():
    results = run_steps([step("a", raise_=RuntimeError("boom")), step("b", out="fine"),
                         ("c", lambda argv: (_ for _ in ()).throw(SystemExit(2)), [])])
    assert [(r.name, r.code) for r in results] == [("a", 1), ("b", 0), ("c", 2)]
    assert "RuntimeError: boom" in results[0].output and "fine" in results[1].output


def test_summary_header_lines_and_cap():
    results = run_steps([step("paper submit", out="\n".join(f"line {i}" for i in range(20)))])
    text = summarize("paper", results, today=date(2026, 10, 5), run_url="https://x/runs/1")
    assert text.startswith("✅ Mon 05 Oct · mode paper")
    assert "line 19" in text and "line 11" not in text          # last 8 lines only
    assert text.endswith("Run: https://x/runs/1")
    big = run_steps([step(f"s{i}", out="y" * 900) for i in range(10)])
    assert len(summarize("paper", big, today=date(2026, 10, 5), run_url=None)) <= 3500


def test_failure_header_and_paused():
    failed = run_steps([step("live run", code=1, out="needs_dry_run")])
    assert summarize("paper+live", failed, today=date(2026, 10, 5), run_url=None).startswith("❌")
    assert "Paused" in summarize("off", [], today=date(2026, 10, 5), run_url=None)


def test_main_notifies_and_exits_by_result(monkeypatch):
    sent = []
    monkeypatch.setattr(daily, "send", lambda text: sent.append(text) or True)
    monkeypatch.setitem(MODES, "paper", [step("ok")])
    assert daily.main(["--mode", "paper", "--notify"]) == 0 and sent
    monkeypatch.setitem(MODES, "paper", [step("bad", code=1)])
    assert daily.main(["--mode", "paper"]) == 1


def test_mode_from_environment(monkeypatch):
    monkeypatch.setenv("RUN_MODE", "off")
    assert daily.main([]) == 0
    monkeypatch.setenv("RUN_MODE", "yolo")
    with pytest.raises(SystemExit):
        daily.main([])
