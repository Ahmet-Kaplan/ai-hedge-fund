import plistlib
from datetime import date, time
from zoneinfo import ZoneInfo

from hedge_fund.live.launchd import (
    calendar_intervals, install_schedule, schedule_installed, uninstall_schedule,
)

WEEK = date(2026, 9, 28)   # a Monday, US daylight time


def test_new_york_time_converted_to_local_weekday_and_hour():
    kolkata = calendar_intervals(time(10, 0), local_tz=ZoneInfo("Asia/Kolkata"), week_of=WEEK)
    assert kolkata[0] == {"Weekday": 1, "Hour": 19, "Minute": 30}
    assert len(kolkata) == 5
    # 10:00 EDT is 03:00 the next day in Auckland (NZDT): Monday NY → Tuesday local.
    auckland = calendar_intervals(time(10, 0), local_tz=ZoneInfo("Pacific/Auckland"), week_of=WEEK)
    assert auckland[0] == {"Weekday": 2, "Hour": 3, "Minute": 0}
    assert auckland[-1]["Weekday"] == 6


def test_install_writes_both_agents_and_bootstraps(tmp_path):
    calls = []
    paths = install_schedule(
        mandate=tmp_path / "m.yaml", universe=tmp_path / "u.txt", log_dir=tmp_path / "logs",
        python="/venv/bin/python", agents_dir=tmp_path / "agents",
        run=lambda cmd, **kw: calls.append(cmd),
    )
    assert [p.name for p in paths] == ["ai.hedgefund.paper.reconcile.plist", "ai.hedgefund.paper.submit.plist"]
    plist = plistlib.loads(paths[1].read_bytes())
    assert plist["ProgramArguments"][:4] == ["/venv/bin/python", "-m", "hedge_fund.live.cli", "submit"]
    assert len(plist["StartCalendarInterval"]) == 5
    assert any(cmd[1] == "bootstrap" for cmd in calls)
    assert schedule_installed(tmp_path / "agents")
    removed = uninstall_schedule(agents_dir=tmp_path / "agents", run=lambda cmd, **kw: None)
    assert len(removed) == 2 and not schedule_installed(tmp_path / "agents")
