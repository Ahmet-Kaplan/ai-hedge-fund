"""Run the paper fund on a schedule with macOS LaunchAgents.

Two jobs, weekdays, in New York time: reconcile at 09:00 (records the
previous session) and submit at 10:00 (does nothing unless it is a rebalance
day). launchd fires in *local* time, so New York times are converted when the
jobs are installed — reinstall after a daylight-saving change on either side.
The Mac must be awake at those times.
"""

from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
from datetime import date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Callable

from hedge_fund.data.sessions import NEW_YORK
from hedge_fund.paths import USER_DIR

LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
LABEL_PREFIX = "ai.hedgefund.paper."
JOBS = {"reconcile": time(9, 0), "submit": time(10, 0)}   # New York times, run in this order


def calendar_intervals(ny_time: time, *, local_tz: tzinfo | None = None, week_of: date | None = None) -> list[dict[str, int]]:
    """launchd StartCalendarInterval entries for `ny_time` on each New York weekday."""
    week_of = week_of or datetime.now(NEW_YORK).date()
    monday = week_of - timedelta(days=week_of.weekday())
    intervals = []
    for offset in range(5):
        ny = datetime.combine(monday + timedelta(days=offset), ny_time, NEW_YORK)
        local = ny.astimezone(local_tz) if local_tz else ny.astimezone()
        intervals.append({"Weekday": local.isoweekday() % 7, "Hour": local.hour, "Minute": local.minute})
    return intervals


def install_schedule(
    *, mandate: Path, universe: Path, log_dir: Path, python: str = sys.executable,
    agents_dir: Path = LAUNCH_AGENTS, run: Callable = subprocess.run,
) -> list[Path]:
    agents_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    USER_DIR.mkdir(parents=True, exist_ok=True)
    domain = f"gui/{os.getuid()}"
    paths = []
    for step, ny_time in JOBS.items():
        path = agents_dir / f"{LABEL_PREFIX}{step}.plist"
        run(["launchctl", "bootout", domain, str(path)], check=False, capture_output=True)
        path.write_bytes(plistlib.dumps({
            "Label": LABEL_PREFIX + step,
            "ProgramArguments": [python, "-m", "hedge_fund.live.cli", step,
                                 "--mandate", str(mandate), "--universe", str(universe)],
            "StartCalendarInterval": calendar_intervals(ny_time),
            "WorkingDirectory": str(USER_DIR),
            "StandardOutPath": str(log_dir / f"launchd-{step}.out.log"),
            "StandardErrorPath": str(log_dir / f"launchd-{step}.err.log"),
        }))
        run(["launchctl", "bootstrap", domain, str(path)], check=True)
        paths.append(path)
    return paths


def uninstall_schedule(*, agents_dir: Path = LAUNCH_AGENTS, run: Callable = subprocess.run) -> list[Path]:
    domain = f"gui/{os.getuid()}"
    removed = []
    for step in JOBS:
        path = agents_dir / f"{LABEL_PREFIX}{step}.plist"
        if path.exists():
            run(["launchctl", "bootout", domain, str(path)], check=False, capture_output=True)
            path.unlink()
            removed.append(path)
    return removed


def schedule_installed(agents_dir: Path = LAUNCH_AGENTS) -> bool:
    return all((agents_dir / f"{LABEL_PREFIX}{step}.plist").exists() for step in JOBS)


def notify(title: str, message: str) -> None:
    """Best-effort macOS notification; never raises."""
    script = f"display notification {json.dumps(message[:200], ensure_ascii=False)} with title {json.dumps(title, ensure_ascii=False)}"
    try:
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True)
    except OSError:
        pass
