"""aihf-daily — the day's routine for a RUN_MODE, summarized for Telegram.

    aihf-daily [--mode off|paper|paper+live-dry|paper+live] [--notify]
    aihf-daily --find-chat-id

The mode defaults to $RUN_MODE, else "paper". Every step runs even if an
earlier one failed; the exit code is 1 if any did. Live safety stays inside
aihf-live: "paper+live" only invokes it, and it still refuses without live
keys, confirm_live: true and a completed dry run.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from datetime import date
from typing import Callable

from pydantic import BaseModel

from hedge_fund.ops.telegram import chat_ids, send

SUMMARY_LIMIT = 3500
LINES_PER_STEP = 8


def _paper(argv: list[str]) -> int:
    from hedge_fund.live.cli import main
    return main(argv)


def _live(argv: list[str]) -> int:
    from hedge_fund.live.account.cli import main
    return main(argv)


Step = tuple[str, Callable[[list[str]], int], list[str]]
_PAPER: list[Step] = [("paper reconcile", _paper, ["reconcile"]), ("paper submit", _paper, ["submit"])]
MODES: dict[str, list[Step]] = {
    "off": [],
    "paper": _PAPER,
    "paper+live-dry": _PAPER + [("live run (dry run)", _live, ["run", "--dry-run"])],
    "paper+live": _PAPER + [("live run", _live, ["run"])],
}


class StepResult(BaseModel):
    name: str
    code: int
    output: str


def run_steps(steps: list[Step]) -> list[StepResult]:
    results = []
    for name, fn, argv in steps:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                code = fn(list(argv))
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        except Exception as exc:   # a step's crash is reported, and the next step still runs
            code = 1
            buf.write(f"{type(exc).__name__}: {exc}\n")
        results.append(StepResult(name=name, code=code or 0, output=buf.getvalue()))
        sys.stdout.write(f"--- {name} (exit {code or 0})\n{buf.getvalue()}")
    return results


def summarize(mode: str, results: list[StepResult], *, today: date, run_url: str | None) -> str:
    ok = all(r.code == 0 for r in results)
    parts = [f"{'✅' if ok else '❌'} {today:%a %d %b} · mode {mode}"]
    if not results:
        parts.append("Paused (RUN_MODE=off)")
    for r in results:
        lines = [line for line in r.output.strip().splitlines() if line.strip()][-LINES_PER_STEP:]
        parts.append(f"\n{'✓' if r.code == 0 else '✗'} {r.name}" + ("\n" + "\n".join(lines) if lines else ""))
    tail = f"\n\nRun: {run_url}" if run_url else ""
    text = "\n".join(parts)
    budget = SUMMARY_LIMIT - len(tail)
    if len(text) > budget:
        text = text[: budget - 1] + "…"
    return text + tail


def _run_url() -> str | None:
    server, repo, run = (os.environ.get(k) for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    return f"{server}/{repo}/actions/runs/{run}" if server and repo and run else None


def main(argv: list[str] | None = None) -> int:
    from hedge_fund.tui.keys import apply_credentials

    apply_credentials()
    parser = argparse.ArgumentParser(prog="aihf-daily", description="Run the day's routine and summarize it.")
    parser.add_argument("--mode", choices=sorted(MODES), default=None)
    parser.add_argument("--notify", action="store_true", help="send the summary to Telegram")
    parser.add_argument("--find-chat-id", action="store_true", help="list chats that messaged the bot")
    args = parser.parse_args(argv)
    if args.find_chat_id:
        found = chat_ids()
        print("\n".join(f"{cid}  {name}" for cid, name in found) or "no chats yet: send your bot a message, then retry")
        return 0
    mode = args.mode or os.environ.get("RUN_MODE") or "paper"
    if mode not in MODES:
        parser.error(f"RUN_MODE must be one of {', '.join(sorted(MODES))}, not {mode!r}")
    results = run_steps(MODES[mode])
    text = summarize(mode, results, today=date.today(), run_url=_run_url())
    print(text)
    if args.notify:
        send(text)
    return 0 if all(r.code == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
