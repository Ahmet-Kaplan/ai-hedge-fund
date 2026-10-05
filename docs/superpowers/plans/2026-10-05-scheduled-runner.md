# Scheduled Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `aihf-daily` runs the day's steps for a `RUN_MODE` and sends a Telegram summary; a GitHub Actions workflow template runs it on a schedule in a private repo with state on a `state` branch.

**Architecture:** `hedge_fund/ops/telegram.py` (sender + chat-ID finder), `hedge_fund/ops/daily.py` (mode → steps, in-process calls of the existing CLIs with stdout captured, summary, exit code), `deploy/github-actions/daily.yml` + `deploy/README.md`.

**Spec:** `docs/superpowers/specs/2026-10-05-scheduled-runner-design.md`

**Test command:** `env FINANCIAL_DATASETS_API_KEY= .venv/bin/python -m pytest <path> -q`

**Decisions made while planning:**
- `aihf-daily --find-chat-id` prints the chat IDs that have messaged the bot (Telegram `getUpdates`), so setup needs no manual API calls.
- Workflow runs `actions/cache/save` only when `state/market.db` exists (a missing path fails that action).
- The workflow template defaults `CODE_REF` to `alpaca-paper-trading` (where this code lives until merged).

---

### Task 1: Telegram sender

**Files:** Create `hedge_fund/ops/__init__.py`, `hedge_fund/ops/telegram.py`; Test `hedge_fund/ops/test_telegram.py`

- [ ] **Step 1: Failing tests**

```python
"""Telegram sender: fake session, no network."""

import requests

from hedge_fund.ops.telegram import chat_ids, send


class FakeResponse:
    def __init__(self, status=200, payload=None, text="ok"):
        self.status_code, self._payload, self.text = status, payload or {}, text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.calls, self._response, self._error = [], response, error

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        if self._error:
            raise self._error
        return self._response

    def get(self, url, timeout=None):
        self.calls.append((url, None))
        return self._response


def test_send_posts_the_message():
    session = FakeSession(FakeResponse())
    assert send("hello", token="T0K", chat_id="42", session=session) is True
    url, body = session.calls[0]
    assert url == "https://api.telegram.org/botT0K/sendMessage"
    assert body == {"chat_id": "42", "text": "hello", "disable_web_page_preview": True}


def test_missing_credentials_skip_quietly(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert send("hi", session=FakeSession()) is False
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().err


def test_errors_never_print_the_token(capsys):
    boom = requests.ConnectionError("HTTPSConnectionPool: /botSECRET123/sendMessage failed")
    assert send("hi", token="SECRET123", chat_id="1", session=FakeSession(error=boom)) is False
    err = capsys.readouterr().err
    assert "SECRET123" not in err and "<token>" in err
    assert send("hi", token="SECRET123", chat_id="1", session=FakeSession(FakeResponse(401, text="bad SECRET123"))) is False
    assert "SECRET123" not in capsys.readouterr().err


def test_long_messages_are_cut_to_telegrams_limit():
    session = FakeSession(FakeResponse())
    send("x" * 5000, token="t", chat_id="1", session=session)
    assert len(session.calls[0][1]["text"]) == 4096


def test_chat_ids_from_updates():
    updates = {"ok": True, "result": [
        {"message": {"chat": {"id": 42, "first_name": "Vivek"}}},
        {"message": {"chat": {"id": 42, "first_name": "Vivek"}}},
        {"message": {"chat": {"id": -100, "title": "Group"}}},
    ]}
    assert chat_ids(token="t", session=FakeSession(FakeResponse(payload=updates))) == [("42", "Vivek"), ("-100", "Group")]
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — `hedge_fund/ops/__init__.py`:

```python
"""Operations: the daily routine and its notifications."""
```

`hedge_fund/ops/telegram.py`:

```python
"""Send the daily summary to a Telegram chat — free, one HTTPS request.

Credentials: TELEGRAM_BOT_TOKEN (from @BotFather) and TELEGRAM_CHAT_ID
(`aihf-daily --find-chat-id` lists it after you message the bot). Missing
credentials skip the notification; a failed send never fails the run. The
token is part of the request URL, so it is redacted from every error printed.
"""

from __future__ import annotations

import os
import sys

import requests

API = "https://api.telegram.org"
MAX_LENGTH = 4096


def send(text: str, *, token: str | None = None, chat_id: str | None = None,
         session=None, timeout: float = 20.0) -> bool:
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("telegram: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to get notifications", file=sys.stderr)
        return False
    http = session or requests
    try:
        resp = http.post(f"{API}/bot{token}/sendMessage", timeout=timeout,
                         json={"chat_id": chat_id, "text": text[:MAX_LENGTH], "disable_web_page_preview": True})
    except requests.RequestException as exc:
        print(f"telegram: send failed: {_redact(str(exc), token)}", file=sys.stderr)
        return False
    if resp.status_code != 200:
        print(f"telegram: HTTP {resp.status_code}: {_redact(resp.text[:200], token)}", file=sys.stderr)
        return False
    return True


def chat_ids(*, token: str | None = None, session=None, timeout: float = 20.0) -> list[tuple[str, str]]:
    """(chat id, name) for every chat that has messaged the bot recently."""
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise ValueError("set TELEGRAM_BOT_TOKEN first")
    http = session or requests
    try:
        data = http.get(f"{API}/bot{token}/getUpdates", timeout=timeout).json()
    except requests.RequestException as exc:
        raise RuntimeError(f"telegram: {_redact(str(exc), token)}") from None
    seen: dict[str, str] = {}
    for update in data.get("result", []):
        chat = (update.get("message") or update.get("channel_post") or {}).get("chat") or {}
        if "id" in chat:
            seen.setdefault(str(chat["id"]), chat.get("title") or chat.get("first_name") or chat.get("username") or "")
    return list(seen.items())


def _redact(text: str, token: str) -> str:
    return text.replace(token, "<token>") if token else text
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** "Send notifications with a Telegram bot".

---

### Task 2: `aihf-daily`

**Files:** Create `hedge_fund/ops/daily.py`; Test `hedge_fund/ops/test_daily.py`; Modify `pyproject.toml`

- [ ] **Step 1: Failing tests**

```python
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
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `hedge_fund/ops/daily.py`:

```python
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
```

`pyproject.toml` scripts: add `aihf-daily = "hedge_fund.ops.daily:main"`; reinstall.

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** "Add aihf-daily: the day's routine with a Telegram summary".

---

### Task 3: Workflow template and setup guide

**Files:** Create `deploy/github-actions/daily.yml`, `deploy/state.gitignore`, `deploy/README.md`; Test `hedge_fund/ops/test_workflow.py`

- [ ] **Step 1: Failing test**

```python
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / "deploy" / "github-actions" / "daily.yml"


def test_workflow_shape():
    wf = yaml.safe_load(WORKFLOW.read_text())
    triggers = wf.get("on") or wf.get(True)                 # YAML 1.1 reads a bare `on` as True
    assert triggers["schedule"] == [{"cron": "0 8 * * 1-5"}]
    assert "workflow_dispatch" in triggers
    assert wf["concurrency"]["group"] == "hedge-fund-daily" and wf["concurrency"]["cancel-in-progress"] is False
    steps = wf["jobs"]["daily"]["steps"]
    run = next(s for s in steps if s.get("id") == "run")
    assert run["run"].strip() == "aihf-daily --notify"
    assert "ANTHROPIC_API_KEY" in run["env"] and "TELEGRAM_BOT_TOKEN" in run["env"]
    save = next(s for s in steps if s.get("name") == "Save state")
    assert save["if"] == "always()" and "git push" in save["run"]
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Create** `deploy/github-actions/daily.yml`:

```yaml
# Copy into the PRIVATE runner repo as .github/workflows/daily.yml (see deploy/README.md).
name: hedge-fund daily

on:
  schedule:
    - cron: "0 8 * * 1-5"          # 08:00 UTC weekdays: US pre-market all year
  workflow_dispatch:
    inputs:
      mode:
        description: "Override RUN_MODE for this run (blank = the RUN_MODE variable)"
        type: choice
        options: ["", "off", "paper", "paper+live-dry", "paper+live"]
        default: ""

concurrency:
  group: hedge-fund-daily
  cancel-in-progress: false

permissions:
  contents: write

jobs:
  daily:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    env:
      RUN_MODE: ${{ inputs.mode || vars.RUN_MODE || 'paper' }}
    steps:
      - name: Code
        uses: actions/checkout@v4
        with:
          repository: ${{ vars.CODE_REPO || 'viveksingh101994/ai-hedge-fund' }}
          ref: ${{ vars.CODE_REF || 'alpaca-paper-trading' }}
          path: code

      - name: State
        uses: actions/checkout@v4
        with:
          ref: state
          path: state

      - name: Link state to ~/.hedge-fund
        run: ln -sfn "$GITHUB_WORKSPACE/state" "$HOME/.hedge-fund"

      - name: Restore market database
        uses: actions/cache/restore@v4
        with:
          path: state/market.db
          key: market-db-${{ github.run_id }}
          restore-keys: market-db-

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install
        run: pip install uv && uv pip install --system -e code/

      - name: Run
        id: run
        env:
          APCA_API_KEY_ID: ${{ secrets.APCA_API_KEY_ID }}
          APCA_API_SECRET_KEY: ${{ secrets.APCA_API_SECRET_KEY }}
          SEC_USER_AGENT: ${{ secrets.SEC_USER_AGENT }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          ALPACA_LIVE_KEY_ID: ${{ secrets.ALPACA_LIVE_KEY_ID }}
          ALPACA_LIVE_SECRET_KEY: ${{ secrets.ALPACA_LIVE_SECRET_KEY }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: aihf-daily --notify

      - name: Save market database
        if: always() && hashFiles('state/market.db') != ''
        uses: actions/cache/save@v4
        with:
          path: state/market.db
          key: market-db-${{ github.run_id }}

      - name: Save state
        if: always()
        working-directory: state
        run: |
          git config user.name "hedge-fund-bot"
          git config user.email "hedge-fund-bot@users.noreply.github.com"
          git add -A
          git diff --cached --quiet || git commit -m "state: $(date -u +%F) $RUN_MODE"
          git push || (git pull --rebase && git push)
```

`deploy/state.gitignore` (becomes `.gitignore` on the state branch):

```
market.db
market.db-*
.env
*.pyc
```

`deploy/README.md`: the five setup steps from spec §4 with exact commands (create private repo, push workflow to `main`, create the orphan `state` branch seeded from `~/.hedge-fund/{paper,cache/llm,crypto}` plus `live.yaml` if present, add secrets and the `RUN_MODE`/`CODE_REF` variables in Settings, first manual run), and how to switch modes, pause (`RUN_MODE=off` or a `KILL` file on the state branch), and read the Telegram message.

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** "Add the GitHub Actions workflow template and setup guide"; push.

---

### Task 4: Local check, then the private repo (needs the user)

- [ ] **Step 1:** User adds `TELEGRAM_BOT_TOKEN` to the project `.env`, messages the bot, runs `.venv/bin/aihf-daily --find-chat-id`, adds `TELEGRAM_CHAT_ID`.
- [ ] **Step 2:** `.venv/bin/aihf-daily --mode paper --notify` → Telegram message arrives.
- [ ] **Step 3:** With the user's approval, create the private repo with `gh`, push the workflow and seed the `state` branch (no market.db, no `.env`). The user adds the secrets in GitHub (never through Claude), sets `RUN_MODE=paper`, presses "Run workflow", and confirms the message.
