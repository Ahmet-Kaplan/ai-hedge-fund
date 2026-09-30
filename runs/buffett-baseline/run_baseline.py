"""Buffett baseline runner with an external Anthropic cost guard.

Runs the same path as `aihf configs/buffett-baseline.yaml --backtest
--universe-schedule ...` (blind Fund, make_data_client, backtest_fund) but
wraps ChatLLM.complete to meter token usage and stop before the budget is hit.
No repo code is modified; the wrapper lives only in this process.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

# Data sources: Tiingo + SEC EDGAR only. Financial Datasets is removed from
# this process entirely, so no code path can reach it.
for var in ("FINANCIAL_DATASETS_API_KEY", "HEDGE_FUND_DATA_PROVIDER", "HEDGE_FUND_DATA_SUPPLEMENT"):
    os.environ.pop(var, None)
MODEL = "claude-opus-5-5"
os.environ["HEDGE_FUND_LLM_MODEL"] = MODEL

BUDGET = float(os.environ.get("BASELINE_BUDGET_USD", "18"))
IN_PER_TOK, OUT_PER_TOK = 4.00 / 1e6, 20.00 / 1e6          # Opus 5.5 $/token
WORST_CALL = 0.15                                          # reserve per call (max_tokens=4096 -> <= $0.082 output)
PROBE_CALLS = int(os.environ.get("BASELINE_PROBE_CALLS", "15"))
MAX_CONSECUTIVE_ERRORS, MAX_TOTAL_ERRORS = 3, 10

START, END = "2016-07-01", "2026-06-30"
OUT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/buffett-baseline")
SCHEDULE = OUT_DIR / "universe_top10.json"
MANDATE = Path("configs/buffett-baseline.yaml")


class RunStop(BaseException):
    """BaseException so LLMAgent's `except Exception` cannot swallow it."""


from hedge_fund.backtesting import backtest_fund  # noqa: E402
from hedge_fund.backtesting.fund import build_schedule  # noqa: E402
from hedge_fund.data import make_data_client  # noqa: E402
from hedge_fund.fund import Fund, load_spec  # noqa: E402
from hedge_fund.llm import cache as cache_mod  # noqa: E402
from hedge_fund.llm import client as client_mod  # noqa: E402
from hedge_fund.universe.builder import load_schedule  # noqa: E402

lock = threading.Lock()
S = {"llm_calls": 0, "cache_hits": 0, "cache_lookups": 0, "input_tokens": 0, "output_tokens": 0,
     "cost_usd": 0.0, "errors": 0, "consecutive_errors": 0, "expected_calls": None,
     "stopped": None, "cycles_done": 0}
log_file = open(OUT_DIR / "run.log", "a", buffering=1)


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, file=sys.stderr, flush=True)
    log_file.write(line + "\n")


_orig_get = cache_mod.PromptCache.get


def metered_get(self, key):
    rec = _orig_get(self, key)
    with lock:
        S["cache_lookups"] += 1
        if rec is not None and "parsed" in rec:
            S["cache_hits"] += 1
    return rec


cache_mod.PromptCache.get = metered_get


def metered_complete(self, system: str, user: str) -> str:
    with lock:
        if S["cost_usd"] + WORST_CALL > BUDGET:
            S["stopped"] = f"budget guard: ${S['cost_usd']:.2f} spent, next call could exceed ${BUDGET:.2f}"
            raise RunStop(S["stopped"])
    try:
        msg = self._chat.invoke([("system", system), ("human", user)])
    except Exception as exc:
        with lock:
            S["errors"] += 1
            S["consecutive_errors"] += 1
            log(f"LLM API error #{S['errors']}: {type(exc).__name__}: {str(exc)[:200]}")
            if S["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS or S["errors"] >= MAX_TOTAL_ERRORS:
                S["stopped"] = f"repeated Anthropic API errors ({S['errors']} total, {S['consecutive_errors']} consecutive)"
                raise RunStop(S["stopped"]) from exc
        raise
    usage = getattr(msg, "usage_metadata", None) or {}
    tin, tout = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    if not tout:
        with lock:
            S["stopped"] = "Anthropic response carried no usage metadata; cannot meter cost"
        raise RunStop(S["stopped"])
    with lock:
        S["consecutive_errors"] = 0
        S["llm_calls"] += 1
        S["input_tokens"] += tin
        S["output_tokens"] += tout
        S["cost_usd"] += tin * IN_PER_TOK + tout * OUT_PER_TOK
        n = S["llm_calls"]
        if n == PROBE_CALLS and S["expected_calls"]:
            per = S["cost_usd"] / n
            remaining = max(S["expected_calls"] - S["cache_lookups"], 0)
            projected = S["cost_usd"] + per * remaining
            log(f"PROBE: {n} calls ${S['cost_usd']:.3f} (${per:.4f}/call), ~{remaining} calls left "
                f"-> projected ${projected:.2f}")
            S["projected_usd"] = round(projected, 2)
            if projected > BUDGET:
                S["stopped"] = (f"projection guard: ${per:.4f}/call x ~{S['expected_calls']} calls "
                                f"projects ${projected:.2f} > ${BUDGET:.2f}")
                raise RunStop(S["stopped"])
    return client_mod._flatten(msg.content)


client_mod.ChatLLM.complete = metered_complete


def on_cycle(i, n, record):
    with lock:
        S["cycles_done"] = i + 1
    log(f"cycle {i + 1}/{n} executed {record.execution_as_of} · NAV ${record.nav:,.0f} · "
        f"LLM calls {S['llm_calls']} · cache hits {S['cache_hits']} · ${S['cost_usd']:.2f}")


def main() -> int:
    spec = load_spec(MANDATE)
    fund = Fund(spec, blind=True)
    pit = load_schedule(SCHEDULE)
    status_path = OUT_DIR / "run_status.json"
    t0 = time.time()
    result = None
    with make_data_client() as fd:
        sched = build_schedule(fd, spec.benchmark, START, END, spec.rebalance)
        S["expected_calls"] = sum(len(pit.members_on(d)) for d in sched.execution_dates)
        log(f"{len(sched.execution_dates)} rebalance dates, ~{S['expected_calls']} agent assessments; budget ${BUDGET}")
        try:
            result = backtest_fund(fund, START, END, fd, pit, on_cycle=on_cycle)
        except RunStop as exc:
            log(f"STOPPED SAFELY: {exc}")
        finally:
            S["requests"] = fd.request_counts()
            S["elapsed_s"] = round(time.time() - t0)
            S["cost_usd"] = round(S["cost_usd"], 4)
            status_path.write_text(json.dumps(S, indent=2))
    if result is None:
        return 4
    (OUT_DIR / "result.json").write_text(result.model_dump_json(indent=2))
    log(f"wrote {OUT_DIR / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
