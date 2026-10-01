"""Compare a cheaper or local model with the agents' reference model.

Every LLM decision is saved with its exact prompt (hedge_fund/llm/cache.py),
so the reference model's answers are a ready-made answer key: replay a sample
of the same prompts through a candidate and measure how often it gives a
valid answer, how often it reaches the same verdict, how often it says the
opposite, and how closely its signed conviction tracks the reference.

    python -m hedge_fund.llm.compare --models gemini-3.8-flash,qwen3:8b --n 100

Agreement with the reference measures similarity, not skill: a candidate
that disagrees could be right. It answers "can this replace the reference
without changing the fund's decisions much?", the question that matters when
the reason to switch is cost.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from hedge_fund.llm.cache import DEFAULT_CACHE_DIR
from hedge_fund.llm.client import LLMClient, make_llm
from hedge_fund.paths import USER_DIR

_SIGN = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
RESULTS_DIR = USER_DIR / "model-compare"


def load_references(cache_dir: Path | str, reference_model: str) -> list[dict]:
    """Saved decisions by the reference model that parsed into a valid answer."""
    refs = []
    for path in sorted(Path(cache_dir).rglob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        parsed = rec.get("parsed")
        if (rec.get("model") == reference_model and isinstance(parsed, dict)
                and parsed.get("signal") in _SIGN and rec.get("system") and rec.get("user")):
            refs.append(rec)
    return refs


def sample(refs: list[dict], n: int, seed: int = 7) -> list[dict]:
    """Up to n references, drawn round-robin across (agent, verdict) groups so
    rare verdicts (an agent's few bearish calls) are represented."""
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in refs:
        groups[(rec["agent"], rec["parsed"]["signal"])].append(rec)
    for members in groups.values():
        rng.shuffle(members)
    keys = sorted(groups)
    picked: list[dict] = []
    while len(picked) < n and any(groups[k] for k in keys):
        for key in keys:
            if groups[key] and len(picked) < n:
                picked.append(groups[key].pop())
    return picked


def run_model(client: LLMClient, refs: list[dict], progress=None) -> list[dict]:
    """The candidate's answer to each reference prompt, with timing and any error."""
    from hedge_fund.signals.llm_agent import parse_view

    results = []
    for i, ref in enumerate(refs):
        started = time.monotonic()
        answer, error, chars = None, None, 0
        try:
            text = client.complete(ref["system"], ref["user"])
            chars = len(text)
            answer = parse_view(text)
        except Exception as exc:   # a failed answer is a data point, not a crash
            error = f"{type(exc).__name__}: {exc}"[:200]
        results.append({"agent": ref["agent"], "ticker": ref.get("ticker"), "reference": ref["parsed"],
                        "candidate": answer, "error": error, "seconds": time.monotonic() - started,
                        "response_chars": chars, "prompt_chars": len(ref["system"]) + len(ref["user"])})
        if progress:
            progress(i + 1, len(refs), results[-1])
    return results


def _value(answer: dict) -> float:
    return _SIGN[answer["signal"]] * answer["confidence"] / 100


def score(results: list[dict]) -> dict:
    valid = [r for r in results if r["candidate"] is not None]
    same = [r for r in valid if r["candidate"]["signal"] == r["reference"]["signal"]]
    opposite = [r for r in valid if _SIGN[r["candidate"]["signal"]] * _SIGN[r["reference"]["signal"]] < 0]
    pairs = [(_value(r["reference"]), _value(r["candidate"])) for r in valid]
    correlation = None
    if len(pairs) >= 3 and len({p[0] for p in pairs}) > 1 and len({p[1] for p in pairs}) > 1:
        correlation = statistics.correlation([p[0] for p in pairs], [p[1] for p in pairs])
    by_agent = {}
    for agent in sorted({r["agent"] for r in valid}):
        rows = [r for r in valid if r["agent"] == agent]
        by_agent[agent] = sum(r["candidate"]["signal"] == r["reference"]["signal"] for r in rows) / len(rows)
    return {
        "n": len(results),
        "valid": len(valid) / len(results) if results else 0.0,
        "agreement": len(same) / len(valid) if valid else 0.0,
        "opposite": len(opposite) / len(valid) if valid else 0.0,
        "value_correlation": correlation,
        "mean_seconds": statistics.mean(r["seconds"] for r in results) if results else 0.0,
        "agreement_by_agent": by_agent,
        "prompt_chars": sum(r["prompt_chars"] for r in results),
        "response_chars": sum(r["response_chars"] for r in results),
    }


def main(argv: list[str] | None = None) -> int:
    from hedge_fund.tui.keys import apply_credentials

    apply_credentials()
    parser = argparse.ArgumentParser(prog="python -m hedge_fund.llm.compare", description=__doc__.splitlines()[0])
    parser.add_argument("--models", required=True, help="comma-separated model ids, e.g. gemini-3.8-flash,qwen3:8b")
    parser.add_argument("--reference", default="claude-opus-5-5")
    parser.add_argument("--n", type=int, default=100, help="prompts per model")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    refs = load_references(DEFAULT_CACHE_DIR, args.reference)
    picked = sample(refs, args.n, args.seed)
    verdicts = {s: sum(r["parsed"]["signal"] == s for r in picked) for s in _SIGN}
    print(f"{len(refs)} saved {args.reference} answers; testing {len(picked)} "
          f"({', '.join(f'{v} {k}' for k, v in verdicts.items())})", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        client = make_llm(model, timeout=300.0, max_tokens=8192)
        results = run_model(client, picked, progress=lambda i, n, r: print(
            f"  {model}: {i}/{n} {'ok' if r['candidate'] else 'FAIL'} {r['seconds']:.0f}s", flush=True)
            if i % 10 == 0 or i == n else None)
        summary[model] = score(results)
        (RESULTS_DIR / f"{stamp}-{model.replace(':', '_').replace('/', '_')}.json").write_text(
            json.dumps({"model": model, "reference": args.reference, "score": summary[model], "results": results}, indent=2))

    print(f"\n{'model':22} {'valid':>6} {'agree':>6} {'opposite':>9} {'corr':>6} {'sec/answer':>11}")
    for model, s in summary.items():
        corr = f"{s['value_correlation']:.2f}" if s["value_correlation"] is not None else "-"
        print(f"{model:22} {s['valid']:>6.0%} {s['agreement']:>6.0%} {s['opposite']:>9.0%} {corr:>6} {s['mean_seconds']:>11.1f}")
        print("    agreement by agent: " + ", ".join(f"{a} {v:.0%}" for a, v in s["agreement_by_agent"].items()))
    print(f"\nsaved to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
