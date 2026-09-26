"""Pure analysis for data-assumption checks — no network, no API key.

Each function takes data a DataClient already returned and classifies it.
The live runner (runner.py) fetches; this module only judges, so every
judgement is unit-testable against synthetic series.

The questions, and why each can invalidate a backtest:

- Split adjustment. SimBroker holds integer shares. If historical closes are
  NOT split-adjusted, a 4:1 split reads as a 75% one-day crash in NAV.
- Valuation timestamps. If `market_cap` / P/E on a historical metrics row
  were computed with today's price, every backtest valuation is lookahead.
- Point-in-time filtering. A row must be invisible until its filing_date,
  not its report_period (which precedes publication by weeks).
- Delisted coverage. Without history for dead companies, any broad-universe
  backtest is survivorship-biased.
"""

from __future__ import annotations

import math
import statistics
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["pass", "fail", "inconclusive", "info", "error", "skipped"]


class CheckResult(BaseModel):
    """One verification outcome. `status` is never guessed: anything the data
    cannot settle is `inconclusive`, and a failed request is `error`."""

    name: str
    status: Status
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Split adjustment
# ---------------------------------------------------------------------------

SplitBasis = Literal["adjusted", "unadjusted", "unknown"]

# A normal day moves a few percent; a split of ratio >= 2 moves by the ratio.
# Anything outside both bands is left unknown rather than forced.
_SPLIT_BAND = 0.25


def classify_split_adjustment(closes: dict[str, float], split_date: str, ratio: float) -> tuple[SplitBasis, dict]:
    """Compare the last close before *split_date* with the first on/after it.

    A pre/post jump near *ratio* means unadjusted history; near 1 means the
    history was restated (split-adjusted). *ratio* must be >= 2 for the two
    bands to stay apart.
    """
    if ratio < 2:
        raise ValueError("ratio must be >= 2 to distinguish a split from a normal move")
    before = [d for d in closes if d < split_date]
    after = [d for d in closes if d >= split_date]
    if not before or not after:
        return "unknown", {"reason": "no closes on both sides of the split date"}
    pre_day, post_day = max(before), min(after)
    pre, post = closes[pre_day], closes[post_day]
    if not (_finite_positive(pre) and _finite_positive(post)):
        return "unknown", {"reason": "non-positive or non-finite close"}
    jump = pre / post
    details = {"pre_day": pre_day, "pre_close": pre, "post_day": post_day, "post_close": post, "jump": round(jump, 4), "ratio": ratio}
    if abs(jump / ratio - 1) < _SPLIT_BAND:
        return "unadjusted", details
    if abs(jump - 1) < _SPLIT_BAND:
        return "adjusted", details
    return "unknown", details


# ---------------------------------------------------------------------------
# Valuation timestamps (market_cap, P/E)
# ---------------------------------------------------------------------------

ValuationStamp = Literal["report_period", "filing_date", "latest", "unknown"]

# Relative distance within which an implied price "matches" a close.
_PRICE_TOLERANCE = 0.03


def close_on_or_before(closes: dict[str, float], day: str) -> float | None:
    """The last close on or before *day* (weekends/holidays roll back)."""
    eligible = [d for d in closes if d <= day]
    return closes[max(eligible)] if eligible else None


def classify_pe_timestamp(pe: float | None, eps: float | None, candidates: dict[str, float | None]) -> tuple[ValuationStamp, dict]:
    """Which date's close does P/E x EPS reproduce?

    *candidates* maps a label ("report_period", "filing_date", "latest") to
    the close on that date. The implied price must sit within tolerance of
    exactly one candidate; if none match, or several do (prices too close to
    tell apart), the answer is unknown.
    """
    if pe is None or eps is None or not math.isfinite(pe) or not math.isfinite(eps) or eps <= 0 or pe <= 0:
        return "unknown", {"reason": "P/E or EPS missing, non-finite, or non-positive"}
    implied = pe * eps
    errors = {label: abs(implied / close - 1) for label, close in candidates.items() if close is not None and _finite_positive(close)}
    details = {"implied_price": round(implied, 4), "candidates": candidates, "relative_errors": {k: round(v, 4) for k, v in errors.items()}}
    matches = [label for label, err in errors.items() if err <= _PRICE_TOLERANCE]
    if len(matches) == 1:
        return matches[0], details  # type: ignore[return-value]
    details["reason"] = "no candidate within tolerance" if not matches else f"ambiguous: {sorted(matches)}"
    return "unknown", details


def classify_market_cap_timestamp(rows: list[dict], closes: dict[str, float]) -> tuple[ValuationStamp, dict]:
    """Infer when historical market caps were struck, from their ratios.

    For consecutive rows i, j: market_cap_i / market_cap_j should track the
    price ratio between the dates they were struck on (share counts drift
    only a few percent). Each hypothesis (report_period vs filing_date) is
    scored by its median log error. Identical market caps on every row mean
    one latest value was stamped onto history — lookahead.

    *rows* are dicts with report_period, filing_date, market_cap.
    """
    usable = [r for r in rows if r.get("market_cap") and r.get("filing_date") and _finite_positive(r["market_cap"])]
    if len(usable) < 3:
        return "unknown", {"reason": f"only {len(usable)} rows with market_cap and filing_date"}
    caps = [r["market_cap"] for r in usable]
    if max(caps) / min(caps) - 1 < 1e-6:
        return "latest", {"reason": "market_cap identical on every historical row", "market_cap": caps[0], "rows": len(usable)}
    scores: dict[str, list[float]] = {"report_period": [], "filing_date": []}
    for a, b in zip(usable, usable[1:]):
        for field in scores:
            pa, pb = close_on_or_before(closes, a[field][:10]), close_on_or_before(closes, b[field][:10])
            if pa and pb:
                scores[field].append(abs(math.log(a["market_cap"] / b["market_cap"]) - math.log(pa / pb)))
    medians = {k: statistics.median(v) for k, v in scores.items() if v}
    details = {"median_log_error": {k: round(v, 4) for k, v in medians.items()}, "pairs": {k: len(v) for k, v in scores.items()}}
    if len(medians) < 2:
        return "unknown", {**details, "reason": "not enough priced pairs"}
    best, other = sorted(medians, key=medians.get)
    # The winner must fit well AND clearly beat the alternative.
    if medians[best] < 0.03 and medians[other] > 2 * medians[best]:
        return best, details  # type: ignore[return-value]
    return "unknown", {**details, "reason": "neither hypothesis fits clearly"}


# ---------------------------------------------------------------------------
# Point-in-time filtering
# ---------------------------------------------------------------------------

def pit_violations(rows: list[dict], end_date: str) -> list[dict]:
    """Rows a point-in-time query as of *end_date* must never return.

    Violations: no filing_date (cannot prove it was public), filed after
    *end_date*, or filed before its own fiscal period ended (corrupt dates).
    """
    bad = []
    for r in rows:
        filed = (r.get("filing_date") or "")[:10]
        if not filed:
            bad.append({**_ident(r), "problem": "missing filing_date"})
        elif filed > end_date:
            bad.append({**_ident(r), "problem": f"filed {filed} after end_date {end_date}"})
        elif filed < r["report_period"][:10]:
            bad.append({**_ident(r), "problem": f"filed {filed} before period end {r['report_period']}"})
    return bad


def pick_gap_probe(rows: list[dict], min_gap_days: int = 7) -> dict | None:
    """Choose a row whose report_period and filing_date are far enough apart
    to query in between. Returns {report_period, filing_date, probe_date}
    with probe_date strictly inside the gap, or None.

    A filter on report_period would return that row on probe_date; a true
    filing_date filter must not.
    """
    for r in rows:
        filed = (r.get("filing_date") or "")[:10]
        if not filed:
            continue
        period = date.fromisoformat(r["report_period"][:10])
        gap = (date.fromisoformat(filed) - period).days
        if gap >= min_gap_days:
            probe = period + timedelta(days=gap // 2)
            return {"report_period": r["report_period"][:10], "filing_date": filed, "probe_date": probe.isoformat()}
    return None


def evaluate_gap_probe(probe: dict, probe_rows: list[dict]) -> CheckResult:
    """Pass if the probed period is absent from a query dated inside its gap."""
    leaked = [r for r in probe_rows if r["report_period"][:10] == probe["report_period"]]
    if leaked:
        return CheckResult(
            name="pit_gap_probe", status="fail",
            summary=f"period {probe['report_period']} (filed {probe['filing_date']}) was returned as of {probe['probe_date']} — the filter is not filing-date based",
            details={"probe": probe, "leaked": [_ident(r) for r in leaked]},
        )
    return CheckResult(
        name="pit_gap_probe", status="pass",
        summary=f"period {probe['report_period']} (filed {probe['filing_date']}) correctly hidden as of {probe['probe_date']}",
        details={"probe": probe, "returned_periods": [r["report_period"][:10] for r in probe_rows]},
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(name: str, verdicts: list[str], *, good: set[str], bad: set[str], details: dict, what: str) -> CheckResult:
    """Fold per-case verdicts into one result.

    Any bad verdict fails the check (one leak is enough to poison a
    backtest). All-good passes. Anything else — unknowns, mixed, or no
    cases — is inconclusive.
    """
    counts = {v: verdicts.count(v) for v in sorted(set(verdicts))}
    details = {**details, "verdict_counts": counts}
    if not verdicts:
        return CheckResult(name=name, status="inconclusive", summary=f"{what}: no cases could be evaluated", details=details)
    if any(v in bad for v in verdicts):
        return CheckResult(name=name, status="fail", summary=f"{what}: {counts}", details=details)
    if all(v in good for v in verdicts):
        return CheckResult(name=name, status="pass", summary=f"{what}: {counts}", details=details)
    return CheckResult(name=name, status="inconclusive", summary=f"{what}: {counts}", details=details)


def _ident(r: dict) -> dict:
    return {"report_period": r.get("report_period"), "filing_date": r.get("filing_date")}


def _finite_positive(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0
