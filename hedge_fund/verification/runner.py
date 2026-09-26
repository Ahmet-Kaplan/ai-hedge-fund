"""Live data-assumption checks against a DataClient.

Every check fetches real responses and hands them to the pure judges in
checks.py. A request that fails becomes an `error` result carrying the
message; nothing is retried into a pass and nothing is filled in. Checks are
independent, so one failing endpoint never hides the others.

Run through the CLI (``python -m hedge_fund.verification``) with a raw,
uncached FDClient: the point is to observe the provider, not the disk cache.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Callable

from hedge_fund.data.client import FDClient, FDClientError
from hedge_fund.features.snapshot import MIN_PERIODS
from hedge_fund.llm.registry import env_var_for, provider_for
from hedge_fund.verification.checks import (
    CheckResult,
    aggregate,
    classify_market_cap_timestamp,
    classify_pe_timestamp,
    classify_split_adjustment,
    close_on_or_before,
    evaluate_gap_probe,
    pick_gap_probe,
    pit_violations,
)

# Well-documented forward splits (effective first split-adjusted session).
SPLIT_CASES = [
    ("AAPL", "2020-08-31", 4.0),
    ("TSLA", "2022-08-25", 3.0),
    ("AMZN", "2022-06-06", 20.0),
    ("GOOGL", "2022-07-18", 20.0),
    ("NVDA", "2024-06-10", 10.0),
]

# No splits in the probed years, so P/E x EPS is comparable to raw closes.
VALUATION_TICKERS = ["KO", "JNJ", "PG", "JPM"]
VALUATION_AS_OF = "2024-12-31"

PIT_TICKERS = ["AAPL", "MSFT", "JPM"]
PIT_AS_OF = "2024-12-31"

# Companies that stopped trading, with their last approximate trading date.
DELISTED_CASES = [
    ("TWTR", "2022-10-27"),  # taken private
    ("ATVI", "2023-10-12"),  # acquired by Microsoft
    ("SIVB", "2023-03-09"),  # bank failure
    ("FRC", "2023-04-28"),   # bank failure
]

BATCH_TICKERS = ["AAPL", "MSFT"]


class CountingFDClient(FDClient):
    """FDClient that counts HTTP requests, pagination pages included."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requests = 0

    def _request(self, method, path, **kwargs):
        self.requests += 1
        return super()._request(method, path, **kwargs)


def required_env(model: str | None = None) -> list[str]:
    """Environment variables a Buffett-baseline backtest needs for *model*.

    Mirrors make_llm's routing: unlisted model ids use the Anthropic
    transport. Kimi also accepts MOONSHOT_API_KEY (checked by missing_env).
    """
    model = model or os.environ.get("HEDGE_FUND_LLM_MODEL") or "claude-opus-5-5"
    provider = provider_for(model) or "Anthropic"
    return ["FINANCIAL_DATASETS_API_KEY", env_var_for(provider) or "ANTHROPIC_API_KEY"]


def missing_env(names: list[str]) -> list[str]:
    """Which of *names* are unset or empty in the process environment."""
    missing = []
    for name in names:
        value = os.environ.get(name) or (os.environ.get("MOONSHOT_API_KEY") if name == "KIMI_API_KEY" else None)
        if not value:
            missing.append(name)
    return missing


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_split_adjustment(client, cases=SPLIT_CASES) -> CheckResult:
    verdicts, per_case = [], {}
    for ticker, split_date, ratio in cases:
        d = date.fromisoformat(split_date)
        bars = client.get_prices(ticker, (d - timedelta(days=10)).isoformat(), (d + timedelta(days=10)).isoformat())
        basis, info = classify_split_adjustment({b.time[:10]: b.close for b in bars}, split_date, ratio)
        verdicts.append(basis)
        per_case[f"{ticker}@{split_date}"] = {"basis": basis, **info}
    return aggregate("split_adjustment", verdicts, good={"adjusted"}, bad={"unadjusted"}, details={"cases": per_case},
                     what="historical closes across known splits")


def check_valuation_timestamps(client, tickers=VALUATION_TICKERS, as_of=VALUATION_AS_OF, latest=None) -> CheckResult:
    """P/E x EPS against closes on report_period / filing_date / latest, and
    the market-cap ratio test across consecutive rows."""
    latest = latest or (date.today() - timedelta(days=1)).isoformat()
    pe_verdicts, cap_verdicts, per_ticker = [], [], {}
    for ticker in tickers:
        rows = [m.model_dump() for m in client.get_financial_metrics(ticker, as_of, period="ttm", limit=8)]
        if not rows:
            per_ticker[ticker] = {"reason": "no metrics rows"}
            continue
        start = (date.fromisoformat(min(r["report_period"][:10] for r in rows)) - timedelta(days=10)).isoformat()
        closes = {b.time[:10]: b.close for b in client.get_prices(ticker, start, as_of)}
        recent = {b.time[:10]: b.close for b in client.get_prices(ticker, (date.fromisoformat(latest) - timedelta(days=10)).isoformat(), latest)}
        latest_close = close_on_or_before(recent, latest)
        cases = []
        for r in rows:
            candidates = {
                "report_period": close_on_or_before(closes, r["report_period"][:10]),
                "filing_date": close_on_or_before(closes, r["filing_date"][:10]) if r.get("filing_date") else None,
                "latest": latest_close,
            }
            stamp, info = classify_pe_timestamp(r.get("price_to_earnings_ratio"), r.get("earnings_per_share"), candidates)
            pe_verdicts.append(stamp)
            cases.append({"report_period": r["report_period"], "filing_date": r.get("filing_date"), "stamp": stamp, **info})
        cap_stamp, cap_info = classify_market_cap_timestamp(rows, closes)
        cap_verdicts.append(cap_stamp)
        per_ticker[ticker] = {"pe_rows": cases, "market_cap": {"stamp": cap_stamp, **cap_info}}
    pit_safe = {"report_period", "filing_date"}
    pe = aggregate("pe_timestamp", pe_verdicts, good=pit_safe, bad={"latest"}, details={}, what="P/E x EPS matched to")
    cap = aggregate("market_cap_timestamp", cap_verdicts, good=pit_safe, bad={"latest"}, details={}, what="market_cap ratios matched to")
    status = _worst([pe.status, cap.status])
    return CheckResult(
        name="valuation_timestamps", status=status,
        summary=f"P/E: {pe.summary} | market_cap: {cap.summary}",
        details={"pe_verdicts": pe.details["verdict_counts"], "market_cap_verdicts": cap.details["verdict_counts"], "tickers": per_ticker},
    )


def check_point_in_time(client, tickers=PIT_TICKERS, as_of=PIT_AS_OF) -> CheckResult:
    violations, probes, results = {}, {}, []
    for ticker in tickers:
        rows = [m.model_dump() for m in client.get_financial_metrics(ticker, as_of, period="ttm", limit=12)]
        bad = pit_violations(rows, as_of)
        if bad:
            violations[ticker] = bad
        probe = pick_gap_probe(rows)
        if probe is None:
            probes[ticker] = {"reason": "no row with a usable report/filing gap"}
            continue
        probe_rows = [m.model_dump() for m in client.get_financial_metrics(ticker, probe["probe_date"], period="ttm", limit=12)]
        probe_bad = pit_violations(probe_rows, probe["probe_date"])
        if probe_bad:
            violations[f"{ticker}@{probe['probe_date']}"] = probe_bad
        result = evaluate_gap_probe(probe, probe_rows)
        results.append(result.status)
        probes[ticker] = result.model_dump()
    details = {"violations": violations, "gap_probes": probes}
    if violations or "fail" in results:
        return CheckResult(name="point_in_time", status="fail", summary="rows visible before their filing_date (see details)", details=details)
    if not results:
        return CheckResult(name="point_in_time", status="inconclusive", summary="no gap probe could be constructed", details=details)
    return CheckResult(name="point_in_time", status="pass",
                       summary=f"no rows after end_date; {len(results)} report/filing gap probes hidden correctly", details=details)


def check_delisted(client, cases=DELISTED_CASES) -> CheckResult:
    verdicts, per_case = [], {}
    for ticker, last_day in cases:
        d = date.fromisoformat(last_day)
        bars = client.get_prices(ticker, (d - timedelta(days=90)).isoformat(), last_day)
        metrics = client.get_financial_metrics(ticker, last_day, period="ttm", limit=4)
        facts = client.get_company_facts(ticker)
        verdict = "covered" if bars and metrics else "partial" if bars or metrics else "missing"
        verdicts.append(verdict)
        per_case[ticker] = {
            "verdict": verdict, "last_trading_day": last_day, "price_bars_90d": len(bars),
            "last_bar": bars[-1].time[:10] if bars else None, "metric_rows": len(metrics),
            "facts_name": facts.name if facts else None, "facts_is_active": facts.is_active if facts else None,
        }
    return aggregate("delisted_coverage", verdicts, good={"covered"}, bad={"missing"}, details={"cases": per_case},
                     what="delisted tickers with prices+fundamentals")


def check_batching(client: CountingFDClient, tickers=BATCH_TICKERS) -> CheckResult:
    """How many requests a long single-ticker series costs, and whether the
    prices endpoint accepts several tickers in one call."""
    before = client.requests
    bars = client.get_prices(tickers[0], "2019-01-01", "2024-12-31")
    single = {"ticker": tickers[0], "range": "2019-01-01..2024-12-31", "bars": len(bars), "http_requests": client.requests - before}

    multi: dict = {"param": "tickers", "value": ",".join(tickers)}
    try:
        resp = client._request("GET", "/prices/", params={
            "tickers": ",".join(tickers), "interval": "day", "interval_multiplier": 1,
            "start_date": "2024-12-02", "end_date": "2024-12-06",
        })
        if resp is None:
            multi["outcome"] = "404"
        else:
            body = resp.json()
            rows = body.get("prices") or []
            multi.update(outcome="200", response_keys=sorted(body), rows=len(rows),
                         tickers_in_rows=sorted({r.get("ticker") for r in rows if isinstance(r, dict) and r.get("ticker")}))
    except FDClientError as exc:
        multi.update(outcome=f"rejected ({exc.status_code})", message=str(exc)[:200])

    supported = multi.get("outcome") == "200" and len(multi.get("tickers_in_rows", [])) > 1
    summary = (f"{single['bars']} daily bars for one ticker cost {single['http_requests']} request(s); "
               f"multi-ticker prices request: {'supported' if supported else 'not supported/unclear — ' + str(multi.get('outcome'))}")
    return CheckResult(name="price_batching", status="info", summary=summary, details={"single_ticker": single, "multi_ticker": multi})


def check_universe(client, universe: list[str], start: str, end: str) -> CheckResult:
    """Each name needs prices over the window and MIN_PERIODS filed TTM rows
    at the start, or the Buffett agent abstains on it."""
    verdicts, per_ticker = [], {}
    for ticker in universe:
        bars = client.get_prices(ticker, start, end)
        metrics = client.get_financial_metrics(ticker, start, period="ttm", limit=MIN_PERIODS)
        facts = client.get_company_facts(ticker)
        verdict = "no_prices" if not bars else "insufficient_history" if len(metrics) < MIN_PERIODS else "ok"
        verdicts.append(verdict)
        per_ticker[ticker] = {"verdict": verdict, "price_bars": len(bars), "first_bar": bars[0].time[:10] if bars else None,
                              "metric_rows_at_start": len(metrics), "facts_name": facts.name if facts else None}
    return aggregate("universe_coverage", verdicts, good={"ok"}, bad={"no_prices"}, details={"window": [start, end], "tickers": per_ticker},
                     what=f"universe data coverage {start}..{end}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_checks(client: CountingFDClient, universe: list[str] | None = None, window: tuple[str, str] | None = None,
               on_result: Callable[[CheckResult], None] | None = None) -> list[CheckResult]:
    """Run every check; a raised error becomes that check's `error` result."""
    plan: list[tuple[str, Callable[[], CheckResult]]] = [
        ("split_adjustment", lambda: check_split_adjustment(client)),
        ("valuation_timestamps", lambda: check_valuation_timestamps(client)),
        ("point_in_time", lambda: check_point_in_time(client)),
        ("delisted_coverage", lambda: check_delisted(client)),
        ("price_batching", lambda: check_batching(client)),
    ]
    if universe:
        start, end = window or ("2025-01-01", "2026-08-31")
        plan.append(("universe_coverage", lambda: check_universe(client, universe, start, end)))
    results = []
    for name, run in plan:
        try:
            result = run()
        except Exception as exc:  # report, never mask: the check is unresolved
            result = CheckResult(name=name, status="error", summary=f"{type(exc).__name__}: {exc}"[:300])
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def _worst(statuses: list[str]) -> str:
    order = ["fail", "error", "inconclusive", "pass", "info", "skipped"]
    return min(statuses, key=order.index)
