"""Live-check orchestration against fake clients — no network, no key."""

from datetime import date, timedelta

import pytest

from hedge_fund.data.client import FDClientError
from hedge_fund.data.models import CompanyFacts, FinancialMetrics, Price
from hedge_fund.verification import __main__ as cli
from hedge_fund.verification.runner import (
    CountingFDClient,
    check_batching,
    check_delisted,
    check_point_in_time,
    check_split_adjustment,
    check_universe,
    check_valuation_timestamps,
    missing_env,
    required_env,
    run_checks,
)


def _bar(day, close):
    return Price(open=close, close=close, high=close, low=close, volume=1, time=f"{day}T00:00:00Z")


class FakeClient:
    """Serves canned closes and metric rows; filters metrics like FD claims to
    (filing_date <= end_date) unless `leak` makes it filter on report_period."""

    def __init__(self, closes=None, metrics=None, facts=None, leak=False):
        self.closes = closes or {}    # ticker -> {day: close}
        self.metrics = metrics or {}  # ticker -> [row dict], newest first
        self.facts = facts or {}
        self.leak = leak

    def get_prices(self, ticker, start_date, end_date, **kw):
        return [_bar(d, c) for d, c in sorted(self.closes.get(ticker, {}).items()) if start_date <= d <= end_date]

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        field = "report_period" if self.leak else "filing_date"
        rows = [r for r in self.metrics.get(ticker, []) if r[field][:10] <= end_date]
        return [FinancialMetrics(ticker=ticker, period="ttm", **r) for r in rows[:limit]]

    def get_company_facts(self, ticker):
        return self.facts.get(ticker)


QUARTERS = [("2024-09-30", "2024-10-31"), ("2024-06-30", "2024-08-01"), ("2024-03-31", "2024-05-02"), ("2023-12-31", "2024-02-01")]


def _metric_rows(**fields):
    return [{"report_period": p, "filing_date": f, **fields} for p, f in QUARTERS]


def _daily(start, end, close):
    d, out = date.fromisoformat(start), {}
    while d <= date.fromisoformat(end):
        out[d.isoformat()] = close(d.isoformat()) if callable(close) else close
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def test_split_check_passes_on_adjusted_and_fails_on_unadjusted():
    adjusted = FakeClient(closes={"AAPL": {"2020-08-28": 124.8, "2020-08-31": 129.0}})
    assert check_split_adjustment(adjusted, [("AAPL", "2020-08-31", 4.0)]).status == "pass"
    raw = FakeClient(closes={"AAPL": {"2020-08-28": 499.2, "2020-08-31": 129.0}})
    assert check_split_adjustment(raw, [("AAPL", "2020-08-31", 4.0)]).status == "fail"


def test_valuation_check_detects_latest_price_stamping():
    # Close is 50 everywhere historically and 80 at "latest"; P/E x EPS = 80.
    closes = {**_daily("2023-12-15", "2024-12-31", 50.0), **_daily("2026-09-10", "2026-09-20", 80.0)}
    client = FakeClient(closes={"KO": closes}, metrics={"KO": _metric_rows(price_to_earnings_ratio=32.0, earnings_per_share=2.5, market_cap=3e11)})
    result = check_valuation_timestamps(client, ["KO"], "2024-12-31", latest="2026-09-20")
    assert result.status == "fail"
    assert result.details["pe_verdicts"] == {"latest": 4}
    assert result.details["market_cap_verdicts"] == {"latest": 1}


def test_valuation_check_accepts_filing_date_stamping():
    filing_prices = {f: 40.0 + 10 * i for i, (_, f) in enumerate(QUARTERS)}
    closes = {**_daily("2023-12-15", "2024-12-31", 20.0), **filing_prices, **_daily("2026-09-10", "2026-09-20", 99.0)}
    rows = [{"report_period": p, "filing_date": f, "price_to_earnings_ratio": filing_prices[f] / 2.0, "earnings_per_share": 2.0,
             "market_cap": filing_prices[f] * 1e9} for p, f in QUARTERS]
    client = FakeClient(closes={"KO": closes}, metrics={"KO": rows})
    result = check_valuation_timestamps(client, ["KO"], "2024-12-31", latest="2026-09-20")
    assert result.status == "pass", result.summary
    assert result.details["pe_verdicts"] == {"filing_date": 4}


def test_point_in_time_check_passes_honest_filter_and_fails_report_period_filter():
    rows = {"AAPL": _metric_rows()}
    assert check_point_in_time(FakeClient(metrics=rows), ["AAPL"], "2024-12-31").status == "pass"
    leaky = check_point_in_time(FakeClient(metrics=rows, leak=True), ["AAPL"], "2024-12-31")
    assert leaky.status == "fail"


def test_point_in_time_inconclusive_without_any_gap():
    rows = {"AAPL": [{"report_period": p, "filing_date": p} for p, _ in QUARTERS]}
    assert check_point_in_time(FakeClient(metrics=rows), ["AAPL"], "2024-12-31").status == "inconclusive"


def test_delisted_check_classifies_coverage():
    covered = FakeClient(
        closes={"TWTR": _daily("2022-09-01", "2022-10-27", 50.0)},
        metrics={"TWTR": [{"report_period": "2022-06-30", "filing_date": "2022-07-29"}]},
        facts={"TWTR": CompanyFacts(ticker="TWTR", name="Twitter", is_active=False)},
    )
    result = check_delisted(covered, [("TWTR", "2022-10-27")])
    assert result.status == "pass"
    assert result.details["cases"]["TWTR"]["facts_is_active"] is False
    assert check_delisted(FakeClient(), [("TWTR", "2022-10-27")]).status == "fail"


def test_universe_check_flags_missing_names_and_short_history():
    client = FakeClient(
        closes={"AAPL": _daily("2025-01-01", "2025-01-10", 1.0), "NEW": _daily("2025-01-01", "2025-01-10", 1.0)},
        metrics={"AAPL": _metric_rows(), "NEW": _metric_rows()[:2]},
    )
    result = check_universe(client, ["AAPL", "NEW", "BRK.B"], "2025-01-01", "2025-01-10")
    verdicts = {t: v["verdict"] for t, v in result.details["tickers"].items()}
    assert verdicts == {"AAPL": "ok", "NEW": "insufficient_history", "BRK.B": "no_prices"}
    assert result.status == "fail"


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload, self.text = status, payload, ""

    def json(self):
        return self._payload


def test_batching_counts_pages_and_reports_rejected_multi_ticker():
    client = CountingFDClient(api_key="test")
    responses = [
        _Resp(200, {"prices": [_bar("2019-01-02", 1.0).model_dump()], "next_page_url": "https://x/page2"}),
        _Resp(200, {"prices": [_bar("2019-01-03", 1.0).model_dump()]}),
        _Resp(400, {}),
    ]
    client._session.request = lambda *a, **k: responses.pop(0)
    result = check_batching(client)
    assert result.status == "info"
    assert result.details["single_ticker"] == {"ticker": "AAPL", "range": "2019-01-01..2024-12-31", "bars": 2, "http_requests": 2}
    assert result.details["multi_ticker"]["outcome"] == "rejected (400)"
    assert client.requests == 3


def test_batching_recognizes_multi_ticker_support():
    client = CountingFDClient(api_key="test")
    rows = [{**_bar("2024-12-02", 1.0).model_dump(), "ticker": t} for t in ("AAPL", "MSFT")]
    responses = [_Resp(200, {"prices": []}), _Resp(200, {"prices": rows})]
    client._session.request = lambda *a, **k: responses.pop(0)
    assert "multi-ticker prices request: supported" in check_batching(client).summary


# ---------------------------------------------------------------------------
# Orchestration and environment
# ---------------------------------------------------------------------------

def test_run_checks_turns_exceptions_into_error_results():
    class Broken(FakeClient):
        def get_prices(self, *a, **k):
            raise FDClientError("GET /prices/ returned 401: bad key", status_code=401)

    seen = []
    results = run_checks(Broken(), ["AAPL"], ("2025-01-01", "2025-02-01"), on_result=seen.append)
    assert [r.name for r in results] == ["split_adjustment", "valuation_timestamps", "point_in_time",
                                         "delisted_coverage", "price_batching", "universe_coverage"]
    assert results[0].status == "error" and "401" in results[0].summary
    assert seen == results


def test_required_env_follows_model_routing(monkeypatch):
    monkeypatch.delenv("HEDGE_FUND_LLM_MODEL", raising=False)
    assert required_env() == ["FINANCIAL_DATASETS_API_KEY", "ANTHROPIC_API_KEY"]
    assert required_env("gpt-6-sol") == ["FINANCIAL_DATASETS_API_KEY", "OPENAI_API_KEY"]
    assert required_env("some-unlisted-model") == ["FINANCIAL_DATASETS_API_KEY", "ANTHROPIC_API_KEY"]


def test_missing_env_accepts_moonshot_alias(monkeypatch):
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.setenv("MOONSHOT_API_KEY", "x")
    assert missing_env(["KIMI_API_KEY"]) == []


def test_cli_stops_cleanly_without_data_key(monkeypatch, capsys):
    monkeypatch.delenv("FINANCIAL_DATASETS_API_KEY", raising=False)
    monkeypatch.setattr(cli, "CountingFDClient", lambda *a, **k: pytest.fail("must not build a client without a key"))
    assert cli.main([]) == 2
    err = capsys.readouterr().err
    assert "FINANCIAL_DATASETS_API_KEY: MISSING" in err and "no data checks were run" in err


def test_cli_preflight_reports_llm_key(monkeypatch, capsys):
    monkeypatch.setenv("FINANCIAL_DATASETS_API_KEY", "x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("HEDGE_FUND_LLM_MODEL", raising=False)
    assert cli.main(["--preflight"]) == 2
    assert "ANTHROPIC_API_KEY: MISSING" in capsys.readouterr().err
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    assert cli.main(["--preflight"]) == 0


def test_load_universe_reads_the_baseline_file(tmp_path):
    path = tmp_path / "u.yaml"
    path.write_text("tickers: [aapl, ' brk.b ']\n")
    assert cli.load_universe(path) == ["AAPL", "BRK.B"]
    path.write_text("tickers: []\n")
    with pytest.raises(ValueError):
        cli.load_universe(path)
