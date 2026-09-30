"""Decision reports built from real receipts: the real backtester, synthetic
prices, and a scripted stub analyst covering every decision path."""

import json
from datetime import date, timedelta

import pytest

from hedge_fund.backtesting.fund import backtest_fund
from hedge_fund.data.models import Price
from hedge_fund.fund.spec import Fund, FundSpec
from hedge_fund.models import Signal
from hedge_fund.pipeline.models import PendingRunResult
from hedge_fund.reporting import __main__ as cli
from hedge_fund.reporting import backtest_report, cycle_report, load_receipt, pending_report, to_html, to_markdown

UNIVERSE = ["KO", "JNJ", "AAPL", "MSFT", "XOM", "BIG", "NOPX"]
PRICES = {"SPY": 400.0, "KO": 60.0, "JNJ": 150.0, "AAPL": 180.0, "MSFT": 350.0, "XOM": 100.0, "BIG": 50_000.0}  # NOPX: no data
HOSTILE = "Durable moat | fair price <script>alert(1)</script>"


def _weekdays(start, end):
    d, out = date.fromisoformat(start), []
    while d <= date.fromisoformat(end):
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DAYS = _weekdays("2024-01-02", "2024-04-30")


class FlatData:
    def get_prices(self, ticker, start_date, end_date, **kw):
        if ticker not in PRICES:
            return []
        p = PRICES[ticker]
        return [Price(open=p, close=p, high=p, low=p, volume=1, time=f"{d}T00:00:00Z") for d in DAYS if start_date <= d <= end_date]


class ScriptedBuffett:
    """January: bullish KO/JNJ/AAPL/BIG, bearish XOM, abstains on MSFT.
    From mid-February: JNJ turns neutral (exit), the rest unchanged."""

    investment_approach = "long_only"
    name = "buffett"

    def predict(self, ticker, as_of, data_client):
        if ticker == "MSFT":
            return Signal(model_name="buffett", ticker=ticker, date=as_of, value=0.0, reasoning="abstained: insufficient data",
                          metadata={"abstained": True, "abstain_reason": "insufficient data: only 2 filed periods (need 4)"})
        views = {"KO": ("bullish", 90), "JNJ": ("bullish", 80), "AAPL": ("bullish", 70), "BIG": ("bullish", 60), "XOM": ("bearish", 80)}
        if as_of >= "2024-02-15":
            views["JNJ"] = ("neutral", 50)
        label, conf = views[ticker]
        sign = {"bullish": 1, "neutral": 0, "bearish": -1}[label]
        return Signal(model_name="buffett", ticker=ticker, date=as_of, value=sign * conf / 100, reasoning=f"{ticker}: {HOSTILE}",
                      metadata={"signal": label, "confidence": conf, "cached": False, "abstained": False})


SPEC = FundSpec(
    schema_version=2, name="buffett-baseline", capital=100_000, rebalance="monthly", benchmark="SPY",
    strategies=[{"name": "buffett-value", "models": [{"name": "buffett"}], "blend": {"mode": "long_only"}}],
    risk={"max_position_pct": 0.15, "max_gross_exposure": 1.0},
)


@pytest.fixture(scope="module")
def result():
    fund = Fund(SPEC, models={"buffett-value": [ScriptedBuffett()]})
    return backtest_fund(fund, DAYS[0], DAYS[-1], FlatData(), UNIVERSE)


def _by_ticker(report):
    return {d.ticker: d for d in report.decisions}


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def test_first_rebalance_opens_capped_positions_and_keeps_cash(result):
    r = cycle_report(result.records[0])
    assert r.status == "executed" and r.execution_as_of == "2024-02-01"
    d = _by_ticker(r)
    assert {x.ticker for x in r.selected} == {"KO", "JNJ", "AAPL"}
    for t in ("KO", "JNJ", "AAPL"):
        assert (d[t].action, d[t].action_detail) == ("BUY", "open")
        assert d[t].shares_before == 0 and d[t].trade_shares == d[t].shares_after > 0
        assert d[t].weight == pytest.approx(0.15, abs=0.002)
        assert any("capped at 15.0% by max_position_pct" in n for n in d[t].notes)
        assert d[t].rejection_reason is None
    assert d["KO"].views[0].signal == "bullish" and d["KO"].views[0].confidence == 90
    assert r.cash_weight == pytest.approx(0.55, abs=0.01)
    assert r.cash_weight + r.invested_weight == pytest.approx(1.0)


def test_every_rejection_has_a_concrete_reason(result):
    d = _by_ticker(cycle_report(result.records[0]))
    assert "buffett bearish (80)" in d["XOM"].rejection_reason and "long-only" in d["XOM"].rejection_reason
    assert "every analyst abstained" in d["MSFT"].rejection_reason and "only 2 filed periods" in d["MSFT"].rejection_reason
    assert d["NOPX"].rejection_reason.startswith("not assessed: no close")
    assert "below one share at $50,000.00" in d["BIG"].rejection_reason
    for t in ("XOM", "MSFT", "NOPX", "BIG"):
        assert d[t].action == "NONE" and not d[t].selected


def test_later_rebalance_holds_and_exits(result):
    d = _by_ticker(cycle_report(result.records[1]))
    assert (d["KO"].action, d["KO"].trade_shares) == ("HOLD", 0)
    assert (d["AAPL"].action, d["AAPL"].action_detail) == ("HOLD", "hold")
    assert (d["JNJ"].action, d["JNJ"].action_detail) == ("SELL", "exit")
    assert d["JNJ"].shares_after == 0 and d["JNJ"].trade_shares < 0
    assert "buffett neutral (50)" in d["JNJ"].rejection_reason


def test_backtest_report_summarizes_every_rebalance(result):
    b = backtest_report(result)
    assert [s.as_of for s in b.rebalances] == ["2024-01-31", "2024-02-29", "2024-03-29"]
    assert b.rebalances[0].buys == ["KO", "JNJ", "AAPL"] and b.rebalances[1].sells == ["JNJ"]
    assert [h.ticker for h in b.rebalances[-1].holdings] == sorted(["KO", "AAPL"], key=lambda t: -dict((h.ticker, h.weight) for h in b.rebalances[-1].holdings)[t])
    assert b.latest.as_of == "2024-03-29"
    assert b.next_proposal.status == "pending" and b.next_proposal.as_of == "2024-04-30"


def test_pending_proposal_states_actions_from_an_empty_book(result):
    p = pending_report(result.pending[-1])
    d = _by_ticker(p)
    assert p.status == "pending" and p.pending_reason
    assert (d["KO"].action, d["KO"].target_weight) == ("BUY", pytest.approx(0.15))
    assert d["JNJ"].action == "NONE" and "neutral" in d["JNJ"].rejection_reason
    assert p.cash_weight == pytest.approx(0.55)
    assert d["KO"].weight is None and d["KO"].trade_shares is None  # nothing executed
    assert any("below one share at $50,000.00" in n for n in d["BIG"].notes)  # flagged, since nothing has executed yet
    assert not any("below one share" in n for n in d["KO"].notes)


def test_load_receipt_dispatches_on_every_receipt_type(result):
    roundtrip = lambda m: json.loads(m.model_dump_json())  # noqa: E731
    assert load_receipt(roundtrip(result)).rebalances
    assert load_receipt(roundtrip(result.records[0])).status == "executed"
    pending = PendingRunResult.model_validate(roundtrip(result.pending[0]))
    assert load_receipt(roundtrip(pending)).status == "pending"
    with pytest.raises(ValueError):
        load_receipt({"fund": "x"})


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_markdown_has_every_section_and_keeps_tables_intact(result):
    md = to_markdown(backtest_report(result))
    for heading in ("# buffett-baseline — backtest", "## Rebalance history", "### Selected portfolio", "### Rejected", "### Analyst reasoning"):
        assert heading in md
    assert "Durable moat \\| fair price" in md  # a pipe in reasoning cannot split a table cell
    assert "<script>" not in md  # many Markdown viewers render inline HTML
    assert "**Cash:**" in md
    assert "| JNJ | SELL (exit" in to_markdown(cycle_report(result.records[1]))


def test_html_escapes_recorded_text_and_embeds_the_chart(result):
    html = to_html(backtest_report(result))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert html.count("<svg") == 1 and "Fund NAV versus SPY" in html
    assert "prefers-color-scheme:dark" in html


def test_html_decision_page_for_a_single_cycle(result):
    html = to_html(cycle_report(result.records[0]))
    assert "<svg" not in html and "Selected portfolio" in html and "Rejected" in html


def test_cli_renders_html_and_markdown(result, tmp_path, capsys):
    receipt = tmp_path / "result.json"
    receipt.write_text(result.model_dump_json())
    out = tmp_path / "report.html"
    assert cli.main([str(receipt), "--out", str(out)]) == 0
    assert out.read_text().startswith("<!doctype html>")
    assert cli.main([str(receipt)]) == 0
    assert "## Rebalance history" in capsys.readouterr().out


def test_cli_rejects_unreadable_receipts(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"fund": "x"}')
    assert cli.main([str(bad)]) == 1
    assert cli.main([str(tmp_path / "missing.json")]) == 1
    assert "cannot build a report" in capsys.readouterr().err
