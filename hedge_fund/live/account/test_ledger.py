import pytest

from hedge_fund.brokers.alpaca import CashFlow
from hedge_fund.live.account.ledger import LiveLedger, LiveNavRow


def row(day, equity, flow=0.0, core_close=500.0, sat=None):
    return LiveNavRow(date=day, equity=equity, cash=0.0, net_flow=flow, core_value=equity,
                      satellite_value=0.0, core_close=core_close, satellite_return=sat)


def test_flows_dedupe_and_net_flow_excludes_dividends(tmp_path):
    ledger = LiveLedger(tmp_path)
    flows = [CashFlow(id="1", kind="CSD", date="2026-10-01", amount=100.0),
             CashFlow(id="2", kind="DIV", date="2026-10-02", amount=0.5),
             CashFlow(id="3", kind="CSW", date="2026-10-02", amount=-20.0)]
    ledger.add_flows(flows)
    ledger.add_flows(flows)
    assert len(ledger.flows()) == 3
    assert ledger.net_flow("2026-09-30", "2026-10-02") == pytest.approx(80.0)
    assert ledger.net_flow("2026-10-01", "2026-10-02") == pytest.approx(-20.0)   # (after, through]
    assert ledger.total_deposited() == pytest.approx(80.0)


def test_nav_rows_holdings_and_markers(tmp_path):
    ledger = LiveLedger(tmp_path)
    ledger.upsert_live_nav(row("2026-10-02", 101.0))
    ledger.upsert_live_nav(row("2026-10-01", 100.0, flow=100.0))
    ledger.upsert_live_nav(row("2026-10-02", 102.0))
    assert [(r.date, r.equity) for r in ledger.live_nav_rows()] == [("2026-10-01", 100.0), ("2026-10-02", 102.0)]
    ledger.save_holdings("2026-10-01", {"SPY": 0.2})
    ledger.save_holdings("2026-10-02", {"SPY": 0.3})
    assert ledger.holdings_before("2026-10-02") == ("2026-10-01", {"SPY": 0.2})
    assert ledger.holdings_before("2026-10-01") is None
    assert not ledger.dry_run_done()
    ledger.mark_dry_run_done()
    assert ledger.dry_run_done()
    ledger.halt_satellite("trailing")
    assert ledger.satellite_halted()
    ledger.clear_satellite_halt()
    assert not ledger.satellite_halted()


def test_reviews_and_satellite_excess(tmp_path):
    ledger = LiveLedger(tmp_path)
    assert ledger.last_review_date() is None
    ledger.append_review("2026-10-01", 0.0, 0.2, paper_excess=0.01, live_excess=None, passed=True)
    assert ledger.last_review_date() == "2026-10-01"
    for day, close, sat in [("2026-10-01", 500.0, None), ("2026-10-02", 510.0, 0.01), ("2026-10-05", 520.2, -0.05)]:
        ledger.upsert_live_nav(row(day, 100.0, core_close=close, sat=sat))
    # satellite 1.01 × 0.95 − 1 = −4.05%; core 520.2/500 − 1 = +4.04%
    assert ledger.satellite_excess(since="2026-10-01") == pytest.approx(0.9595 - 1.0404, abs=1e-6)
    assert ledger.satellite_excess(since="2026-10-06") is None


def test_crypto_limit_attempts_round_trip(tmp_path):
    ledger = LiveLedger(tmp_path / "live")
    assert ledger.crypto_limits() == {}
    ledger.save_crypto_limits({"BTC/USD": {"side": "buy", "tries": 2}})
    assert ledger.crypto_limits() == {"BTC/USD": {"side": "buy", "tries": 2}}
