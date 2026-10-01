import pytest

from hedge_fund.brokers.alpaca import CashFlow
from hedge_fund.live.account.ledger import LiveLedger, LiveNavRow
from hedge_fund.live.account.report import build_live_report
from hedge_fund.live.account.settings import LiveSettings


def test_time_weighted_return_ignores_deposits(tmp_path):
    live = LiveLedger(tmp_path)
    live.add_flows([CashFlow(id="1", kind="CSD", date="2026-10-01", amount=100.0),
                    CashFlow(id="2", kind="CSD", date="2026-11-02", amount=100.0)])
    for d, eq, flow, close in [("2026-10-01", 100.0, 100.0, 500.0), ("2026-10-30", 110.0, 0.0, 525.0),
                               ("2026-11-02", 210.0, 100.0, 525.0), ("2026-11-30", 231.0, 0.0, 551.25)]:
        live.upsert_live_nav(LiveNavRow(date=d, equity=eq, cash=0, net_flow=flow, core_value=eq,
                                        satellite_value=0, core_close=close))
    r = build_live_report(live, LiveSettings())
    assert r.deposited == 200.0 and r.equity == 231.0 and r.gain == pytest.approx(31.0)
    assert r.time_weighted_return == pytest.approx(1.1 * 1.1 - 1)
    assert r.core_return == pytest.approx(551.25 / 500 - 1)


def test_empty_ledger(tmp_path):
    assert build_live_report(LiveLedger(tmp_path), LiveSettings()) is None
