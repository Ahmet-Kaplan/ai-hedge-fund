"""Live runner tests — a fake live account and fake prices; nothing reaches Alpaca."""

from datetime import datetime

import pytest

from hedge_fund.brokers.alpaca import Account, CashFlow, LiveOrder
from hedge_fund.data.sessions import NEW_YORK
from hedge_fund.live.account.ledger import LiveLedger
from hedge_fund.live.account.runner import reconcile_live, submit_live
from hedge_fund.live.account.settings import LiveSettings
from hedge_fund.live.ledger import Ledger
from hedge_fund.live.test_plan import FakeDataClient

SESSIONS = ["2026-08-19", "2026-08-20", "2026-08-21", "2026-08-24", "2026-08-25"]
CLOSES = {"SPY": {d: 500.0 for d in SESSIONS}, "NVDA": {d: 100.0 for d in SESSIONS}}
MON = datetime(2026, 8, 24, 4, 0, tzinfo=NEW_YORK)    # 09:00 UK
TUE = datetime(2026, 8, 25, 4, 0, tzinfo=NEW_YORK)


class FakeLive:
    def __init__(self, holdings=None, cash=0.0, flows=(), orders=()):
        self._holdings, self.cash, self._flows, self.orders = dict(holdings or {}), cash, list(flows), list(orders)
        self.sent = []

    def calendar(self, start, end):
        return [s for s in SESSIONS if start <= s <= end]

    def account(self):
        return Account(cash=self.cash, equity=self.cash, last_equity=self.cash, status="ACTIVE")

    def holdings(self):
        return dict(self._holdings)

    def cash_flows(self, after):
        return [f for f in self._flows if f.date > after]

    def list_orders(self, after):
        return list(self.orders)

    def _order(self, cid, ticker, side, **kw):
        self.sent.append((ticker, side, kw))
        return LiveOrder(client_order_id=cid, ticker=ticker, side=side, status="accepted", **kw)

    def buy_notional(self, ticker, dollars, cid):
        return self._order(cid, ticker, "buy", notional=dollars)

    def sell_qty(self, ticker, qty, cid):
        return self._order(cid, ticker, "sell", qty=qty)

    def trade_shares(self, ticker, side, shares, cid):
        return self._order(cid, ticker, side, qty=float(shares))


@pytest.fixture
def env(tmp_path):
    paper = Ledger(tmp_path / "paper")
    paper.write_plan("2026-08-24", {"plan": {"decision": {"final_weights": {"NVDA": 0.04}}}})
    return dict(live=LiveLedger(tmp_path / "live"), paper=paper, kill=tmp_path / "KILL")


def run(env, client, now=MON, dry_run=False, **settings):
    s = LiveSettings(confirm_live=True, **settings)
    return submit_live(s, client, FakeDataClient(CLOSES), env["live"], env["paper"], now=now,
                       dry_run=dry_run, kill_path=env["kill"])


def test_guards_in_order(env):
    env["kill"].touch()
    assert run(env, FakeLive(cash=100.0)).status == "killed"
    env["kill"].unlink()
    assert submit_live(LiveSettings(), FakeLive(cash=100.0), FakeDataClient(CLOSES), env["live"], env["paper"],
                       now=MON, kill_path=env["kill"]).status == "not_confirmed"
    assert run(env, FakeLive(cash=100.0)).status == "needs_dry_run"


def test_dry_run_then_first_deposit_goes_all_to_core(env):
    client = FakeLive(cash=100.0)
    dry = run(env, client, dry_run=True)
    assert dry.status == "dry_run" and client.sent == []
    result = run(env, client, cash_buffer_pct=0.0)
    assert result.status == "submitted"
    assert client.sent == [("SPY", "buy", {"notional": 100.0})]


def test_cash_day_invests_deposit_without_selling(env):
    env["live"].mark_dry_run_done()
    client = FakeLive(holdings={"SPY": 0.2}, cash=50.0)
    result = run(env, client, now=TUE, cash_buffer_pct=0.0)
    assert result.status == "submitted"
    assert client.sent == [("SPY", "buy", {"notional": 50.0})]


def test_nothing_to_do_without_cash_on_a_cash_day(env):
    env["live"].mark_dry_run_done()
    assert run(env, FakeLive(holdings={"SPY": 0.2}, cash=0.5), now=TUE).status == "nothing_to_do"


def test_satellite_mirrors_paper_when_share_set(env):
    env["live"].mark_dry_run_done()
    client = FakeLive(holdings={"SPY": 2.0}, cash=0.0)       # $1,000 in SPY
    run(env, client, agent_share=0.2, cash_buffer_pct=0.0)
    # NVDA's 20% share is capped at the 10% per-name limit; the rest stays in SPY: sell $100 of SPY
    assert ("SPY", "sell", {"qty": pytest.approx(0.2)}) in client.sent
    # buys wait for cash (no same-day proceeds): next cash day buys NVDA
    assert all(side == "sell" for _, side, _ in client.sent)


def test_halted_satellite_counts_as_zero_share(env):
    env["live"].mark_dry_run_done()
    env["live"].halt_satellite("trailing")
    client = FakeLive(holdings={"SPY": 1.6, "NVDA": 2.0}, cash=0.0)
    run(env, client, agent_share=0.2)
    assert ("NVDA", "sell", {"qty": pytest.approx(2.0)}) in client.sent


def test_reconcile_separates_deposits_and_tracks_satellite(env):
    live = env["live"]
    client = FakeLive(holdings={"SPY": 0.2}, cash=0.0,
                      flows=[CashFlow(id="d1", kind="CSD", date="2026-08-20", amount=100.0)])
    r1 = reconcile_live(LiveSettings(), client, FakeDataClient(CLOSES), live, session="2026-08-20")
    assert (r1.equity, r1.net_flow, r1.core_value) == (100.0, 100.0, 100.0)
    closes = {"SPY": CLOSES["SPY"], "NVDA": {**CLOSES["NVDA"], "2026-08-21": 110.0}}
    client._holdings = {"SPY": 0.2, "NVDA": 1.0}
    client.cash = -100.0  # bought NVDA on margin-free cash in this toy; equity math only
    live.save_holdings("2026-08-20", {"SPY": 0.2, "NVDA": 1.0})
    r2 = reconcile_live(LiveSettings(), client, FakeDataClient(closes), live, session="2026-08-21")
    assert r2.satellite_return == pytest.approx(0.10)
    assert r2.net_flow == 0.0


def test_reconcile_halts_a_trailing_satellite(env):
    live = env["live"]
    live.append_review("2026-08-20", 0.0, 0.2, paper_excess=0.01, live_excess=None, passed=True)
    live.save_holdings("2026-08-20", {"NVDA": 1.0})
    live.upsert_live_nav(__import__("hedge_fund.live.account.ledger", fromlist=["LiveNavRow"]).LiveNavRow(
        date="2026-08-20", equity=100.0, cash=0.0, net_flow=0.0, core_value=0.0, satellite_value=100.0, core_close=500.0))
    closes = {"SPY": CLOSES["SPY"], "NVDA": {**CLOSES["NVDA"], "2026-08-21": 85.0}}
    reconcile_live(LiveSettings(agent_share=0.2), FakeLive(holdings={"NVDA": 1.0}), FakeDataClient(closes), live,
                   session="2026-08-21")
    assert live.satellite_halted()


def test_core_only_run_ignores_unpriceable_paper_names(env):
    env["live"].mark_dry_run_done()
    env["paper"].write_plan("2026-08-24", {"plan": {"decision": {"final_weights": {"ZZZ": 0.05}}}})
    client = FakeLive(cash=100.0)
    result = run(env, client, cash_buffer_pct=0.0)            # share 0: ZZZ has no price, and needs none
    assert result.status == "submitted"
    assert client.sent == [("SPY", "buy", {"notional": 100.0})]
