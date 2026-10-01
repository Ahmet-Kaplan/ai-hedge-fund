"""submit/reconcile tests — a fake Alpaca account, a tmp ledger, real planning."""

from datetime import datetime

import pytest

from hedge_fund.brokers.alpaca import Account, OrderResult
from hedge_fund.data.sessions import NEW_YORK
from hedge_fund.live.ledger import Ledger, NavRow
from hedge_fund.live.runner import reconcile, session_to_reconcile, submit
from hedge_fund.live.test_plan import SERIES, FakeDataClient, make_fund, registered_fakes  # noqa: F401

SESSIONS = ["2024-06-06", "2024-06-07", "2024-06-10", "2024-06-11"]
MON_10AM = datetime(2024, 6, 10, 10, 0, tzinfo=NEW_YORK)
TUE_10AM = datetime(2024, 6, 11, 10, 0, tzinfo=NEW_YORK)


class FakeAlpaca:
    def __init__(self, positions=None, cash=100_000.0, orders=None, reject=()):
        self.sessions = SESSIONS
        self._positions = dict(positions or {})
        self.cash = cash
        self.orders = list(orders or [])
        self.reject = set(reject)
        self.submitted = []

    def calendar(self, start, end):
        return [s for s in self.sessions if start <= s <= end]

    def list_orders(self, after):
        return list(self.orders)

    def positions(self):
        return dict(self._positions)

    def account(self):
        return Account(cash=self.cash, equity=self.cash, last_equity=self.cash, status="ACTIVE")

    def submit_order(self, ticker, side, quantity, client_order_id):
        self.submitted.append((ticker, side, quantity, client_order_id))
        status = "rejected" if ticker in self.reject else "accepted"
        return OrderResult(client_order_id=client_order_id, ticker=ticker, side=side, quantity=quantity, status=status)


def nav(day, equity):
    return NavRow(date=day, equity=equity, cash=equity, long_exposure=0.0,
                  short_exposure=0.0, gross=0.0, benchmark_close=500.0)


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger")


def run_submit(client, ledger, tmp_path, *, now=MON_10AM, dry_run=False):
    fund, analyst = make_fund({"AAPL": 1.0})
    result = submit(fund, ["AAPL"], client, FakeDataClient(SERIES), ledger,
                    now=now, dry_run=dry_run, kill_path=tmp_path / "KILL")
    return result, analyst


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def test_submits_moc_orders_on_rebalance_day(ledger, tmp_path):
    client = FakeAlpaca()
    result, _ = run_submit(client, ledger, tmp_path)
    assert result.status == "submitted"
    assert client.submitted == [("AAPL", "buy", 500, "paper-test-2024-06-10-AAPL")]
    saved = ledger.read_plan("2024-06-10")
    assert saved["status"] == "submitted"
    assert saved["orders"][0]["status"] == "accepted"


def test_kill_switch_stops_everything(ledger, tmp_path):
    (tmp_path / "KILL").touch()
    client = FakeAlpaca()
    result, _ = run_submit(client, ledger, tmp_path)
    assert result.status == "killed"
    assert client.submitted == []


def test_not_a_trading_day(ledger, tmp_path):
    result, _ = run_submit(FakeAlpaca(), ledger, tmp_path, now=datetime(2024, 6, 8, 10, 0, tzinfo=NEW_YORK))
    assert result.status == "not_trading_day"


def test_too_late_for_moc(ledger, tmp_path):
    result, _ = run_submit(FakeAlpaca(), ledger, tmp_path, now=datetime(2024, 6, 10, 15, 51, tzinfo=NEW_YORK))
    assert result.status == "too_late"


def test_already_submitted_is_a_no_op(ledger, tmp_path):
    existing = OrderResult(client_order_id="paper-test-2024-06-10-AAPL", ticker="AAPL", side="buy", quantity=500, status="new")
    client = FakeAlpaca(orders=[existing])
    result, _ = run_submit(client, ledger, tmp_path)
    assert result.status == "already_submitted"
    assert client.submitted == []


def test_not_a_rebalance_day_asks_no_analysts(ledger, tmp_path):
    result, analyst = run_submit(FakeAlpaca(), ledger, tmp_path, now=TUE_10AM)
    assert result.status == "not_rebalance_day"
    assert analyst.calls == []


def test_dry_run_plans_any_day_and_sends_nothing(ledger, tmp_path):
    client = FakeAlpaca()
    result, _ = run_submit(client, ledger, tmp_path, now=TUE_10AM, dry_run=True)
    assert result.status == "dry_run"
    assert [(o.side, o.ticker, o.quantity) for o in result.plan.orders] == [("buy", "AAPL", 500)]
    assert client.submitted == []
    assert ledger.read_plan("2024-06-11") is None
    assert (ledger.root / "plans" / "2024-06-11.dryrun.json").exists()


def test_rejections_are_recorded(ledger, tmp_path):
    result, _ = run_submit(FakeAlpaca(reject={"AAPL"}), ledger, tmp_path)
    assert result.orders[0].status == "rejected"
    assert ledger.read_plan("2024-06-10")["orders"][0]["status"] == "rejected"


def test_drawdown_breach_halts_and_flattens_any_trading_day(ledger, tmp_path):
    ledger.upsert_nav(nav("2024-06-05", 100_000.0))
    ledger.upsert_nav(nav("2024-06-07", 84_000.0))
    client = FakeAlpaca(positions={"AAPL": 10}, cash=83_000.0)
    result, analyst = run_submit(client, ledger, tmp_path, now=TUE_10AM)
    assert result.status == "flattening"
    assert ledger.is_halted()
    assert analyst.calls == []
    assert client.submitted == [("AAPL", "sell", 10, "paper-test-2024-06-11-AAPL")]


def test_small_drawdown_does_not_halt(ledger, tmp_path):
    ledger.upsert_nav(nav("2024-06-05", 100_000.0))
    ledger.upsert_nav(nav("2024-06-07", 86_000.0))
    result, _ = run_submit(FakeAlpaca(), ledger, tmp_path)
    assert result.status == "submitted"
    assert not ledger.is_halted()


def test_halted_and_flat_stays_put(ledger, tmp_path):
    ledger.halt("manual")
    client = FakeAlpaca()
    result, _ = run_submit(client, ledger, tmp_path)
    assert result.status == "halted"
    assert client.submitted == []


def test_dry_run_never_writes_halted(ledger, tmp_path):
    ledger.upsert_nav(nav("2024-06-05", 100_000.0))
    ledger.upsert_nav(nav("2024-06-07", 80_000.0))
    result, _ = run_submit(FakeAlpaca(positions={"AAPL": 10}), ledger, tmp_path, dry_run=True)
    assert result.status == "dry_run"
    assert result.plan.flatten
    assert not ledger.is_halted()


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------

RECON_SERIES = {"AAPL": {"2024-06-10": 102.0}, "SPY": {"2024-06-10": 505.0}}


def filled(side, qty, price, ticker="AAPL"):
    return OrderResult(client_order_id=f"paper-test-2024-06-10-{ticker}", ticker=ticker, side=side,
                       quantity=qty, status="filled", filled_qty=qty, filled_avg_price=price)


def seed_plan(ledger, marks=None, positions=None):
    ledger.write_plan("2024-06-10", {"status": "submitted", "orders": [],
                                     "plan": {"marks": marks or {"AAPL": 100.0}, "positions": positions or {}}})


def test_reconcile_records_fills_slippage_and_nav(ledger):
    seed_plan(ledger)
    client = FakeAlpaca(positions={"AAPL": 500}, cash=49_500.0, orders=[filled("buy", 500, 101.0)])
    spec = make_fund({})[0].spec
    result = reconcile(spec, client, FakeDataClient(RECON_SERIES), ledger, session="2024-06-10")
    assert result.fills[0].slippage_bps == pytest.approx(100.0)
    assert result.nav.equity == pytest.approx(49_500.0 + 500 * 102.0)
    assert result.nav.long_exposure == pytest.approx(51_000.0)
    assert result.nav.gross == pytest.approx(51_000.0 / 100_500.0, abs=1e-6)
    assert result.nav.benchmark_close == 505.0
    assert result.mismatches == []
    assert ledger.read_fills("2024-06-10")["fills"][0]["fill_price"] == 101.0


def test_reconcile_is_idempotent(ledger):
    client = FakeAlpaca(positions={}, cash=100_000.0)
    spec = make_fund({})[0].spec
    for _ in range(2):
        reconcile(spec, client, FakeDataClient(RECON_SERIES), ledger, session="2024-06-10")
    assert [r.date for r in ledger.nav_rows()] == ["2024-06-10"]


def test_sell_slippage_is_positive_when_filled_below_reference(ledger):
    seed_plan(ledger, positions={"AAPL": 100})
    client = FakeAlpaca(positions={"AAPL": 50}, cash=100_000.0, orders=[filled("sell", 50, 99.0)])
    result = reconcile(make_fund({})[0].spec, client, FakeDataClient(RECON_SERIES), ledger, session="2024-06-10")
    assert result.fills[0].slippage_bps == pytest.approx(100.0)
    assert result.mismatches == []


def test_reconcile_flags_book_mismatch_and_trusts_alpaca(ledger):
    seed_plan(ledger)
    client = FakeAlpaca(positions={"AAPL": 400}, cash=59_500.0, orders=[filled("buy", 500, 101.0)])
    result = reconcile(make_fund({})[0].spec, client, FakeDataClient(RECON_SERIES), ledger, session="2024-06-10")
    assert len(result.mismatches) == 1 and "AAPL" in result.mismatches[0]
    assert result.nav.equity == pytest.approx(59_500.0 + 400 * 102.0)


def test_session_to_reconcile():
    client = FakeAlpaca()
    assert session_to_reconcile(client, datetime(2024, 6, 11, 9, 0, tzinfo=NEW_YORK)) == "2024-06-10"
    assert session_to_reconcile(client, datetime(2024, 6, 8, 17, 0, tzinfo=NEW_YORK)) == "2024-06-07"
    with pytest.raises(ValueError, match="09:30"):
        session_to_reconcile(client, datetime(2024, 6, 11, 9, 31, tzinfo=NEW_YORK))


def test_force_rebalance_trades_mid_week_but_keeps_other_guards(ledger, tmp_path):
    fund, _ = make_fund({"AAPL": 1.0})
    client = FakeAlpaca()
    result = submit(fund, ["AAPL"], client, FakeDataClient(SERIES), ledger, now=TUE_10AM,
                    force_rebalance=True, kill_path=tmp_path / "KILL")
    assert result.status == "submitted"
    assert client.submitted == [("AAPL", "buy", 500, "paper-test-2024-06-11-AAPL")]
    late = submit(fund, ["AAPL"], FakeAlpaca(), FakeDataClient(SERIES), ledger,
                  now=datetime(2024, 6, 11, 15, 51, tzinfo=NEW_YORK), force_rebalance=True, kill_path=tmp_path / "KILL")
    assert late.status == "too_late"


# ---------------------------------------------------------------------------
# retry_rejected
# ---------------------------------------------------------------------------

def test_retry_resends_only_rejected_orders(ledger, tmp_path):
    from hedge_fund.live.runner import retry_rejected
    first = FakeAlpaca(reject={"AAPL"})
    run_submit(first, ledger, tmp_path)
    assert ledger.read_plan("2024-06-10")["orders"][0]["status"] == "rejected"
    again = FakeAlpaca()
    retried = retry_rejected(again, ledger, now=datetime(2024, 6, 10, 14, 0, tzinfo=NEW_YORK), kill_path=tmp_path / "KILL")
    assert [(o.ticker, o.status) for o in retried] == [("AAPL", "accepted")]
    assert again.submitted == [("AAPL", "buy", 500, "paper-test-2024-06-10-AAPL")]
    assert ledger.read_plan("2024-06-10")["orders"][0]["status"] == "accepted"
    assert retry_rejected(FakeAlpaca(), ledger, now=datetime(2024, 6, 10, 14, 0, tzinfo=NEW_YORK), kill_path=tmp_path / "KILL") == []


def test_retry_respects_the_guards(ledger, tmp_path):
    from hedge_fund.live.runner import retry_rejected
    run_submit(FakeAlpaca(reject={"AAPL"}), ledger, tmp_path)
    with pytest.raises(ValueError, match="15:50"):
        retry_rejected(FakeAlpaca(), ledger, now=datetime(2024, 6, 10, 15, 51, tzinfo=NEW_YORK), kill_path=tmp_path / "KILL")
    (tmp_path / "KILL").touch()
    with pytest.raises(ValueError, match="KILL"):
        retry_rejected(FakeAlpaca(), ledger, now=datetime(2024, 6, 10, 14, 0, tzinfo=NEW_YORK), kill_path=tmp_path / "KILL")
    with pytest.raises(ValueError, match="no plan"):
        retry_rejected(FakeAlpaca(), ledger, now=datetime(2024, 6, 11, 10, 0, tzinfo=NEW_YORK), kill_path=tmp_path / "NOKILL")



def test_long_to_short_flip_is_sent_as_close_then_open(ledger, tmp_path):
    fund, _ = make_fund({"AAPL": -1.0})                 # target: short 50% → -500 shares
    client = FakeAlpaca(positions={"AAPL": 5}, cash=99_500.0)
    result = submit(fund, ["AAPL"], client, FakeDataClient(SERIES), ledger,
                    now=MON_10AM, kill_path=tmp_path / "KILL")
    assert result.status == "submitted"
    assert client.submitted == [("AAPL", "sell", 5, "paper-test-2024-06-10-AAPL"),
                                ("AAPL", "sell", 500, "paper-test-2024-06-10-AAPL-open")]
