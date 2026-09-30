from hedge_fund.live.ledger import Ledger, NavRow


def nav(day, equity, benchmark=500.0):
    return NavRow(date=day, equity=equity, cash=equity, long_exposure=0.0,
                  short_exposure=0.0, gross=0.0, benchmark_close=benchmark)


def test_nav_upsert_is_idempotent_and_sorted(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.upsert_nav(nav("2024-06-11", 101.0))
    ledger.upsert_nav(nav("2024-06-10", 100.0))
    ledger.upsert_nav(nav("2024-06-11", 99.0))
    assert [(r.date, r.equity) for r in ledger.nav_rows()] == [("2024-06-10", 100.0), ("2024-06-11", 99.0)]
    assert ledger.peak_equity() == 100.0
    assert ledger.latest_equity() == 99.0


def test_empty_ledger(tmp_path):
    ledger = Ledger(tmp_path)
    assert ledger.nav_rows() == []
    assert ledger.peak_equity() is None
    assert ledger.latest_equity() is None
    assert ledger.read_plan("2024-06-10") is None
    assert ledger.plan_sessions() == []


def test_plans_and_dry_runs_are_separate(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.write_plan("2024-06-10", {"a": 1})
    ledger.write_plan("2024-06-11", {"b": 2}, dry_run=True)
    assert ledger.read_plan("2024-06-10") == {"a": 1}
    assert ledger.read_plan("2024-06-11") is None
    assert ledger.plan_sessions() == ["2024-06-10"]


def test_fills_round_trip(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.write_fills("2024-06-10", {"fills": []})
    assert ledger.read_fills("2024-06-10") == {"fills": []}
    assert ledger.fill_sessions() == ["2024-06-10"]


def test_halt_and_resume(tmp_path):
    ledger = Ledger(tmp_path)
    assert not ledger.is_halted()
    ledger.halt("drawdown")
    assert ledger.is_halted()
    assert "drawdown" in ledger.halted_path.read_text()
    assert ledger.resume() is True
    assert ledger.resume() is False


def test_log_path_creates_dir(tmp_path):
    path = Ledger(tmp_path).log_path("2024-06-10", "submit")
    assert path.parent.is_dir()
    assert path.name == "2024-06-10-submit.log"
