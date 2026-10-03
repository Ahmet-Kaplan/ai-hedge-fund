"""MarketStore tests — a throwaway SQLite file per test."""

from hedge_fund.data.models import Price
from hedge_fund.data.store import Fact, MarketStore


def bar(day, close):
    return Price(open=close, high=close, low=close, close=close, volume=100, time=f"{day}T00:00:00Z")


def fact(concept, end, value, filed, start=None, form="10-Q", accn="a1", unit="USD"):
    return Fact(concept=concept, unit=unit, start=start, end=end, value=value,
                fy=2026, fp="Q1", form=form, filed=filed, accn=accn)


def test_prices_upsert_is_idempotent_and_range_queries(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    store.upsert_prices("AAPL", [bar("2026-09-28", 1.0), bar("2026-09-29", 2.0)])
    store.upsert_prices("AAPL", [bar("2026-09-29", 3.0)])
    assert [(p.time[:10], p.close) for p in store.prices("AAPL", "2026-09-01", "2026-09-30")] == [
        ("2026-09-28", 1.0), ("2026-09-29", 3.0)]
    assert store.prices("AAPL", "2026-09-29", "2026-09-29")[0].close == 3.0
    assert store.price_range("AAPL") is None
    store.set_price_range("AAPL", "2016-01-01", "2026-09-29")
    assert store.price_range("AAPL") == ("2016-01-01", "2026-09-29")


def test_facts_are_point_in_time_and_replaced_per_company(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    store.replace_facts("1", [
        fact("NetIncomeLoss", "2026-03-31", 10.0, "2026-04-30", start="2026-01-01"),
        fact("NetIncomeLoss", "2026-06-30", 12.0, "2026-07-30", start="2026-04-01", accn="a2"),
        fact("AssetsCurrent", "2026-06-30", 5.0, "2026-07-30", accn="a2"),
    ])
    seen = store.facts("1", ["NetIncomeLoss", "AssetsCurrent"], filed_lte="2026-05-15")
    assert [f.value for f in seen["NetIncomeLoss"]] == [10.0]
    assert seen["AssetsCurrent"] == []
    store.replace_facts("1", [fact("NetIncomeLoss", "2026-06-30", 99.0, "2026-07-30")])
    assert [f.value for f in store.facts("1", ["NetIncomeLoss"], filed_lte="2026-12-31")["NetIncomeLoss"]] == [99.0]


def test_company_filings_sync_and_runs(tmp_path):
    store = MarketStore(tmp_path / "m.db")
    store.upsert_company("BRK.B", "1067983", "Berkshire", 6331, "Fire, Marine & Casualty Insurance")
    assert store.company("BRK.B")["cik"] == "1067983"
    assert store.company("NOPE") is None
    store.replace_filings("1", [
        {"accn": "x1", "form": "8-K", "filed": "2026-07-30", "report_date": "2026-07-30", "items": "2.02,9.01"},
        {"accn": "x2", "form": "10-Q", "filed": "2026-07-31", "report_date": "2026-06-27", "items": ""},
    ])
    assert [f["accn"] for f in store.filings("1", ["8-K"])] == ["x1"]
    assert store.sync_times("1") == (None, None)
    store.mark_synced("1", facts_at="2026-09-30T10:00:00+00:00")
    store.mark_synced("1", submissions_at="2026-09-30T11:00:00+00:00")
    assert store.sync_times("1") == ("2026-09-30T10:00:00+00:00", "2026-09-30T11:00:00+00:00")
    assert store.get_run("k") is None
    store.put_run("k", "fund", "done", '{"x": 1}')
    assert store.get_run("k")["status"] == "done"
    store.put_run("k", "fund", "failed", None, error="boom")
    assert store.get_run("k")["error"] == "boom"
