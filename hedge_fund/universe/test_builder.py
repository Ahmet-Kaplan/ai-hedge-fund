"""Point-in-time universe builder on a synthetic market — no network, no key.

A small SEC + Tiingo world is written in the real cache layouts, so the real
EdgarClient (offline) and TiingoClient (fake HTTP) code paths run:

    BIG  (101)  large, listed throughout
    MID  (102)  mid-size, listed throughout
    DEAD (103)  large, stops filing 2019-05 and trading 2019-09; only its
                filings' cover pages know its symbol (not in today's map)
    NEW  (104)  IPO 2019-02, first 10-Q 2019-05
    NWNM (105)  renamed from OLDN in 2021; its prices live under NWNM
    GLD  (106)  a commodity trust (SIC 6221)
    ERR  (107)  public float tagged 1,000x too large
    GOOGL (1652044) successor of CIK 1288776 after 2015-10-01 (bundled lineage)
"""

from __future__ import annotations

import gzip
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from hedge_fund.data import tiingo as tiingo_mod
from hedge_fund.data.edgar import EdgarClient
from hedge_fund.data.test_tiingo import FakeTiingo, bar, weekdays
from hedge_fund.data.tiingo import TiingoClient
from hedge_fund.universe import (
    PriceBudgetExceeded,
    UniverseBuilder,
    UniverseConfig,
    UniverseSchedule,
    reconstitution_dates,
)
from hedge_fund.universe.builder import _looks_common

STAMP = "2026-09-30T12:00:00-04:00"


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump({"fetched_at": STAMP, "data": data}, fh)


def _d(s):
    return date.fromisoformat(s)


def _plus(s, days):
    return (_d(s) + timedelta(days=days)).isoformat()


class Company:
    def __init__(self, cik, name, *, shares, price, first_period="2012-03-31", last_period="2021-12-31",
                 sic=3571, entity_type="operating", float_multiplier=1.0, floats=None, predecessor_until=None):
        self.cik, self.name, self.shares, self.price = cik, name, shares, price
        self.first_period, self.last_period = first_period, last_period
        self.sic, self.entity_type, self.float_multiplier = sic, entity_type, float_multiplier
        self.floats = floats or {}          # year -> float value override
        self.predecessor_until = predecessor_until

    def periods(self):
        """(start, end, form, filed, accn) for each quarter in range."""
        out, n = [], 0
        for year in range(2011, 2023):
            for q, end in enumerate((f"{year}-03-31", f"{year}-06-30", f"{year}-09-30", f"{year}-12-31"), start=1):
                if not self.first_period <= end <= self.last_period:
                    continue
                form = "10-K" if q == 4 else "10-Q"
                filed = _plus(end, 55 if form == "10-K" else 35)
                if self.predecessor_until and filed > self.predecessor_until:
                    continue
                start = f"{year}-01-01" if form == "10-K" else _plus(end, -89)
                n += 1
                out.append((start, end, form, filed, f"{self.cik:010d}-{year % 100:02d}-{n:06d}"))
        return out

    def float_value(self, year):
        return self.floats.get(year, 0.9 * self.shares * self.price * self.float_multiplier)

    def companyfacts(self):
        rev, ni, eq, cover, flt = [], [], [], [], []
        for start, end, form, filed, accn in self.periods():
            common = {"accn": accn, "form": form, "filed": filed, "fy": int(end[:4]), "fp": "FY" if form == "10-K" else "Q"}
            days = (_d(end) - _d(start)).days + 1
            rev.append({"start": start, "end": end, "val": 1e9 * days / 91, **common})
            ni.append({"start": start, "end": end, "val": 1e8 * days / 91, **common})
            eq.append({"end": end, "val": 5e9, **common})
            cover.append({"end": _plus(filed, -5), "val": self.shares, **common})
            if form == "10-K":
                year = int(end[:4])
                flt.append({"end": f"{year}-06-30", "val": self.float_value(year), **common})
        return {"cik": self.cik, "entityName": self.name, "trim_version": 2, "facts": {
            "us-gaap": {"Revenues": {"units": {"USD": rev}}, "NetIncomeLoss": {"units": {"USD": ni}},
                        "StockholdersEquity": {"units": {"USD": eq}}},
            "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": cover}},
                    "EntityPublicFloat": {"units": {"USD": flt}}},
        }}


def build_world(root: Path, companies, current, cover_symbols, prices):
    """SEC cache under root/edgar, Tiingo histories returned for the fake server."""
    edgar = root / "edgar"
    for c in companies:
        _write(edgar / f"companyfacts/CIK{c.cik:010d}.json.gz", c.companyfacts())
        _write(edgar / f"submissions/CIK{c.cik:010d}.json.gz",
               {"cik": str(c.cik), "name": c.name, "sic": str(c.sic), "entityType": c.entity_type, "tickers": []})
    _write(edgar / "company_tickers.json.gz", current)
    for c in companies:
        for start, end, form, filed, accn in c.periods():
            if c.cik in cover_symbols:
                _write(edgar / f"symbols/{c.cik}/{accn}.json.gz", cover_symbols[c.cik](filed))
    for year in range(2012, 2023):
        for q, end in enumerate(("03-31", "06-30", "09-30", "12-31"), start=1):
            rows = [{"cik": c.cik, "entityName": c.name, "end": f"{year}-{end}", "val": c.float_value(year),
                     "accn": "x"} for c in companies if c.first_period[:4] <= str(year) <= c.last_period[:4]]
            _write(edgar / f"frames/dei/EntityPublicFloat/USD/CY{year}Q{q}I.json.gz", rows)
    return {sym: [bar(d, px) for d, px in series.items()] for sym, series in prices.items()}


DAYS = weekdays("2012-01-02", "2022-06-30")


def flat(price, start="2012-01-02", end="2022-06-30"):
    return {d: price for d in DAYS if start <= d <= end}


COMPANIES = [
    Company(101, "BIG CORP", shares=1e9, price=100.0),
    Company(102, "MID CORP", shares=1e9, price=20.0),
    Company(103, "DEAD CORP", shares=1e9, price=80.0, last_period="2019-03-31"),
    Company(104, "NEW CORP", shares=1e9, price=60.0, first_period="2019-03-31"),
    Company(105, "RENAMED INC", shares=1e9, price=40.0),
    Company(106, "GOLD TRUST", shares=1e9, price=150.0, sic=6221),
    Company(107, "ERROR CORP", shares=1e6, price=10.0, float_multiplier=1000.0),
    Company(1288776, "GOOGLE INC", shares=1e9, price=70.0, predecessor_until="2015-10-01"),
    Company(1652044, "ALPHABET INC", shares=1e9, price=70.0, first_period="2015-09-30"),
]
CURRENT = {"BIG": 101, "MID": 102, "NEW": 104, "NWNM": 105, "GLD": 106, "ERR": 107, "GOOGL": 1652044, "GOOG": 1652044}
COVER = {103: lambda filed: ["DEAD"], 105: lambda filed: ["OLDN" if filed < "2021-01-01" else "NWNM"]}
PRICES = {
    "BIG": flat(100.0), "MID": flat(20.0), "DEAD": flat(80.0, end="2019-09-30"), "NEW": flat(60.0, start="2019-02-01"),
    "NWNM": flat(40.0), "SPY": flat(300.0), "GLD": flat(150.0), "ERR": flat(10.0), "GOOGL": flat(70.0), "GOOG": flat(70.0),
}


@pytest.fixture
def world(tmp_path, monkeypatch):
    histories = build_world(tmp_path, COMPANIES, CURRENT, COVER, PRICES)
    server = FakeTiingo(histories)
    monkeypatch.setattr(tiingo_mod.requests, "Session", lambda: server)
    tiingo_mod.clear_process_cache()
    yield tmp_path, server
    tiingo_mod.clear_process_cache()


def make(root: Path, config=None, **kw):
    tiingo = TiingoClient(api_key="fixture", cache_dir=root / "tiingo", max_age_hours=None)
    edgar = EdgarClient(cache_dir=root / "edgar", offline=True, today="2026-09-30", price_source=tiingo)
    return UniverseBuilder(edgar, tiingo, config or UniverseConfig(top_n=4, pool_multiple=2.0),
                           cache_dir=root / "universe", **kw)


def reasons(snap):
    return {e.cik: e.reason for e in snap.excluded}


# ---------------------------------------------------------------------------
# Eligibility over time
# ---------------------------------------------------------------------------

def test_members_ranked_by_point_in_time_market_cap(world):
    snap = make(world[0]).snapshot("2018-07-02")
    assert snap.tickers == ["BIG", "DEAD", "GOOGL", "NWNM"]
    assert [m.rank for m in snap.members] == [1, 2, 3, 4]
    big = snap.members[0]
    assert big.market_cap == pytest.approx(1e9 * 100.0) and big.price_date == "2018-07-02"
    assert reasons(snap)[102] == "below_top_n"


def test_newly_listed_company_is_not_eligible_before_it_has_history(world):
    b = make(world[0])
    assert 104 not in {m.cik for m in b.snapshot("2019-07-01").members}
    assert reasons(b.snapshot("2019-07-01"))[104] in ("insufficient_filings", "not_yet_filing")
    assert reasons(b.snapshot("2018-07-02")).get(104) is None     # not even discovered: no frames yet
    later = make(world[0], UniverseConfig(top_n=8)).snapshot("2020-07-01")
    assert 104 in {m.cik for m in later.members}


def test_delisted_company_included_before_and_excluded_after(world):
    b = make(world[0])
    before = b.snapshot("2019-07-01")
    dead = next(m for m in before.members if m.cik == 103)
    assert (dead.ticker, dead.symbol_source) == ("DEAD", "cover_page")
    assert b.edgar.resolve("DEAD", "2019-07-01").cik == 103          # registered for the agents
    after = b.snapshot("2020-07-01")
    assert 103 not in {m.cik for m in after.members}
    assert reasons(after)[103] == "stopped_filing"


def test_recently_listed_prices_are_required(tmp_path, monkeypatch):
    # MID files with the SEC for years but only starts trading on 2019-04-01
    histories = build_world(tmp_path, COMPANIES, CURRENT, COVER, {**PRICES, "MID": flat(20.0, start="2019-04-01")})
    monkeypatch.setattr(tiingo_mod.requests, "Session", lambda: FakeTiingo(histories))
    tiingo_mod.clear_process_cache()
    b = make(tmp_path, UniverseConfig(top_n=8))
    assert reasons(b.snapshot("2019-05-15"))[102] == "recently_listed"
    assert 102 in {m.cik for m in b.snapshot("2019-07-01").members}      # 91 days later


def test_excluded_security_types_and_scale_errors(world):
    snap = make(world[0], UniverseConfig(top_n=8)).snapshot("2019-07-01")
    r = reasons(snap)
    assert r[106] == "excluded_security_type"
    assert r[107] in ("implausible_float", "float_market_cap_mismatch")   # 1,000x float never ranks
    assert {106, 107}.isdisjoint(m.cik for m in snap.members)


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------

def _truncate(root: Path, cutoff: str) -> None:
    for path in (root / "edgar" / "companyfacts").glob("*.json.gz"):
        with gzip.open(path, "rt") as fh:
            payload = json.load(fh)
        for ns in payload["data"]["facts"].values():
            for body in ns.values():
                for unit, rows in body["units"].items():
                    body["units"][unit] = [r for r in rows if r["filed"] <= cutoff]
        with gzip.open(path, "wt") as fh:
            json.dump(payload, fh)


@pytest.mark.parametrize("as_of", ["2016-07-01", "2018-07-02", "2019-07-01", "2020-07-01"])
def test_universe_is_unchanged_when_everything_later_is_deleted(tmp_path, monkeypatch, as_of):
    """Delete every SEC fact filed after D and every price after D: the
    universe as of D must not change by a single field."""
    def build(root, truncate):
        histories = build_world(root, COMPANIES, CURRENT, COVER, PRICES)
        if truncate:
            _truncate(root, as_of)
            histories = {s: [r for r in rows if r["date"][:10] <= as_of] for s, rows in histories.items()}
        monkeypatch.setattr(tiingo_mod.requests, "Session", lambda: FakeTiingo(histories))
        tiingo_mod.clear_process_cache()
        return make(root).snapshot(as_of).model_dump()

    assert build(tmp_path / "full", False) == build(tmp_path / "cut", True)


def test_a_float_filed_after_the_date_is_invisible(tmp_path, monkeypatch):
    # MID's FY2018 10-K (filed 2019-02-24) reports a huge float; before that
    # filing it must not lift MID into the universe
    mid = Company(102, "MID CORP", shares=1e9, price=20.0, floats={2018: 5e12})
    companies = [c if c.cik != 102 else mid for c in COMPANIES]
    histories = build_world(tmp_path, companies, CURRENT, COVER, PRICES)
    monkeypatch.setattr(tiingo_mod.requests, "Session", lambda: FakeTiingo(histories))
    tiingo_mod.clear_process_cache()
    b = make(tmp_path, UniverseConfig(top_n=4, pool_multiple=1.0))
    assert reasons(b.snapshot("2019-02-22")).get(102) in ("outside_float_pool", "below_top_n")
    assert reasons(b.snapshot("2019-02-25"))[102] in ("implausible_float", "float_market_cap_mismatch")


def test_price_after_the_date_never_used(world):
    snap = make(world[0]).snapshot("2019-07-03")
    assert all(m.price_date <= "2019-07-03" for m in snap.members)


# ---------------------------------------------------------------------------
# Splits, ticker changes, lineage, duplicates
# ---------------------------------------------------------------------------

def test_split_after_the_last_filing_is_applied_to_shares(tmp_path, monkeypatch):
    prices = dict(PRICES)
    split_day = "2018-06-01"   # after the 2018-05-05 10-Q, before D
    histories = build_world(tmp_path, COMPANIES, CURRENT, COVER, prices)
    histories["BIG"] = [bar(d, 50.0 if d >= split_day else 100.0, split=2.0 if d == split_day else 1.0)
                        for d in DAYS]
    monkeypatch.setattr(tiingo_mod.requests, "Session", lambda: FakeTiingo(histories))
    tiingo_mod.clear_process_cache()
    big = next(m for m in make(tmp_path).snapshot("2018-07-02").members if m.cik == 101)
    assert (big.price, big.shares) == (50.0, 2e9)
    assert big.market_cap == pytest.approx(1e11)       # unchanged by the split


def test_renamed_company_appears_once_under_its_current_symbol(world):
    b = make(world[0], UniverseConfig(top_n=8))
    for d in ("2018-07-02", "2022-01-03"):
        snap = b.snapshot(d)
        entries = [m for m in snap.members if m.cik == 105]
        assert len(entries) == 1 and entries[0].ticker == "NWNM"
        assert "OLDN" not in snap.tickers


def test_predecessor_is_merged_into_successor(world):
    b = make(world[0], UniverseConfig(top_n=8))
    early = b.snapshot("2015-07-01")
    assert reasons(early)[1652044] == "successor_not_yet"
    google = [m for m in early.members if m.cik in (1288776, 1652044)]
    assert [(m.cik, m.ticker) for m in google] == [(1288776, "GOOGL")]    # history under the successor's symbol
    late = b.snapshot("2016-07-01")
    assert reasons(late)[1288776] == "predecessor"
    alphabet = [m for m in late.members if m.cik == 1652044]
    assert len(alphabet) == 1 and alphabet[0].ticker == "GOOGL"      # one company, curated symbol
    assert "GOOG" not in late.tickers


def test_recycled_symbol_is_rejected_or_deduplicated(tmp_path, monkeypatch):
    # CIK 108: a dead company whose cover symbol "BIG" now belongs to CIK 101
    ghost = Company(108, "OLD BIG CO", shares=1e9, price=99.0, last_period="2019-03-31")
    tiny = Company(109, "TINY CO", shares=1e6, price=1.0, float_multiplier=2e4)   # cover symbol MID: mismatch
    companies = COMPANIES + [ghost, tiny]
    cover = {**COVER, 108: lambda filed: ["BIG"], 109: lambda filed: ["MID"]}
    histories = build_world(tmp_path, companies, CURRENT, cover, PRICES)
    monkeypatch.setattr(tiingo_mod.requests, "Session", lambda: FakeTiingo(histories))
    tiingo_mod.clear_process_cache()
    snap = make(tmp_path, UniverseConfig(top_n=10)).snapshot("2019-07-01")
    assert [m.ticker for m in snap.members].count("BIG") == 1
    assert next(m for m in snap.members if m.ticker == "BIG").cik == 101
    assert reasons(snap)[108] == "duplicate_symbol"
    assert reasons(snap)[109] in ("float_market_cap_mismatch", "implausible_float")


def test_one_entry_per_company_and_per_symbol(world):
    b = make(world[0], UniverseConfig(top_n=20))
    for d in ("2015-07-01", "2016-07-01", "2019-07-01", "2021-07-01"):
        snap = b.snapshot(d)
        assert len({m.cik for m in snap.members}) == len(snap.members) == len(set(snap.tickers))


# ---------------------------------------------------------------------------
# Cache, budget, reproducibility
# ---------------------------------------------------------------------------

def test_rebuild_is_identical_and_free(world):
    root, server = world
    first = make(root).schedule(["2018-07-02", "2019-07-01"])
    calls = len(server.calls)
    tiingo_mod.clear_process_cache()
    again = make(root)
    second = again.schedule(["2019-07-01", "2018-07-02"])
    assert second.model_dump_json() == first.model_dump_json()
    assert len(server.calls) == calls and again.tiingo.requests == 0 and again.edgar.requests == 0


def test_full_price_history_downloaded_once_per_symbol(world):
    root, server = world
    make(root).schedule(["2016-07-01", "2017-07-03", "2018-07-02", "2019-07-01"])
    symbols = [url.rsplit("/daily/", 1)[1].split("/")[0] for url, _ in server.calls]
    assert len(symbols) == len(set(symbols))


def test_price_budget_stops_cleanly_and_resumes(world):
    root, server = world
    with pytest.raises(PriceBudgetExceeded):
        make(root, max_price_downloads=2).snapshot("2019-07-01")
    first_calls = len(server.calls)
    assert first_calls == 2
    snap = make(root, max_price_downloads=50).snapshot("2019-07-01")
    assert snap.members and len(server.calls) > first_calls
    symbols = [url.rsplit("/daily/", 1)[1].split("/")[0] for url, _ in server.calls]
    assert len(symbols) == len(set(symbols))                     # nothing downloaded twice


def test_config_change_gets_its_own_cache(world):
    root, _ = world
    a = make(root, UniverseConfig(top_n=2)).snapshot("2019-07-01")
    b = make(root, UniverseConfig(top_n=3)).snapshot("2019-07-01")
    assert len(a.members) == 2 and len(b.members) == 3 and a.config_digest != b.config_digest


# ---------------------------------------------------------------------------
# Schedule and backtest integration
# ---------------------------------------------------------------------------

def test_schedule_members_on_uses_latest_snapshot_not_later_ones():
    from hedge_fund.universe.models import UniverseSnapshot
    snaps = [UniverseSnapshot(as_of=d, config_digest="x", candidates=0, members=[]) for d in ("2019-07-01", "2020-07-01")]
    schedule = UniverseSchedule(config=UniverseConfig(), snapshots=snaps)
    assert schedule.members_on("2019-06-28") == [] and schedule.members_on("2020-06-30") == []


def test_reconstitution_dates():
    sessions = weekdays("2019-06-01", "2021-08-31")
    assert reconstitution_dates(sessions, "2019-06-29", "2021-07-15") == ["2019-07-01", "2020-06-29", "2021-06-29"]
    assert reconstitution_dates(weekdays("2020-01-01", "2020-12-31"), "2020-01-31", "2020-04-30", "quarterly") == [
        "2020-01-31", "2020-04-30"]
    assert reconstitution_dates(sessions, "2019-07-01", "2019-12-31", "quarterly") == ["2019-07-01", "2019-10-01"]
    with pytest.raises(ValueError):
        reconstitution_dates(sessions, "2019-07-01", "2019-12-31", "weekly")


@pytest.mark.parametrize("ticker, ok", [("AAPL", True), ("BRK.B", True), ("GOOGL", True), ("JPM.PC", False),
                                         ("XYZ.WS", False), ("ABCDEF", False), ("BRK27", False), ("A1B", False)])
def test_common_stock_symbol_filter(ticker, ok):
    assert _looks_common(ticker) is ok


def test_backtest_follows_the_schedule(world, monkeypatch):
    from hedge_fund.backtesting.fund import backtest_fund
    from hedge_fund.backtesting.test_fund import FakeAnalyst
    from hedge_fund.data.factory import CompositeDataClient
    from hedge_fund.fund.spec import Fund, FundSpec
    from hedge_fund.signals import ALPHA_MODEL_REGISTRY

    root, _ = world
    monkeypatch.setitem(ALPHA_MODEL_REGISTRY, "a", FakeAnalyst)
    b = make(root, UniverseConfig(top_n=2))
    schedule = b.schedule(["2019-07-01", "2020-07-01"])
    assert schedule.snapshots[0].tickers == ["BIG", "DEAD"] and "DEAD" not in schedule.snapshots[1].tickers
    spec = FundSpec(schema_version=2, name="t", capital=100_000.0, rebalance="monthly",
                    strategies=[{"name": "solo", "models": [{"name": "a"}], "blend": {"mode": "long_short"}}],
                    risk={"max_position_pct": 0.5, "max_gross_exposure": 1.0})
    views = {t: 1.0 for t in ("BIG", "DEAD", "NWNM", "GOOGL", "MID")}
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views=views)]})
    result = backtest_fund(fund, "2019-07-01", "2020-12-31", CompositeDataClient(b.tiingo, b.edgar), schedule)
    assert result.universe_schedule == {s.as_of: s.tickers for s in schedule.snapshots}
    for r in result.records:
        assert set(r.universe) <= set(schedule.members_on(r.as_of))
    assert all("DEAD" not in r.positions for r in result.records if r.execution_as_of > "2020-07-01")
