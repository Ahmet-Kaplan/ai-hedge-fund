"""MarketPanel and its as-of view.

Synthetic Tiingo histories served by the fake Tiingo server from
data/test_tiingo.py, so the real TiingoClient code path (store, split
adjustment, dividends, raw closes) is exercised with no network and no key.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hedge_fund.data import tiingo as tiingo_mod
from hedge_fund.data.security_events import SecurityEvent
from hedge_fund.data.test_tiingo import KEY, bar, fake, weekdays  # noqa: F401  (fixture)
from hedge_fund.data.tiingo import TiingoClient
from hedge_fund.systematic import AsOfView, FutureDataError, MarketPanel
from hedge_fund.universe.builder import load_schedule

DAYS = weekdays("2023-01-02", "2023-03-31")
LAST = "2023-02-15"          # DLST's last real trading day
HALT = "2023-01-25"          # AAA prints with zero volume
DIV_DAY = "2023-01-17"       # AAA pays $1.00 per pre-split share
SPLIT_DAY = "2023-02-01"     # AAA splits 2-for-1
RECON = "2023-02-01"         # universe reconstitution: DLST out, BBB in


def _aaa(day: str) -> dict:
    raw = 100.0 + DAYS.index(day)                    # rising, so every bar is distinct
    if day >= SPLIT_DAY:
        raw /= 2
    return bar(day, raw, split=2.0 if day == SPLIT_DAY else 1.0,
               div=1.0 if day == DIV_DAY else 0.0, volume=0 if day == HALT else 1000)


def histories() -> dict[str, list[dict]]:
    return {
        "SPY": [bar(d, 400.0) for d in DAYS],
        "AAA": [_aaa(d) for d in DAYS],
        "BBB": [bar(d, 20.0 + 0.1 * i) for i, d in enumerate(DAYS)],
        # Real trades until LAST, then the vendor keeps printing the last close.
        "DLST": [bar(d, 50.0 + (0.5 * i if d < LAST else 0.0)) if d <= LAST else bar(d, 50.0, volume=100)
                 for i, d in enumerate(DAYS)],
    }


class Schedule:
    """members_on/all_tickers, like UniverseSchedule, with one reconstitution."""

    def members_on(self, day: str) -> list[str]:
        return ["AAA", "DLST"] if day < RECON else ["AAA", "BBB"]

    def all_tickers(self) -> list[str]:
        return ["AAA", "DLST", "BBB"]


class Data:
    """TiingoClient plus curated delisting events — the CompositeDataClient
    surface the panel reads, without SEC."""

    def __init__(self, tiingo: TiingoClient, events: dict[str, SecurityEvent] | None = None,
                 hide_after: str | None = None) -> None:
        self.tiingo, self.events, self.hide_after = tiingo, events or {}, hide_after

    def _end(self, end: str) -> str:
        return min(end, self.hide_after) if self.hide_after else end

    def get_prices(self, ticker, start_date, end_date, **kw):
        return self.tiingo.get_prices(ticker, start_date, self._end(end_date), **kw)

    def raw_closes(self, ticker, start_date, end_date):
        return self.tiingo.raw_closes(ticker, start_date, self._end(end_date))

    def dividends(self, ticker, start_date, end_date):
        return self.tiingo.dividends(ticker, start_date, self._end(end_date))

    def split_events(self, ticker, start_date, end_date):
        return self.tiingo.split_events(ticker, start_date, end_date)

    def security_event(self, ticker):
        return self.events.get(ticker)


DLST_EVENT = {"DLST": SecurityEvent(ticker="DLST", last_trading_day=LAST, event_type="acquisition")}


@pytest.fixture
def served(fake):  # noqa: F811  (the fake Tiingo server fixture)
    fake.histories.update(histories())
    return fake


@pytest.fixture
def client(tmp_path, served):
    with TiingoClient(api_key=KEY, cache_dir=tmp_path / "tiingo", max_age_hours=None) as c:
        yield c


def build(client, hide_after=None, **kw) -> MarketPanel:
    kw.setdefault("schedule", Schedule())
    return MarketPanel.build(Data(client, DLST_EVENT, hide_after), None, DAYS[0], DAYS[-1], **kw)


# ---------------------------------------------------------------------------
# 1. Future-data access raises
# ---------------------------------------------------------------------------

def test_every_read_after_the_as_of_date_raises(client):
    view = build(client).as_of("2023-02-10")
    future = "2023-02-13"
    with pytest.raises(FutureDataError):
        view.bars("close", end=future)
    with pytest.raises(FutureDataError):
        view.bars("volume", start=future)
    with pytest.raises(FutureDataError):
        view.history("AAA", end=future)
    with pytest.raises(FutureDataError):
        view.close("AAA", future)
    with pytest.raises(FutureDataError):
        view.tradable(end=future)
    with pytest.raises(FutureDataError):
        view.members(future)
    with pytest.raises(FutureDataError):
        view.membership(end=future)
    # the as-of day itself is readable
    assert view.close("AAA", "2023-02-10") == view.close("AAA")


def test_a_view_of_an_incomplete_day_is_refused(client):
    panel = build(client)
    with pytest.raises(FutureDataError):
        panel.as_of("2999-01-01")
    with pytest.raises(ValueError):
        panel.as_of("2022-12-30")              # before the first session


def test_a_view_cannot_be_constructed_holding_later_rows(client):
    panel = build(client)
    frames = {f: df for f, df in panel._frames.items()}
    with pytest.raises(FutureDataError):
        AsOfView(as_of="2023-02-10", session="2023-02-10", frames=frames,
                 tradable=panel._tradable, members=panel._members)


# ---------------------------------------------------------------------------
# 2. No strategy can obtain bars after its as-of date
# ---------------------------------------------------------------------------

def _stored_frames(view: AsOfView) -> list[pd.DataFrame]:
    """Every frame the view object holds, found by walking its slots."""
    frames = [*view._frames.values(), view._tradable]
    return frames + ([view._members] if view._members is not None else [])


@pytest.mark.parametrize("as_of", ["2023-01-02", HALT, "2023-02-11", LAST, "2023-03-31"])
def test_a_view_holds_and_returns_nothing_after_its_session(client, as_of):
    view = build(client).as_of(as_of)
    assert view.session <= as_of
    assert all(df.index.max() <= view.session for df in _stored_frames(view))
    for field in ("open", "high", "low", "close", "volume", "dividend"):
        for tradable_only in (True, False):
            assert view.bars(field, tradable_only=tradable_only).index.max() == view.session
    assert view.history("AAA").index.max() == view.session
    assert view.sessions[-1] == view.session


def _greedy_strategy(view: AsOfView) -> dict:
    """Reads everything it is allowed to; its output is all it could see."""
    last = {t: view.close(t) for t in view.tickers}
    out = {"members": view.members(), "last": {t: None if np.isnan(v) else round(v, 10) for t, v in last.items()}}
    for field in ("open", "high", "low", "close", "volume", "dividend"):
        frame = view.bars(field, tradable_only=False)
        out[field] = frame.round(10).fillna(-1).to_dict()
    out["tradable"] = view.tradable().to_dict()
    return out


@pytest.mark.parametrize("as_of", ["2023-01-05", DIV_DAY, HALT, "2023-01-31", SPLIT_DAY, LAST, "2023-03-01"])
def test_strategy_output_is_identical_whether_or_not_the_future_exists(tmp_path, served, as_of):
    """Build one panel from the full vendor history and one from a vendor
    whose history ends at as_of (no later bars, splits, dividends or
    placeholder prints exist anywhere). Any strategy must see exactly the same
    thing through either view — including before the 2:1 split, where a naive
    split-adjusted series would already reveal it."""
    tiingo_mod.clear_process_cache()
    with TiingoClient(api_key=KEY, cache_dir=tmp_path / "full", max_age_hours=None) as c:
        full = MarketPanel.build(Data(c, DLST_EVENT), None, DAYS[0], DAYS[-1], schedule=Schedule())
        seen_full = _greedy_strategy(full.as_of(as_of))

    served.histories.clear()
    served.histories.update({s: [r for r in rows if r["date"][:10] <= as_of] for s, rows in histories().items()})
    tiingo_mod.clear_process_cache()
    with TiingoClient(api_key=KEY, cache_dir=tmp_path / "cut", max_age_hours=None) as c:
        cut = MarketPanel.build(Data(c, DLST_EVENT), None, DAYS[0], DAYS[-1], schedule=Schedule())
        seen_cut = _greedy_strategy(cut.as_of(as_of))
    assert seen_full == seen_cut


def test_pre_split_view_shows_prices_as_traded(client):
    """Before the split the view shows the raw price level of the time."""
    view = build(client).as_of("2023-01-31")
    assert view.close("AAA") == pytest.approx(100.0 + DAYS.index("2023-01-31"))
    later = build(client).as_of("2023-03-31")
    assert later.close("AAA", "2023-01-31") == pytest.approx((100.0 + DAYS.index("2023-01-31")) / 2)


def test_returned_frames_are_copies(client):
    panel = build(client)
    view = panel.as_of(LAST)
    frame = view.bars("close")
    frame.iloc[:, :] = -1.0
    assert (view.bars("close").fillna(0) >= 0).all().all()
    assert (panel.as_of(LAST).bars("close").fillna(0) >= 0).all().all()


# ---------------------------------------------------------------------------
# 3. Delisted securities
# ---------------------------------------------------------------------------

def test_delisted_name_is_untradable_after_its_last_day(client):
    view = build(client).as_of("2023-03-31")
    mask = view.tradable(tickers=["DLST"])["DLST"]
    assert mask.loc[:LAST].all()
    assert not mask.loc[mask.index > LAST].any()
    close = view.bars("close", tickers=["DLST"])["DLST"]
    assert close.loc[LAST] == pytest.approx(50.0)
    assert close.loc[close.index > LAST].isna().all()
    assert np.isnan(view.close("DLST"))
    # the vendor's placeholder prints are kept for audit, not for trading
    raw = view.bars("close", tickers=["DLST"], tradable_only=False)["DLST"]
    assert (raw.loc[raw.index > LAST] == 50.0).all()


def test_delisting_event_is_not_used_before_it_happened(client):
    """Before LAST the event cannot mask anything: the view before the
    delisting sees DLST as an ordinary tradable name."""
    view = build(client).as_of("2023-02-10")
    assert view.tradable(tickers=["DLST"])["DLST"].all()
    assert view.close("DLST") > 0


# ---------------------------------------------------------------------------
# 4. Halted / stale placeholder prints are masked
# ---------------------------------------------------------------------------

def test_zero_volume_halt_print_is_masked(client):
    view = build(client).as_of("2023-03-31")
    assert not view.tradable(tickers=["AAA"])["AAA"].loc[HALT]
    assert np.isnan(view.bars("close", tickers=["AAA"])["AAA"].loc[HALT])
    assert np.isnan(view.close("AAA", HALT))
    assert view.bars("close", tickers=["AAA"], tradable_only=False)["AAA"].loc[HALT] > 0
    others = view.tradable(tickers=["AAA"])["AAA"].drop(HALT)
    assert others.all()


# ---------------------------------------------------------------------------
# 5. Point-in-time universe membership
# ---------------------------------------------------------------------------

def test_membership_follows_the_schedule_and_never_leads_it(client):
    panel = build(client)
    before = panel.as_of("2023-01-31")
    assert before.members() == ["AAA", "DLST"]
    assert "BBB" not in before.members()             # joins at RECON, not earlier
    after = panel.as_of("2023-03-31")
    assert after.members() == ["AAA", "BBB"]
    assert after.members("2023-01-31") == ["AAA", "DLST"]
    grid = after.membership()
    for day in grid.index:
        assert sorted(t for t in grid.columns if grid.loc[day, t]) == sorted(Schedule().members_on(day))


def test_membership_matches_the_committed_top10_schedule(tmp_path, served):
    """Every session's mask equals UniverseSchedule.members_on for the
    Buffett baseline's committed universe (prices are irrelevant here)."""
    path = Path(__file__).resolve().parents[2] / "runs" / "buffett-baseline" / "universe_top10.json"
    if not path.exists():
        pytest.skip("committed universe file not present")
    schedule = load_schedule(path)
    served.histories["SPY"] = [bar(d, 400.0) for d in weekdays("2016-06-01", "2026-06-30")]
    tiingo_mod.clear_process_cache()
    with TiingoClient(api_key=KEY, cache_dir=tmp_path / "u", max_age_hours=None) as c:
        panel = MarketPanel.build(Data(c), None, "2016-06-01", "2026-06-30", schedule=schedule)
    view = panel.as_of("2026-06-30")
    grid = view.membership()
    for day in grid.index:
        assert sorted(t for t in grid.columns if grid.loc[day, t]) == sorted(schedule.members_on(day)), day
    assert view.members("2016-06-30") == []                  # before the first snapshot
    assert "GE" in view.members("2016-07-01") and "GE" not in view.members("2017-07-03")


# ---------------------------------------------------------------------------
# 6. Warm-cache construction makes zero network requests
# ---------------------------------------------------------------------------

def test_warm_cache_build_makes_no_requests(tmp_path, served):
    cache = tmp_path / "warm"
    with TiingoClient(api_key=KEY, cache_dir=cache, max_age_hours=None) as cold:
        MarketPanel.build(Data(cold, DLST_EVENT), None, DAYS[0], DAYS[-1], schedule=Schedule())
        assert cold.requests == 4                            # SPY, AAA, DLST, BBB once each
    calls = len(served.calls)

    tiingo_mod.clear_process_cache()                         # a new process: disk store only
    with TiingoClient(api_key=KEY, cache_dir=cache, max_age_hours=None) as warm:
        again = MarketPanel.build(Data(warm, DLST_EVENT), None, DAYS[0], DAYS[-1], schedule=Schedule())
        assert warm.requests == 0
    assert len(served.calls) == calls

    tiingo_mod.clear_process_cache()
    with TiingoClient(cache_dir=cache, offline=True) as offline:  # no key, cannot fetch
        same = MarketPanel.build(Data(offline, DLST_EVENT), None, DAYS[0], DAYS[-1], schedule=Schedule())
    a, b = again.as_of(DAYS[-1]), same.as_of(DAYS[-1])
    pd.testing.assert_frame_equal(a.bars("close", tradable_only=False), b.bars("close", tradable_only=False))


# ---------------------------------------------------------------------------
# Prices and dividends share one split-adjusted basis
# ---------------------------------------------------------------------------

def test_dividend_is_restated_per_split_adjusted_share(client):
    view = build(client).as_of("2023-03-31")
    div = view.bars("dividend", tickers=["AAA"])["AAA"]
    assert div.loc[DIV_DAY] == pytest.approx(0.5)            # $1.00 before a 2:1 split
    assert (div.drop(DIV_DAY) == 0).all()
    close = view.bars("close", tickers=["AAA"])["AAA"]
    raw_before = 100.0 + DAYS.index("2023-01-31")
    assert close.loc["2023-01-31"] == pytest.approx(raw_before / 2)
    # seen before the split, the same dividend is the $1.00 actually paid
    assert build(client).as_of("2023-01-31").bars("dividend", tickers=["AAA"])["AAA"].loc[DIV_DAY] == pytest.approx(1.0)
    # a dividend is invisible before its ex-date
    early = build(client).as_of("2023-01-13").bars("dividend", tickers=["AAA"])["AAA"]
    assert (early == 0).all()


def test_panel_calendar_is_the_benchmark_sessions(client):
    panel = build(client)
    assert panel.info.sessions == len(DAYS)
    assert panel.sessions_through("2023-01-08") == ["2023-01-02", "2023-01-03", "2023-01-04",
                                                   "2023-01-05", "2023-01-06"]
    view = panel.as_of("2023-01-08")                          # a Sunday
    assert view.session == "2023-01-06"


# ---------------------------------------------------------------------------
# Real local cache (skipped when absent): offline, so no request is possible
# ---------------------------------------------------------------------------

def test_real_cache_top10_panel_builds_offline():
    cache = Path.home() / ".hedge-fund" / "cache" / "tiingo"
    path = Path(__file__).resolve().parents[2] / "runs" / "buffett-baseline" / "universe_top10.json"
    if not (cache / "SPY.json.gz").exists() or not path.exists():
        pytest.skip("no local Tiingo store / committed universe")
    schedule = load_schedule(path)
    tiingo_mod.clear_process_cache()
    try:
        with TiingoClient(cache_dir=cache, offline=True) as c:
            panel = MarketPanel.build(Data(c), None, "2016-07-01", "2026-06-30", schedule=schedule)
            assert c.requests == 0
    except tiingo_mod.TiingoClientError as exc:
        pytest.skip(f"local store incomplete: {exc}")
    finally:
        tiingo_mod.clear_process_cache()
    view = panel.as_of("2026-06-30")
    assert view.members() == [t for t in panel.info.tickers if t in schedule.members_on(view.session)]
    assert view.bars("close").index.max() == view.session
