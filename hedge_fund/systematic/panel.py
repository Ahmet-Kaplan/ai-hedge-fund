"""Market panel — daily bars for a set of names, read only through an as-of view.

The systematic (price-based) strategies read the market as one panel instead
of one `get_prices` call per ticker per date. The panel is built once from the
same sources the rest of the fund uses, and nothing is fetched here that the
data client would not fetch anyway:

    calendar     sessions.session_closes on the benchmark (today never counts)
    OHLCV        data_client.get_prices — split-adjusted daily bars (Tiingo)
    dividends    data_client.dividends — cash per share on the ex-date, restated
                 per split-adjusted share so it matches the price basis
    splits       data_client.split_events — to undo, in each view, adjustment
                 for splits that happen after the view's session
    tradable     tradability.tradable_closes — positive close and volume, and
                 not a vendor carry-forward after a known delisting
    membership   a point-in-time universe schedule (members_on), when given

The panel itself is not readable: strategies get an `AsOfView` from
`MarketPanel.as_of(day)`. A view holds only rows dated on or before its as-of
session — the rows after it are sliced off when the view is made, so there is
nothing later inside it to leak — and every accessor raises `FutureDataError`
for a date after that session instead of clipping silently. A view at day D
includes D's close; acting on it is a matter for execution (next session).

Untradable bars (halts, zero volume, post-delisting placeholder prints) are
NaN in the price fields by default, with the vendor's prints still available
behind `tradable_only=False` for audit.

Split basis. A vendor's split-adjusted history divides every earlier price by
every later split, so a 2016 price level already "knows" about a 2020 split.
Returns are unaffected, but levels (and volumes, and dividends per share) are
not what anyone saw at the time. Each view therefore restates its rows to the
split basis in force at its session: prices and dividends are multiplied, and
volumes divided, by the product of split factors dated after the session. A
view's output is identical whether or not the panel's source knew the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite

import numpy as np
import pandas as pd

from hedge_fund.data.protocol import DataClient
from hedge_fund.data.sessions import completed_through, session_closes
from hedge_fund.data.tradability import tradable_closes

PRICE_FIELDS = ("open", "high", "low", "close", "volume")
FIELDS = (*PRICE_FIELDS, "dividend")


class FutureDataError(LookupError):
    """A read asked for market data dated after the view's as-of session."""


def _day(value: str) -> str:
    """Validate and normalize a YYYY-MM-DD date string."""
    return date.fromisoformat(str(value)[:10]).isoformat()


@dataclass(frozen=True)
class PanelInfo:
    """What a panel covers — facts, no market data."""

    start: str
    end: str
    benchmark: str
    tickers: tuple[str, ...]
    sessions: int
    has_membership: bool


class MarketPanel:
    """Daily bars for *tickers* over the benchmark's sessions in [start, end].

    Build with `MarketPanel.build(...)`; read with `panel.as_of(day)`.
    """

    def __init__(self, frames: dict[str, pd.DataFrame], tradable: pd.DataFrame,
                 members: pd.DataFrame | None, info: PanelInfo,
                 splits: dict[str, list[tuple[str, float]]] | None = None) -> None:
        self._frames = frames
        self._splits = splits or {}
        self._tradable = tradable
        self._members = members
        self._sessions: list[str] = list(tradable.index)
        self.info = info

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, data_client: DataClient, tickers: list[str] | None, start: str, end: str, *,
              benchmark: str = "SPY", schedule=None) -> MarketPanel:
        """Assemble the panel from *data_client*.

        *tickers* defaults to every name the *schedule* ever holds. With a
        *schedule* (anything with `members_on(day)`, e.g. a UniverseSchedule),
        membership on each session is the snapshot in force that day.
        Costs one history read per ticker plus the benchmark; with a warm
        Tiingo store that is zero network requests.
        """
        start, end = _day(start), _day(end)
        if start > end:
            raise ValueError(f"panel start {start} is after end {end}")
        if tickers is None:
            if schedule is None:
                raise ValueError("give tickers, a schedule, or both")
            tickers = schedule.all_tickers()
        tickers = list(dict.fromkeys(t.strip().upper() for t in tickers))
        if not tickers:
            raise ValueError("a panel needs at least one ticker")

        sessions = list(session_closes(data_client, benchmark, start, end))
        if not sessions:
            raise ValueError(f"{benchmark} has no completed sessions in [{start}, {end}]")
        index = pd.Index(sessions, name="date")
        frames = {f: pd.DataFrame(np.nan, index=index, columns=tickers) for f in FIELDS}
        tradable = pd.DataFrame(False, index=index, columns=tickers)
        first, last = sessions[0], sessions[-1]
        dividends = getattr(data_client, "dividends", None)
        raw_closes = getattr(data_client, "raw_closes", None)
        split_events = getattr(data_client, "split_events", None)
        splits: dict[str, list[tuple[str, float]]] = {}

        for t in tickers:
            bars = {b.time[:10]: b for b in data_client.get_prices(t, first, last)
                    if first <= b.time[:10] <= last}
            days = [d for d in sessions if d in bars]
            for f in PRICE_FIELDS:
                frames[f].loc[days, t] = [float(getattr(bars[d], f)) for d in days]
            if callable(split_events):
                # Through the end of the stored history, not the panel: the
                # adjusted prices already divide by those later splits too.
                splits[t] = sorted((d[:10], float(f)) for d, f in split_events(t, first, "9999-12-31").items()
                                   if f and isfinite(f) and f != 1.0)
            ok = [d for d in tradable_closes(data_client, t, first, last) if d in tradable.index]
            tradable.loc[ok, t] = True
            if callable(dividends):
                raw = raw_closes(t, first, last) if callable(raw_closes) else {}
                for d, cash in dividends(t, first, last).items():
                    d = d[:10]
                    if d not in bars or not cash or not isfinite(cash):
                        continue
                    # Restate per split-adjusted share: adjusted = raw / factor.
                    factor = raw[d] / bars[d].close if raw.get(d) and bars[d].close else 1.0
                    frames["dividend"].loc[d, t] = float(cash) / factor
        frames["dividend"] = frames["dividend"].fillna(0.0)

        members = None
        if schedule is not None:
            members = pd.DataFrame(False, index=index, columns=tickers)
            for d in sessions:
                held = [t for t in schedule.members_on(d) if t in members.columns]
                members.loc[d, held] = True

        for f in FIELDS:
            frames[f] = frames[f].astype(float)
        info = PanelInfo(start=first, end=last, benchmark=benchmark, tickers=tuple(tickers),
                         sessions=len(sessions), has_membership=members is not None)
        return cls(frames, tradable, members, info, splits)

    # ------------------------------------------------------------------
    # The only way in
    # ------------------------------------------------------------------

    def sessions_through(self, day: str) -> list[str]:
        """Trading sessions on or before *day* — the calendar, not the data."""
        day = _day(day)
        return [s for s in self._sessions if s <= day]

    def as_of(self, day: str) -> AsOfView:
        """A read-only view of everything known at the close of *day*.

        The view's session is the last panel session on or before *day*.
        Raises if *day* is before the first session, or is not yet a
        completed day in New York.
        """
        day = _day(day)
        if day > completed_through():
            raise FutureDataError(f"{day} is not a completed trading day yet")
        n = int(np.searchsorted(np.array(self._sessions), day, side="right"))
        if n == 0:
            raise ValueError(f"{day} is before the panel's first session {self._sessions[0]}")
        session = self._sessions[n - 1]
        basis = pd.Series({t: float(np.prod([f for d, f in self._splits.get(t, []) if d > session]))
                           for t in self._tradable.columns}, dtype=float)
        return AsOfView(
            as_of=day, session=session, basis=basis,
            frames={f: df.iloc[:n] for f, df in self._frames.items()},
            tradable=self._tradable.iloc[:n],
            members=None if self._members is None else self._members.iloc[:n],
        )


class AsOfView:
    """Market data as known at one as-of date. Holds no row after it.

    Every accessor returns a fresh copy, so a strategy cannot alter what the
    next view (or another strategy) sees.
    """

    __slots__ = ("_as_of", "_session", "_frames", "_tradable", "_members", "_basis")

    def __init__(self, *, as_of: str, session: str, frames: dict[str, pd.DataFrame],
                 tradable: pd.DataFrame, members: pd.DataFrame | None,
                 basis: pd.Series | None = None) -> None:
        for name, df in (*frames.items(), ("tradable", tradable),
                         *((("members", members),) if members is not None else ())):
            if len(df.index) and df.index[-1] > session:
                raise FutureDataError(f"refusing a view whose {name} extends past {session}")
        self._as_of = as_of
        self._session = session
        self._frames = frames
        self._tradable = tradable
        self._members = members
        # Per-ticker product of splits after the session (1.0 = none); see
        # "Split basis" in the module docstring.
        self._basis = basis if basis is not None else pd.Series(1.0, index=tradable.columns)

    @property
    def as_of(self) -> str:
        return self._as_of

    @property
    def session(self) -> str:
        """The last trading session included — on or before `as_of`."""
        return self._session

    @property
    def tickers(self) -> list[str]:
        return list(self._tradable.columns)

    @property
    def sessions(self) -> list[str]:
        return list(self._tradable.index)

    def _check(self, day: str | None, what: str) -> str | None:
        if day is None:
            return None
        day = _day(day)
        if day > self._as_of:
            raise FutureDataError(f"{what} {day} is after the as-of date {self._as_of}")
        return day

    def _rows(self, df: pd.DataFrame, start: str | None, end: str | None, lookback: int | None,
              tickers: list[str] | None) -> pd.DataFrame:
        start, end = self._check(start, "start"), self._check(end, "end")
        if lookback is not None and lookback < 1:
            raise ValueError("lookback must be at least 1 session")
        out = df
        if end is not None:
            out = out.loc[:end]
        if start is not None:
            out = out.loc[start:]
        if lookback is not None:
            out = out.iloc[-lookback:]
        if tickers is not None:
            missing = [t for t in tickers if t not in out.columns]
            if missing:
                raise KeyError(f"not in panel: {', '.join(missing)}")
            out = out[tickers]
        return out.copy()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def bars(self, field: str, *, start: str | None = None, end: str | None = None,
             lookback: int | None = None, tickers: list[str] | None = None,
             tradable_only: bool = True) -> pd.DataFrame:
        """A sessions x tickers frame of *field* up to `end` (default: as-of).

        *field* is one of open, high, low, close, volume, dividend. Price
        fields are NaN where the bar is not tradable unless *tradable_only*
        is False. *lookback* keeps the last N sessions.
        """
        if field not in FIELDS:
            raise ValueError(f"unknown field {field!r}; expected one of {', '.join(FIELDS)}")
        out = self._rows(self._frames[field], start, end, lookback, tickers)
        scale = self._basis.reindex(out.columns).fillna(1.0)
        out = out / scale if field == "volume" else out * scale
        if tradable_only and field in PRICE_FIELDS:
            mask = self._rows(self._tradable, start, end, lookback, tickers)
            out = out.where(mask)
        return out

    def history(self, ticker: str, *, start: str | None = None, end: str | None = None,
                lookback: int | None = None, tradable_only: bool = True) -> pd.DataFrame:
        """One ticker's OHLCV + dividend rows (sessions x fields)."""
        cols = {f: self.bars(f, start=start, end=end, lookback=lookback, tickers=[ticker],
                             tradable_only=tradable_only)[ticker] for f in FIELDS}
        return pd.DataFrame(cols)

    def close(self, ticker: str, day: str | None = None, *, tradable_only: bool = True) -> float:
        """The close on *day* (default: the view's session); NaN if none."""
        day = self._check(day, "day") or self._session
        frame = self.bars("close", end=day, lookback=1, tickers=[ticker], tradable_only=tradable_only)
        if frame.empty or frame.index[-1] != day:
            return float("nan")
        return float(frame.iloc[-1, 0])

    def tradable(self, *, start: str | None = None, end: str | None = None,
                 lookback: int | None = None, tickers: list[str] | None = None) -> pd.DataFrame:
        """Boolean mask: True where the bar is a real, tradable close."""
        return self._rows(self._tradable, start, end, lookback, tickers)

    def members(self, day: str | None = None) -> list[str]:
        """Universe members in force on *day* (default: the view's session)."""
        if self._members is None:
            raise LookupError("this panel was built without a universe schedule")
        day = self._check(day, "day") or self._session
        rows = self._members.loc[:day]
        if rows.empty:
            return []
        last = rows.iloc[-1]
        return [t for t in rows.columns if bool(last[t])]

    def membership(self, *, start: str | None = None, end: str | None = None,
                   lookback: int | None = None) -> pd.DataFrame:
        """Boolean sessions x tickers: point-in-time universe membership."""
        if self._members is None:
            raise LookupError("this panel was built without a universe schedule")
        return self._rows(self._members, start, end, lookback, None)
