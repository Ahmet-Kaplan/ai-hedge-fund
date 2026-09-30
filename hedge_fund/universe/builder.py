"""Build a survivorship-aware, point-in-time US stock universe.

For a date D the builder answers "which companies were the largest US common
stocks on D, knowing only what was public on D" — from SEC filings and Tiingo
prices, never from today's ticker list or market caps.

1. Discovery (candidates only). SEC XBRL frames list every filer's public
   float for a calendar quarter; the top filers of each quarterly frame
   that ended on or before D (over `discovery_years`) are candidates. Frame
   values are "latest filed", so they only nominate — every decision below
   uses point-in-time data. Companies that later delisted are in the frames
   of the years they filed, so they are discovered like any other.
2. Point-in-time screen, SEC facts filed on or before D only:
   - a periodic report (10-K/10-Q) filed within `max_filing_age_days`
     (stopped filing = delisted, acquired, or reorganized);
   - at least `min_filings` periodic reports (what the Buffett snapshot needs);
   - an operating company that is not a commodity trust, investment company,
     asset-backed issuer or SPAC (SIC code, SEC entity type);
   - a plausible public float: implied price per cover share below
     `max_implied_price` and no `max_float_jump`x jump from the prior year
     (filers do tag values 1,000x too large).
   A registrant lineage (ticker_history.csv: Google Inc. -> Alphabet) is one
   candidate: the predecessor up to the reorganization, the successor after,
   with the successor's facts including the predecessor's history.
3. Pool: the `pool_size` largest by that public float.
4. Pricing: a symbol per company — curated ticker history, then SEC's
   current map (renamed companies keep their history under today's symbol at
   Tiingo), then the trading symbol printed on the company's own latest
   filing (delisted companies). A symbol is accepted only if Tiingo has a
   tradable close within `price_lookback_days` of D, at least
   `min_price_history_days` of history, and its market cap agrees with the
   filed float within `float_to_market_cap` (rejects recycled tickers).
   Market cap = latest filed shares (in the symbol's class, adjusted for
   splits since the filing) x the raw close on or before D.
5. Rank by market cap, one entry per company and per symbol; keep `top_n`.

Every exclusion is recorded with its reason. Snapshots are cached per
config digest and date, so a rebuild is identical and free.
"""

from __future__ import annotations

import json
from calendar import monthrange
from datetime import date, timedelta
from math import prod
from pathlib import Path

from hedge_fund.data.edgar.client import EdgarClient
from hedge_fund.data.edgar.concepts import COVER_SHARES_TAG
from hedge_fund.data.edgar.facts import FactStore, Filing
from hedge_fund.data.edgar.identity import SHARE_CLASSES, Identity, load_history, normalize_ticker
from hedge_fund.data.factory import CompositeDataClient
from hedge_fund.data.tiingo import TiingoClient, tiingo_symbol
from hedge_fund.data.tradability import tradable_closes
from hedge_fund.paths import CACHE_DIR
from hedge_fund.universe.models import (
    UniverseConfig,
    UniverseExclusion,
    UniverseMember,
    UniverseSchedule,
    UniverseSnapshot,
)

DEFAULT_CACHE_DIR = CACHE_DIR / "universe"
_ABSURD_FLOAT = 1e13   # > $10T: a tagging error, not a company


class PriceBudgetExceeded(RuntimeError):
    """More new Tiingo downloads were needed than allowed. Everything fetched
    so far is stored; rerun later to continue."""


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _days(a: str, b: str) -> int:
    return (_d(b) - _d(a)).days


def _quarter_end(year: int, q: int) -> str:
    return {1: f"{year}-03-31", 2: f"{year}-06-30", 3: f"{year}-09-30", 4: f"{year}-12-31"}[q]


class _Candidate:
    def __init__(self, cik: int, name: str | None, store: FactStore, latest: Filing,
                 public_float: float, float_filed: str) -> None:
        self.cik, self.name, self.store, self.latest = cik, name, store, latest
        self.public_float, self.float_filed = public_float, float_filed


class UniverseBuilder:
    def __init__(self, edgar: EdgarClient, tiingo: TiingoClient, config: UniverseConfig | None = None, *,
                 cache_dir: Path | str = DEFAULT_CACHE_DIR, max_price_downloads: int | None = None,
                 refresh: bool = False) -> None:
        self.edgar = edgar
        self.tiingo = tiingo
        self.config = config or UniverseConfig()
        self.data = CompositeDataClient(tiingo, edgar)
        self._dir = Path(cache_dir) / self.config.digest()
        self._budget = max_price_downloads
        self._refresh = refresh
        self.price_downloads = 0
        history = load_history()
        self._successor: dict[int, tuple[int, str]] = {}       # predecessor cik -> (successor, until)
        self._predecessor: dict[int, tuple[int, str]] = {}     # successor cik -> (predecessor, until)
        self._history_tickers: dict[int, list] = {}
        for rows in history.values():
            for r in rows:
                if r.cik is None:
                    continue
                self._history_tickers.setdefault(r.cik, []).append(r)
                if r.predecessor_cik is not None:
                    self._successor[r.predecessor_cik] = (r.cik, r.predecessor_until)
                    self._predecessor[r.cik] = (r.predecessor_cik, r.predecessor_until)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self, dates: list[str]) -> UniverseSchedule:
        return UniverseSchedule(config=self.config, snapshots=[self.snapshot(d) for d in sorted(set(dates))])

    def snapshot(self, as_of: str) -> UniverseSnapshot:
        path = self._dir / f"{as_of}.json"
        if path.exists() and not self._refresh:
            return UniverseSnapshot.model_validate_json(path.read_text())
        snap = self._build(as_of)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(snap.model_dump_json(indent=1))
        discovered = {m.ticker: m.cik for m in snap.members if m.symbol_source == "cover_page"}
        if discovered:
            self.edgar.register_tickers(discovered)
        return snap

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def discover(self, as_of: str) -> dict[int, str | None]:
        """CIK -> name for the top public-float filers of each quarterly frame
        that ended on or before *as_of*."""
        year = _d(as_of).year
        found: dict[int, str | None] = {}
        for y in range(year - self.config.discovery_years, year + 1):
            for q in (1, 2, 3, 4):
                if _quarter_end(y, q) >= as_of:
                    continue
                rows = [r for r in self.edgar.frame("EntityPublicFloat", "USD", f"CY{y}Q{q}I")
                        if r.get("val") and 0 < r["val"] < _ABSURD_FLOAT and r.get("cik")]
                rows.sort(key=lambda r: (-r["val"], r["cik"]))
                for r in rows[:self.config.per_frame]:
                    found.setdefault(int(r["cik"]), r.get("entityName"))
        return found

    def _build(self, as_of: str) -> UniverseSnapshot:
        cfg = self.config
        discovered = self.discover(as_of)
        excluded: list[UniverseExclusion] = []
        canonical: dict[int, str | None] = {}
        for cik, name in sorted(discovered.items()):
            # One economic company per lineage: the predecessor registrant up
            # to its reorganization date, the successor after it.
            succ, pred = self._successor.get(cik), self._predecessor.get(cik)
            if succ is not None and as_of > succ[1]:
                canonical.setdefault(succ[0], name)
                excluded.append(UniverseExclusion(cik=cik, name=name, reason="predecessor",
                                                  detail=f"represented by CIK {succ[0]} after {succ[1]}"))
            elif pred is not None and as_of <= pred[1]:
                canonical.setdefault(pred[0], name)
                excluded.append(UniverseExclusion(cik=cik, name=name, reason="successor_not_yet",
                                                  detail=f"represented by CIK {pred[0]} until {pred[1]}"))
            else:
                canonical.setdefault(cik, name)

        screened: list[_Candidate] = []
        for cik, name in sorted(canonical.items()):
            reason, detail, cand = self._screen(cik, name, as_of)
            if cand is None:
                excluded.append(UniverseExclusion(cik=cik, name=name, reason=reason, detail=detail))
            else:
                screened.append(cand)
        screened.sort(key=lambda c: (-c.public_float, c.cik))

        priced: list[UniverseMember] = []
        seen_symbols: dict[str, int] = {}
        for cand in screened:
            if len(priced) >= cfg.pool_size:
                excluded.append(UniverseExclusion(cik=cand.cik, name=cand.name, reason="outside_float_pool"))
                continue
            member, reason, detail = self._price(cand, as_of)
            if member is None:
                excluded.append(UniverseExclusion(cik=cand.cik, name=cand.name, reason=reason, detail=detail))
                continue
            key = tiingo_symbol(member.ticker)
            if key in seen_symbols:
                excluded.append(UniverseExclusion(cik=cand.cik, name=cand.name, reason="duplicate_symbol",
                                                  detail=f"{member.ticker} already used by CIK {seen_symbols[key]}"))
                continue
            seen_symbols[key] = cand.cik
            priced.append(member)

        priced.sort(key=lambda m: (-m.market_cap, m.cik))
        members = [m.model_copy(update={"rank": i + 1}) for i, m in enumerate(priced[:cfg.top_n])]
        for m in priced[cfg.top_n:]:
            excluded.append(UniverseExclusion(cik=m.cik, name=m.name, reason="below_top_n",
                                              detail=f"{m.ticker} market cap {m.market_cap:,.0f}"))
        return UniverseSnapshot(as_of=as_of, config_digest=cfg.digest(), candidates=len(canonical),
                                members=members, excluded=sorted(excluded, key=lambda e: (e.reason, e.cik)))

    def _store(self, cik: int) -> FactStore | None:
        pred = self._predecessor.get(cik)
        if pred is None:
            return self.edgar.store_for_cik(cik)
        return self.edgar._store(Identity(ticker="", cik=cik, predecessor_cik=pred[0], predecessor_until=pred[1]))

    def _screen(self, cik: int, name: str | None, as_of: str) -> tuple[str, str, _Candidate | None]:
        cfg = self.config
        store = self._store(cik)
        if store is None:
            return "no_sec_facts", "", None
        filings = [f for f in store.filings() if f.filed <= as_of]
        if not filings:
            return "not_yet_filing", "", None
        latest = filings[-1]
        if _days(latest.filed, as_of) > cfg.max_filing_age_days:
            return "stopped_filing", f"last periodic report filed {latest.filed}", None
        if len(filings) < cfg.min_filings:
            return "insufficient_filings", f"{len(filings)} periodic reports filed", None
        profile = self.edgar.company_profile(cik) or {}
        try:
            sic = int(profile.get("sic") or 0)
        except ValueError:
            sic = 0
        if sic in cfg.excluded_sic:
            return "excluded_security_type", f"SIC {sic}: {cfg.excluded_sic[sic]}", None
        entity_type = (profile.get("entityType") or "operating").lower()
        if entity_type != "operating":
            return "excluded_security_type", f"entity type {entity_type}", None
        floats = sorted((f for f in store.public_floats(as_of) if f.val > 0), key=lambda f: (f.end, f.filed))
        if not floats:
            return "no_public_float", "", None
        current = floats[-1]
        if current.val >= _ABSURD_FLOAT:
            return "implausible_float", f"{current.val:,.0f}", None
        prior = [f for f in floats if _days(f.end, current.end) >= 300]
        if prior and not (1 / cfg.max_float_jump <= current.val / prior[-1].val <= cfg.max_float_jump):
            return "implausible_float", f"{current.val:,.0f} vs {prior[-1].val:,.0f} a year earlier", None
        if cik not in SHARE_CLASSES:
            covers = sorted((f for f in store.facts if f.tag == COVER_SHARES_TAG and f.filed <= as_of and f.val > 0),
                            key=lambda f: (f.filed, f.end))
            if covers and current.val / covers[-1].val > cfg.max_implied_price:
                return "implausible_float", f"implies ${current.val / covers[-1].val:,.0f} per share", None
        name = name or profile.get("name")
        return "", "", _Candidate(cik, name, store, latest, current.val, current.filed)

    def _symbols(self, cand: _Candidate, as_of: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        rows = [r for r in self._history_tickers.get(cand.cik, [])
                if (r.start_date or "") <= as_of and (r.end_date is None or as_of <= r.end_date)]
        succ = self._successor.get(cand.cik)
        if succ is not None:   # same company: the vendor keeps its history under the successor's symbol
            rows += [r for r in self._history_tickers.get(succ[0], []) if r.predecessor_cik == cand.cik]
        classing = SHARE_CLASSES.get(cand.cik)
        if classing is not None:   # most liquid class first (fewest units per share)
            units = classing.units(as_of)
            rows.sort(key=lambda r: units.get(classing.ticker_class.get(r.ticker, ""), 1e9))
        out += [(r.ticker, "history") for r in rows]
        current = [normalize_ticker(t) for t, c in self.edgar.current_tickers().items() if int(c) == cand.cik]
        out += [(t, "current") for t in current if _looks_common(t)]
        seen, unique = set(), []
        for t, src in out:
            if t not in seen:
                seen.add(t)
                unique.append((t, src))
        return unique

    def _cover_symbols(self, cand: _Candidate) -> list[tuple[str, str]]:
        symbols = self.edgar.trading_symbols(cand.latest.cik or cand.cik, cand.latest.accn)
        return [(normalize_ticker(s), "cover_page") for s in symbols if _looks_common(normalize_ticker(s))][:3]

    def _price(self, cand: _Candidate, as_of: str) -> tuple[UniverseMember | None, str, str]:
        reason, detail = "no_symbol", "no symbol in ticker history, current map or cover page"
        tried: set[str] = set()
        for source_list in (lambda: self._symbols(cand, as_of), lambda: self._cover_symbols(cand)):
            for ticker, source in source_list():
                if ticker in tried:
                    continue
                tried.add(ticker)
                member, reason, detail = self._try_symbol(cand, ticker, source, as_of)
                if member is not None:
                    return member, "", ""
        return None, reason, detail

    def _try_symbol(self, cand: _Candidate, ticker: str, source: str, as_of: str):
        cfg = self.config
        if not self.tiingo.is_stored(ticker):
            if self._budget is not None and self.price_downloads >= self._budget:
                raise PriceBudgetExceeded(
                    f"needs a new Tiingo download ({ticker}) beyond max_price_downloads={self._budget}; "
                    "progress is stored — rerun later to continue")
            self.price_downloads += 1
        start = (_d(as_of) - timedelta(days=cfg.price_lookback_days)).isoformat()
        closes = tradable_closes(self.data, ticker, start, as_of)
        if not closes:
            return None, "no_price", f"{ticker}: no tradable close in {start}..{as_of}"
        first = self.tiingo.history_range(ticker)[0]
        if _days(first, as_of) < cfg.min_price_history_days:
            return None, "recently_listed", f"{ticker}: prices start {first}"
        price_date = max(closes)
        raw = self.tiingo.raw_closes(ticker, price_date, price_date).get(price_date)
        if not raw:
            return None, "no_price", f"{ticker}: no raw close on {price_date}"
        view = cand.store.view(cand.latest.filed)
        shares, shares_source = self.edgar._shares(ticker, cand.store, view, cand.latest)
        if not shares:
            return None, "no_shares", f"no share count on the {cand.latest.filed} filing"
        after = (_d(cand.latest.filed) + timedelta(days=1)).isoformat()
        factor = prod(f for d, f in self.tiingo.split_events(ticker, after, price_date).items() if f > 0)
        shares *= factor
        market_cap = shares * raw
        lo, hi = cfg.float_to_market_cap
        ratio = cand.public_float / market_cap
        if not lo <= ratio <= hi:
            return None, "float_market_cap_mismatch", f"{ticker}: float/market cap {ratio:.3g}"
        return UniverseMember(
            rank=0, ticker=ticker, cik=cand.cik, name=cand.name, market_cap=market_cap, price=raw,
            price_date=price_date, shares=shares, shares_source=shares_source or "", public_float=cand.public_float,
            float_filed=cand.float_filed, latest_filing=cand.latest.filed, symbol_source=source,
        ), "", ""


def _looks_common(ticker: str) -> bool:
    """Common-stock-looking symbol: letters, optionally one class letter
    (BRK.B); not a preferred, warrant, unit or right (JPM.PC, XYZ.WS)."""
    base, _, suffix = ticker.partition(".")
    return base.isalpha() and 1 <= len(base) <= 5 and (suffix == "" or (len(suffix) == 1 and suffix in "ABC"))


def save_schedule(schedule: UniverseSchedule, path: Path | str) -> None:
    Path(path).write_text(schedule.model_dump_json(indent=1))


def load_schedule(path: Path | str) -> UniverseSchedule:
    return UniverseSchedule.model_validate(json.loads(Path(path).read_text()))


def reconstitution_dates(sessions: list[str], start: str, end: str, cadence: str = "annual") -> list[str]:
    """The first trading session on or after each anchor date: *start*, then
    every 12 (annual) / 3 (quarterly) months, up to *end*."""
    step = {"annual": 12, "quarterly": 3}.get(cadence)
    if step is None:
        raise ValueError(f"unknown reconstitution cadence {cadence!r}")
    days = sorted(d for d in sessions if start <= d <= end)
    s = _d(start)
    out: list[str] = []
    k = 0
    while True:
        month = s.month - 1 + k * step
        year, month = s.year + month // 12, month % 12 + 1
        anchor = date(year, month, min(s.day, monthrange(year, month)[1])).isoformat()
        if anchor > end:
            break
        nxt = next((d for d in days if d >= anchor), None)
        if nxt is not None and nxt not in out:
            out.append(nxt)
        k += 1
    return out
