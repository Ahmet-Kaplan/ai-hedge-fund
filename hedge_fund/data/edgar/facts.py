"""Point-in-time XBRL fact store and TTM arithmetic.

Every SEC fact carries the date its filing was accepted (`filed`). A
`KnowledgeView` is the store cut at a date: facts filed later do not exist
in it, so nothing computed from a view can see the future. Restatements are
separate facts with later filing dates, so a view shows the figure as it
was known on its cutoff, not as later revised.

TTM for a flow concept (revenue, cash flow, ...) at a period end E:

1. a fiscal-year value ending on E (a 10-K), else
2. year-to-date(E) + prior fiscal year - prior-year year-to-date. 10-Q
   cash-flow statements are *only* year-to-date, so this is the normal path
   for them; income statements take it too.
3. else the sum of four discrete quarters, where a quarter is a direct
   three-month value or the difference of two year-to-date values (Q4 is
   the fiscal year minus nine months).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from hedge_fund.data.edgar.concepts import CONCEPTS, COVER_SHARES_TAG, KNOWLEDGE_FORMS, PERIODIC_FORMS, PUBLIC_FLOAT_TAG

QUARTER_DAYS = (80, 100)
ANNUAL_DAYS = (350, 380)
_MATCH_TOLERANCE = 10  # days: 52/53-week fiscal calendars shift period ends


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _days(start: str, end: str) -> int:
    return (_d(end) - _d(start)).days + 1


@dataclass(frozen=True)
class Fact:
    tag: str
    unit: str
    start: str | None
    end: str
    val: float
    accn: str
    form: str
    filed: str
    fp: str | None = None
    fy: int | None = None
    cik: int | None = None


@dataclass(frozen=True)
class Filing:
    """One periodic report — the unit a FinancialMetrics row describes."""

    accn: str
    cik: int | None
    form: str
    filed: str
    report_period: str
    fiscal_period: str | None


class FactStore:
    """All facts of one company lineage, queryable as of any date."""

    def __init__(self, facts: list[Fact]) -> None:
        self.facts = sorted(facts, key=lambda f: (f.filed, f.accn))

    @classmethod
    def from_companyfacts(cls, doc: dict, filed_until: str | None = None) -> FactStore:
        """Parse a (trimmed) companyfacts document.

        *filed_until* drops facts filed after that date — used for a
        predecessor registrant, whose later filings describe a subsidiary.
        """
        cik = doc.get("cik")
        facts: list[Fact] = []
        for ns in ("us-gaap", "dei"):
            for tag, body in ((doc.get("facts") or {}).get(ns) or {}).items():
                for unit, rows in (body.get("units") or {}).items():
                    for r in rows:
                        if r.get("form") not in KNOWLEDGE_FORMS or r.get("val") is None:
                            continue
                        if filed_until is not None and r["filed"] > filed_until:
                            continue
                        facts.append(Fact(
                            tag=tag, unit=unit, start=r.get("start"), end=r["end"], val=float(r["val"]),
                            accn=r["accn"], form=r["form"], filed=r["filed"],
                            fp=r.get("fp"), fy=r.get("fy"), cik=int(cik) if cik is not None else None,
                        ))
        return cls(facts)

    @classmethod
    def merged(cls, stores: list[FactStore]) -> FactStore:
        return cls([f for s in stores for f in s.facts])

    def filings(self) -> list[Filing]:
        """Periodic reports, oldest first, one per report period.

        A report period is the latest end date among the filing's duration
        facts. When two registrants of one lineage both report a period, the
        earliest filing wins — it is when the numbers became public.
        """
        by_accn: dict[str, list[Fact]] = {}
        for f in self.facts:
            if f.form in PERIODIC_FORMS and f.tag not in (COVER_SHARES_TAG, PUBLIC_FLOAT_TAG):
                by_accn.setdefault(f.accn, []).append(f)
        filings: dict[str, Filing] = {}
        for accn, facts in by_accn.items():
            durations = [f for f in facts if f.start is not None]
            if not durations:
                continue
            period = max(f.end for f in durations)
            fps = [f.fp for f in durations if f.end == period and f.fp]
            filing = Filing(accn=accn, cik=facts[0].cik, form=facts[0].form, filed=min(f.filed for f in facts),
                            report_period=period, fiscal_period=fps[0] if fps else None)
            current = filings.get(period)
            if current is None or (filing.filed, filing.accn) < (current.filed, current.accn):
                filings[period] = filing
        return sorted(filings.values(), key=lambda x: (x.filed, x.accn))

    def public_floats(self, cutoff: str) -> list[Fact]:
        """Public-float facts filed on or before *cutoff*, oldest first."""
        return [f for f in self.facts if f.tag == PUBLIC_FLOAT_TAG and f.unit == "USD" and f.filed <= cutoff]

    def cover_shares(self, accn: str) -> list[Fact]:
        return [f for f in self.facts if f.accn == accn and f.tag == COVER_SHARES_TAG]

    def view(self, cutoff: str) -> KnowledgeView:
        """Everything filed on or before *cutoff*, and nothing else."""
        return KnowledgeView([f for f in self.facts if f.filed <= cutoff])


class KnowledgeView:
    """Concept values as known at one cutoff (see module docstring)."""

    def __init__(self, facts: list[Fact]) -> None:
        self._facts = facts
        self._resolved: dict[str, dict[tuple[str | None, str], float]] = {}

    # -- resolution ---------------------------------------------------------

    def series(self, concept: str) -> dict[tuple[str | None, str], float]:
        """(start, end) -> value, choosing per period the highest-priority
        tag, and within a tag the latest filing visible in this view."""
        if concept in self._resolved:
            return self._resolved[concept]
        c = CONCEPTS[concept]
        latest: dict[tuple[str, str | None, str], Fact] = {}
        for f in self._facts:
            if f.tag not in c.tags or f.unit != c.unit:
                continue
            if (c.kind == "instant") != (f.start is None):
                continue
            key = (f.tag, f.start, f.end)
            if key not in latest or (f.filed, f.accn) >= (latest[key].filed, latest[key].accn):
                latest[key] = f
        out: dict[tuple[str | None, str], float] = {}
        for tag in c.tags:
            for (t, start, end), f in latest.items():
                if t == tag and (start, end) not in out:
                    out[(start, end)] = f.val
        self._resolved[concept] = out
        return out

    def instant(self, concept: str, end: str) -> float | None:
        return self.series(concept).get((None, end))

    def _durations_ending(self, concept: str, end: str, tolerance: int = 0) -> list[tuple[str, str, float]]:
        target = _d(end)
        return [(s, e, v) for (s, e), v in self.series(concept).items()
                if s is not None and abs((_d(e) - target).days) <= tolerance]

    def _annual(self, concept: str, end: str, tolerance: int = 0) -> float | None:
        hits = [(abs((_d(e) - _d(end)).days), v) for s, e, v in self._durations_ending(concept, end, tolerance)
                if ANNUAL_DAYS[0] <= _days(s, e) <= ANNUAL_DAYS[1]]
        return min(hits)[1] if hits else None

    # -- TTM ----------------------------------------------------------------

    def ttm(self, concept: str, end: str) -> float | None:
        annual = self._annual(concept, end)
        if annual is not None:
            return annual
        via_ytd = self._ttm_from_ytd(concept, end)
        if via_ytd is not None:
            return via_ytd
        return self._ttm_from_quarters(concept, end)

    def _ttm_from_ytd(self, concept: str, end: str) -> float | None:
        ytds = [(s, v) for s, _, v in self._durations_ending(concept, end)
                if QUARTER_DAYS[0] <= _days(s, end) < ANNUAL_DAYS[0]]
        if not ytds:
            return None
        start, value = min(ytds)  # earliest start = the full year-to-date
        length = _days(start, end)
        prior_fy_end = (_d(start) - timedelta(days=1)).isoformat()
        prior_fy = self._annual(concept, prior_fy_end, tolerance=3)
        if prior_fy is None:
            return None
        prior_end = (_d(end) - timedelta(days=365)).isoformat()
        prior = [(abs(_days(s, e) - length), v) for s, e, v in self._durations_ending(concept, prior_end, _MATCH_TOLERANCE)
                 if abs(_days(s, e) - length) <= _MATCH_TOLERANCE]
        if not prior:
            return None
        return value + prior_fy - min(prior)[1]

    def quarter(self, concept: str, end: str) -> tuple[float, str] | None:
        """(value, start) of the fiscal quarter ending on *end*."""
        durations = self._durations_ending(concept, end)
        for s, _, v in durations:
            if QUARTER_DAYS[0] <= _days(s, end) <= QUARTER_DAYS[1]:
                return v, s
        # Derived: a longer YTD minus the YTD one quarter shorter, same start.
        for s, _, v in sorted(durations):
            if _days(s, end) <= QUARTER_DAYS[1]:
                continue
            shorter_end = (_d(end) - timedelta(days=91)).isoformat()
            for s2, e2, v2 in self._durations_ending(concept, shorter_end, _MATCH_TOLERANCE):
                if abs((_d(s2) - _d(s)).days) <= 3:
                    q_start = (_d(e2) + timedelta(days=1)).isoformat()
                    if QUARTER_DAYS[0] <= _days(q_start, end) <= QUARTER_DAYS[1]:
                        return v - v2, q_start
        return None

    def _ttm_from_quarters(self, concept: str, end: str) -> float | None:
        total, cursor = 0.0, end
        for _ in range(4):
            q = self.quarter(concept, cursor)
            if q is None:
                return None
            total += q[0]
            cursor = (_d(q[1]) - timedelta(days=1)).isoformat()
        return total

    def latest_duration(self, concept: str, end: str) -> float | None:
        """Value of the shortest period ending on *end* (e.g. a quarter's
        weighted-average share count)."""
        hits = sorted((_days(s, end), v) for s, _, v in self._durations_ending(concept, end))
        return hits[0][1] if hits else None

    def prior_period_end(self, end: str) -> str | None:
        """A reported period end about one year before *end*, if any."""
        target = _d(end) - timedelta(days=365)
        ends = {e for series in (self.series("net_income"), self.series("revenue"), self.series("equity"))
                for (_, e) in series if abs((_d(e) - target).days) <= _MATCH_TOLERANCE}
        return min(ends, key=lambda e: abs((_d(e) - target).days)) if ends else None
