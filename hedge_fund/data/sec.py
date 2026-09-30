"""SEC EDGAR — free, official company filings and XBRL financial facts.

No key, but SEC requires every client to identify itself: SEC_USER_AGENT must
hold a name and contact email ("Jane Doe jane@example.com"). Requests are
throttled below SEC's 10-per-second limit and retried on 429/5xx.

sync_company() stores what the fund needs for one company: its SIC industry,
its 8-K/10-Q/10-K filing index, and the XBRL facts listed in
fundamentals.CONCEPTS from 10-Q/10-K filings (with the date each was filed —
that date is what makes backtests point-in-time).
"""

from __future__ import annotations

import os
import time
from typing import Callable

import requests

from hedge_fund.data.errors import DataSourceError
from hedge_fund.data.fundamentals import DEI, US_GAAP
from hedge_fund.data.store import Fact, MarketStore

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:0>10}.json"

_MIN_INTERVAL = 0.125                 # 8 requests/second, under SEC's 10
_RETRY_DELAYS = (1.0, 2.0, 4.0)
_FILING_FORMS = {"8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A"}

# Standard SIC divisions — the broad sector the agents see.
_SIC_DIVISIONS = [
    (100, 999, "Agriculture, Forestry & Fishing"), (1000, 1499, "Mining"), (1500, 1799, "Construction"),
    (2000, 3999, "Manufacturing"), (4000, 4999, "Transportation, Communications & Utilities"),
    (5000, 5199, "Wholesale Trade"), (5200, 5999, "Retail Trade"),
    (6000, 6799, "Finance, Insurance & Real Estate"), (7000, 8999, "Services"),
    (9100, 9729, "Public Administration"),
]


def sec_ticker(ticker: str) -> str:
    """SEC spells share classes with a dash: BRK.B → BRK-B."""
    return ticker.upper().replace(".", "-")


def sic_sector(sic: int | None) -> str | None:
    if sic is None:
        return None
    return next((name for low, high, name in _SIC_DIVISIONS if low <= sic <= high), None)


class SecSource:
    def __init__(self, user_agent: str | None = None, *, timeout: float = 30.0,
                 session: requests.Session | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        user_agent = user_agent or os.environ.get("SEC_USER_AGENT", "")
        if not user_agent.strip():
            raise ValueError('set SEC_USER_AGENT="Your Name your@email.com" in .env (SEC requires a contact on every request)')
        self._timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._clock, self._sleep = clock, sleep
        self._last_request = None
        self._ticker_map: dict[str, str] | None = None

    def ticker_map(self) -> dict[str, str]:
        """SEC ticker (BRK-B style) → CIK, fetched once per SecSource."""
        if self._ticker_map is None:
            rows = self._get(TICKERS_URL).values()
            self._ticker_map = {row["ticker"]: str(row["cik_str"]) for row in rows}
        return self._ticker_map

    def sync_company(self, ticker: str, store: MarketStore, *, now: str) -> str:
        """Fetch and store one company's industry, filings and facts; returns its CIK."""
        known = store.company(ticker)
        if known:
            cik = known["cik"]
        else:
            cik = self.ticker_map().get(sec_ticker(ticker))
            if cik is None:
                raise DataSourceError(f"{ticker}: no SEC registrant with ticker {sec_ticker(ticker)}")

        subs = self._get(SUBMISSIONS_URL.format(cik=cik))
        sic = int(subs["sic"]) if str(subs.get("sic") or "").isdigit() else None
        store.upsert_company(ticker, cik, subs.get("name"), sic, subs.get("sicDescription"))
        recent = subs.get("filings", {}).get("recent", {})
        filings = [
            {"accn": accn, "form": form, "filed": filed, "report_date": report or None, "items": items or ""}
            for accn, form, filed, report, items in zip(
                recent.get("accessionNumber", []), recent.get("form", []), recent.get("filingDate", []),
                recent.get("reportDate", []), recent.get("items", []))
            if form in _FILING_FORMS
        ]
        store.replace_filings(cik, filings)
        store.mark_synced(cik, submissions_at=now)

        body = self._get(FACTS_URL.format(cik=cik))
        store.replace_facts(cik, _facts(body))
        store.mark_synced(cik, facts_at=now)
        return cik

    def _get(self, url: str) -> dict:
        for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
            if self._last_request is not None:
                wait = _MIN_INTERVAL - (self._clock() - self._last_request)
                if wait > 0:
                    self._sleep(wait)
            self._last_request = self._clock()
            try:
                resp = self._session.get(url, timeout=self._timeout)
            except requests.RequestException as exc:
                if delay is None:
                    raise DataSourceError(f"SEC {url}: {exc}") from exc
                self._sleep(delay)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if delay is None:
                    raise DataSourceError(f"SEC {url}: HTTP {resp.status_code}", status_code=resp.status_code)
                self._sleep(delay)
                continue
            if resp.status_code >= 400:
                raise DataSourceError(f"SEC {url}: HTTP {resp.status_code}: {resp.text[:200]}", status_code=resp.status_code)
            return resp.json()
        raise AssertionError("unreachable")


def _facts(body: dict) -> list[Fact]:
    """The facts the fund uses, from 10-Q/10-K filings only (8-K exhibits are unaudited duplicates)."""
    wanted = {"us-gaap": US_GAAP, "dei": DEI}
    out: list[Fact] = []
    for taxonomy, concepts in wanted.items():
        available = body.get("facts", {}).get(taxonomy, {})
        for concept in concepts:
            for unit, rows in available.get(concept, {}).get("units", {}).items():
                for row in rows:
                    form = row.get("form", "")
                    if not form.startswith("10-") or not row.get("filed") or row.get("val") is None:
                        continue
                    out.append(Fact(concept=concept, unit=unit, start=row.get("start"), end=row["end"],
                                    value=float(row["val"]), fy=row.get("fy"), fp=row.get("fp"),
                                    form=form, filed=row["filed"], accn=row["accn"]))
    return out
