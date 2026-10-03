"""Who was in the S&P 500 on a given date — free, from Wikipedia.

"List of S&P 500 companies" holds today's members; "Historical components
of the S&P 500" holds a dated log of additions and removals. Replaying the
log backwards from today's list gives the index on any past date, including
companies later acquired or delisted: the difference between an honest
backtest universe and a list of today's winners. Checked: the reconstruction
holds 504-507 lines at every year-start back to 2016 (the index has ~503
because some companies list two share classes).
"""

from __future__ import annotations

import html
import re
from datetime import date, datetime

import requests

CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
USER_AGENT = "ai-hedge-fund (https://github.com/virattt/ai-hedge-fund) research"

Change = tuple[date, str | None, str | None]   # (effective date, added ticker, removed ticker)


def _table(page: str, table_id: str) -> str:
    match = re.search(rf'<table[^>]*id="{table_id}".*?</table>', page, re.S)
    if not match:
        raise ValueError(f"table {table_id!r} not found; the Wikipedia page layout may have changed")
    return match.group(0)


def _rows(table: str) -> list[list[str]]:
    rows = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
        cells = re.findall(r"<t([dh])[^>]*>(.*?)</t[dh]>", row, re.S)
        rows.append([(kind, html.unescape(re.sub(r"<[^>]+>", "", text)).strip()) for kind, text in cells])
    return rows


def parse_current(page: str) -> list[str]:
    """Tickers in the constituents table, as listed (e.g. BRK.B)."""
    return [cells[0][1] for cells in _rows(_table(page, "constituents"))
            if cells and cells[0][0] == "d" and cells[0][1]]


def parse_changes(page: str) -> list[Change]:
    """(date, added, removed) rows, newest first; rows without a parseable date are skipped."""
    out: list[Change] = []
    for cells in _rows(_table(page, "changes")):
        values = [text for kind, text in cells if kind == "d"]
        if len(values) < 5:
            continue
        try:
            day = datetime.strptime(values[0], "%B %d, %Y").date()
        except ValueError:
            continue
        out.append((day, values[1] or None, values[3] or None))
    return sorted(out, key=lambda c: c[0], reverse=True)


def members_as_of(as_of: str, current: list[str], changes: list[Change]) -> set[str]:
    """Index members on `as_of` (changes take effect on their date)."""
    day = date.fromisoformat(as_of)
    members = set(current)
    for effective, added, removed in sorted(changes, key=lambda c: c[0], reverse=True):
        if effective <= day:
            break
        if added:
            members.discard(added)
        if removed:
            members.add(removed)
    return members


def fetch(session: requests.Session | None = None, timeout: float = 30.0) -> tuple[list[str], list[Change]]:
    session = session or requests.Session()
    pages = []
    for url in (CURRENT_URL, CHANGES_URL):
        resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
        pages.append(resp.text)
    return parse_current(pages[0]), parse_changes(pages[1])
