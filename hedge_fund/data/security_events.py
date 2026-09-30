"""Curated corporate events that end a listing (take-privates, acquisitions,
failures), used by the backtester's delisting policy.

Each row records the listing's last trading day, what ended it, and — when
the same shares kept trading under another symbol — that successor. Rows are
static facts; the backtester applies one only on sessions *after* its last
trading day, so nothing here is visible before it happened.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

EVENTS_PATH = Path(__file__).with_name("security_events.csv")

EventType = Literal["acquisition", "take_private", "bankruptcy", "bank_failure", "delisted"]


class SecurityEvent(BaseModel):
    ticker: str
    last_trading_day: str
    event_type: EventType
    successor: str | None = None
    note: str = ""


def _normalize(ticker: str) -> str:
    return ticker.strip().upper().replace("-", ".").replace("/", ".")


@lru_cache(maxsize=None)
def load_events(path: Path = EVENTS_PATH) -> dict[str, SecurityEvent]:
    with open(path, newline="") as fh:
        return {
            _normalize(row["ticker"]): SecurityEvent(
                ticker=_normalize(row["ticker"]), last_trading_day=row["last_trading_day"].strip(),
                event_type=row["event_type"].strip(), successor=_normalize(row["successor"]) if row["successor"].strip() else None,
                note=row.get("note", "").strip(),
            )
            for row in csv.DictReader(fh)
        }


def security_event(ticker: str) -> SecurityEvent | None:
    """The listing-ending event recorded for *ticker*, if any."""
    return load_events().get(_normalize(ticker))
