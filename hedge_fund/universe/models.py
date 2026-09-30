"""Point-in-time universe records: config, per-date snapshots, schedule."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field

from hedge_fund.features.snapshot import MIN_PERIODS

# Bumped whenever selection logic changes, so cached snapshots are rebuilt.
UNIVERSE_VERSION = 2

# SIC codes of registrants that file 10-Ks but are not operating companies'
# common stock: commodity/currency trusts (GLD, SLV, UUP), investment
# companies and unit trusts, asset-backed issuers, blank checks (SPACs).
EXCLUDED_SIC = {6221: "commodity trust", 6726: "investment company", 6189: "asset-backed issuer", 6770: "blank check (SPAC)"}


class UniverseConfig(BaseModel):
    top_n: int = Field(15, ge=1)
    pool_multiple: float = Field(2.0, ge=1.0)       # price-check this many x top_n, by public float
    discovery_per_frame: int | None = None          # default 4 x top_n filers per SEC frame
    discovery_years: int = Field(2, ge=1)           # frames from this many calendar years before D
    min_filings: int = Field(MIN_PERIODS, ge=1)     # the Buffett snapshot needs this many filed periods
    max_filing_age_days: int = 200                  # a periodic report must have been filed this recently
    min_price_history_days: int = 90                # listed at least this long before D
    price_lookback_days: int = 7                    # last tradable close within this window
    max_implied_price: float = 10_000.0             # float / cover shares above this = scale error
    max_float_jump: float = 20.0                    # vs the previous year's float
    float_to_market_cap: tuple[float, float] = (0.05, 20.0)
    excluded_sic: dict[int, str] = Field(default_factory=lambda: dict(EXCLUDED_SIC))

    @property
    def per_frame(self) -> int:
        return self.discovery_per_frame or 4 * self.top_n

    @property
    def pool_size(self) -> int:
        return int(round(self.pool_multiple * self.top_n))

    def digest(self) -> str:
        payload = json.dumps({"v": UNIVERSE_VERSION, **self.model_dump(mode="json")}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class UniverseMember(BaseModel):
    rank: int
    ticker: str
    cik: int
    name: str | None = None
    market_cap: float
    price: float                     # raw (unadjusted) close used
    price_date: str
    shares: float                    # in units of `ticker`'s class, split-adjusted to price_date
    shares_source: str
    public_float: float
    float_filed: str
    latest_filing: str               # filing date of the latest periodic report used
    symbol_source: Literal["history", "current", "cover_page"]


class UniverseExclusion(BaseModel):
    cik: int
    name: str | None = None
    reason: str
    detail: str = ""


class UniverseSnapshot(BaseModel):
    as_of: str
    config_digest: str
    candidates: int
    members: list[UniverseMember]
    excluded: list[UniverseExclusion] = Field(default_factory=list)

    @property
    def tickers(self) -> list[str]:
        return [m.ticker for m in self.members]


class UniverseSchedule(BaseModel):
    """Snapshots by date; a date's universe is the latest snapshot on or
    before it (reconstitution in between is not seen early)."""

    schema_version: Literal[1] = 1
    kind: Literal["point_in_time_universe"] = "point_in_time_universe"
    config: UniverseConfig
    snapshots: list[UniverseSnapshot]

    def members_on(self, day: str) -> list[str]:
        eligible = [s for s in self.snapshots if s.as_of <= day]
        return max(eligible, key=lambda s: s.as_of).tickers if eligible else []

    def all_tickers(self) -> list[str]:
        seen: list[str] = []
        for s in sorted(self.snapshots, key=lambda s: s.as_of):
            for t in s.tickers:
                if t not in seen:
                    seen.append(t)
        return seen
