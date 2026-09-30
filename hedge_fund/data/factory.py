"""Pick the data source for a run: free (Alpaca + SEC, the default) or Financial Datasets.

    with open_data_client() as data:
        run_cycle(fund, as_of, broker, data, universe)

HEDGE_FUND_DATA=free|fd chooses. Each `with` opens its own client, so threads
(the TUI's workers) never share an HTTP session or a SQLite connection.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from hedge_fund.data.alpaca_prices import AlpacaPriceSource
from hedge_fund.data.cached import CachedDataClient
from hedge_fund.data.client import FDClient
from hedge_fund.data.free import FreeDataClient
from hedge_fund.data.protocol import DataClient
from hedge_fund.data.sec import SecSource
from hedge_fund.data.store import MarketStore
from hedge_fund.paths import MARKET_DB_PATH

REQUIRED_KEYS = {
    "free": ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "SEC_USER_AGENT"],
    "fd": ["FINANCIAL_DATASETS_API_KEY"],
}


def data_source() -> str:
    source = (os.environ.get("HEDGE_FUND_DATA") or "free").strip().lower()
    if source not in REQUIRED_KEYS:
        raise ValueError(f"HEDGE_FUND_DATA must be one of {sorted(REQUIRED_KEYS)}, not {source!r}")
    return source


def missing_data_keys() -> list[str]:
    """Environment variables the selected data source needs but doesn't have."""
    return [name for name in REQUIRED_KEYS[data_source()] if not os.environ.get(name, "").strip()]


@contextmanager
def open_data_client() -> Iterator[DataClient]:
    if data_source() == "fd":
        with FDClient() as raw:
            yield CachedDataClient(raw)
        return
    with FreeDataClient(MarketStore(MARKET_DB_PATH), AlpacaPriceSource(), SecSource()) as client:
        yield client
