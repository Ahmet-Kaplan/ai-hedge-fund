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
from hedge_fund.data.composite import CompositeDataClient
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
    """The selected source, with a keyed half for what the free one lacks.

    `free` is the default because it needs no paid subscription. It serves
    prices and SEC fundamentals only, so when a Financial Datasets key is
    *also* present the client is composed: the free source answers prices and
    fundamentals, and the keyed one answers news, insider trades and earnings.

    That composition is the point. Without it, `free` would answer "no news"
    for every ticker, and the two news models would abstain as though the
    company were silent — a data-source gap recorded as a fact about a company.
    With it, a free run keeps everything the free sources do serve and loses
    only what they never had, and only if no key is available at all.
    """
    if data_source() == "fd":
        with FDClient() as raw:
            yield CachedDataClient(raw)
        return

    free = FreeDataClient(MarketStore(MARKET_DB_PATH), AlpacaPriceSource(), SecSource())
    if not os.environ.get("FINANCIAL_DATASETS_API_KEY", "").strip():
        with free:
            yield free
        return

    # Nested rather than composed-into-one-manager: CachedDataClient is a
    # wrapper, not a context manager, so the resources are opened here.
    with free, FDClient() as raw:
        yield CompositeDataClient(free, CachedDataClient(raw))
