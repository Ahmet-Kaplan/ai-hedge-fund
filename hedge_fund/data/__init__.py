"""v2 data pipeline — data provider protocol, FD client, and response models.

Empty collections / ``None`` mean the data does not exist. Infrastructure
failures raise ``FDClientError`` (auth, quota, network, HTTP >= 400 except
404). ``get_news`` returning ``[]`` is genuine no-articles, not a failed
fetch; there is no sentiment endpoint — empty news means no narrative
input, not a zero signal. See ``DataClient`` and ``FDClient.get_news``.
"""

from hedge_fund.data.cached import CachedDataClient
from hedge_fund.data.client import FDClient, FDClientError
from hedge_fund.data.fxmacrodata import FXMacroDataClient, FXMacroDataClientError
from hedge_fund.data.models import (
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsData,
    EarningsRecord,
    Filing,
    FinancialMetrics,
    InsiderTrade,
    Price,
)
from hedge_fund.data.factory import data_source, missing_data_keys, open_data_client
from hedge_fund.data.free import FreeDataClient
from hedge_fund.data.protocol import DataClient

__all__ = [
    "CachedDataClient",
    "CompanyFacts",
    "CompanyNews",
    "DataClient",
    "Earnings",
    "EarningsData",
    "EarningsRecord",
    "FDClient",
    "FDClientError",
    "Filing",
    "FXMacroDataClient",
    "FXMacroDataClientError",
    "FinancialMetrics",
    "FreeDataClient",
    "InsiderTrade",
    "Price",
    "data_source",
    "missing_data_keys",
    "open_data_client",
]
