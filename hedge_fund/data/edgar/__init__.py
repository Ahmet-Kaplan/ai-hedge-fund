"""SEC EDGAR fundamentals adapter (point-in-time, no API key)."""

from hedge_fund.data.edgar.client import (
    USER_AGENT_ENV,
    EdgarClient,
    EdgarClientError,
    EdgarConfigError,
    validate_user_agent,
)
from hedge_fund.data.edgar.prices import RawPriceSource, StaticRawPrices

__all__ = [
    "USER_AGENT_ENV",
    "EdgarClient",
    "EdgarClientError",
    "EdgarConfigError",
    "RawPriceSource",
    "StaticRawPrices",
    "validate_user_agent",
]
