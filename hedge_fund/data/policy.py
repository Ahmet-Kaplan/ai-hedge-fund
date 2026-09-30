"""Data-source policy: Financial Datasets is off unless explicitly opted in.

This project runs on Tiingo prices and SEC EDGAR fundamentals. The Financial
Datasets key may still be present in a shared environment, so its presence
must never be enough to reach that API. Access requires the explicit opt-in

    AIHF_ALLOW_FINANCIAL_DATASETS=1

and is checked fail-closed at every entry point: building an FDClient, every
HTTP request it makes, and the data-client factory. Anything other than the
exact value "1" (unset, "0", "true", "yes", ...) means disabled.
"""

from __future__ import annotations

import os

POLICY_ENV = "AIHF_ALLOW_FINANCIAL_DATASETS"


class FinancialDatasetsDisabled(RuntimeError):
    """Financial Datasets was reached while the project policy forbids it."""


def financial_datasets_allowed() -> bool:
    return os.environ.get(POLICY_ENV, "").strip() == "1"


def require_financial_datasets(what: str = "Financial Datasets access") -> None:
    """Raise unless the explicit opt-in is set. Never mentions the API key."""
    if not financial_datasets_allowed():
        raise FinancialDatasetsDisabled(
            f"{what} is disabled by project policy (Tiingo + SEC EDGAR only). "
            f"Set {POLICY_ENV}=1 to opt in deliberately."
        )
