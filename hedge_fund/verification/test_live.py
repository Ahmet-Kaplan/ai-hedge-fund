"""Live data-assumption checks against the configured provider (Tiingo +
SEC EDGAR by default). Opt-in: set LIVE_DATA_TESTS=1 with TIINGO_API_KEY and
SEC_USER_AGENT. Prices land in the shared local store, so a repeat run costs
almost no requests.

These observe the provider; they assert only that each check reached a
verdict (no `error`), because the verdict itself is the finding. Read the
printed summaries (pytest -s), or run the CLI for the full JSON report:

    python -m hedge_fund.verification --universe-file configs/baseline-universe.yaml
"""

import os

import pytest

from hedge_fund.data.factory import make_data_client, required_data_env
from hedge_fund.verification.runner import (
    check_delisted,
    check_point_in_time,
    check_price_cache,
    check_split_adjustment,
    check_valuation_timestamps,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("LIVE_DATA_TESTS") or not all(os.environ.get(v) for v in required_data_env()),
    reason="live data verification: set LIVE_DATA_TESTS=1 and the data provider's keys",
)


@pytest.fixture(scope="module")
def client():
    with make_data_client() as c:
        yield c


@pytest.mark.parametrize("check", [check_split_adjustment, check_valuation_timestamps, check_point_in_time, check_delisted, check_price_cache])
def test_check_reaches_a_verdict(client, check):
    result = check(client)
    print(f"\n[{result.status}] {result.name}: {result.summary}")
    assert result.status != "error", result.summary
