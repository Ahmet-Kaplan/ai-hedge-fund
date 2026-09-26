"""Live data-assumption checks — require FINANCIAL_DATASETS_API_KEY.

These observe the provider; they assert only that each check reached a
verdict (no `error`), because the verdict itself is the finding. Read the
printed summaries (pytest -s), or run the CLI for the full JSON report:

    python -m hedge_fund.verification --universe-file configs/baseline-universe.yaml
"""

import os

import pytest

from hedge_fund.verification.runner import (
    CountingFDClient,
    check_batching,
    check_delisted,
    check_point_in_time,
    check_split_adjustment,
    check_valuation_timestamps,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("FINANCIAL_DATASETS_API_KEY"),
    reason="live data verification requires FINANCIAL_DATASETS_API_KEY",
)


@pytest.fixture(scope="module")
def client():
    with CountingFDClient() as c:
        yield c


@pytest.mark.parametrize("check", [check_split_adjustment, check_valuation_timestamps, check_point_in_time, check_delisted, check_batching])
def test_check_reaches_a_verdict(client, check):
    result = check(client)
    print(f"\n[{result.status}] {result.name}: {result.summary}")
    assert result.status != "error", result.summary
