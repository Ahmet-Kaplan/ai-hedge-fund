import pytest

from hedge_fund.live.calendar import client_order_prefix, is_rebalance_day, ny_midnight


@pytest.mark.parametrize("session, previous, expected", [
    ("2024-06-10", "2024-06-07", True),    # Monday after Friday
    ("2024-06-11", "2024-06-10", False),   # Tuesday after Monday
    ("2024-05-28", "2024-05-24", True),    # Tuesday after Memorial Day weekend
    ("2025-01-02", "2024-12-31", False),   # same ISO week across New Year
])
def test_weekly_rebalances_on_first_session_of_iso_week(session, previous, expected):
    assert is_rebalance_day(session, previous, "weekly") is expected


def test_daily_and_monthly():
    assert is_rebalance_day("2024-06-11", "2024-06-10", "daily") is True
    assert is_rebalance_day("2025-01-02", "2024-12-31", "monthly") is True
    assert is_rebalance_day("2024-06-11", "2024-06-10", "monthly") is False
    with pytest.raises(ValueError, match="cadence"):
        is_rebalance_day("2024-06-11", "2024-06-10", "hourly")


def test_helpers():
    assert ny_midnight("2024-06-10") == "2024-06-10T00:00:00-04:00"
    assert client_order_prefix("paper-fund", "2024-06-10") == "paper-fund-2024-06-10-"
