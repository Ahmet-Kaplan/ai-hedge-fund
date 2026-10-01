"""When the paper fund trades — the backtest's cadence, restated for live sessions.

The backtester assesses on the last session of each period and executes at
the next session's close. Live, the same rule reads: execute on the first
session of a new period, with data through the previous one.
"""

from __future__ import annotations

from datetime import date, datetime, time

from hedge_fund.data.sessions import NEW_YORK

# Orders are day market orders: sent before the open they fill at the open;
# sent during the session they fill at once. After this a day order could
# miss the close and expire, so the runner stops sending.
ORDER_CUTOFF = time(15, 50)
# Fills land from the open on; the previous session's closing book is only
# observable before then.
RECONCILE_DEADLINE = time(9, 30)


def is_rebalance_day(session: str, previous_session: str, cadence: str) -> bool:
    """True when `session` opens a new rebalance period relative to the previous session."""
    today, prev = date.fromisoformat(session), date.fromisoformat(previous_session)
    if cadence == "daily":
        return True
    if cadence == "weekly":
        return today.isocalendar()[:2] != prev.isocalendar()[:2]
    if cadence == "monthly":
        return (today.year, today.month) != (prev.year, prev.month)
    raise ValueError(f"unknown rebalance cadence {cadence!r}")


def ny_midnight(session: str) -> str:
    """Start of `session` in New York, as an ISO timestamp for order queries."""
    return datetime.combine(date.fromisoformat(session), time(0), NEW_YORK).isoformat()


def client_order_prefix(fund_name: str, session: str) -> str:
    """Every order a fund sends on a session starts with this; the ticker completes it."""
    return f"{fund_name}-{session}-"
