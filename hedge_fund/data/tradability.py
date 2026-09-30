"""Which daily bars are real, tradable closes.

Vendors keep printing a halted or delisted stock at its last price: Tiingo
carries SIVB at 106.04 through its March 2023 halt, FRC at 3.51 after its
last NYSE day, TWTR at 53.70 on the suspension day — sometimes with zero
volume, sometimes with a handful of shares. Those prints are not prices
anyone could trade at. A bar counts as a tradable close only if:

- its close is finite and positive and its volume is positive, and
- it is not a carry-forward of the listing's last close after a known
  listing-ending event (data_client.security_event): once the last trading
  day has passed, a close identical to the last listed close is the
  vendor's placeholder, not a trade.

Only bars in [start, end] are read, so nothing after *end* can leak in.
"""

from __future__ import annotations

from datetime import date, timedelta
from math import isclose, isfinite

from hedge_fund.data.protocol import DataClient


def tradable_closes(data_client: DataClient, ticker: str, start: str, end: str) -> dict[str, float]:
    """{day: close} for tradable closes of *ticker* in [start, end]."""
    valid = _traded(data_client, ticker, start, end)
    lookup = getattr(data_client, "security_event", None)
    event = lookup(ticker) if callable(lookup) else None
    if event is None or end <= event.last_trading_day:
        return dict(valid)
    # The listing's last close, looked up on its own (it may predate *start*).
    ltd = event.last_trading_day
    listed = _traded(data_client, ticker, (date.fromisoformat(ltd) - timedelta(days=_LISTED_LOOKBACK_DAYS)).isoformat(), ltd)
    last_listed = listed[-1][1] if listed else None
    return {d: c for d, c in valid
            if d <= ltd or last_listed is None or not isclose(c, last_listed, rel_tol=1e-9)}


_LISTED_LOOKBACK_DAYS = 45


def _traded(data_client: DataClient, ticker: str, start: str, end: str) -> list[tuple[str, float]]:
    """(day, close) with a finite positive close and positive volume, sorted."""
    return sorted((b.time[:10], b.close) for b in data_client.get_prices(ticker, start, end)
                  if start <= b.time[:10] <= end and b.close is not None and isfinite(b.close) and b.close > 0
                  and (b.volume or 0) > 0)
