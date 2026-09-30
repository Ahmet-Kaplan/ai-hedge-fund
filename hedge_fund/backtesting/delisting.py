"""What happens to a held position when its stock stops trading.

Run once per session, before execution and valuation. For every held name
without a close on the session, in order:

0. Continuation. If the name still has a close, nothing changes — but when
   a curated event says its listing ended and a successor carries the same
   shares, and the vendor serves the successor's prices under this symbol,
   the crossing is recorded once as a DELISTING event (no cash moves).
1. Successor. If a curated security event (data_client.security_event)
   says the listing ended before this session and names a successor symbol
   for the same shares (SIVB -> SIVBQ, FRC -> FRCB), and the successor has
   a close on this session, the position is re-registered under the
   successor share for share. No cash moves; it is marked at the
   successor's close from here on.
2. Grace. Otherwise the position is frozen at its last real close while
   the gap is short — at most `grace_sessions` sessions for an unexplained
   gap (a halt may resume), at most `successor_wait_sessions` while a mapped
   successor has not yet started trading, and not at all once a known
   take-private/acquisition has ended the listing. Frozen names are valued
   at that close but never traded.
3. Liquidation. When the grace runs out, the position is closed at the last
   valid close on or before the listing's last trading day (or on or before
   the session, when no event is known), through the broker's normal fill
   path, and recorded as a DELISTING event.

Only tradable closes count (data/tradability.py: vendors' carry-forward
prints during halts and after delistings are excluded). Nothing reads a
price dated after the session, a curated event is used only
after its last trading day, no price is invented, and carrying a stale close
is bounded. If no close is found at all within `lookback_days`, the last
price the backtest itself filled the name at is used and the event says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import isclose
from typing import Literal

from pydantic import BaseModel

from hedge_fund.brokers.models import Order
from hedge_fund.brokers.sim import SimBroker
from hedge_fund.data.protocol import DataClient
from hedge_fund.data.tradability import tradable_closes

DelistingPolicyName = Literal[
    "successor_symbol",
    "successor_prices_under_original_symbol",
    "known_delisting_last_close",
    "successor_not_trading_last_close",
    "no_close_within_grace_last_close",
    "last_fill_price",
]


@dataclass(frozen=True)
class DelistingPolicy:
    grace_sessions: int = 5
    successor_wait_sessions: int = 20
    lookback_days: int = 45


DEFAULT_POLICY = DelistingPolicy()


class DelistingEvent(BaseModel):
    """One held position that stopped trading, and what the backtest did."""

    kind: Literal["DELISTING"] = "DELISTING"
    ticker: str
    session: str                       # backtest session on which it was processed
    delisting_date: str                # last trading day (event) or last close seen
    action: Literal["liquidated", "converted_to_successor", "continued_as_successor"]
    policy: DelistingPolicyName
    event_type: str | None = None      # acquisition | take_private | bank_failure | ... | None if unexplained
    shares: int                        # signed shares affected
    price: float                       # liquidation reference, or the successor-era close on conversion/continuation
    price_date: str
    proceeds: float                    # cash credited (negative when covering a short); 0 on conversion
    successor: str | None = None
    note: str = ""


@dataclass
class Resolution:
    events: list[DelistingEvent]
    frozen: dict[str, tuple[float, str]]   # ticker -> (last close, its date), held but not trading today


def resolve_delistings(
    broker: SimBroker, session: str, sessions: list[str], data_client: DataClient,
    last_fills: dict[str, float], policy: DelistingPolicy = DEFAULT_POLICY,
) -> Resolution:
    """Apply the policy to every held name lacking a close on *session*.

    *sessions*: the benchmark sessions of the backtest up to and including
    *session* (the trading calendar used to count missed sessions).
    """
    lookup = getattr(data_client, "security_event", None)
    events: list[DelistingEvent] = []
    frozen: dict[str, tuple[float, str]] = {}
    for ticker, position in sorted(broker.positions().items()):
        closes = _closes(data_client, ticker, session, policy.lookback_days)
        event = lookup(ticker) if callable(lookup) else None
        ended = event is not None and session > event.last_trading_day
        if closes and max(closes) == session:
            # Trading today. If the listing ended but the vendor serves the
            # successor's real trading under this symbol (Tiingo: SIVB's OTC
            # trading; FRC served from FRCB), the position carries on at
            # those prices — record the delisting on the first such day.
            first_after = ended and not any(event.last_trading_day < d < session for d in closes)
            if ended and event.successor and first_after:
                events.append(DelistingEvent(
                    ticker=ticker, session=session, delisting_date=event.last_trading_day,
                    action="continued_as_successor", policy="successor_prices_under_original_symbol",
                    event_type=event.event_type, shares=position.shares, price=closes[session], price_date=session,
                    proceeds=0.0, successor=event.successor, note=event.note,
                ))
            continue

        listed = {d: c for d, c in closes.items() if d <= event.last_trading_day} if ended else {}
        if ended and event.successor:
            successor_close = _closes(data_client, event.successor, session, policy.lookback_days).get(session)
            # the successor series may repeat the dead listing's last close
            # as a placeholder (Tiingo: SIVBQ at 106.04 during SIVB's halt)
            last_listed = listed[max(listed)] if listed else None
            if successor_close is not None and (last_listed is None or not isclose(successor_close, last_listed, rel_tol=1e-9)):
                shares = broker.convert_position(ticker, event.successor)
                events.append(DelistingEvent(
                    ticker=ticker, session=session, delisting_date=event.last_trading_day,
                    action="converted_to_successor", policy="successor_symbol", event_type=event.event_type,
                    shares=shares, price=successor_close, price_date=session, proceeds=0.0,
                    successor=event.successor, note=event.note,
                ))
                continue

        if ended:
            reference = listed or closes
            wait = policy.successor_wait_sessions if event.successor else 0
        else:
            reference, wait = closes, policy.grace_sessions
        last_day = max(reference) if reference else None
        missed = sum(1 for d in sessions if last_day is not None and last_day < d <= session)
        if last_day is not None and missed <= wait:
            frozen[ticker] = (reference[last_day], last_day)
            continue

        if last_day is not None:
            price, price_date = reference[last_day], last_day
            name: DelistingPolicyName = (
                ("successor_not_trading_last_close" if event.successor else "known_delisting_last_close")
                if ended else "no_close_within_grace_last_close")
        elif ticker in last_fills:
            price, price_date, name = last_fills[ticker], session, "last_fill_price"
        else:
            raise ValueError(f"{ticker}: held on {session} with no close in {policy.lookback_days} days "
                             "and no fill price — the position was never priced")
        shares = position.shares
        broker.place_order(Order(ticker=ticker, side="sell" if shares > 0 else "buy", quantity=abs(shares), price=price))
        events.append(DelistingEvent(
            ticker=ticker, session=session,
            delisting_date=event.last_trading_day if ended else (last_day or session),
            action="liquidated", policy=name, event_type=event.event_type if ended else None,
            shares=shares, price=price, price_date=price_date, proceeds=shares * price,
            successor=event.successor if ended else None, note=event.note if ended else "",
        ))
    return Resolution(events=events, frozen=frozen)


def _closes(data_client: DataClient, ticker: str, session: str, lookback_days: int) -> dict[str, float]:
    """Tradable closes in [session - lookback, session]; never after *session*."""
    start = (date.fromisoformat(session) - timedelta(days=lookback_days)).isoformat()
    return tradable_closes(data_client, ticker, start, session)
