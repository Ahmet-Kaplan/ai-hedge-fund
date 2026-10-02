"""Analysts that follow a member of Congress's disclosed trades.

Each instance follows one member. The view on a name is built from that
member's filings in it, and nothing else — no prices, no fundamentals — so
the track record the attribution page builds is a clean answer to one
question: was following this person worth it?

Three decisions do the real work here.

*Disclosure dating.* A filing enters the model on its disclosure date, never
its transaction date. The STOCK Act allows up to 45 days between the two,
and keying on the transaction would let the backtest buy alongside a member
weeks before anyone could have known they bought. That is not a better
signal, it is a fabricated one, and it is the single easiest way to produce
a congressional-trading strategy that looks extraordinary and is worthless.

*Decay.* A filing stops being news. Conviction falls linearly to zero across
the lookback window rather than persisting, because a purchase disclosed
three months ago is not a reason to buy today — and a model that held its
view forever would mostly be measuring whether the market rose.

*Abstention over neutrality.* A member who has disclosed nothing in a name
has not said it is fairly valued; they have said nothing. The model abstains,
which keeps it out of the pod's blend entirely instead of diluting the
analysts who did have a view.
"""

from __future__ import annotations

import logging
import threading
from datetime import date as _date, timedelta

from hedge_fund.data.congress import Chamber, CongressClient, Disclosure
from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel

logger = logging.getLogger(__name__)

# How long a filing stays relevant. Roughly a quarter: long enough that the
# 45-day disclosure deadline leaves usable life in the signal, short enough
# that the view is about now rather than about last year.
LOOKBACK_DAYS = 90

# How many pages of filings to pull per chamber. The endpoint returns the
# most recent disclosures, so this bounds how far back the roster can see —
# a backtest reaching past it will find members abstaining for lack of data
# rather than for lack of trades, which is a limit of the feed and is stated
# here rather than hidden.
#
# One page because entry FMP plans serve page 0 and 402 on anything past it.
# Asking for more does not degrade to less: the error surfaces inside
# predict(), where it takes down the whole tick rather than one analyst. At
# 25 filings that is a few days of disclosures, far short of the 90-day
# window above — raise this as soon as the plan allows paging.
DEFAULT_PAGES = 1

# A filed range of $15,001-$50,000 is most of what Congress discloses, so
# size is a mild tilt and never the whole signal. An unparseable amount
# lands on the neutral floor instead of silencing a filing whose direction
# was disclosed perfectly well.
_SIZE_FLOOR = 0.5
_SIZE_REFERENCE = 100_000.0

_cache: dict[Chamber, list[Disclosure]] = {}
_cache_lock = threading.Lock()


def disclosures_for(chamber: Chamber, *, pages: int = DEFAULT_PAGES) -> list[Disclosure]:
    """Every cached filing for *chamber*, fetching once per process.

    Fifty analysts asking per ticker per rebalance would otherwise make
    thousands of identical requests. The cache is process-wide and never
    invalidated: a backtest is a fixed historical window, so refetching
    mid-run could only introduce filings the earlier sessions could not see.
    """
    with _cache_lock:
        if chamber not in _cache:
            with CongressClient() as client:
                _cache[chamber] = client.latest(chamber, pages=pages)
            logger.info("loaded %d %s disclosures", len(_cache[chamber]), chamber)
        return _cache[chamber]


def reset_cache() -> None:
    """Drop cached filings. For tests and long-lived processes."""
    with _cache_lock:
        _cache.clear()


class CongressModel(QuantModel):
    """Follows one member of Congress's disclosed trades in a name."""

    # Members disclose both purchases and sales, so the model takes both
    # sides. A long-only pod will clamp it; that is the pod's decision.
    investment_approach = "long_short"

    def __init__(
        self,
        member: str,
        chamber: Chamber,
        *,
        lookback_days: int = LOOKBACK_DAYS,
        pages: int = DEFAULT_PAGES,
    ) -> None:
        self.member = member
        self.chamber = chamber
        self.lookback_days = lookback_days
        self.pages = pages

    @property
    def name(self) -> str:
        return slug(self.member)

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        """Form a view on *ticker* from filings disclosed on or before *date*."""
        filings = self._relevant(ticker, date)
        if not filings:
            return Signal(
                model_name=self.name, ticker=ticker, date=date, value=0.0,
                reasoning=f"{self.member} disclosed no trades in {ticker} in the last {self.lookback_days} days",
                metadata={"abstained": True, "member": self.member, "chamber": self.chamber},
            )

        asof = _date.fromisoformat(date)
        score = sum(
            filing.direction * _recency(filing, asof, self.lookback_days) * _size(filing)
            for filing in filings
        )
        return Signal(
            model_name=self.name, ticker=ticker, date=date,
            value=self._sigmoid(score, scale=1.0),
            reasoning=self._reasoning(filings),
            metadata={
                "member": self.member,
                "chamber": self.chamber,
                "filings": len(filings),
                # Kept so a reader can check the gap themselves rather than
                # taking on trust that the model dated the view honestly.
                "latest_disclosure": filings[0].disclosure_date,
                "latest_transaction": filings[0].transaction_date,
            },
        )

    def _relevant(self, ticker: str, date: str) -> list[Disclosure]:
        """This member's filings in *ticker* that were public by *date*.

        The upper bound is what prevents lookahead and the lower bound is
        what makes the view expire. Both compare disclosure dates.
        """
        asof = _date.fromisoformat(date)
        earliest = (asof - timedelta(days=self.lookback_days)).isoformat()
        wanted = ticker.strip().upper()

        found = [
            filing for filing in disclosures_for(self.chamber, pages=self.pages)
            if filing.ticker == wanted
            and filing.member == self.member
            and filing.direction != 0
            and earliest <= filing.known_by <= date
        ]
        found.sort(key=lambda f: f.known_by, reverse=True)
        return found

    def _reasoning(self, filings: list[Disclosure]) -> str:
        bought = sum(1 for f in filings if f.direction > 0)
        sold = len(filings) - bought
        parts = []
        if bought:
            parts.append(f"{bought} purchase{'' if bought == 1 else 's'}")
        if sold:
            parts.append(f"{sold} sale{'' if sold == 1 else 's'}")
        latest = filings[0]
        lag = _lag_days(latest)
        return (
            f"{self.member} disclosed {' and '.join(parts)}; most recent traded "
            f"{latest.transaction_date}, disclosed {latest.disclosure_date}"
            + (f" ({lag} days later)" if lag is not None else "")
        )


def slug(member: str) -> str:
    """Registry key for a member: lowercase, hyphenated, prefixed.

    Prefixed so a member can never collide with a built-in analyst name, and
    so a mandate reads as following a person rather than a strategy.
    """
    cleaned = "".join(char if char.isalnum() else " " for char in member.lower())
    return "congress-" + "-".join(cleaned.split())


def _recency(filing: Disclosure, asof: _date, lookback_days: int) -> float:
    """Weight falling linearly to zero at the edge of the window."""
    age = (asof - _date.fromisoformat(filing.known_by)).days
    return max(0.0, 1.0 - age / lookback_days)


def _size(filing: Disclosure) -> float:
    """A mild tilt for disclosed size, floored so direction always counts."""
    from math import tanh
    return _SIZE_FLOOR + (1.0 - _SIZE_FLOOR) * tanh(filing.amount_mid / _SIZE_REFERENCE)


def _lag_days(filing: Disclosure) -> int | None:
    """Days between trading and disclosing — the window this model respects."""
    try:
        traded = _date.fromisoformat(filing.transaction_date)
        disclosed = _date.fromisoformat(filing.disclosure_date)
    except ValueError:
        return None
    return (disclosed - traded).days
