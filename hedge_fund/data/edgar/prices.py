"""The price inputs EDGAR valuation needs, and nothing more.

Market cap and P/E are computed as of each filing: shares from that filing
times the ticker's close on (or just before) its filing date. SEC share
counts are as reported at the time, so the close must be the *raw,
unadjusted* close of that day — a split-adjusted close would mis-state
market cap by the cumulative split ratio. (The FD `get_prices` series is
split-adjusted, so it cannot be used here.)

Per-share fields (EPS, book value, FCF per share) are also as reported;
split events let the adapter restate older rows onto the share basis of the
query date, using only splits effective on or before that date.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RawPriceSource(Protocol):
    def raw_closes(self, ticker: str, start_date: str, end_date: str) -> dict[str, float]:
        """{YYYY-MM-DD: unadjusted close} for sessions in [start, end]."""
        ...

    def split_events(self, ticker: str, start_date: str, end_date: str) -> dict[str, float]:
        """{YYYY-MM-DD: split factor (new shares per old share)} for splits
        effective in [start, end]; e.g. a 4-for-1 split is 4.0."""
        ...


class StaticRawPrices:
    """In-memory RawPriceSource for tests and pre-loaded data."""

    def __init__(self, closes: dict[str, dict[str, float]] | None = None,
                 splits: dict[str, dict[str, float]] | None = None) -> None:
        self._closes = {t.upper(): dict(v) for t, v in (closes or {}).items()}
        self._splits = {t.upper(): dict(v) for t, v in (splits or {}).items()}

    def raw_closes(self, ticker, start_date, end_date):
        return {d: c for d, c in self._closes.get(ticker.upper(), {}).items() if start_date <= d <= end_date}

    def split_events(self, ticker, start_date, end_date):
        return {d: f for d, f in self._splits.get(ticker.upper(), {}).items() if start_date <= d <= end_date}
