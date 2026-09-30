"""Benchmark series that match what the portfolio is credited with.

A portfolio that is credited dividends must be compared with a total-return
benchmark; a price-only benchmark would flatter it by the benchmark's yield.
`benchmark_curve` returns both and says which one is comparable.
"""

from __future__ import annotations

import pandas as pd


def benchmark_curve(panel, symbol: str, sessions: list[str], capital: float,
                    *, total_return: bool) -> pd.Series:
    """Capital invested in *symbol* at the first session's close.

    total_return=True reinvests each ex-date dividend at that session's close.
    Uses only data up to each session (read through as-of views).
    """
    if not sessions:
        return pd.Series(dtype=float)
    view = panel.as_of(sessions[-1])
    close = view.bars("close", tickers=[symbol], tradable_only=False)[symbol].reindex(sessions)
    div = view.bars("dividend", tickers=[symbol])[symbol].reindex(sessions).fillna(0.0)
    if close.isna().any():
        close = close.ffill()
    # Work in the final view's split basis: price levels there are consistent over
    # the window, and dividends/splits restated the same way.
    units = capital / close.iloc[0]
    values = []
    for i, s in enumerate(sessions):
        if i and total_return and div.iloc[i]:
            units += units * div.iloc[i] / close.iloc[i]
        values.append(units * close.iloc[i])
    return pd.Series(values, index=sessions, name=f"{symbol}_{'tr' if total_return else 'price'}")
