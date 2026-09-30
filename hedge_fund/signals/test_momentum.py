"""MomentumModel tests — synthetic daily closes."""

import math
from datetime import date, timedelta

import pytest

from hedge_fund.data.models import Price
from hedge_fund.signals.momentum import MomentumModel


class Series:
    def __init__(self, closes):
        self.closes = closes    # list of (date, close), oldest first

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        return [Price(open=c, high=c, low=c, close=c, volume=1, time=f"{d}T00:00:00Z")
                for d, c in self.closes if start_date <= d <= end_date]


def trend(days, daily_return, wiggle=0.01, start=100.0):
    """`days` weekday closes growing by `daily_return`, with alternating noise so volatility isn't zero."""
    out, day, price = [], date(2025, 1, 1), start
    while len(out) < days:
        if day.weekday() < 5:
            price *= 1 + daily_return + (wiggle if len(out) % 2 else -wiggle)
            out.append((day.isoformat(), price))
        day += timedelta(days=1)
    return out


def test_uptrend_is_bullish_and_downtrend_bearish():
    up, down = trend(300, 0.002), trend(300, -0.002)
    as_of = up[-1][0]
    bull = MomentumModel().predict("UP", as_of, Series(up))
    bear = MomentumModel().predict("DN", as_of, Series(down))
    assert 0 < bull.value < 1 and -1 < bear.value < 0
    assert bull.metadata["return_12_1"] > 0 and bull.metadata["volatility"] > 0


def test_skips_the_most_recent_month():
    closes = trend(300, 0.001)
    crash = [(d, c * 0.5) for d, c in closes[-10:]]    # last two weeks crash
    as_of = closes[-1][0]
    calm = MomentumModel().predict("X", as_of, Series(closes)).metadata
    crashed = MomentumModel().predict("X", as_of, Series(closes[:-10] + crash)).metadata
    # The crash is inside the skipped month: the return is untouched; only volatility sees it.
    assert crashed["return_12_1"] == pytest.approx(calm["return_12_1"])
    assert crashed["volatility"] > calm["volatility"]


def test_value_is_return_over_volatility_squashed():
    closes = trend(300, 0.001)
    signal = MomentumModel().predict("X", closes[-1][0], Series(closes))
    m = signal.metadata
    assert signal.value == pytest.approx(math.tanh(m["return_12_1"] / m["volatility"]))


def test_abstains_without_a_year_of_history():
    closes = trend(200, 0.002)
    signal = MomentumModel().predict("NEW", closes[-1][0], Series(closes))
    assert signal.value == 0.0 and signal.metadata["abstained"] is True


def test_registered_as_long_short_quant_model():
    from hedge_fund.signals import ALPHA_MODEL_REGISTRY, get_investment_approach
    assert ALPHA_MODEL_REGISTRY["momentum"] is MomentumModel
    assert get_investment_approach("momentum") == "long_short"
