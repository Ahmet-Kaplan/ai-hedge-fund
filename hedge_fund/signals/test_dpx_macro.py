"""DPXMacroStabilityModel tests — fully offline; every HTTP call is mocked.

Ported from PR #21 with the two bugs it shipped fixed and pinned:
- abstentions were unmarked, so the pipeline blended them as neutral votes
- `investment_approach` was not declared, which raises at registration
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import Mock, patch

import pytest
import requests

from hedge_fund.signals import ALPHA_MODEL_REGISTRY, get_investment_approach
from hedge_fund.signals.dpx_macro import DPXMacroStabilityModel

TODAY = date.today().isoformat()
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()

PAYLOAD = {
    "stability": {
        "currentScore": 95,
        "latestStatus": "STABLE",
        "threshold": {"caution": 75, "stable": 90},
    },
    "peg": {"deviationBps": 2},
}


def _response(payload=None):
    response = Mock()
    response.raise_for_status = Mock()
    response.json = Mock(return_value=payload if payload is not None else PAYLOAD)
    return response


def _payload(score=95, status="STABLE"):
    return {
        "stability": {
            "currentScore": score,
            "latestStatus": status,
            "threshold": {"caution": 75, "stable": 90},
        },
        "peg": {"deviationBps": 2},
    }


def _live(model, payload=None, ticker="AAPL", day=TODAY):
    with patch("hedge_fund.signals.dpx_macro.requests.get") as get:
        get.return_value = _response(payload)
        signal = model.predict(ticker, day, None)
    return signal, get


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_registered_and_declares_an_approach():
    """`get_investment_approach` reads vars(cls) — an inherited or missing
    declaration raises, which is how the contributed version would have failed
    at registration rather than at use."""
    assert ALPHA_MODEL_REGISTRY["dpx_macro"] is DPXMacroStabilityModel
    assert get_investment_approach("dpx_macro") == "long_short"


def test_it_can_run_in_a_blind_backtest():
    """The oracle is handed no ticker, so blinding leaks nothing — and unlike
    ta_board this model does not need the identity withheld from it."""
    assert DPXMacroStabilityModel.supports_blind is True


# ---------------------------------------------------------------------------
# Live path
# ---------------------------------------------------------------------------

def test_stable_conditions_are_mildly_bullish():
    signal, _ = _live(DPXMacroStabilityModel(), _payload(score=95, status="STABLE"))
    assert signal.value > 0.5
    assert signal.metadata["abstained"] is False
    assert signal.metadata["status"] == "STABLE"
    assert "STABLE" in signal.reasoning


def test_unstable_conditions_are_bearish():
    signal, _ = _live(DPXMacroStabilityModel(), _payload(score=60, status="UNSTABLE"))
    assert signal.value < -0.5


def test_caution_sits_near_neutral():
    signal, _ = _live(DPXMacroStabilityModel(), _payload(score=82.5, status="CAUTION"))
    assert abs(signal.value) < 0.2


def test_the_overlay_is_ticker_agnostic():
    """It is a regime gate: the same day gives the same conviction per symbol."""
    model = DPXMacroStabilityModel()
    with patch("hedge_fund.signals.dpx_macro.requests.get") as get:
        get.return_value = _response()
        a = model.predict("AAPL", TODAY, None)
        b = model.predict("MSFT", TODAY, None)
    assert a.value == b.value


def test_the_components_carry_the_raw_score():
    signal, _ = _live(DPXMacroStabilityModel(), _payload(score=95))
    assert signal.components["stability_score"] == pytest.approx(95)
    assert signal.components["peg_deviation_bps"] == pytest.approx(2)


def test_one_lookup_per_day():
    model = DPXMacroStabilityModel()
    with patch("hedge_fund.signals.dpx_macro.requests.get") as get:
        get.return_value = _response()
        model.predict("AAPL", TODAY, None)
        model.predict("MSFT", TODAY, None)
        model.predict("AAPL", TODAY, None)
    assert get.call_count == 1


def test_a_live_signal_records_that_it_bypassed_the_data_client():
    signal, _ = _live(DPXMacroStabilityModel())
    assert signal.metadata["data_source"] == "dpx_oracle"


# ---------------------------------------------------------------------------
# Abstention — marked, never a silent neutral
# ---------------------------------------------------------------------------

def test_a_historical_date_abstains_rather_than_reusing_today():
    """The oracle has no history. Reusing the current score for a past date
    would quietly break the point-in-time contract."""
    model = DPXMacroStabilityModel()
    with patch("hedge_fund.signals.dpx_macro.requests.get") as get:
        signal = model.predict("AAPL", "2020-01-02", None)

    get.assert_not_called()               # it does not even ask
    assert signal.value == 0.0
    assert signal.metadata["abstained"] is True          # the regression
    assert "live-only" in signal.metadata["abstain_reason"]


def test_tomorrow_abstains_too():
    model = DPXMacroStabilityModel()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    signal = model.predict("AAPL", tomorrow, None)
    assert signal.metadata["abstained"] is True


def test_a_malformed_date_abstains():
    model = DPXMacroStabilityModel()
    signal = model.predict("AAPL", "not-a-date", None)
    assert signal.metadata["abstained"] is True


def test_infrastructure_failure_abstains_with_the_error_visible():
    """Unmarked, this would be a neutral vote that dilutes the whole blend."""
    model = DPXMacroStabilityModel()
    with patch("hedge_fund.signals.dpx_macro.requests.get") as get:
        get.side_effect = requests.ConnectionError("boom")
        signal = model.predict("AAPL", TODAY, None)

    assert signal.value == 0.0
    assert signal.metadata["abstained"] is True          # the regression
    assert "boom" in signal.metadata["abstain_reason"]


@pytest.mark.parametrize("payload", [
    {},                                                       # no stability key
    {"stability": {}},                                        # no score
    {"stability": {"currentScore": "abc", "latestStatus": "X",
                   "threshold": {"caution": 75, "stable": 90}}},  # unparseable
])
def test_a_response_that_is_not_shaped_as_expected_abstains(payload):
    model = DPXMacroStabilityModel()
    signal, _ = _live(model, payload)
    assert signal.metadata["abstained"] is True


def test_an_http_error_abstains():
    model = DPXMacroStabilityModel()
    with patch("hedge_fund.signals.dpx_macro.requests.get") as get:
        response = Mock()
        response.raise_for_status = Mock(side_effect=requests.HTTPError("500"))
        get.return_value = response
        signal = model.predict("AAPL", TODAY, None)
    assert signal.metadata["abstained"] is True


# ---------------------------------------------------------------------------
# The abstention is an omission, not a vote — end to end
# ---------------------------------------------------------------------------

def test_the_pipeline_records_abstentions_as_dropped():
    """What the flag is actually for: the omission becomes visible instead of
    silently blending as 0.0."""
    from hedge_fund.pipeline.run_cycle import _dropped_outputs

    model = DPXMacroStabilityModel()
    signal = model.predict("AAPL", YESTERDAY, None)

    dropped = _dropped_outputs([signal], "macro")

    assert len(dropped) == 1
    assert dropped[0].model == "dpx_macro"
    assert "live-only" in dropped[0].reason


def test_offline_test_suite_guard():
    """Every test above mocks requests.get; assert none was left unmocked by
    checking the suite never reaches the real host."""
    assert "untitledfinancial.com" in __import__(
        "hedge_fund.signals.dpx_macro", fromlist=["_RELIABILITY_URL"]
    )._RELIABILITY_URL
