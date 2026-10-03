"""Tests for alpha models (AlphaModel/QuantModel + PEADModel)."""

from __future__ import annotations

import pytest

from hedge_fund.data.client import FDClientError
from hedge_fund.data.models import EarningsData, EarningsRecord
from hedge_fund.signals import MeanReversionModel, MomentumModel, PEADModel, QuantModel
from hedge_fund.signals.base import AlphaModel
from hedge_fund.models import Signal


class MockFDClient:
    """Returns canned earnings history for testing without API calls."""

    def __init__(self, earnings=None):
        self._earnings = earnings or []

    def get_earnings_history(self, ticker, limit=12):
        if isinstance(self._earnings, Exception):
            raise self._earnings
        return self._earnings


def _rec(report_period, filing_date, surprise, source_type="8-K"):
    return EarningsRecord(
        ticker="TEST", report_period=report_period, source_type=source_type,
        filing_date=filing_date,
        quarterly=EarningsData(eps_surprise=surprise) if surprise else None,
    )


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class TestInterface:
    def test_quant_model_is_alpha_model(self):
        assert issubclass(QuantModel, AlphaModel)
        assert issubclass(PEADModel, QuantModel)
        assert issubclass(MomentumModel, QuantModel)
        assert issubclass(MeanReversionModel, QuantModel)

    def test_name(self):
        assert PEADModel().name == "pead"
        assert MomentumModel().name == "momentum"
        assert MeanReversionModel().name == "mean_reversion"

    def test_helpers(self):
        assert QuantModel._safe_float(None) == 0.0
        assert QuantModel._safe_float("3.5") == 3.5
        assert QuantModel._normalize_to_signal(2.0) == 1.0
        assert QuantModel._normalize_to_signal(-2.0) == -1.0


# ---------------------------------------------------------------------------
# PEADModel.predict
# ---------------------------------------------------------------------------

class TestPEADPredict:
    def test_beat_fires_long(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 1.0
        assert sig.model_name == "pead"
        assert "BEAT" in sig.reasoning

    def test_miss_fires_short(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "MISS")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == -1.0

    def test_meet_is_neutral(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "MEET")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 0.0

    def test_no_earnings_is_neutral(self):
        fd = MockFDClient([])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 0.0

    def test_earnings_infra_failure_is_not_neutral(self):
        """Empty history is no-view; a data-client infra error must raise."""
        fd = MockFDClient(FDClientError("API down", status_code=500, path="/earnings/"))
        with pytest.raises(FDClientError) as exc_info:
            PEADModel().predict("TEST", "2025-08-01", fd)
        assert exc_info.value.status_code == 500

    def test_stale_event_is_neutral(self):
        # Event filed 30 days before the query date — outside the freshness window
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-08-31", fd)
        assert sig.value == 0.0

    def test_point_in_time_ignores_future_filings(self):
        # A filing dated after the query date must not be visible (no lookahead)
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-07-15", fd)
        assert sig.value == 0.0

    def test_freshness_window_bridges_weekend(self):
        # Filed Saturday 2025-08-02; queried Monday 2025-08-04 (2 days) → still fresh
        fd = MockFDClient([_rec("2025-06-30", "2025-08-02", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-08-04", fd)
        assert sig.value == 1.0

    def test_45_day_retrospective_filter(self):
        # Filing is 100+ days after the report period → retrospective, excluded
        fd = MockFDClient([_rec("2025-12-31", "2026-04-13", "BEAT")])
        sig = PEADModel().predict("TEST", "2026-04-13", fd)
        assert sig.value == 0.0

    def test_dedup_prefers_8k(self):
        # Same report period via 8-K and 10-Q; 8-K should be the chosen source
        fd = MockFDClient([
            _rec("2025-06-30", "2025-08-01", "BEAT", source_type="8-K"),
            _rec("2025-06-30", "2025-08-02", "BEAT", source_type="10-Q"),
        ])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 1.0
        assert sig.metadata["source_type"] == "8-K"

    def test_10q_only_period_is_neutral(self):
        # No 8-K for the period — 10-Q filing date is not the announcement
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT", source_type="10-Q")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 0.0

    def test_10q_fallback_when_announcement_only_false(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT", source_type="10-Q")])
        sig = PEADModel(announcement_only=False).predict("TEST", "2025-08-01", fd)
        assert sig.value == 1.0
        assert sig.metadata["source_type"] == "10-Q"

    def test_dedup_earliest_8k_wins(self):
        # Two 8-Ks for the same period; keep the earlier filing date
        fd = MockFDClient([
            _rec("2025-06-30", "2025-08-15", "BEAT", source_type="8-K"),
            _rec("2025-06-30", "2025-08-01", "BEAT", source_type="8-K"),
        ])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 1.0
        assert sig.metadata["filing_date"] == "2025-08-01"

    def test_returns_signal_type(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert isinstance(sig, Signal)


class TestPEADAbstainsRatherThanVoting:
    """PEAD's silence must be an omission, not a neutral opinion.

    `blend_signals` keeps any signal that is not flagged `abstained`
    (portfolio/construction.py), so an unmarked 0.0 dilutes every model that
    did form a view and never shows up in the receipt's dropped outputs.
    `momentum`, `mean_reversion` and `news_sentiment` already comply; PEAD was
    the outlier. Ported from PR #24's `de6c727`, which caught it.
    """

    def _abstained(self, sig):
        assert sig.value == 0.0
        assert sig.metadata["abstained"] is True
        assert sig.metadata["abstain_reason"]
        assert sig.reasoning.startswith("abstained:")

    def test_no_surprise_on_file_abstains(self):
        self._abstained(PEADModel().predict("TEST", "2025-08-01", MockFDClient([])))

    def test_a_meet_abstains(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "MEET")])
        self._abstained(PEADModel().predict("TEST", "2025-08-01", fd))

    def test_a_stale_event_abstains_and_says_how_stale(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-08-31", fd)
        self._abstained(sig)
        assert "30 days old" in sig.metadata["abstain_reason"]
        assert "drift window" in sig.metadata["abstain_reason"]

    def test_a_future_filing_abstains(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        self._abstained(PEADModel().predict("TEST", "2025-07-15", fd))

    def test_a_real_surprise_still_votes(self):
        """The fix must not silence the signal PEAD exists to produce."""
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 1.0
        assert sig.metadata.get("abstained") is not True

    def test_an_abstention_is_excluded_from_the_blend(self):
        """What the flag is for, measured: an unmarked 0.0 halves a real view."""
        from hedge_fund.models import Signal
        from hedge_fund.portfolio.construction import blend_signals

        weights = {"pead": 1.0, "buffett": 1.0}
        approaches = {"pead": "long_short", "buffett": "long_short"}
        voted = Signal(model_name="buffett", ticker="TEST", date="2025-08-01", value=1.0)

        def blend(pead):
            return blend_signals([pead, voted], weights, gross_target=1.0,
                                 mode="long_short", investment_approaches=approaches)

        abstention = PEADModel().predict("TEST", "2025-08-01", MockFDClient([]))
        voting = Signal(model_name="pead", ticker="TEST", date="2025-08-01", value=0.0)

        assert blend(abstention).convictions["TEST"] == pytest.approx(1.0)
        # The bug this replaced: PEAD's silence counted as a neutral opinion
        # and diluted the only model that actually had a view.
        assert blend(voting).convictions["TEST"] == pytest.approx(0.5)
