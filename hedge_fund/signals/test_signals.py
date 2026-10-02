"""Tests for alpha models (AlphaModel/QuantModel + PEADModel)."""

from __future__ import annotations

from hedge_fund.data.models import EarningsData, EarningsRecord
from hedge_fund.signals import PEADModel, QuantModel
from hedge_fund.signals.base import AlphaModel
from hedge_fund.models import Signal


class MockFDClient:
    """Returns canned earnings history for testing without API calls."""

    def __init__(self, earnings=None):
        self._earnings = earnings or []

    def get_earnings_history(self, ticker, limit=12):
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

    def test_name(self):
        assert PEADModel().name == "pead"

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
        # A real view must not be filed as an abstention, or attribution would
        # stop scoring the only calls this model actually makes.
        assert sig.metadata.get("abstained") is not True

    def test_miss_fires_short(self):
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "MISS")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == -1.0

    def test_meet_abstains(self):
        # Reporting in line is no surprise, so there is no drift to lean on.
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "MEET")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 0.0
        assert sig.metadata["abstained"] is True

    def test_no_earnings_abstains(self):
        fd = MockFDClient([])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 0.0
        assert sig.metadata["abstained"] is True

    def test_stale_event_abstains(self):
        # Event filed 30 days before the query date — outside the freshness window
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-08-31", fd)
        assert sig.value == 0.0
        assert sig.metadata["abstained"] is True
        # The reason names the event it found and why it no longer counts, so a
        # stale surprise is distinguishable from no surprise at all.
        assert "30 days ago" in sig.metadata["abstain_reason"]
        assert "BEAT" in sig.metadata["abstain_reason"]

    def test_point_in_time_ignores_future_filings(self):
        # A filing dated after the query date must not be visible (no lookahead)
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT")])
        sig = PEADModel().predict("TEST", "2025-07-15", fd)
        assert sig.value == 0.0
        assert sig.metadata["abstained"] is True

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
        assert sig.metadata["abstained"] is True

    def test_dedup_prefers_8k(self):
        # Same report period via 8-K and 10-Q; 8-K should be the chosen source
        fd = MockFDClient([
            _rec("2025-06-30", "2025-08-01", "BEAT", source_type="8-K"),
            _rec("2025-06-30", "2025-08-02", "BEAT", source_type="10-Q"),
        ])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 1.0
        assert sig.metadata["source_type"] == "8-K"

    def test_10q_only_period_abstains(self):
        # No 8-K for the period — 10-Q filing date is not the announcement
        fd = MockFDClient([_rec("2025-06-30", "2025-08-01", "BEAT", source_type="10-Q")])
        sig = PEADModel().predict("TEST", "2025-08-01", fd)
        assert sig.value == 0.0
        assert sig.metadata["abstained"] is True

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
