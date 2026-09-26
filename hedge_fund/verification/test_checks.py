"""Pure verification judges — synthetic series, no network."""

import pytest

from hedge_fund.verification.checks import (
    aggregate,
    classify_market_cap_timestamp,
    classify_pe_timestamp,
    classify_split_adjustment,
    close_on_or_before,
    evaluate_gap_probe,
    pick_gap_probe,
    pit_violations,
)

# ---------------------------------------------------------------------------
# Split adjustment
# ---------------------------------------------------------------------------

def test_unadjusted_history_shows_the_split_as_a_jump():
    closes = {"2020-08-27": 500.0, "2020-08-28": 499.2, "2020-08-31": 129.0}
    basis, info = classify_split_adjustment(closes, "2020-08-31", 4.0)
    assert basis == "unadjusted"
    assert info["pre_day"] == "2020-08-28" and info["post_day"] == "2020-08-31"


def test_adjusted_history_is_continuous_across_the_split():
    closes = {"2020-08-28": 124.8, "2020-08-31": 129.0}
    assert classify_split_adjustment(closes, "2020-08-31", 4.0)[0] == "adjusted"


def test_move_outside_both_bands_is_unknown():
    closes = {"2020-08-28": 200.0, "2020-08-31": 100.0}  # 2x move against a 4:1 split
    assert classify_split_adjustment(closes, "2020-08-31", 4.0)[0] == "unknown"


def test_missing_side_is_unknown():
    assert classify_split_adjustment({"2020-08-31": 129.0}, "2020-08-31", 4.0)[0] == "unknown"
    assert classify_split_adjustment({}, "2020-08-31", 4.0)[0] == "unknown"


def test_small_ratio_rejected():
    with pytest.raises(ValueError):
        classify_split_adjustment({}, "2020-08-31", 1.5)


# ---------------------------------------------------------------------------
# P/E timestamp
# ---------------------------------------------------------------------------

CANDIDATES = {"report_period": 60.0, "filing_date": 64.0, "latest": 72.0}


@pytest.mark.parametrize("price,expected", [(60.0, "report_period"), (64.3, "filing_date"), (71.5, "latest")])
def test_pe_times_eps_matches_one_close(price, expected):
    stamp, info = classify_pe_timestamp(pe=price / 2.5, eps=2.5, candidates=CANDIDATES)
    assert stamp == expected
    assert info["implied_price"] == pytest.approx(price)


def test_pe_matching_nothing_is_unknown():
    assert classify_pe_timestamp(40.0, 2.5, CANDIDATES)[0] == "unknown"  # implies 100


def test_pe_ambiguous_when_closes_are_indistinguishable():
    stamp, info = classify_pe_timestamp(20.0, 3.0, {"report_period": 60.0, "filing_date": 60.5, "latest": 90.0})
    assert stamp == "unknown" and "ambiguous" in info["reason"]


@pytest.mark.parametrize("pe,eps", [(None, 2.0), (20.0, None), (20.0, -1.0), (float("nan"), 2.0), (-5.0, 2.0)])
def test_pe_unusable_inputs_are_unknown(pe, eps):
    assert classify_pe_timestamp(pe, eps, CANDIDATES)[0] == "unknown"


def test_close_on_or_before_rolls_back_over_weekends():
    closes = {"2024-06-28": 10.0, "2024-07-01": 11.0}
    assert close_on_or_before(closes, "2024-06-30") == 10.0
    assert close_on_or_before(closes, "2024-07-01") == 11.0
    assert close_on_or_before(closes, "2024-06-01") is None


# ---------------------------------------------------------------------------
# Market-cap timestamp
# ---------------------------------------------------------------------------

# Prices: flat 100 at each report_period, 150 at each filing date — so the
# two hypotheses predict very different market-cap ratios.
PERIODS = [("2024-09-30", "2024-10-30"), ("2024-06-30", "2024-07-30"), ("2024-03-31", "2024-04-30"), ("2023-12-31", "2024-01-30")]


def _closes(at_period, at_filing):
    closes = {}
    for i, (period, filed) in enumerate(PERIODS):
        closes[period] = at_period[i]
        closes[filed] = at_filing[i]
    return closes


def _rows(caps):
    return [{"report_period": p, "filing_date": f, "market_cap": c} for (p, f), c in zip(PERIODS, caps)]


def test_market_cap_struck_at_filing_date():
    closes = _closes([100, 100, 100, 100], [150, 120, 90, 110])
    stamp, _ = classify_market_cap_timestamp(_rows([1.5e12, 1.2e12, 0.9e12, 1.1e12]), closes)
    assert stamp == "filing_date"


def test_market_cap_struck_at_report_period():
    closes = _closes([150, 120, 90, 110], [100, 100, 100, 100])
    stamp, _ = classify_market_cap_timestamp(_rows([1.5e12, 1.2e12, 0.9e12, 1.1e12]), closes)
    assert stamp == "report_period"


def test_identical_market_caps_mean_latest_value_stamped_on_history():
    stamp, info = classify_market_cap_timestamp(_rows([2e12] * 4), _closes([1] * 4, [1] * 4))
    assert stamp == "latest" and "identical" in info["reason"]


def test_market_cap_needs_three_usable_rows():
    rows = _rows([1e12, 1e12, None, None])
    assert classify_market_cap_timestamp(rows, {})[0] == "unknown"


def test_market_cap_unclear_fit_is_unknown():
    closes = _closes([100, 100, 100, 100], [100, 100, 100, 100])  # both hypotheses predict flat caps
    stamp, _ = classify_market_cap_timestamp(_rows([1.5e12, 1.2e12, 0.9e12, 1.1e12]), closes)
    assert stamp == "unknown"


# ---------------------------------------------------------------------------
# Point in time
# ---------------------------------------------------------------------------

def test_pit_violations_flags_each_problem():
    rows = [
        {"report_period": "2024-06-30", "filing_date": "2024-08-01"},  # fine
        {"report_period": "2024-09-30", "filing_date": "2024-11-01"},  # after end_date
        {"report_period": "2024-03-31", "filing_date": None},          # unprovable
        {"report_period": "2024-03-31", "filing_date": "2024-03-01"},  # filed before period end
    ]
    problems = [v["problem"] for v in pit_violations(rows, "2024-10-15")]
    assert len(problems) == 3
    assert any("after end_date" in p for p in problems)
    assert any("missing filing_date" in p for p in problems)
    assert any("before period end" in p for p in problems)


def test_pit_violations_accepts_filing_on_end_date():
    assert pit_violations([{"report_period": "2024-06-30", "filing_date": "2024-08-01T16:05:00"}], "2024-08-01") == []


def test_gap_probe_lands_strictly_inside_the_gap():
    probe = pick_gap_probe([{"report_period": "2024-06-30", "filing_date": "2024-08-01"}])
    assert "2024-06-30" < probe["probe_date"] < "2024-08-01"


def test_gap_probe_skips_rows_without_a_usable_gap():
    rows = [{"report_period": "2024-06-30", "filing_date": None}, {"report_period": "2024-06-30", "filing_date": "2024-07-02"}]
    assert pick_gap_probe(rows) is None


def test_gap_probe_passes_when_period_hidden_and_fails_when_leaked():
    probe = {"report_period": "2024-06-30", "filing_date": "2024-08-01", "probe_date": "2024-07-16"}
    hidden = [{"report_period": "2024-03-31", "filing_date": "2024-05-01"}]
    assert evaluate_gap_probe(probe, hidden).status == "pass"
    leaked = hidden + [{"report_period": "2024-06-30", "filing_date": "2024-08-01"}]
    assert evaluate_gap_probe(probe, leaked).status == "fail"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("verdicts,expected", [
    (["ok", "ok"], "pass"),
    (["ok", "bad"], "fail"),
    (["ok", "unknown"], "inconclusive"),
    ([], "inconclusive"),
])
def test_aggregate(verdicts, expected):
    result = aggregate("x", verdicts, good={"ok"}, bad={"bad"}, details={}, what="t")
    assert result.status == expected
