import pytest

from hedge_fund.live.account.target import latest_paper_weights, satellite_weights, target_book
from hedge_fund.live.ledger import Ledger

PAPER = {"NVDA": 0.04, "MSFT": 0.04, "WMT": -0.02}   # gross 0.10


def test_no_share_is_all_core():
    assert target_book(satellite_weights(PAPER, 0.0, shorts_ok=True, max_name=0.1, core_ticker="SPY"), "SPY") == {"SPY": 1.0}


def test_long_only_stage_drops_shorts_and_scales_to_the_share():
    sat = satellite_weights(PAPER, 0.2, shorts_ok=False, max_name=0.5, core_ticker="SPY")
    assert sat == {"NVDA": pytest.approx(0.1), "MSFT": pytest.approx(0.1)}
    assert target_book(sat, "SPY") == {"SPY": pytest.approx(0.8), "NVDA": pytest.approx(0.1), "MSFT": pytest.approx(0.1)}


def test_shorts_stage_keeps_signs_and_gross_equals_share():
    sat = satellite_weights(PAPER, 0.2, shorts_ok=True, max_name=0.5, core_ticker="SPY")
    assert sat == {"NVDA": pytest.approx(0.08), "MSFT": pytest.approx(0.08), "WMT": pytest.approx(-0.04)}
    assert target_book(sat, "SPY")["SPY"] == pytest.approx(0.8)


def test_name_cap_overflow_goes_to_core():
    sat = satellite_weights({"NVDA": 0.5}, 0.3, shorts_ok=False, max_name=0.1, core_ticker="SPY")
    assert sat == {"NVDA": 0.1}
    assert target_book(sat, "SPY")["SPY"] == pytest.approx(0.9)


def test_core_ticker_in_paper_weights_is_not_satellite():
    assert satellite_weights({"SPY": 0.5}, 0.2, shorts_ok=False, max_name=0.1, core_ticker="SPY") == {}


def test_latest_paper_weights_freshness(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.write_plan("2026-09-28", {"plan": {"decision": {"final_weights": {"NVDA": 0.04, "X": 0.0}}}})
    assert latest_paper_weights(ledger, "2026-10-01", 8) == ({"NVDA": 0.04}, "2026-09-28", True)
    assert latest_paper_weights(ledger, "2026-10-08", 8)[2] is False
    ledger.write_plan("2026-09-29", {"plan": {"decision": None}})          # a flatten plan: no bets
    assert latest_paper_weights(ledger, "2026-10-01", 8) == ({}, "2026-09-29", True)
    assert latest_paper_weights(Ledger(tmp_path / "none"), "2026-10-01", 8) == ({}, None, False)


from hedge_fund.live.account.settings import LiveSettings  # noqa: E402
from hedge_fund.live.account.target import account_book  # noqa: E402


def test_account_book_splits_halves():
    book = account_book({"SPY": 0.8, "NVDA": 0.2}, LiveSettings(crypto_share=0.5))
    assert book == {"SPY": pytest.approx(0.4), "NVDA": pytest.approx(0.1), "BTC/USD": pytest.approx(0.3),
                    "ETH/USD": pytest.approx(0.15), "SOL/USD": pytest.approx(0.05)}
    assert sum(book.values()) == pytest.approx(1.0)


def test_no_crypto_share_leaves_the_stock_book_alone():
    assert account_book({"SPY": 1.0}, LiveSettings()) == {"SPY": 1.0}
