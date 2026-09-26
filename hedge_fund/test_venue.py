"""Venue registry tests — which book gets opened, and what we say about it.

Order submission itself is covered by the Alpaca broker tests; these pin the
*selection* and the plain-language status a caller shows the operator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hedge_fund.brokers.paper import PaperBroker
from hedge_fund.brokers.sim import SimBroker
from hedge_fund.venue import open_venue, VENUES

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ledger"


def _clear_alpaca_env(monkeypatch):
    for var in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_API_SECRET",
                "APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_PAPER",
                "ALPACA_TRADING_ENABLED", "ALPACA_LIVE_TRADING_CONFIRMED"):
        monkeypatch.delenv(var, raising=False)


def _fake_alpaca(monkeypatch):
    """Replace the SDK client so no network call is ever made."""
    import hedge_fund.venue as venue_module
    monkeypatch.setattr(venue_module, "AlpacaBroker", lambda settings: SimBroker(cash=1.0))


# ---------------------------------------------------------------------------
# Offline venues
# ---------------------------------------------------------------------------

def test_the_three_books_are_registered():
    assert VENUES == ("paper", "sim", "alpaca")


def test_unknown_venue_lists_the_alternatives(tmp_path):
    with pytest.raises(ValueError, match="unknown venue 'ibkr'"):
        open_venue("ibkr", fund_name="d", capital=1.0, receipts=tmp_path)


def test_paper_opens_a_paper_book_at_capital(tmp_path):
    opened = open_venue("paper", fund_name="desk", capital=50_000.0, receipts=tmp_path)

    assert isinstance(opened.broker, PaperBroker)
    assert opened.label == "paper"
    assert opened.live is False
    assert opened.reference is None
    assert opened.broker.cash() == pytest.approx(50_000.0)
    assert "no live venue" in opened.note


def test_sim_opens_a_sim_book(tmp_path):
    opened = open_venue("sim", fund_name="desk", capital=75_000.0, receipts=tmp_path)

    assert isinstance(opened.broker, SimBroker)
    assert opened.label == "sim"
    assert opened.broker.cash() == pytest.approx(75_000.0)


def _place_receipt(directory: Path) -> None:
    """Put the canned receipt where the ledger globs for it."""
    import shutil
    shutil.copy(FIXTURES / "valid_cycle.json",
                directory / "alpha-one-run-2024-06-03-120000.json")


def test_an_existing_receipt_becomes_the_reconciliation_reference(tmp_path):
    """The ledger's claim about the account travels with the opened venue."""
    _place_receipt(tmp_path)

    opened = open_venue("paper", fund_name="alpha-one", capital=1.0,
                        receipts=tmp_path)

    assert opened.reference is not None
    assert opened.reference.source.startswith("alpha-one")
    # And the paper book is seeded from it, as before.
    assert opened.broker.positions()


# ---------------------------------------------------------------------------
# Alpaca: selection and the gates it reports
# ---------------------------------------------------------------------------

def test_alpaca_without_keys_fails_with_the_variable_to_set(tmp_path, monkeypatch):
    _clear_alpaca_env(monkeypatch)

    with pytest.raises(ValueError, match="ALPACA_API_KEY"):
        open_venue("alpaca", fund_name="d", capital=1.0, receipts=tmp_path)


def test_alpaca_is_read_only_until_trading_is_enabled(tmp_path, monkeypatch):
    _clear_alpaca_env(monkeypatch)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")   # the alias real setups use
    _fake_alpaca(monkeypatch)

    opened = open_venue("alpaca", fund_name="d", capital=1.0, receipts=tmp_path)

    assert opened.label == "alpaca-paper"
    assert opened.live is False
    assert opened.trading_enabled is False
    assert "READ-ONLY" in opened.note
    assert "ALPACA_TRADING_ENABLED" in opened.note


def test_alpaca_reports_a_paper_venue_that_will_trade(tmp_path, monkeypatch):
    _clear_alpaca_env(monkeypatch)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_TRADING_ENABLED", "1")
    _fake_alpaca(monkeypatch)

    opened = open_venue("alpaca", fund_name="d", capital=1.0, receipts=tmp_path)

    assert opened.trading_enabled is True
    assert opened.live is False
    assert "paper account" in opened.note
    assert "READ-ONLY" not in opened.note


def test_a_live_account_says_so_in_plain_words(tmp_path, monkeypatch):
    """The note is the last thing between an operator and real money."""
    _clear_alpaca_env(monkeypatch)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_PAPER", "false")
    monkeypatch.setenv("ALPACA_TRADING_ENABLED", "1")
    monkeypatch.setenv("ALPACA_LIVE_TRADING_CONFIRMED", "1")
    _fake_alpaca(monkeypatch)

    opened = open_venue("alpaca", fund_name="d", capital=1.0, receipts=tmp_path)

    assert opened.live is True
    assert opened.trading_enabled is True
    assert opened.label == "alpaca-live"
    assert "LIVE" in opened.note and "for real" in opened.note


def test_a_live_account_without_the_second_confirmation_stays_read_only(tmp_path, monkeypatch):
    _clear_alpaca_env(monkeypatch)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("ALPACA_PAPER", "false")
    monkeypatch.setenv("ALPACA_TRADING_ENABLED", "1")
    _fake_alpaca(monkeypatch)

    opened = open_venue("alpaca", fund_name="d", capital=1.0, receipts=tmp_path)

    assert opened.live is True
    assert opened.trading_enabled is False
    assert "READ-ONLY" in opened.note
