"""Tests for the congressional-trading analysts.

The expensive failure here is lookahead. A congressional strategy keyed on
transaction dates backtests beautifully and could never have been traded,
because the filing did not exist yet. Most of what follows exists to make
that specific mistake impossible to reintroduce quietly.
"""

from __future__ import annotations

import json

import pytest

from hedge_fund.data.congress import (
    CongressClient,
    CongressDataError,
    Disclosure,
    parse_amount,
    parse_disclosure,
)
from hedge_fund.signals import congress, roster


def filing(**overrides) -> Disclosure:
    base = dict(
        member="Jane Doe", chamber="house", ticker="AAPL",
        transaction_date="2026-05-01", disclosure_date="2026-06-01",
        kind="Purchase", amount_low=15_001.0, amount_high=50_000.0,
    )
    return Disclosure(**{**base, **overrides})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Nothing in this module may reach the vendor."""
    congress.reset_cache()
    yield
    congress.reset_cache()


def with_filings(monkeypatch, filings: list[Disclosure]) -> None:
    monkeypatch.setattr(congress, "disclosures_for", lambda chamber, pages=0: filings)


def model(**kwargs) -> congress.CongressModel:
    return congress.CongressModel("Jane Doe", "house", **kwargs)


# ---- the vendor payload is read strictly ------------------------------------

def test_a_row_missing_a_field_raises_rather_than_being_skipped() -> None:
    """A renamed field must not become an analyst who silently abstains.

    Skipping would turn a broken integration into what looks like a quiet
    market, and nobody would investigate a quiet market.
    """
    with pytest.raises(CongressDataError, match="disclosureDate"):
        parse_disclosure({
            "symbol": "AAPL", "transactionDate": "2026-05-01",
            "type": "Purchase", "firstName": "Jane", "lastName": "Doe",
        }, "house")


def test_a_filing_with_no_ticker_is_skipped_not_fatal() -> None:
    """Members disclose LLCs and real estate, not only listed equities.

    A live Senate page had 23 of 25 rows carrying no symbol. Raising on
    those made the whole chamber unreadable, which is how this was found.
    """
    assert parse_disclosure({
        "symbol": "", "transactionDate": "2026-05-01",
        "disclosureDate": "2026-06-01", "type": "Purchase",
        "firstName": "Jane", "lastName": "Doe",
        "assetDescription": "MH Built to Last LLC",
    }, "senate") is None


def test_a_dateless_equity_filing_still_raises() -> None:
    """The symbol says this is tradeable, so a blank disclosure date is the
    lookahead guarantee arriving empty — not a non-equity filing."""
    with pytest.raises(CongressDataError, match="AAPL"):
        parse_disclosure({
            "symbol": "AAPL", "transactionDate": "2026-05-01",
            "disclosureDate": "", "type": "Purchase",
            "firstName": "Jane", "lastName": "Doe",
        }, "house")


def test_a_page_of_untradeable_filings_does_not_stop_paging() -> None:
    """A page of nothing but LLCs is still a full page; paging must not
    mistake it for the end of the data."""
    rows = [{"symbol": "", "transactionDate": "2026-05-01",
             "disclosureDate": "2026-06-01", "type": "Purchase",
             "firstName": "Jane", "lastName": "Doe"}] * CongressClient.PAGE_SIZE

    client = CongressClient(api_key="unused")
    pages: list[int] = []

    def fake_get(path, params):
        pages.append(params["page"])
        return rows if params["page"] == 0 else []

    client._get = fake_get  # type: ignore[method-assign]
    assert client.latest("senate", pages=3) == []
    assert pages == [0, 1]  # it kept going rather than stopping on an empty yield


def test_a_filed_range_is_read_into_its_bounds() -> None:
    assert parse_amount("$15,001 - $50,000") == (15_001.0, 50_000.0)


def test_an_unreadable_amount_does_not_discard_the_filing() -> None:
    """Direction is disclosed reliably; size is not. Keep the useful half."""
    assert parse_amount("") == (0.0, 0.0)
    assert parse_amount(None) == (0.0, 0.0)


@pytest.mark.parametrize("kind,expected", [
    ("Purchase", 1),
    ("Sale (Full)", -1),
    ("Sale (Partial)", -1),
    ("Exchange", 0),
])
def test_transaction_type_maps_to_a_direction(kind: str, expected: int) -> None:
    """An exchange moves an asset without expressing a view."""
    assert filing(kind=kind).direction == expected


# ---- the model cannot see a filing before it was filed ---------------------

def test_a_trade_disclosed_after_the_decision_date_is_invisible(monkeypatch) -> None:
    """The whole point. The member traded in May; nobody knew until June.

    A model keyed on the transaction date would buy on 2026-05-02 and show
    an edge that was never available to anyone.
    """
    with_filings(monkeypatch, [filing(transaction_date="2026-05-01",
                                      disclosure_date="2026-06-01")])

    signal = model().predict("AAPL", "2026-05-02", data_client=None)

    assert signal.metadata["abstained"] is True
    assert signal.value == 0.0


def test_the_same_trade_is_visible_once_it_has_been_disclosed(monkeypatch) -> None:
    with_filings(monkeypatch, [filing(transaction_date="2026-05-01",
                                      disclosure_date="2026-06-01")])

    signal = model().predict("AAPL", "2026-06-01", data_client=None)

    assert signal.metadata.get("abstained") is not True
    assert signal.value > 0


def test_the_view_carries_both_dates_so_the_lag_can_be_checked(monkeypatch) -> None:
    """A reader should be able to verify the dating rather than trust it."""
    with_filings(monkeypatch, [filing()])

    signal = model().predict("AAPL", "2026-06-02", data_client=None)

    assert signal.metadata["latest_transaction"] == "2026-05-01"
    assert signal.metadata["latest_disclosure"] == "2026-06-01"
    assert "traded 2026-05-01" in signal.reasoning
    assert "disclosed 2026-06-01" in signal.reasoning
    assert "31 days later" in signal.reasoning


# ---- views expire ----------------------------------------------------------

def test_a_filing_older_than_the_lookback_carries_no_weight(monkeypatch) -> None:
    """A purchase disclosed last year is not a reason to buy today."""
    with_filings(monkeypatch, [filing(disclosure_date="2025-01-01")])

    signal = model(lookback_days=90).predict("AAPL", "2026-06-01", data_client=None)

    assert signal.metadata["abstained"] is True


def test_conviction_decays_as_a_filing_ages(monkeypatch) -> None:
    with_filings(monkeypatch, [filing(disclosure_date="2026-06-01")])
    fresh = model().predict("AAPL", "2026-06-02", data_client=None).value
    stale = model().predict("AAPL", "2026-08-01", data_client=None).value

    assert 0 < stale < fresh


# ---- direction and abstention ----------------------------------------------

def test_a_disclosed_sale_is_a_bearish_view(monkeypatch) -> None:
    with_filings(monkeypatch, [filing(kind="Sale (Full)")])

    assert model().predict("AAPL", "2026-06-02", data_client=None).value < 0


def test_silence_is_an_abstention_not_a_neutral_view(monkeypatch) -> None:
    """Abstaining keeps the member out of the blend; a zero would dilute it."""
    with_filings(monkeypatch, [filing(ticker="MSFT")])

    signal = model().predict("AAPL", "2026-06-02", data_client=None)

    assert signal.metadata["abstained"] is True


def test_another_members_trade_is_not_attributed_to_this_one(monkeypatch) -> None:
    with_filings(monkeypatch, [filing(member="John Roe")])

    signal = model().predict("AAPL", "2026-06-02", data_client=None)

    assert signal.metadata["abstained"] is True


def test_repeated_purchases_strengthen_the_view(monkeypatch) -> None:
    one = [filing(disclosure_date="2026-06-01")]
    three = one + [filing(disclosure_date="2026-05-28"), filing(disclosure_date="2026-05-20")]

    with_filings(monkeypatch, one)
    single = model().predict("AAPL", "2026-06-02", data_client=None).value
    with_filings(monkeypatch, three)
    repeated = model().predict("AAPL", "2026-06-02", data_client=None).value

    assert repeated > single


def test_conviction_stays_inside_the_allowed_range(monkeypatch) -> None:
    """Twenty purchases must not produce a conviction the blend cannot hold."""
    with_filings(monkeypatch, [filing(disclosure_date="2026-06-01",
                                      amount_high=5_000_000.0)] * 20)

    assert model().predict("AAPL", "2026-06-02", data_client=None).value <= 1.0


# ---- the roster is a file, not a live query --------------------------------

def test_an_absent_roster_registers_no_analysts(tmp_path) -> None:
    assert roster.load_roster(tmp_path / "missing.json") == []


def test_a_malformed_roster_costs_the_analysts_not_the_process(tmp_path) -> None:
    """A bad roster must not break imports for everything else in the fund."""
    path = tmp_path / "roster.json"
    path.write_text("{ not json")

    assert roster.load_roster(path) == []


def test_a_roster_entry_becomes_a_registrable_analyst(tmp_path) -> None:
    path = tmp_path / "roster.json"
    path.write_text(json.dumps([{"member": "Jane Doe", "chamber": "house"}]))

    entries = roster.registry_entries(path)

    assert "congress-jane-doe" in entries
    built = entries["congress-jane-doe"]()
    assert built.member == "Jane Doe"
    assert built.name == "congress-jane-doe"


def test_a_registered_member_declares_its_own_approach(tmp_path) -> None:
    """get_investment_approach reads vars(cls) and will not look up the chain."""
    path = tmp_path / "roster.json"
    path.write_text(json.dumps([{"member": "Jane Doe", "chamber": "senate"}]))

    cls = roster.registry_entries(path)["congress-jane-doe"]

    assert vars(cls)["investment_approach"] in ("long_only", "long_short")


def test_an_entry_with_an_unknown_chamber_is_dropped(tmp_path) -> None:
    path = tmp_path / "roster.json"
    path.write_text(json.dumps([{"member": "Jane Doe", "chamber": "parliament"}]))

    assert roster.load_roster(path) == []
