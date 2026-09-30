"""Ticker history resolution, share classes, and cover-page parsing."""

import pytest

from hedge_fund.data.edgar.cover import parse_cover_shares
from hedge_fund.data.edgar.identity import (
    Identity,
    TickerResolver,
    load_history,
    normalize_ticker,
    shares_in_ticker_units,
)


def test_normalize_share_class_separators():
    assert normalize_ticker(" brk-b ") == normalize_ticker("BRK/B") == "BRK.B"


def test_bundled_history_is_well_formed():
    history = load_history()
    for rows in history.values():
        for r in rows:
            assert r.start_date is None or r.end_date is None or r.start_date <= r.end_date
            assert (r.predecessor_cik is None) == (r.predecessor_until is None)
    assert history["FRC"][0].cik is None  # FDIC filer: known, no SEC fundamentals


@pytest.mark.parametrize("ticker, as_of, cik", [
    ("TWTR", "2020-01-01", 1418091),
    ("TWTR", "2024-01-01", 1418091),  # after delisting: still its own filings
    ("ATVI", "2023-10-31", 718877),
    ("SIVB", "2022-06-30", 719739),
    ("SIVBQ", "2024-01-01", 719739),
    ("GOOGL", "2020-01-01", 1652044),
    ("BRK-B", "2016-08-31", 1067983),
])
def test_history_resolution(ticker, as_of, cik):
    assert TickerResolver(current={}).resolve(ticker, as_of).cik == cik


def test_alphabet_lineage_includes_google_inc_until_reorganization():
    ident = TickerResolver(current={}).resolve("GOOGL", "2016-01-01")
    assert ident.lineage() == [(1652044, None), (1288776, "2015-10-01")]


def test_recycled_ticker_resolves_by_date():
    history = {"XYZ": [
        Identity("XYZ", 111, start_date="2000-01-01", end_date="2010-06-30"),
        Identity("XYZ", 222, start_date="2015-01-01"),
    ]}
    r = TickerResolver(current={"XYZ": 222}, history=history)
    assert r.resolve("XYZ", "2005-01-01").cik == 111
    assert r.resolve("XYZ", "2012-01-01").cik == 111   # between listings: last known holder
    assert r.resolve("XYZ", "2020-01-01").cik == 222
    assert r.resolve("XYZ", "1990-01-01").cik == 111   # before any start: earliest


def test_current_map_is_the_fallback_and_history_wins():
    r = TickerResolver(current={"KO": 21344, "TWTR": 999}, history=load_history())
    assert r.resolve("KO", "2020-01-01").cik == 21344
    assert r.resolve("KO").source == "sec_current"
    assert r.resolve("TWTR", "2020-01-01").cik == 1418091
    assert r.resolve("NOPE", "2020-01-01") is None


def test_delisted_identity_is_inactive_after_end():
    ident = TickerResolver(current={}).resolve("ATVI", "2023-01-01")
    assert ident.is_active("2023-10-12") and not ident.is_active("2023-10-13")


def test_berkshire_class_b_units():
    by_class = {"A": 788_894.0, "B": 1_282_442_561.0}
    assert shares_in_ticker_units(1067983, "BRK.B", by_class, "2016-08-05") == 788_894 * 1500 + 1_282_442_561
    assert shares_in_ticker_units(1067983, "BRK.A", by_class, "2016-08-05") == pytest.approx(
        788_894 + 1_282_442_561 / 1500)
    # before the 2010-01-21 50:1 Class B split one A was 30 B
    assert shares_in_ticker_units(1067983, "BRK.B", {"A": 1.0, "B": 3.0}, "2009-11-06") == 33.0


def test_berkshire_refuses_unlabelled_or_unknown_classes():
    assert shares_in_ticker_units(1067983, "BRK.B", {None: 941_481.0}, "2011-05-06") is None
    assert shares_in_ticker_units(1067983, "BRK.B", {"C": 5.0}, "2016-01-01") is None
    assert shares_in_ticker_units(1067983, "OTHER", {"A": 1.0, "B": 1.0}, "2016-01-01") is None


def test_equal_classes_are_summed():
    assert shares_in_ticker_units(1652044, "GOOGL", {"A": 3.0, "B": 1.0, "C": 3.0}, "2020-01-01") == 7.0


# Cover pages -----------------------------------------------------------------

_R1_2016 = """<table><tr><td>Trading Symbol</td><td>BRKA</td></tr>
<tr><td>Entity Filer Category</td><td>Large Accelerated Filer</td></tr>
<tr><td><strong>Class A [Member]</strong></td><td>&#160;</td></tr>
<tr><td>Document Information [Line Items]</td><td>&#160;</td></tr>
<tr><td>Entity Common Stock, Shares Outstanding</td><td>&#160;</td><td>788,894</td></tr>
<tr><td><strong>Class B [Member]</strong></td><td>&#160;</td></tr>
<tr><td>Document Information [Line Items]</td><td>&#160;</td></tr>
<tr><td>Entity Common Stock, Shares Outstanding</td><td>&#160;</td><td>1,282,442,561</td></tr></table>"""

_R1_2026 = """<tr><td>Document Transition Report</td><td>false</td></tr>
<tr><td>Common Class A [Member]</td></tr><tr><td>Document Information [Line Items]</td></tr>
<tr><td>Entity Common Stock, Shares Outstanding</td><td>&#160;</td><td>488,450</td></tr>
<tr><td>Security 12b Title</td><td>Class A Common Stock</td></tr><tr><td>Trading Symbol</td><td>BRK.A</td></tr>
<tr><td>Common Class B [Member]</td></tr><tr><td>Document Information [Line Items]</td></tr>
<tr><td>Entity Common Stock, Shares Outstanding</td><td>&#160;</td><td>1,408,035,161</td></tr>"""


def test_parse_cover_two_layouts():
    assert parse_cover_shares(_R1_2016) == {"A": 788_894.0, "B": 1_282_442_561.0}
    # the Class B block's label is preceded by "Class A Common Stock" text
    assert parse_cover_shares(_R1_2026) == {"A": 488_450.0, "B": 1_408_035_161.0}


def test_parse_cover_entity_wide_and_empty():
    doc = "<td>Entity Common Stock, Shares Outstanding</td><td>&#160;</td><td>687,725,164</td>"
    assert parse_cover_shares(doc) == {None: 687_725_164.0}
    assert parse_cover_shares("<html>no cover here</html>") == {}
