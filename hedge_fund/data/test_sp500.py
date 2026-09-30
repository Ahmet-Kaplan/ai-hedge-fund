"""S&P 500 membership from Wikipedia's current list + change log (fixture HTML, no network)."""

from datetime import date

from hedge_fund.data.sp500 import members_as_of, parse_changes, parse_current

CURRENT = """<table class="wikitable sortable" id="constituents"><tbody>
<tr><th>Symbol</th><th>Security</th></tr>
<tr><td><a href="#">AAPL</a></td><td>Apple</td></tr>
<tr><td>BRK.B</td><td>Berkshire</td></tr>
<tr><td>NEW</td><td>Newco</td></tr>
</tbody></table>"""

CHANGES = """<table class="wikitable sortable" id="changes"><tbody>
<tr><th>Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th>Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
<tr><td>March 3, 2025</td><td>NEW</td><td>Newco</td><td>OLD</td><td>Oldco</td><td>Acquired.</td></tr>
<tr><td>June 1, 2020</td><td>OLD</td><td>Oldco</td><td></td><td></td><td>Spin-off.</td></tr>
<tr><td>Bad date</td><td>X</td><td>X</td><td>Y</td><td>Y</td><td>junk</td></tr>
</tbody></table>"""


def test_parse_current_and_changes():
    assert parse_current(CURRENT) == ["AAPL", "BRK.B", "NEW"]
    assert parse_changes(CHANGES) == [(date(2025, 3, 3), "NEW", "OLD"), (date(2020, 6, 1), "OLD", None)]


def test_members_replay_the_log_backwards():
    current, changes = parse_current(CURRENT), parse_changes(CHANGES)
    assert members_as_of("2026-01-01", current, changes) == {"AAPL", "BRK.B", "NEW"}
    assert members_as_of("2025-03-02", current, changes) == {"AAPL", "BRK.B", "OLD"}   # before the swap
    assert members_as_of("2025-03-03", current, changes) == {"AAPL", "BRK.B", "NEW"}   # effective that day
    assert members_as_of("2019-01-01", current, changes) == {"AAPL", "BRK.B"}          # before OLD was added
