"""Live Tiingo + SEC EDGAR checks. Opt-in: LIVE_DATA_TESTS=1 with
TIINGO_API_KEY and SEC_USER_AGENT set.

They use the normal user-level stores (~/.hedge-fund/cache/{tiingo,edgar}),
so each ticker's history is downloaded at most once across runs — about 20
Tiingo requests and ~30 SEC requests the first time, ~0 after.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hedge_fund.data.edgar import EdgarClient
from hedge_fund.data.factory import make_data_client
from hedge_fund.data.tiingo import TiingoClient
from hedge_fund.verification.checks import classify_split_adjustment
from hedge_fund.verification.runner import DELISTED_CASES, SPLIT_CASES

pytestmark = pytest.mark.skipif(
    not (os.environ.get("LIVE_DATA_TESTS") and os.environ.get("TIINGO_API_KEY") and os.environ.get("SEC_USER_AGENT")),
    reason="live provider tests: set LIVE_DATA_TESTS=1, TIINGO_API_KEY and SEC_USER_AGENT",
)

EDGAR_FIXTURES = Path(__file__).parent / "edgar" / "testdata"


@pytest.fixture(scope="module")
def client():
    with make_data_client() as c:
        yield c
        print(f"\nrequests: {c.request_counts()}")


@pytest.mark.parametrize("ticker, split_date, ratio", SPLIT_CASES)
def test_known_splits(client, ticker, split_date, ratio):
    events = client.split_events(ticker, "2019-01-01", "2025-12-31")
    assert events.get(split_date) == pytest.approx(ratio)
    window = {p.time[:10]: p.close for p in client.get_prices(ticker, "2019-01-01", "2025-12-31")}
    basis, info = classify_split_adjustment(window, split_date, ratio)
    assert basis == "adjusted", info
    raw = client.raw_closes(ticker, "2019-01-01", "2025-12-31")
    raw_basis, raw_info = classify_split_adjustment(raw, split_date, ratio)
    assert raw_basis == "unadjusted", raw_info


@pytest.mark.parametrize("ticker, years", [("SPY", 20), ("AAPL", 10), ("KO", 10), ("JPM", 10)])
def test_long_history(client, ticker, years):
    bars = client.get_prices(ticker, "1990-01-01", "2026-08-31")
    first, last = bars[0].time[:10], bars[-1].time[:10]
    assert int(last[:4]) - int(first[:4]) >= years
    assert len(bars) >= 250 * years


@pytest.mark.parametrize("ticker, last_day", DELISTED_CASES)
def test_delisted_prices_through_last_trading_day(client, ticker, last_day):
    bars = client.get_prices(ticker, "2022-01-01", last_day)
    assert bars and (bars[-1].time[:10] >= last_day[:8] + "01")


def test_successor_symbols(client):
    assert client.get_prices("SIVBQ", "2023-03-01", "2024-12-31")
    assert client.get_prices("FRC", "2023-04-01", "2023-04-28")  # served from FRCB


@pytest.mark.parametrize("ticker, as_of, cap", [
    ("AAPL", "2020-10-30", 1.85e12),   # 10-K filed 2020-10-30
    ("KO", "2020-02-24", 2.516e11),    # 10-K filed 2020-02-24
    ("BRK.B", "2016-08-05", 3.59e11),  # 10-Q filed 2016-08-05 (per-class cover page)
])
def test_edgar_market_cap_from_tiingo_raw_close(tmp_path, ticker, as_of, cap):
    """Committed SEC fixtures (no SEC requests) valued with live Tiingo closes."""
    import shutil
    shutil.copytree(EDGAR_FIXTURES, tmp_path / "edgar")
    with TiingoClient() as tiingo:
        edgar = EdgarClient(cache_dir=tmp_path / "edgar", offline=True, price_source=tiingo)
        row = edgar.get_financial_metrics(ticker, as_of, limit=1)[0]
        raw_close = tiingo.raw_closes(ticker, as_of, as_of)[as_of]
    assert row.filing_date == as_of
    assert row.market_cap == pytest.approx(cap, rel=0.02)
    assert row.price_to_earnings_ratio * row.earnings_per_share == pytest.approx(raw_close)


@pytest.mark.parametrize("ticker", ["KO", "JPM", "AAPL", "MSFT"])
def test_point_in_time_live(client, ticker):
    for as_of in ("2016-06-30", "2020-02-28", "2024-12-31"):
        rows = client.get_financial_metrics(ticker, as_of, limit=12)
        assert len(rows) >= 4 and all(r.filing_date <= as_of for r in rows)
