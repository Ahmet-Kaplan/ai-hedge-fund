"""Re-download the EDGAR test fixtures (live SEC requests; public-domain data).

    SEC_USER_AGENT="Name you@example.com" python hedge_fund/data/edgar/testdata/refresh_fixtures.py

Writes trimmed documents in the EdgarClient cache layout next to this file:
companyfacts + submissions for the CIKs below, company_tickers restricted to
TICKERS, and the cover pages the tests rely on (~20 requests). Values in the
tests are pinned to published figures, so a refresh that changes history
(e.g. a restatement filed since) will show up as a test failure to review.
"""

from __future__ import annotations

from pathlib import Path

from hedge_fund.data.edgar import EdgarClient

HERE = Path(__file__).resolve().parent
CIKS = [21344, 19617, 320193, 1652044, 1288776, 1067983, 1418091, 718877, 719739]
PROFILES = [21344, 19617, 320193, 1652044, 1067983, 1418091, 718877, 719739]
TICKERS = {"KO", "JPM", "AAPL", "GOOGL", "GOOG", "BRK-B", "BRK-A", "MSFT"}
COVER_PAGES = [
    (1067983, "0001193125-16-673492"),  # Berkshire 10-Q, 2016-06-30
    (1067983, "0001193125-26-341032"),  # Berkshire 10-Q, 2026-06-30
    (1288776, "0001288776-15-000035"),  # Google Inc. 10-Q, 2015-06-30
    (1652044, "0001652044-15-000005"),  # Alphabet 10-Q, 2015-09-30
]


def main() -> None:
    with EdgarClient(cache_dir=HERE, max_age_hours=0) as sec:
        for cik in CIKS:
            sec._companyfacts(cik)
        for cik in PROFILES:
            sec._submissions(cik)
        for cik, accn in COVER_PAGES:
            (HERE / f"cover/{cik}/{accn}.json.gz").unlink(missing_ok=True)
            sec._cover_page_shares(cik, accn)
        current = sec._company_tickers()
        sec._write(HERE / "company_tickers.json.gz",
                   {"fetched_at": "fixture", "data": {t: c for t, c in current.items() if t in TICKERS}})
        print(f"{sec.requests} SEC requests")


if __name__ == "__main__":
    main()
