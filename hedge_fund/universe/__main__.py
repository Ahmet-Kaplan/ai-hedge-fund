"""Build a point-in-time universe schedule for a backtest.

    python -m hedge_fund.universe --start 2016-07-01 --end 2025-07-01 --top-n 15 --out universe.json
    aihf configs/buffett-baseline.yaml --backtest --universe-schedule universe.json --start ... --date ...

Needs TIINGO_API_KEY and SEC_USER_AGENT. Everything is stored locally
(~/.hedge-fund/cache/{edgar,tiingo,universe}); a rerun with the same
settings costs no requests. --max-price-downloads caps new Tiingo downloads
so a build stays inside the free tier's hourly limit; when it is reached the
build stops with everything fetched so far kept — rerun later to continue.
"""

from __future__ import annotations

import argparse
import sys

from hedge_fund.data.edgar import EdgarClient
from hedge_fund.data.tiingo import TiingoClient
from hedge_fund.universe.builder import PriceBudgetExceeded, UniverseBuilder, reconstitution_dates, save_schedule
from hedge_fund.universe.models import UniverseConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hedge_fund.universe", description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--cadence", choices=["annual", "quarterly"], default="annual")
    parser.add_argument("--benchmark", default="SPY", help="trading calendar source")
    parser.add_argument("--max-price-downloads", type=int, default=45,
                        help="new Tiingo downloads allowed this run (free tier: 50 requests/hour)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    config = UniverseConfig(top_n=args.top_n)
    with TiingoClient(max_age_hours=None) as tiingo, EdgarClient(price_source=tiingo) as edgar:
        sessions = [p.time[:10] for p in tiingo.get_prices(args.benchmark, args.start, args.end)]
        dates = reconstitution_dates(sessions, args.start, args.end, args.cadence)
        builder = UniverseBuilder(edgar, tiingo, config, max_price_downloads=args.max_price_downloads)
        try:
            schedule = builder.schedule(dates)
        except PriceBudgetExceeded as exc:
            print(f"stopped: {exc}\n(Tiingo {tiingo.requests} requests, SEC {edgar.requests} requests)", file=sys.stderr)
            return 3
        save_schedule(schedule, args.out)
        for snap in schedule.snapshots:
            print(f"{snap.as_of}: {', '.join(snap.tickers)}  ({snap.candidates} candidates)", file=sys.stderr)
        print(f"wrote {args.out} · Tiingo {tiingo.requests} requests · SEC {edgar.requests} requests", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
