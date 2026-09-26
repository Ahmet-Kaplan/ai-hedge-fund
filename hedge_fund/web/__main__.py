"""Run the read-only dashboard.

Usage::

    python -m hedge_fund.web
        Serve the desk view on http://127.0.0.1:8765. The token is taken from
        HEDGE_FUND_WEB_TOKEN or generated and printed on startup.

    python -m hedge_fund.web --venue alpaca --port 9000
        Watch a specific venue. Alpaca reads work with no trading opt-in; the
        dashboard cannot place orders whatever the settings say.

The page is unencrypted and holds an account token, so the bind address must
be loopback unless --allow-remote is passed explicitly.
"""

from __future__ import annotations

import argparse
import sys

from hedge_fund.venue import VENUES
from hedge_fund.web.app import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_REFRESH_SECONDS,
    check_host,
    resolve_token,
    serve,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hedge_fund.web",
        description="Read-only desk dashboard on loopback. Never submits orders.",
    )
    parser.add_argument("--venue", choices=VENUES,
                        help="which account to watch (default: alpaca when its "
                             "keys are set, else paper)")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"bind address (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port (default: {DEFAULT_PORT})")
    parser.add_argument("--refresh", type=float, default=DEFAULT_REFRESH_SECONDS,
                        help="seconds between stream updates")
    parser.add_argument("--allow-remote", action="store_true",
                        help="permit a non-loopback bind (the page is not encrypted)")
    args = parser.parse_args()

    try:
        check_host(args.host, allow_remote=args.allow_remote)
    except ValueError as exc:
        parser.error(str(exc))

    token = resolve_token()
    if token != (__import__("os").environ.get("HEDGE_FUND_WEB_TOKEN") or "").strip():
        sys.stderr.write(f"dashboard token: {token}\n")
        sys.stderr.write("set HEDGE_FUND_WEB_TOKEN to pin it between runs\n")
    sys.stderr.write(
        f"read-only desk on http://{args.host}:{args.port}  ·  Ctrl-C to stop\n"
    )
    serve(venue=args.venue, host=args.host, port=args.port, token=token,
          allow_remote=args.allow_remote, refresh_seconds=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
