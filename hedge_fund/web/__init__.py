"""Read-only local dashboard.

`python -m hedge_fund.web` serves the desk view on loopback. Nothing here can
submit an order; see `hedge_fund/web/app.py`.

FastAPI and uvicorn are optional and imported lazily, so importing this package
costs nothing when the dashboard is not used.
"""

from hedge_fund.web.app import (
    TOKEN_ENV,
    ReadOnlyBroker,
    check_host,
    create_app,
    load_snapshot,
    read_only,
    resolve_token,
    serve,
)

__all__ = [
    "TOKEN_ENV",
    "ReadOnlyBroker",
    "check_host",
    "create_app",
    "load_snapshot",
    "read_only",
    "resolve_token",
    "serve",
]
