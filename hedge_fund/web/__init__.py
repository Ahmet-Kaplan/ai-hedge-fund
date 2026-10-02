"""The browser dashboard for the hedge fund.

Install with the `web` extra and serve with:

    uvicorn hedge_fund.web.app:app

Authentication is not implemented here. In deployment, Azure Container Apps'
built-in auth signs the user in against Entra ID before the request reaches
this process; `hedge_fund.web.auth` only reads the resulting headers. Set
AIHF_WEB_REQUIRE_AUTH=false to run locally without a tenant.
"""

from __future__ import annotations

__all__ = ["app"]


def __getattr__(name: str):
    # Imported lazily so that `import hedge_fund.web` does not drag FastAPI
    # into a terminal-only install that skipped the `web` extra.
    if name == "app":
        from hedge_fund.web.app import app

        return app
    raise AttributeError(name)
