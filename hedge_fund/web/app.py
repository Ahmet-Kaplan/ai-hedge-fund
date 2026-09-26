"""Read-only local dashboard — the account on a web page, and nothing else.

Deliberately narrow. This surface shows what `hedge_fund.desk` shows and has no
way to submit anything:

- **Never writes.** Only GET routes exist, and the broker handed to the read
  model is wrapped in `ReadOnlyBroker`, which raises on `place_order`. A future
  edit cannot turn a view into a trade by accident.
- **Loopback by default.** `serve` refuses a non-loopback host unless you pass
  `allow_remote=True` explicitly.
- **Token required.** Every route needs one, compared with
  `secrets.compare_digest` so a wrong token cannot be probed byte by byte.
  Without an explicit token or `HEDGE_FUND_WEB_TOKEN` one is generated per
  process and printed, so the default is never "no auth".
- **No files served.** The page is a string in this module, so there is no
  static path to traverse. (The repository has history here: an earlier
  contributed web app shipped an unauthenticated SPA catch-all that resolved
  `../` straight out of the deployment directory.)

FastAPI and uvicorn are optional: this module imports them lazily inside
`create_app`, and no other part of the package depends on them.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

# Imported at module scope so FastAPI can resolve the annotations below.
# `from __future__ import annotations` makes them strings, and FastAPI
# evaluates them in this module's globals — a type imported inside the
# factory would be resolved as a request parameter instead. The import is
# guarded so the package still loads without the optional web dependency.
try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, StreamingResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    WEB_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:  # pragma: no cover - depends on the environment
    WEB_IMPORT_ERROR = _exc

from hedge_fund.desk import DeskSnapshot, desk_snapshot
from hedge_fund.paths import ensure_mandates_dir
from hedge_fund.venue import VENUES, OpenVenue, open_venue

TOKEN_ENV = "HEDGE_FUND_WEB_TOKEN"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_REFRESH_SECONDS = 5.0
# Only addresses that cannot be reached from another machine. 0.0.0.0 is NOT
# one of them: it means every interface, which would publish an unencrypted
# account view to the network.
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


def resolve_token(explicit: str | None = None) -> str:
    """The token to require. Never empty: one is generated when unset."""
    token = (explicit or os.environ.get(TOKEN_ENV) or "").strip()
    return token or secrets.token_urlsafe(24)


def check_host(host: str, *, allow_remote: bool = False) -> None:
    """Refuse to listen anywhere but loopback unless asked in so many words."""
    if allow_remote or host in _LOOPBACK:
        return
    raise ValueError(
        f"refusing to bind {host!r}: the dashboard is unencrypted and meant for "
        f"this machine. Pass allow_remote=True (or --allow-remote) to override."
    )


class ReadOnlyBroker:
    """A broker that can be read and cannot be traded through.

    Wraps any broker and forwards the read methods the desk view needs;
    `place_order` raises. This is the enforcement behind "the dashboard never
    submits orders" — not a comment asking future code to behave.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def venue(self) -> str:
        return getattr(self._inner, "venue", type(self._inner).__name__)

    def positions(self):
        return self._inner.positions()

    def cash(self) -> float:
        return self._inner.cash()

    def account(self):
        probe = getattr(self._inner, "account", None)
        return probe() if callable(probe) else None

    def position_details(self):
        probe = getattr(self._inner, "position_details", None)
        return probe() if callable(probe) else None

    def open_orders(self):
        probe = getattr(self._inner, "open_orders", None)
        return probe() if callable(probe) else {}

    def calendar(self, start: str, end: str):
        probe = getattr(self._inner, "calendar", None)
        return probe(start, end) if callable(probe) else []

    def is_market_open(self) -> bool:
        probe = getattr(self._inner, "is_market_open", None)
        return bool(probe()) if callable(probe) else False

    def place_order(self, order: Any) -> Any:
        raise PermissionError(
            "the dashboard is read-only: it never submits orders. "
            "Use the CLI or the Desk screen's ticket."
        )


def read_only(opened: OpenVenue) -> OpenVenue:
    """The same venue with a broker that cannot place orders."""
    return replace(opened, broker=ReadOnlyBroker(opened.broker))


def load_snapshot(
    venue: str | None = None,
    *,
    open_fn: Callable[..., OpenVenue] = open_venue,
    receipts: Path | None = None,
) -> DeskSnapshot:
    """Open a venue read-only and snapshot it. Blocking; call off the loop."""
    name = venue or _default_venue()
    opened = open_fn(name, fund_name="desk", capital=0.0,
                     receipts=receipts or ensure_mandates_dir())
    return desk_snapshot(read_only(opened))


def _default_venue() -> str:
    """Alpaca when it is configured, else the local paper book."""
    if os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID"):
        return "alpaca"
    return "paper"


def _html(token: str) -> str:
    """The page. A string, so there is nothing on disk to traverse to."""
    return _PAGE.replace("__TOKEN__", json.dumps(token))


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>hedge fund · desk</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 2rem; background: #0b100e; color: #d9e6e0;
         font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  h1 { font-size: 15px; letter-spacing: .3em; color: #f2f7f4; margin: 0 0 1rem; }
  .badge { padding: .15rem .5rem; border-radius: 3px; font-weight: 700; }
  .paper { background: #2bd97c; color: #06251a; }
  .live  { background: #f87171; color: #2a0a0a; }
  .readonly { background: #22302a; color: #5f7268; }
  table { border-collapse: collapse; margin-top: 1rem; width: 100%; }
  th, td { padding: .3rem .8rem .3rem 0; text-align: right; }
  th:first-child, td:first-child { text-align: left; color: #22d3ee; }
  th { color: #5f7268; border-bottom: 1px solid #1f2b25; }
  .muted { color: #5f7268; }
  .pos { color: #2bd97c; } .neg { color: #f87171; }
  .warn { color: #f87171; margin-top: 1rem; }
  .card { border: 1px solid #1f2b25; border-radius: 6px; padding: 1rem 1.2rem; }
</style>
</head>
<body>
<h1>DESK</h1>
<div class="card" id="card" data-token='__TOKEN__'>
  <div id="head" class="muted">connecting…</div>
  <div id="stats"></div>
  <div id="positions"></div>
  <div id="warnings"></div>
</div>
<script>
const token = document.getElementById('card').dataset.token;
const money = v => v === null || v === undefined ? '—'
  : (Math.abs(v) >= 10000 ? '$' + (v/1000).toFixed(0) + 'k' : '$' + v.toFixed(0));
const price = v => v === null || v === undefined ? '—' : '$' + v.toFixed(2);
const signed = v => v === null || v === undefined ? '—'
  : `<span class="${v > 0 ? 'pos' : v < 0 ? 'neg' : 'muted'}">${v >= 0 ? '+' : ''}${v.toFixed(2)}</span>`;

function render(s) {
  const cls = {LIVE: 'live', PAPER: 'paper', 'READ-ONLY': 'readonly'}[s.badge] || 'readonly';
  const badge = `<span class="badge ${cls}">${s.badge}</span>`;
  document.getElementById('head').innerHTML = badge + ' &nbsp; ' + s.venue;

  const rows = [
    ['cash', money(s.cash)],
    ['buying power', s.buying_power === null ? '—' : money(s.buying_power)],
    ['equity', s.equity === null ? '—' : money(s.equity)],
    ['gross exposure', s.gross_exposure === null ? '—' : money(s.gross_exposure)],
    ['unrealized', s.unrealized_pnl === null ? '—' : signed(s.unrealized_pnl)],
    ['working orders', s.working_orders],
  ];
  document.getElementById('stats').innerHTML = '<table>' + rows.map(
    ([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join('') + '</table>';

  const head = `<tr><th>ticker</th><th>side</th><th>shares</th><th>avg</th>
                <th>mark</th><th>value</th><th>unrealized</th></tr>`;
  const body = s.positions.length ? s.positions.map(p => `<tr>
      <td>${p.ticker}</td><td class="muted">${p.shares >= 0 ? 'long' : 'short'}</td>
      <td>${p.shares.toLocaleString()}</td><td>${price(p.avg_entry_price)}</td>
      <td>${price(p.current_price)}</td><td>${money(p.market_value)}</td>
      <td>${signed(p.unrealized_pnl)}</td></tr>`).join('')
    : '<tr><td class="muted" colspan="7">flat — no open positions</td></tr>';
  document.getElementById('positions').innerHTML = '<table>' + head + body + '</table>';

  document.getElementById('warnings').innerHTML =
    (s.warnings || []).map(w => `<div class="warn">! ${w}</div>`).join('');
}

fetch(`/api/desk?token=${encodeURIComponent(token)}`)
  .then(r => r.ok ? r.json() : Promise.reject(r.status))
  .then(render)
  .catch(e => document.getElementById('head').textContent = 'unavailable (' + e + ')');

const stream = new EventSource(`/api/stream?token=${encodeURIComponent(token)}`);
stream.onmessage = e => render(JSON.parse(e.data));
</script>
</body>
</html>
"""


def create_app(
    *,
    venue: str | None = None,
    token: str | None = None,
    refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
    max_stream_events: int | None = None,
    open_fn: Callable[..., OpenVenue] = open_venue,
    snapshot_fn: Callable[[str | None], DeskSnapshot] | None = None,
):
    """Build the dashboard app.

    `open_fn` / `snapshot_fn` are seams for tests; production uses the venue
    registry and the real desk snapshot. `max_stream_events` closes the stream
    after that many frames — a bounded read for tests and scripts, where the
    default stream would legitimately never end.
    """
    if WEB_IMPORT_ERROR is not None:  # pragma: no cover - environment dependent
        raise ImportError(
            "the dashboard needs the optional web dependencies: "
            "pip install fastapi uvicorn"
        ) from WEB_IMPORT_ERROR

    expected = resolve_token(token)
    fetch = snapshot_fn or (lambda name: load_snapshot(name, open_fn=open_fn))
    bearer = HTTPBearer(auto_error=False)

    app = FastAPI(
        title="hedge fund desk (read-only)",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def authorise(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        """Token in the Authorization header, or `token=` for EventSource.

        EventSource cannot set headers, so the stream accepts the query
        parameter; everything else should prefer the header.
        """
        presented = (
            (credentials.credentials if credentials else None)
            or request.query_params.get("token")
            or ""
        )
        if not secrets.compare_digest(presented, expected):
            raise HTTPException(status_code=401, detail="a valid token is required")

    def render(name: str | None) -> dict:
        snapshot = fetch(name or venue)
        return json.loads(snapshot.model_dump_json())

    @app.get("/", dependencies=[Depends(authorise)])
    def index() -> Any:
        return HTMLResponse(_html(expected))

    @app.get("/api/health", dependencies=[Depends(authorise)])
    def health() -> dict:
        return {"ok": True, "venue": venue or _default_venue(), "read_only": True}

    @app.get("/api/desk", dependencies=[Depends(authorise)])
    def desk_endpoint(name: str | None = None) -> dict:
        try:
            return render(name)
        except Exception as exc:  # a view that cannot load is a 503, not a stack trace
            raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.get("/api/stream", dependencies=[Depends(authorise)])
    def stream(request: Request, name: str | None = None) -> Any:
        async def events() -> Any:
            previous: str | None = None
            sent = 0
            while True:
                if await request.is_disconnected():
                    return
                if max_stream_events is not None and sent >= max_stream_events:
                    return
                try:
                    # Blocking network work belongs off the event loop.
                    payload = await asyncio.to_thread(_stream_payload, fetch, name or venue)
                except Exception as exc:
                    payload = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                if payload != previous:
                    previous = payload
                    sent += 1
                    yield f"data: {payload}\n\n"
                else:
                    yield ": unchanged\n\n"
                await asyncio.sleep(refresh_seconds)

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _stream_payload(fetch: Callable[[str | None], DeskSnapshot], name: str | None) -> str:
    return fetch(name).model_dump_json()


def serve(
    *,
    venue: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    token: str | None = None,
    allow_remote: bool = False,
    refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
) -> str:
    """Run the dashboard until interrupted. Returns the URL it served."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "serving the dashboard needs uvicorn: pip install fastapi uvicorn"
        ) from exc

    check_host(host, allow_remote=allow_remote)
    app = create_app(venue=venue, token=token, refresh_seconds=refresh_seconds)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return f"http://{host}:{port}"
