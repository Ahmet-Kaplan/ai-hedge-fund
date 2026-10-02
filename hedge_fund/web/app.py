"""The browser dashboard: run simulations, read deployed funds.

Writes are confined to research/. Paper funds are rendered read-only and no
route here appends to a ledger — that stays with the scheduled job, because
the chain has exactly one safe writer and a web process is the wrong one to
make it two.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date as _date, timedelta
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from hedge_fund.backtesting.fund import FundBacktestResult
from hedge_fund.data.sessions import completed_through
from hedge_fund.fund import load_spec, normalize_universe
from hedge_fund.paper.deployed import list_deployed, load_deployed
from hedge_fund.paper.ledger import Ledger, LedgerError
from hedge_fund.paths import ensure_mandates_dir, MANDATES_DIR, PAPER_DIR
from hedge_fund.signals import ALPHA_MODEL_REGISTRY
from hedge_fund.web.auth import NotAuthenticated, principal_from_headers
from hedge_fund.web.charts import nav_chart
from hedge_fund.web.runs import Runs

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DEFAULT_WINDOW_WEEKS = 26

app = FastAPI(title="AI Hedge Fund", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(HERE / "templates"))
runs = Runs()


# ---------------------------------------------------------------------------
# cross-cutting
# ---------------------------------------------------------------------------

@app.exception_handler(NotAuthenticated)
async def _unauthenticated(request: Request, exc: NotAuthenticated) -> HTMLResponse:
    return HTMLResponse("<h1>Sign-in required</h1>", status_code=403)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> HTMLResponse:
    # Generic text to the browser, detail to the logs under a correlation id.
    correlation = uuid.uuid4().hex[:12]
    log.exception("unhandled error [%s] on %s", correlation, request.url.path)
    return HTMLResponse(
        f"<h1>Something went wrong</h1><p>Reference: <code>{correlation}</code></p>",
        status_code=500,
    )


@app.get("/healthz")
async def healthz() -> JSONResponse:
    """Liveness probe. Left unauthenticated on purpose so the platform can
    reach it; it reveals nothing but that the process is up."""
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    principal = principal_from_headers(request.headers)
    return templates.TemplateResponse(request, "dashboard.html", {
        "principal": principal,
        "funds": _paper_funds(),
        "runs": runs.recent(),
        "busy": runs.active(),
    })


@app.get("/simulate", response_class=HTMLResponse)
async def simulate_form(request: Request) -> HTMLResponse:
    principal = principal_from_headers(request.headers)
    end = completed_through()
    start = (_date.fromisoformat(end) - timedelta(weeks=DEFAULT_WINDOW_WEEKS)).isoformat()
    return templates.TemplateResponse(request, "simulate.html", {
        "principal": principal,
        "mandates": _mandates(),
        "analysts": sorted(ALPHA_MODEL_REGISTRY),
        "start": start,
        "end": end,
        "busy": runs.active(),
        "error": request.query_params.get("error"),
    })


@app.post("/simulate")
async def simulate_start(
    request: Request,
    mandate: str = Form(...),
    universe: str = Form(...),
    start: str = Form(...),
    end: str = Form(...),
) -> RedirectResponse:
    principal = principal_from_headers(request.headers)

    # The mandate arrives as a bare filename and is resolved strictly inside
    # the mandates directory: anything that escapes it is a traversal attempt,
    # not a typo.
    path = (MANDATES_DIR / mandate).resolve()
    if path.parent != MANDATES_DIR.resolve() or not path.is_file():
        return RedirectResponse("/simulate?error=unknown-mandate", status_code=303)

    try:
        spec = load_spec(path)
        tickers = normalize_universe(universe.replace(",", " ").split())
        _validate_window(start, end)
    except ValueError:
        return RedirectResponse("/simulate?error=invalid-input", status_code=303)

    run = runs.submit(spec, tickers, start, end, requested_by=principal.display)
    return RedirectResponse(f"/runs/{run.id}", status_code=303)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str) -> HTMLResponse:
    principal = principal_from_headers(request.headers)
    run = runs.get(run_id)
    if run is None:
        return HTMLResponse("<h1>No such run</h1>", status_code=404)

    result = chart = None
    path = runs.result_path(run)
    if path is not None:
        result = FundBacktestResult.model_validate_json(path.read_text())
        chart = nav_chart(result.dates, result.nav, result.benchmark_nav, result.benchmark)

    return templates.TemplateResponse(request, "run.html", {
        "principal": principal, "run": run, "result": result, "chart": chart,
    })


@app.get("/api/runs/{run_id}")
async def run_status(request: Request, run_id: str) -> JSONResponse:
    """Polled by the run page for progress. Cheap and side-effect free."""
    principal_from_headers(request.headers)
    run = runs.get(run_id)
    if run is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(run.as_dict())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _validate_window(start: str, end: str) -> None:
    first, last = _date.fromisoformat(start), _date.fromisoformat(end)
    if first >= last:
        raise ValueError("start must precede end")


def _mandates() -> list[str]:
    ensure_mandates_dir()
    return sorted(p.name for p in MANDATES_DIR.glob("*.yaml"))


def _paper_funds() -> list[dict]:
    """A read-only snapshot of every deployed fund.

    One unreadable fund must not blank the dashboard, so a broken chain is
    reported in its own row rather than raised.
    """
    funds = []
    for directory in list_deployed(PAPER_DIR):
        try:
            deployed = load_deployed(directory)
            ledger = Ledger(directory)
            latest = ledger.latest()
            funds.append({
                "name": deployed.name,
                "mandate": deployed.spec.name,
                "universe": deployed.universe,
                "rebalance": deployed.spec.rebalance,
                "benchmark": deployed.spec.benchmark,
                "capital": deployed.spec.capital,
                "nav": latest.nav if latest else deployed.spec.capital,
                "last_session": latest.session if latest else None,
                "sessions": len(ledger.sessions()),
                "halted": ledger.halted(),
                "error": None,
            })
        except (ValueError, LedgerError) as exc:
            log.warning("paper fund at %s is unreadable: %s", directory, exc)
            funds.append({"name": directory.name, "error": "unreadable", "universe": [], "halted": None})
    return funds
