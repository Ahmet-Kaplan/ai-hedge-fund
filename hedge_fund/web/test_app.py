"""Tests for the dashboard, concentrated on the ways it could leak or corrupt.

The interesting cases here are not the happy paths — they are: does an
unauthenticated request get in, can a crafted mandate name escape the
mandates directory, and does any route touch a paper fund's ledger.
"""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from hedge_fund.web import auth
from hedge_fund.web.app import app, runs
from hedge_fund.web.charts import nav_chart


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def signed_in(monkeypatch) -> dict[str, str]:
    """Headers as Container Apps' built-in auth would inject them."""
    monkeypatch.setattr(auth, "REQUIRE_AUTH", True)
    return {
        auth.NAME_HEADER: "philip.chung@utoronto.ca",
        auth.ID_HEADER: "00000000-1111-2222-3333-444444444444",
    }


# ---- authentication --------------------------------------------------------

def test_healthz_needs_no_principal(client: TestClient, monkeypatch) -> None:
    """The probe must answer before anyone has signed in."""
    monkeypatch.setattr(auth, "REQUIRE_AUTH", True)
    assert client.get("/healthz").status_code == 200


@pytest.mark.parametrize("path", ["/", "/simulate", "/analysts", "/analysts/graham", "/api/runs/anything"])
def test_pages_refuse_an_unauthenticated_request(client: TestClient, monkeypatch, path: str) -> None:
    monkeypatch.setattr(auth, "REQUIRE_AUTH", True)
    assert client.get(path).status_code == 403


def test_pages_render_for_a_signed_in_user(client: TestClient, signed_in: dict) -> None:
    response = client.get("/", headers=signed_in)
    assert response.status_code == 200
    assert "philip.chung@utoronto.ca" in response.text


def test_principal_falls_back_to_the_claims_blob(monkeypatch) -> None:
    """The convenience headers are absent on some configurations."""
    monkeypatch.setattr(auth, "REQUIRE_AUTH", True)
    blob = base64.b64encode(json.dumps({
        "claims": [
            {"typ": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/upn", "val": "someone@utoronto.ca"},
            {"typ": "http://schemas.microsoft.com/identity/claims/objectidentifier", "val": "abc"},
        ]
    }).encode()).decode()

    principal = auth.principal_from_headers({auth.PRINCIPAL_HEADER: blob})

    assert principal.name == "someone@utoronto.ca"
    assert principal.object_id == "abc"


def test_a_corrupt_claims_blob_is_not_a_way_in(monkeypatch) -> None:
    monkeypatch.setattr(auth, "REQUIRE_AUTH", True)
    with pytest.raises(auth.NotAuthenticated):
        auth.principal_from_headers({auth.PRINCIPAL_HEADER: "not-base64-at-all"})


def test_local_development_may_opt_out(monkeypatch) -> None:
    monkeypatch.setattr(auth, "REQUIRE_AUTH", False)
    assert auth.principal_from_headers({}) is auth.ANONYMOUS


# ---- input handling --------------------------------------------------------

@pytest.fixture
def no_real_backtests(monkeypatch) -> list:
    """Capture submissions instead of running hours of inference."""
    submitted: list = []

    class _Stub:
        id = "stub0run"

    def _submit(spec, universe, start, end, requested_by):
        submitted.append((spec, universe, start, end, requested_by))
        return _Stub()

    monkeypatch.setattr(runs, "submit", _submit)
    return submitted


@pytest.mark.parametrize("mandate", [
    "../../../etc/passwd",
    "/etc/passwd",
    "..%2Fexample.yaml",
    "subdir/example.yaml",
])
def test_a_mandate_cannot_escape_the_mandates_directory(
    client: TestClient, signed_in: dict, no_real_backtests: list, mandate: str
) -> None:
    response = client.post("/simulate", headers=signed_in, follow_redirects=False, data={
        "mandate": mandate, "universe": "AAPL", "start": "2025-01-01", "end": "2025-06-01",
    })
    assert response.status_code == 303
    assert response.headers["location"] == "/simulate?error=unknown-mandate"
    assert no_real_backtests == []


def test_an_inverted_window_is_rejected(
    client: TestClient, signed_in: dict, no_real_backtests: list
) -> None:
    response = client.post("/simulate", headers=signed_in, follow_redirects=False, data={
        "mandate": "example.yaml", "universe": "AAPL", "start": "2025-06-01", "end": "2025-01-01",
    })
    assert response.headers["location"] == "/simulate?error=invalid-input"
    assert no_real_backtests == []


def test_an_empty_universe_is_rejected(
    client: TestClient, signed_in: dict, no_real_backtests: list
) -> None:
    response = client.post("/simulate", headers=signed_in, follow_redirects=False, data={
        "mandate": "example.yaml", "universe": " , , ", "start": "2025-01-01", "end": "2025-06-01",
    })
    assert response.headers["location"] == "/simulate?error=invalid-input"
    assert no_real_backtests == []


def test_a_valid_submission_normalizes_the_universe(
    client: TestClient, signed_in: dict, no_real_backtests: list
) -> None:
    response = client.post("/simulate", headers=signed_in, follow_redirects=False, data={
        "mandate": "example.yaml", "universe": "aapl, nvda  msft, aapl",
        "start": "2025-01-01", "end": "2025-06-01",
    })
    assert response.status_code == 303
    assert response.headers["location"] == "/runs/stub0run"
    _spec, universe, _start, _end, who = no_real_backtests[0]
    assert universe == ["AAPL", "NVDA", "MSFT"]
    assert who == "philip.chung@utoronto.ca"


# ---- analysts --------------------------------------------------------------

def test_analysts_page_renders_with_no_results_yet(client: TestClient, signed_in: dict) -> None:
    """A fresh deployment has no stored runs; the page must still load."""
    response = client.get("/analysts", headers=signed_in)
    assert response.status_code == 200
    assert "Nothing to score yet" in response.text


@pytest.mark.parametrize("name", ["../../../etc/passwd", "not-an-analyst", "../runs"])
def test_an_unknown_analyst_is_refused(client: TestClient, signed_in: dict, name: str) -> None:
    """The path segment only ever names a registered model, never a file."""
    assert client.get(f"/analysts/{name}", headers=signed_in).status_code == 404


def test_a_registered_analyst_renders_without_any_calls(client: TestClient, signed_in: dict) -> None:
    response = client.get("/analysts/graham", headers=signed_in)
    assert response.status_code == 200
    assert "has not been consulted" in response.text


def test_an_unreadable_result_file_does_not_break_the_page(
    client: TestClient, signed_in: dict, monkeypatch, tmp_path
) -> None:
    """One corrupt artifact must not take down attribution for the rest."""
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json")
    monkeypatch.setattr(runs, "result_files", lambda limit=25: [broken])

    assert client.get("/analysts", headers=signed_in).status_code == 200


# ---- the ledger stays untouched -------------------------------------------

def test_no_route_can_write_to_a_paper_fund() -> None:
    """The hash chain has one safe writer and it is the scheduled job.

    A future route that ticks a fund from a web request would race the cron
    and could break the chain, so the mutating surface is asserted empty.
    """
    mutating = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
        if method in {"POST", "PUT", "PATCH", "DELETE"}
    }
    assert mutating == {("POST", "/simulate")}


# ---- rendering -------------------------------------------------------------

def test_chart_plots_both_series() -> None:
    svg = nav_chart(["2025-01-02", "2025-01-03"], [100.0, 110.0], [100.0, 105.0], "SPY")
    assert svg.count("<polyline") == 2
    assert "2025-01-02" in svg and "2025-01-03" in svg


def test_chart_survives_a_flat_curve() -> None:
    """A zero-height band would divide by zero."""
    svg = nav_chart(["2025-01-02", "2025-01-03"], [100.0, 100.0], [100.0, 100.0], "SPY")
    assert "<polyline" in svg
    assert "nan" not in svg.lower()


def test_chart_handles_no_sessions() -> None:
    assert "No sessions" in nav_chart([], [], [], "SPY")
