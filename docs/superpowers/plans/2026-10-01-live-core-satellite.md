# Live Core + Satellite Account Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A real Alpaca account that starts at $100, takes deposits, holds an S&P 500 ETF core, and gives the paper fund's active bets a satellite share only as reviews earn it.

**Architecture:** New package `hedge_fund/live/account/` (settings, target, orders, ledger, runner, review, report, cli) beside the paper runner. A new `AlpacaLiveClient` shares a REST base with the unchanged `AlpacaPaperClient` but has its own URL and keys. The live account never assesses anything: the satellite copies the latest paper plan's final weights.

**Tech Stack:** Python 3.12, pydantic v2, requests, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-live-core-satellite-design.md`

**Test command:** `FINANCIAL_DATASETS_API_KEY= .venv/bin/python -m pytest <path> -q` (blanking the key keeps the live-API data tests skipped).

**Decisions made while planning (refine the spec):**
- Buys are **notional** (dollar) market day orders, so a buy can never spend more than intended; long sells are fractional-quantity market day orders; shorts and covers are whole shares.
- Buys are funded only by the cash already in the account, not by same-day sell proceeds (no reliance on unsettled funds). Underweights left by that rule are bought on the next run's cash-investing pass.
- The satellite-vs-core comparison window (halt check, live review evidence) starts at the latest applied review.
- The first run with no holdings counts as a rebalance day, so a first deposit is invested immediately.
- Rebalance cadence comes from `LiveSettings.rebalance` (default `weekly`, matching the paper mandate).

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `hedge_fund/paths.py` | modify | `LIVE_DIR`, `LIVE_SETTINGS_PATH` |
| `hedge_fund/brokers/alpaca.py` | modify | `_AlpacaRest` base; `AlpacaLiveClient`, `LiveOrder`, `CashFlow` |
| `hedge_fund/live/account/__init__.py` | create | package docstring |
| `hedge_fund/live/account/settings.py` | create | `LiveSettings`, load/save |
| `hedge_fund/live/account/target.py` | create | `satellite_weights`, `target_book`, `latest_paper_weights` |
| `hedge_fund/live/account/orders.py` | create | `PlannedOrder`, `size_orders`, `check_orders` |
| `hedge_fund/live/account/ledger.py` | create | `LiveLedger`, `LiveNavRow` |
| `hedge_fund/live/account/runner.py` | create | `reconcile_live`, `submit_live` |
| `hedge_fund/live/account/review.py` | create | `propose_review`, `apply_review` |
| `hedge_fund/live/account/report.py` | create | `build_live_report` |
| `hedge_fund/live/account/cli.py` | create | `aihf-live` |
| `pyproject.toml`, `.env.example`, `README.md` | modify | entry point, keys, docs |

---

### Task 1: Paths and the live Alpaca client

**Files:** Modify `hedge_fund/paths.py`, `hedge_fund/brokers/alpaca.py`; Test `hedge_fund/brokers/test_alpaca_live.py`

- [ ] **Step 1: Paths** — after `MARKET_DB_PATH` in `hedge_fund/paths.py` add:

```python
LIVE_DIR = USER_DIR / "live"                 # the real-money account's ledger
LIVE_SETTINGS_PATH = USER_DIR / "live.yaml"  # its settings (agent share, core ticker, ...)
```

- [ ] **Step 2: Failing tests** — create `hedge_fund/brokers/test_alpaca_live.py`:

```python
"""AlpacaLiveClient tests — fake session, no network, never a real order."""

import pytest

from hedge_fund.brokers.alpaca import LIVE_BASE_URL, AlpacaLiveClient, AlpacaPaperClient
from hedge_fund.brokers.test_alpaca import FakeResponse, FakeSession


def live(*responses):
    session = FakeSession(*responses)
    return AlpacaLiveClient("live-key", "live-secret", session=session), session


def test_uses_live_host_and_live_keys_only(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "paper-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "paper-secret")
    monkeypatch.delenv("ALPACA_LIVE_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_LIVE_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="ALPACA_LIVE_KEY_ID"):
        AlpacaLiveClient(session=FakeSession())          # paper keys are never picked up
    with pytest.raises(ValueError, match="paper"):
        AlpacaLiveClient("paper-key", "x", session=FakeSession())
    client, session = live(FakeResponse(payload={"cash": "5", "equity": "5", "last_equity": "5", "status": "ACTIVE"}))
    client.account()
    assert session.calls[0]["url"] == f"{LIVE_BASE_URL}/v2/account"


def test_paper_client_still_refuses_the_live_host():
    with pytest.raises(ValueError, match="paper"):
        AlpacaPaperClient("k", "s", base_url=LIVE_BASE_URL, session=FakeSession())


def test_holdings_are_signed_fractions():
    client, _ = live(FakeResponse(payload=[
        {"symbol": "SPY", "qty": "0.1308", "side": "long"},
        {"symbol": "XYZ", "qty": "-3", "side": "short"},
    ]))
    assert client.holdings() == {"SPY": pytest.approx(0.1308), "XYZ": -3.0}


def test_buy_notional_and_sell_fraction():
    order = {"id": "o1", "client_order_id": "c", "symbol": "SPY", "side": "buy", "qty": None,
             "notional": "100.00", "filled_qty": "0", "filled_avg_price": None, "status": "accepted"}
    client, session = live(FakeResponse(payload=order), FakeResponse(payload={**order, "side": "sell", "qty": "0.05", "notional": None}))
    bought = client.buy_notional("SPY", 100.0, "c")
    assert session.calls[0]["json"] == {"symbol": "SPY", "notional": "100.00", "side": "buy", "type": "market",
                                        "time_in_force": "day", "client_order_id": "c"}
    assert (bought.notional, bought.qty, bought.status) == (100.0, None, "accepted")
    client.sell_qty("SPY", 0.05, "c")
    assert session.calls[1]["json"]["qty"] == "0.05"


def test_rejection_is_recorded():
    client, _ = live(FakeResponse(403, {"message": "insufficient buying power"}))
    result = client.buy_notional("SPY", 100.0, "c")
    assert result.status == "rejected" and "buying power" in result.reason


def test_cash_flows_map_and_paginate():
    page1 = [{"id": f"a{i}", "activity_type": "CSD", "date": "2026-10-01", "net_amount": "50"} for i in range(100)]
    page2 = [{"id": "b", "activity_type": "DIV", "date": "2026-10-02", "net_amount": "0.12"}]
    client, session = live(FakeResponse(payload=page1), FakeResponse(payload=page2))
    flows = client.cash_flows(after="2026-09-30")
    assert len(flows) == 101 and flows[-1].kind == "DIV" and flows[-1].amount == pytest.approx(0.12)
    assert session.calls[1]["params"]["page_token"] == "a99"
```

- [ ] **Step 3: Run** — `FINANCIAL_DATASETS_API_KEY= .venv/bin/python -m pytest hedge_fund/brokers/test_alpaca_live.py -q` → FAIL (ImportError).

- [ ] **Step 4: Implement** — in `hedge_fund/brokers/alpaca.py`:

Replace the module docstring's second paragraph ("Paper only, by construction: ... no live-money URL anywhere in the codebase.") with:

```
Two clients share one REST base. AlpacaPaperClient talks only to the paper
endpoint with the APCA_* paper keys. AlpacaLiveClient talks only to the live
endpoint with its own ALPACA_LIVE_* keys and refuses a paper key, so a paper
setting can never reach real money and vice versa.
```

Add after `PAPER_BASE_URL`:

```python
LIVE_BASE_URL = "https://api.alpaca.markets"
```

Insert before `class AlpacaPaperClient`:

```python
class LiveOrder(BaseModel):
    """A live order: a dollar (notional) buy, or a fractional / whole-share quantity."""

    client_order_id: str
    ticker: str
    side: Literal["buy", "sell"]
    status: str                          # Alpaca's order status, or "rejected"
    qty: float | None = None
    notional: float | None = None
    order_id: str | None = None
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    reason: str | None = None


class CashFlow(BaseModel):
    """One account activity: a deposit/withdrawal/journal (a flow) or a dividend/fee (a return)."""

    id: str
    kind: str                            # Alpaca activity type: CSD, CSW, JNLC, DIV, FEE
    date: str
    amount: float                        # signed: + into the account


class _AlpacaRest:
    """Shared transport: auth headers, JSON requests, error mapping."""

    def __init__(self, key_id: str, secret_key: str, base_url: str, *, timeout: float,
                 session: requests.Session | None) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._session = session or requests.Session()
        self._session.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key})

    def account(self) -> Account:
        row = self._request("GET", "/v2/account")
        return Account(cash=float(row["cash"]), equity=float(row["equity"]),
                       last_equity=float(row["last_equity"]), status=row["status"])

    def calendar(self, start: str, end: str) -> list[str]:
        """Trading-session dates (YYYY-MM-DD) in [start, end]."""
        rows = self._request("GET", "/v2/calendar", params={"start": start, "end": end})
        return sorted(row["date"] for row in rows)

    def _request(self, method: str, path: str, *, params: dict | None = None, json: dict | None = None) -> Any:
        try:
            resp = self._session.request(method, self._base_url + path, params=params, json=json, timeout=self._timeout)
        except requests.RequestException as exc:
            raise AlpacaError(f"{method} {path}: {exc}") from exc
        if resp.status_code >= 400:
            raise AlpacaError(f"{method} {path}: HTTP {resp.status_code}: {resp.text[:300]}", status_code=resp.status_code)
        return resp.json()
```

Change `class AlpacaPaperClient:` to `class AlpacaPaperClient(_AlpacaRest):`. In its `__init__`, replace the last three lines (`self._timeout = ...`, `self._session = ...`, `self._session.headers.update(...)`) with:

```python
        super().__init__(key_id, secret_key, PAPER_BASE_URL, timeout=timeout, session=session)
```

Delete `AlpacaPaperClient.account`, `AlpacaPaperClient.calendar` and `AlpacaPaperClient._request` (now inherited).

Append after `AlpacaPaperClient` (before `_order_result`):

```python
_FLOW_PAGE = 100


class AlpacaLiveClient(_AlpacaRest):
    """The real-money account. Live endpoint and ALPACA_LIVE_* keys only."""

    def __init__(self, key_id: str | None = None, secret_key: str | None = None, *,
                 timeout: float = 30.0, session: requests.Session | None = None) -> None:
        key_id = key_id or os.environ.get("ALPACA_LIVE_KEY_ID", "")
        secret_key = secret_key or os.environ.get("ALPACA_LIVE_SECRET_KEY", "")
        if not key_id or not secret_key:
            raise ValueError("set ALPACA_LIVE_KEY_ID and ALPACA_LIVE_SECRET_KEY (live keys) in ~/.hedge-fund/.env")
        if key_id == os.environ.get("APCA_API_KEY_ID"):
            raise ValueError("ALPACA_LIVE_KEY_ID is the paper key; the live account needs its own live keys")
        super().__init__(key_id, secret_key, LIVE_BASE_URL, timeout=timeout, session=session)

    def holdings(self) -> dict[str, float]:
        """Signed share quantities, fractions included. Negative = short."""
        held: dict[str, float] = {}
        for row in self._request("GET", "/v2/positions"):
            qty = abs(float(row["qty"]))
            if qty > 1e-9:
                held[row["symbol"]] = -qty if row.get("side") == "short" else qty
        return held

    def list_orders(self, after: str) -> list[LiveOrder]:
        rows = self._request("GET", "/v2/orders", params={
            "status": "all", "after": after, "limit": 500, "direction": "asc",
        })
        return [_live_order(row) for row in rows]

    def buy_notional(self, ticker: str, dollars: float, client_order_id: str) -> LiveOrder:
        """Buy `dollars` worth (fractional shares) with a market day order."""
        return self._send({"symbol": ticker, "notional": f"{dollars:.2f}", "side": "buy"}, client_order_id)

    def sell_qty(self, ticker: str, qty: float, client_order_id: str) -> LiveOrder:
        """Sell a (fractional) quantity of a long with a market day order."""
        return self._send({"symbol": ticker, "qty": _qty(qty), "side": "sell"}, client_order_id)

    def trade_shares(self, ticker: str, side: Literal["buy", "sell"], shares: int, client_order_id: str) -> LiveOrder:
        """Whole-share market day order — shorts and covers."""
        return self._send({"symbol": ticker, "qty": str(shares), "side": side}, client_order_id)

    def cash_flows(self, after: str) -> list[CashFlow]:
        """Deposits, withdrawals, journals, dividends and fees after date `after` (YYYY-MM-DD)."""
        flows: list[CashFlow] = []
        params = {"activity_types": "CSD,CSW,JNLC,DIV,FEE", "after": after, "direction": "asc", "page_size": _FLOW_PAGE}
        while True:
            rows = self._request("GET", "/v2/account/activities", params=dict(params))
            flows.extend(CashFlow(id=r["id"], kind=r["activity_type"], date=str(r.get("date") or r.get("transaction_time", ""))[:10],
                                  amount=float(r.get("net_amount") or 0)) for r in rows)
            if len(rows) < _FLOW_PAGE:
                return flows
            params["page_token"] = rows[-1]["id"]

    def _send(self, body: dict, client_order_id: str) -> LiveOrder:
        body = {**body, "type": "market", "time_in_force": "day", "client_order_id": client_order_id}
        try:
            return _live_order(self._request("POST", "/v2/orders", json=body))
        except AlpacaError as exc:
            if exc.status_code in _REJECTION_CODES:
                return LiveOrder(client_order_id=client_order_id, ticker=body["symbol"], side=body["side"],
                                 status="rejected", qty=float(body["qty"]) if "qty" in body else None,
                                 notional=float(body["notional"]) if "notional" in body else None, reason=str(exc))
            raise


def _qty(qty: float) -> str:
    return f"{qty:.9f}".rstrip("0").rstrip(".")


def _live_order(row: dict) -> LiveOrder:
    def num(key: str) -> float | None:
        return float(row[key]) if row.get(key) not in (None, "") else None
    return LiveOrder(
        client_order_id=row["client_order_id"], ticker=row["symbol"], side=row["side"], status=row["status"],
        qty=num("qty"), notional=num("notional"), order_id=row.get("id"),
        filled_qty=num("filled_qty") or 0.0, filled_avg_price=num("filled_avg_price"),
    )
```

- [ ] **Step 5: Run** — `FINANCIAL_DATASETS_API_KEY= .venv/bin/python -m pytest hedge_fund/brokers -q` → PASS (old paper tests included).

- [ ] **Step 6: Commit** — `git add hedge_fund/paths.py hedge_fund/brokers/alpaca.py hedge_fund/brokers/test_alpaca_live.py && git commit -m "Add a separately keyed live Alpaca client"`

---

### Task 2: Live settings

**Files:** Create `hedge_fund/live/account/__init__.py`, `hedge_fund/live/account/settings.py`; Test `hedge_fund/live/account/test_settings.py`

- [ ] **Step 1: Failing tests** — `hedge_fund/live/account/__init__.py`:

```python
"""The real-money account: an S&P 500 core plus a satellite that mirrors the paper fund."""
```

`hedge_fund/live/account/test_settings.py`:

```python
import pytest

from hedge_fund.live.account.settings import LiveSettings, load_settings, save_settings


def test_defaults_are_safe():
    s = LiveSettings()
    assert (s.confirm_live, s.agent_share, s.core_ticker, s.shorts_enabled) == (False, 0.0, "SPY", False)


def test_round_trip(tmp_path):
    path = tmp_path / "live.yaml"
    save_settings(LiveSettings(confirm_live=True, agent_share=0.2), path)
    assert load_settings(path) == LiveSettings(confirm_live=True, agent_share=0.2)


def test_share_capped_and_unknown_keys_rejected(tmp_path):
    with pytest.raises(ValueError):
        LiveSettings(agent_share=0.6)
    path = tmp_path / "live.yaml"
    path.write_text("confirm_live: true\nleverage: 3\n")
    with pytest.raises(ValueError, match="leverage"):
        load_settings(path)


def test_missing_file_explains(tmp_path):
    with pytest.raises(ValueError, match="aihf-live init"):
        load_settings(tmp_path / "nope.yaml")
```

- [ ] **Step 2: Run** → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement** `hedge_fund/live/account/settings.py`:

```python
"""Settings for the real-money account (~/.hedge-fund/live.yaml).

Defaults are the safe state: not confirmed for live trading, no agent
share, no shorts. A review changes agent_share; the user changes the rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class LiveSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm_live: bool = Field(default=False, description="must be true before any real order is sent")
    core_ticker: str = Field(default="SPY", description="the S&P 500 ETF the core holds")
    paper_fund: str = Field(default="paper-fund", description="the paper fund whose bets the satellite mirrors")
    rebalance: Literal["daily", "weekly", "monthly"] = "weekly"
    agent_share: float = Field(default=0.0, ge=0, le=0.5, description="fraction of the account the satellite gets")
    satellite_max_name_pct: float = Field(default=0.10, gt=0, le=1)
    shorts_enabled: bool = Field(default=False, description="set true only after enabling margin at Alpaca")
    short_min_equity: float = Field(default=2000.0, ge=2000.0)
    satellite_halt_relative: float = Field(default=0.10, gt=0, lt=1)
    min_order_usd: float = Field(default=1.0, ge=1.0)
    min_trade_pct: float = Field(default=0.005, ge=0, lt=1)
    cash_buffer_pct: float = Field(default=0.01, ge=0, lt=0.5, description="cash left unspent for price moves and fees")
    stale_plan_days: int = Field(default=8, ge=1)


def load_settings(path: str | Path) -> LiveSettings:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{path} not found; run `aihf-live init` to create it")
    try:
        data = yaml.safe_load(path.read_text()) or {}
        return LiveSettings.model_validate(data)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def save_settings(settings: LiveSettings, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(settings.model_dump(), sort_keys=False))
```

- [ ] **Step 4: Run** `FINANCIAL_DATASETS_API_KEY= .venv/bin/python -m pytest hedge_fund/live/account -q` → PASS.
- [ ] **Step 5: Commit** — `git add hedge_fund/live/account && git commit -m "Add live account settings"`

---

### Task 3: Target book

**Files:** Create `hedge_fund/live/account/target.py`; Test `hedge_fund/live/account/test_target.py`

- [ ] **Step 1: Failing tests:**

```python
import pytest

from hedge_fund.live.account.target import latest_paper_weights, satellite_weights, target_book
from hedge_fund.live.ledger import Ledger

PAPER = {"NVDA": 0.04, "MSFT": 0.04, "WMT": -0.02}   # gross 0.10


def test_no_share_is_all_core():
    assert target_book(satellite_weights(PAPER, 0.0, shorts_ok=True, max_name=0.1, core_ticker="SPY"), "SPY") == {"SPY": 1.0}


def test_long_only_stage_drops_shorts_and_scales_to_the_share():
    sat = satellite_weights(PAPER, 0.2, shorts_ok=False, max_name=0.5, core_ticker="SPY")
    assert sat == {"NVDA": pytest.approx(0.1), "MSFT": pytest.approx(0.1)}
    assert target_book(sat, "SPY") == {"SPY": pytest.approx(0.8), "NVDA": pytest.approx(0.1), "MSFT": pytest.approx(0.1)}


def test_shorts_stage_keeps_signs_and_gross_equals_share():
    sat = satellite_weights(PAPER, 0.2, shorts_ok=True, max_name=0.5, core_ticker="SPY")
    assert sat == {"NVDA": pytest.approx(0.08), "MSFT": pytest.approx(0.08), "WMT": pytest.approx(-0.04)}
    assert target_book(sat, "SPY")["SPY"] == pytest.approx(0.8)


def test_name_cap_overflow_goes_to_core():
    sat = satellite_weights({"NVDA": 0.5}, 0.3, shorts_ok=False, max_name=0.1, core_ticker="SPY")
    assert sat == {"NVDA": 0.1}
    assert target_book(sat, "SPY")["SPY"] == pytest.approx(0.9)


def test_core_ticker_in_paper_weights_is_not_satellite():
    assert satellite_weights({"SPY": 0.5}, 0.2, shorts_ok=False, max_name=0.1, core_ticker="SPY") == {}


def test_latest_paper_weights_freshness(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.write_plan("2026-09-28", {"plan": {"decision": {"final_weights": {"NVDA": 0.04, "X": 0.0}}}})
    assert latest_paper_weights(ledger, "2026-10-01", 8) == ({"NVDA": 0.04}, "2026-09-28", True)
    assert latest_paper_weights(ledger, "2026-10-08", 8)[2] is False
    ledger.write_plan("2026-09-29", {"plan": {"decision": None}})          # a flatten plan: no bets
    assert latest_paper_weights(ledger, "2026-10-01", 8) == ({}, "2026-09-29", True)
    assert latest_paper_weights(Ledger(tmp_path / "none"), "2026-10-01", 8) == ({}, None, False)
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `hedge_fund/live/account/target.py`:

```python
"""What the live account should hold: the core plus the agents' share of the paper fund's bets."""

from __future__ import annotations

from datetime import date

from hedge_fund.live.ledger import Ledger


def satellite_weights(
    paper_weights: dict[str, float], share: float, *, shorts_ok: bool, max_name: float, core_ticker: str,
) -> dict[str, float]:
    """The paper fund's active book rescaled so its gross equals `share` of the account.

    Without shorts_ok, short bets are dropped and the longs fill the share.
    Each name is capped at max_name; what a cap removes goes to the core,
    never to other names.
    """
    bets = {t: w for t, w in paper_weights.items() if w and t != core_ticker and (shorts_ok or w > 0)}
    gross = sum(abs(w) for w in bets.values())
    if share <= 0 or gross <= 0:
        return {}
    return {t: max(-max_name, min(max_name, share * w / gross)) for t, w in bets.items()}


def target_book(satellite: dict[str, float], core_ticker: str) -> dict[str, float]:
    """Satellite weights plus the core, which takes everything the satellite doesn't use."""
    return {core_ticker: 1.0 - sum(abs(w) for w in satellite.values()), **satellite}


def latest_paper_weights(paper: Ledger, today: str, stale_days: int) -> tuple[dict[str, float], str | None, bool]:
    """The newest real paper plan's final (pre-equitization) weights, its session, and whether it's fresh."""
    sessions = [s for s in paper.plan_sessions() if s <= today]
    if not sessions:
        return {}, None, False
    session = sessions[-1]
    decision = (paper.read_plan(session) or {}).get("plan", {}).get("decision") or {}
    weights = {t: w for t, w in decision.get("final_weights", {}).items() if w}
    fresh = (date.fromisoformat(today) - date.fromisoformat(session)).days <= stale_days
    return weights, session, fresh
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** — `git add hedge_fund/live/account && git commit -m "Compute the live target book from the paper fund's bets"`

---

### Task 4: Order sizing and limits

**Files:** Create `hedge_fund/live/account/orders.py`; Test `hedge_fund/live/account/test_orders.py`

- [ ] **Step 1: Failing tests:**

```python
import pytest

from hedge_fund.live.account.orders import PlannedOrder, check_orders, size_orders

MARKS = {"SPY": 500.0, "NVDA": 100.0, "WMT": 100.0}
KW = dict(min_order_usd=1.0, min_trade_pct=0.005, cash_buffer_pct=0.0)


def plan(holdings, cash, target, rebalance=True, **kw):
    return size_orders(holdings, cash, MARKS, target, rebalance=rebalance, **{**KW, **kw})


def test_first_deposit_buys_the_core_by_dollars():
    assert plan({}, 100.0, {"SPY": 1.0}) == [PlannedOrder(ticker="SPY", side="buy", dollars=100.0)]


def test_cash_buffer_is_left_unspent():
    assert plan({}, 100.0, {"SPY": 1.0}, cash_buffer_pct=0.01)[0].dollars == 99.0


def test_deposit_day_only_buys_with_cash():
    # Holding 0.2 SPY ($100) + $50 new cash; target 80/20 SPY/NVDA. No sells on a cash day.
    orders = plan({"SPY": 0.2}, 50.0, {"SPY": 0.8, "NVDA": 0.2}, rebalance=False)
    assert all(o.side == "buy" for o in orders)
    assert sum(o.dollars for o in orders) == pytest.approx(50.0)
    assert {o.ticker for o in orders} == {"SPY", "NVDA"}


def test_rebalance_sells_fractions_and_closes_unwanted_names():
    orders = plan({"SPY": 1.0, "WMT": 0.004}, 0.0, {"SPY": 1.0})
    assert orders == [PlannedOrder(ticker="WMT", side="sell", qty=0.004)]   # $0.40 close is never skipped


def test_dust_trades_skipped_on_rebalance():
    # $1,000 account ($994 SPY + $6 cash): SPY +$2 and NVDA +$4 are both under the 0.5% floor ($5).
    orders = plan({"SPY": 1.988}, 6.0, {"SPY": 0.996, "NVDA": 0.004}, rebalance=True)
    assert orders == []


def test_buys_scaled_to_available_cash_not_sale_proceeds():
    orders = plan({"SPY": 1.0}, 10.0, {"SPY": 0.5, "NVDA": 0.5})
    sells = [o for o in orders if o.side == "sell"]
    buys = [o for o in orders if o.side == "buy"]
    assert sells == [PlannedOrder(ticker="SPY", side="sell", qty=pytest.approx(0.49))]
    assert buys == [PlannedOrder(ticker="NVDA", side="buy", dollars=10.0)]


def test_shorts_use_whole_shares():
    orders = plan({"SPY": 4.0}, 2_000.0, {"SPY": 0.9, "WMT": -0.1})   # equity 4,000 → short $400 = 4 sh
    assert PlannedOrder(ticker="WMT", side="sell", shares=4) in orders


def test_check_blocks_overspend_shorts_and_oversized_names():
    check_orders([PlannedOrder(ticker="SPY", side="buy", dollars=100.0)], {}, 100.0, MARKS, "SPY",
                 shorts_ok=False, max_name=0.1)
    with pytest.raises(ValueError, match="cash"):
        check_orders([PlannedOrder(ticker="SPY", side="buy", dollars=101.0)], {}, 100.0, MARKS, "SPY",
                     shorts_ok=False, max_name=0.1)
    with pytest.raises(ValueError, match="short"):
        check_orders([PlannedOrder(ticker="WMT", side="sell", shares=1)], {}, 1_000.0, MARKS, "SPY",
                     shorts_ok=False, max_name=0.5)
    with pytest.raises(ValueError, match="NVDA"):
        check_orders([PlannedOrder(ticker="NVDA", side="buy", dollars=50.0)], {}, 100.0, MARKS, "SPY",
                     shorts_ok=False, max_name=0.1)
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `hedge_fund/live/account/orders.py`:

```python
"""Turn a target book into live orders, and refuse orders that break the account's limits.

Buys are dollar amounts (fractional shares), funded only by cash already in
the account. Long sells are fractional quantities. Shorts and covers are
whole shares. Sells are listed first.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel

_TOLERANCE = 0.005   # 0.5% of equity: price moves between sizing and fill


class PlannedOrder(BaseModel):
    ticker: str
    side: Literal["buy", "sell"]
    dollars: float | None = None   # notional buy
    qty: float | None = None       # fractional long sell
    shares: int | None = None      # whole-share short / cover


def size_orders(
    holdings: dict[str, float], cash: float, marks: dict[str, float], target: dict[str, float], *,
    rebalance: bool, min_order_usd: float, min_trade_pct: float, cash_buffer_pct: float,
) -> list[PlannedOrder]:
    """Orders that move `holdings` toward `target` (weights of equity).

    On a rebalance day every name is realigned (dust under the larger of
    min_order_usd and min_trade_pct of equity skipped, closes never skipped).
    Otherwise only uninvested cash is put to work, toward the most
    underweight names. Buys never exceed cash × (1 − cash_buffer_pct).
    """
    equity = cash + sum(q * marks[t] for t, q in holdings.items())
    if equity <= 0:
        return []
    floor = max(min_order_usd, min_trade_pct * equity)
    sells: list[PlannedOrder] = []
    covers: list[PlannedOrder] = []
    wanted: dict[str, float] = {}
    for t in sorted(set(target) | set(holdings)):
        weight, q, mark = target.get(t, 0.0), holdings.get(t, 0.0), marks[t]
        delta = weight * equity - q * mark
        if weight < 0 or q < 0:                                  # short side: whole shares
            if not rebalance:
                continue
            target_shares = math.trunc(weight * equity / mark) if weight < 0 else 0
            change = target_shares - round(q)
            if change < 0 and -change * mark >= floor:
                sells.append(PlannedOrder(ticker=t, side="sell", shares=-change))
            elif change > 0 and (target_shares == 0 or change * mark >= floor):
                covers.append(PlannedOrder(ticker=t, side="buy", shares=change))
            continue
        if delta < 0 and q > 0 and rebalance:
            closing = weight <= 0
            if closing or -delta >= floor:
                sells.append(PlannedOrder(ticker=t, side="sell", qty=round(q if closing else min(q, -delta / mark), 9)))
        elif delta > 0 and (not rebalance or delta >= floor):
            wanted[t] = delta

    budget = cash * (1 - cash_buffer_pct) - sum(o.shares * marks[o.ticker] for o in covers)
    total = sum(wanted.values())
    buys: list[PlannedOrder] = []
    if budget >= min_order_usd and total > 0:
        scale = min(1.0, budget / total)
        for t in sorted(wanted):
            dollars = math.floor(wanted[t] * scale * 100) / 100
            if dollars >= min_order_usd:
                buys.append(PlannedOrder(ticker=t, side="buy", dollars=dollars))
    return sells + covers + buys


def check_orders(
    orders: list[PlannedOrder], holdings: dict[str, float], cash: float, marks: dict[str, float],
    core_ticker: str, *, shorts_ok: bool, max_name: float,
) -> None:
    """Raise if these orders would overspend cash, short without permission, or oversize a name."""
    equity = cash + sum(q * marks[t] for t, q in holdings.items())
    spend = sum(o.dollars for o in orders if o.dollars) + sum(o.shares * marks[o.ticker] for o in orders if o.shares and o.side == "buy")
    if spend > cash + 1e-6:
        raise ValueError(f"orders spend ${spend:,.2f} but the account has ${cash:,.2f} cash")
    projected = dict(holdings)
    for o in orders:
        if o.dollars:
            change = o.dollars / marks[o.ticker]
        elif o.qty:
            change = -o.qty
        else:
            change = o.shares if o.side == "buy" else -o.shares
        projected[o.ticker] = projected.get(o.ticker, 0.0) + change
    for t, q in projected.items():
        if q < -1e-9 and not shorts_ok:
            raise ValueError(f"{t}: would short without shorts enabled (margin and ${2000:,}+ equity)")
        if t != core_ticker and equity > 0 and abs(q * marks[t]) / equity > max_name + _TOLERANCE:
            raise ValueError(f"{t}: would be {abs(q * marks[t]) / equity:.1%} of the account (max {max_name:.0%})")
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** — `git add hedge_fund/live/account && git commit -m "Size live orders with fractional buys and hard limits"`

---

### Task 5: Live ledger

**Files:** Create `hedge_fund/live/account/ledger.py`; Test `hedge_fund/live/account/test_ledger.py`

- [ ] **Step 1: Failing tests:**

```python
import pytest

from hedge_fund.brokers.alpaca import CashFlow
from hedge_fund.live.account.ledger import LiveLedger, LiveNavRow


def row(day, equity, flow=0.0, core_close=500.0, sat=None):
    return LiveNavRow(date=day, equity=equity, cash=0.0, net_flow=flow, core_value=equity,
                      satellite_value=0.0, core_close=core_close, satellite_return=sat)


def test_flows_dedupe_and_net_flow_excludes_dividends(tmp_path):
    ledger = LiveLedger(tmp_path)
    flows = [CashFlow(id="1", kind="CSD", date="2026-10-01", amount=100.0),
             CashFlow(id="2", kind="DIV", date="2026-10-02", amount=0.5),
             CashFlow(id="3", kind="CSW", date="2026-10-02", amount=-20.0)]
    ledger.add_flows(flows)
    ledger.add_flows(flows)
    assert len(ledger.flows()) == 3
    assert ledger.net_flow("2026-09-30", "2026-10-02") == pytest.approx(80.0)
    assert ledger.net_flow("2026-10-01", "2026-10-02") == pytest.approx(-20.0)   # (after, through]
    assert ledger.total_deposited() == pytest.approx(80.0)


def test_nav_rows_holdings_and_markers(tmp_path):
    ledger = LiveLedger(tmp_path)
    ledger.upsert_live_nav(row("2026-10-02", 101.0))
    ledger.upsert_live_nav(row("2026-10-01", 100.0, flow=100.0))
    ledger.upsert_live_nav(row("2026-10-02", 102.0))
    assert [(r.date, r.equity) for r in ledger.live_nav_rows()] == [("2026-10-01", 100.0), ("2026-10-02", 102.0)]
    ledger.save_holdings("2026-10-01", {"SPY": 0.2})
    ledger.save_holdings("2026-10-02", {"SPY": 0.3})
    assert ledger.holdings_before("2026-10-02") == ("2026-10-01", {"SPY": 0.2})
    assert ledger.holdings_before("2026-10-01") is None
    assert not ledger.dry_run_done()
    ledger.mark_dry_run_done()
    assert ledger.dry_run_done()
    ledger.halt_satellite("trailing")
    assert ledger.satellite_halted()
    ledger.clear_satellite_halt()
    assert not ledger.satellite_halted()


def test_reviews_and_satellite_excess(tmp_path):
    ledger = LiveLedger(tmp_path)
    assert ledger.last_review_date() is None
    ledger.append_review("2026-10-01", 0.0, 0.2, paper_excess=0.01, live_excess=None, passed=True)
    assert ledger.last_review_date() == "2026-10-01"
    for day, close, sat in [("2026-10-01", 500.0, None), ("2026-10-02", 510.0, 0.01), ("2026-10-05", 520.2, -0.05)]:
        ledger.upsert_live_nav(row(day, 100.0, core_close=close, sat=sat))
    # satellite 1.01 × 0.95 − 1 = −4.05%; core 520.2/500 − 1 = +4.04%
    assert ledger.satellite_excess(since="2026-10-01") == pytest.approx(0.9595 - 1.0404, abs=1e-6)
    assert ledger.satellite_excess(since="2026-10-06") is None
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `hedge_fund/live/account/ledger.py`:

```python
"""The live account's books: daily NAV with deposits separated, cash flows, holdings, reviews."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import BaseModel

from hedge_fund.brokers.alpaca import CashFlow
from hedge_fund.live.ledger import Ledger
from hedge_fund.paths import LIVE_DIR

FLOW_KINDS = frozenset({"CSD", "CSW", "JNLC"})   # money in or out; dividends and fees are returns


class LiveNavRow(BaseModel):
    date: str
    equity: float
    cash: float
    net_flow: float                     # deposits − withdrawals since the previous row
    core_value: float
    satellite_value: float
    core_close: float
    satellite_return: float | None = None   # satellite holdings' return since the previous row


class LiveLedger(Ledger):
    """Plans, logs and JSON helpers come from the paper Ledger; NAV, flows and reviews are live-specific."""

    @classmethod
    def default(cls) -> "LiveLedger":
        return cls(LIVE_DIR / "main")

    # -- NAV ---------------------------------------------------------------
    def upsert_live_nav(self, row: LiveNavRow) -> None:
        rows = {r.date: r for r in self.live_nav_rows()}
        rows[row.date] = row
        _write_csv(self.root / "nav.csv", list(LiveNavRow.model_fields), [rows[d].model_dump() for d in sorted(rows)])

    def live_nav_rows(self) -> list[LiveNavRow]:
        return [LiveNavRow.model_validate({k: (v if v != "" else None) for k, v in r.items()})
                for r in _read_csv(self.root / "nav.csv")]

    # -- flows ---------------------------------------------------------------
    def add_flows(self, flows: list[CashFlow]) -> None:
        known = {f.id: f for f in self.flows()}
        known.update({f.id: f for f in flows})
        rows = sorted(known.values(), key=lambda f: (f.date, f.id))
        _write_csv(self.root / "flows.csv", list(CashFlow.model_fields), [f.model_dump() for f in rows])

    def flows(self) -> list[CashFlow]:
        return [CashFlow.model_validate(r) for r in _read_csv(self.root / "flows.csv")]

    def net_flow(self, after: str | None, through: str) -> float:
        return sum(f.amount for f in self.flows()
                   if f.kind in FLOW_KINDS and (after is None or f.date > after) and f.date <= through)

    def total_deposited(self) -> float:
        return sum(f.amount for f in self.flows() if f.kind in FLOW_KINDS)

    # -- holdings snapshots ----------------------------------------------------
    def save_holdings(self, session: str, holdings: dict[str, float]) -> None:
        self._write_json("holdings", f"{session}.json", holdings)

    def holdings_before(self, session: str) -> tuple[str, dict[str, float]] | None:
        earlier = [s for s in self._sessions("holdings") if s < session]
        return (earlier[-1], self._read_json("holdings", f"{earlier[-1]}.json")) if earlier else None

    # -- markers -----------------------------------------------------------------
    def dry_run_done(self) -> bool:
        return (self.root / "DRY_RUN_DONE").exists()

    def mark_dry_run_done(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "DRY_RUN_DONE").touch()

    @property
    def satellite_halt_path(self) -> Path:
        return self.root / "SATELLITE_HALTED"

    def satellite_halted(self) -> bool:
        return self.satellite_halt_path.exists()

    def halt_satellite(self, reason: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.satellite_halt_path.write_text(reason + "\n")

    def clear_satellite_halt(self) -> None:
        self.satellite_halt_path.unlink(missing_ok=True)

    # -- reviews -------------------------------------------------------------------
    def append_review(self, day: str, old_share: float, new_share: float, *,
                      paper_excess: float, live_excess: float | None, passed: bool) -> None:
        rows = _read_csv(self.root / "reviews.csv")
        rows.append({"date": day, "old_share": old_share, "new_share": new_share, "paper_excess": paper_excess,
                     "live_excess": "" if live_excess is None else live_excess, "passed": passed})
        _write_csv(self.root / "reviews.csv", ["date", "old_share", "new_share", "paper_excess", "live_excess", "passed"], rows)

    def last_review_date(self) -> str | None:
        rows = _read_csv(self.root / "reviews.csv")
        return rows[-1]["date"] if rows else None

    def satellite_excess(self, since: str) -> float | None:
        """Satellite's compounded return minus the core's, over rows after `since` that held a satellite."""
        rows = [r for r in self.live_nav_rows() if r.date >= since]
        sat, core, used = 1.0, 1.0, False
        for prev, cur in zip(rows, rows[1:]):
            if cur.satellite_return is None:
                continue
            sat *= 1 + cur.satellite_return
            core *= cur.core_close / prev.core_close
            used = True
        return sat - core if used else None

    def satellite_plans(self) -> list[tuple[str, dict]]:
        return [(s, self.read_plan(s) or {}) for s in self.plan_sessions()]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
```

Note `satellite_excess` compares the satellite with the core only on days a satellite was held, starting at `since`.

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** — `git add hedge_fund/live/account && git commit -m "Add the live account ledger"`

---

### Task 6: Reconcile and submit

**Files:** Create `hedge_fund/live/account/runner.py`; Test `hedge_fund/live/account/test_runner.py`

- [ ] **Step 1: Failing tests:**

```python
"""Live runner tests — a fake live account and fake prices; nothing reaches Alpaca."""

from datetime import datetime

import pytest

from hedge_fund.brokers.alpaca import Account, CashFlow, LiveOrder
from hedge_fund.data.sessions import NEW_YORK
from hedge_fund.live.account.ledger import LiveLedger
from hedge_fund.live.account.runner import reconcile_live, submit_live
from hedge_fund.live.account.settings import LiveSettings
from hedge_fund.live.ledger import Ledger
from hedge_fund.live.test_plan import FakeDataClient

SESSIONS = ["2026-08-19", "2026-08-20", "2026-08-21", "2026-08-24", "2026-08-25"]
CLOSES = {"SPY": {d: 500.0 for d in SESSIONS}, "NVDA": {d: 100.0 for d in SESSIONS}}
MON = datetime(2026, 8, 24, 4, 0, tzinfo=NEW_YORK)    # 09:00 UK
TUE = datetime(2026, 8, 25, 4, 0, tzinfo=NEW_YORK)


class FakeLive:
    def __init__(self, holdings=None, cash=0.0, flows=(), orders=()):
        self._holdings, self.cash, self._flows, self.orders = dict(holdings or {}), cash, list(flows), list(orders)
        self.sent = []

    def calendar(self, start, end):
        return [s for s in SESSIONS if start <= s <= end]

    def account(self):
        return Account(cash=self.cash, equity=self.cash, last_equity=self.cash, status="ACTIVE")

    def holdings(self):
        return dict(self._holdings)

    def cash_flows(self, after):
        return [f for f in self._flows if f.date > after]

    def list_orders(self, after):
        return list(self.orders)

    def _order(self, cid, ticker, side, **kw):
        self.sent.append((ticker, side, kw))
        return LiveOrder(client_order_id=cid, ticker=ticker, side=side, status="accepted", **kw)

    def buy_notional(self, ticker, dollars, cid):
        return self._order(cid, ticker, "buy", notional=dollars)

    def sell_qty(self, ticker, qty, cid):
        return self._order(cid, ticker, "sell", qty=qty)

    def trade_shares(self, ticker, side, shares, cid):
        return self._order(cid, ticker, side, qty=float(shares))


@pytest.fixture
def env(tmp_path):
    paper = Ledger(tmp_path / "paper")
    paper.write_plan("2026-08-24", {"plan": {"decision": {"final_weights": {"NVDA": 0.04}}}})
    return dict(live=LiveLedger(tmp_path / "live"), paper=paper, kill=tmp_path / "KILL")


def run(env, client, now=MON, dry_run=False, **settings):
    s = LiveSettings(confirm_live=True, **settings)
    return submit_live(s, client, FakeDataClient(CLOSES), env["live"], env["paper"], now=now,
                       dry_run=dry_run, kill_path=env["kill"])


def test_guards_in_order(env):
    env["kill"].touch()
    assert run(env, FakeLive(cash=100.0)).status == "killed"
    env["kill"].unlink()
    assert submit_live(LiveSettings(), FakeLive(cash=100.0), FakeDataClient(CLOSES), env["live"], env["paper"],
                       now=MON, kill_path=env["kill"]).status == "not_confirmed"
    assert run(env, FakeLive(cash=100.0)).status == "needs_dry_run"


def test_dry_run_then_first_deposit_goes_all_to_core(env):
    client = FakeLive(cash=100.0)
    dry = run(env, client, dry_run=True)
    assert dry.status == "dry_run" and client.sent == []
    result = run(env, client, cash_buffer_pct=0.0)
    assert result.status == "submitted"
    assert client.sent == [("SPY", "buy", {"notional": 100.0})]


def test_cash_day_invests_deposit_without_selling(env):
    env["live"].mark_dry_run_done()
    client = FakeLive(holdings={"SPY": 0.2}, cash=50.0)
    result = run(env, client, now=TUE, cash_buffer_pct=0.0)
    assert result.status == "submitted"
    assert client.sent == [("SPY", "buy", {"notional": 50.0})]


def test_nothing_to_do_without_cash_on_a_cash_day(env):
    env["live"].mark_dry_run_done()
    assert run(env, FakeLive(holdings={"SPY": 0.2}, cash=0.5), now=TUE).status == "nothing_to_do"


def test_satellite_mirrors_paper_when_share_set(env):
    env["live"].mark_dry_run_done()
    client = FakeLive(holdings={"SPY": 2.0}, cash=0.0)       # $1,000 in SPY
    run(env, client, agent_share=0.2, cash_buffer_pct=0.0)
    # NVDA's 20% share is capped at the 10% per-name limit; the rest stays in SPY: sell $100 of SPY
    assert ("SPY", "sell", {"qty": pytest.approx(0.2)}) in client.sent
    # buys wait for cash (no same-day proceeds): next cash day buys NVDA
    assert all(side == "sell" for _, side, _ in client.sent)


def test_halted_satellite_counts_as_zero_share(env):
    env["live"].mark_dry_run_done()
    env["live"].halt_satellite("trailing")
    client = FakeLive(holdings={"SPY": 1.6, "NVDA": 2.0}, cash=0.0)
    run(env, client, agent_share=0.2)
    assert ("NVDA", "sell", {"qty": pytest.approx(2.0)}) in client.sent


def test_reconcile_separates_deposits_and_tracks_satellite(env):
    live = env["live"]
    client = FakeLive(holdings={"SPY": 0.2}, cash=0.0,
                      flows=[CashFlow(id="d1", kind="CSD", date="2026-08-20", amount=100.0)])
    r1 = reconcile_live(LiveSettings(), client, FakeDataClient(CLOSES), live, session="2026-08-20")
    assert (r1.equity, r1.net_flow, r1.core_value) == (100.0, 100.0, 100.0)
    closes = {"SPY": CLOSES["SPY"], "NVDA": {**CLOSES["NVDA"], "2026-08-21": 110.0}}
    client._holdings = {"SPY": 0.2, "NVDA": 1.0}
    client.cash = -100.0  # bought NVDA on margin-free cash in this toy; equity math only
    live.save_holdings("2026-08-20", {"SPY": 0.2, "NVDA": 1.0})
    r2 = reconcile_live(LiveSettings(), client, FakeDataClient(closes), live, session="2026-08-21")
    assert r2.satellite_return == pytest.approx(0.10)
    assert r2.net_flow == 0.0


def test_reconcile_halts_a_trailing_satellite(env):
    live = env["live"]
    live.append_review("2026-08-20", 0.0, 0.2, paper_excess=0.01, live_excess=None, passed=True)
    live.save_holdings("2026-08-20", {"NVDA": 1.0})
    live.upsert_live_nav(__import__("hedge_fund.live.account.ledger", fromlist=["LiveNavRow"]).LiveNavRow(
        date="2026-08-20", equity=100.0, cash=0.0, net_flow=0.0, core_value=0.0, satellite_value=100.0, core_close=500.0))
    closes = {"SPY": CLOSES["SPY"], "NVDA": {**CLOSES["NVDA"], "2026-08-21": 85.0}}
    reconcile_live(LiveSettings(agent_share=0.2), FakeLive(holdings={"NVDA": 1.0}), FakeDataClient(closes), live,
                   session="2026-08-21")
    assert live.satellite_halted()
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `hedge_fund/live/account/runner.py`:

```python
"""The live account's daily steps: reconcile (record yesterday) and submit (trade today).

submit never assesses anything: the satellite copies the paper fund's latest
plan, scaled by the agent share. Every guard runs before any order is sent,
and the plan is written to disk before the first order.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from hedge_fund.brokers.alpaca import LiveOrder
from hedge_fund.data.protocol import DataClient
from hedge_fund.data.sessions import NEW_YORK
from hedge_fund.live.account.ledger import LiveLedger, LiveNavRow
from hedge_fund.live.account.orders import PlannedOrder, check_orders, size_orders
from hedge_fund.live.account.settings import LiveSettings
from hedge_fund.live.account.target import latest_paper_weights, satellite_weights, target_book
from hedge_fund.live.calendar import ORDER_CUTOFF, is_rebalance_day, ny_midnight
from hedge_fund.live.ledger import Ledger
from hedge_fund.paths import KILL_PATH
from hedge_fund.pipeline.run_cycle import exact_marks

logger = logging.getLogger(__name__)

LiveStatus = Literal["killed", "not_confirmed", "needs_dry_run", "not_trading_day", "too_late",
                     "already_submitted", "nothing_to_do", "dry_run", "submitted"]


class LiveSubmitResult(BaseModel):
    status: LiveStatus
    session: str
    detail: str = ""
    target: dict[str, float] = Field(default_factory=dict)
    orders: list[PlannedOrder] = Field(default_factory=list)
    results: list[LiveOrder] = Field(default_factory=list)


def reconcile_live(settings: LiveSettings, client, data_client: DataClient, ledger: LiveLedger, *, session: str) -> LiveNavRow:
    """Record `session`'s closing value, deposits and satellite return; halt a trailing satellite."""
    rows = ledger.live_nav_rows()
    previous = max((r.date for r in rows if r.date < session), default=None)
    since = previous or (datetime.fromisoformat(session) - timedelta(days=30)).date().isoformat()
    ledger.add_flows(client.cash_flows(after=since))
    holdings = client.holdings()
    cash = client.account().cash
    core = settings.core_ticker
    closes = exact_marks(sorted(set(holdings) | {core}), session, data_client)
    core_value = holdings.get(core, 0.0) * closes[core]
    satellite_value = sum(q * closes[t] for t, q in holdings.items() if t != core)

    satellite_return = None
    before = ledger.holdings_before(session)
    if before:
        prev_day, prev_holdings = before
        sat = {t: q for t, q in prev_holdings.items() if t != core and q}
        if sat:
            start = exact_marks(sorted(sat), prev_day, data_client)
            end = exact_marks(sorted(sat), session, data_client)
            base = sum(abs(q) * start[t] for t, q in sat.items())
            satellite_return = sum(q * (end[t] - start[t]) for t, q in sat.items()) / base

    row = LiveNavRow(
        date=session, equity=round(cash + core_value + satellite_value, 2), cash=round(cash, 2),
        net_flow=round(ledger.net_flow(previous, session), 2), core_value=round(core_value, 2),
        satellite_value=round(satellite_value, 2), core_close=closes[core], satellite_return=satellite_return,
    )
    ledger.upsert_live_nav(row)
    ledger.save_holdings(session, holdings)

    since_review = ledger.last_review_date()
    excess = ledger.satellite_excess(since_review) if since_review else None
    if excess is not None and excess <= -settings.satellite_halt_relative and not ledger.satellite_halted():
        ledger.halt_satellite(f"satellite trails the core by {-excess:.1%} since {since_review}; "
                              "it is sold into the core until a review sets a new share")
        logger.warning("satellite halted: trails the core by %.1f%%", -excess * 100)
    logger.info("reconcile %s: equity %.2f, net flow %.2f", session, row.equity, row.net_flow)
    return row


def submit_live(
    settings: LiveSettings, client, data_client: DataClient, ledger: LiveLedger, paper: Ledger, *,
    now: datetime, dry_run: bool = False, kill_path: Path = KILL_PATH,
) -> LiveSubmitResult:
    now = now.astimezone(NEW_YORK)
    session = now.date().isoformat()

    def done(status: LiveStatus, detail: str = "", **extra) -> LiveSubmitResult:
        logger.info("live submit %s: %s %s", session, status, detail)
        return LiveSubmitResult(status=status, session=session, detail=detail, **extra)

    if kill_path.exists():
        return done("killed", f"{kill_path} exists")
    if not settings.confirm_live:
        return done("not_confirmed", "set confirm_live: true in live.yaml to allow real orders")
    if not dry_run and not ledger.dry_run_done():
        return done("needs_dry_run", "run `aihf-live submit --dry-run` once and review it first")
    sessions = client.calendar((now.date() - timedelta(days=10)).isoformat(), session)
    prefix = f"live-{session}-"
    if not dry_run:
        if not sessions or sessions[-1] != session:
            return done("not_trading_day")
        if now.time() > ORDER_CUTOFF:
            return done("too_late", f"after {ORDER_CUTOFF:%H:%M} ET")
        if any(o.client_order_id.startswith(prefix) for o in client.list_orders(after=ny_midnight(session))):
            return done("already_submitted")
    previous = [s for s in sessions if s < session]
    if not previous:
        raise ValueError("no completed session in the last 10 days to price the account")
    mark_day = previous[-1]

    holdings = client.holdings()
    cash = client.account().cash
    paper_weights, paper_session, fresh = latest_paper_weights(paper, session, settings.stale_plan_days)
    share = 0.0 if ledger.satellite_halted() else settings.agent_share
    core = settings.core_ticker
    names = sorted(set(holdings) | set(paper_weights) | {core} | set(_last_fresh_satellite(ledger)))
    marks = exact_marks(names, mark_day, data_client)
    equity = cash + sum(q * marks[t] for t, q in holdings.items())
    shorts_ok = settings.shorts_enabled and equity >= settings.short_min_equity
    if fresh:
        satellite = satellite_weights(paper_weights, share, shorts_ok=shorts_ok,
                                      max_name=settings.satellite_max_name_pct, core_ticker=core)
    else:   # the paper fund missed its run: hold the satellite as last set
        satellite = _last_fresh_satellite(ledger) if share > 0 else {}
    target = target_book(satellite, core)
    rebalance = not holdings or is_rebalance_day(session, mark_day, settings.rebalance)
    orders = size_orders(holdings, cash, marks, target, rebalance=rebalance,
                         min_order_usd=settings.min_order_usd, min_trade_pct=settings.min_trade_pct,
                         cash_buffer_pct=settings.cash_buffer_pct)
    check_orders(orders, holdings, cash, marks, core, shorts_ok=shorts_ok, max_name=settings.satellite_max_name_pct)
    payload = {"session": session, "rebalance": rebalance, "equity": equity, "cash": cash, "holdings": holdings,
               "marks": marks, "paper_plan": paper_session, "paper_fresh": fresh, "agent_share": share,
               "satellite": satellite, "target": target, "orders": [o.model_dump() for o in orders], "results": []}
    if dry_run:
        ledger.write_plan(session, payload, dry_run=True)
        ledger.mark_dry_run_done()
        return done("dry_run", f"{len(orders)} orders", target=target, orders=orders)
    if not orders:
        return done("nothing_to_do", target=target)

    ledger.write_plan(session, payload)
    results: list[LiveOrder] = []
    for o in orders:
        cid = f"{prefix}{o.ticker}-{o.side}"
        if o.dollars is not None:
            results.append(client.buy_notional(o.ticker, o.dollars, cid))
        elif o.qty is not None:
            results.append(client.sell_qty(o.ticker, o.qty, cid))
        else:
            results.append(client.trade_shares(o.ticker, o.side, o.shares, cid))
        payload["results"] = [r.model_dump() for r in results]
        ledger.write_plan(session, payload)
    rejected = sum(r.status == "rejected" for r in results)
    return done("submitted", f"{len(results)} orders, {rejected} rejected", target=target, orders=orders, results=results)


def _last_fresh_satellite(ledger: LiveLedger) -> dict[str, float]:
    for _, plan in reversed(ledger.satellite_plans()):
        if plan.get("paper_fresh"):
            return plan.get("satellite", {})
    return {}
```

- [ ] **Step 4: Run** `FINANCIAL_DATASETS_API_KEY= .venv/bin/python -m pytest hedge_fund/live/account -q` → PASS. (Fix only genuine mismatches between test arithmetic and the spec; record any in the commit message.)
- [ ] **Step 5: Commit** — `git add hedge_fund/live/account && git commit -m "Reconcile and submit the live core + satellite account"`

---

### Task 7: Review

**Files:** Create `hedge_fund/live/account/review.py`; Test `hedge_fund/live/account/test_review.py`

- [ ] **Step 1: Failing tests:**

```python
import pytest

from hedge_fund.live.account.ledger import LiveLedger
from hedge_fund.live.account.review import apply_review, propose_review
from hedge_fund.live.account.settings import LiveSettings, load_settings, save_settings
from hedge_fund.live.ledger import Ledger, NavRow


def paper_with(tmp_path, n_rebalances, fund_end, spy_end):
    paper = Ledger(tmp_path / "paper")
    days = [f"2026-{10 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n_rebalances)]
    for d in days:
        paper.write_plan(d, {"plan": {}})
    paper.upsert_nav(NavRow(date=days[0], equity=100.0, cash=0, long_exposure=0, short_exposure=0, gross=0, benchmark_close=100.0))
    paper.upsert_nav(NavRow(date=days[-1], equity=fund_end, cash=0, long_exposure=0, short_exposure=0, gross=0, benchmark_close=spy_end))
    return paper


def test_too_early(tmp_path):
    with pytest.raises(ValueError, match="11 of 12"):
        propose_review(paper_with(tmp_path, 11, 105, 100), LiveLedger(tmp_path / "live"), LiveSettings())


def test_first_pass_grants_twenty_percent(tmp_path):
    r = propose_review(paper_with(tmp_path, 12, 105, 102), LiveLedger(tmp_path / "live"), LiveSettings())
    assert r.passed and r.paper_excess == pytest.approx(0.03) and (r.old_share, r.new_share) == (0.0, 0.2)


def test_pass_steps_up_to_cap_and_fail_steps_down(tmp_path):
    paper = paper_with(tmp_path, 12, 105, 102)
    live = LiveLedger(tmp_path / "live")
    assert propose_review(paper, live, LiveSettings(agent_share=0.5)).new_share == 0.5
    assert propose_review(paper, live, LiveSettings(agent_share=0.3)).new_share == pytest.approx(0.4)
    losing = paper_with(tmp_path / "b", 12, 101, 102)
    assert propose_review(losing, live, LiveSettings(agent_share=0.3)).new_share == pytest.approx(0.2)
    assert propose_review(losing, live, LiveSettings(agent_share=0.0)).new_share == 0.0


def test_apply_writes_settings_logs_and_clears_halt(tmp_path):
    path = tmp_path / "live.yaml"
    save_settings(LiveSettings(confirm_live=True), path)
    live = LiveLedger(tmp_path / "live")
    live.halt_satellite("x")
    r = propose_review(paper_with(tmp_path, 12, 105, 102), live, load_settings(path))
    apply_review(r, load_settings(path), path, live, today="2026-12-20")
    assert load_settings(path).agent_share == 0.2
    assert live.last_review_date() == "2026-12-20"
    assert not live.satellite_halted()
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `hedge_fund/live/account/review.py`:

```python
"""The agent-share review: evidence from the paper fund (and the live satellite), proposed, then applied by hand."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from hedge_fund.live.account.ledger import LiveLedger
from hedge_fund.live.account.settings import LiveSettings, save_settings
from hedge_fund.live.ledger import Ledger
from hedge_fund.live.report import MIN_REBALANCES_FOR_VERDICT

FIRST_SHARE, STEP, CAP = 0.2, 0.1, 0.5


class ReviewResult(BaseModel):
    window_start: str
    window_end: str
    paper_rebalances: int
    paper_excess: float              # paper fund return − benchmark return over the window
    live_excess: float | None        # satellite − core, if a satellite was held
    passed: bool
    old_share: float
    new_share: float


def propose_review(paper: Ledger, live: LiveLedger, settings: LiveSettings) -> ReviewResult:
    sessions = paper.plan_sessions()
    if not sessions:
        raise ValueError("the paper fund has no rebalances yet")
    start = live.last_review_date() or sessions[0]
    rebalances = [s for s in sessions if s >= start]
    if len(rebalances) < MIN_REBALANCES_FOR_VERDICT:
        raise ValueError(f"{len(rebalances)} of {MIN_REBALANCES_FOR_VERDICT} paper rebalances since {start}; review not due")
    rows = [r for r in paper.nav_rows() if r.date >= start]
    if len(rows) < 2:
        raise ValueError(f"the paper fund has fewer than two reconciled sessions since {start}")
    paper_excess = rows[-1].equity / rows[0].equity - rows[-1].benchmark_close / rows[0].benchmark_close
    live_excess = live.satellite_excess(since=start)
    passed = paper_excess > 0 and (live_excess is None or live_excess > 0)
    old = settings.agent_share
    if passed:
        new = FIRST_SHARE if old == 0 else min(CAP, old + STEP)
    else:
        new = max(0.0, old - STEP)
    return ReviewResult(window_start=start, window_end=rows[-1].date, paper_rebalances=len(rebalances),
                        paper_excess=round(paper_excess, 6), live_excess=None if live_excess is None else round(live_excess, 6),
                        passed=passed, old_share=old, new_share=round(new, 6))


def apply_review(result: ReviewResult, settings: LiveSettings, settings_path: Path, live: LiveLedger, *, today: str) -> None:
    save_settings(settings.model_copy(update={"agent_share": result.new_share}), settings_path)
    live.append_review(today, result.old_share, result.new_share, paper_excess=result.paper_excess,
                       live_excess=result.live_excess, passed=result.passed)
    live.clear_satellite_halt()
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** — `git add hedge_fund/live/account && git commit -m "Add the agent-share review"`

---

### Task 8: Report

**Files:** Create `hedge_fund/live/account/report.py`; Test `hedge_fund/live/account/test_report.py`

- [ ] **Step 1: Failing tests:**

```python
import pytest

from hedge_fund.brokers.alpaca import CashFlow
from hedge_fund.live.account.ledger import LiveLedger, LiveNavRow
from hedge_fund.live.account.report import build_live_report
from hedge_fund.live.account.settings import LiveSettings


def test_time_weighted_return_ignores_deposits(tmp_path):
    live = LiveLedger(tmp_path)
    live.add_flows([CashFlow(id="1", kind="CSD", date="2026-10-01", amount=100.0),
                    CashFlow(id="2", kind="CSD", date="2026-11-02", amount=100.0)])
    for d, eq, flow, close in [("2026-10-01", 100.0, 100.0, 500.0), ("2026-10-30", 110.0, 0.0, 525.0),
                               ("2026-11-02", 210.0, 100.0, 525.0), ("2026-11-30", 231.0, 0.0, 551.25)]:
        live.upsert_live_nav(LiveNavRow(date=d, equity=eq, cash=0, net_flow=flow, core_value=eq,
                                        satellite_value=0, core_close=close))
    r = build_live_report(live, LiveSettings())
    assert r.deposited == 200.0 and r.equity == 231.0 and r.gain == pytest.approx(31.0)
    assert r.time_weighted_return == pytest.approx(1.1 * 1.1 - 1)
    assert r.core_return == pytest.approx(551.25 / 500 - 1)


def test_empty_ledger(tmp_path):
    assert build_live_report(LiveLedger(tmp_path), LiveSettings()) is None
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `hedge_fund/live/account/report.py`:

```python
"""How the live account is doing, with deposits taken out of the return."""

from __future__ import annotations

from pydantic import BaseModel

from hedge_fund.live.account.ledger import LiveLedger
from hedge_fund.live.account.settings import LiveSettings


class LiveReport(BaseModel):
    start: str
    end: str
    deposited: float                 # deposits − withdrawals, all time
    equity: float
    gain: float                      # equity − deposited
    time_weighted_return: float      # unaffected by when money arrived
    core_return: float               # the core ETF alone over the same dates
    core_value: float
    satellite_value: float
    agent_share: float
    satellite_halted: bool
    satellite_vs_core: float | None  # since the latest review


def build_live_report(ledger: LiveLedger, settings: LiveSettings) -> LiveReport | None:
    rows = ledger.live_nav_rows()
    if not rows:
        return None
    twr = 1.0
    for prev, cur in zip(rows, rows[1:]):
        if prev.equity > 0:
            twr *= (cur.equity - cur.net_flow) / prev.equity
    last = rows[-1]
    review = ledger.last_review_date()
    return LiveReport(
        start=rows[0].date, end=last.date, deposited=round(ledger.total_deposited(), 2), equity=last.equity,
        gain=round(last.equity - ledger.total_deposited(), 2), time_weighted_return=round(twr - 1, 6),
        core_return=round(last.core_close / rows[0].core_close - 1, 6), core_value=last.core_value,
        satellite_value=last.satellite_value, agent_share=settings.agent_share,
        satellite_halted=ledger.satellite_halted(),
        satellite_vs_core=ledger.satellite_excess(review) if review else None,
    )
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** — `git add hedge_fund/live/account && git commit -m "Report the live account with deposits separated"`

---

### Task 9: `aihf-live` CLI, entry point, docs

**Files:** Create `hedge_fund/live/account/cli.py`; Test `hedge_fund/live/account/test_cli.py`; Modify `pyproject.toml`, `.env.example`, `README.md`

- [ ] **Step 1: Failing tests:**

```python
from hedge_fund.live.account.cli import build_parser, main
from hedge_fund.live.account.settings import load_settings


def test_parser():
    args = build_parser().parse_args(["submit", "--dry-run"])
    assert args.command == "submit" and args.dry_run


def test_init_writes_safe_settings_once(tmp_path, capsys):
    path = tmp_path / "live.yaml"
    assert main(["init", "--settings", str(path)]) == 0
    s = load_settings(path)
    assert s.confirm_live is False and s.agent_share == 0.0
    assert main(["init", "--settings", str(path)]) == 1          # never overwrites
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `hedge_fund/live/account/cli.py`:

```python
"""aihf-live — the real-money account: S&P 500 core + a satellite mirroring the paper fund.

    aihf-live init                     write a safe ~/.hedge-fund/live.yaml (confirm_live: false)
    aihf-live status                   account, settings, halts
    aihf-live run [--dry-run]          reconcile yesterday, then submit today
    aihf-live reconcile | submit [--dry-run]
    aihf-live report                   deposits, value, time-weighted return vs the core ETF
    aihf-live review [--apply]         propose (and apply) a new agent share

Real orders need live keys (ALPACA_LIVE_KEY_ID / ALPACA_LIVE_SECRET_KEY),
confirm_live: true, and one reviewed dry run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from hedge_fund.brokers.alpaca import AlpacaLiveClient
from hedge_fund.data.factory import open_data_client
from hedge_fund.data.sessions import NEW_YORK
from hedge_fund.live.account.ledger import LiveLedger
from hedge_fund.live.account.report import build_live_report
from hedge_fund.live.account.review import apply_review, propose_review
from hedge_fund.live.account.runner import reconcile_live, submit_live
from hedge_fund.live.account.settings import LiveSettings, load_settings, save_settings
from hedge_fund.live.launchd import notify
from hedge_fund.live.ledger import Ledger
from hedge_fund.live.runner import session_to_reconcile
from hedge_fund.paths import KILL_PATH, LIVE_SETTINGS_PATH
from hedge_fund.tui.keys import apply_credentials

logger = logging.getLogger("aihf-live")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aihf-live", description="Run the real-money core + satellite account.")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help)
        p.add_argument("--settings", default=str(LIVE_SETTINGS_PATH))
        return p

    command("init", "write a safe live.yaml (never overwrites)")
    command("status", "account, settings, halts")
    command("reconcile", "record the previous session's value and deposits")
    command("submit", "trade today").add_argument("--dry-run", action="store_true")
    command("run", "reconcile then submit").add_argument("--dry-run", action="store_true")
    command("report", "deposits, value, return vs the core ETF")
    command("review", "propose a new agent share").add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    apply_credentials()
    args = build_parser().parse_args(argv)
    path = Path(args.settings)
    if args.command == "init":
        if path.exists():
            print(f"{path} already exists; not overwriting")
            return 1
        save_settings(LiveSettings(), path)
        print(f"wrote {path} (confirm_live: false, agent_share: 0). Review it, then set confirm_live: true.")
        return 0
    ledger = LiveLedger.default()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True,
        handlers=[logging.StreamHandler(), logging.FileHandler(ledger.log_path(date.today().isoformat(), args.command))],
    )
    try:
        settings = load_settings(path)
        return _COMMANDS[args.command](args, settings, path, ledger)
    except Exception as exc:
        logger.exception("%s failed", args.command)
        notify(f"aihf-live {args.command} failed", str(exc))
        return 1


def _reconcile(args, settings, path, ledger) -> int:
    client = AlpacaLiveClient()
    session = session_to_reconcile(client, datetime.now(NEW_YORK))
    with open_data_client() as data:
        row = reconcile_live(settings, client, data, ledger, session=session)
    print(f"{session}: equity ${row.equity:,.2f} · core ${row.core_value:,.2f} · satellite ${row.satellite_value:,.2f}"
          f" · deposits ${row.net_flow:,.2f}")
    return 0


def _submit(args, settings, path, ledger) -> int:
    client = AlpacaLiveClient()
    with open_data_client() as data:
        result = submit_live(settings, client, data, ledger, Ledger.for_fund(settings.paper_fund),
                             now=datetime.now(NEW_YORK), dry_run=args.dry_run, kill_path=KILL_PATH)
    print(f"{result.session}: {result.status} {result.detail}")
    for t, w in sorted(result.target.items(), key=lambda kv: -abs(kv[1])):
        print(f"  target {t:6} {w:+.1%}")
    for o in result.orders:
        what = f"${o.dollars:,.2f}" if o.dollars is not None else (f"{o.qty:g} sh" if o.qty is not None else f"{o.shares} sh")
        print(f"  {o.side:4} {o.ticker:6} {what}")
    for r in result.results:
        if r.status == "rejected":
            print(f"  REJECTED {r.ticker}: {r.reason}")
    return 0


def _run(args, settings, path, ledger) -> int:
    try:
        _reconcile(args, settings, path, ledger)
    except Exception:
        logger.exception("reconcile failed; submitting anyway")
    return _submit(args, settings, path, ledger)


def _status(args, settings, path, ledger) -> int:
    account = AlpacaLiveClient().account()
    print(f"live account {account.status} · equity ${account.equity:,.2f} · cash ${account.cash:,.2f}")
    print(f"core {settings.core_ticker} · agent share {settings.agent_share:.0%} · shorts "
          f"{'on' if settings.shorts_enabled else 'off'} · confirm_live {settings.confirm_live}")
    print(f"kill switch {'ON' if KILL_PATH.exists() else 'off'} · satellite halted: {ledger.satellite_halted()}"
          f" · dry run done: {ledger.dry_run_done()}")
    return 0


def _report(args, settings, path, ledger) -> int:
    report = build_live_report(ledger, settings)
    print(report.model_dump_json(indent=2) if report else "no reconciled sessions yet")
    return 0


def _review(args, settings, path, ledger) -> int:
    result = propose_review(Ledger.for_fund(settings.paper_fund), ledger, settings)
    print(result.model_dump_json(indent=2))
    if args.apply:
        apply_review(result, settings, path, ledger, today=date.today().isoformat())
        print(f"agent share {result.old_share:.0%} → {result.new_share:.0%}")
    else:
        print("re-run with --apply to set this share")
    return 0


_COMMANDS = {"status": _status, "reconcile": _reconcile, "submit": _submit, "run": _run,
             "report": _report, "review": _review}


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Entry point and docs** — `pyproject.toml` under `[tool.poetry.scripts]` add `aihf-live = "hedge_fund.live.account.cli:main"`; reinstall with `uv pip install --python .venv/bin/python -e .`. Append to `.env.example`:

```
# Real-money Alpaca account (aihf-live). LIVE keys, separate from the paper
# keys above; the live client refuses a paper key and vice versa.
ALPACA_LIVE_KEY_ID=your-alpaca-live-key-id
ALPACA_LIVE_SECRET_KEY=your-alpaca-live-secret-key
```

Add a README section "Real-money account (core + satellite)" after "Paper trading on Alpaca": what `aihf-live` does, `init` → edit `live.yaml` → `run --dry-run` → `confirm_live: true` → daily `aihf-live run` after the paper command; the agent share rule; the user-side checks (UK availability, FX/transfer fees, W-8BEN, not an ISA — not tax advice); "educational project, not investment advice".

- [ ] **Step 5: Run** — `FINANCIAL_DATASETS_API_KEY= .venv/bin/python -m pytest hedge_fund -q && .venv/bin/aihf-live --help` → all pass; help lists the commands.
- [ ] **Step 6: Commit** — `git add hedge_fund/live/account pyproject.toml .env.example README.md && git commit -m "Add the aihf-live command"`

---

### Task 10: Agent model decision (operational)

- [ ] **Step 1:** `which ollama && ollama list` — if Ollama isn't installed, stop and tell the user (installing it and pulling a model is their call; models are several GB).
- [ ] **Step 2:** Run `.venv/bin/python -m hedge_fund.llm.compare --models <installed candidates> --n 200` against the saved Opus prompts.
- [ ] **Step 3:** Apply the spec's pre-committed bar (valid ≥ 98%, agreement ≥ 75%, opposite ≤ 5%, correlation ≥ 0.7, every agent ≥ 60%). Report the table; switch `HEDGE_FUND_LLM_MODEL` only with the user's go-ahead, and note the change date in the paper ledger's log.

### Task 11: Acceptance (needs the user's live account)

- [ ] **Step 1:** User opens/funds the live account and adds `ALPACA_LIVE_*` keys.
- [ ] **Step 2:** `aihf-live init`, review `~/.hedge-fund/live.yaml`, `aihf-live status`.
- [ ] **Step 3:** `aihf-live run --dry-run` — review the plan with the user (expect: all cash → core ETF).
- [ ] **Step 4:** User sets `confirm_live: true` and runs `aihf-live run` themselves.
