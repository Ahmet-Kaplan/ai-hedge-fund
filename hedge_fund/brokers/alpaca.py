"""Alpaca paper-trading client — the fund's link to a simulated brokerage account.

Two clients share one REST base. AlpacaPaperClient talks only to the paper
endpoint with the APCA_* paper keys. AlpacaLiveClient talks only to the live
endpoint with its own ALPACA_LIVE_* keys and refuses a paper key, so a paper
setting can never reach real money and vice versa.

Deliberately not a `Broker`: the protocol promises a complete fill or a raise,
and an order sent before the open fills hours later. The live path is
submit-then-reconcile instead (hedge_fund/live/runner.py).
"""

from __future__ import annotations

import os
from typing import Any, Literal

import requests
from pydantic import BaseModel

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"

# Alpaca answers a refused order with 403 (e.g. buying power) or 422 (e.g.
# not shortable). Those are rejections to record, not crashes.
_REJECTION_CODES = (403, 422)


class AlpacaError(Exception):
    """An Alpaca request failed (auth, network, server error, bad request)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class Account(BaseModel):
    cash: float
    equity: float
    last_equity: float
    status: str


class OrderResult(BaseModel):
    """What Alpaca did with one order: accepted (with its current status) or rejected."""

    client_order_id: str
    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    status: str                          # Alpaca's order status, or "rejected"
    order_id: str | None = None
    filled_qty: int = 0
    filled_avg_price: float | None = None
    reason: str | None = None


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


class AlpacaPaperClient(_AlpacaRest):
    """Thin REST client for the handful of paper-account endpoints the fund needs."""

    def __init__(
        self,
        key_id: str | None = None,
        secret_key: str | None = None,
        *,
        base_url: str = PAPER_BASE_URL,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        if base_url.rstrip("/") != PAPER_BASE_URL:
            raise ValueError(f"refusing Alpaca endpoint {base_url!r}: only {PAPER_BASE_URL} (paper trading) is allowed")
        key_id = key_id or os.environ.get("APCA_API_KEY_ID", "")
        secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY", "")
        if not key_id or not secret_key:
            raise ValueError("set APCA_API_KEY_ID and APCA_API_SECRET_KEY (Alpaca paper keys) in ~/.hedge-fund/.env")
        super().__init__(key_id, secret_key, PAPER_BASE_URL, timeout=timeout, session=session)

    def positions(self) -> dict[str, int]:
        """Signed whole shares per ticker. Negative = short.

        The fund trades whole shares, so any fractional remainder is left
        out here and reported by fractional_holdings() instead.
        """
        held: dict[str, int] = {}
        for row in self._request("GET", "/v2/positions"):
            shares = int(abs(float(row["qty"])))
            if shares:
                held[row["symbol"]] = -shares if row.get("side") == "short" else shares
        return held

    def fractional_holdings(self) -> dict[str, float]:
        """Signed fractional-share remainders the fund cannot trade or value (e.g. from manual trades)."""
        fractions: dict[str, float] = {}
        for row in self._request("GET", "/v2/positions"):
            qty = abs(float(row["qty"]))
            remainder = qty - int(qty)
            if remainder > 1e-9:
                fractions[row["symbol"]] = -remainder if row.get("side") == "short" else remainder
        return fractions

    def list_orders(self, after: str) -> list[OrderResult]:
        """Every order (any status) submitted after the ISO timestamp `after`."""
        rows = self._request("GET", "/v2/orders", params={
            "status": "all", "after": after, "limit": 500, "direction": "asc",
        })
        return [_order_result(row) for row in rows]

    def submit_order(self, ticker: str, side: Literal["buy", "sell"], quantity: int, client_order_id: str,
                     time_in_force: str = "day") -> OrderResult:
        """Submit a market order. A refused order comes back as status "rejected".

        "day" (default): sent before the open, it fills at the opening price —
        reliable on Alpaca's paper engine, which fills market-on-close ("cls")
        orders only sporadically.
        """
        body = {
            "symbol": ticker, "qty": str(quantity), "side": side, "type": "market",
            "time_in_force": time_in_force, "client_order_id": client_order_id,
        }
        try:
            row = self._request("POST", "/v2/orders", json=body)
        except AlpacaError as exc:
            if exc.status_code in _REJECTION_CODES:
                return OrderResult(client_order_id=client_order_id, ticker=ticker, side=side,
                                   quantity=quantity, status="rejected", reason=str(exc))
            raise
        return _order_result(row)


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


def _order_result(row: dict) -> OrderResult:
    avg = row.get("filled_avg_price")
    return OrderResult(
        client_order_id=row["client_order_id"],
        ticker=row["symbol"],
        side=row["side"],
        quantity=int(float(row.get("qty") or 0)),
        status=row["status"],
        order_id=row.get("id"),
        filled_qty=int(float(row.get("filled_qty") or 0)),
        filled_avg_price=float(avg) if avg is not None else None,
    )
