"""Real-broker adapters — integration-ready, paper endpoints only, live locked.

Neither adapter can reach a live trading endpoint. Live access would need all
of: an adapter constructed with `live=True`, the environment variable
`AIHF_ALLOW_LIVE_TRADING=1`, and a code change that removes the
`LIVE_TRADING_ENABLED = False` constant below — i.e. a reviewed human
decision, never a runtime switch. Without credentials the adapters raise
`BrokerNotConfigured` before any network call.

Alpaca: REST v2 against the paper host; the HTTP transport is injectable so
tests exercise payloads without a network.
IBKR: requires a running IB Gateway/TWS in paper mode and the `ib_insync`
(or `ib_async`) package, neither of which is installed here.
"""

from __future__ import annotations

import os
from typing import Callable

from hedge_fund.core.orders import OrderRequest, OrderState, OrderStatus

LIVE_TRADING_ENABLED = False          # hard lock; changing it is a human, reviewed decision


class LiveTradingDisabled(PermissionError):
    pass


class BrokerNotConfigured(RuntimeError):
    pass


def _refuse_live(live: bool) -> None:
    if live and not (LIVE_TRADING_ENABLED and os.environ.get("AIHF_ALLOW_LIVE_TRADING") == "1"):
        raise LiveTradingDisabled(
            "live trading is locked in this codebase (LIVE_TRADING_ENABLED=False); "
            "only paper endpoints are available")


class AlpacaAdapter:
    PAPER_URL = "https://paper-api.alpaca.markets"
    KEY_ENV, SECRET_ENV = "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"

    def __init__(self, *, live: bool = False, transport: Callable | None = None) -> None:
        _refuse_live(live)
        key, secret = os.environ.get(self.KEY_ENV), os.environ.get(self.SECRET_ENV)
        if not key or not secret:
            raise BrokerNotConfigured(f"set {self.KEY_ENV} and {self.SECRET_ENV} for the Alpaca PAPER account")
        self.base_url = self.PAPER_URL
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._transport = transport or self._requests_transport

    def _requests_transport(self, method: str, url: str, **kw):
        import requests
        resp = requests.request(method, url, headers=self._headers, timeout=15, **kw)
        resp.raise_for_status()
        return resp.json()

    def _call(self, method: str, path: str, **kw):
        url = self.base_url + path
        if not url.startswith(self.PAPER_URL):
            raise LiveTradingDisabled(f"refusing non-paper URL {url}")
        return self._transport(method, url, **kw)

    @staticmethod
    def order_payload(order: OrderRequest) -> dict:
        tif = {"day": "day", "gtc": "gtc", "opg": "opg", "cls": "cls"}[order.time_in_force]
        typ = {"market": "market", "limit": "limit", "moo": "market", "moc": "market"}[order.order_type]
        if order.order_type == "moo":
            tif = "opg"
        if order.order_type == "moc":
            tif = "cls"
        payload = {"symbol": order.symbol, "side": order.side, "qty": str(order.quantity), "type": typ,
                   "time_in_force": tif, "client_order_id": order.with_client_id().client_order_id}
        if order.order_type == "limit":
            payload["limit_price"] = str(order.limit_price)
        return payload

    def submit(self, order: OrderRequest) -> OrderState:
        order = order.with_client_id()
        body = self._call("POST", "/v2/orders", json=self.order_payload(order))
        state = OrderState(request=order)
        status = str(body.get("status", "accepted"))
        state.transition(OrderStatus.REJECTED if status == "rejected" else OrderStatus.ACCEPTED,
                         body.get("reject_reason"))
        return state

    def cancel(self, client_order_id: str):
        return self._call("DELETE", f"/v2/orders:by_client_order_id?client_order_id={client_order_id}")

    def positions(self) -> dict[str, float]:
        return {p["symbol"]: float(p["qty"]) for p in self._call("GET", "/v2/positions")}

    def cash(self) -> float:
        return float(self._call("GET", "/v2/account")["cash"])


class IBKRAdapter:
    """Interactive Brokers via a local IB Gateway in paper mode (port 4002)."""

    PAPER_PORT = 4002

    def __init__(self, *, live: bool = False, host: str = "127.0.0.1", port: int = PAPER_PORT) -> None:
        _refuse_live(live)
        if port != self.PAPER_PORT:
            raise LiveTradingDisabled(f"port {port} is not the paper gateway port {self.PAPER_PORT}")
        try:
            import ib_insync  # noqa: F401
        except ImportError as exc:
            raise BrokerNotConfigured("ib_insync is not installed and no IB Gateway is configured") from exc
        self.host, self.port = host, port
        raise BrokerNotConfigured("IB Gateway connection is not configured in this environment")
