"""The platform's seams, as structural protocols.

    MarketDataProvider -> MarketPanel / AsOfView -> Strategy -> Signal
      -> Ensemble -> PortfolioConstructor -> RiskEngine -> ExecutionModel / Broker
      -> Fill -> Ledger -> Reconciliation

No protocol names an asset class, a vendor or a broker. The existing
equities stack is one implementation: `CompositeDataClient` (Tiingo prices +
SEC EDGAR) already satisfies `MarketDataProvider`; `SimBroker` keeps serving
the Buffett pipeline; `hedge_fund.systematic` provides the rest.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hedge_fund.core.orders import FillEvent, OrderRequest, OrderState
from hedge_fund.data.models import Price


@runtime_checkable
class MarketDataProvider(Protocol):
    """Daily bars and corporate actions for any instrument the vendor knows.

    Contract: prices are split-adjusted consistently over the vendor's whole
    stored history; `split_events`/`dividends` let consumers restate them to
    any as-of basis. Infrastructure failures raise; empty means no data.
    """

    def get_prices(self, ticker: str, start_date: str, end_date: str, **kwargs) -> list[Price]: ...

    def dividends(self, ticker: str, start_date: str, end_date: str) -> dict[str, float]: ...

    def split_events(self, ticker: str, start_date: str, end_date: str) -> dict[str, float]: ...


@runtime_checkable
class Strategy(Protocol):
    """Forms views from an as-of view only. No sizing, no broker, no clock."""

    name: str
    version: str

    def config_hash(self) -> str: ...

    def generate(self, view) -> list: ...           # AsOfView -> list[Signal]


@runtime_checkable
class ExecutionModel(Protocol):
    """Turns an order decided at session D into fills at a later session."""

    def execute(self, order: OrderRequest, session: str, panel) -> tuple[OrderState, list[FillEvent]]: ...


@runtime_checkable
class TradingVenue(Protocol):
    """A broker with a real order lifecycle (paper or live)."""

    def submit(self, order: OrderRequest) -> OrderState: ...

    def cancel(self, client_order_id: str) -> OrderState: ...

    def order(self, client_order_id: str) -> OrderState: ...

    def positions(self) -> dict[str, float]: ...

    def cash(self) -> float: ...


@runtime_checkable
class RiskGate(Protocol):
    """Has the last word: may shrink or reject any target or order."""

    def apply(self, targets: dict[str, float], context) -> object: ...
