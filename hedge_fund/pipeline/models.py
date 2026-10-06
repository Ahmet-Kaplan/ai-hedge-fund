"""Pipeline records — the serialized truth of every cycle.

A CycleRecord captures one tick of the fund end to end: what the analysts
saw, what they said, which views were dropped before blending, how views
became weights, what risk clamped, what was ordered and filled, and what
the book looks like after. The ledger persists these and, on the next
live-clock paper run, seeds PaperBroker from the newest receipt so cash,
positions, and NAV carry forward. `fund why AAPL` will answer from them
alone.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from hedge_fund.brokers.models import Fill, Order
from hedge_fund.fund.spec import FundSpec
from hedge_fund.models import Signal
from hedge_fund.portfolio.construction import FlatReason
from hedge_fund.brokers.account import PreflightReport
from hedge_fund.reconciliation import ReconciliationReport
from hedge_fund.risk.limits import ClampEvent


class TickerSkip(BaseModel):
    """A requested name that could not be traded this cycle, and why."""

    ticker: str
    reason: str


class DroppedOutput(BaseModel):
    """An analyst view that was produced but excluded from the book.

    Abstained views (insufficient history, an LLM call/parse failure) stay
    on `StrategyRecord.signals` so the thesis is still auditable, and are
    listed here so the exclusion is explicit — never a silent omit.
    """

    ticker: str
    model: str
    strategy: str
    reason: str


class StrategyRecord(BaseModel):
    """One strategy's slice of a cycle: its analysts' views and its sleeve."""

    name: str
    slice: float                        # normalized capital slice of the fund
    signals: list[Signal]               # this strategy's analysts x tradeable tickers
    convictions: dict[str, float]       # blended views, pre-scaling
    weights: dict[str, float]           # the sleeve, before netting across strategies
    eligible_scores: dict[str, float] = Field(default_factory=dict)
    flat_reason: FlatReason | None = None
    final_contribution: dict[str, float] = Field(default_factory=dict)  # fraction of fund equity


class DecisionRecord(BaseModel):
    """An auditable assessment, without orders or broker accounting."""

    schema_version: Literal[2] = 2
    fund: str
    as_of: str
    spec: FundSpec
    universe: list[str]
    marks: dict[str, float]
    skipped: list[TickerSkip]
    strategies: list[StrategyRecord]
    target_weights: dict[str, float]
    clamps: list[ClampEvent]
    final_weights: dict[str, float]
    risk_scale_factor: float | None = None
    dropped: list[DroppedOutput] = Field(default_factory=list)


class PendingRunResult(BaseModel):
    """A saved proposal awaiting an explicit run with completed session data."""

    schema_version: Literal[2] = 2
    status: Literal["pending"] = "pending"
    execution_policy: Literal["next_close"] = "next_close"
    fund: str
    as_of: str
    proposal: DecisionRecord
    reason: str
    scheduled_execution_date: str | None = None


class ExecutionSlippage(BaseModel):
    """One fill measured against the price the order was sized on.

    Signed so the sign always means the same thing: positive is worse for the
    fund — it paid more to buy, or received less to sell. A backtest reports
    zero here by construction, which is exactly the cost a backtest cannot see.
    """

    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    reference_price: float              # the mark the fund sized against
    fill_price: float                   # what the venue charged
    per_share: float                    # signed: + is worse
    notional: float                     # per_share * quantity


class CycleRecord(BaseModel):
    """One tick of the fund, fully serialized — every stage's inputs and
    outputs. `model_dump_json()` round-trips; nothing about a decision
    lives anywhere else."""

    schema_version: Literal[2] = 2
    fund: str
    as_of: str
    spec: FundSpec                      # self-contained audit copy
    universe: list[str]                 # the tickers this cycle was asked to trade
    marks: dict[str, float]             # ticker -> close used for sizing and NAV
    skipped: list[TickerSkip]
    dropped: list[DroppedOutput] = Field(default_factory=list)
    strategies: list[StrategyRecord]    # every sleeve, incl. each thesis
    target_weights: dict[str, float]    # the NETTED book, pre-risk
    clamps: list[ClampEvent]
    final_weights: dict[str, float]     # post-risk
    equity_before: float
    cash_before: float
    orders: list[Order]
    fills: list[Fill]
    positions: dict[str, int]           # signed shares after fills
    cash: float
    nav: float                          # cash + sum(shares * mark)
    risk_scale_factor: float | None = None
    original_assessment: DecisionRecord | None = None
    refreshed_assessment: DecisionRecord | None = None
    execution_as_of: str | None = None
    # next_close: the backtest/paper path, priced and settled in one session
    # that has already ended. live_now: priced from the newest completed close
    # and sent to the venue immediately, which is the only way to trade during
    # a session.
    execution_policy: Literal["next_close", "live_now"] | None = None
    # What the venue actually charged against the price the fund sized on.
    # Zero for the in-process books; non-zero whenever a real venue is involved,
    # because the market moves between the assessment and the submission.
    slippage: list[ExecutionSlippage] = Field(default_factory=list)
    # What the broker held versus what the last receipt claimed, as of this
    # execution. None when no reference was supplied.
    reconciliation: "ReconciliationReport | None" = None
    # The venue's own account checks (buying power, short permission, day
    # trades) as of this execution. None for venues without an account.
    preflight: "PreflightReport | None" = None
