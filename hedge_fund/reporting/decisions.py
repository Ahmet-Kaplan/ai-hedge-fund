"""Decision reports — what the fund decided, per name, and why.

A pure consumer of the receipts the engine already writes (CycleRecord,
PendingRunResult, FundBacktestResult). Nothing here re-runs an analyst, reads
market data, or changes a weight: every field is derived from the record, so
a report can be rebuilt from a saved receipt at any time.

Per name the report answers four questions:

- action:   BUY / SELL / HOLD / NONE — the trade the cycle made (or, for a
            pending proposal, would make from an empty book)
- views:    each analyst's signal, confidence and written reasoning
- outcome:  selected (held after the cycle) or rejected
- why:      for rejected names, the concrete reason; for selected names,
            any risk clamp that cut the requested weight
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from hedge_fund.backtesting.fund import FundBacktestMetrics, FundBacktestResult
from hedge_fund.models import Signal
from hedge_fund.pipeline.models import CycleRecord, DecisionRecord, PendingRunResult

Action = Literal["BUY", "SELL", "HOLD", "NONE"]
ActionDetail = Literal["open", "add", "trim", "exit", "hold", "none"]

# Weights and cash below this are rounding noise, not positions.
_EPS = 1e-9


class AnalystView(BaseModel):
    """One analyst's opinion on one name, as recorded."""

    strategy: str
    analyst: str
    signal: str                      # bullish | neutral | bearish | abstained
    confidence: float | None = None  # 0-100; None when abstained
    value: float                     # signed conviction in [-1, +1]
    reasoning: str | None = None
    cached: bool | None = None


class TickerDecision(BaseModel):
    ticker: str
    action: Action
    action_detail: ActionDetail
    selected: bool                          # held after the cycle (or proposed > 0)
    rejection_reason: str | None = None     # set exactly when not selected
    notes: list[str] = Field(default_factory=list)  # clamps and other adjustments
    views: list[AnalystView] = Field(default_factory=list)
    convictions: dict[str, float] = Field(default_factory=dict)  # strategy -> blended view
    requested_weight: float = 0.0           # netted target before risk
    target_weight: float = 0.0              # after risk limits
    weight: float | None = None             # actual weight at execution marks (None if not executed)
    shares_before: int | None = None
    shares_after: int | None = None
    trade_shares: int | None = None         # signed; None if not executed
    price: float | None = None


class DecisionReport(BaseModel):
    """One cycle's decisions, with the resulting book."""

    fund: str
    as_of: str                              # assessment cutoff
    status: Literal["executed", "pending"]
    execution_as_of: str | None = None
    pending_reason: str | None = None
    capital: float
    max_position_pct: float
    rebalance: str
    benchmark: str
    universe: list[str]
    analysts: list[str]
    equity: float | None = None             # NAV after execution
    cash: float | None = None
    cash_weight: float                      # executed: cash / NAV; pending: 1 - sum(target weights)
    invested_weight: float
    decisions: list[TickerDecision]
    gross_clamp: str | None = None          # the portfolio-level clamp, if it fired

    @property
    def selected(self) -> list[TickerDecision]:
        return sorted((d for d in self.decisions if d.selected), key=lambda d: -(d.weight if d.weight is not None else d.target_weight))

    @property
    def rejected(self) -> list[TickerDecision]:
        return [d for d in self.decisions if not d.selected]


class Holding(BaseModel):
    ticker: str
    shares: int
    weight: float


class RebalanceSummary(BaseModel):
    as_of: str
    executed: str | None
    nav: float
    cash_weight: float
    holdings: list[Holding]
    buys: list[str]
    sells: list[str]


class BacktestReport(BaseModel):
    fund: str
    start: str
    end: str
    rebalance: str
    benchmark: str
    universe: list[str]
    capital: float
    metrics: FundBacktestMetrics
    dates: list[str]
    nav: list[float]
    benchmark_nav: list[float]
    rebalances: list[RebalanceSummary]
    latest: DecisionReport | None = None    # full detail for the last executed rebalance
    next_proposal: DecisionReport | None = None


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def cycle_report(record: CycleRecord) -> DecisionReport:
    """Decisions from an executed cycle: actions come from the fills."""
    after = dict(record.positions)
    before = dict(after)
    for fill in record.fills:
        before[fill.ticker] = before.get(fill.ticker, 0) - (fill.quantity if fill.side == "buy" else -fill.quantity)
    before = {t: s for t, s in before.items() if s}
    decisions = []
    for ticker in _tickers(record, before, after):
        b, a = before.get(ticker, 0), after.get(ticker, 0)
        mark = record.marks.get(ticker)
        weight = a * mark / record.nav if mark else 0.0
        d = _decide(record, ticker, selected=a > 0)
        action, detail = _action(b, a)
        d.action, d.action_detail = action, detail
        d.shares_before, d.shares_after, d.trade_shares = b, a, a - b
        d.weight, d.price = weight, mark
        if not d.selected and d.rejection_reason is None:
            d.rejection_reason = _below_one_share(d, mark, record.equity_before)
        decisions.append(d)
    invested = sum(d.weight or 0.0 for d in decisions)
    return _report(record, decisions, status="executed", execution_as_of=record.execution_as_of,
                   equity=record.nav, cash=record.cash, cash_weight=record.cash / record.nav, invested_weight=invested)


def proposal_report(proposal: DecisionRecord, reason: str | None = None) -> DecisionReport:
    """Decisions from an assessment that has not executed. Actions are stated
    relative to an empty book — a run starts from the mandate's cash."""
    decisions = []
    for ticker in _tickers(proposal, {}, {}):
        target = proposal.final_weights.get(ticker, 0.0)
        d = _decide(proposal, ticker, selected=target > _EPS)
        d.action, d.action_detail = ("BUY", "open") if d.selected else ("NONE", "none")
        d.price = proposal.marks.get(ticker)
        if d.selected and d.price and target * proposal.spec.capital < d.price:
            d.notes.append(f"target ${target * proposal.spec.capital:,.0f} at ${proposal.spec.capital:,.0f} capital is below one share "
                           f"at ${d.price:,.2f} — likely to round to zero shares at execution")
        decisions.append(d)
    invested = sum(max(w, 0.0) for w in proposal.final_weights.values())
    return _report(proposal, decisions, status="pending", pending_reason=reason, cash_weight=max(0.0, 1.0 - invested), invested_weight=invested)


def pending_report(result: PendingRunResult) -> DecisionReport:
    return proposal_report(result.proposal, result.reason)


def backtest_report(result: FundBacktestResult) -> BacktestReport:
    rebalances = []
    for record in result.records:
        cycle = cycle_report(record)
        rebalances.append(RebalanceSummary(
            as_of=record.as_of, executed=record.execution_as_of, nav=record.nav, cash_weight=cycle.cash_weight,
            holdings=[Holding(ticker=d.ticker, shares=d.shares_after or 0, weight=d.weight or 0.0) for d in cycle.selected],
            buys=[d.ticker for d in cycle.decisions if d.action == "BUY"],
            sells=[d.ticker for d in cycle.decisions if d.action == "SELL"],
        ))
    return BacktestReport(
        fund=result.fund, start=result.start, end=result.end, rebalance=result.rebalance, benchmark=result.benchmark,
        universe=result.universe, capital=result.capital, metrics=result.metrics, dates=result.dates, nav=result.nav,
        benchmark_nav=result.benchmark_nav, rebalances=rebalances,
        latest=cycle_report(result.records[-1]) if result.records else None,
        next_proposal=pending_report(result.pending[-1]) if result.pending else None,
    )


def load_receipt(data: dict) -> DecisionReport | BacktestReport:
    """Build the right report from a receipt's parsed JSON."""
    if "metrics" in data and "records" in data:
        return backtest_report(FundBacktestResult.model_validate(data))
    if data.get("status") == "pending":
        return pending_report(PendingRunResult.model_validate(data))
    if "fills" in data:
        return cycle_report(CycleRecord.model_validate(data))
    raise ValueError("not a recognised receipt: expected a run record, a pending proposal, or a backtest result")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _tickers(record: DecisionRecord | CycleRecord, before: dict, after: dict) -> list[str]:
    """Every name the cycle touched: the universe plus anything held."""
    seen = list(record.universe)
    for t in [*before, *after]:
        if t not in seen:
            seen.append(t)
    return seen


def _decide(record: DecisionRecord | CycleRecord, ticker: str, *, selected: bool) -> TickerDecision:
    views = [_view(s.name, sig) for s in record.strategies for sig in s.signals if sig.ticker == ticker]
    convictions = {s.name: s.convictions[ticker] for s in record.strategies if ticker in s.convictions}
    d = TickerDecision(
        ticker=ticker, action="NONE", action_detail="none", selected=selected, views=views, convictions=convictions,
        requested_weight=record.target_weights.get(ticker, 0.0), target_weight=record.final_weights.get(ticker, 0.0),
    )
    for clamp in record.clamps:
        if clamp.ticker == ticker:
            d.notes.append(f"capped at {abs(clamp.after):.1%} by {clamp.limit} (requested {abs(clamp.before):.1%})")
    if not selected:
        d.rejection_reason = _rejection(record, ticker, d)
    return d


def _view(strategy: str, signal: Signal) -> AnalystView:
    meta = signal.metadata
    if meta.get("abstained") is True:
        return AnalystView(strategy=strategy, analyst=signal.model_name, signal="abstained", value=signal.value,
                           reasoning=meta.get("abstain_reason") or signal.reasoning)
    label = meta.get("signal") or ("bullish" if signal.value > 0 else "bearish" if signal.value < 0 else "neutral")
    confidence = meta.get("confidence", abs(signal.value) * 100)
    return AnalystView(strategy=strategy, analyst=signal.model_name, signal=label, confidence=confidence,
                       value=signal.value, reasoning=signal.reasoning, cached=meta.get("cached"))


def _rejection(record, ticker: str, d: TickerDecision) -> str | None:
    """Why a name ended with no position. None when the only explanation is
    share rounding, which needs execution marks (see _below_one_share)."""
    skip = next((s for s in record.skipped if s.ticker == ticker), None)
    if skip is not None:
        return f"not assessed: {skip.reason}"
    if not d.views:
        return "no analyst covered this name"
    voting = [v for v in d.views if v.signal != "abstained"]
    if not voting:
        reasons = sorted({v.reasoning or "no reason given" for v in d.views})
        return "no view — every analyst abstained: " + "; ".join(reasons)
    if all(c <= _EPS for c in d.convictions.values()):
        views = ", ".join(f"{v.analyst} {v.signal}" + (f" ({v.confidence:.0f})" if v.confidence is not None else "") for v in voting)
        return f"no positive conviction — {views}; a long-only book holds no position"
    flat = [s.flat_reason for s in record.strategies if s.flat_reason and ticker in s.convictions]
    if flat and d.target_weight <= _EPS:
        return f"strategy targeted no positions ({', '.join(sorted(set(flat)))})"
    if d.target_weight <= _EPS:
        return "target weight is zero after risk limits"
    return None


def _below_one_share(d: TickerDecision, mark: float | None, equity: float) -> str:
    if mark and d.target_weight > _EPS:
        return f"target {d.target_weight:.2%} (${d.target_weight * equity:,.0f}) is below one share at ${mark:,.2f}"
    return "no position after execution"


def _action(before: int, after: int) -> tuple[Action, ActionDetail]:
    if after > before:
        return "BUY", "open" if before == 0 else "add"
    if after < before:
        return "SELL", "exit" if after == 0 else "trim"
    return ("HOLD", "hold") if after else ("NONE", "none")


def _report(record, decisions, *, status, cash_weight, invested_weight, execution_as_of=None, pending_reason=None, equity=None, cash=None) -> DecisionReport:
    spec = record.spec
    gross = next((c for c in record.clamps if c.ticker is None), None)
    return DecisionReport(
        fund=record.fund, as_of=record.as_of, status=status, execution_as_of=execution_as_of, pending_reason=pending_reason,
        capital=spec.capital, max_position_pct=spec.risk.max_position_pct, rebalance=spec.rebalance, benchmark=spec.benchmark,
        universe=list(record.universe), analysts=list(dict.fromkeys(m.name for s in spec.strategies for m in s.models)),
        equity=equity, cash=cash, cash_weight=cash_weight, invested_weight=invested_weight, decisions=decisions,
        gross_clamp=f"gross exposure scaled from {gross.before:.1%} to {gross.after:.1%}" if gross else None,
    )
