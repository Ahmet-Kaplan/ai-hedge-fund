"""Daily systematic backtester with a full audit trail.

    MarketPanel -> Strategy -> Signals -> (Ensemble) -> Portfolio -> Risk
      -> Orders -> ExecutionModel -> Fills -> Ledger -> Performance

One loop over the benchmark's sessions. At each session S, in this order:

    1. corporate actions effective on S (splits, then ex-date dividends)
    2. orders decided at the previous decision session fill at S's open or
       close (explicit FillTiming) — sells first, buys limited by cash
    3. held names that have stopped trading are liquidated at their last
       tradable close once untradable for `delist_grace` sessions
    4. mark to market at S's close; update drawdown / daily-loss state
    5. on a rebalance session: strategies read panel.as_of(S), the
       ensemble/portfolio/risk stages produce targets, and orders are queued
       for the next session. Nothing decided on S can fill on S.

Every fill is written to `trades` with the decision session, the data
timestamp, strategy versions and config hashes, the signal contributions,
the risk interventions for that name, the expected (reference) price, the
simulated fill price and its cost breakdown.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from hedge_fund.backtesting.fund import rebalance_grid
from hedge_fund.core.instruments import InstrumentRegistry
from hedge_fund.core.orders import FillEvent, OrderRequest, OrderStatus
from hedge_fund.systematic import metrics
from hedge_fund.systematic.benchmark import benchmark_curve
from hedge_fund.systematic.ensemble import Ensemble, EnsembleConfig, StrategyEvidence
from hedge_fund.systematic.execution import CostModel, FillTiming, SimulatedExecution
from hedge_fund.systematic.ledger import Ledger
from hedge_fund.systematic.portfolio import PortfolioConfig, target_weights
from hedge_fund.systematic.risk import RiskConfig, RiskEngine, RiskState


class BacktestConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: str
    end: str
    capital: float = Field(100_000.0, gt=0)
    rebalance: str = "monthly"
    timing: FillTiming
    costs: CostModel = CostModel()
    risk: RiskConfig = RiskConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    benchmark: str = "SPY"
    delist_grace: int = Field(5, ge=1)
    rf_annual: float = 0.0

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class BacktestResult:
    config_hash: str
    strategies: list[dict]
    sessions: list[str]
    equity: pd.Series
    exposure: pd.Series
    benchmark_total_return: pd.Series
    benchmark_price_return: pd.Series
    metrics: dict[str, float]
    decisions: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    liquidations: list[dict] = field(default_factory=list)
    reconciliation_error: float = 0.0
    halted: str | None = None

    def to_dict(self) -> dict:
        return {
            "config_hash": self.config_hash, "strategies": self.strategies,
            "sessions": self.sessions, "equity": [round(x, 6) for x in self.equity],
            "exposure": [round(x, 6) for x in self.exposure],
            "benchmark_total_return": [round(x, 6) for x in self.benchmark_total_return],
            "benchmark_price_return": [round(x, 6) for x in self.benchmark_price_return],
            "metrics": self.metrics, "decisions": self.decisions, "trades": self.trades,
            "rejected": self.rejected, "liquidations": self.liquidations,
            "reconciliation_error": self.reconciliation_error, "halted": self.halted,
        }


class SystematicBacktester:
    def __init__(self, panel, strategies: list, config: BacktestConfig, *,
                 evidence: dict[str, StrategyEvidence] | None = None, regime=None,
                 instruments: InstrumentRegistry | None = None, kill_switch_on: str | None = None) -> None:
        if not strategies:
            raise ValueError("at least one strategy is required")
        if len(strategies) > 1 and evidence is None:
            raise ValueError("combining strategies needs validation evidence for the ensemble")
        self.panel, self.strategies, self.config = panel, strategies, config
        self.evidence, self.regime = evidence, regime
        self.instruments = instruments or InstrumentRegistry()
        self.execution = SimulatedExecution(panel, config.costs, config.timing, self.instruments)
        self.risk = RiskEngine(config.risk)
        self.ensemble = Ensemble(config.ensemble)
        self.kill_switch_on = kill_switch_on        # test/drill hook: engage the kill switch from this session

    # ------------------------------------------------------------------

    def _scores(self, view, signals: dict[str, list], regime_state):
        if self.evidence is None:                        # single strategy, no ensemble
            (name, sigs), = signals.items()
            scores = {s.ticker: s.value for s in sigs if not s.metadata.get("abstained")}
            detail = {t: {"decision": "TRADE", "score": v, "contributions": {name: v}} for t, v in scores.items()}
            return scores, detail, {name: 1.0}
        decision = self.ensemble.combine(view.session, signals, self.evidence, regime_state)
        detail = {t: d.model_dump() for t, d in decision.instruments.items()}
        return decision.scores(), detail, decision.strategy_weights

    def _marks(self, view, symbols) -> dict[str, float]:
        closes = view.bars("close", tickers=sorted(symbols), tradable_only=False) if symbols else pd.DataFrame()
        tradable = view.tradable(tickers=sorted(symbols)) if symbols else pd.DataFrame()
        out = {}
        for s in symbols:
            good = closes[s].where(tradable[s]).dropna()
            out[s] = float(good.iloc[-1]) if len(good) else float(closes[s].dropna().iloc[-1])
        return out

    def run(self) -> BacktestResult:
        c = self.config
        all_sessions = [s for s in self.panel.sessions_through(c.end) if s >= c.start]
        if len(all_sessions) < 2:
            raise ValueError("backtest window needs at least two sessions")
        rebalance_days = set(rebalance_grid(all_sessions, c.rebalance))
        ledger = Ledger(cash=c.capital, instruments=self.instruments, allow_short=c.risk.allow_short)
        state = RiskState(peak_equity=c.capital, day_start_equity=c.capital)
        pending: list[OrderRequest] = []
        untradable_run: dict[str, int] = {}
        equity, exposure, decisions, trades, rejected, liquidations = [], [], [], [], [], []
        last_equity = c.capital

        for session in all_sessions:
            view = self.panel.as_of(session)
            ledger.apply_corporate_actions(view)
            if pending:
                splits = view.bars("split", lookback=1)
                pending = [self._split_adjust(o, splits) for o in pending]
                trades_now, rej = self._execute(pending, session, ledger)
                trades.extend(trades_now)
                rejected.extend(rej)
                pending = []

            tradable = view.tradable(lookback=1)
            for sym in sorted(ledger.positions):
                ok = sym in tradable.columns and bool(tradable[sym].iloc[-1])
                untradable_run[sym] = 0 if ok else untradable_run.get(sym, 0) + 1
                if untradable_run[sym] >= c.delist_grace:
                    liquidations.append(self._liquidate(sym, view, ledger))
                    untradable_run.pop(sym, None)

            marks = self._marks(view, ledger.positions)
            eq = ledger.equity(marks)
            state.update(last_equity, new_session=True)      # day starts at the prior close
            state.update(eq, new_session=False)
            if self.kill_switch_on and session >= self.kill_switch_on:
                state.kill_switch = True
            equity.append(eq)
            exposure.append(sum(abs(self.instruments.get(s).notional(q, marks[s])) for s, q in ledger.positions.items()) / eq
                            if eq > 0 else 0.0)
            last_equity = eq

            if session in rebalance_days and session != all_sessions[-1]:
                record, pending = self._decide(view, ledger, marks, eq, state)
                decisions.append(record)

        sessions = all_sessions
        eq_series = pd.Series(equity, index=sessions, name="equity")
        bench_tr = benchmark_curve(self.panel, c.benchmark, sessions, c.capital, total_return=True)
        bench_px = benchmark_curve(self.panel, c.benchmark, sessions, c.capital, total_return=False)
        stats = metrics.summarize(eq_series, fills=ledger.fills, exposure=pd.Series(exposure),
                                  rf_annual=c.rf_annual)
        stats["benchmark_total_return"] = metrics.total_return(bench_tr)
        stats["excess_return_vs_total_return_benchmark"] = stats["total_return"] - stats["benchmark_total_return"]
        stats["dividends"] = ledger.total("dividend")
        return BacktestResult(
            config_hash=c.config_hash(), strategies=[s.spec() for s in self.strategies],
            sessions=sessions, equity=eq_series, exposure=pd.Series(exposure, index=sessions),
            benchmark_total_return=bench_tr, benchmark_price_return=bench_px, metrics=stats,
            decisions=decisions, trades=trades, rejected=rejected, liquidations=liquidations,
            reconciliation_error=ledger.reconcile(), halted=state.halted_reason or (
                "kill switch engaged" if state.kill_switch else None),
        )

    # ------------------------------------------------------------------

    def _decide(self, view, ledger: Ledger, marks, eq, state):
        signals = {s.name: s.generate(view) for s in self.strategies}
        regime_state = self.regime.classify(view) if self.regime is not None else None
        scores, detail, strategy_weights = self._scores(view, signals, regime_state)
        targets = target_weights(scores, view, self.config.portfolio)
        current = {s: self.instruments.get(s).notional(q, marks[s]) / eq for s, q in ledger.positions.items()} if eq > 0 else {}
        risk = self.risk.apply(targets, equity=eq, current=current, state=state, view=view)
        closes = self._marks(view, set(risk.weights) | set(ledger.positions))
        orders = []
        interventions: dict[str, list] = {}
        for i in risk.interventions:
            for t in ([i["ticker"]] if "ticker" in i else i.get("members", ["*"])):
                interventions.setdefault(t, []).append(i["check"])
        versions = {s.name: {"version": s.version, "config_hash": s.config_hash()} for s in self.strategies}
        for sym in sorted(set(risk.weights) | set(ledger.positions)):
            price = closes.get(sym)
            if not price or not math.isfinite(price):
                continue
            inst = self.instruments.get(sym)
            target_q = inst.round_quantity(risk.weights.get(sym, 0.0) * eq / (price * inst.multiplier))
            delta = target_q - ledger.quantity(sym)
            if abs(delta) < 1e-12:
                continue
            reason = {"data_as_of": view.session, "target_weight": risk.weights.get(sym, 0.0),
                      "pre_risk_weight": targets.get(sym, 0.0), "ensemble": detail.get(sym),
                      "risk_checks": interventions.get(sym, []) + interventions.get("*", []),
                      "strategies": versions, "regime": regime_state.model_dump() if regime_state else None}
            orders.append(OrderRequest(symbol=sym, side="buy" if delta > 0 else "sell", quantity=abs(delta),
                                       decision_session=view.session, reference_price=price,
                                       strategy="+".join(sorted(versions)), reason=reason).with_client_id())
        record = {"session": view.session, "equity": eq, "regime": regime_state.model_dump() if regime_state else None,
                  "strategy_weights": strategy_weights,
                  "signals": {n: [{"ticker": s.ticker, "value": s.value, "abstained": bool(s.metadata.get("abstained"))}
                                  for s in sigs] for n, sigs in signals.items()},
                  "ensemble": detail, "pre_risk_targets": targets, "risk": {
                      "weights": risk.weights, "interventions": risk.interventions, "halted": risk.halted,
                      "reason": risk.reason},
                  "orders": [o.client_order_id for o in orders]}
        return record, orders

    @staticmethod
    def _split_adjust(order: OrderRequest, splits: pd.DataFrame) -> OrderRequest:
        if order.symbol not in splits.columns or not len(splits):
            return order
        f = float(splits[order.symbol].iloc[-1])
        if f == 1.0 or not math.isfinite(f) or f <= 0:
            return order
        return order.model_copy(update={"quantity": order.quantity * f, "reference_price": order.reference_price / f})

    def _execute(self, orders: list[OrderRequest], session: str, ledger: Ledger):
        trades, rejected = [], []
        for order in sorted(orders, key=lambda o: (o.side != "sell", o.symbol)):
            if order.side == "sell" and not self.config.risk.allow_short:
                held = ledger.quantity(order.symbol)
                if held <= 0:
                    continue
                order = order.model_copy(update={"quantity": min(order.quantity, held)})
            state = self.execution.execute(order, session)
            if order.side == "buy" and state.fills:
                state = self._fit_cash(order, state, session, ledger)
            if state.status is OrderStatus.REJECTED or not state.fills:
                rejected.append({"session": session, "order": order.client_order_id, "symbol": order.symbol,
                                 "side": order.side, "quantity": order.quantity, "reason": state.reject_reason})
                continue
            for f in state.fills:
                ledger.apply_fill(f)
                trades.append(self._audit(order, f, state.status.value, state.reject_reason))
        return trades, rejected

    def _fit_cash(self, order, state, session, ledger):
        f = state.fills[0]
        inst = self.instruments.get(order.symbol)
        need = inst.notional(f.quantity, f.price) + f.commission
        if need <= ledger.cash + 1e-9:
            return state
        per_unit = inst.notional(1.0, f.price) * (1 + 1e-9)
        affordable = inst.round_quantity(max(ledger.cash - f.commission, 0.0) / per_unit)
        if affordable <= 0:
            state.fills.clear()
            state.status = OrderStatus.REJECTED
            state.reject_reason = "insufficient cash"
            return state
        return self.execution.execute(order.model_copy(update={"quantity": affordable}), session)

    @staticmethod
    def _audit(order: OrderRequest, fill: FillEvent, status: str, note: str | None) -> dict:
        return {
            "client_order_id": order.client_order_id, "symbol": fill.symbol, "side": fill.side,
            "quantity": fill.quantity, "decision_session": order.decision_session, "fill_session": fill.session,
            "data_as_of": order.reason.get("data_as_of"), "expected_price": order.reference_price,
            "fill_reference_price": fill.mid_price, "fill_price": fill.price,
            "commission": fill.commission, "spread_cost": fill.spread_cost, "impact_cost": fill.impact_cost,
            "status": status, "note": note, "target_weight": order.reason.get("target_weight"),
            "pre_risk_weight": order.reason.get("pre_risk_weight"), "risk_checks": order.reason.get("risk_checks"),
            "ensemble": order.reason.get("ensemble"), "strategies": order.reason.get("strategies"),
        }

    def _liquidate(self, sym: str, view, ledger: Ledger) -> dict:
        closes = view.bars("close", tickers=[sym], tradable_only=True)[sym].dropna()
        price = float(closes.iloc[-1]) if len(closes) else 0.0
        q = ledger.quantity(sym)
        if price > 0:
            ledger.apply_fill(FillEvent(client_order_id=f"delist-{sym}-{view.session}", symbol=sym,
                                        side="sell" if q > 0 else "buy", quantity=abs(q), price=price,
                                        session=view.session, mid_price=price))
        else:
            ledger.positions.pop(sym, None)
        return {"session": view.session, "symbol": sym, "quantity": q, "price": price,
                "last_tradable_session": closes.index[-1] if len(closes) else None}
