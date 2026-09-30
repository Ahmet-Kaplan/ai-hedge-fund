"""One decision step, shared by the backtester and paper trading.

    view -> strategies -> signals -> (ensemble) -> portfolio -> risk -> orders

Given the as-of view of a decision session, current positions, marks and
equity, returns the decision record and the orders to queue for a later
session. Identical code for simulation and paper, so paper results test the
same logic that was backtested.
"""

from __future__ import annotations

import math


from hedge_fund.core.instruments import InstrumentRegistry
from hedge_fund.core.orders import OrderRequest
from hedge_fund.systematic.ensemble import Ensemble, StrategyEvidence
from hedge_fund.systematic.portfolio import target_weights
from hedge_fund.systematic.risk import RiskEngine, RiskState


def marks_for(view, symbols) -> dict[str, float]:
    """Last tradable close on or before the view's session (else last print)."""
    symbols = sorted(symbols)
    if not symbols:
        return {}
    closes = view.bars("close", tickers=symbols, tradable_only=False)
    tradable = view.tradable(tickers=symbols)
    out = {}
    for s in symbols:
        good = closes[s].where(tradable[s]).dropna()
        out[s] = float(good.iloc[-1]) if len(good) else float(closes[s].dropna().iloc[-1])
    return out


class DecisionEngine:
    def __init__(self, strategies: list, config, *, evidence: dict[str, StrategyEvidence] | None = None,
                 regime=None, instruments: InstrumentRegistry | None = None) -> None:
        if not strategies:
            raise ValueError("at least one strategy is required")
        if len(strategies) > 1 and evidence is None:
            raise ValueError("combining strategies needs validation evidence for the ensemble")
        self.strategies, self.config, self.evidence, self.regime = strategies, config, evidence, regime
        self.instruments = instruments or InstrumentRegistry()
        self.risk = RiskEngine(config.risk)
        self.ensemble = Ensemble(config.ensemble)

    def _scores(self, view, signals: dict[str, list], regime_state):
        if self.evidence is None:                        # single strategy, no ensemble
            (name, sigs), = signals.items()
            scores = {s.ticker: s.value for s in sigs if not s.metadata.get("abstained")}
            detail = {t: {"decision": "TRADE", "score": v, "contributions": {name: v}} for t, v in scores.items()}
            return scores, detail, {name: 1.0}
        decision = self.ensemble.combine(view.session, signals, self.evidence, regime_state)
        detail = {t: d.model_dump() for t, d in decision.instruments.items()}
        return decision.scores(), detail, decision.strategy_weights

    def decide(self, view, positions: dict[str, float], marks: dict[str, float], eq: float, state: RiskState):
        signals = {s.name: s.generate(view) for s in self.strategies}
        regime_state = self.regime.classify(view) if self.regime is not None else None
        scores, detail, strategy_weights = self._scores(view, signals, regime_state)
        targets = target_weights(scores, view, self.config.portfolio)
        current = {s: self.instruments.get(s).notional(q, marks[s]) / eq for s, q in positions.items()} if eq > 0 else {}
        risk = self.risk.apply(targets, equity=eq, current=current, state=state, view=view)
        closes = marks_for(view, set(risk.weights) | set(positions))
        orders = []
        interventions: dict[str, list] = {}
        for i in risk.interventions:
            for t in ([i["ticker"]] if "ticker" in i else i.get("members", ["*"])):
                interventions.setdefault(t, []).append(i["check"])
        versions = {s.name: {"version": s.version, "config_hash": s.config_hash()} for s in self.strategies}
        for sym in sorted(set(risk.weights) | set(positions)):
            price = closes.get(sym)
            if not price or not math.isfinite(price):
                continue
            inst = self.instruments.get(sym)
            target_q = inst.round_quantity(risk.weights.get(sym, 0.0) * eq / (price * inst.multiplier))
            delta = target_q - positions.get(sym, 0.0)
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

