"""Paper trading loop — PAPER ONLY. No real money, no live endpoints.

    market data (panel) -> DecisionEngine (strategies, ensemble, portfolio, risk)
      -> orders -> TradingVenue (PaperBroker) -> fills -> reconciliation -> audit

`run_session(session)` is one end-of-day cycle, safe to call repeatedly:

    clock        refuses sessions that have not completed yet, and non-trading days
    idempotency  a session already processed is skipped; every order carries a
                 deterministic client id and ids already submitted are never resent
    kill switch  a KILL_SWITCH file in the state dir cancels open orders, queues a
                 flatten of every position, and halts further decisions
    health       stale data (panel does not reach the session) -> no trading
    retries      transient broker errors (BrokerUnavailable) are retried with backoff;
                 a persistent failure is recorded and the cycle stops without partial state
    reconcile    broker positions vs the engine's expected positions after fills
    audit        every step is appended to audit.jsonl

State (risk peak/day-start equity, expected positions, submitted ids, last
session) persists as JSON so a restarted process resumes where it stopped.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from hedge_fund.backtesting.fund import rebalance_grid
from hedge_fund.brokers.paper import BrokerUnavailable
from hedge_fund.core.orders import OrderRequest
from hedge_fund.data.sessions import completed_through
from hedge_fund.systematic.decision import DecisionEngine, marks_for
from hedge_fund.systematic.risk import RiskState

KILL_SWITCH = "KILL_SWITCH"


class PaperTradingError(RuntimeError):
    pass


class PaperTradingEngine:
    def __init__(self, panel, decider: DecisionEngine, broker, state_dir: Path | str, *,
                 rebalance: str = "monthly", retries: int = 3, backoff: float = 1.0,
                 clock: Callable[[], str] = completed_through, sleep: Callable[[float], None] = time.sleep) -> None:
        if getattr(broker, "live", True):
            raise PaperTradingError("paper trading requires a paper venue (broker.live must be False)")
        self.panel, self.decider, self.broker = panel, decider, broker
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rebalance, self.retries, self.backoff = rebalance, retries, backoff
        self.clock, self.sleep = clock, sleep

    # -- persistence ----------------------------------------------------------

    @property
    def _state_path(self) -> Path:
        return self.dir / "state.json"

    def load_state(self) -> dict:
        if self._state_path.exists():
            return json.loads(self._state_path.read_text())
        eq = self.broker.cash()
        return {"last_session": None, "peak_equity": eq, "day_start_equity": eq, "submitted": [],
                "expected_positions": {}, "halted": None}

    def _save(self, state: dict) -> None:
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True, indent=1))
        os.replace(tmp, self._state_path)

    def audit(self, event: str, **data) -> None:
        line = {"at": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        with open(self.dir / "audit.jsonl", "a") as fh:
            fh.write(json.dumps(line, sort_keys=True, default=str) + "\n")

    def _retry(self, what: str, fn, *args):
        for attempt in range(self.retries + 1):
            try:
                return fn(*args)
            except BrokerUnavailable as exc:
                self.audit("broker_retry", what=what, attempt=attempt + 1, error=str(exc))
                if attempt == self.retries:
                    raise PaperTradingError(f"{what}: broker unavailable after {self.retries} retries") from exc
                self.sleep(self.backoff * (2 ** attempt))

    # -- one cycle ---------------------------------------------------------------

    def run_session(self, session: str) -> dict:
        state = self.load_state()
        report = {"session": session, "status": "ok", "orders": [], "fills": 0}
        if session > self.clock():
            report["status"] = "refused_future_session"
            self.audit("refused", session=session, reason="session not completed")
            return report
        if state["last_session"] is not None and session <= state["last_session"]:
            report["status"] = "already_processed"
            return report
        sessions = self.panel.sessions_through(session)
        if not sessions or sessions[-1] != session:
            last = sessions[-1] if sessions else None
            status = "stale_data" if last is None or session > self.panel.info.end else "market_closed"
            report["status"] = status
            self.audit(status, session=session, panel_last_session=last)
            return report

        view = self.panel.as_of(session)
        # 1) fills for orders submitted after earlier sessions
        before = self.broker.positions()
        touched = self._retry("process", self.broker.process, session) or []
        for o in touched:
            for f in o.fills:
                if f.session == session:
                    exp = state["expected_positions"]
                    exp[f.symbol] = exp.get(f.symbol, 0.0) + f.signed_quantity
                    if abs(exp[f.symbol]) < 1e-9:
                        exp.pop(f.symbol)
                    report["fills"] += 1
        after = self.broker.positions()
        self.audit("fills", session=session, before=before, after=after,
                   orders=[{"id": o.request.client_order_id, "status": o.status.value} for o in touched])
        rec = self.broker.reconcile(state["expected_positions"], self.broker.cash())
        report["reconciled"] = rec.ok
        if not rec.ok:
            report["status"] = "reconciliation_break"
            self.audit("reconciliation_break", session=session, differences=rec.position_differences)

        # 2) valuation and risk state
        marks = marks_for(view, after)
        equity = self.broker.cash() + sum(q * marks[s] for s, q in after.items())
        risk_state = RiskState(peak_equity=max(state["peak_equity"], equity), day_start_equity=state["day_start_equity"])
        report["equity"] = equity

        # 3) kill switch
        if (self.dir / KILL_SWITCH).exists() or state.get("halted"):
            for o in list(getattr(self.broker, "open_orders", lambda: [])()):
                self.broker.cancel(o.request.client_order_id)
            orders = [OrderRequest(symbol=s, side="sell" if q > 0 else "buy", quantity=abs(q), decision_session=session,
                                   reference_price=marks[s], strategy="kill-switch",
                                   reason={"kill_switch": True}).with_client_id()
                      for s, q in sorted(after.items())]
            state["halted"] = state.get("halted") or "kill switch"
            report["status"] = "halted"
            self._submit(orders, state, report, session)
            self._finish(state, session, equity, risk_state)
            return report

        # 4) decision on rebalance sessions
        if self.is_rebalance_session(session) and rec.ok:
            record, orders = self.decider.decide(view, after, marks, equity, risk_state)
            self.audit("decision", session=session, record=record)
            if risk_state.halted_reason:
                state["halted"] = risk_state.halted_reason
            self._submit(orders, state, report, session)
        self._finish(state, session, equity, risk_state)
        return report

    def is_rebalance_session(self, session: str) -> bool:
        """Last session of its week/month, judged from the trading *calendar*.

        The calendar (which dates are sessions) is public in advance; no price
        after `session` is read. Beyond the panel's last session, the next
        weekday stands in for the next session.
        """
        if self.rebalance == "daily":
            return True
        calendar = self.panel.sessions_through(self.panel.info.end)
        if session in calendar and session != calendar[-1]:
            nxt = calendar[calendar.index(session) + 1]
        else:
            d = date.fromisoformat(session) + timedelta(days=1)
            while d.weekday() >= 5:
                d += timedelta(days=1)
            nxt = d.isoformat()
        return rebalance_grid([session, nxt], self.rebalance)[0] == session

    def _submit(self, orders, state, report, session) -> None:
        for o in orders:
            if o.client_order_id in state["submitted"]:
                self.audit("duplicate_suppressed", session=session, id=o.client_order_id)
                continue
            s = self._retry("submit", self.broker.submit, o)
            state["submitted"].append(o.client_order_id)
            report["orders"].append({"id": o.client_order_id, "symbol": o.symbol, "side": o.side,
                                     "quantity": o.quantity, "status": s.status.value})
            self.audit("order", session=session, id=o.client_order_id, symbol=o.symbol, side=o.side,
                       quantity=o.quantity, status=s.status.value, reason=s.reject_reason)

    def _finish(self, state, session, equity, risk_state) -> None:
        state.update(last_session=session, peak_equity=max(risk_state.peak_equity, equity), day_start_equity=equity)
        self._save(state)
