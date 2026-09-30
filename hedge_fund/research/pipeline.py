"""Controlled self-improvement: candidates move forward only through gates.

    CANDIDATE -> BACKTESTED -> VALIDATED -> STRESS_TESTED -> HOLDOUT_PASSED
      -> PAPER -> LIVE_CANDIDATE (human approval)          any stage -> REJECTED

Who may do what:

    AI (actor "ai:<model>")   propose a candidate (a spec for a registered strategy
                              type), attach reviews/critiques/failure analyses.
                              Nothing else.
    system                    advance a candidate one stage at a time, only with the
                              evidence that stage requires (Python-computed gate
                              results, a locked-holdout result, paper metrics).
    human ("human:<name>")    the only actor who can approve LIVE_CANDIDATE, and the
                              only one allowed to perform protected actions.

Protected actions — changing risk limits, validation thresholds or the locked
holdout, promoting to live, raising leverage, disabling the kill switch — are
refused for any non-human actor, whatever the request says. Every event is
written to the experiment registry.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from hedge_fund.systematic.strategies import strategy_from_spec
from hedge_fund.validation.registry import ExperimentRegistry, spec_hash


class Stage(str, Enum):
    CANDIDATE = "candidate"
    BACKTESTED = "backtested"
    VALIDATED = "validated"
    STRESS_TESTED = "stress_tested"
    HOLDOUT_PASSED = "holdout_passed"
    PAPER = "paper"
    LIVE_CANDIDATE = "live_candidate"
    REJECTED = "rejected"


ORDER = [Stage.CANDIDATE, Stage.BACKTESTED, Stage.VALIDATED, Stage.STRESS_TESTED, Stage.HOLDOUT_PASSED,
         Stage.PAPER, Stage.LIVE_CANDIDATE]

PROTECTED_ACTIONS = frozenset({
    "change_risk_limits", "change_validation_gates", "change_locked_holdout", "promote_to_live",
    "increase_leverage", "disable_kill_switch",
})

AI_ALLOWED_ACTIONS = frozenset({"propose_candidate", "propose_hypothesis", "propose_features",
                                "analyze_failure", "adversarial_review"})


class GovernanceViolation(PermissionError):
    pass


def _kind(actor: str) -> str:
    kind = actor.split(":", 1)[0]
    if kind not in ("ai", "system", "human") or (kind != "system" and ":" not in actor):
        raise GovernanceViolation(f"unknown actor {actor!r}; use ai:<model>, system or human:<name>")
    return kind


def authorize(actor: str, action: str) -> None:
    kind = _kind(actor)
    if action in PROTECTED_ACTIONS and kind != "human":
        raise GovernanceViolation(f"{actor} may not {action}: protected action, human only")
    if kind == "ai" and action not in AI_ALLOWED_ACTIONS:
        raise GovernanceViolation(f"{actor} may not {action}; AI may only {sorted(AI_ALLOWED_ACTIONS)}")


class Candidate(BaseModel):
    spec: dict
    family: str
    stage: Stage = Stage.CANDIDATE
    proposed_by: str
    parent: dict | None = None
    notes: list[dict] = Field(default_factory=list)

    @property
    def id(self) -> str:
        return spec_hash(self.spec)


_REQUIRED_EVIDENCE = {
    Stage.BACKTESTED: "backtest",
    Stage.VALIDATED: "gates",
    Stage.STRESS_TESTED: "stress",
    Stage.HOLDOUT_PASSED: "holdout",
    Stage.PAPER: "paper",
}


class ResearchPipeline:
    def __init__(self, registry: ExperimentRegistry) -> None:
        self.registry = registry

    def _log(self, c: Candidate, stage: str, actor: str, metrics: dict) -> None:
        self.registry.record(family=c.family, spec=c.spec, stage=stage, metrics=metrics,
                             notes=f"actor={actor}")

    def propose(self, spec: dict, *, family: str, actor: str, parent: dict | None = None) -> Candidate:
        authorize(actor, "propose_candidate")
        strategy_from_spec(spec)            # must be a registered, valid strategy type + config
        c = Candidate(spec=spec, family=family, proposed_by=actor, parent=parent)
        self._log(c, Stage.CANDIDATE.value, actor, {"proposed": True})
        return c

    def review(self, c: Candidate, *, actor: str, kind: str, text: str) -> None:
        authorize(actor, kind)
        c.notes.append({"actor": actor, "kind": kind, "text": text})
        self._log(c, f"note:{kind}", actor, {"chars": len(text)})

    def advance(self, c: Candidate, target: Stage, *, actor: str, evidence: dict | None = None) -> Candidate:
        kind = _kind(actor)
        if c.stage in (Stage.REJECTED, Stage.LIVE_CANDIDATE):
            raise GovernanceViolation(f"candidate is {c.stage.value}; it cannot move")
        if target is Stage.REJECTED:
            if kind == "ai":
                raise GovernanceViolation("AI may analyse failures, not change a candidate's status")
            c.stage = Stage.REJECTED
            self._log(c, Stage.REJECTED.value, actor, dict(evidence or {}))
            return c
        if ORDER.index(target) != ORDER.index(c.stage) + 1:
            raise GovernanceViolation(f"{c.stage.value} -> {target.value} skips a stage")
        if target is Stage.LIVE_CANDIDATE:
            authorize(actor, "promote_to_live")
            if not (evidence or {}).get("approval"):
                raise GovernanceViolation("live-candidate status needs a recorded human approval")
        else:
            if kind != "system":
                raise GovernanceViolation(f"only the system advances candidates on evidence, not {actor}")
            key = _REQUIRED_EVIDENCE[target]
            result = (evidence or {}).get(key)
            passed = getattr(result, "passed", None) if not isinstance(result, dict) else result.get("passed")
            if passed is not True:
                c.stage = Stage.REJECTED
                self._log(c, Stage.REJECTED.value, actor, {"failed_at": target.value})
                raise GovernanceViolation(f"{target.value} evidence missing or failing; candidate rejected")
        c.stage = target
        self._log(c, target.value, actor, {"advanced": True})
        return c
