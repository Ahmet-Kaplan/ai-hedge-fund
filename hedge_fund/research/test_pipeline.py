"""Self-improvement governance: AI proposes, Python gates decide, humans approve live."""

from __future__ import annotations

import pytest

from hedge_fund.research.pipeline import (
    PROTECTED_ACTIONS, GovernanceViolation, ResearchPipeline, Stage, authorize,
)
from hedge_fund.validation import ExperimentRegistry

SPEC = {"strategy": "tsmom", "version": "1.0.0", "config": {"lookback": 126, "skip": 5}}
OK = {"passed": True}


@pytest.fixture
def pipe(tmp_path):
    return ResearchPipeline(ExperimentRegistry(tmp_path / "exp.jsonl"))


def walk(pipe, c, upto):
    for stage, key in [(Stage.BACKTESTED, "backtest"), (Stage.VALIDATED, "gates"), (Stage.STRESS_TESTED, "stress"),
                       (Stage.HOLDOUT_PASSED, "holdout"), (Stage.PAPER, "paper")]:
        pipe.advance(c, stage, actor="system", evidence={key: OK})
        if stage is upto:
            return c
    return c


def test_ai_can_propose_and_review_only(pipe):
    c = pipe.propose(SPEC, family="tsmom", actor="ai:kimi-k3")
    pipe.review(c, actor="ai:claude-opus-5-5", kind="adversarial_review", text="check survivorship")
    with pytest.raises(GovernanceViolation):
        pipe.advance(c, Stage.BACKTESTED, actor="ai:claude-opus-5-5", evidence={"backtest": OK})
    with pytest.raises(GovernanceViolation):
        pipe.advance(c, Stage.REJECTED, actor="ai:kimi-k3")
    assert c.stage is Stage.CANDIDATE


def test_unknown_or_invalid_strategies_cannot_be_proposed(pipe):
    with pytest.raises(ValueError):
        pipe.propose({"strategy": "secret_sauce", "config": {}}, family="x", actor="ai:kimi-k3")
    with pytest.raises(ValueError):
        pipe.propose({"strategy": "tsmom", "config": {"lookback": 2}}, family="x", actor="ai:kimi-k3")


def test_stages_advance_in_order_on_passing_evidence(pipe):
    c = pipe.propose(SPEC, family="tsmom", actor="ai:kimi-k3")
    with pytest.raises(GovernanceViolation, match="skips"):
        pipe.advance(c, Stage.VALIDATED, actor="system", evidence={"gates": OK})
    walk(pipe, c, Stage.PAPER)
    assert c.stage is Stage.PAPER


def test_failing_evidence_rejects_for_good(pipe):
    c = pipe.propose(SPEC, family="tsmom", actor="ai:kimi-k3")
    pipe.advance(c, Stage.BACKTESTED, actor="system", evidence={"backtest": OK})
    with pytest.raises(GovernanceViolation, match="rejected"):
        pipe.advance(c, Stage.VALIDATED, actor="system", evidence={"gates": {"passed": False}})
    assert c.stage is Stage.REJECTED
    with pytest.raises(GovernanceViolation):
        pipe.advance(c, Stage.VALIDATED, actor="system", evidence={"gates": OK})


def test_missing_evidence_fails_closed(pipe):
    c = pipe.propose(SPEC, family="tsmom", actor="ai:kimi-k3")
    with pytest.raises(GovernanceViolation):
        pipe.advance(c, Stage.BACKTESTED, actor="system")


def test_live_candidate_needs_a_human_approval(pipe):
    c = walk(pipe, pipe.propose(SPEC, family="tsmom", actor="ai:kimi-k3"), Stage.PAPER)
    for actor in ("system", "ai:claude-opus-5-5"):
        with pytest.raises(GovernanceViolation):
            pipe.advance(c, Stage.LIVE_CANDIDATE, actor=actor, evidence={"approval": "yes"})
    with pytest.raises(GovernanceViolation, match="approval"):
        pipe.advance(c, Stage.LIVE_CANDIDATE, actor="human:owner")
    pipe.advance(c, Stage.LIVE_CANDIDATE, actor="human:owner", evidence={"approval": "signed 2026-10-01"})
    assert c.stage is Stage.LIVE_CANDIDATE


@pytest.mark.parametrize("action", sorted(PROTECTED_ACTIONS))
def test_protected_actions_are_human_only(action):
    for actor in ("ai:claude-opus-5-5", "ai:kimi-k3", "system"):
        with pytest.raises(GovernanceViolation):
            authorize(actor, action)
    authorize("human:owner", action)


def test_every_event_is_in_the_registry(pipe):
    c = pipe.propose(SPEC, family="tsmom", actor="ai:kimi-k3")
    pipe.advance(c, Stage.BACKTESTED, actor="system", evidence={"backtest": OK})
    stages = [r["stage"] for r in pipe.registry.records()]
    assert stages == ["candidate", "backtested"]
    pipe.registry.verify()


def test_actor_names_are_strict():
    with pytest.raises(GovernanceViolation):
        authorize("claude", "propose_candidate")
    with pytest.raises(GovernanceViolation):
        authorize("ai", "propose_candidate")
