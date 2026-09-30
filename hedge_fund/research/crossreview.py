"""Multi-model research review: one model proposes, another critiques.

    Claude proposes -> Kimi critiques        (or the reverse)
    Python quantitative validation            -> the only PASS/FAIL

The models only write text: a hypothesis and an adversarial critique. They
never size positions, never place orders, and cannot change a verdict. The
final outcome of a review is the validation gate's result, computed in Python
from backtest evidence and passed in by the caller; the critique can at most
add concerns for a human to read.

Every model is named explicitly (a model id from the registry, never a router
alias), and every paid response goes through the prompt cache, so a review
replays for free and the cache records exactly which model answered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Protocol

from hedge_fund.llm import PromptCache, extract_json, make_llm, prompt_key
from hedge_fund.llm.registry import provider_for

# Router-style aliases that would let the served model change between runs.
_FORBIDDEN_MODEL_IDS = {"auto", "default", "router", "best", "cheapest", "fastest"}

PROPOSER_SYSTEM = (
    "You are a quantitative research assistant. Propose ONE testable, systematic "
    "trading hypothesis for the brief. Reply as JSON with keys: hypothesis, "
    "rationale, signal_definition, data_required, failure_modes (list). Do not "
    "recommend buying or selling any security; describe a rule to be backtested."
)
CRITIC_SYSTEM = (
    "You are an adversarial reviewer of quantitative research. Look for look-ahead "
    "bias, survivorship bias, data snooping, unrealistic costs or fills, regime "
    "dependence and overfitting. Reply as JSON with keys: concerns (list of "
    "strings), severity (low|medium|high), tests_to_add (list)."
)


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


class GateResult(Protocol):
    """Anything with a Python-computed boolean verdict (validation.gates)."""

    passed: bool


def check_explicit_model(model: str) -> str:
    """Reject router aliases; a research run must name one concrete model."""
    name = (model or "").strip()
    if not name or name.lower() in _FORBIDDEN_MODEL_IDS or name.lower().startswith("auto"):
        raise ValueError(f"model {model!r} is not an explicit model id; routing aliases are not allowed")
    return name


@dataclass(frozen=True)
class ReviewStep:
    role: Literal["proposer", "critic"]
    model: str
    cache_key: str
    cached: bool
    content: dict


@dataclass(frozen=True)
class CrossReview:
    brief: str
    proposal: ReviewStep
    critique: ReviewStep
    gate_passed: bool | None = None
    notes: list[str] = field(default_factory=list)

    # AI output can never promote anything on its own.
    ai_can_promote: bool = False

    @property
    def verdict(self) -> Literal["PASS", "FAIL", "NOT_VALIDATED"]:
        """The Python gate's verdict — the critique does not enter into it."""
        if self.gate_passed is None:
            return "NOT_VALIDATED"
        return "PASS" if self.gate_passed else "FAIL"

    @property
    def needs_human_attention(self) -> bool:
        return self.critique.content.get("severity") == "high" or bool(self.notes)


class CrossReviewer:
    """Run proposer -> critic with caching; attach a Python gate result."""

    def __init__(self, proposer: LLM, critic: LLM, cache: PromptCache | None = None) -> None:
        check_explicit_model(proposer.model)
        check_explicit_model(critic.model)
        if proposer.model == critic.model:
            raise ValueError("proposer and critic must be different models")
        self.proposer, self.critic = proposer, critic
        self.cache = cache if cache is not None else PromptCache()

    @classmethod
    def from_models(cls, proposer_model: str, critic_model: str, cache: PromptCache | None = None) -> CrossReviewer:
        """Build both clients by explicit model id (e.g. claude-opus-5-5, kimi-k3).

        Raises the client factory's error naming the missing variable when a
        key is absent (e.g. KIMI_API_KEY for Kimi) — nothing is called.
        """
        for m in (proposer_model, critic_model):
            check_explicit_model(m)
            if provider_for(m) is None:
                raise ValueError(f"{m!r} is not in the model registry; name a registered model")
        return cls(make_llm(proposer_model), make_llm(critic_model), cache)

    def _ask(self, role: str, llm: LLM, system: str, user: str) -> ReviewStep:
        key = prompt_key(f"crossreview:{role}", llm.model, system, user)
        hit = self.cache.get(key)
        if hit is not None and "parsed" in hit:
            return ReviewStep(role=role, model=llm.model, cache_key=key, cached=True, content=hit["parsed"])
        record = {"agent": f"crossreview:{role}", "model": llm.model, "system": system, "user": user}
        response = llm.complete(system, user)
        record["response"] = response
        try:
            parsed = extract_json(response)
        except Exception as exc:  # keep the paid response for audit, then fail loudly
            self.cache.put(key, {**record, "parse_error": str(exc)})
            raise
        self.cache.put(key, {**record, "parsed": parsed})
        return ReviewStep(role=role, model=llm.model, cache_key=key, cached=False, content=parsed)

    def review(self, brief: str, gate: GateResult | None = None) -> CrossReview:
        proposal = self._ask("proposer", self.proposer, PROPOSER_SYSTEM, brief)
        critic_input = json.dumps({"brief": brief, "proposal": proposal.content}, sort_keys=True)
        critique = self._ask("critic", self.critic, CRITIC_SYSTEM, critic_input)
        return CrossReview(brief=brief, proposal=proposal, critique=critique,
                           gate_passed=None if gate is None else bool(gate.passed))
