"""Cross-model review with mocked models: no network, no keys, no spend."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from hedge_fund.llm import PromptCache
from hedge_fund.llm.registry import ALIAS_ENV_VARS, PROVIDER_ENV_VARS
from hedge_fund.research import CrossReviewer, check_explicit_model


class Scripted:
    def __init__(self, model: str, reply: dict) -> None:
        self.model, self.reply, self.calls = model, reply, 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return "Here you go:\n" + json.dumps(self.reply)


@dataclass
class Gate:
    passed: bool


PROPOSAL = {"hypothesis": "12-1 momentum persists", "rationale": "...", "signal_definition": "...",
            "data_required": ["daily closes"], "failure_modes": ["crashes"]}
CRITIQUE = {"concerns": ["survivorship in universe"], "severity": "high", "tests_to_add": ["delisted names"]}


def reviewer(tmp_path, proposer="claude-opus-5-5", critic="kimi-k3"):
    p, c = Scripted(proposer, PROPOSAL), Scripted(critic, CRITIQUE)
    return CrossReviewer(p, c, PromptCache(tmp_path / "llm")), p, c


def test_claude_proposes_kimi_critiques_and_python_decides(tmp_path):
    r, p, c = reviewer(tmp_path)
    out = r.review("trend following on liquid ETFs", gate=Gate(passed=False))
    assert out.proposal.model == "claude-opus-5-5" and out.critique.model == "kimi-k3"
    assert out.proposal.content == PROPOSAL and out.critique.content == CRITIQUE
    assert out.verdict == "FAIL"                      # the gate, not the models
    assert out.ai_can_promote is False
    assert out.needs_human_attention


def test_reverse_direction_is_symmetric(tmp_path):
    r, _, _ = reviewer(tmp_path, proposer="kimi-k3", critic="claude-opus-5-5")
    out = r.review("mean reversion", gate=Gate(passed=True))
    assert (out.proposal.model, out.critique.model) == ("kimi-k3", "claude-opus-5-5")
    assert out.verdict == "PASS"


def test_a_glowing_critique_cannot_pass_a_failed_gate(tmp_path):
    p = Scripted("claude-opus-5-5", PROPOSAL)
    c = Scripted("kimi-k3", {"concerns": [], "severity": "low", "tests_to_add": [], "verdict": "PASS"})
    out = CrossReviewer(p, c, PromptCache(tmp_path / "llm")).review("x", gate=Gate(passed=False))
    assert out.verdict == "FAIL"


def test_without_a_gate_nothing_is_validated(tmp_path):
    r, _, _ = reviewer(tmp_path)
    assert r.review("x").verdict == "NOT_VALIDATED"


def test_responses_are_cached_and_never_repaid(tmp_path):
    r, p, c = reviewer(tmp_path)
    first = r.review("breakout on futures")
    again = r.review("breakout on futures")
    assert (p.calls, c.calls) == (1, 1)
    assert not first.proposal.cached and again.proposal.cached and again.critique.cached
    assert again.proposal.cache_key == first.proposal.cache_key


@pytest.mark.parametrize("alias", ["auto", "AUTO", "auto:cheap", "router", "default", ""])
def test_router_aliases_are_rejected(alias):
    with pytest.raises(ValueError):
        check_explicit_model(alias)


def test_same_model_cannot_review_itself(tmp_path):
    with pytest.raises(ValueError):
        CrossReviewer(Scripted("kimi-k3", PROPOSAL), Scripted("kimi-k3", CRITIQUE), PromptCache(tmp_path))


def test_missing_kimi_key_names_the_variable_and_calls_nothing(monkeypatch, tmp_path):
    for v in (*PROVIDER_ENV_VARS.values(), *ALIAS_ENV_VARS.values(), "HEDGE_FUND_LLM_MODEL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("AIHF_ANTHROPIC_API_KEY", "test-key-not-real")
    with pytest.raises(ValueError, match="KIMI_API_KEY"):
        CrossReviewer.from_models("claude-opus-5-5", "kimi-k3", PromptCache(tmp_path))


def test_kimi_client_is_openai_compatible_with_configurable_base_url(monkeypatch):
    import langchain_openai
    from unittest.mock import Mock
    from hedge_fund.llm import make_llm
    chat = Mock()
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", chat)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.setenv("KIMI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("MOONSHOT_BASE_URL", raising=False)
    assert make_llm("kimi-k3").model == "kimi-k3"
    assert chat.call_args.kwargs["base_url"] == "https://api.moonshot.ai/v1"
    monkeypatch.setenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
    make_llm("kimi-k3")
    assert chat.call_args.kwargs["base_url"] == "https://api.moonshot.cn/v1"
    assert chat.call_args.kwargs["model"] == "kimi-k3"


def test_unregistered_model_is_refused_for_research(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="registry"):
        CrossReviewer.from_models("claude-opus-5-5", "some-unlisted-model", PromptCache(tmp_path))
