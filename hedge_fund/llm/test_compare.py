"""Model comparison tests — synthetic saved decisions, fake clients, no network."""

import json

import pytest

from hedge_fund.llm.compare import load_references, run_model, sample, score


def record(i, agent, signal, confidence, model="claude-opus-5-5"):
    return {"agent": agent, "model": model, "ticker": f"T{i}", "system": f"sys-{agent}", "user": f"user-{i}",
            "parsed": {"signal": signal, "confidence": confidence, "reasoning": "r"}}


def write_cache(tmp_path, records):
    for i, rec in enumerate(records):
        (tmp_path / f"{i:04d}.json").write_text(json.dumps(rec))
    (tmp_path / "broken.json").write_text("{not json")
    return tmp_path


def test_references_are_the_reference_models_parsed_answers(tmp_path):
    cache = write_cache(tmp_path, [record(0, "buffett", "bullish", 80), record(1, "graham", "bearish", 60, model="other"),
                                   {"agent": "lynch", "model": "claude-opus-5-5", "system": "s", "user": "u", "parse_error": "x"}])
    refs = load_references(cache, "claude-opus-5-5")
    assert [(r["agent"], r["parsed"]["signal"]) for r in refs] == [("buffett", "bullish")]


def test_sample_balances_agents_and_verdicts():
    refs = [record(i, agent, signal, 50) for i, (agent, signal) in enumerate(
        [("buffett", "bullish")] * 20 + [("buffett", "bearish")] * 2 + [("graham", "neutral")] * 20)]
    picked = sample(refs, 6, seed=1)
    assert len(picked) == 6
    assert {(r["agent"], r["parsed"]["signal"]) for r in picked} == {("buffett", "bullish"), ("buffett", "bearish"), ("graham", "neutral")}


def test_run_and_score_against_the_reference():
    refs = [record(0, "a", "bullish", 80), record(1, "a", "bearish", 70), record(2, "a", "neutral", 50), record(3, "a", "bullish", 60)]

    class Fake:
        model = "fake"
        replies = iter(['{"signal": "bullish", "confidence": 70, "reasoning": "x"}',
                        '{"signal": "bullish", "confidence": 55, "reasoning": "x"}',   # opposite of bearish
                        'no json here',
                        '{"signal": "bullish", "confidence": 60, "reasoning": "x"}'])

        def complete(self, system, user):
            return next(self.replies)

    results = run_model(Fake(), refs)
    s = score(results)
    assert s["n"] == 4 and s["valid"] == pytest.approx(0.75)
    assert s["agreement"] == pytest.approx(2 / 3)        # of the valid answers
    assert s["opposite"] == pytest.approx(1 / 3)
    assert -1 <= s["value_correlation"] <= 1
