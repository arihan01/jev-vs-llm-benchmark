"""Fairness rule 1: Jev and the LLMs receive the same words for every question."""

import json

import pytest

from bench import prompts
from bench.adapters.jev import JevAdapter
from bench.adapters.llm import LLMAdapter

JEV = JevAdapter("jev", {"kind": "jev", "model": "typesafe/jev-1.13"})
LLM = LLMAdapter("low", {"kind": "llm", "model": "x/y"})

Q2_SCORE = {
    "id": "Q2",
    "type": "score",
    "instructions": "How likely is the reviewer to return?",
    "options": {"0": "Will not return", "1": "Might return", "2": "Will certainly return"},
    "gold": "2",
}
QUESTIONS = [prompts.q1("choice", "age_limit"), prompts.q1("score", "3"), prompts.q1("noul", "yes"), Q2_SCORE]


def _criteria_strings(jq):
    c = jq["criteria"]
    return list(c.values()) if isinstance(c, dict) else list(c)


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: f"{q['type']}-{q['id']}")
def test_same_words_for_both_engines(q):
    jq = JEV.build_request(q, "hello")["questions"][q["id"]]
    user = LLM.build_request(q, "hello")["messages"][1]["content"]
    assert jq["instructions"] == q["instructions"] and q["instructions"] in user
    assert sorted(_criteria_strings(jq)) == sorted(q["options"].values())
    for key, text in q["options"].items():
        assert f"{key}: {text}" in user


def test_llm_template_matches_brief():
    user = prompts.llm_messages(Q2_SCORE, "Nice place.")[1]["content"]
    assert user == (
        "How likely is the reviewer to return?\n"
        "Options:\n0: Will not return\n1: Might return\n2: Will certainly return\n"
        "Input:\nNice place.\n"
        'Return {"label": <one option key>, "confidence": <0-1>}.'
    )


def test_dict_state_is_rendered_as_json_for_the_llm():
    state = {"passage": "P", "question": "Q"}
    assert prompts.render_state(state) == json.dumps(state, indent=2)
    assert json.dumps(state, indent=2) in prompts.llm_messages(prompts.q1("noul", "yes"), state)[1]["content"]


def test_score_criteria_follow_numeric_level_order():
    q = {**Q2_SCORE, "options": {str(i): f"level {i}" for i in (10 - 1, 0, 2, 1, 3, 4, 5, 6, 7, 8)}}
    assert prompts.jev_question(q)["criteria"] == [f"level {i}" for i in range(10)]


def test_noul_maps_yes_no_to_true_false():
    assert prompts.jev_question(prompts.q1("noul", "no"))["criteria"] == {
        "true": prompts.NOUL_CRITERIA["yes"],
        "false": prompts.NOUL_CRITERIA["no"],
    }


def test_q1_defaults():
    assert len(prompts.q1("choice", "x")["options"]) == 77
    assert prompts.option_keys(prompts.q1("score", "0")) == ["0", "1", "2", "3", "4"]
    with pytest.raises(ValueError):
        prompts.default_options("nope")
