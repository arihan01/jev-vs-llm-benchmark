import pytest

from bench import prompts
from bench.adapters.jev import JevAdapter, find_answer

JEV = JevAdapter("jev", {"kind": "jev", "model": "typesafe/jev-1.13"})
Q_SCORE = {"id": "Q2", "type": "score", "instructions": "i", "options": {"0": "a", "1": "b", "2": "c"}, "gold": "2"}
ANSWER = {"type": "noul", "noul": 0.8}


@pytest.mark.parametrize(
    "envelope",
    [
        {"Q1": ANSWER},
        {"answers": {"Q1": ANSWER}},
        {"data": {"results": {"Q1": ANSWER}}},
        {"answers": [{"name": "Q1", **ANSWER}]},
        {"results": [{"other": 1}, {"Q1": ANSWER}]},
    ],
)
def test_find_answer_locates_nested_shapes(envelope):
    found = find_answer(envelope, "Q1")
    assert found is not None and found["noul"] == 0.8


def test_find_answer_ignores_non_answer_dicts():
    assert find_answer({"Q1": {"no_type": True}}, "Q1") is None
    assert find_answer({"Q2": ANSWER}, "Q1") is None


def test_request_uses_question_id_and_state():
    body = JEV.build_request(Q_SCORE, {"passage": "p", "question": "q"})
    assert body == {
        "model": "typesafe/jev-1.13",
        "state": {"passage": "p", "question": "q"},
        "questions": {"Q2": {"type": "score", "instructions": "i", "criteria": ["a", "b", "c"]}},
    }


@pytest.mark.parametrize("p_yes, label", [(0.5, "yes"), (0.4999, "no"), (1.0, "yes"), (0.0, "no")])
def test_noul_threshold(p_yes, label):
    p = JEV.parse(prompts.q1("noul", "yes"), {"Q1": {"type": "noul", "noul": p_yes}})
    assert p.label == label and p.value == p_yes
    assert p.confidence == pytest.approx(max(p_yes, 1 - p_yes))


@pytest.mark.parametrize("score, label", [(-0.7, "0"), (0.49, "0"), (1.51, "2"), (9.0, "2")])
def test_score_rounding_clamps(score, label):
    assert JEV.parse(Q_SCORE, {"Q2": {"type": "score", "score": score}}).label == label


def test_score_confidence_falls_back_to_reported_confidence():
    p = JEV.parse(Q_SCORE, {"Q2": {"type": "score", "score": 1.0, "confidence": 0.42}})
    assert p.confidence == 0.42


@pytest.mark.parametrize(
    "usage, expected",
    [
        ({"prompt_tokens": 11, "completion_tokens": 2}, (11, 2)),
        ({"input_tokens": 7, "output_tokens": 1}, (7, 1)),
        ({}, (0, 0)),
    ],
)
def test_token_field_fallbacks(usage, expected):
    p = JEV.parse(prompts.q1("noul", "yes"), {"Q1": ANSWER, "usage": usage})
    assert (p.tokens_in, p.tokens_out) == expected


def test_choice_outside_options_is_invalid():
    q = {"id": "Q1", "type": "choice", "instructions": "i", "options": {"a": "A"}, "gold": "a"}
    p = JEV.parse(q, {"Q1": {"type": "choice", "choice": "zzz", "probabilities": {"zzz": 1.0}}})
    assert not p.valid and p.confidence == 0.0


def test_malformed_answer_is_invalid():
    p = JEV.parse(Q_SCORE, {"Q2": {"type": "score", "score": "high"}})
    assert not p.valid and p.error.startswith("malformed answer")
