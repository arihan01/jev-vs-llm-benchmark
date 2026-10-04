import pytest

from bench import prompts
from bench.adapters.llm import LLMAdapter, parse_content

Q_SCORE = {
    "id": "Q2",
    "type": "score",
    "instructions": "How likely is the reviewer to return?",
    "options": {"0": "Will not return", "1": "Might return", "2": "Will certainly return"},
    "gold": "2",
}


def llm(**spec):
    return LLMAdapter("low", {"kind": "llm", "model": "x/y", **spec})


def resp(content, **usage):
    return {"choices": [{"message": {"content": content}}], "usage": usage}


def test_request_shape():
    body = llm(reasoning="off").build_request(Q_SCORE, "t", temperature=0.0)
    schema = body["response_format"]["json_schema"]
    assert body["response_format"]["type"] == "json_schema" and schema["strict"] is True
    assert schema["schema"]["properties"]["label"]["enum"] == ["0", "1", "2"]
    assert schema["schema"]["required"] == ["label", "confidence"]
    assert body["temperature"] == 0.0 and body["usage"] == {"include": True}
    assert body["reasoning"] == {"enabled": False}
    assert body["messages"][0]["content"] == prompts.SYSTEM


@pytest.mark.parametrize("reasoning", [None, "on", True])
def test_reasoning_only_disabled_when_configured(reasoning):
    spec = {} if reasoning is None else {"reasoning": reasoning}
    assert "reasoning" not in llm(**spec).build_request(Q_SCORE, "t")


def test_default_temperature_omits_field():
    assert "temperature" not in llm().build_request(Q_SCORE, "t", temperature=None)


@pytest.mark.parametrize(
    "content, expected",
    [
        ('{"label": "1", "confidence": 0.5}', {"label": "1", "confidence": 0.5}),
        ('```json\n{"label": "1", "confidence": 0.5}\n```', {"label": "1", "confidence": 0.5}),
        ('Sure! {"label": "1", "confidence": 0.5} hope that helps', {"label": "1", "confidence": 0.5}),
        ("[1, 2]", None),
        ("not json at all", None),
        ("{broken", None),
        ("", None),
    ],
)
def test_parse_content(content, expected):
    assert parse_content(content) == expected


def test_probs_spread_remaining_mass_evenly():
    p = llm().parse(Q_SCORE, resp('{"label": "0", "confidence": 0.6}', prompt_tokens=9, completion_tokens=3, cost=1e-6))
    assert p.valid and p.label == "0" and p.confidence == pytest.approx(0.6)
    assert p.probs == pytest.approx({"0": 0.6, "1": 0.2, "2": 0.2})
    assert (p.tokens_in, p.tokens_out, p.cost) == (9, 3, 1e-6)
    assert p.value == 0.0


def test_noul_value_is_p_yes():
    q = prompts.q1("noul", "yes")
    assert llm().parse(q, resp('{"label": "yes", "confidence": 0.9}')).value == pytest.approx(0.9)
    assert llm().parse(q, resp('{"label": "no", "confidence": 0.9}')).value == pytest.approx(0.1)


def test_choice_has_no_value():
    q = {"id": "Q1", "type": "choice", "instructions": "i", "options": {"a": "A", "b": "B"}, "gold": "a"}
    assert llm().parse(q, resp('{"label": "b", "confidence": 1}')).value is None


@pytest.mark.parametrize(
    "content, conf",
    [('{"label": "2", "confidence": 7}', 1.0), ('{"label": "2", "confidence": -1}', 0.0), ('{"label": "2"}', 0.0)],
)
def test_confidence_is_clamped(content, conf):
    assert llm().parse(Q_SCORE, resp(content)).confidence == conf


def test_non_numeric_confidence_becomes_zero():
    assert llm().parse(Q_SCORE, resp('{"label": "2", "confidence": "high"}')).confidence == 0.0


@pytest.mark.parametrize(
    "response, error",
    [
        ({"choices": []}, "no message content"),
        ({}, "no message content"),
        (resp("three"), "unparseable JSON"),
        (resp('{"confidence": 1}'), "unparseable JSON"),
        (resp('{"label": "3", "confidence": 1}'), "label not in option set"),
    ],
)
def test_invalid_outputs(response, error):
    p = llm().parse(Q_SCORE, response)
    assert not p.valid and p.error == error
