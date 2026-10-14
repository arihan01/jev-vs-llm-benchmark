import asyncio
import json

from bench import metrics
from bench.adapters import baseline
from bench.adapters.base import provider_error
from bench.prompts import q1


def test_provider_error_in_a_200_body_is_detected():
    assert provider_error({"error": {"message": "overloaded", "code": 503}}) == 503
    assert provider_error({"choices": [{"finish_reason": "error", "message": {"content": '{"'}}]}) == 502
    assert provider_error({"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}) is None
    assert provider_error({"answers": {}}) is None


def test_prob_spread_sees_drift_the_label_hides():
    same = [json.dumps({"a": 0.9, "b": 0.1})] * 3
    drift = [json.dumps({"a": 0.9, "b": 0.1}), json.dumps({"a": 0.6, "b": 0.4}), json.dumps({"a": 0.9, "b": 0.1})]
    assert metrics.prob_spread(same) == 0
    assert abs(metrics.prob_spread(drift) - 0.3) < 1e-9


def test_baseline_answers_from_its_classifier(monkeypatch):
    class Pipe:
        classes_ = ("no", "yes")

        def predict_proba(self, xs):
            assert xs == ["is it? text"]
            return [[0.2, 0.8]]

    monkeypatch.setattr(baseline, "model", lambda task: Pipe())
    a = baseline.BaselineAdapter("stat", {"kind": "baseline", "model": "tfidf-logreg"})
    q = q1("noul", "yes")
    body = a.build_request(q, {"passage": "text", "question": "is it?"})
    p = a.parse(q, asyncio.run(a.call(None, body, {}))["response"])
    assert (p.label, p.valid, p.value) == ("yes", True, 0.8)


def test_cache_hit_requires_same_input(monkeypatch, tmp_path):
    from bench import cache, runner

    monkeypatch.setattr(cache, "CACHE", tmp_path)
    a = baseline.BaselineAdapter("stat", {"kind": "baseline", "model": "m"})
    q = q1("score", "3")
    job = runner.Job("E3", "score", {"id": "S1", "text": "good"}, q, "stat", 0)
    k = cache.key("stat", "E3", "S1", "Q1", 0)
    cache.put(a.cache_id, "E3", k, {"request": a.build_request(q, "good"), "response": {"probabilities": {}}})
    assert not runner._needs_call(a, job)
    job.sample["text"] = "edited"
    assert runner._needs_call(a, job)


def test_disabled_engine_runs_only_when_named():
    from bench import runner
    from bench.build_experiments import load_experiments

    exps = load_experiments()
    assert {j.engine for j in runner.build_jobs(exps)} == {"jev", "low", "high"}
    assert {j.engine for j in runner.build_jobs(exps, engine=["stat"])} == {"stat"}


def test_turning_schema_off_invalidates_schema_answers(monkeypatch, tmp_path):
    from bench import cache, runner
    from bench.adapters.llm import LLMAdapter

    monkeypatch.setattr(cache, "CACHE", tmp_path)
    q = q1("score", "3")
    job = runner.Job("E3", "score", {"id": "S1", "text": "good"}, q, "low", 0)
    with_schema = LLMAdapter("low", {"kind": "llm", "model": "m"})
    k = cache.key("low", "E3", "S1", "Q1", 0)
    cache.put(with_schema.cache_id, "E3", k, {"request": with_schema.build_request(q, "good"), "response": {}})
    assert not runner._needs_call(with_schema, job)
    assert runner._needs_call(LLMAdapter("low", {"kind": "llm", "model": "m", "structured_output": False}), job)
