"""Shared offline fixtures: a fake OpenRouter gateway and a filled-in copy of
the experiments file, so the whole pipeline runs without a key or network."""

import random

import pytest

from bench import build_experiments, cache, report, runner
from bench.adapters.base import Adapter


def fake_response(body: dict) -> dict:
    """Deterministic per request body, shaped like the real Jev / chat responses."""
    rng = random.Random(str(body)[:600])
    if "questions" in body:
        qid, q = next(iter(body["questions"].items()))
        if q["type"] == "choice":
            keys = list(q["criteria"])
            pick = rng.choice(keys)
            ans = {
                "type": "choice",
                "choice": pick,
                "probabilities": {k: float(k == pick) for k in keys},
                "confidence": 0.9,
            }
        elif q["type"] == "score":
            s = rng.uniform(0, len(q["criteria"]) - 1)
            ans = {"type": "score", "score": s, "probabilities": {str(round(s)): 1.0}}
        else:
            ans = {"type": "noul", "noul": rng.random()}
        return {"answers": {qid: ans}, "usage": {"prompt_tokens": 100, "cost": 4.2e-6}}
    keys = body["response_format"]["json_schema"]["schema"]["properties"]["label"]["enum"]
    label = rng.choice(keys) if rng.random() > 0.05 else "bogus"
    return {
        "choices": [{"message": {"content": f'{{"label": "{label}", "confidence": {rng.random():.2f}}}'}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 8, "cost": 2e-5},
    }


def make_ready(exps: dict) -> dict:
    for spec in exps.values():
        for s in spec["samples"]:
            if s["perturbation"] == build_experiments.PENDING:
                s["perturbation"] = "paraphrase"
            q2 = s["questions"][1]
            q2["instructions"] = "Second question"
            if q2["type"] == "score":
                q2["options"] = {"0": "low", "1": "mid", "2": "high"}
            elif q2["type"] == "choice":
                q2["options"] = {"a": "first", "b": "second"}
            q2["gold"] = next(iter(q2["options"]))
    return exps


@pytest.fixture
def ready_experiments() -> dict:
    return make_ready(build_experiments.build())


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Cache and results in tmp_path, Adapter.call replaced by the fake gateway.
    Returns (tmp_path, calls) where calls["n"] counts gateway hits."""
    monkeypatch.setattr(cache, "CACHE", tmp_path / "cache")
    for mod in (runner, report):
        monkeypatch.setattr(mod, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(runner, "headers", lambda: {})
    calls = {"n": 0}

    async def fake_call(self, client, body, headers):
        calls["n"] += 1
        return {"response": fake_response(body), "latency_ms": 100.0 + calls["n"] % 7, "status": 200, "retries_429": 0}

    monkeypatch.setattr(Adapter, "call", fake_call)
    return tmp_path, calls
