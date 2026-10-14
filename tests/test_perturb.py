import json
import random

import httpx
import pytest
import respx

from bench import build_experiments, perturb


def test_typo_lowercases_and_strips_punctuation():
    assert perturb.typo("Hello, World! How ARE you?", random.Random(0), rate=0.0) == "hello world how are you"


def test_padding_only_touches_passage():
    text = {"passage": "Sky is blue.", "question": "is the sky blue"}
    out = perturb.map_text(text, lambda s: perturb.pad(s, random.Random(0)), fields=("passage",))
    assert out["question"] == text["question"] and len(out["passage"]) > 2 * len(text["passage"])


def test_build_sample_is_deterministic_and_keeps_gold():
    item = {"item_id": "score-1", "text": "Great food, slow service.", "gold": "3"}
    kinds = ["typo", "padding", "distractor", "clean", "paraphrase"]
    a = [build_experiments.build_sample("score", i, item, k, [item], 42) for i, k in enumerate(kinds)]
    b = [build_experiments.build_sample("score", i, item, k, [item], 42) for i, k in enumerate(kinds)]
    assert a == b
    assert all(s["questions"][0]["gold"] == "3" and s["questions"][1]["gold"] is None for s in a)
    assert a[3]["perturbation"] == "clean" and a[3]["text"] == item["text"]
    assert a[4]["perturbation"] == build_experiments.PENDING and a[4]["text"] == item["text"]


def test_paraphrase_prompt_targets_passage_for_noul():
    prompt = perturb.paraphrase_prompt("noul", {"passage": "The sky is blue.", "question": "is it"})
    assert prompt.endswith("The sky is blue.") and "passage" in prompt


def test_distractor_per_task():
    rng = random.Random(0)
    choice = perturb.distractor("choice", {"item_id": "c", "text": "Card?", "gold": "card_arrival"}, rng, [])
    assert choice.startswith("Card? This isn't about ") and "card arrival" not in choice
    assert perturb.distractor("score", {"item_id": "s", "text": "Meh.", "gold": "2"}, rng, []).endswith("either.")
    pool = [
        {"item_id": "n1", "text": {"passage": "P.", "question": "is a"}, "gold": "yes"},
        {"item_id": "n2", "text": {"passage": "Q.", "question": "is b"}, "gold": "no"},
    ]
    out = perturb.distractor("noul", pool[0], rng, pool)
    assert out["passage"] == "P. This passage is not about whether is b." and out["question"] == "is a"


@respx.mock
def test_paraphrase_calls_generator_once_then_uses_cache(tmp_path, monkeypatch):
    from bench import cache

    monkeypatch.setattr(cache, "CACHE", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "  A new passage.  "}}]})
    )
    text = {"passage": "Old passage.", "question": "q"}
    assert perturb.paraphrase("noul", "noul-1", text) == {"passage": "A new passage.", "question": "q"}
    assert perturb.paraphrase("noul", "noul-1", text) == {"passage": "A new passage.", "question": "q"}
    assert route.call_count == 1
    assert route.calls[0].request.headers["Authorization"] == "Bearer sk-or-v1-test"


@respx.mock
def test_generator_cache_key_includes_model(tmp_path, monkeypatch):
    from bench import cache, config

    monkeypatch.setattr(cache, "CACHE", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=lambda req: httpx.Response(
            200, json={"choices": [{"message": {"content": "from " + json.loads(req.content)["model"]}}]}
        )
    )
    assert perturb.paraphrase("score", "s-1", "Good food.") == "from google/gemini-3.5-flash-lite"
    config.use_config(config.ROOT / "config.free.yaml")
    try:
        assert perturb.paraphrase("score", "s-1", "Good food.") == "from nvidia/nemotron-3-super-120b-a12b:free"
    finally:
        config.use_config(None)
    assert route.call_count == 2


@respx.mock
def test_generator_daily_limit_exits_cleanly(tmp_path, monkeypatch):
    from bench import cache

    monkeypatch.setattr(cache, "CACHE", tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "Rate limit exceeded: free-models-per-day"}})
    )
    with pytest.raises(SystemExit, match="daily request limit; re-run tomorrow"):
        perturb.paraphrase("score", "s-1", "Good food.")
