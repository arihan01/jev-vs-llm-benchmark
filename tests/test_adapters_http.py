import asyncio

import httpx
import pytest
import respx
from tenacity import wait_none

from bench.adapters import base
from bench.adapters.jev import JevAdapter

URL = "https://openrouter.ai/api/alpha/decisions"


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(base, "RETRY_WAIT", wait_none())


@pytest.fixture
def adapter():
    return JevAdapter("jev", {"kind": "jev", "model": "typesafe/jev-1.13"})


async def _call(adapter):
    async with httpx.AsyncClient() as client:
        return await adapter.call(client, {"model": "m"}, {})


@respx.mock
async def test_429_then_200_retries_once_and_times_final_attempt(adapter):
    async def slow_429(request):
        await asyncio.sleep(0.2)
        return httpx.Response(429, text="slow down")

    route = respx.post(URL).mock(side_effect=[slow_429, httpx.Response(200, json={"ok": True})])
    rec = await _call(adapter)
    assert route.call_count == 2
    assert rec["status"] == 200 and rec["retries_429"] == 1 and rec["response"] == {"ok": True}
    assert rec["latency_ms"] < 150, "backoff and failed attempts must not count toward latency"


@respx.mock
async def test_persistent_500_raises_after_all_attempts(adapter):
    route = respx.post(URL).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(base.HTTPStatusError) as e:
        await _call(adapter)
    assert e.value.status == 500 and route.call_count == base.RETRY_ATTEMPTS


@respx.mock
async def test_400_is_not_retried(adapter):
    route = respx.post(URL).mock(return_value=httpx.Response(400, text="bad request"))
    with pytest.raises(base.HTTPStatusError, match="HTTP 400"):
        await _call(adapter)
    assert route.call_count == 1


@respx.mock
async def test_timeout_is_retried(adapter):
    route = respx.post(URL).mock(side_effect=[httpx.ReadTimeout("slow"), httpx.Response(200, json={"ok": 1})])
    rec = await _call(adapter)
    assert route.call_count == 2 and rec["retries_429"] == 0


def test_base_adapter_is_abstract():
    a = base.Adapter("x", {"model": "m"})
    with pytest.raises(NotImplementedError):
        a.build_request({}, "t")
    with pytest.raises(NotImplementedError):
        a.parse({}, {})


@respx.mock
async def test_daily_cap_429_is_not_retried(adapter):
    route = respx.post(URL).mock(
        return_value=httpx.Response(429, json={"error": {"message": "Rate limit exceeded: free-models-per-day"}})
    )
    with pytest.raises(base.DailyLimitError):
        await _call(adapter)
    assert route.call_count == 1


class CountingLimiter:
    def __init__(self):
        self.waits = 0

    async def wait(self):
        self.waits += 1


@respx.mock
async def test_every_attempt_waits_on_the_limiter(adapter):
    respx.post(URL).mock(
        side_effect=[
            httpx.Response(429, text="slow"),
            httpx.Response(500, text="x"),
            httpx.Response(200, json={"ok": 1}),
        ]
    )
    limiter = CountingLimiter()
    async with httpx.AsyncClient() as client:
        rec = await adapter.call(client, {"model": "m"}, {}, limiter)
    assert limiter.waits == 3 and rec["status"] == 200


CHAT = "https://openrouter.ai/api/v1/chat/completions"
REJECT = {"error": {"message": "Provider returned error: response_format is not supported for this model"}}


@pytest.fixture
def chat_adapter():
    from bench.adapters.llm import LLMAdapter

    return LLMAdapter("low", {"kind": "llm", "model": "qwen/qwen3.8-27b:free"})


@respx.mock
async def test_schema_rejection_resends_once_without_response_format(chat_adapter):
    from bench import prompts

    body = chat_adapter.build_request(prompts.q1("noul", "yes"), "t")
    ok = {"choices": [{"message": {"content": '{"label": "yes", "confidence": 0.9}'}}]}
    route = respx.post(CHAT).mock(side_effect=[httpx.Response(400, json=REJECT), httpx.Response(200, json=ok)])
    async with httpx.AsyncClient() as client:
        rec = await chat_adapter.call(client, body, {})
    assert route.call_count == 2
    assert b"response_format" in route.calls[0].request.content
    assert b"response_format" not in route.calls[1].request.content
    assert "response_format" not in rec["request"] and rec["response"] == ok


@respx.mock
@pytest.mark.parametrize("error", [{"error": {"message": "invalid model id"}}, REJECT])
async def test_other_400s_are_not_resent(chat_adapter, error):
    from bench import prompts

    body = chat_adapter.build_request(prompts.q1("noul", "yes"), "t")
    if error is REJECT:
        body.pop("response_format")
    route = respx.post(CHAT).mock(return_value=httpx.Response(400, json=error))
    async with httpx.AsyncClient() as client:
        with pytest.raises(base.HTTPStatusError):
            await chat_adapter.call(client, body, {})
    assert route.call_count == 1
