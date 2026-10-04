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


def test_prediction_to_dict_round_trips():
    p = base.Prediction(label="yes", probs={"yes": 0.7}, valid=True)
    d = p.to_dict()
    assert d["label"] == "yes" and d["probs"] == {"yes": 0.7} and d["valid"] is True


def test_base_adapter_is_abstract():
    a = base.Adapter("x", {"model": "m"})
    with pytest.raises(NotImplementedError):
        a.build_request({}, "t")
    with pytest.raises(NotImplementedError):
        a.parse({}, {})
