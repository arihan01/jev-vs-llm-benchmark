import asyncio

import httpx
import pytest
import respx
from tenacity import wait_none

from bench import cache, config, runner
from bench.adapters import base

CHAT = "https://openrouter.ai/api/v1/chat/completions"
DAILY = {"error": {"message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000."}}


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


async def test_rate_limiter_spaces_calls_by_60_over_rpm():
    clock = FakeClock()
    limiter = runner.RateLimiter(20, clock=clock, sleep=clock.sleep)
    for _ in range(4):
        await limiter.wait()
    assert clock.sleeps == pytest.approx([3.0, 3.0, 3.0])


async def test_rate_limiter_does_not_wait_after_idle_time():
    clock = FakeClock()
    limiter = runner.RateLimiter(60, clock=clock, sleep=clock.sleep)
    await limiter.wait()
    clock.now += 10
    await limiter.wait()
    assert clock.sleeps == []


async def test_rate_limiter_serialises_concurrent_waiters():
    clock = FakeClock()
    limiter = runner.RateLimiter(30, clock=clock, sleep=clock.sleep)
    await asyncio.gather(*(limiter.wait() for _ in range(3)))
    assert sum(clock.sleeps) == pytest.approx(4.0)


@pytest.fixture
def live_free(tmp_path, monkeypatch, free_config, ready_experiments):
    """Free-trial config with real adapters and HTTP mocked by respx."""
    monkeypatch.setattr(cache, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(config, "results_dir", lambda: tmp_path / "results")
    monkeypatch.setattr(base, "RETRY_WAIT", wait_none())
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    monkeypatch.setattr(runner.RateLimiter, "wait", lambda self: asyncio.sleep(0))
    return ready_experiments


@respx.mock
def test_daily_cap_defers_that_engine_without_recording_rows(live_free, caplog):
    low = respx.post(CHAT).mock(return_value=httpx.Response(429, json=DAILY))
    respx.post("https://api.typesafe.ai/v1/systemone").mock(
        return_value=httpx.Response(200, json={"answers": {"Q1": {"noul": 0.8}}, "usage": {"input_tokens": 9}})
    )
    jobs = runner.build_jobs(live_free, exp=["E2"], engine=["low", "jev"], question=["Q1"], repeats=1)
    with caplog.at_level("WARNING"):
        raw = runner.run(jobs)
    assert low.call_count == 1, "after the daily cap, remaining jobs for that engine must not be sent"
    assert set(raw.engine) == {"jev"} and len(raw) == 5, "deferred jobs are not attempts and are not saved"
    assert "5 job(s) deferred" in caplog.text and "re-run tomorrow" in caplog.text
    assert not (cache.CACHE / "low").exists()


@respx.mock
def test_free_config_routes_and_keys_per_engine(live_free):
    def chat(request):
        body = request.read().decode()
        assert request.headers["Authorization"] == "Bearer sk-or-v1-test"
        assert '"usage"' in body
        label = "yes" if '"enum"' in body or "nemotron" in body else "no"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": f'{{"label": "{label}", "confidence": 0.9}}'}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            },
        )

    def typesafe(request):
        assert request.headers["Authorization"] == "Bearer ts-test"
        return httpx.Response(
            200,
            json={"answers": {"Q1": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 300, "output_tokens": 0}},
        )

    respx.post(CHAT).mock(side_effect=chat)
    ts = respx.post("https://api.typesafe.ai/v1/systemone").mock(side_effect=typesafe)
    raw = runner.run(runner.build_jobs(live_free, exp=["E2"], question=["Q1"], repeats=1))
    assert ts.call_count == 5 and len(raw) == 15
    jev = raw[raw.engine == "jev"]
    assert (jev.model == "jev-1.13.0").all() and jev.cost.iloc[0] == pytest.approx(300 * 0.042 / 1e6)
    high_requests = [c.request for c in respx.calls if b"nemotron" in c.request.content]
    assert high_requests and all(b"response_format" not in r.content for r in high_requests)


def test_cache_only_rerun_needs_no_key(sandbox, ready_experiments, monkeypatch):
    runner.run(runner.build_jobs(ready_experiments, exp=["E1"], repeats=1))

    def no_key(spec=None):
        raise SystemExit("key required")

    monkeypatch.setattr(runner, "headers", no_key)
    raw = runner.run(runner.build_jobs(ready_experiments, exp=["E1"], repeats=1))
    assert raw[raw.exp == "E1"].cached.all()
    with pytest.raises(SystemExit, match="key required"):
        runner.run(runner.build_jobs(ready_experiments, exp=["E2"], repeats=1))


@respx.mock
def test_schema_fallback_is_recorded_per_row(live_free):
    reject = {"error": {"message": "json_schema response_format unsupported"}}
    ok = {"choices": [{"message": {"content": '{"label": "yes", "confidence": 0.9}'}}], "usage": {}}

    def chat(request):
        return (
            httpx.Response(400, json=reject) if b"response_format" in request.content else httpx.Response(200, json=ok)
        )

    respx.post(CHAT).mock(side_effect=chat)
    respx.post("https://api.typesafe.ai/v1/systemone").mock(
        return_value=httpx.Response(200, json={"answers": {"Q1": {"noul": 0.8}}})
    )
    raw = runner.run(runner.build_jobs(live_free, exp=["E2"], sample=["S1"], question=["Q1"], repeats=1))
    by_engine = raw.set_index("engine")
    assert by_engine.loc["low", "valid"] and not by_engine.loc["low", "structured_output"]
    assert not by_engine.loc["high", "structured_output"], "high never sends a schema"
    assert by_engine.loc["jev", "structured_output"]
    cached = list((cache.CACHE / "low").rglob("*.json"))
    assert cached and b"response_format" not in cached[0].read_bytes()
