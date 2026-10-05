import asyncio
import re
import time
from dataclasses import dataclass, field

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from ..config import engine_url

RETRY_ATTEMPTS = 6
RETRY_WAIT = wait_exponential_jitter(initial=1, max=30)


@dataclass
class Prediction:
    label: str | None
    probs: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0  # probability of the predicted label
    value: float | None = None  # Score: position 0-4; Noul: P(yes)
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    valid: bool = False
    error: str | None = None


class RateLimiter:
    """Spaces calls at least 60/rpm seconds apart (e.g. OpenRouter's 20/min cap
    on free models)."""

    def __init__(self, rpm: float, clock=time.monotonic, sleep=asyncio.sleep):
        self.interval = 60.0 / rpm
        self._next = 0.0
        self._lock = asyncio.Lock()
        self._clock = clock
        self._sleep = sleep

    async def wait(self) -> None:
        async with self._lock:
            delay = self._next - self._clock()
            if delay > 0:
                await self._sleep(delay)
            self._next = max(self._clock(), self._next) + self.interval


class HTTPStatusError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


class DailyLimitError(HTTPStatusError):
    """A 429 for an exhausted daily quota (e.g. OpenRouter free models). Retrying
    within minutes cannot succeed, so it is never retried."""


DAILY_LIMIT = re.compile(r"per[- ]day|daily", re.I)


def _retryable(e: BaseException) -> bool:
    if isinstance(e, DailyLimitError):
        return False
    if isinstance(e, HTTPStatusError):
        return e.status == 429 or e.status >= 500
    return isinstance(e, (httpx.TransportError, httpx.TimeoutException))


class Adapter:
    """Subclasses build the request body and parse the response; HTTP, retries
    and timing live here so every engine is measured the same way."""

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.spec = spec
        self.model = spec["model"]
        self.url = engine_url(spec)

    @property
    def cache_id(self) -> str:
        """Cache namespace: engine name plus model, so swapping a model under the
        same engine name (e.g. the free-trial config) never reuses its answers."""
        return f"{self.name}/{re.sub(r'[^A-Za-z0-9._-]', '_', self.model)}"

    def cost(self, usage: dict, tokens_in: int, tokens_out: int) -> float:
        """usage.cost when the provider reports it, else tokens × the configured
        price (USD per million tokens)."""
        if usage.get("cost") is not None:
            return float(usage["cost"])
        return (tokens_in * self.spec.get("price_in", 0) + tokens_out * self.spec.get("price_out", 0)) / 1e6

    def build_request(self, q: dict, text, temperature=0.0) -> dict:
        raise NotImplementedError

    def parse(self, q: dict, response: dict) -> Prediction:
        raise NotImplementedError

    async def call(
        self, client: httpx.AsyncClient, body: dict, headers: dict, limiter: RateLimiter | None = None
    ) -> dict:
        """Returns {response, latency_ms, status, retries_429}. Every attempt,
        retries included, waits its turn on `limiter`. Latency is of the final
        attempt only, so backoff and limiter waits never inflate it."""
        n429 = 0
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_retryable),
            wait=RETRY_WAIT,
            stop=stop_after_attempt(RETRY_ATTEMPTS),
            reraise=True,
        ):
            with attempt:
                if limiter:
                    await limiter.wait()
                t0 = time.perf_counter()
                r = await client.post(self.url, json=body, headers=headers)
                latency = (time.perf_counter() - t0) * 1000
                if r.status_code == 429 and DAILY_LIMIT.search(r.text):
                    raise DailyLimitError(429, r.text)
                if r.status_code != 200:
                    n429 += r.status_code == 429
                    raise HTTPStatusError(r.status_code, r.text)
                return {"response": r.json(), "latency_ms": latency, "status": 200, "retries_429": n429}
        raise RuntimeError("retry loop exited without a result")
