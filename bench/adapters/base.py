import time
from dataclasses import asdict, dataclass, field

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

RETRY_ATTEMPTS = 6
RETRY_WAIT = wait_exponential_jitter(initial=1, max=30)


@dataclass
class Prediction:
    label: str | None
    probs: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0  # probability of the predicted label
    value: float | None = None  # Score: position 0-4; Noul: P(yes)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    valid: bool = False
    status: int = 0
    retries_429: int = 0
    error: str | None = None
    raw: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class HTTPStatusError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def _retryable(e: BaseException) -> bool:
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

    url: str = ""

    def build_request(self, q: dict, text, temperature=0.0) -> dict:
        raise NotImplementedError

    def parse(self, q: dict, response: dict) -> Prediction:
        raise NotImplementedError

    async def call(self, client: httpx.AsyncClient, body: dict, headers: dict) -> dict:
        """Returns {response, latency_ms, status, retries_429}. Latency is of the
        final successful attempt only, so backoff sleeps never inflate it."""
        n429 = 0
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_retryable),
            wait=RETRY_WAIT,
            stop=stop_after_attempt(RETRY_ATTEMPTS),
            reraise=True,
        ):
            with attempt:
                t0 = time.perf_counter()
                r = await client.post(self.url, json=body, headers=headers)
                latency = (time.perf_counter() - t0) * 1000
                if r.status_code != 200:
                    n429 += r.status_code == 429
                    raise HTTPStatusError(r.status_code, r.text)
                return {"response": r.json(), "latency_ms": latency, "status": 200, "retries_429": n429}
        raise RuntimeError("retry loop exited without a result")
