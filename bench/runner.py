"""Runs the job grid: one HTTP call per (experiment, sample, question, engine,
repeat), through a per-engine lane that holds its key, concurrency and rate
limit. Responses are cached on disk, so a re-run re-parses without new calls and
a run cut short by a daily quota resumes where it stopped."""

import asyncio
import json
import logging
from dataclasses import dataclass

import httpx
import pandas as pd

from . import cache, config
from .adapters.base import Adapter, DailyLimitError, Prediction, RateLimiter, provider_error
from .adapters.baseline import BaselineAdapter
from .adapters.jev import JevAdapter
from .adapters.llm import LLMAdapter
from .build_experiments import todo_report
from .config import engines, headers, load

log = logging.getLogger(__name__)

# Column order of raw.parquet. Passed explicitly so an empty run still writes the schema.
COLUMNS = [
    "exp",
    "type",
    "sample",
    "question",
    "perturbation",
    "engine",
    "model",
    "repeat",
    "gold",
    "n_options",
    "structured_output",
    "label",
    "value",
    "probs",
    "confidence",
    "correct",
    "valid",
    "latency_ms",
    "tokens_in",
    "tokens_out",
    "cost",
    "status",
    "retries_429",
    "error",
    "cached",
]


@dataclass
class Job:
    exp: str
    exp_type: str
    sample: dict
    question: dict
    engine: str
    repeat: int


def build_jobs(exps: dict, *, exp=None, engine=None, sample=None, question=None, repeats=None) -> list[Job]:
    cfg = load()
    n_rep = repeats or cfg["repeats"]
    jobs: list[Job] = []
    for e, spec in exps.items():
        if exp and e not in exp:
            continue
        for s in spec["samples"]:
            if sample and s["id"] not in sample:
                continue
            for q in s["questions"]:
                if question and q["id"] not in question:
                    continue
                # A disabled engine runs only when named in the filter, for one-off runs.
                for name, espec in engines(include_disabled=bool(engine)).items():
                    if engine and name not in engine:
                        continue
                    if espec["kind"] == "baseline" and q["id"] != "Q1":
                        continue
                    jobs.extend(Job(e, spec["type"], s, q, name, r) for r in range(n_rep))
    return jobs


def check_ready(exps: dict) -> None:
    problems = todo_report(exps)
    if problems:
        listing = "\n  - ".join(problems)
        raise SystemExit(
            f"data/experiments.json is not ready ({len(problems)} issue(s)); no API calls were made:\n  - {listing}"
        )


@dataclass
class Lane:
    """Per-engine call settings: its own key, concurrency and rate limit, and a
    stop flag set when the provider reports an exhausted daily quota."""

    adapter: Adapter
    headers: dict
    sem: asyncio.Semaphore
    limiter: RateLimiter | None = None
    stopped: str | None = None


def _failure(status: int, error: str, deferred: bool = False) -> dict:
    return {
        "response": None,
        "latency_ms": 0.0,
        "status": status,
        "retries_429": 0,
        "error": error[:500],
        "deferred": deferred,
    }


async def _call(lane: Lane, client: httpx.AsyncClient, body: dict) -> dict:
    async with lane.sem:
        if lane.stopped:
            return _failure(429, lane.stopped, deferred=True)
        try:
            return await lane.adapter.call(client, body, lane.headers, lane.limiter)
        except DailyLimitError as e:
            if not lane.stopped:
                lane.stopped = f"daily request limit reached for {lane.adapter.model}; re-run tomorrow ({e})"
                log.warning("  %s: %s", lane.adapter.name, lane.stopped)
            return _failure(429, lane.stopped, deferred=True)
        except Exception as e:
            # -1 marks transport failures (timeout, DNS) that never produced an HTTP status.
            return _failure(getattr(e, "status", -1), str(e))


def _log_response(key: str, body: dict, rec: dict) -> None:
    """Appends every live call, failures included, to responses.jsonl for later inspection."""
    path = config.results_dir() / "responses.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"key": key, "request": body, **rec}, default=str) + "\n")


async def _one(lane: Lane, client: httpx.AsyncClient, job: Job) -> dict | None:
    """One row for raw.parquet, or None for a job deferred by a daily cap: it was
    never attempted, so recording it would count as a failure."""
    adapter = lane.adapter
    s, q = job.sample, job.question
    k = cache.key(adapter.name, job.exp, s["id"], q["id"], job.repeat)
    rec = _cached(adapter, job)
    cached = rec is not None
    log.debug("%s %s", "cache hit " if cached else "calling  ", k)
    if rec is None:
        body = adapter.build_request(q, s["text"], temperature=0.0)
        rec = await _call(lane, client, body)
        _log_response(k, body, rec)
        if rec.get("deferred"):
            return None
        rec = {"request": body, **rec}
        # Failures are not cached, so the next run retries them instead of replaying the error.
        if rec["response"] is not None:
            cache.put(adapter.cache_id, job.exp, k, rec)
    # The recorded request, not the config, says whether a schema was attached: the LLM
    # adapter may have dropped it after a 400. Jev's output is structured by design.
    structured = isinstance(adapter, (JevAdapter, BaselineAdapter)) or "response_format" in (rec.get("request") or {})
    p = (
        adapter.parse(q, rec["response"])
        if rec["response"] is not None
        else Prediction(label=None, error=rec.get("error"))
    )
    return {
        "exp": job.exp,
        "type": job.exp_type,
        "sample": s["id"],
        "question": q["id"],
        "perturbation": s["perturbation"],
        "engine": adapter.name,
        "model": adapter.model,
        "repeat": job.repeat,
        "gold": q["gold"],
        "n_options": len(q["options"]),
        "structured_output": structured,
        "label": p.label,
        "value": p.value,
        "probs": json.dumps(p.probs),
        "confidence": p.confidence,
        "correct": bool(p.valid and p.label == q["gold"]),
        "valid": p.valid,
        "latency_ms": rec["latency_ms"],
        "tokens_in": p.tokens_in,
        "tokens_out": p.tokens_out,
        "cost": p.cost,
        "status": rec["status"],
        "retries_429": rec.get("retries_429", 0),
        "error": p.error,
        "cached": cached,
    }


# The parts of a request that carry the input and question; other fields (schema, routing,
# token limits) may change between versions without invalidating an answer.
INPUT_FIELDS = ("messages", "state", "questions", "text", "task")


def _cached(adapter: Adapter, job: Job) -> dict | None:
    """The cached record, unless it holds a provider error or was made for a
    different input (the sample or question was edited since)."""
    k = cache.key(adapter.name, job.exp, job.sample["id"], job.question["id"], job.repeat)
    rec = cache.get(adapter.cache_id, job.exp, k)
    if rec is None or provider_error(rec.get("response")):
        return None
    body = adapter.build_request(job.question, job.sample["text"], temperature=0.0)
    old = rec.get("request") or {}
    if any(old.get(f) != body.get(f) for f in INPUT_FIELDS):
        return None
    # A schema-constrained answer is stale once the schema is turned off. The reverse is not
    # checked: a cached plain request may be the adapter's fallback after the schema was rejected.
    if "response_format" in old and "response_format" not in body:
        return None
    return rec


def _needs_call(adapter: Adapter, job: Job) -> bool:
    return _cached(adapter, job) is None


def make(name: str, spec: dict) -> Adapter:
    cls = {"jev": JevAdapter, "baseline": BaselineAdapter}.get(spec["kind"], LLMAdapter)
    return cls(name, spec)


def lanes(jobs: list[Job]) -> dict[str, Lane]:
    """Resolves keys up front, and only for engines that will make a live call,
    so a cache-only re-run needs no key and a missing key fails before any call."""
    cfg = load()
    out = {}
    for name, spec in engines(include_disabled=True).items():
        adapter = make(name, spec)
        live = any(j.engine == name and _needs_call(adapter, j) for j in jobs)
        rpm = spec.get("rpm")
        out[name] = Lane(
            adapter=adapter,
            headers=headers(spec) if live and spec["kind"] != "baseline" else {},
            sem=asyncio.Semaphore(spec.get("concurrency", cfg.get("concurrency", 5))),
            limiter=RateLimiter(rpm) if rpm else None,
        )
    return out


async def execute(jobs: list[Job]) -> list[dict]:
    cfg = load()
    by_engine = lanes(jobs)
    async with httpx.AsyncClient(timeout=cfg.get("timeout_s", 60)) as client:
        rows = await asyncio.gather(*(_one(by_engine[j.engine], client, j) for j in jobs))
    deferred = sum(r is None for r in rows)
    if deferred:
        log.warning("  %d job(s) deferred by a daily limit and not recorded; re-run tomorrow to finish", deferred)
    return [r for r in rows if r is not None]


def run(jobs: list[Job]) -> pd.DataFrame:
    rows = asyncio.run(execute(jobs))
    df = pd.DataFrame(rows, columns=COLUMNS)
    for (exp, engine), g in df.groupby(["exp", "engine"]):
        log.info(
            "  %s %-5s %3d calls  %3d cached  %2d invalid  %3d correct",
            exp,
            engine,
            len(g),
            int(g.cached.sum()),
            int((~g.valid).sum()),
            int(g.correct.sum()),
        )
    return save(df)


def save(new: pd.DataFrame) -> pd.DataFrame:
    """Rows for the same (exp, sample, question, engine, repeat) replace older ones,
    so partial re-runs (--exp, --engine ...) merge into one table."""
    out = config.results_dir() / "raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["exp", "sample", "question", "engine", "repeat"]
    if out.exists():
        old = pd.read_parquet(out)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(keys, keep="last")
    new = new.sort_values(keys, kind="stable").reset_index(drop=True)
    new.to_parquet(out, index=False)
    return new
