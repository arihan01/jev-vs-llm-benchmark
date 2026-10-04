import asyncio
import json
import logging
from dataclasses import dataclass

import httpx
import pandas as pd

from . import cache
from .adapters import Prediction, make
from .build_experiments import todo_report
from .config import RESULTS, engines, headers, load

log = logging.getLogger(__name__)

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
    "injection_target",
    "n_options",
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
                for name in engines():
                    if engine and name not in engine:
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


async def _one(adapter, client, hdrs, job: Job, sem: asyncio.Semaphore) -> dict:
    s, q = job.sample, job.question
    k = cache.key(adapter.name, job.exp, s["id"], q["id"], job.repeat)
    rec = cache.get(adapter.name, job.exp, k)
    cached = rec is not None
    log.debug("%s %s", "cache hit " if cached else "calling  ", k)
    if rec is None:
        body = adapter.build_request(q, s["text"], temperature=0.0)
        async with sem:
            try:
                rec = await adapter.call(client, body, hdrs)
            except Exception as e:
                rec = {
                    "response": None,
                    "latency_ms": 0.0,
                    "status": getattr(e, "status", -1),
                    "retries_429": 0,
                    "error": str(e)[:500],
                }
        if rec["response"] is not None:
            cache.put(adapter.name, job.exp, k, {"request": body, **rec})
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
        "injection_target": s.get("injection_target") if q["id"] == "Q1" else None,
        "n_options": len(q["options"]),
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


async def execute(jobs: list[Job]) -> list[dict]:
    cfg = load()
    adapters = {name: make(name, spec) for name, spec in engines().items()}
    sem = asyncio.Semaphore(cfg.get("concurrency", 5))
    hdrs = headers() if any(_needs_call(adapters[j.engine], j) for j in jobs) else {}
    async with httpx.AsyncClient(timeout=cfg.get("timeout_s", 60)) as client:
        return list(await asyncio.gather(*(_one(adapters[j.engine], client, hdrs, j, sem) for j in jobs)))


def _needs_call(adapter, job: Job) -> bool:
    return (
        cache.get(
            adapter.name, job.exp, cache.key(adapter.name, job.exp, job.sample["id"], job.question["id"], job.repeat)
        )
        is None
    )


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
    out = RESULTS / "raw.parquet"
    RESULTS.mkdir(exist_ok=True)
    keys = ["exp", "sample", "question", "engine", "repeat"]
    if out.exists():
        old = pd.read_parquet(out)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(keys, keep="last")
    new = new.sort_values(keys, kind="stable").reset_index(drop=True)
    new.to_parquet(out, index=False)
    return new
