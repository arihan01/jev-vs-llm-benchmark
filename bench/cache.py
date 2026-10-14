"""Disk cache of raw provider responses, one JSON file per call, under
.cache/<engine>/<model>/<experiment>/. Re-runs re-parse from here instead of
calling again, which is what makes partial runs resumable."""

import hashlib
import json

from .config import CACHE


def key(engine: str, exp: str, sample: str, question: str, repeat: int) -> str:
    return f"{engine}|{exp}|{sample}|{question}|{repeat}"


def _path(engine: str, scope: str, k: str):
    # The key contains '|' and ids; hashing gives a filesystem-safe, fixed-length name.
    return CACHE / engine / scope / (hashlib.sha1(k.encode()).hexdigest() + ".json")


def get(engine: str, scope: str, k: str) -> dict | None:
    p = _path(engine, scope, k)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def put(engine: str, scope: str, k: str, record: dict) -> None:
    p = _path(engine, scope, k)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so an interrupted run never leaves a half-written file that a
    # later run would read as a cache hit.
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"key": k, **record}, f)
    tmp.replace(p)
