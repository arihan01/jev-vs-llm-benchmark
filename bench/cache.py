import hashlib
import json

from .config import CACHE


def key(engine: str, exp: str, sample: str, question: str, repeat: int) -> str:
    return f"{engine}|{exp}|{sample}|{question}|{repeat}"


def _path(engine: str, scope: str, k: str):
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
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"key": k, **record}, f)
    tmp.replace(p)
