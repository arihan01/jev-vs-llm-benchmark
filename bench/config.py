import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"
TASKS = ("choice", "score", "noul")

load_dotenv(ROOT / ".env")


@lru_cache
def load() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key or key.startswith("sk-or-v1-..."):
        raise SystemExit("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")
    return key


def headers() -> dict:
    h = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}
    if url := os.environ.get("OPENROUTER_APP_URL"):
        h["HTTP-Referer"] = url
    if name := os.environ.get("OPENROUTER_APP_NAME"):
        h["X-Title"] = name
    return h


def engines() -> dict:
    return dict(load()["engines"])


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
