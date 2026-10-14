"""Config file selection, engine endpoint and key resolution. The active config
is chosen once per process: --config, then $BENCH_CONFIG, then config.yaml."""

import logging
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / ".cache"
TASKS = ("choice", "score", "noul")
DEFAULT_KEY_ENV = "OPENROUTER_API_KEY"
# The example values in .env.example. A copied-but-unedited file must fail like a missing key.
PLACEHOLDERS = ("sk-or-v1-...", "apikey_...", "sk-proj-...")

load_dotenv(ROOT / ".env")

_config_path: Path | None = None


def use_config(path: str | Path | None) -> None:
    """Select the config file for this process (CLI --config). Without it,
    BENCH_CONFIG or config.yaml is used."""
    global _config_path
    _config_path = Path(path).resolve() if path else None


def config_path() -> Path:
    if _config_path:
        return _config_path
    env = os.environ.get("BENCH_CONFIG")
    return Path(env).resolve() if env else ROOT / "config.yaml"


def load() -> dict:
    return _load(config_path())


# Cached per path, so tests that switch configs within one process get a fresh load.
@lru_cache
def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def results_dir() -> Path:
    return ROOT / load().get("results_dir", "results")


def engines(include_disabled: bool = False) -> dict:
    """Engines in config order, skipping any marked `enabled: false` unless asked."""
    return {k: v for k, v in load()["engines"].items() if include_disabled or v.get("enabled", True)}


def engine_order(names) -> list[str]:
    """Configured engines in config order, then any others sorted."""
    names, known = set(names), list(engines(include_disabled=True))
    return [e for e in known if e in names] + sorted(names - set(known))


def engine_url(spec: dict) -> str:
    """Full endpoint for an engine. `base_url` is the complete endpoint for Jev
    and the API base (…/v1) for chat models."""
    cfg = load()
    if spec.get("kind") == "jev":
        return spec.get("base_url") or cfg["jev_decisions_url"]
    return (spec.get("base_url") or cfg["openrouter_base"]).rstrip("/") + "/chat/completions"


def is_openrouter(spec: dict) -> bool:
    return (urlparse(engine_url(spec)).hostname or "").endswith("openrouter.ai")


def is_free(spec: dict) -> bool:
    return is_openrouter(spec) and str(spec.get("model", "")).endswith(":free")


def api_key(env: str = DEFAULT_KEY_ENV) -> str:
    key = os.environ.get(env, "")
    if not key or key in PLACEHOLDERS:
        raise SystemExit(f"{env} is not set. Copy .env.example to .env and add your key.")
    return key


def headers(spec: dict | None = None) -> dict:
    spec = spec or {"kind": "llm"}
    return {
        "Authorization": f"Bearer {api_key(spec.get('api_key_env', DEFAULT_KEY_ENV))}",
        "Content-Type": "application/json",
    }


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
