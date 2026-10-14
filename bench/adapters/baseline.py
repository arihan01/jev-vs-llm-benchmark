"""Standard statistical classifier as an engine: TF-IDF + logistic regression
trained on each dataset's own train split (the benchmark samples come from the
test/validation splits). Runs locally, needs no key, and only answers Q1, since
Q2 is a hand-written question with no training data."""

import asyncio
import pickle
import time
from functools import lru_cache

import pandas as pd

from .. import prompts
from ..config import CACHE, load
from .base import Adapter, Prediction

BANKING77_TRAIN = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
MAX_TRAIN = 20_000  # Yelp's train split is 650k reviews; a seeded sample keeps training to seconds.


def as_text(x) -> str:
    return f"{x['question']} {x['passage']}" if isinstance(x, dict) else x


def train_data(task: str) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    from ..data import norm_intent

    if task == "choice":
        df = pd.read_csv(BANKING77_TRAIN)
        return list(df["text"]), list(df["category"].map(norm_intent))
    spec = load()["tasks"][task]
    ds = load_dataset(spec["dataset"], split="train").to_pandas()
    ds = ds.sample(min(len(ds), MAX_TRAIN), random_state=load()["seed"])
    if task == "score":
        return list(ds["text"].str.slice(0, spec.get("truncate_chars", 2000))), list(ds["label"].astype(str))
    return [f"{q} {p}" for q, p in zip(ds["question"], ds["passage"], strict=True)], list(
        ds["answer"].map({True: "yes", False: "no"})
    )


@lru_cache
def model(task: str):
    path = CACHE / "baseline" / f"{task}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2), LogisticRegression(max_iter=2000)
    )
    pipe.fit(*train_data(task))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipe, f)
    return pipe


class BaselineAdapter(Adapter):
    def build_request(self, q, text, temperature=None):
        return {"model": self.model, "task": q["type"], "text": text}

    async def call(self, client, body, headers, limiter=None) -> dict:
        pipe = await asyncio.to_thread(model, body["task"])
        t0 = time.perf_counter()
        probs = pipe.predict_proba([as_text(body["text"])])[0]
        latency = (time.perf_counter() - t0) * 1000
        resp = {"probabilities": {str(c): float(p) for c, p in zip(pipe.classes_, probs, strict=True)}}
        return {"response": resp, "latency_ms": latency, "status": 200, "retries_429": 0}

    def parse(self, q, response) -> Prediction:
        keys = prompts.option_keys(q)
        p = Prediction(label=None, probs={k: response["probabilities"].get(k, 0.0) for k in keys})
        p.label = max(p.probs, key=p.probs.__getitem__)
        p.valid, p.confidence = True, p.probs[p.label]
        if q["type"] == "score":
            p.value = sum(int(k) * v for k, v in p.probs.items())
        elif q["type"] == "noul":
            p.value = p.probs["yes"]
        return p
