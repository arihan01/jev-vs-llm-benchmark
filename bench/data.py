"""Sample 300 items per task (stratified, seed 42) and freeze to data/{task}.jsonl.
Experiments draw their samples from the head of each file, so they are fixed too."""

import json
import logging
import re

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .config import DATA, TASKS, load, setup_logging

log = logging.getLogger(__name__)

BANKING77_CSV = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"


def norm_intent(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", name.lower())


def _raw(task: str) -> pd.DataFrame:
    spec = load()["tasks"][task]
    if task == "choice":
        # PolyAI/banking77 on the Hub is a loading script, which `datasets` no longer
        # runs; the script itself downloads this CSV, so it is the same data.
        ds = load_dataset("csv", data_files=BANKING77_CSV, split="train")
        df = ds.to_pandas()
        return pd.DataFrame({"text": df["text"], "gold": df["category"].map(norm_intent)})
    ds = load_dataset(spec["dataset"], split=spec["split"]).to_pandas()
    if task == "score":
        n = spec.get("truncate_chars", 2000)
        return pd.DataFrame({"text": ds["text"].str.slice(0, n), "gold": ds["label"].astype(str)})
    return pd.DataFrame(
        {
            "text": [{"passage": p, "question": q} for p, q in zip(ds["passage"], ds["question"], strict=True)],
            "gold": ds["answer"].map({True: "yes", False: "no"}),
        }
    )


def sample(task: str, n: int, seed: int) -> list[dict]:
    df = _raw(task)
    df["item_id"] = [f"{task}-{i}" for i in df.index]
    picked, _ = train_test_split(df, train_size=n, stratify=df["gold"], random_state=seed)
    return [{"item_id": r.item_id, "text": r.text, "gold": r.gold} for r in picked.itertuples()]


def path(task: str):
    return DATA / f"{task}.jsonl"


def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(p) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def items(task: str) -> list[dict]:
    if not path(task).exists():
        raise SystemExit(f"{path(task)} missing; run `make data` first.")
    return read_jsonl(path(task))


def main(force: bool = False):
    cfg = load()
    for task in TASKS:
        if path(task).exists() and not force:
            log.info("%s already frozen; skipping (use --force to resample)", path(task).name)
            continue
        rows = sample(task, cfg.get("n_items", 300), cfg["seed"])
        write_jsonl(path(task), rows)
        log.info("wrote %d items -> %s", len(rows), path(task).relative_to(DATA.parent))


if __name__ == "__main__":
    import sys

    setup_logging()
    main(force="--force" in sys.argv)
