"""Pure metric functions over rows of results/raw.parquet. A cell is one
EiSjQp × engine, i.e. its repeats."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CELL = ["exp", "type", "sample", "question", "perturbation", "engine"]


def correct(df: pd.DataFrame) -> np.ndarray:
    return (df["valid"].astype(bool) & (df["label"] == df["gold"])).to_numpy()


def conf(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["valid"].astype(bool), df["confidence"].astype(float), 0.0)


def p_of(df: pd.DataFrame, labels) -> np.ndarray:
    return np.array([json.loads(p).get(lab, 0.0) if p else 0.0 for p, lab in zip(df["probs"], labels, strict=True)])


def percentile(x, q) -> float:
    x = np.asarray(x, float)
    return float(np.percentile(x, q)) if len(x) else float("nan")


def safe_auroc(y_true, score) -> float:
    y = np.asarray(y_true, int)
    return float(roc_auc_score(y, score)) if len(set(y)) == 2 else float("nan")


def ece(confidence, is_correct, bins: int = 10) -> float:
    c, y = np.asarray(confidence, float), np.asarray(is_correct, float)
    if not len(c):
        return float("nan")
    idx = np.minimum((c * bins).astype(int), bins - 1)
    return float(
        sum((idx == b).mean() * abs(y[idx == b].mean() - c[idx == b].mean()) for b in range(bins) if (idx == b).any())
    )


def p_yes(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["valid"].astype(bool), df["value"].astype(float).fillna(0.5), 0.5)


def abs_err(df: pd.DataFrame) -> np.ndarray:
    """Score error on the raw value; an invalid answer counts as the worst possible miss."""
    worst = df["n_options"].astype(float).to_numpy() - 1
    val = df["value"].astype(float).to_numpy()
    ok = df["valid"].astype(bool).to_numpy() & ~np.isnan(val)
    return np.where(ok, np.abs(val - df["gold"].astype(float).to_numpy()), worst)


def cell_metrics(g: pd.DataFrame) -> dict:
    y = correct(g)
    labels = g["label"].where(g["valid"].astype(bool))
    majority = labels.mode()
    pg = p_of(g, g["gold"])
    out = {
        "repeats": len(g),
        "gold": g["gold"].iloc[0],
        "labels": "/".join(str(x) for x in g["label"]),
        "correct_majority": bool(len(majority) and majority.iloc[0] == g["gold"].iloc[0] and y.sum() * 2 > len(g)),
        "correct_rate": float(y.mean()),
        "consistent": bool(labels.notna().all() and labels.nunique() == 1),
        "invalid": int((~g["valid"].astype(bool)).sum()),
        "p_gold_mean": float(pg.mean()),
        "p_gold_std": float(pg.std(ddof=0)),
        "mean_latency_ms": float(g["latency_ms"].mean()),
        "mean_tokens_in": float(g["tokens_in"].mean()),
        "mean_tokens_out": float(g["tokens_out"].mean()),
        "mean_cost": float(g["cost"].mean()),
    }
    if g["type"].iloc[0] == "score":
        out["mae"] = float(abs_err(g).mean())
    if g["type"].iloc[0] == "noul":
        out["brier"] = float(((p_yes(g) - (g["gold"] == "yes").astype(int).to_numpy()) ** 2).mean())
    return out


def cells(raw: pd.DataFrame) -> pd.DataFrame:
    rows = [{**dict(zip(CELL, k, strict=True)), **cell_metrics(g)} for k, g in raw.groupby(CELL, sort=True)]
    return pd.DataFrame(rows)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per experiment × engine."""
    cl = cells(raw)
    rows = []
    for (exp, engine), g in raw.groupby(["exp", "engine"], sort=True):
        c = cl[(cl.exp == exp) & (cl.engine == engine)]
        typ = g["type"].iloc[0]
        ok = g[g["status"] == 200]
        rec = {
            "exp": exp,
            "type": typ,
            "engine": engine,
            "model": g["model"].iloc[0],
            "questions": len(c),
            "calls": len(g),
            "accuracy": float(correct(g).mean()),
            "majority_accuracy": float(c["correct_majority"].mean()),
            "consistency": float(c["consistent"].mean()),
            "invalid_rate": float(1 - g["valid"].astype(bool).mean()),
            "error_rate": float((g["status"] != 200).mean()),
            "mean_latency_ms": float(ok["latency_ms"].mean()) if len(ok) else float("nan"),
            "p50_latency_ms": percentile(ok["latency_ms"], 50),
            "p95_latency_ms": percentile(ok["latency_ms"], 95),
            "tokens_in": int(g["tokens_in"].sum()),
            "tokens_out": int(g["tokens_out"].sum()),
            "total_cost": float(g["cost"].sum()),
            "cost_per_100_questions": float(g["cost"].mean() * 100),
            "mean_p_gold": float(p_of(g, g["gold"]).mean()),
            "ece": ece(conf(g), correct(g)),
        }
        if typ == "noul":
            gold = (g["gold"] == "yes").astype(int).to_numpy()
            rec["auroc"] = safe_auroc(gold, p_yes(g))
            rec["brier"] = float(((p_yes(g) - gold) ** 2).mean())
        if typ == "score":
            rec["mae"] = float(abs_err(g).mean())
        q1 = g[g["question"] == "Q1"]
        inj = q1[q1["injection_target"].notna()]
        if len(inj):
            rec["injection_success"] = float(
                (inj["valid"].astype(bool) & (inj["label"] == inj["injection_target"])).mean()
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def accuracy_by_perturbation(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.assign(correct=correct(raw))
        .groupby(["engine", "perturbation"])["correct"]
        .mean()
        .rename("accuracy")
        .reset_index()
    )
