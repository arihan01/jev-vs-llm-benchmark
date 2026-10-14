"""Pure metric functions over rows of results/raw.parquet. A cell is one
EiSjQp × engine, i.e. its repeats."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CELL = ["exp", "type", "sample", "question", "perturbation", "engine"]


def correct(df: pd.DataFrame) -> np.ndarray:
    # Labels and gold are both option keys (strings), including score levels like "3".
    return (df["valid"].astype(bool) & (df["label"] == df["gold"])).to_numpy()


def conf(df: pd.DataFrame) -> np.ndarray:
    """Confidence of valid answers; invalid answers and a missing value count as 0."""
    return np.where(df["valid"].astype(bool), df["confidence"].astype(float).fillna(0.0), 0.0)


def p_of(df: pd.DataFrame, labels) -> np.ndarray:
    """Probability each row assigned to the given label. `probs` is stored as a JSON
    string; a label absent from the distribution has probability 0."""
    return np.array([json.loads(p).get(lab, 0.0) if p else 0.0 for p, lab in zip(df["probs"], labels, strict=True)])


def prob_spread(probs) -> float:
    """Largest total-variation distance between any two repeats' distributions:
    0 when every repeat returned the same probabilities, 1 when two put all their
    mass on different options. Catches drift that leaves the argmax label unchanged."""
    ds = [json.loads(p) if p else {} for p in probs]
    keys = set().union(*ds) if ds else set()
    return max(
        (0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) for i, a in enumerate(ds) for b in ds[i + 1 :]),
        default=0.0,
    )


def safe_auroc(y_true, score) -> float:
    # AUROC is undefined unless both classes occur in gold.
    y = np.asarray(y_true, int)
    return float(roc_auc_score(y, score)) if len(set(y)) == 2 else float("nan")


def ece(confidence, is_correct, bins: int = 10) -> float:
    """Expected calibration error: per confidence bin, |accuracy - mean confidence|,
    weighted by the bin's share of answers. Confidence 1.0 falls in the top bin."""
    c, y = np.asarray(confidence, float), np.asarray(is_correct, float)
    if not len(c):
        return float("nan")
    idx = np.minimum((c * bins).astype(int), bins - 1)
    return float(
        sum((idx == b).mean() * abs(y[idx == b].mean() - c[idx == b].mean()) for b in range(bins) if (idx == b).any())
    )


def p_yes(df: pd.DataFrame) -> np.ndarray:
    # Noul `value` is P(yes). An invalid answer gets 0.5 (no information), so Brier and
    # AUROC penalise it instead of skipping it.
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
    pg = p_of(g, g["gold"])
    out = {
        "repeats": len(g),
        "gold": g["gold"].iloc[0],
        "labels": "/".join(str(x) for x in g["label"]),
        "correct_majority": bool(y.sum() * 2 > len(g)),
        "correct_rate": float(y.mean()),
        # Every repeat valid and identical; one invalid reply makes the cell inconsistent.
        "consistent": bool(labels.notna().all() and labels.nunique() == 1),
        "invalid": int((~g["valid"].astype(bool)).sum()),
        "prob_spread": prob_spread(g["probs"]),
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


def answered(raw: pd.DataFrame) -> pd.DataFrame:
    """Calls that got an HTTP 200. Quality metrics use only these; transport and
    provider errors are reported separately as error_rate. An unparseable or
    out-of-set answer is still a 200 and still counts as wrong."""
    return raw[raw["status"] == 200]


def cells(raw: pd.DataFrame) -> pd.DataFrame:
    ok = answered(raw)
    rows = [{**dict(zip(CELL, k, strict=True)), **cell_metrics(g)} for k, g in ok.groupby(CELL, sort=True)]
    return pd.DataFrame(rows, columns=None if rows else CELL)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per experiment × engine."""
    cl = cells(raw)
    nan = float("nan")
    rows = []
    for (exp, engine), attempted in raw.groupby(["exp", "engine"], sort=True):
        g = answered(attempted)
        c = cl[(cl.exp == exp) & (cl.engine == engine)] if len(cl) else cl
        typ = attempted["type"].iloc[0]
        has = len(g) > 0
        rec = {
            "exp": exp,
            "type": typ,
            "engine": engine,
            "model": attempted["model"].iloc[0],
            "questions": len(c),
            "attempted": len(attempted),
            "answered": len(g),
            "error_rate": float((attempted["status"] != 200).mean()),
            "accuracy": float(correct(g).mean()) if has else nan,
            "majority_accuracy": float(c["correct_majority"].mean()) if has else nan,
            "consistency": float(c["consistent"].mean()) if has else nan,
            # 1 - mean spread of the repeats' probability distributions; 1 = identical every time.
            "prob_consistency": float(1 - c["prob_spread"].mean()) if has else nan,
            "invalid_rate": float(1 - g["valid"].astype(bool).mean()) if has else nan,
            "invalid_reasons": "; ".join(
                f"{n}× {e}"
                for e, n in g.reindex(columns=["error"])
                .loc[~g["valid"].astype(bool), "error"]
                .fillna("?")
                .value_counts()
                .items()
            ),
            "mean_latency_ms": float(g["latency_ms"].mean()) if has else nan,
            # quantile of an empty series is already NaN, so no `has` guard is needed here
            "p50_latency_ms": float(g["latency_ms"].quantile(0.5)),
            "p95_latency_ms": float(g["latency_ms"].quantile(0.95)),
            "tokens_in": int(attempted["tokens_in"].sum()),
            "tokens_out": int(attempted["tokens_out"].sum()),
            "total_cost": float(attempted["cost"].sum()),
            "cost_per_100_questions": float(g["cost"].mean() * 100) if has else nan,
            "cost_per_correct": float(g["cost"].sum() / correct(g).sum()) if has and correct(g).any() else nan,
            "mean_p_gold": float(p_of(g, g["gold"]).mean()) if has else nan,
            "ece": ece(conf(g), correct(g)),
        }
        if typ == "noul":
            gold = (g["gold"] == "yes").astype(int).to_numpy()
            rec["auroc"] = safe_auroc(gold, p_yes(g)) if has else nan
            rec["brier"] = float(((p_yes(g) - gold) ** 2).mean()) if has else nan
        if typ == "score":
            rec["mae"] = float(abs_err(g).mean()) if has else nan
        rows.append(rec)
    return pd.DataFrame(rows)


def accuracy_by_perturbation(raw: pd.DataFrame) -> pd.DataFrame:
    ok = answered(raw)
    return (
        ok.assign(correct=correct(ok))
        .groupby(["engine", "perturbation"])["correct"]
        .mean()
        .rename("accuracy")
        .reset_index()
    )
