"""results/raw.parquet -> results/cells.csv, scorecard.csv, verdict.md"""

import logging

import numpy as np
import pandas as pd

from . import metrics
from .config import RESULTS, setup_logging

log = logging.getLogger(__name__)

ACCURACY_TOLERANCE = 0.05  # with 30 calls per cell group, smaller gaps are noise
NAMES = {"choice": "Choice (E1 · Banking77)", "noul": "Noul (E2 · BoolQ)", "score": "Score (E3 · Yelp)"}


def load_raw() -> pd.DataFrame:
    p = RESULTS / "raw.parquet"
    if not p.exists():
        raise SystemExit("results/raw.parquet missing; run `make bench` first.")
    return pd.read_parquet(p)


def _f(x, pct=False, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:.0f}%" if pct else f"{x:.{digits}g}"


def verdict(sc: pd.DataFrame, raw: pd.DataFrame) -> str:
    lines = [
        "# Verdict: Jev vs LLMs",
        "",
        f"Rule: pick the cheapest engine whose accuracy is within {ACCURACY_TOLERANCE * 100:.0f} points of the "
        "best engine on that experiment. Each experiment is 5 perturbed samples × 2 questions × 3 repeats "
        "= 30 calls per engine, so treat differences of one or two answers as noise.",
        "",
    ]
    for exp, t in sc.groupby("exp", sort=True):
        best = t.accuracy.max()
        ok = t[t.accuracy >= best - ACCURACY_TOLERANCE].sort_values("cost_per_100_questions")
        pick = ok.iloc[0]
        lines += [
            f"## {NAMES.get(t.type.iloc[0], exp)}",
            "",
            f"**Use `{pick.engine}` ({pick.model}).** Accuracy {_f(pick.accuracy, True)}, "
            f"p50 {_f(pick.p50_latency_ms, digits=4)} ms, {pick.tokens_in + pick.tokens_out:,} tokens, "
            f"${pick.cost_per_100_questions:.4f} per 100 questions, invalid {_f(pick.invalid_rate, True)}, "
            f"errors {_f(pick.error_rate, True)}.",
            "",
            "| engine | accuracy | consistent | p50 ms | tokens in/out | $ / 100 q | invalid | inj. success |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, r in t.iterrows():
            lines.append(
                f"| {r.engine} | {_f(r.accuracy, True)} | {_f(r.consistency, True)} | "
                f"{_f(r.p50_latency_ms, digits=4)} | {r.tokens_in:,} / {r.tokens_out:,} | "
                f"{r.cost_per_100_questions:.4f} | {_f(r.invalid_rate, True)} | "
                f"{_f(r.get('injection_success'), True)} |"
            )
        extra = [f"{r.engine} MAE {_f(r.mae)}" for _, r in t.iterrows() if "mae" in r and pd.notna(r.get("mae"))]
        extra += [
            f"{r.engine} AUROC {_f(r.auroc)} / Brier {_f(r.brier)}"
            for _, r in t.iterrows()
            if "auroc" in r and pd.notna(r.get("brier"))
        ]
        if extra:
            lines += ["", "; ".join(extra) + "."]
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- LLM confidence is self-reported (asked for in the JSON), not a probability; Jev's is a model "
        "probability. Compare calibration with that in mind.",
        "- Jev's run-to-run variance shows up in the `consistent` column; it is reported, not hidden.",
        "- Invalid LLM outputs count as wrong (and as maximum error for Score MAE).",
        "- Q2 questions and their gold answers are hand-written; Q1 uses the dataset's human label.",
        "- Latency is measured over the shared OpenRouter gateway at concurrency 5; compare engines, not absolutes.",
    ]
    return "\n".join(lines) + "\n"


def main():
    raw = load_raw()
    cells = metrics.cells(raw)
    sc = metrics.aggregate(raw)
    cells.to_csv(RESULTS / "cells.csv", index=False)
    sc.to_csv(RESULTS / "scorecard.csv", index=False)
    (RESULTS / "verdict.md").write_text(verdict(sc, raw))
    print(
        sc[["exp", "engine", "accuracy", "consistency", "p50_latency_ms", "tokens_in", "total_cost"]].to_string(
            index=False
        )
    )
    log.info("wrote results/cells.csv, results/scorecard.csv, results/verdict.md")


if __name__ == "__main__":
    setup_logging()
    main()
