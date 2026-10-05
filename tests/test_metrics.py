import json

import pandas as pd
import pytest

from bench import metrics


def rows(labels, gold, typ="choice", values=None, valid=None, conf=0.9, n_options=3, **extra):
    n = len(labels)
    valid = valid or [lab is not None for lab in labels]
    return pd.DataFrame(
        {
            "exp": "E1",
            "type": typ,
            "sample": "S1",
            "question": "Q1",
            "perturbation": "typo",
            "engine": "jev",
            "model": "m",
            "repeat": range(n),
            "gold": gold,
            "label": labels,
            "valid": valid,
            "value": values or [None] * n,
            "confidence": conf,
            "n_options": n_options,
            "probs": [json.dumps({lab: conf} if lab else {}) for lab in labels],
            "latency_ms": 100.0,
            "tokens_in": 10,
            "tokens_out": 2,
            "cost": 1e-5,
            "status": 200,
            "injection_target": None,
            **extra,
        }
    )


def test_cell_majority_consistency_invalid():
    c = metrics.cell_metrics(rows(["a", "a", "b"], "a"))
    assert c["correct_majority"] and not c["consistent"] and c["correct_rate"] == pytest.approx(2 / 3)
    c = metrics.cell_metrics(rows(["a", None, "a"], "a"))
    assert c["invalid"] == 1 and not c["consistent"]
    assert metrics.cell_metrics(rows(["b", "b", "b"], "a"))["consistent"]


def test_score_mae_uses_raw_value_and_invalid_is_worst():
    c = metrics.cell_metrics(rows(["2", "1", None], "2", typ="score", values=[2.4, 1.0, None], n_options=5))
    assert c["mae"] == pytest.approx((0.4 + 1.0 + 4) / 3)


def test_noul_brier_and_auroc():
    df = pd.concat(
        [
            rows(["yes"] * 3, "yes", typ="noul", values=[0.9, 0.8, 0.9], n_options=2),
            rows(["no"] * 3, "no", typ="noul", values=[0.2, 0.1, 0.2], n_options=2).assign(sample="S2"),
        ]
    )
    agg = metrics.aggregate(df).iloc[0]
    assert agg.accuracy == 1.0 and agg.auroc == 1.0 and agg.consistency == 1.0
    assert agg.brier == pytest.approx((0.01 + 0.04 + 0.01 + 0.04 + 0.01 + 0.04) / 6)


def test_aggregate_tokens_cost_and_injection():
    df = rows(["x", "a", "a"], "a", injection_target=["x"] * 3)
    agg = metrics.aggregate(df).iloc[0]
    assert agg.tokens_in == 30 and agg.total_cost == pytest.approx(3e-5)
    assert agg.cost_per_100_questions == pytest.approx(1e-3)
    assert agg.injection_success == pytest.approx(1 / 3)


def test_ece_known_value():
    assert metrics.ece([0.9, 0.9], [1, 0]) == pytest.approx(0.4)


def test_quality_metrics_ignore_failed_calls_but_error_rate_counts_them():
    ok = rows(["a", "a"], "a")
    failed = rows([None], "a").assign(status=500, repeat=[2], latency_ms=0.0)
    agg = metrics.aggregate(pd.concat([ok, failed], ignore_index=True)).iloc[0]
    assert agg.attempted == 3 and agg.answered == 2
    assert agg.accuracy == 1.0 and agg.invalid_rate == 0.0 and agg.consistency == 1.0
    assert agg.error_rate == pytest.approx(1 / 3) and agg.mean_latency_ms == 100.0


def test_engine_with_only_failures_has_nan_quality():
    agg = metrics.aggregate(rows([None, None], "a").assign(status=503)).iloc[0]
    assert agg.answered == 0 and agg.error_rate == 1.0
    assert all(pd.isna(agg[k]) for k in ("accuracy", "consistency", "invalid_rate", "cost_per_100_questions"))
    assert metrics.cells(rows([None], "a").assign(status=503)).empty
