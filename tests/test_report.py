import math

import pandas as pd
import pytest

from bench import report


def row(engine, typ="choice", accuracy=0.5, cost=0.01, **extra):
    return {
        "exp": {"choice": "E1", "noul": "E2", "score": "E3"}[typ],
        "type": typ,
        "engine": engine,
        "model": f"{engine}-model",
        "accuracy": accuracy,
        "consistency": 1.0,
        "p50_latency_ms": 120.0,
        "tokens_in": 1000,
        "tokens_out": 20,
        "cost_per_100_questions": cost,
        "invalid_rate": 0.0,
        "error_rate": 0.0,
        "injection_success": float("nan"),
        **extra,
    }


def test_picks_cheapest_within_tolerance():
    sc = pd.DataFrame([row("jev", accuracy=0.80, cost=0.001), row("high", accuracy=0.84, cost=0.5)])
    assert "**Use `jev`" in report.verdict(sc, pd.DataFrame())


def test_picks_best_when_cheap_engine_is_too_far_behind():
    sc = pd.DataFrame([row("jev", accuracy=0.60, cost=0.001), row("high", accuracy=0.90, cost=0.5)])
    assert "**Use `high`" in report.verdict(sc, pd.DataFrame())


def test_nan_formats_as_na_and_type_specific_lines():
    sc = pd.DataFrame(
        [
            row("jev", "noul", auroc=0.9, brier=0.1, mae=math.nan),
            row("jev", "score", mae=0.75, auroc=math.nan, brier=math.nan),
            row("jev", "choice", mae=math.nan, auroc=math.nan, brier=math.nan),
        ]
    )
    text = report.verdict(sc, pd.DataFrame())
    noul, score, choice = (text.split(f"## {report.NAMES[t]}")[1].split("##")[0] for t in ("noul", "score", "choice"))
    assert "AUROC 0.9 / Brier 0.1" in noul and "MAE" not in noul
    assert "MAE 0.75" in score and "AUROC" not in score
    assert "MAE" not in choice and "AUROC" not in choice
    assert "| n/a |" in choice


@pytest.mark.parametrize("value, pct, out", [(0.123, True, "12%"), (1234.5, False, "1.23e+03"), (None, False, "n/a")])
def test_formatter(value, pct, out):
    assert report._f(value, pct) == out


def test_main_writes_three_files(sandbox, ready_experiments):
    from bench import runner

    tmp, _ = sandbox
    runner.run(runner.build_jobs(ready_experiments, repeats=1))
    report.main()
    for name in ("cells.csv", "scorecard.csv", "verdict.md"):
        assert (tmp / "results" / name).stat().st_size > 0


def test_missing_raw_exits(sandbox):
    with pytest.raises(SystemExit, match="make bench"):
        report.load_raw()
