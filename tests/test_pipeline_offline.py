"""All 270 calls against the fake gateway: row count, gold-null guard, caching,
cache-only reproducibility and failure rows."""

import copy

import pandas as pd
import pytest

from bench import build_experiments, metrics, report, runner
from bench.adapters.base import Adapter, HTTPStatusError


def test_unfinished_q2_aborts_before_any_call(sandbox):
    _, calls = sandbox
    with pytest.raises(SystemExit, match="gold is null"):
        runner.check_ready(build_experiments.build())
    assert calls["n"] == 0


def test_full_grid(sandbox, ready_experiments):
    tmp, calls = sandbox
    runner.check_ready(ready_experiments)
    raw = runner.run(runner.build_jobs(ready_experiments))
    assert len(raw) == 270 and calls["n"] == 270
    assert set(raw.groupby(["exp", "engine"]).size()) == {30}
    sc = metrics.aggregate(raw)
    assert len(sc) == 9 and len(metrics.cells(raw)) == 90
    assert "Verdict" in report.verdict(sc, raw)

    (tmp / "results" / "raw.parquet").unlink()
    raw2 = runner.run(runner.build_jobs(copy.deepcopy(ready_experiments)))
    assert calls["n"] == 270, "re-run must be served entirely from cache"
    assert raw2.cached.all()
    pd.testing.assert_frame_equal(sc, metrics.aggregate(raw2))


def test_partial_rerun_merges(sandbox, ready_experiments):
    runner.run(runner.build_jobs(ready_experiments, exp=["E1"], engine=["jev"]))
    raw = runner.run(runner.build_jobs(ready_experiments, exp=["E2"], engine=["jev"], repeats=1))
    assert set(raw.exp) == {"E1", "E2"} and len(raw) == 30 + 10


def test_failing_engine_yields_invalid_rows_and_report_still_builds(sandbox, ready_experiments, monkeypatch):
    tmp, _ = sandbox
    working = Adapter.call

    async def flaky(self, client, body, headers):
        if self.name == "high":
            raise HTTPStatusError(500, "upstream exploded")
        return await working(self, client, body, headers)

    monkeypatch.setattr(Adapter, "call", flaky)
    raw = runner.run(runner.build_jobs(ready_experiments, exp=["E1"], repeats=1))
    high = raw[raw.engine == "high"]
    assert len(high) == 10 and not high.valid.any() and (high.status == 500).all()
    assert high.error.str.contains("HTTP 500").all() and not high.correct.any()
    report.main()
    sc = pd.read_csv(tmp / "results" / "scorecard.csv")
    assert sc.loc[sc.engine == "high", "error_rate"].item() == 1.0
    assert not (tmp / "cache" / "high").exists(), "failed calls must not be cached"
