import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

from bench import metrics, runner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dashboard"))
import figures


@pytest.fixture
def results(sandbox, ready_experiments):
    raw = runner.run(runner.build_jobs(ready_experiments))
    return raw, metrics.aggregate(raw)


def test_correctness_grid_shape(results):
    raw, _ = results
    fig = figures.correctness_grid(raw, "E1")
    heat = fig.data[0]
    assert isinstance(fig, go.Figure)
    assert [x.split("<br>")[0] for x in heat.x] == [f"S{s}Q{q}" for s in range(1, 6) for q in (1, 2)]
    assert list(heat.y) == ["jev", "low", "high"]
    assert all(set(cell.replace(" ", "")) <= {"✓", "✗"} for row in heat.text for cell in row)


def test_every_figure_builds(results):
    raw, sc = results
    for fig in (
        figures.latency_box(raw),
        figures.tokens_cost(sc),
        figures.consistency_bars(sc),
        figures.accuracy_by_perturbation(raw),
    ):
        assert isinstance(fig, go.Figure) and fig.data
