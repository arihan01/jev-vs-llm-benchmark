import pandas as pd

from bench import export_md, runner


def test_table_escapes_pipes_and_handles_empty():
    df = pd.DataFrame({"a": ["x|y"], "b": [float("nan")], "c": [True]})
    lines = export_md.table(df, headers={"a": "A"})
    assert lines[0] == "| A | b | c |" and lines[2] == "| x\\|y |  | yes |"
    assert export_md.table(pd.DataFrame()) == ["_none_", ""]


def test_marks_one_per_repeat():
    assert export_md._marks("a/b/a", "a") == "✓✗✓"


def test_export_reads_like_the_dashboard(sandbox, ready_experiments, monkeypatch):
    tmp, _ = sandbox
    monkeypatch.setattr(export_md, "load_experiments", lambda: ready_experiments)
    runner.run(runner.build_jobs(ready_experiments))
    md = export_md.main(["--out", str(tmp / "report.md")]).read_text()
    for heading in (
        "## 1. The experiment",
        "## 2. What is measured",
        "## 3. The inputs: text types and questions",
        "## 4. Run report",
        "### What was run",
        "### Headline numbers",
        "### Verdict (cost rule)",
        "#### Correct vs T per sample × question",
        "#### What stands out",
        "### Speed and cost",
        "### Accuracy by type of change",
        "## 5. How the core is implemented",
        "## 6. Observations",
    ):
        assert heading in md
    assert "Calls recorded: **270 / 270**" in md and "The grid is complete." in md
    # the inputs section shows one example per experiment with its truth
    assert md.count("Text as sent to the engines:") == 3 and "**T = `" in md
    # code excerpts are real source, with file:line pointers
    assert "```python" in md and "bench/prompts.py:" in md and "def llm_messages(" in md
    # stage 2 is a placeholder until observations.md exists
    assert "_Not written yet._" in md
    assert "mean P(T)" in md and "$ / 100 q" in md
    # the text heatmap: one row per engine per experiment
    assert md.count("| **jev** |") == 3
    assert "| _T_ |" in md and "| _change_ |" in md
    # aggregates only: no collapsible appendices, no per-call or per-cell rows
    assert "<details>" not in md
    assert "Every call" not in md and "Cell detail" not in md and "Inputs and questions" not in md
    assert "| S1 | Q1 |" not in md
    assert "**Prompt injection**" in md
    assert "raw.parquet" in md


def test_export_marks_incomplete_grid(sandbox, ready_experiments, monkeypatch):
    tmp, _ = sandbox
    monkeypatch.setattr(export_md, "load_experiments", lambda: ready_experiments)
    runner.run(runner.build_jobs(ready_experiments, exp=["E1"], engine=["jev"]))
    md = export_md.main(["--out", str(tmp / "r.md")]).read_text()
    assert "calls are still deferred" in md
    assert "·" in md  # unrun cells in the grid


def test_export_hides_cost_for_free_tier_and_includes_observations(
    sandbox, ready_experiments, monkeypatch, free_config
):
    tmp, _ = sandbox
    monkeypatch.setattr(export_md, "load_experiments", lambda: ready_experiments)
    runner.run(runner.build_jobs(ready_experiments, exp=["E1"], repeats=1))
    obs = tmp / "observations.md"
    obs.write_text("Jev wins E1.")
    md = export_md.main(["--out", str(tmp / "r.md"), "--observations", str(obs)]).read_text()
    assert "**Cost is not reported for this run.**" in md
    assert "$ / 100 q" not in md and "total cost" not in md and "### Verdict" not in md
    assert "### Speed and tokens" in md
    assert "Jev wins E1." in md and "_Not written yet._" not in md
