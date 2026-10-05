import pytest

from bench import run


def test_dry_lists_all_jobs_without_readiness_check(monkeypatch, capsys):
    monkeypatch.setattr(run, "check_ready", lambda exps: pytest.fail("--dry must not check readiness"))
    run.main(["--dry"])
    out = capsys.readouterr().out
    assert "\n270 jobs\n" in out and "total ~$" in out
    assert "E1S1Q1  jev   r0  typo" in out


def test_no_arguments_is_a_usage_error():
    with pytest.raises(SystemExit) as e:
        run.main([])
    assert e.value.code == 2


def test_filters_compose(capsys):
    run.main(["--dry", "--exp", "E1", "--engine", "jev", "--repeats", "1"])
    out = capsys.readouterr().out
    assert "\n10 jobs\n" in out and " low " not in out and "E2" not in out
    run.main(["--dry", "--exp", "E2,E3", "--sample", "S1", "--question", "Q2"])
    assert "\n18 jobs\n" in capsys.readouterr().out


def test_all_checks_readiness_before_running(monkeypatch):
    order = []
    monkeypatch.setattr(run, "check_ready", lambda exps: order.append("check") or (_ for _ in ()).throw(SystemExit(1)))
    monkeypatch.setattr(run, "run", lambda jobs: order.append("run"))
    with pytest.raises(SystemExit):
        run.main(["--all"])
    assert order == ["check"]


def test_all_runs_then_reports(monkeypatch):
    order = []
    monkeypatch.setattr(run, "check_ready", lambda exps: order.append("check"))
    monkeypatch.setattr(run, "run", lambda jobs: order.append(("run", len(jobs))))
    monkeypatch.setattr(run.report, "main", lambda: order.append("report"))
    run.main(["--all", "-v"])
    assert order == ["check", ("run", 270), "report"]


def test_dry_with_free_config_reports_days(capsys):
    try:
        run.main(["--dry", "--config", "config.free.yaml"])
    finally:
        run.use_config(None)
    out = capsys.readouterr().out
    assert "config: config.free.yaml" in out
    assert "free tier: 180 uncached call(s) on :free models (high, low); at 50/day that is 4 day(s)" in out


def test_dry_with_default_config_has_no_free_line(capsys):
    run.main(["--dry"])
    assert "free tier" not in capsys.readouterr().out
