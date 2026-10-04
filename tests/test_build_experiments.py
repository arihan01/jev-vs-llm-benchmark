import json

import pytest

from bench import build_experiments as bx
from bench.config import load
from tests.conftest import make_ready


@pytest.fixture
def tmp_path_file(tmp_path, monkeypatch):
    path = tmp_path / "experiments.json"
    monkeypatch.setattr(bx, "PATH", path)
    return path


def test_build_assigns_perturbations_in_config_order():
    exps = bx.build()
    kinds = load()["sample_perturbations"]
    assert list(exps) == ["E1", "E2", "E3"]
    for spec in exps.values():
        assert [s["id"] for s in spec["samples"]] == ["S1", "S2", "S3", "S4", "S5"]
        got = [s["perturbation"] for s in spec["samples"]]
        assert got == [bx.PENDING if k == "paraphrase" else k for k in kinds]
        assert all([q["id"] for q in s["questions"]] == ["Q1", "Q2"] for s in spec["samples"])


def test_injection_only_on_injection_sample():
    for spec in bx.build().values():
        for s in spec["samples"]:
            assert (s["injection_target"] is not None) == (s["perturbation"] == "injection")
            if s["injection_target"]:
                assert s["injection_target"] != s["questions"][0]["gold"]


def test_ready_file_has_no_todos_and_passes_schema(ready_experiments):
    assert bx.todo_report(ready_experiments) == []
    assert bx.schema_errors(ready_experiments) == []


def test_committed_file_matches_schema():
    assert bx.schema_errors(json.loads(bx.PATH.read_text())) == []


def _first_q2(exps):
    return exps["E1"]["samples"][0]["questions"][1]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda e: _first_q2(e).update(gold=None), "gold is null"),
        (lambda e: _first_q2(e).update(gold="zzz"), "is not an option key"),
        (lambda e: _first_q2(e).update(instructions="TODO later"), "still contains TODO"),
        (lambda e: e["E1"]["samples"][1].update(perturbation=bx.PENDING), "paraphrase pending"),
        (
            lambda e: e["E2"]["samples"][0]["questions"][1].update(options={"y": "a", "n": "b"}, gold="y"),
            "noul options must be exactly yes/no",
        ),
        (
            lambda e: e["E3"]["samples"][0]["questions"][1].update(options={"1": "a", "2": "b"}, gold="1"),
            "score options must be keyed 0..n-1",
        ),
    ],
)
def test_todo_report_flags_each_rule(ready_experiments, mutate, message):
    mutate(ready_experiments)
    problems = bx.todo_report(ready_experiments)
    assert len(problems) == 1 and message in problems[0]


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda e: _first_q2(e).update(type="maybe"), "is not one of"),
        (lambda e: e["E2"]["samples"][0]["questions"][1].update(options={"yes": "a"}), "'no' is a required property"),
        (lambda e: e["E3"]["samples"][0]["questions"][1].update(options={"x": "a", "y": "b"}), "does not match"),
        (lambda e: e.update(X1={}), "does not match any of the regexes"),
        (lambda e: e["E1"]["samples"][0].pop("clean_text"), "'clean_text' is a required property"),
    ],
)
def test_schema_rejects_malformed_files(ready_experiments, mutate, fragment):
    mutate(ready_experiments)
    errors = bx.schema_errors(ready_experiments)
    assert errors and any(fragment in e for e in errors), errors


def test_main_writes_then_refuses_to_overwrite(tmp_path_file):
    bx.main([])
    first = tmp_path_file.read_text()
    assert json.loads(first)["E1"]["samples"][0]["questions"][1]["gold"] is None
    with pytest.raises(SystemExit, match="use --force"):
        bx.main([])
    edited = json.loads(first)
    edited["E1"]["samples"][0]["questions"][1]["gold"] = "kept"
    tmp_path_file.write_text(json.dumps(edited))
    with pytest.raises(SystemExit):
        bx.main([])
    assert json.loads(tmp_path_file.read_text())["E1"]["samples"][0]["questions"][1]["gold"] == "kept"
    bx.main(["--force"])
    assert json.loads(tmp_path_file.read_text())["E1"]["samples"][0]["questions"][1]["gold"] is None


def test_llm_flag_fills_only_pending_paraphrases(tmp_path_file, monkeypatch):
    bx.main([])
    exps = json.loads(tmp_path_file.read_text())
    exps["E1"]["samples"][0]["questions"][1]["gold"] = "hand-written"
    tmp_path_file.write_text(json.dumps(exps))
    monkeypatch.setattr(bx.perturb, "paraphrase", lambda task, sid, text: f"PARA {task}")
    bx.main(["--llm"])
    out = json.loads(tmp_path_file.read_text())
    s2 = [spec["samples"][1] for spec in out.values()]
    assert all(
        s["perturbation"] == "paraphrase" and s["text"] == f"PARA {t}"
        for s, t in zip(s2, ["choice", "noul", "score"], strict=True)
    )
    assert out["E1"]["samples"][0]["questions"][1]["gold"] == "hand-written"


@pytest.mark.parametrize("ready", [False, True])
def test_validate_passes_schema_valid_files_even_with_todos(tmp_path_file, ready):
    exps = bx.build()
    tmp_path_file.write_text(json.dumps(make_ready(exps) if ready else exps))
    with pytest.raises(SystemExit) as e:
        bx.main(["--validate"])
    assert e.value.code == 0


def test_validate_fails_on_schema_error(tmp_path_file):
    exps = bx.build()
    exps["E1"]["type"] = "nope"
    tmp_path_file.write_text(json.dumps(exps))
    with pytest.raises(SystemExit) as e:
        bx.main(["--validate"])
    assert e.value.code == 1


def test_load_missing_file_exits(tmp_path_file):
    with pytest.raises(SystemExit, match="make experiments"):
        bx.load_experiments()


def test_unknown_perturbation_rejected():
    with pytest.raises(ValueError):
        bx.build_sample("score", 0, {"item_id": "x", "text": "t", "gold": "1"}, "rot13", [], 42)
