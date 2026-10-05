import collections

import pandas as pd
import pytest

from bench import data


def fake_raw(task):
    golds = ["a"] * 60 + ["b"] * 30 + ["c"] * 10
    return pd.DataFrame({"text": [f"t{i}" for i in range(100)], "gold": golds})


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "_raw", fake_raw)
    monkeypatch.setattr(data, "DATA", tmp_path)
    return tmp_path


def test_sample_is_stratified_and_deterministic(fake):
    a = data.sample("choice", 20, 42)
    assert a == data.sample("choice", 20, 42)
    assert a != data.sample("choice", 20, 7)
    assert collections.Counter(r["gold"] for r in a) == {"a": 12, "b": 6, "c": 2}
    assert all(r["item_id"].startswith("choice-") for r in a)


@pytest.mark.parametrize(
    "raw, norm",
    [("Refund_not_showing_up", "refund_not_showing_up"), ("reverted_card_payment?", "reverted_card_payment")],
)
def test_norm_intent(raw, norm):
    assert data.norm_intent(raw) == norm


def test_jsonl_round_trip(tmp_path):
    rows = [{"item_id": "x", "text": {"passage": "é", "question": "q"}, "gold": "yes"}, {"item_id": "y", "text": "t"}]
    p = tmp_path / "f.jsonl"
    data.write_jsonl(p, rows)
    assert data.read_jsonl(p) == rows


def test_main_writes_each_task_once(fake, monkeypatch):
    monkeypatch.setattr(data, "load", lambda: {"seed": 42})
    monkeypatch.setattr(data, "N_ITEMS", 10)
    data.main()
    written = {p.name: p.read_text() for p in fake.glob("*.jsonl")}
    assert set(written) == {"choice.jsonl", "score.jsonl", "noul.jsonl"}
    (fake / "score.jsonl").write_text('{"item_id": "frozen", "text": "t", "gold": "a"}\n')
    data.main()
    assert data.items("score")[0]["item_id"] == "frozen", "existing frozen files must not be resampled"
    data.main(force=True)
    assert data.items("score")[0]["item_id"] != "frozen"


def test_items_missing_file_exits(fake):
    with pytest.raises(SystemExit, match="make data"):
        data.items("choice")


def test_frozen_files_are_complete():
    for task in ("choice", "score", "noul"):
        rows = data.read_jsonl(data.path(task))
        assert len(rows) == 300 and len({r["item_id"] for r in rows}) == 300
