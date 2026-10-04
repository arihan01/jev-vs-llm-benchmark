import pytest

from bench import cache


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE", tmp_path)
    return tmp_path


def test_key_is_unique_per_coordinate():
    base = ("jev", "E1", "S1", "Q1", 0)
    keys = {cache.key(*base)}
    for i, alt in enumerate(["low", "E2", "S2", "Q2", 1]):
        coords = list(base)
        coords[i] = alt
        keys.add(cache.key(*coords))
    assert len(keys) == 6


def test_miss_returns_none():
    assert cache.get("jev", "E1", cache.key("jev", "E1", "S1", "Q1", 0)) is None


def test_round_trip_preserves_record(tmp_cache):
    k = cache.key("jev", "E1", "S1", "Q1", 2)
    record = {"response": {"a": [1, 2, {"b": None}]}, "latency_ms": 12.5, "status": 200}
    cache.put("jev", "E1", k, record)
    assert cache.get("jev", "E1", k) == {"key": k, **record}
    assert not list(tmp_cache.rglob("*.tmp")), "put must not leave temp files behind"


def test_put_overwrites_atomically(tmp_cache):
    k = cache.key("low", "E3", "S5", "Q2", 1)
    cache.put("low", "E3", k, {"v": 1})
    cache.put("low", "E3", k, {"v": 2})
    assert cache.get("low", "E3", k)["v"] == 2
    assert len(list(tmp_cache.rglob("*.json"))) == 1
