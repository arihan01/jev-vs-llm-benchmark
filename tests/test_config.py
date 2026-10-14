import pytest
import yaml

from bench import config


@pytest.mark.parametrize("value", ["", "sk-or-v1-..."])
def test_missing_or_placeholder_key_exits_with_hint(monkeypatch, value):
    monkeypatch.setenv("OPENROUTER_API_KEY", value)
    with pytest.raises(SystemExit, match=r"Copy \.env\.example to \.env"):
        config.api_key()


def test_headers(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    assert config.headers() == {"Authorization": "Bearer sk-or-v1-test", "Content-Type": "application/json"}


def test_engines_follow_config_order():
    raw = yaml.safe_load((config.ROOT / "config.yaml").read_text())
    assert list(raw["engines"]) == ["jev", "low", "high", "stat"]
    assert list(config.engines()) == ["jev", "low", "high"]  # stat ships disabled


def test_engine_order_puts_configured_engines_first():
    assert config.engine_order(["zeta", "high", "jev", "alpha"]) == ["jev", "high", "alpha", "zeta"]


def test_config_shape():
    cfg = config.load()
    assert cfg["repeats"] == 3 and cfg["samples"] == 5
    assert len(cfg["sample_perturbations"]) == cfg["samples"]
    assert {e["type"] for e in cfg["experiments"].values()} == {"choice", "noul", "score"}
    assert cfg["engines"]["low"]["reasoning"] in (False, "off")


def test_setup_logging_quiets_http_libraries():
    import logging

    config.setup_logging(verbose=True)
    assert logging.getLogger("httpx").level == logging.WARNING


TYPESAFE = {
    "kind": "jev",
    "model": "jev-1.13.0",
    "base_url": "https://api.typesafe.ai/v1/systemone",
    "api_key_env": "TYPESAFE_API_KEY",
}


def test_engine_url_defaults_and_overrides():
    assert config.engine_url({"kind": "jev"}) == "https://openrouter.ai/api/alpha/decisions"
    assert config.engine_url({"kind": "llm"}) == "https://openrouter.ai/api/v1/chat/completions"
    assert config.engine_url(TYPESAFE) == "https://api.typesafe.ai/v1/systemone"
    assert config.engine_url({"kind": "llm", "base_url": "https://api.anthropic.com/v1/"}) == (
        "https://api.anthropic.com/v1/chat/completions"
    )


def test_is_openrouter_and_is_free():
    assert config.is_openrouter({"kind": "llm"}) and not config.is_openrouter(TYPESAFE)
    assert config.is_free({"kind": "llm", "model": "qwen/qwen3.8-27b:free"})
    assert not config.is_free({"kind": "llm", "model": "qwen/qwen3.5-9b"})
    assert not config.is_free({"kind": "llm", "model": "x:free", "base_url": "https://other.example/v1"})


def test_headers_use_the_engine_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-real")
    assert config.headers(TYPESAFE)["Authorization"] == "Bearer ts-real"


@pytest.mark.parametrize("value", [None, "", "apikey_..."])
def test_missing_typesafe_key_names_the_variable(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("TYPESAFE_API_KEY", value)
    with pytest.raises(SystemExit, match="TYPESAFE_API_KEY is not set"):
        config.headers(TYPESAFE)


def test_config_selection_precedence(monkeypatch, tmp_path):
    alt = tmp_path / "alt.yaml"
    alt.write_text("engines: {}\nresults_dir: results/alt\n")
    monkeypatch.delenv("BENCH_CONFIG", raising=False)
    assert config.config_path() == config.ROOT / "config.yaml"
    monkeypatch.setenv("BENCH_CONFIG", str(alt))
    assert config.config_path() == alt and config.results_dir() == config.ROOT / "results/alt"
    config.use_config(config.ROOT / "config.yaml")
    try:
        assert config.config_path() == config.ROOT / "config.yaml", "--config beats BENCH_CONFIG"
        assert config.results_dir() == config.ROOT / "results"
    finally:
        config.use_config(None)


def test_missing_config_file_exits(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCH_CONFIG", str(tmp_path / "nope.yaml"))
    with pytest.raises(SystemExit, match="config file not found"):
        config.load()


def test_free_config_matches_default_except_engines(free_config):
    default = yaml.safe_load((config.ROOT / "config.yaml").read_text())
    for key in ("seed", "repeats", "samples", "sample_perturbations", "experiments", "tasks"):
        assert free_config[key] == default[key], key
    assert list(free_config["engines"]) == list(default["engines"])
    assert free_config["results_dir"] != default.get("results_dir", "results")
    engines = free_config["engines"]
    assert engines["jev"]["api_key_env"] == "TYPESAFE_API_KEY" and not config.is_openrouter(engines["jev"])
    assert all(config.is_free(engines[e]) and engines[e]["rpm"] <= 20 for e in ("low", "high"))
    assert engines["high"]["structured_output"] is False
