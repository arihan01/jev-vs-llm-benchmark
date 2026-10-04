import pytest
import yaml

from bench import config


@pytest.mark.parametrize("value", ["", "sk-or-v1-..."])
def test_missing_or_placeholder_key_exits_with_hint(monkeypatch, value):
    monkeypatch.setenv("OPENROUTER_API_KEY", value)
    with pytest.raises(SystemExit, match=r"Copy \.env\.example to \.env"):
        config.api_key()


def test_headers_with_optional_attribution(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.delenv("OPENROUTER_APP_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_APP_NAME", raising=False)
    assert config.headers() == {"Authorization": "Bearer sk-or-v1-test", "Content-Type": "application/json"}
    monkeypatch.setenv("OPENROUTER_APP_URL", "https://example.test")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "bench")
    h = config.headers()
    assert h["HTTP-Referer"] == "https://example.test" and h["X-Title"] == "bench"


def test_engines_follow_config_order():
    raw = yaml.safe_load((config.ROOT / "config.yaml").read_text())
    assert list(config.engines()) == list(raw["engines"]) == ["jev", "low", "high"]


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
