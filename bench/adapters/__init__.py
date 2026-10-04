from .base import Adapter, Prediction
from .jev import JevAdapter
from .llm import LLMAdapter


def make(name: str, spec: dict) -> Adapter:
    cls = JevAdapter if spec["kind"] == "jev" else LLMAdapter
    return cls(name, spec)


__all__ = ["Adapter", "JevAdapter", "LLMAdapter", "Prediction", "make"]
