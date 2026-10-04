import json
import re

from .. import prompts
from ..config import load
from .base import Adapter, Prediction


def response_format(q: dict) -> dict:
    keys = prompts.option_keys(q)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": keys},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["label", "confidence"],
                "additionalProperties": False,
            },
        },
    }


def parse_content(content: str) -> dict | None:
    if not content:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


class LLMAdapter(Adapter):
    def __init__(self, name, spec):
        super().__init__(name, spec)
        self.url = load()["openrouter_base"].rstrip("/") + "/chat/completions"

    def build_request(self, q, text, temperature=0.0):
        body = {
            "model": self.model,
            "messages": prompts.llm_messages(q, text),
            "response_format": response_format(q),
            "max_tokens": 200,
            "usage": {"include": True},
        }
        if temperature is not None:
            body["temperature"] = temperature
        if self.spec.get("reasoning") is False or self.spec.get("reasoning") == "off":
            body["reasoning"] = {"enabled": False}
        return body

    def parse(self, q, response) -> Prediction:
        usage = response.get("usage") or {}
        p = Prediction(
            label=None,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            cost=float(usage.get("cost") or 0.0),
            raw=response,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            p.error = "no message content"
            return p
        obj = parse_content(content)
        keys = prompts.option_keys(q)
        if obj is None or "label" not in obj:
            p.error = "unparseable JSON"
            return p
        label = str(obj["label"]).strip()
        if label not in keys:
            p.label = label
            p.error = "label not in option set"
            return p
        try:
            conf = min(max(float(obj.get("confidence", 0.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.0
        rest = (1 - conf) / (len(keys) - 1)
        p.label, p.valid, p.confidence = label, True, conf
        p.probs = {k: (conf if k == label else rest) for k in keys}
        if q["type"] == "score":
            p.value = float(label)
        elif q["type"] == "noul":
            p.value = p.probs["yes"]
        return p
