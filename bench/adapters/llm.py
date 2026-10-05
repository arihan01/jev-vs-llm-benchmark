import json
import math
import re

from .. import prompts
from ..config import is_openrouter
from .base import Adapter, HTTPStatusError, Prediction


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


def confidence(value, n_options: int) -> float:
    """Self-reported confidence clamped to [0, 1]. A missing, non-numeric or
    non-finite value becomes the neutral 1/n rather than 0, so the chosen label
    never ends up less likely than the options the model did not pick."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        return 1 / n_options
    return min(max(c, 0.0), 1.0) if math.isfinite(c) else 1 / n_options


SCHEMA_REJECTED = re.compile(r"response_format|json_schema|structured", re.I)


class LLMAdapter(Adapter):
    async def call(self, client, body, headers, limiter=None) -> dict:
        """Some free models reject `response_format` with a 400. Resend once
        without it; the prompt's own "Return {...}" line still asks for JSON, and
        the returned `request` records which body actually produced the answer."""
        try:
            return await super().call(client, body, headers, limiter)
        except HTTPStatusError as e:
            if e.status != 400 or "response_format" not in body or not SCHEMA_REJECTED.search(e.body):
                raise
            plain = {k: v for k, v in body.items() if k != "response_format"}
            rec = await super().call(client, plain, headers, limiter)
            return {**rec, "request": plain}

    def build_request(self, q, text, temperature=0.0):
        body: dict = {
            "model": self.model,
            "messages": prompts.llm_messages(q, text),
            "max_tokens": self.spec.get("max_tokens", 200),
        }
        if self.spec.get("structured_output", True):
            body["response_format"] = response_format(q)
        if temperature is not None:
            body["temperature"] = temperature
        if is_openrouter(self.spec):
            body["usage"] = {"include": True}
            if self.spec.get("reasoning") is False or self.spec.get("reasoning") == "off":
                body["reasoning"] = {"enabled": False}
        return body

    def parse(self, q, response) -> Prediction:
        usage = response.get("usage") or {}
        p = Prediction(
            label=None,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
        )
        p.cost = self.cost(usage, p.tokens_in, p.tokens_out)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            p.error = "no message content"
            return p
        if not (content or "").strip():
            finish = response["choices"][0].get("finish_reason")
            p.error = f"empty content (finish_reason={finish}; a reasoning model may have spent max_tokens thinking)"
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
        conf = confidence(obj.get("confidence"), len(keys))
        rest = (1 - conf) / (len(keys) - 1)
        p.label, p.valid, p.confidence = label, True, conf
        p.probs = {k: (conf if k == label else rest) for k in keys}
        if q["type"] == "score":
            p.value = float(label)
        elif q["type"] == "noul":
            p.value = p.probs["yes"]
        return p
