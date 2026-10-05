from .. import prompts
from .base import Adapter, Prediction


def find_answer(resp, qname: str):
    """The alpha API's envelope is not pinned down, so locate the answer object
    for `qname` wherever it is nested rather than hard-coding one path. At the
    two known positions (top level, `answers`) `type` is optional, since
    TypeSafe's direct API may omit it; deeper matches must carry `type`."""
    if isinstance(resp, dict):
        for container in (resp, resp.get("answers")):
            if isinstance(container, dict) and isinstance(container.get(qname), dict):
                return container[qname]
    return _search(resp, qname)


def _search(resp, qname: str):
    if isinstance(resp, dict):
        hit = resp.get(qname)
        if isinstance(hit, dict) and "type" in hit:
            return hit
        for v in resp.values():
            if (found := _search(v, qname)) is not None:
                return found
    elif isinstance(resp, list):
        for v in resp:
            if isinstance(v, dict) and v.get("name") == qname and "type" in v:
                return v
            if (found := _search(v, qname)) is not None:
                return found
    return None


class JevAdapter(Adapter):
    def build_request(self, q, text, temperature=None):
        return {"model": self.model, "state": text, "questions": {q["id"]: prompts.jev_question(q)}}

    def parse(self, q, response) -> Prediction:
        usage = response.get("usage") or {}
        p = Prediction(
            label=None,
            tokens_in=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        )
        p.cost = self.cost(usage, p.tokens_in, p.tokens_out)
        ans = find_answer(response, q["id"])
        if ans is None:
            p.error = "no answer in response"
            return p
        keys = prompts.option_keys(q)
        kind = q["type"]
        try:
            if kind == "choice":
                p.probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
                p.label = ans.get("choice")
            elif kind == "score":
                p.value = float(ans["score"])
                p.probs = {str(k): float(v) for k, v in (ans.get("probabilities") or {}).items()}
                p.label = str(min(max(round(p.value), 0), len(keys) - 1))
            else:
                py = float(ans["noul"])
                p.value = py
                p.probs = {"yes": py, "no": 1 - py}
                p.label = "yes" if py >= 0.5 else "no"
        except (KeyError, TypeError, ValueError) as e:
            p.error = f"malformed answer: {e}"
            return p
        p.valid = p.label in keys
        p.confidence = p.probs.get(p.label, float(ans.get("confidence") or 0.0)) if p.valid else 0.0
        return p
