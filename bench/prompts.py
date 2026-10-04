"""Question wording and rendering. A question is a dict with `id`, `type`
(choice / score / noul), `instructions` and `options` ({key: description}).
Jev receives instructions + option text as fields; the LLM receives the same
strings in the brief's fixed template, so neither engine gets words the other
doesn't."""

import json

from .config import DATA

SYSTEM = "You are a classifier. Reply only with JSON matching the schema."

INSTRUCTIONS = {
    "choice": "Which intent does this banking customer message express?",
    "score": "How positive is this review?",
    "noul": "Given the passage, is the answer to the question yes?",
}

SCORE_LEVELS = [
    "Very negative (1 star)",
    "Negative (2 stars)",
    "Mixed or neutral (3 stars)",
    "Positive (4 stars)",
    "Very positive (5 stars)",
]

NOUL_CRITERIA = {
    "yes": "The passage supports answering the question with yes.",
    "no": "The passage supports answering the question with no.",
}


def banking77_labels() -> dict[str, str]:
    with open(DATA / "banking77_labels.json") as f:
        return json.load(f)


def default_options(task: str) -> dict[str, str]:
    if task == "choice":
        return banking77_labels()
    if task == "score":
        return {str(i): d for i, d in enumerate(SCORE_LEVELS)}
    if task == "noul":
        return dict(NOUL_CRITERIA)
    raise ValueError(task)


def q1(task: str, gold: str) -> dict:
    """The dataset's own task as a question, with the dataset's gold label."""
    return {
        "id": "Q1",
        "type": task,
        "instructions": INSTRUCTIONS[task],
        "options": default_options(task),
        "gold": gold,
    }


def option_keys(q: dict) -> list[str]:
    return list(q["options"])


def render_state(text) -> str:
    return text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, indent=2)


def llm_messages(q: dict, text) -> list[dict]:
    lines = "\n".join(f"{k}: {d}" for k, d in q["options"].items())
    user = (
        f"{q['instructions']}\n"
        f"Options:\n{lines}\n"
        f"Input:\n{render_state(text)}\n"
        'Return {"label": <one option key>, "confidence": <0-1>}.'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def jev_question(q: dict) -> dict:
    opts = q["options"]
    criteria: dict[str, str] | list[str]
    if q["type"] == "choice":
        criteria = dict(opts)
    elif q["type"] == "score":
        criteria = [opts[k] for k in sorted(opts, key=int)]
    else:
        # Jev's Noul criteria are keyed true/false; the text is identical to the LLM's yes/no lines.
        criteria = {"true": opts["yes"], "false": opts["no"]}
    return {"type": q["type"], "instructions": q["instructions"], "criteria": criteria}
