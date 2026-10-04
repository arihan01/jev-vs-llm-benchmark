"""5-minute demo: one customer message, all three Jev question types in a single
request, side by side with the same three questions asked of an LLM."""

import asyncio
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.adapters.jev import find_answer
from bench.adapters.llm import LLMAdapter, parse_content
from bench.config import RESULTS, headers, load

MESSAGE = (
    "I was charged twice for the same coffee this morning and the app still shows both payments. "
    "This is the second time this month and I'm getting really fed up."
)

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "Which intent does this banking customer message express?",
        "criteria": {
            "transaction_charged_twice": "Customer was charged twice for the same transaction",
            "card_payment_not_recognised": "Customer sees a card payment they do not recognise",
            "request_refund": "Customer wants to request a refund for a purchase",
            "pending_card_payment": "Customer's card payment is still showing as pending",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Mildly annoyed", "Frustrated", "Very angry"],
    },
    "escalate": {
        "type": "noul",
        "instructions": "Should this be escalated to a human agent?",
        "criteria": {
            "true": "The customer is upset about a repeated problem.",
            "false": "A routine automated reply is enough.",
        },
    },
}


def llm_prompt(q: dict) -> tuple[str, list[str]]:
    crit = q["criteria"]
    if q["type"] == "score":
        opts = [(str(i), d) for i, d in enumerate(crit)]
    elif q["type"] == "noul":
        opts = [("yes", crit["true"]), ("no", crit["false"])]
    else:
        opts = list(crit.items())
    lines = "\n".join(f"{k}: {d}" for k, d in opts)
    return (
        f"{q['instructions']}\nOptions:\n{lines}\nInput:\n{MESSAGE}\n"
        'Return {"label": <one option key>, "confidence": <0-1>}.'
    ), [k for k, _ in opts]


async def main():
    cfg = load()
    llm = LLMAdapter("low", cfg["engines"]["low"])
    async with httpx.AsyncClient(timeout=60) as client:
        print(f"Input: {MESSAGE}\n")
        t0 = time.perf_counter()
        r = await client.post(
            cfg["jev_decisions_url"],
            headers=headers(),
            json={"model": cfg["engines"]["jev"]["model"], "state": MESSAGE, "questions": QUESTIONS},
        )
        r.raise_for_status()
        jev_ms = (time.perf_counter() - t0) * 1000
        resp = r.json()
        print(
            f"Jev ({cfg['engines']['jev']['model']}): one request, {jev_ms:.0f} ms, "
            f"${(resp.get('usage') or {}).get('cost', 0):.6f}"
        )
        for name in QUESTIONS:
            a = find_answer(resp, name) or {}
            shown = {k: a.get(k) for k in ("choice", "score", "noul", "confidence", "probabilities") if k in a}
            print(f"  {name:<12} {shown}")

        print(f"\nLLM ({llm.model}): three requests")
        for name, q in QUESTIONS.items():
            user, keys = llm_prompt(q)
            body = {
                "model": llm.model,
                "temperature": 0,
                "max_tokens": 200,
                "usage": {"include": True},
                "messages": [
                    {"role": "system", "content": "You are a classifier. Reply only with JSON matching the schema."},
                    {"role": "user", "content": user},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "c",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["label", "confidence"],
                            "properties": {"label": {"type": "string", "enum": keys}, "confidence": {"type": "number"}},
                        },
                    },
                },
            }
            rec = await llm.call(client, body, headers())
            out = parse_content(rec["response"]["choices"][0]["message"]["content"])
            print(f"  {name:<12} {out}  ({rec['latency_ms']:.0f} ms)")

    sc = RESULTS / "scorecard.csv"
    if sc.exists():
        import pandas as pd

        cols = ["exp", "engine", "accuracy", "consistency", "p50_latency_ms", "cost_per_100_questions"]
        df = pd.read_csv(sc)
        print("\nScorecard:\n" + df[[c for c in cols if c in df]].to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
