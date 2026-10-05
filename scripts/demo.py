"""5-minute demo: one customer message, all three Jev question types in a single
request, side by side with the same three questions asked of an LLM."""

import asyncio
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.adapters.jev import find_answer
from bench.adapters.llm import LLMAdapter
from bench.config import engine_url, headers, load, results_dir

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


def as_question(q: dict) -> dict:
    """Jev's `criteria` in the benchmark's `options` form, so the LLM side goes
    through the same adapter code as `make bench`."""
    crit = q["criteria"]
    if q["type"] == "score":
        options = {str(i): d for i, d in enumerate(crit)}
    elif q["type"] == "noul":
        options = {"yes": crit["true"], "no": crit["false"]}
    else:
        options = dict(crit)
    return {"type": q["type"], "instructions": q["instructions"], "options": options}


async def main():
    cfg = load()
    llm = LLMAdapter("low", cfg["engines"]["low"])
    async with httpx.AsyncClient(timeout=60) as client:
        print(f"Input: {MESSAGE}\n")
        t0 = time.perf_counter()
        jev = cfg["engines"]["jev"]
        r = await client.post(
            engine_url(jev),
            headers=headers(jev),
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
            question = as_question(q)
            rec = await llm.call(client, llm.build_request(question, MESSAGE), headers(llm.spec))
            p = llm.parse(question, rec["response"])
            out = {"label": p.label, "confidence": p.confidence} if p.valid else {"error": p.error}
            print(f"  {name:<12} {out}  ({rec['latency_ms']:.0f} ms)")

    sc = results_dir() / "scorecard.csv"
    if sc.exists():
        import pandas as pd

        cols = ["exp", "engine", "accuracy", "consistency", "p50_latency_ms", "cost_per_100_questions"]
        df = pd.read_csv(sc)
        print("\nScorecard:\n" + df[[c for c in cols if c in df]].to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
