"""Optional: draft Banking77 intent descriptions with the generator model.
Writes data/banking77_labels.draft.json for side-by-side review; the frozen
data/banking77_labels.json is only ever edited by hand."""

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.config import DATA, headers, load
from bench.prompts import banking77_labels


def main():
    cfg = load()
    keys = list(banking77_labels())
    prompt = (
        "For each banking customer-support intent id below, write a one-line description of what the "
        "customer is asking, starting with 'Customer'. Example: card_arrival -> \"Customer asks when their "
        'new card will arrive". Reply with a JSON object mapping each id to its description.\n\n' + "\n".join(keys)
    )
    r = httpx.post(
        cfg["openrouter_base"].rstrip("/") + "/chat/completions",
        headers=headers(),
        timeout=120,
        json={
            "model": cfg["perturb_generator"]["model"],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    r.raise_for_status()
    draft = json.loads(r.json()["choices"][0]["message"]["content"])
    missing = set(keys) - set(draft)
    out = DATA / "banking77_labels.draft.json"
    out.write_text(json.dumps({k: draft.get(k, "") for k in keys}, indent=2) + "\n")
    note = f"; missing: {', '.join(sorted(missing))}" if missing else ""
    print(f"wrote {out} ({len(keys) - len(missing)}/{len(keys)} ids){note}")


if __name__ == "__main__":
    main()
