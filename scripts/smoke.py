"""One live call per engine per experiment on S1Q1 (no cache). Prints the raw
response and the parsed prediction; exits non-zero if anything is unparseable."""

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.adapters import make
from bench.build_experiments import load_experiments
from bench.config import engines, headers


async def main() -> int:
    failures = 0
    exps = load_experiments()
    async with httpx.AsyncClient(timeout=60) as client:
        for name, spec in engines().items():
            adapter = make(name, spec)
            for exp, e in exps.items():
                s = e["samples"][0]
                q = s["questions"][0]
                print(f"\n=== {name} ({adapter.model}) / {exp}{s['id']}{q['id']} / gold={q['gold']} ===")
                try:
                    rec = await adapter.call(client, adapter.build_request(q, s["text"]), headers())
                except Exception as ex:
                    print(f"  FAILED: {ex}")
                    failures += 1
                    continue
                print(json.dumps(rec["response"], indent=2)[:1500])
                p = adapter.parse(q, rec["response"])
                print(
                    f"  -> label={p.label} conf={p.confidence:.3f} value={p.value} valid={p.valid} "
                    f"latency={rec['latency_ms']:.0f}ms tokens={p.tokens_in}/{p.tokens_out} "
                    f"cost=${p.cost:.6f} error={p.error}"
                )
                failures += not p.valid
    print(f"\n{'OK' if not failures else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
