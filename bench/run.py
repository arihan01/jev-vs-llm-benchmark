import argparse
import json

from . import report
from .adapters import make
from .build_experiments import load_experiments
from .config import engines, setup_logging
from .runner import build_jobs, check_ready, run


def _split(s):
    return s.split(",") if s else None


def estimate(jobs) -> None:
    """Rough cost from prompt length (~4 chars per token) and the config prices."""
    specs = engines()
    total = 0.0
    per: dict[str, float] = {}
    for j in jobs:
        spec = specs[j.engine]
        body = make(j.engine, spec).build_request(j.question, j.sample["text"])
        t_in = len(json.dumps(body)) / 4
        t_out = 0 if spec["kind"] == "jev" else 20
        c = (t_in * spec.get("price_in", 0) + t_out * spec.get("price_out", 0)) / 1e6
        per[j.engine] = per.get(j.engine, 0.0) + c
        total += c
    for name, c in per.items():
        print(f"  {name:<5} ~${c:.4f}")
    print(f"  total ~${total:.4f} (estimate; real cost comes from usage.cost)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Jev vs LLM mini-benchmark")
    ap.add_argument("--all", action="store_true", help="every experiment, sample, question and engine")
    ap.add_argument("--exp", help="comma list, e.g. E1,E3")
    ap.add_argument("--engine", help="comma list of engine names from config.yaml")
    ap.add_argument("--sample", help="comma list, e.g. S1,S2")
    ap.add_argument("--question", help="comma list, e.g. Q1")
    ap.add_argument("--repeats", type=int, help="override repeats from config.yaml")
    ap.add_argument("--dry", action="store_true", help="print the job list and a cost estimate; no calls")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging (cache hits, call keys)")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    if not (args.all or args.exp or args.engine or args.sample or args.question or args.dry):
        ap.error("pass --all, --dry, or a filter such as --exp E1")

    exps = load_experiments()
    jobs = build_jobs(
        exps,
        exp=_split(args.exp),
        engine=_split(args.engine),
        sample=_split(args.sample),
        question=_split(args.question),
        repeats=args.repeats,
    )
    if args.dry:
        for j in jobs:
            print(f"{j.exp}{j.sample['id']}{j.question['id']}  {j.engine:<5} r{j.repeat}  {j.sample['perturbation']}")
        print(f"\n{len(jobs)} jobs")
        estimate(jobs)
        return
    check_ready(exps)
    run(jobs)
    report.main()


if __name__ == "__main__":
    main()
