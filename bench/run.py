"""CLI entry point: builds the job list from the filters, runs it and writes
the report. `--dry` prints the jobs and a cost estimate without calling anything."""

import argparse
import json
import math

from . import report
from .build_experiments import load_experiments
from .config import config_path, engines, is_free, load, setup_logging, use_config
from .runner import _needs_call, build_jobs, check_ready, make, run

FREE_DAILY_LIMIT = 50


def _split(s):
    return s.split(",") if s else None


def estimate(jobs) -> None:
    """Rough cost from prompt length (~4 chars per token) and the config prices."""
    specs = engines(include_disabled=True)
    total = 0.0
    per: dict[str, float] = {}
    for j in jobs:
        spec = specs[j.engine]
        body = make(j.engine, spec).build_request(j.question, j.sample["text"])
        t_in = len(json.dumps(body)) / 4
        # Jev returns no completion text; an LLM's JSON answer is about 20 tokens.
        t_out = 0 if spec["kind"] != "llm" else 20
        c = (t_in * spec.get("price_in", 0) + t_out * spec.get("price_out", 0)) / 1e6
        per[j.engine] = per.get(j.engine, 0.0) + c
        total += c
    for name, c in per.items():
        print(f"  {name:<5} ~${c:.4f}")
    print(f"  total ~${total:.4f} (estimate; real cost comes from usage.cost or tokens x price)")
    free_days(jobs)


def free_days(jobs) -> None:
    """OpenRouter's :free models share one per-account daily request cap."""
    specs = engines(include_disabled=True)
    free = [j for j in jobs if is_free(specs[j.engine])]
    if not free:
        return
    adapters = {name: make(name, specs[name]) for name in {j.engine for j in free}}
    pending = sum(_needs_call(adapters[j.engine], j) for j in free)
    cap = load().get("free_daily_limit", FREE_DAILY_LIMIT)
    names = ", ".join(sorted(adapters))
    print(
        f"  free tier: {pending} uncached call(s) on :free models ({names}); at {cap}/day "
        f"that is {math.ceil(pending / cap)} day(s) of re-running `make bench`"
    )


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
    ap.add_argument("--config", help="config file (default: $BENCH_CONFIG or config.yaml)")
    args = ap.parse_args(argv)
    setup_logging(args.verbose)
    use_config(args.config)
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
        print(f"config: {config_path().name}")
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
