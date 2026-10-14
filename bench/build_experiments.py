"""Build data/experiments.json: per experiment, the first N frozen items, one
perturbation each, and two questions per sample (Q1 = dataset task, Q2 =
hand-written stub). Never overwrites an existing file without --force, because
the Q2s in it are written by hand. `--llm` only fills pending paraphrases."""

import argparse
import json
import logging
import random

from . import perturb, prompts
from .config import DATA, load, setup_logging
from .data import items

PATH = DATA / "experiments.json"
log = logging.getLogger(__name__)
# Paraphrases need a model call, so build() marks them and `--llm` fills them in a second pass.
PENDING = "paraphrase (pending)"


def q2_stub(task: str) -> dict:
    if task == "noul":
        return {
            "id": "Q2",
            "type": "noul",
            "instructions": "TODO: a second yes/no question about the passage",
            "options": dict(prompts.NOUL_CRITERIA),
            "gold": None,
        }
    if task == "score":
        return {
            "id": "Q2",
            "type": "score",
            "instructions": "TODO: e.g. How likely is the reviewer to return?",
            "options": {"0": "TODO", "1": "TODO", "2": "TODO"},
            "gold": None,
        }
    return {
        "id": "Q2",
        "type": "choice",
        "instructions": "TODO: a second question about the message",
        "options": {"TODO_a": "TODO", "TODO_b": "TODO"},
        "gold": None,
    }


def build_sample(task: str, idx: int, item: dict, kind: str, pool: list[dict], seed: int) -> dict:
    # Seeded per item and perturbation, so a rebuild reproduces the same typos, filler
    # and distractor.
    rng = random.Random(f"{seed}-{item['item_id']}-{kind}")
    text = item["text"]
    if kind == "typo":
        text = perturb.map_text(text, lambda s: perturb.typo(s, rng))
    elif kind == "padding":
        # BoolQ: pad the passage only; the question must stay intact.
        text = perturb.map_text(text, lambda s: perturb.pad(s, rng), fields=("passage",))
    elif kind == "distractor":
        text = perturb.distractor(task, item, rng, pool)
    elif kind == "clean":
        pass  # the unaltered dataset text, as a control
    elif kind == "paraphrase":
        kind = PENDING
    else:
        raise ValueError(kind)
    return {
        "id": f"S{idx + 1}",
        "source_id": item["item_id"],
        "perturbation": kind,
        "clean_text": item["text"],
        "text": text,
        "questions": [prompts.q1(task, item["gold"]), q2_stub(task)],
    }


def build() -> dict:
    cfg = load()
    out = {}
    for exp, spec in cfg["experiments"].items():
        task = spec["task"]
        pool = items(task)[: cfg["samples"]]
        kinds = cfg["sample_perturbations"]
        out[exp] = {
            "type": spec["type"],
            "task": task,
            "samples": [
                # S1..S5 take the perturbations in config order
                build_sample(task, i, it, kinds[i % len(kinds)], pool, cfg["seed"])
                for i, it in enumerate(pool)
            ],
        }
    return out


def fill_paraphrases(exps: dict) -> int:
    n = 0
    for spec in exps.values():
        for s in spec["samples"]:
            if s["perturbation"] == PENDING:
                s["text"] = perturb.paraphrase(spec["task"], s["source_id"], s["clean_text"])
                s["perturbation"] = "paraphrase"
                n += 1
    return n


def load_experiments() -> dict:
    if not PATH.exists():
        raise SystemExit(f"{PATH} missing; run `make experiments` first.")
    return json.loads(PATH.read_text())


def todo_report(exps: dict) -> list[str]:
    """Everything that still needs a human before the benchmark may run."""
    problems = []
    for exp, spec in exps.items():
        for s in spec["samples"]:
            if s["perturbation"] == PENDING:
                problems.append(f"{exp}{s['id']}: paraphrase pending (run `make experiments-llm`)")
            for q in s["questions"]:
                tag = f"{exp}{s['id']}{q['id']}"
                if q.get("gold") is None:
                    problems.append(f"{tag}: gold is null")
                elif q["gold"] not in q["options"]:
                    problems.append(f"{tag}: gold {q['gold']!r} is not an option key")
                if "TODO" in json.dumps(q):
                    problems.append(f"{tag}: still contains TODO text")
                if q["type"] == "noul" and set(q["options"]) != {"yes", "no"}:
                    problems.append(f"{tag}: noul options must be exactly yes/no")
                if q["type"] == "score" and sorted(q["options"], key=int) != [str(i) for i in range(len(q["options"]))]:
                    problems.append(f"{tag}: score options must be keyed 0..n-1")
    return problems


def save(exps: dict, path=None) -> None:
    # PATH is looked up at call time so tests that redirect it never touch data/.
    (path or PATH).write_text(json.dumps(exps, indent=2, ensure_ascii=False) + "\n")


def original_path():
    """Pristine copy next to PATH, written by build/--llm and never by the dashboard;
    "Reset sample" restores from it."""
    return PATH.with_name("experiments.original.json")


def original() -> dict:
    """The untouched build. Older checkouts have no snapshot yet, so the current
    file becomes the master the first time it is asked for."""
    if not original_path().exists():
        save(load_experiments(), original_path())
    return json.loads(original_path().read_text())


def reset_sample(exps: dict, exp: str, sample_id: str) -> dict:
    """Replace one sample (text, questions, golds) with its original and return it."""
    src = next(s for s in original()[exp]["samples"] if s["id"] == sample_id)
    samples = exps[exp]["samples"]
    samples[[s["id"] for s in samples].index(sample_id)] = src
    return src


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild from scratch (discards hand-written Q2s)")
    ap.add_argument("--llm", action="store_true", help="fill pending paraphrases via the generator model")
    ap.add_argument("--validate", action="store_true", help="report what still needs attention in the existing file")
    args = ap.parse_args(argv)
    setup_logging()
    if args.validate:
        exps = load_experiments()
        todo = todo_report(exps)
        log.info("%d item(s) still need attention before `make bench`", len(todo))
        for t in todo:
            log.info("  - %s", t)
        raise SystemExit(0)
    if args.llm:
        exps = load_experiments()
        log.info("filled %d paraphrase(s)", fill_paraphrases(exps))
    elif PATH.exists() and not args.force:
        raise SystemExit(f"{PATH.name} exists and may hold hand-written Q2s; use --force to rebuild it.")
    else:
        exps = build()
    save(exps)
    save(exps, original_path())
    todo = todo_report(exps)
    log.info("wrote %s; %d item(s) still need attention", PATH.name, len(todo))
    for t in todo:
        log.info("  - %s", t)


if __name__ == "__main__":
    main()
