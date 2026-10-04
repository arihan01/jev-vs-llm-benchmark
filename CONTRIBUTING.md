# Contributing

## Setup

```bash
make setup               # .venv with the package in editable mode + dev tools
make precommit-install   # optional: ruff, mypy and file hygiene on every commit
```

No API key is needed for development. Everything except `make smoke`, `make experiments-llm`,
`make bench` and `make demo` runs offline.

## Before you push

```bash
make check               # ruff check, ruff format --check, mypy, pytest (coverage >= 85%)
```

CI runs the same commands on Python 3.11, 3.12 and 3.14.

## Tests

- Tests live in `tests/`, one file per module (`test_<module>.py`), plus `test_pipeline_offline.py` for
  the end-to-end grid and `test_cli.py` for `bench.run`.
- Shared fixtures are in `tests/conftest.py`: `sandbox` (fake gateway, cache and results in `tmp_path`)
  and `ready_experiments` (a filled-in copy of the experiments file).
- Tests never hit the network. Use the `sandbox` fixture or `respx` for anything HTTP.

## Rules that keep the benchmark fair

- Write Q2 instructions, options and gold answers **before** looking at any engine output.
- Jev and the LLMs must receive identical wording. Change question text only in
  `data/experiments.json` or `bench/prompts.py`, never inside an adapter.
- Don't edit frozen data (`data/*.jsonl`, `data/banking77_labels.json`) after results exist. If you
  must, re-run the whole grid.
- `.cache/` is the record of what each engine actually returned. Don't hand-edit it; delete an entry
  to force a fresh call.

## Commits

Short imperative subject line (≤ 72 chars), a blank line, then the why. One logical change per commit,
and each commit should pass `make check`.
