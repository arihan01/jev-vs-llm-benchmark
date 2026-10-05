# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Free-trial mode: `config.free.yaml` (Jev via TypeSafe direct, two OpenRouter `:free` LLMs), selected
  with `--config`, `BENCH_CONFIG` or `make ... CONFIG=`.
- Per-engine `base_url`, `api_key_env`, `structured_output`, `concurrency`, `rpm` and `results_dir`.
- Rate limiter for per-minute caps. A 429 for an exhausted daily quota stops that engine's remaining
  jobs instead of retrying them.
- `--dry` reports how many days of free-tier calls remain.

- Per-engine `max_tokens`; free `high` engine runs with `reasoning: off` and `max_tokens: 400`.
- A 400 that rejects `response_format` is resent once without the schema, and the new
  `structured_output` column in `raw.parquet` records which calls used it.
- Scorecard `attempted` / `answered` / `expected` / `deferred` columns, and an "Incomplete" banner in
  `verdict.md` until the grid is complete.

### Changed
- Quality metrics are computed over answered (HTTP 200) calls only; `error_rate` still covers every
  attempted call.
- Jobs deferred by a daily limit are not saved to `raw.parquet`.
- Retries go through the per-minute rate limiter.
- The verdict breaks cost ties by p50 latency and notes engines that report $0 (free tier).
- The paraphrase generator retries 429/5xx, exits cleanly on a daily limit, and includes the model in
  its cache key.
- Jev answers at the top level or under `answers` no longer need a `type` field (TypeSafe direct).
- Cost falls back to tokens × configured price when a provider omits `usage.cost`.
- `usage` and `reasoning` request fields are sent only to OpenRouter.
- The response cache is keyed by engine name and model.

### Fixed
- Free-trial generator model id (`google/gemma-4-26b-a4b-it:free`).
- TypeSafe key placeholder is `apikey_...`, matching real keys.

### Removed
- `data/experiments.schema.json` and the `jsonschema` dependency; `make validate` reports readiness only.
- `scripts/demo.py`'s hand-built LLM request; the demo goes through `LLMAdapter`.
- Unused `Prediction` fields (`latency_ms`, `status`, `retries_429`, `raw`) and `to_dict`.
- PNG export (`dashboard/figures.py` `__main__`) and the `static` extra.
- `OPENROUTER_APP_URL` / `OPENROUTER_APP_NAME` attribution headers.
- `n_items` config key (fixed at 300), `metrics.percentile`, and the `bench.adapters` re-exports
  (`make` lives in `bench.runner`). Engine order now follows config instead of a hardcoded list.

## [0.1.0] - 2026-10-04

### Added
- Mini-benchmark grid: 3 experiments (Choice / Noul / Score on Banking77 / BoolQ / Yelp) × 5 perturbed
  samples × 2 questions × 3 engines (`jev`, `low`, `high`) × 3 repeats = 270 calls through OpenRouter.
- Frozen data: 300 stratified items per task (seed 42), hand-written Banking77 intent descriptions,
  `data/experiments.json` with Q2 stubs for hand-written questions, and its JSON Schema.
- Jev Decisions API and chat-completion adapters with shared timing, retries on 429 / 5xx / timeouts,
  and a disk cache that makes re-runs free and reproducible.
- Metrics per cell and per experiment × engine; `cells.csv`, `scorecard.csv`, `verdict.md`.
- Streamlit dashboard: correctness grid, latency, tokens and cost, consistency, accuracy by perturbation.
- CLI `jev-bench` / `python -m bench.run` with `--dry` cost estimate and filters.
- Tooling: ruff, mypy, pytest with coverage gate, pre-commit, GitHub Actions CI on Python 3.11 / 3.12 / 3.14.
- Docs: README, `docs/experiment.md` (Mermaid diagrams), CONTRIBUTING.
