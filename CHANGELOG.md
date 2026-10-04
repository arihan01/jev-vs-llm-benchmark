# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
