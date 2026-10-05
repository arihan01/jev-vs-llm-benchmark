# Experiment design

## Pipeline

```mermaid
flowchart LR
  subgraph Data["1 · Data (frozen, seed 42)"]
    HF[(HF datasets<br/>Banking77 · BoolQ · Yelp)] --> S["5 samples per experiment<br/>S1 typo · S2 paraphrase · S3 padding<br/>S4 distractor · S5 injection"]
    S --> Q["2 questions per sample<br/>Q1 dataset task + gold T<br/>Q2 hand-written + gold T"]
  end
  Q --> X[data/experiments.json]
  subgraph Exp["2 · Experiments"]
    E1[E1 Choice] & E2[E2 Noul] & E3[E3 Score]
  end
  X --> E1 & E2 & E3
  subgraph Eng["3 · Engines (one OpenRouter key, temp 0, 3 repeats)"]
    J[jev · typesafe/jev-1.13<br/>Decisions API]
    L[low · qwen3.5-9b<br/>JSON schema mimics Jev]
    H[high · claude-opus-5.5<br/>JSON schema mimics Jev]
  end
  E1 & E2 & E3 --> J & L & H
  J & L & H --> R["EiSjQp × engine × repeat<br/>label · probs · confidence<br/>tokens · latency · cost · valid"]
  R --> C[(.cache/ + results/raw.parquet)]
  C --> M["4 · Metrics<br/>correct vs T · consistency<br/>latency · tokens · cost"]
  M --> O["cells.csv · scorecard.csv<br/>verdict.md · dashboard"]
```

## Index scheme

Every call is addressed as `EiSjQp × engine × repeat`, e.g. `E2S4Q1 · low · r2`.
3 × 5 × 2 × 3 × 3 = **270 calls**.

```mermaid
flowchart LR
  E["E1 Choice · E2 Noul · E3 Score"] --> S["S1 … S5<br/>one perturbation each"]
  S --> Q["Q1 dataset task<br/>Q2 hand-written"]
  Q --> N["jev · low · high"]
  N --> R["r0 · r1 · r2"]
  R --> Row["one row in raw.parquet<br/>correct vs T"]
```

## Extending

### Adding an engine

1. Add it under `engines:` in `config.yaml` with `kind` (`jev` or `llm`), `model`, and `price_in` /
   `price_out` for the dry-run estimate. Any chat model on OpenRouter that supports JSON-schema output
   works as `kind: llm` with no code changes.
2. For a different API shape, subclass `bench.adapters.base.Adapter`: implement
   `build_request(q, text, temperature)` and `parse(q, response) -> Prediction`, then route to it in
   `bench/adapters/__init__.py::make`. HTTP, retries and timing come from the base class.
3. Add the engine's colour to `COLORS` in `dashboard/figures.py` and a parsing test next to
   `tests/test_jev_adapter.py`.

### Adding a perturbation

1. Write the transformation in `bench/perturb.py`. It must keep the gold answer unchanged and be
   deterministic given the `random.Random` it receives.
2. Add a branch for it in `bench/build_experiments.py::build_sample`, and use it in `sample_perturbations` in `config.yaml`.
3. Rebuild with `make experiments FORCE=1`. This throws away the hand-written Q2s, so copy them out
   first.
