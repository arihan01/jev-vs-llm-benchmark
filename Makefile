PY ?= .venv/bin/python
BIN = $(dir $(PY))

# CONFIG=config.free.yaml switches every target to the free-trial engines
ifdef CONFIG
export BENCH_CONFIG := $(CONFIG)
endif

.DEFAULT_GOAL := help
.PHONY: help setup data experiments experiments-llm validate smoke dry bench report export dashboard diagram demo \
        test lint fmt typecheck check precommit-install clean-results

help:              ## list targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

setup:             ## create .venv and install with dev extras
	python3 -m venv .venv
	$(PY) -m pip install -q -e '.[dev]'

data:              ## sample + freeze data/{task}.jsonl (no key needed)
	$(PY) -m bench.data

experiments:       ## build data/experiments.json (refuses to overwrite; FORCE=1 to rebuild)
	$(PY) -m bench.build_experiments $(if $(FORCE),--force,)

experiments-llm:   ## fill the pending S2 paraphrases via the generator model
	$(PY) -m bench.build_experiments --llm

validate:          ## check data/experiments.json against its readiness rules
	$(PY) -m bench.build_experiments --validate

smoke:             ## one live call per engine per experiment (S1Q1)
	$(PY) scripts/smoke.py

dry:               ## job list + cost estimate, no calls
	$(PY) -m bench.run --dry

bench:             ## all 270 calls, then the report
	$(PY) -m bench.run --all

report:            ## rebuild cells / scorecard / verdict from results/raw.parquet
	$(PY) -m bench.report

export:            ## results as report.md; PDF=1 also writes report.pdf (needs: pip install -e '.[pdf]')
	$(PY) -m bench.export_md $(if $(PDF),--pdf,)

dashboard:         ## streamlit dashboard over results/
	$(PY) -m streamlit run dashboard/app.py

diagram:           ## export Mermaid diagrams to SVG (needs: npm i -g @mermaid-js/mermaid-cli)
	mmdc -i docs/experiment.md -o docs/experiment.svg

demo:              ## live side-by-side demo (needs key)
	$(PY) scripts/demo.py

test:              ## pytest with coverage gate
	$(PY) -m pytest

lint:              ## ruff lint + format check
	$(BIN)ruff check .
	$(BIN)ruff format --check .

fmt:               ## apply ruff fixes and formatting
	$(BIN)ruff check --fix .
	$(BIN)ruff format .

typecheck:         ## mypy over bench/
	$(BIN)mypy

check: lint typecheck test  ## everything CI runs

precommit-install: ## install the git pre-commit hook
	$(BIN)pre-commit install

clean-results:     ## keeps .cache/, so re-running bench is free and identical
	rm -f results/raw.parquet results/*.csv results/verdict.md
