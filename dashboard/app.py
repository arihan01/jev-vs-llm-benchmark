"""Streamlit dashboard over results/: correctness grids, latency, tokens,
consistency and the verdict. Run with `make dashboard`."""

import sys
from pathlib import Path

import streamlit as st

# Streamlit runs this file as a script; make the sibling module importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures

from bench.export_md import DATASETS

st.set_page_config(page_title="Jev vs LLM Benchmark", layout="wide")
figures.pick_config()
st.title("Jev vs LLM Benchmark")
st.caption(f"Results in `{figures.results_dir().relative_to(figures.config.ROOT)}`")

if not (figures.results_dir() / "scorecard.csv").exists():
    st.warning("No results yet for this config. Start one on the Run page.")
    st.stop()

raw, sc = figures.load()
st.caption(
    "One experiment per answer format. Each has 5 texts: four altered in one way each (typo, paraphrase, "
    "padding, distractor) and S5 unaltered as a control. Two questions per text: Q1 is the dataset's own task, "
    "Q2 a hand-written question about the same text. Cells show whether each engine's answer matched the human "
    "answer T, one mark per repeat."
)
for exp in sorted(raw.exp.unique()):
    kind = raw.loc[raw.exp == exp, "type"].iloc[0]
    name, what = DATASETS.get(kind, (kind, ""))
    st.markdown(f"**{exp} · {kind}** — {name}: {what}.")
    st.plotly_chart(figures.correctness_grid(raw, exp), use_container_width=True)

for col, fig in zip(st.columns(3), figures.tradeoffs(sc), strict=True):
    col.plotly_chart(fig, use_container_width=True)
st.plotly_chart(figures.latency_box(raw), use_container_width=True)
left, right = st.columns(2)
left.plotly_chart(figures.tokens_cost(sc), use_container_width=True)
right.plotly_chart(figures.consistency_bars(sc), use_container_width=True)
st.plotly_chart(figures.accuracy_by_perturbation(raw), use_container_width=True)

st.subheader("Scorecard")
st.dataframe(sc, use_container_width=True, hide_index=True)

verdict = figures.results_dir() / "verdict.md"
if verdict.exists():
    with st.expander("Verdict", expanded=True):
        st.markdown(verdict.read_text())
