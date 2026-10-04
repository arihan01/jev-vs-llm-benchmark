import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures

st.set_page_config(page_title="Jev vs LLM Benchmark", layout="wide")
st.title("Jev vs LLM Benchmark")
st.caption("3 experiments × 5 perturbed samples × 2 questions × 3 engines × 3 repeats")

if not (figures.RESULTS / "scorecard.csv").exists():
    st.warning("No results yet. Run `make bench`, then reload.")
    st.stop()

raw, sc = figures.load()
for exp in sorted(raw.exp.unique()):
    st.plotly_chart(figures.correctness_grid(raw, exp), use_container_width=True)

st.plotly_chart(figures.latency_box(raw), use_container_width=True)
left, right = st.columns(2)
left.plotly_chart(figures.tokens_cost(sc), use_container_width=True)
right.plotly_chart(figures.consistency_bars(sc), use_container_width=True)
st.plotly_chart(figures.accuracy_by_perturbation(raw), use_container_width=True)

st.subheader("Scorecard")
st.dataframe(sc, use_container_width=True, hide_index=True)

verdict = figures.RESULTS / "verdict.md"
if verdict.exists():
    with st.expander("Verdict", expanded=True):
        st.markdown(verdict.read_text())
