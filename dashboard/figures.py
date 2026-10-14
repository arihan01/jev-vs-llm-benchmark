"""Plotly figures for the Streamlit dashboard, built from raw.parquet and
scorecard.csv."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Streamlit runs this directory as scripts, so the repo root is not on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import config, metrics
from bench.config import engine_order, results_dir

COLORS = {"jev": "#2a6fdb", "low": "#e07b39", "high": "#3a9d5d", "stat": "#7f8c8d"}


def pick_config() -> Path:
    """Sidebar choice of config*.yaml, shared by every page and passed to bench."""
    import streamlit as st

    names = sorted(p.name for p in config.ROOT.glob("config*.yaml"))
    # Streamlit drops a widget's state on pages that don't render it; re-assigning keeps it.
    st.session_state.config = st.session_state.get("config", "config.yaml")
    name = st.sidebar.selectbox("Config", names, key="config")
    config.use_config(config.ROOT / name)
    return config.config_path()


def load():
    out = results_dir()
    return pd.read_parquet(out / "raw.parquet"), pd.read_csv(out / "scorecard.csv")


def correctness_grid(raw, exp):
    """Rows engines, columns SjQp; cell = share of repeats correct, one ✓/✗ per
    repeat, the answers themselves on hover."""
    c = metrics.cells(raw[raw.exp == exp])
    c["col"] = c["sample"] + c["question"]
    c["marks"] = [
        " ".join("✓" if a == g else "✗" for a in labels.split("/"))
        for labels, g in zip(c["labels"], c["gold"], strict=True)
    ]
    # Numeric sample order (S10 after S9), then question id.
    cols = sorted(c["col"].unique(), key=lambda s: (int(s[1 : s.index("Q")]), s[s.index("Q") :]))
    z = c.pivot(index="engine", columns="col", values="correct_rate").reindex(
        index=engine_order(c.engine), columns=cols
    )
    txt = c.pivot(index="engine", columns="col", values="marks").reindex(index=z.index, columns=cols)
    answers = c.pivot(index="engine", columns="col", values="labels").reindex(index=z.index, columns=cols)
    gold = c.drop_duplicates("col").set_index("col")["gold"].reindex(cols)
    fig = go.Figure(
        go.Heatmap(
            z=z.values,
            x=[f"{k}<br>T={str(gold[k])[:14]}" for k in cols],
            y=z.index,
            zmin=0,
            zmax=1,
            colorscale=[[0, "#d64545"], [0.5, "#f2c94c"], [1, "#3a9d5d"]],
            text=txt.values,
            customdata=answers.values,
            texttemplate="%{text}",
            textfont={"size": 14},
            hovertemplate="%{y} · %{x}<br>answers %{customdata}<br>correct %{z:.0%}<extra></extra>",
            colorbar=dict(title="correct", tickformat=".0%"),
        )
    )
    fig.update_layout(title=f"{exp}: correct vs T per sample × question (3 repeats)", height=320)
    return fig


def latency_box(raw):
    d = raw[raw.status == 200]
    return px.box(
        d,
        x="engine",
        y="latency_ms",
        color="engine",
        facet_col="exp",
        points="all",
        color_discrete_map=COLORS,
        category_orders={"engine": engine_order(d.engine)},
        title="Latency per call (ms)",
    )


def tokens_cost(sc):
    """Input and output tokens per engine and experiment, stacked; label = total cost."""
    fig = go.Figure()
    for e in engine_order(sc.engine):
        d = sc[sc.engine == e]
        x = [d.exp, [e] * len(d)]
        fig.add_bar(x=x, y=d.tokens_in, name=f"{e} in", marker_color=COLORS.get(e), legendgroup=e)
        fig.add_bar(
            x=x,
            y=d.tokens_out,
            name=f"{e} out",
            marker_color=COLORS.get(e),
            marker_pattern_shape="/",
            legendgroup=e,
            text=[f"${c:.4f}" for c in d.total_cost],
            textposition="outside",
        )
    fig.update_layout(title="Tokens in / out (hatched) per experiment; label = total cost", barmode="stack")
    return fig


TRADEOFFS = [
    ("cost_per_100_questions", "accuracy", "Cost vs accuracy"),
    ("p50_latency_ms", "accuracy", "Latency vs accuracy"),
    ("p50_latency_ms", "cost_per_100_questions", "Latency vs cost"),
]


def tradeoffs(sc):
    """The three pairs of cost, accuracy and time, one point per engine × experiment.
    Up-and-left is better on every chart except Latency vs cost, where down-and-left is."""
    return [
        px.scatter(
            sc,
            x=x,
            y=y,
            color="engine",
            symbol="exp",
            color_discrete_map=COLORS,
            category_orders={"engine": engine_order(sc.engine)},
            hover_data=["exp", "model", "accuracy", "p50_latency_ms", "cost_per_100_questions"],
            title=title,
        )
        .update_traces(marker_size=12)
        .update_layout(legend=dict(orientation="h", y=-0.3), height=480)
        for x, y, title in TRADEOFFS
    ]


def consistency_bars(sc):
    d = sc.melt(
        id_vars=["exp", "engine"],
        value_vars=[c for c in ("consistency", "prob_consistency") if c in sc],
        var_name="measure",
    )
    return px.bar(
        d,
        x="exp",
        y="value",
        color="engine",
        pattern_shape="measure",
        barmode="group",
        range_y=[0, 1.05],
        color_discrete_map=COLORS,
        category_orders={"engine": engine_order(sc.engine)},
        title="Consistency: same label every repeat (plain) and same probabilities (1 - TV spread, hatched)",
    ).update_layout(yaxis_tickformat=".0%")


def accuracy_by_perturbation(raw):
    d = metrics.accuracy_by_perturbation(raw)
    return px.bar(
        d,
        x="perturbation",
        y="accuracy",
        color="engine",
        barmode="group",
        range_y=[0, 1.05],
        color_discrete_map=COLORS,
        category_orders={"engine": engine_order(d.engine)},
        title="Accuracy by perturbation (all experiments)",
    ).update_layout(yaxis_tickformat=".0%")
