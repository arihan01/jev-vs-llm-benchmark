import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench import metrics
from bench.config import engine_order, results_dir

COLORS = {"jev": "#2a6fdb", "low": "#e07b39", "high": "#3a9d5d"}


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
    fig = go.Figure()
    for e in engine_order(sc.engine):
        d = sc[sc.engine == e]
        fig.add_bar(
            x=d.exp,
            y=d.tokens_in + d.tokens_out,
            name=f"{e} tokens",
            marker_color=COLORS.get(e),
            customdata=d.total_cost,
            hovertemplate="%{x}: %{y:,} tokens<br>$%{customdata:.5f}<extra></extra>",
            text=[f"${c:.4f}" for c in d.total_cost],
            textposition="outside",
        )
    fig.update_layout(title="Tokens per experiment (label = total cost)", barmode="group", yaxis_title="tokens")
    return fig


def consistency_bars(sc):
    return px.bar(
        sc,
        x="exp",
        y="consistency",
        color="engine",
        barmode="group",
        range_y=[0, 1.05],
        color_discrete_map=COLORS,
        category_orders={"engine": engine_order(sc.engine)},
        title="Consistency: share of questions with 3 identical valid answers",
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
