"""Edit the active config (engines, repeats) and data/experiments.json."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import figures

from bench import build_experiments, config

st.set_page_config(page_title="Config", layout="wide")
path = figures.pick_config()
cfg = config.load()
st.title(f"Config · {path.name}")

KINDS = ["jev", "llm", "baseline"]
COLS = ["enabled", "kind", "model", "price_in", "price_out"]

st.subheader("Engines")
st.caption(
    "Toggle, rename or swap models here. Other per-engine settings (base_url, api_key_env, rpm, reasoning, "
    "temperature …) are kept as they are; edit them under *Raw YAML*. Prices are USD per million tokens and "
    "only used when the provider reports no cost."
)
rows = [{"name": n, **{c: s.get(c) for c in COLS}} for n, s in cfg["engines"].items()]
for r in rows:
    r["enabled"] = r["enabled"] is not False
edited = st.data_editor(
    pd.DataFrame(rows, columns=["name", *COLS]),
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    column_config={
        "enabled": st.column_config.CheckboxColumn(default=True),
        "kind": st.column_config.SelectboxColumn(options=KINDS, required=True),
        "price_in": st.column_config.NumberColumn(min_value=0.0, format="%.4f"),
        "price_out": st.column_config.NumberColumn(min_value=0.0, format="%.4f"),
    },
)

left, right = st.columns(2)
repeats = left.number_input("Repeats per question", 1, 20, int(cfg.get("repeats", 3)))
concurrency = right.number_input("Default concurrency", 1, 50, int(cfg.get("concurrency", 5)))


def engines_from(table: pd.DataFrame) -> dict:
    out = {}
    for r in table.to_dict("records"):
        name = str(r.get("name") or "").strip()
        if not name or not r.get("model"):
            continue
        spec = dict(cfg["engines"].get(name, {}))
        for c in COLS:
            v = r.get(c)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                spec[c] = v.item() if hasattr(v, "item") else v
        if spec.get("enabled", True):
            spec.pop("enabled", None)
        out[name] = spec
    return out


with st.expander("Raw YAML (whole config)"):
    raw_text = st.text_area("config", path.read_text(), height=400, label_visibility="collapsed")

save_as = st.text_input("Save as", path.name, help="A new name (config.<something>.yaml) keeps the original file.")
a, b = st.columns(2)
if a.button("Save table changes", type="primary"):
    new = {**cfg, "repeats": int(repeats), "concurrency": int(concurrency), "engines": engines_from(edited)}
    target = config.ROOT / Path(save_as).name
    target.write_text(yaml.safe_dump(new, sort_keys=False, allow_unicode=True))
    config._load.cache_clear()
    st.success(f"Saved {target.name}. Comments in the file are not preserved by this form.")
if b.button("Save raw YAML"):
    try:
        yaml.safe_load(raw_text)
    except yaml.YAMLError as e:
        st.error(f"Not valid YAML: {e}")
    else:
        (config.ROOT / Path(save_as).name).write_text(raw_text)
        config._load.cache_clear()
        st.success(f"Saved {Path(save_as).name}.")

st.divider()
st.subheader("Experiments (data/experiments.json)")
if not build_experiments.PATH.exists():
    st.info("No experiments file yet. Run `make data experiments experiments-llm` first.")
    st.stop()
exps = build_experiments.load_experiments()
c1, c2 = st.columns(2)
e = c1.selectbox("Experiment", list(exps))
samples = {s["id"]: s for s in exps[e]["samples"]}
sid = c2.selectbox("Sample", list(samples))
s = samples[sid]
st.caption(f"Perturbation: **{s['perturbation']}** · source `{s['source_id']}`")
# Widgets keep their typed value across reruns while their key is unchanged; a reset bumps
# this counter so every field below is recreated with the restored content.
wk = f"{e}{sid}{st.session_state.get('reset_n', 0)}"
is_dict = isinstance(s["text"], dict)


def as_editable(x):
    return json.dumps(x, ensure_ascii=False, indent=2) if is_dict else x


st.text_area("Original text (dataset)", as_editable(s["clean_text"]), disabled=True, key=f"clean{wk}")
text = st.text_area("Perturbed text sent to the engines", as_editable(s["text"]), key=f"text{wk}")
q1 = s["questions"][0]
gold1 = st.text_input(
    "Q1 gold (an option key)",
    q1["gold"] or "",
    key=f"g1{wk}",
    help="Starts as the dataset's own label; change it only if that label is wrong for this text.",
)
q2 = next((q for q in s["questions"] if q["id"] == "Q2"), None)
if q2:
    instr = st.text_input("Q2 instructions", q2["instructions"], key=f"i{wk}")
    opts = st.text_area("Q2 options (JSON: key → description)", json.dumps(q2["options"], indent=2), key=f"o{wk}")
    gold = st.text_input("Q2 gold (an option key)", q2["gold"] or "", key=f"g{wk}")

save_col, reset_col = st.columns([1, 3])
if reset_col.button(
    "Reset sample to original", help="Restores this sample's text, questions and golds from the build."
):
    build_experiments.reset_sample(exps, e, sid)
    build_experiments.save(exps)
    st.session_state["reset_n"] = st.session_state.get("reset_n", 0) + 1
    st.rerun()
if save_col.button("Save sample"):
    try:
        s["text"] = json.loads(text) if is_dict else text
        q1["gold"] = gold1 or None
        if q2:
            q2.update(instructions=instr, options=json.loads(opts), gold=gold or None)
    except json.JSONDecodeError as err:
        st.error(f"Not valid JSON: {err}")
    else:
        build_experiments.save(exps)
        problems = build_experiments.todo_report(exps)
        st.success("Saved. Cached answers for this sample are reused only if the request is identical.")
        if problems:
            st.warning("Not ready yet:\n- " + "\n- ".join(problems))
