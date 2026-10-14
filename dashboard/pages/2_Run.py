"""Start a benchmark run (or a dry run) for the active config and stream its log."""

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import figures

from bench import build_experiments, config

st.set_page_config(page_title="Run", layout="wide")
path = figures.pick_config()
st.title(f"Run · {path.name}")

engines = config.engines(include_disabled=True)
st.subheader("Engines and keys")
for name, spec in engines.items():
    on = spec.get("enabled", True)
    env = None if spec["kind"] == "baseline" else spec.get("api_key_env", config.DEFAULT_KEY_ENV)
    ok = env is None or os.environ.get(env, "") not in ("", *config.PLACEHOLDERS)
    key = "no key needed" if env is None else f"`{env}` {'set' if ok else '**missing** (add it to .env)'}"
    icon = "❌" if not ok else "✅" if on else "⏸️"
    state = "" if on else " · *off: runs only when picked under Engines below*"
    st.markdown(f"{icon} **{name}** · `{spec['model']}` · {key}{state}")
st.caption(
    "✅ runs by default · ⏸️ off in the config · ❌ key missing. Turn engines on or off on the Config page. "
    "Cached answers are reused, so re-running is free."
)

exps = build_experiments.load_experiments() if build_experiments.PATH.exists() else {}
c = st.columns(5)
exp = c[0].multiselect("Experiments", list(exps), placeholder="all")
eng = c[1].multiselect(
    "Engines",
    list(engines),
    default=[n for n, s in engines.items() if s.get("enabled", True)],
    placeholder="pick at least one",
    key=f"engines-{path.name}",
)
samples = sorted({s["id"] for e in exps.values() for s in e["samples"]}, key=lambda x: int(x[1:]))
smp = c[2].multiselect("Samples", samples, placeholder="all")
qs = c[3].multiselect("Questions", ["Q1", "Q2"], placeholder="all")
rep = c[4].number_input("Repeats", 1, 20, int(config.load().get("repeats", 3)))

args = [sys.executable, "-m", "bench.run", "--config", str(path), "--repeats", str(rep)]
for flag, vals in (("--exp", exp), ("--engine", eng), ("--sample", smp), ("--question", qs)):
    if vals:
        args += [flag, ",".join(vals)]

a, b, _ = st.columns([2, 1, 4])
dry = a.button("Dry run (estimate)", disabled=not eng)
live = b.button("Run", type="primary", disabled=not (exps and eng))
if dry or live:
    cmd = [*args, "--dry"] if dry else args
    st.code(" ".join(cmd[1:]), language="bash")
    box, lines = st.empty(), []
    with subprocess.Popen(
        cmd, cwd=config.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as proc:
        for line in proc.stdout or []:
            lines.append(line.rstrip())
            box.code("\n".join(lines[-400:]))
    if proc.returncode:
        st.error(f"Exited with code {proc.returncode}.")
    elif live:
        st.success("Done. The report was rebuilt; open the main page for results.")
