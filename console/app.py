"""Live console over runs/**/events.jsonl: streamlit run console/app.py"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
STRUCTURE_ORDER = ["vote", "average", "leader", "elected_leader", "representatives", "single"]
COMMS_ORDER = ["none", "sub_dm", "cross_dm", "all_dm", "global_open"]

st.set_page_config(page_title="Collusion console", layout="wide")


class RunTail:
    """Incrementally parsed events.jsonl. Prompts are dropped from memory and re-read from disk on demand."""

    def __init__(self, path):
        self.path, self.offset, self.events, self.lock = path, 0, [], threading.Lock()

    def update(self):
        with self.lock:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                chunk = f.read()
            pos = self.offset
            for line in chunk.split(b"\n")[:-1]:  # the last piece is empty or a line still being written
                ev = json.loads(line)
                if ev["type"] == "llm_call":
                    ev.pop("messages", None)
                    ev["_offset"] = pos
                self.events.append(ev)
                pos += len(line) + 1
            self.offset = pos
        return self

    def prompt(self, offset):
        with open(self.path, "rb") as f:
            f.seek(offset)
            return json.loads(f.readline())["messages"]

    def of(self, kind):
        return [e for e in self.events if e["type"] == kind]


@st.cache_resource
def tails():
    return {}


def tail(run_dir):
    return tails().setdefault(str(run_dir), RunTail(run_dir / "events.jsonl")).update()


@st.cache_data(max_entries=256)
def judgments(path, mtime):
    return {j["message_index"]: j for j in map(json.loads, open(path)) if j.get("deception_score") is not None}


def judged(run_dir):
    p = run_dir / "judgments.jsonl"
    return judgments(str(p), p.stat().st_mtime) if p.exists() else {}


def cross_comms(channels):
    return "+".join(c for c in channels if c not in ("team_open", "team_dm")) or "none"


def run_info(t):
    start = next(iter(t.of("run_start")), None)
    if start is None:
        return None
    cfg, markets, last = start["config"], t.of("market"), t.events[-1]
    ended = bool(t.of("run_end"))
    idle = time.time() - last["t"]
    return {
        "structure": cfg["team"]["structure"], "cross comms": cross_comms(cfg["communication"]["channels"]),
        "seed": cfg["seed"], "state": "done" if ended else ("stalled?" if idle > 600 else "running"),
        "round": f"{len(markets)}/{cfg['rounds']}",
        "now": "-" if ended else f"r{last.get('round', '-')} {last.get('phase', last['type'])}",
        "collusion (last 5)": np.mean([m["collusion_index"] for m in markets[-5:]]) if markets else np.nan,
        "msgs/round": len(t.of("message")) / max(len(markets), 1),
        "errors": len(t.of("llm_error")) + len(t.of("parse_error")),
        "cost $": sum((c.get("usage") or {}).get("cost", 0) or 0 for c in t.of("llm_call")),
        "idle (s)": round(idle),
    }


def ordered(values, order):
    return sorted(values, key=lambda v: (order.index(v) if v in order else len(order), v))


def overview(group):
    rows = []
    for d in sorted(p.parent for p in group.glob("**/events.jsonl")):
        info = run_info(tail(d))
        if info:
            rows.append({"run": d.name, **info})
    table = pd.DataFrame(rows)
    if table.empty:
        st.info("No runs here yet.")
        return
    c = st.columns(5)
    c[0].metric("runs", len(table))
    c[1].metric("running", (table["state"] == "running").sum())
    c[2].metric("done", (table["state"] == "done").sum())
    c[3].metric("stalled?", (table["state"] == "stalled?").sum())
    c[4].metric("cost so far", f"${table['cost $'].sum():.2f}")
    grid = table.pivot_table(index="cross comms", columns="structure", values="collusion (last 5)", aggfunc="mean")
    grid = grid.reindex(index=ordered(grid.index, COMMS_ORDER), columns=ordered(grid.columns, STRUCTURE_ORDER))
    st.plotly_chart(px.imshow(grid, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-0.5, zmax=1.5, aspect="auto",
                              labels={"color": "collusion index"}), width="stretch")
    st.dataframe(table, width="stretch", hide_index=True)


def run_view(run_dir):
    t = tail(run_dir)
    info, start = run_info(t), next(iter(t.of("run_start")), None)
    if start is None:
        st.info("Run has not started logging yet.")
        return
    b = start["benchmarks"]
    st.caption(" | ".join(f"{k}: {v}" for k, v in info.items() if k in ("structure", "cross comms", "seed", "state", "round", "now")))
    markets = t.of("market")
    if markets:
        prices = pd.DataFrame([{"round": m["round"], "firm": f, **v} for m in markets for f, v in m["firms"].items()])
        c1, c2 = st.columns(2)
        fig = px.line(prices, x="round", y="price", color="firm", markers=True)
        for name, y in (("Nash", b["nash_price"]), ("monopoly", b["monopoly_price"])):
            fig.add_hline(y=y, line_dash="dash", annotation_text=name)
        c1.plotly_chart(fig, width="stretch")
        fig = px.line(pd.DataFrame(markets), x="round", y="collusion_index", markers=True)
        for name, y in (("Nash", 0), ("monopoly", 1)):
            fig.add_hline(y=y, line_dash="dash", annotation_text=name)
        c2.plotly_chart(fig, width="stretch")
        msgs = t.of("message")
        if msgs:
            counts = pd.DataFrame(msgs).groupby(["round", "scope"]).size().reset_index(name="messages")
            c1.plotly_chart(px.bar(counts, x="round", y="messages", color="scope"), width="stretch")
        decisions = [d for d in t.of("team_decision") if d.get("leader")]
        if decisions:
            gaps = pd.DataFrame([{"round": d["round"], "firm": d["team"], "leader price - mean advice":
                                  d["price"] - np.mean([p for a, p in d["proposals"].items() if p is not None and a != d["leader"]] or [np.nan])}
                                 for d in decisions])
            c2.plotly_chart(px.line(gaps, x="round", y="leader price - mean advice", color="firm", markers=True), width="stretch")

    tabs = st.tabs(["messages", "decisions", "reasoning", "elections", "errors", "setup"])
    with tabs[0]:
        msgs = t.of("message")
        if msgs:
            m = pd.DataFrame(msgs)
            j = judged(run_dir)
            m["deception"] = [j.get(i, {}).get("deception_score") for i in range(len(m))]
            m["collusion?"] = [j.get(i, {}).get("explicit_collusion") for i in range(len(m))]
            f = st.columns(4)
            rnd = f[0].selectbox("round", ["all"] + sorted(m["round"].unique().tolist(), reverse=True))
            scope = f[1].multiselect("scope", sorted(m["scope"].unique()), default=sorted(m["scope"].unique()))
            agent = f[2].selectbox("agent (sent or received)", ["all"] + sorted(set(m["sender"])))
            search = f[3].text_input("text contains")
            m = m[m["scope"].isin(scope)]
            m = m if rnd == "all" else m[m["round"] == rnd]
            m = m if agent == "all" else m[(m["sender"] == agent) | m["recipients"].map(lambda r: agent in r)]
            m = m[m["text"].str.contains(search, case=False, regex=False)] if search else m
            st.dataframe(m.iloc[::-1][["round", "subround", "channel", "scope", "sender", "recipients", "text", "deception", "collusion?"]],
                         width="stretch", hide_index=True, height=500)
    with tabs[1]:
        d = t.of("team_decision")
        if d:
            st.dataframe(pd.DataFrame(d[::-1]).drop(columns=["t"], errors="ignore"), width="stretch", hide_index=True)
    with tabs[2]:
        calls = t.of("llm_call")
        if calls:
            c = pd.DataFrame(calls)
            f = st.columns(3)
            rnd = f[0].selectbox("round ", sorted(c["round"].unique().tolist(), reverse=True))
            c = c[c["round"] == rnd]
            agent = f[1].selectbox("agent", sorted(c["agent"].unique()))
            c = c[c["agent"] == agent].reset_index(drop=True)
            labels = [f"{p} (call {cid}, attempt {a})" for p, cid, a in zip(c["phase"], c["call_id"], c["attempt"])]
            call = c.iloc[labels.index(f[2].selectbox("call", labels))]
            st.markdown(f"**model** `{call['model']}` | **usage** `{call['usage']}` | **latency** {call['latency_s']}s"
                        + (f" | **error** `{call['error']}`" if call.get("error") else ""))
            st.text_area("reasoning", call["reasoning"] or "", height=300)
            st.text_area("response", call["response"] or "", height=150)
            with st.expander("prompt"):
                for msg in t.prompt(call["_offset"]):
                    st.markdown(f"**{msg['role']}**")
                    st.text(msg["content"])
    with tabs[3]:
        e = t.of("election")
        if e:
            st.dataframe(pd.DataFrame(e[::-1]).drop(columns=["t", "call_ids"], errors="ignore"), width="stretch", hide_index=True)
    with tabs[4]:
        errs = t.of("llm_error") + t.of("parse_error")
        if errs:
            st.dataframe(pd.DataFrame(errs).sort_values("t", ascending=False), width="stretch", hide_index=True)
    with tabs[5]:
        st.json({"benchmarks": b, "teams": start["teams"], "leaders": start["leaders"], "config": start["config"]})


root = Path(st.sidebar.text_input("runs directory", str(REPO / "runs")))
groups = [root] + sorted({p.parent.parent for p in root.glob("**/events.jsonl")} - {root}, key=lambda p: p.stat().st_mtime, reverse=True)
group = st.sidebar.selectbox("group", groups, format_func=lambda p: str(p.relative_to(root)) if p != root else "(all)")
runs = sorted((p.parent for p in group.glob("**/events.jsonl")), key=lambda p: p.stat().st_mtime, reverse=True)
run = st.sidebar.selectbox("run", [None] + runs, format_func=lambda p: "(overview)" if p is None else p.name)
refresh = st.sidebar.number_input("auto-refresh (s, 0 = off)", 0, 600, 5)

with st.sidebar.expander("start a run"):
    configs = sorted(str(p.relative_to(REPO)) for p in (REPO / "configs").glob("*.yaml"))
    config = st.selectbox("config", configs, index=configs.index("configs/smoke.yaml") if "configs/smoke.yaml" in configs else 0)
    sweep = st.checkbox("sweep (config has a sweep: block)", value=config.endswith(("sweep.yaml", "grid.yaml")))
    overrides = st.text_area("overrides, one dotted key=value per line", placeholder="rounds=10\nteam.structure=leader")
    mock = st.checkbox("mock LLM (offline, free)")
    parallel = st.number_input("parallel runs (sweep)", 1, 64, 4) if sweep else 1
    if st.button("start"):
        cmd = [sys.executable, "-m", "collusion", "sweep" if sweep else "run", config]
        cmd += [a for o in overrides.splitlines() if o.strip() for a in ("--set", o.strip())]
        cmd += ["--mock"] * mock + (["--parallel", str(parallel)] if sweep else [])
        log = REPO / "runs" / f"launch-{time.strftime('%Y%m%d-%H%M%S')}.log"
        log.parent.mkdir(exist_ok=True)
        subprocess.Popen(cmd, cwd=REPO, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        st.success(f"started; output in {log.relative_to(REPO)}. It appears under group/run once round 1 logs.")


@st.fragment(run_every=refresh or None)
def live():
    if run is None:
        overview(group)
    else:
        run_view(run)


live()
