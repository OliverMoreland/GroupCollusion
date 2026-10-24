"""CLI.

    python -m collusion run configs/base.yaml [--set team.structure=vote ...] [--mock] [--echo]
    python -m collusion sweep configs/sweep.yaml [--mock] [--dry-run] [--parallel N]
    python -m collusion judge runs/<run> [--model ...] [--mock] [--limit N]
    python -m collusion report runs/            # one row per run -> runs/report.csv
    python -m collusion show runs/<run> [--round 3] [--agent A1] [--phase message] [--no-prompt]
    python -m collusion benchmarks [--set market.n_firms=3]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

from .config import apply_overrides, config_from_dict, load_config
from .logger import read_events
from .market import Market


def _headline(s: dict) -> str:
    p, c, d = s["pricing"], s["communication"], s.get("deception")
    out = (f"{s['name']}: mean price (tail) {p['mean_price_tail']:.3f} | Nash {s['benchmarks']['nash_price']:.3f} "
           f"| monopoly {s['benchmarks']['monopoly_price']:.3f} | Δ {p['collusion_index_tail']:+.2f} "
           f"| msgs {c['n_messages']} | calls {s['llm']['n_calls']} | cost ${s['llm']['cost_usd']:.4f}")
    if d:
        out += f" | deception {d['mean_score']:.2f}"
    return out


def cmd_run(a):
    from .simulation import Simulation
    ov = list(a.set or [])
    if a.mock:
        ov.append("llm.provider=mock")
    cfg = load_config(a.config, ov)
    sim = Simulation(cfg, echo=a.echo)
    print(f"run dir: {sim.run_dir}", flush=True)
    s = sim.run()
    print(_headline(s))
    if a.judge:
        from .judge import judge_run
        s = judge_run(sim.run_dir)
        print(_headline(s))


def expand_sweep(path: str, extra_overrides=None) -> list[tuple[str, dict]]:
    """A sweep file is a base config plus `sweep: {dotted.key: [values...]}` (full grid)."""
    raw = yaml.safe_load(open(path))
    grid = raw.pop("sweep", {})
    named = raw.pop("presets", {})  # optional: named communication presets
    base = apply_overrides(raw, extra_overrides)
    keys = list(grid)
    out, seen = [], set()
    for combo in itertools.product(*(grid[k] for k in keys)):
        ov, parts = {}, []
        for k, v in zip(keys, combo):
            if k == "communication.channels" and isinstance(v, str):
                parts.append(v)
                v = named[v]
            else:
                parts.append(f"{k.split('.')[-1]}={v}")
            ov[k] = v
        data = apply_overrides(base, ov)
        # structure=single implies a team of one, so within-team channels don't exist, and the
        # sole agent is its firm's decision-maker, so sub_dm is closed to it; drop combos that
        # are then identical to one already listed
        if data.get("team", {}).get("structure") == "single":
            data["team"]["size"] = 1
            chans = data.get("communication", {}).get("channels") or []
            chans = ["cross_dm" if c == "all_dm" else c for c in chans if c not in ("team_dm", "team_open", "sub_dm")]
            data.setdefault("communication", {})["channels"] = list(dict.fromkeys(chans))
            key = json.dumps({k: v for k, v in data.items() if k != "name"}, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
        name = f"{base.get('name', 'sweep')}__" + "__".join(parts)
        data["name"] = name.replace(" ", "")
        out.append((data["name"], data))
    return out


def cmd_sweep(a):
    from .simulation import Simulation
    runs = expand_sweep(a.config, (["llm.provider=mock"] if a.mock else []) + list(a.set or []))
    print(f"{len(runs)} runs")
    if a.dry_run:
        for name, _ in runs:
            print("  ", name)
        return

    def one(item):
        name, data = item
        try:
            sim = Simulation(config_from_dict(data), echo=a.echo)
            s = sim.run()
            if a.judge:
                from .judge import judge_run
                s = judge_run(sim.run_dir)
            print(_headline(s), flush=True)
        except Exception as e:  # one failed run shouldn't stop the rest of the grid
            print(f"FAILED {name}: {type(e).__name__}: {e}", flush=True)

    # each run already fans out its own LLM calls; --parallel runs several games at once
    with ThreadPoolExecutor(a.parallel) as pool:
        list(pool.map(one, runs))
    cmd_report(argparse.Namespace(runs_dir=runs[0][1].get("output_dir", "runs")))


def cmd_judge(a):
    from .judge import judge_run
    s = judge_run(a.run_dir, model=a.model, provider="mock" if a.mock else None, limit=a.limit)
    print(_headline(s))


def cmd_report(a):
    rows = []
    for p in sorted(Path(a.runs_dir).glob("*/summary.json")):
        s = json.loads(p.read_text())
        d = s.get("deception") or {}
        rows.append({
            "run": p.parent.name, "structure": s["structure"], "team_size": s["team_size"],
            "channels": "+".join(s["channels"]) or "none", "n_firms": s["n_firms"], "seed": s["seed"],
            "model": s["model"], "rounds": s["pricing"]["rounds"],
            "mean_price_tail": s["pricing"]["mean_price_tail"], "nash": s["benchmarks"]["nash_price"],
            "monopoly": s["benchmarks"]["monopoly_price"],
            "collusion_index_tail": s["pricing"]["collusion_index_tail"],
            "profit_index_tail": s["pricing"]["profit_index_tail"],
            "n_messages": s["communication"]["n_messages"],
            "msgs_intra": s["communication"]["by_scope"].get("intra_team", 0),
            "msgs_inter": s["communication"]["by_scope"].get("inter_team", 0),
            "chars": s["communication"]["total_chars"],
            "deception_mean": d.get("mean_score"), "explicit_collusion_rate": d.get("explicit_collusion_rate"),
            "proposal_sd": s["team_dynamics"]["mean_within_team_proposal_sd"],
            "parse_errors": s["llm"]["n_parse_errors"], "cost_usd": s["llm"]["cost_usd"],
        })
    if not rows:
        print("no runs with summary.json found")
        return
    df = pd.DataFrame(rows)
    out = Path(a.runs_dir) / "report.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.drop(columns=["run", "model"]).round(3).to_string(index=False))
    print(f"\nwrote {out}")


def cmd_show(a):
    """Print LLM calls (prompt, reasoning, response) filtered by round/agent/phase."""
    for e in read_events(a.run_dir):
        if e["type"] != "llm_call":
            continue
        if (a.round is not None and e["round"] != a.round) or (a.agent and e["agent"] != a.agent) \
                or (a.phase and e["phase"] != a.phase):
            continue
        print("=" * 100)
        print(f"{e['call_id']} attempt {e['attempt']} | round {e['round']} | {e['agent']} | {e['phase']} | "
              f"{e['model']} | {e['latency_s']}s | usage {e.get('usage')}")
        if not a.no_prompt:
            for m in e["messages"]:
                print(f"--- {m['role'].upper()} ---\n{m['content']}")
        print(f"--- REASONING ---\n{e.get('reasoning') or '(none)'}")
        print(f"--- RESPONSE ---\n{e['response']}")
        if e.get("error"):
            print(f"--- ERROR ---\n{e['error']}")


def cmd_benchmarks(a):
    cfg = load_config(a.config, a.set)
    m = Market(**{k: v for k, v in vars(cfg.market).items() if k != "max_wtp"})
    print(json.dumps(m.benchmarks(), indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="collusion")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("config", nargs="?")
    r.add_argument("--set", action="append", help="dotted override, e.g. team.structure=vote")
    r.add_argument("--mock", action="store_true", help="use the offline mock LLM")
    r.add_argument("--echo", action="store_true", help="print messages/decisions as they happen")
    r.add_argument("--judge", action="store_true", help="run the deception judge afterwards")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("sweep")
    s.add_argument("config")
    s.add_argument("--set", action="append")
    s.add_argument("--mock", action="store_true")
    s.add_argument("--echo", action="store_true")
    s.add_argument("--judge", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--parallel", type=int, default=1, help="number of runs to play at the same time")
    s.set_defaults(fn=cmd_sweep)

    j = sub.add_parser("judge")
    j.add_argument("run_dir")
    j.add_argument("--model")
    j.add_argument("--mock", action="store_true")
    j.add_argument("--limit", type=int)
    j.set_defaults(fn=cmd_judge)

    rep = sub.add_parser("report")
    rep.add_argument("runs_dir", nargs="?", default="runs")
    rep.set_defaults(fn=cmd_report)

    sh = sub.add_parser("show")
    sh.add_argument("run_dir")
    sh.add_argument("--round", type=int)
    sh.add_argument("--agent")
    sh.add_argument("--phase", choices=["message", "propose", "leader", "ballot", "election"])
    sh.add_argument("--no-prompt", action="store_true")
    sh.set_defaults(fn=cmd_show)

    b = sub.add_parser("benchmarks")
    b.add_argument("config", nargs="?")
    b.add_argument("--set", action="append")
    b.set_defaults(fn=cmd_benchmarks)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
