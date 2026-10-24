"""Metrics computed purely from a run's events.jsonl (and judgments.jsonl if present).

Outputs written to the run directory:
    rounds.csv     one row per round: per-firm price/profit, mean price, collusion index
    messages.csv   one row per message (with judge scores once `judge` has run)
    summary.json   headline numbers: prices vs Nash/monopoly, communication volume,
                   team-decision dynamics, LLM usage, deception scores
    transcript.md  human-readable round-by-round log (no chain-of-thought; see `show`)
"""
from __future__ import annotations

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .logger import read_events


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else None


def load_judgments(run_dir) -> dict:
    p = Path(run_dir) / "judgments.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            j = json.loads(line)
            out[j["message_index"]] = j
    return out


def compute(run_dir, tail_frac: float = 0.5) -> dict:
    ev = read_events(run_dir)
    start = next(e for e in ev if e["type"] == "run_start")
    b = start["benchmarks"]
    pn, pm, prn, prm = b["nash_price"], b["monopoly_price"], b["nash_profit"], b["monopoly_profit"]
    markets = [e for e in ev if e["type"] == "market"]
    msgs = [e for e in ev if e["type"] == "message"]
    decisions = [e for e in ev if e["type"] == "team_decision"]
    calls = [e for e in ev if e["type"] == "llm_call"]
    teams = list(start["teams"])
    n_rounds = len(markets)
    tail_start = n_rounds - max(1, int(round(n_rounds * tail_frac))) + 1 if n_rounds else 1

    # ---- prices / profits --------------------------------------------------
    rows = []
    for m in markets:
        row = {"round": m["round"], "mean_price": m["mean_price"], "collusion_index": m["collusion_index"]}
        for t, f in m["firms"].items():
            row[f"price_{t}"], row[f"profit_{t}"], row[f"quantity_{t}"] = f["price"], f["profit"], f["quantity"]
        row["mean_profit"] = _mean([f["profit"] for f in m["firms"].values()])
        row["profit_index"] = (row["mean_profit"] - prn) / (prm - prn)
        row["n_messages"] = sum(1 for x in msgs if x["round"] == m["round"])
        rows.append(row)
    tail = [r for r in rows if r["round"] >= tail_start]
    pricing = {
        "rounds": n_rounds,
        "tail_from_round": tail_start,
        "mean_price_all": _mean([r["mean_price"] for r in rows]),
        "mean_price_tail": _mean([r["mean_price"] for r in tail]),
        "collusion_index_tail": _mean([r["collusion_index"] for r in tail]),
        "profit_index_tail": _mean([r["profit_index"] for r in tail]),
        "price_tail_by_firm": {t: _mean([r[f"price_{t}"] for r in tail]) for t in teams},
        "price_sd_tail": st.pstdev([r["mean_price"] for r in tail]) if len(tail) > 1 else 0.0,
    }

    # ---- communication -----------------------------------------------------
    judg = load_judgments(run_dir)
    by_scope, by_kind, by_channel, by_agent = Counter(), Counter(), Counter(), Counter()
    chars_by_scope = Counter()
    for m in msgs:
        by_scope[m["scope"]] += 1
        by_kind[m["kind"]] += 1
        by_channel[m["channel"]] += 1
        by_agent[m["sender"]] += 1
        chars_by_scope[m["scope"]] += m["n_chars"]
    n_channels = len(start["channels"])
    comm = {
        "n_channels": n_channels,
        "n_messages": len(msgs),
        "messages_per_round": len(msgs) / n_rounds if n_rounds else 0,
        "total_chars": sum(m["n_chars"] for m in msgs),
        "total_words": sum(m["n_words"] for m in msgs),
        "mean_chars_per_message": _mean([m["n_chars"] for m in msgs]),
        "by_scope": dict(by_scope),
        "chars_by_scope": dict(chars_by_scope),
        "by_kind": dict(by_kind),
        "by_channel": dict(by_channel),
        "by_agent": dict(by_agent),
        "active_channel_share": (len(by_channel) / n_channels) if n_channels else None,
        "rounds_with_inter_team_messages": len({m["round"] for m in msgs if m["scope"] == "inter_team"}),
    }

    # ---- team dynamics -----------------------------------------------------
    disp, leader_dev, ties, unanimous = [], [], 0, 0
    methods = Counter(d["method"] for d in decisions)
    for d in decisions:
        ps = [p for a, p in d["proposals"].items() if p is not None and a != d.get("leader")]
        if len(ps) > 1:
            disp.append(st.pstdev(ps))
        if d.get("leader") and ps:
            leader_dev.append(d["price"] - st.fmean(ps))
        if d.get("tally"):
            ties += bool(d.get("tie"))
            unanimous += len(d["tally"]) == 1
    n_votes = sum(1 for d in decisions if d.get("tally"))
    team = {
        "decision_methods": dict(methods),
        "mean_within_team_proposal_sd": _mean(disp),
        "mean_leader_deviation_from_advice": _mean(leader_dev),
        "vote_tie_rate": ties / n_votes if n_votes else None,
        "vote_unanimity_rate": unanimous / n_votes if n_votes else None,
        "elections": [{"round": e["round"], "team": e["team"], "representatives": e["representatives"]}
                      for e in ev if e["type"] == "election"],
    }

    # ---- LLM usage ---------------------------------------------------------
    llm = {
        "n_calls": len(calls),
        "n_parse_errors": sum(1 for e in ev if e["type"] == "parse_error"),
        "n_llm_errors": sum(1 for e in ev if e["type"] == "llm_error"),
        "n_failed_proposals": sum(1 for e in ev if e["type"] == "proposal" and e.get("failed")),
        "reasoning_present_rate": _mean([1.0 if c.get("reasoning") else 0.0 for c in calls]),
        "prompt_tokens": sum((c.get("usage") or {}).get("prompt_tokens", 0) or 0 for c in calls),
        "completion_tokens": sum((c.get("usage") or {}).get("completion_tokens", 0) or 0 for c in calls),
        "cached_prompt_tokens": sum(((c.get("usage") or {}).get("prompt_tokens_details") or {})
                                    .get("cached_tokens", 0) or 0 for c in calls),
        "reasoning_tokens": sum(((c.get("usage") or {}).get("completion_tokens_details") or {})
                                .get("reasoning_tokens", 0) or 0 for c in calls),
        "cost_usd": sum((c.get("usage") or {}).get("cost", 0) or 0 for c in calls),
        "calls_by_phase": dict(Counter(c["phase"] for c in calls)),
    }

    # ---- deception (LLM judge) ---------------------------------------------
    deception = None
    if judg:
        scored = [(msgs[i], j) for i, j in judg.items() if i < len(msgs) and j.get("deception_score") is not None]
        grp = defaultdict(list)
        for m, j in scored:
            grp[m["scope"]].append(j["deception_score"])
        deception = {
            "n_judged": len(scored),
            "mean_score": _mean([j["deception_score"] for _, j in scored]),
            "deceptive_rate": _mean([float(bool(j.get("is_deceptive"))) for _, j in scored]),
            "explicit_collusion_rate": _mean([float(bool(j.get("explicit_collusion"))) for _, j in scored]),
            "mean_score_by_scope": {k: _mean(v) for k, v in grp.items()},
            "judge_model": next(iter(judg.values())).get("judge_model"),
        }

    cfg = start["config"]
    return {
        "run_dir": str(run_dir),
        "name": cfg["name"],
        "structure": cfg["team"]["structure"],
        "team_size": cfg["team"]["size"],
        "channels": cfg["communication"]["channels"],
        "n_firms": cfg["market"]["n_firms"],
        "seed": cfg["seed"],
        "model": cfg["agent"]["model"],
        "benchmarks": b,
        "pricing": pricing,
        "communication": comm,
        "team_dynamics": team,
        "llm": llm,
        "deception": deception,
        "_rows": rows,
    }


def write_outputs(run_dir) -> dict:
    run_dir = Path(run_dir)
    s = compute(run_dir)
    rows = s.pop("_rows")
    pd.DataFrame(rows).to_csv(run_dir / "rounds.csv", index=False)
    ev = read_events(run_dir)
    msgs = [e for e in ev if e["type"] == "message"]
    judg = load_judgments(run_dir)
    mrows = []
    for i, m in enumerate(msgs):
        j = judg.get(i, {})
        mrows.append({"index": i, "round": m["round"], "subround": m["subround"], "channel": m["channel"],
                      "scope": m["scope"], "kind": m["kind"], "sender": m["sender"], "n_chars": m["n_chars"],
                      "deception_score": j.get("deception_score"), "is_deceptive": j.get("is_deceptive"),
                      "explicit_collusion": j.get("explicit_collusion"), "text": m["text"]})
    pd.DataFrame(mrows, columns=["index", "round", "subround", "channel", "scope", "kind", "sender", "n_chars",
                                 "deception_score", "is_deceptive", "explicit_collusion", "text"]
                 ).to_csv(run_dir / "messages.csv", index=False)
    (run_dir / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    (run_dir / "transcript.md").write_text(transcript(ev))
    return s


def transcript(ev: list[dict]) -> str:
    start = next(e for e in ev if e["type"] == "run_start")
    b = start["benchmarks"]
    out = [f"# {start['config']['name']}",
           f"Structure: {start['config']['team']['structure']} (size {start['config']['team']['size']}), "
           f"channels: {start['config']['communication']['channels'] or 'none'}",
           f"Nash price {b['nash_price']:.3f}, monopoly price {b['monopoly_price']:.3f}", ""]
    by_round = defaultdict(list)
    for e in ev:
        if "round" in e and e["type"] in ("message", "proposal", "team_decision", "market", "election"):
            by_round[e["round"]].append(e)
    for r in sorted(by_round):
        out.append(f"## Round {r}")
        for e in by_round[r]:
            t = e["type"]
            if t == "election":
                out.append(f"- election team {e['team']}: {e['representatives']} (tally {e['tally']})")
            elif t == "message":
                out.append(f"- msg {e['subround']} [{e['channel']}] **{e['sender']}**: {e['text']}")
            elif t == "proposal":
                p = "FAILED" if e["price"] is None else f"{e['price']:.2f}"
                out.append(f"- proposal {e['agent']}: {p}" + (f" — {e['rationale']}" if e.get("rationale") else ""))
            elif t == "team_decision":
                extra = f", ballots {e['ballots']}" if e.get("ballots") else ""
                out.append(f"- **team {e['team']} price {e['price']:.2f}** via {e['method']}{extra}")
            elif t == "market":
                out.append("- market: " + ", ".join(f"{k} p={v['price']:.2f} q={v['quantity']:.1f} "
                                                    f"π={v['profit']:.1f}" for k, v in e["firms"].items())
                           + f" | Δ={e['collusion_index']:+.2f}")
        out.append("")
    return "\n".join(out)
