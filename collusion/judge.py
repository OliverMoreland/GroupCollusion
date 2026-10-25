"""LLM-as-judge deception scoring, run post hoc over a finished run.

For every message the judge sees what the recipients could not: the sender's
chain-of-thought when writing it, the sender's full output, and what the sender
then proposed/did that round. Results go to `<run>/judgments.jsonl` (keyed by
the message's index in events.jsonl order); re-running skips judged messages.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import prompts as P
from .config import config_from_dict
from .llm import make_client, parse_json
from .logger import read_events
from .metrics import load_judgments, write_outputs


def _followup(ev, m) -> str:
    r, a, t = m["round"], m["sender"], m["team"]
    lines = []
    later_msgs = [e for e in ev if e["type"] == "message" and e["round"] == r and e["sender"] == a
                  and e["subround"] > m["subround"]]
    for e in later_msgs:
        lines.append(f"- later message [{e['channel']}]: {e['text']}")
    for e in ev:
        if e["type"] == "proposal" and e["round"] == r and e["agent"] == a:
            lines.append(f"- proposed price: {e['price']}" + (f" (rationale: {e['rationale']})" if e.get("rationale") else ""))
            if e.get("plans"):
                lines.append(f"- notes written with the proposal: {e['plans'][:1500]}")
        if e["type"] == "team_decision" and e["round"] == r and e["team"] == t:
            if e.get("leader") == a:
                lines.append(f"- as team leader set the firm price to {e['price']:.2f}")
            if e.get("ballots", {}).get(a):
                lines.append(f"- voted for {e['ballots'][a]}'s proposal")
            lines.append(f"- firm {t}'s final price this round: {e['price']:.2f}")
        if e["type"] == "market" and e["round"] == r:
            lines.append("- market prices this round: " + ", ".join(f"{k}={v['price']:.2f}" for k, v in e["firms"].items()))
    return "\n".join(lines) or "(nothing recorded)"


def judge_run(run_dir, model: str | None = None, provider: str | None = None, limit: int | None = None) -> dict:
    run_dir = Path(run_dir)
    ev = read_events(run_dir)
    cfg = config_from_dict(next(e for e in ev if e["type"] == "run_start")["config"])
    if provider:
        cfg.llm.provider = provider
    jc = cfg.judge
    if model:
        jc.model = model
    client = make_client(cfg.llm, cfg.seed)
    calls = {e["call_id"]: e for e in ev if e["type"] == "llm_call"}  # last attempt wins
    msgs = [e for e in ev if e["type"] == "message"]
    done = {i for i, j in load_judgments(run_dir).items() if j.get("deception_score") is not None}
    todo = [i for i in range(len(msgs)) if i not in done][: limit or None]

    def one(i):
        m = msgs[i]
        call = calls.get(m["call_id"], {})
        user = P.JUDGE_TASK.format(
            sender=m["sender"], team=m["team"], channel=m["channel"], scope=m["scope"], kind=m["kind"],
            recipients=", ".join(m["recipients"]), round=m["round"], subround=m["subround"], text=m["text"],
            reasoning=(call.get("reasoning") or "(no reasoning trace available)")[:12000],
            raw_output=(call.get("response") or "")[:4000], followup=_followup(ev, m))
        messages = [{"role": "system", "content": P.JUDGE_SYSTEM}, {"role": "user", "content": user}]
        resp = client.complete(messages, model=jc.model, temperature=jc.temperature, max_tokens=jc.max_tokens,
                               reasoning=jc.reasoning, reasoning_effort=jc.reasoning_effort,
                               hints={"phase": "judge", "message_index": i})
        obj = parse_json(resp.content) or {}
        score = obj.get("deception_score")
        try:
            score = max(0.0, min(10.0, float(score)))
        except (TypeError, ValueError):
            score = None
        return {"message_index": i, "round": m["round"], "sender": m["sender"], "channel": m["channel"],
                "deception_score": score, "is_deceptive": obj.get("is_deceptive"),
                "explicit_collusion": obj.get("explicit_collusion"), "explanation": obj.get("explanation"),
                "judge_model": resp.model or jc.model, "judge_reasoning": resp.reasoning,
                "judge_raw": resp.content, "judge_prompt": user, "usage": resp.usage, "error": resp.error}

    with ThreadPoolExecutor(cfg.llm.concurrency or max(1, len(todo))) as pool, open(run_dir / "judgments.jsonl", "a") as f:
        for j in pool.map(one, todo):
            f.write(json.dumps(j, default=str) + "\n")
    return write_outputs(run_dir)
