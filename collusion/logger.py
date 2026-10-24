"""Append-only JSONL event log. Every LLM call (full prompt, response, reasoning,
usage), every message, proposal, vote and market outcome is one line in
`<run_dir>/events.jsonl`, so all metrics can be recomputed from the log alone."""
from __future__ import annotations

import itertools
import json
import threading
import time
from pathlib import Path


class RunLogger:
    def __init__(self, run_dir: Path, echo: bool = False):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self._f = open(self.path, "a", buffering=1)
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self.echo = echo

    def new_call_id(self) -> str:
        with self._lock:
            return f"c{next(self._ids):06d}"

    def log(self, type_: str, **fields):
        ev = {"t": round(time.time(), 3), "type": type_, **fields}
        line = json.dumps(ev, default=str)
        with self._lock:
            self._f.write(line + "\n")
        if self.echo and type_ in ("market", "message", "team_decision", "parse_error", "llm_error"):
            print(_short(ev), flush=True)

    def close(self):
        self._f.close()


def _short(ev: dict) -> str:
    t = ev["type"]
    if t == "market":
        return f"[r{ev['round']}] prices " + " ".join(f"{k}={v['price']:.2f}" for k, v in ev["firms"].items()) \
            + f"  Δ={ev['collusion_index']:+.2f}"
    if t == "message":
        return f"[r{ev['round']}.{ev['subround']}] {ev['sender']} -> {ev['channel']}: {ev['text'][:120]}"
    if t == "team_decision":
        return f"[r{ev['round']}] team {ev['team']} ({ev['structure']}) -> {ev['price']:.2f}"
    return f"[{t}] {json.dumps(ev, default=str)[:200]}"


def read_events(run_dir) -> list[dict]:
    with open(Path(run_dir) / "events.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]
