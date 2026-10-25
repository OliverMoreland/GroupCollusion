"""Repeated Bertrand game played by teams of LLM agents.

Each round:
  1. (representatives / elected_leader) elections, on round 1 and every `reelect_every` rounds
  2. communication: up to `subrounds` simultaneous messaging turns on the configured channels
  3. proposals: every member (except a leader) proposes a price (and, with memory=notes, rewrites its notes)
  4. aggregation by team structure: single | average | vote | leader | elected_leader | representatives
  5. market clears; every firm sees its price, quantity, profit (and competitors' prices)

With agent.memory=conversation each agent keeps one growing chat: every call appends a user turn with
only what is new (round, market results, messages, status changes) plus the task, and the agent's
answer is appended after it. Reasoning traces are logged but never sent back.
"""
from __future__ import annotations

import json
import random
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import prompts as P
from .channels import ChannelHub, Message, build_channels
from .config import Config
from .llm import make_client, parse_json
from .logger import RunLogger
from .market import Market


@dataclass
class AgentState:
    id: str
    team: str
    notes: dict = field(default_factory=lambda: {"plans": "", "insights": ""})
    proposals: dict = field(default_factory=dict)  # round -> proposed price
    # memory=conversation: the agent's chat and what it has already been told
    system: str | None = None
    turns: list = field(default_factory=list)  # [{"round": r, "messages": [user, assistant, ...]}]
    seen: dict = field(default_factory=lambda: {"round": 0, "market": 0, "msgs": 0, "status": ""})


class Simulation:
    def __init__(self, cfg: Config, client=None, run_dir: str | Path | None = None, echo: bool = False):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.client = client or make_client(cfg.llm, cfg.seed)
        self.market = Market(**{k: v for k, v in vars(cfg.market).items() if k != "max_wtp"})
        self.bench = self.market.benchmarks()

        letters = string.ascii_uppercase[: cfg.market.n_firms]
        self.teams = {t: [f"{t}{i + 1}" for i in range(cfg.team.size)] for t in letters}
        self.agents = {a: AgentState(a, t) for t, ms in self.teams.items() for a in ms}
        self.leaders = {}
        if cfg.team.structure == "leader":
            self.leaders = {t: (ms[0] if cfg.team.leader == "first" else self.rng.choice(ms))
                            for t, ms in self.teams.items()}
        self.reps: dict[str, list[str]] = {}
        self.hub = ChannelHub(build_channels(self.teams, cfg.communication.channels),
                              cfg.communication.max_chars)
        self.history: list[dict] = []

        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(run_dir or Path(cfg.output_dir) / f"{stamp}_{cfg.name}")
        self.log = RunLogger(self.run_dir, echo=echo)
        self.pool = ThreadPoolExecutor(max_workers=cfg.llm.concurrency or len(self.agents))
        # separate pool for per-team decisions, which themselves fan out ballots on self.pool
        self.team_pool = ThreadPoolExecutor(max_workers=cfg.market.n_firms)

    # ------------------------------------------------------------------ LLM
    @property
    def conversation(self):
        return self.cfg.agent.memory == "conversation"

    def ask(self, agent_id, phase, round_no, task, validate, hints=None, before=None):
        """One logged LLM call with one repair attempt. Returns (value | None, call_id)."""
        mc = self.cfg.agent
        a = self.agents[agent_id]
        if self.conversation:
            if a.system is None:
                a.system = self.system_prompt(agent_id)
            new, seen = self.update(agent_id, round_no, before)
            prior = [{"role": "system", "content": a.system}] + [m for t in a.turns for m in t["messages"]]
            user = new + "\n\n" + task if new else task
        else:
            prior = [{"role": "system", "content": self.system_prompt(agent_id)}]
            user = self.context(agent_id, round_no, before) + "\n\n" + task
        messages = prior + [{"role": "user", "content": user}]
        hints = {"phase": phase, "agent": agent_id, "round": round_no,
                 "price_range": (self.bench["nash_price"] * 0.9, self.bench["monopoly_price"] * 1.05),
                 **(hints or {})}
        call_id = self.log.new_call_id()
        for attempt in range(2):
            resp = self.client.complete(messages, model=mc.model, temperature=mc.temperature,
                                        max_tokens=mc.max_tokens, reasoning=mc.reasoning,
                                        reasoning_effort=mc.reasoning_effort, hints=hints)
            self.log.log("llm_call", call_id=call_id, attempt=attempt, agent=agent_id, phase=phase,
                         round=round_no, model=resp.model or mc.model, provider=resp.provider, finish_reason=resp.finish_reason, messages=messages,
                         response=resp.content, reasoning=resp.reasoning, usage=resp.usage,
                         latency_s=round(resp.latency_s, 3), error=resp.error, hints=hints)
            if resp.error:
                self.log.log("llm_error", call_id=call_id, agent=agent_id, phase=phase,
                             round=round_no, error=resp.error)
                return None, call_id
            obj = parse_json(resp.content)
            value, err = (None, "no JSON object found") if obj is None else validate(obj)
            if err is None:
                if self.conversation:
                    a.turns.append({"round": round_no, "messages": messages[len(prior):] + [
                        {"role": "assistant", "content": resp.content}]})
                    a.seen = seen
                    self.trim(a)
                return value, call_id
            self.log.log("parse_error", call_id=call_id, attempt=attempt, agent=agent_id, phase=phase,
                         round=round_no, error=err, response=resp.content[:2000])
            messages = messages + [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": f"Your response could not be used ({err}). "
                                            "Respond again with only the requested JSON object."}]
        return None, call_id

    def pmap(self, fn, items):
        return list(self.pool.map(fn, items))

    # -------------------------------------------------------------- prompts
    def system_prompt(self, agent_id):
        a = self.agents[agent_id]
        if self.conversation:  # built once and never changed: elections and closed channels go in the updates
            fixed = self.cfg.team.structure == "leader"
            return P.system_prompt(self.cfg, agent_id, a.team, self.teams[a.team],
                                   [c for c in self.hub.channels.values() if agent_id in c.members],
                                   leader=self.leaders.get(a.team) if fixed else None)
        return P.system_prompt(self.cfg, agent_id, a.team, self.teams[a.team],
                               self.hub.channels_for(agent_id), leader=self.leaders.get(a.team),
                               reps=self.reps.get(a.team))

    def context(self, agent_id, round_no, before=None):
        a, c = self.agents[agent_id], self.cfg
        hist = P.market_history(self.history, a.team, c.agent.history_window, c.agent.show_competitor_prices,
                                a.proposals if c.team.size > 1 else None)
        msgs_txt = None
        if self.hub.channels_for(agent_id):
            since = round_no - c.communication.history_rounds + 1
            msgs_txt = P.messages_block(self.hub.visible(agent_id, since, before), agent_id)
        return P.context(round_no, hist, a.notes, msgs_txt)

    def update(self, agent_id, round_no, before=None):
        """Conversation memory: text for everything the agent hasn't been told yet, and the new `seen` marks."""
        a, seen = self.agents[agent_id], dict(self.agents[agent_id].seen)
        closed = [c.id for c in self.hub.channels.values()
                  if agent_id in c.members and c not in self.hub.channels_for(agent_id)]
        status = P.status_line(a.team, leader=self.leaders.get(a.team) if self.cfg.team.structure == "elected_leader"
                               else None, reps=self.reps.get(a.team), closed=closed)
        new_round = round_no if round_no > seen["round"] else None
        new_status = status if status != seen["status"] else None
        hist_txt = None
        fresh = [h for h in self.history if h["round"] > seen["market"]]
        if fresh:
            hist_txt = P.market_history(fresh, a.team, len(fresh), self.cfg.agent.show_competitor_prices,
                                        a.proposals if self.cfg.team.size > 1 else None)
        elif seen["round"] == 0:
            hist_txt = P.market_history([], a.team, 0, False)
        msgs_txt = None
        if self.hub.channels_for(agent_id) or closed:
            vis = self.hub.visible(agent_id, 1, before)
            others = [m for m in vis[seen["msgs"]:] if m.sender != agent_id]
            if others:
                msgs_txt = P.messages_block(others, agent_id).replace("on your channels", "since your last turn", 1)
            seen["msgs"] = len(vis)
        seen.update(round=max(round_no, seen["round"]), market=self.history[-1]["round"] if self.history else 0,
                    status=status)
        return P.update(new_round, new_status, hist_txt, msgs_txt), seen

    def trim(self, a):
        """Past context_chars, drop the oldest turns (to about half the budget) and recap their market data.

        Trimming changes the prompt prefix, so it costs one cache miss; dropping to half keeps that rare."""
        size = lambda turns: len(a.system) + sum(len(m["content"]) for t in turns for m in t["messages"])
        limit = self.cfg.agent.context_chars
        if size(a.turns) <= limit or len(a.turns) < 2:
            return
        if a.turns[0].get("recap"):  # an old recap is rebuilt below for the new cut-off
            a.turns[0] = {**a.turns[0], "messages": [{**a.turns[0]["messages"][0], "content": a.turns[0]["recap"]},
                                                     *a.turns[0]["messages"][1:]]}
        turns = a.turns
        while len(turns) > 1:
            turns = turns[1:]
            upto = turns[0]["round"] - 1
            old = [h for h in self.history if h["round"] <= upto]
            recap = (P.RECAP.format(upto=upto) + "\n\n" + P.market_history(
                old, a.team, len(old), self.cfg.agent.show_competitor_prices,
                a.proposals if self.cfg.team.size > 1 else None)) if old else ""
            if size(turns) + len(recap) <= limit // 2:
                break
        first = turns[0]["messages"]
        a.turns = [{**turns[0], "recap": first[0]["content"],
                    "messages": [{**first[0], "content": recap + "\n\n" + first[0]["content"] if recap
                                  else first[0]["content"]}, *first[1:]]}, *turns[1:]]

    # ---------------------------------------------------------------- phases
    def phase_election(self, r):
        leader_vote = self.cfg.team.structure == "elected_leader"
        k = 1 if leader_vote else self.cfg.team.n_representatives
        task = P.LEADER_ELECTION_TASK if leader_vote else P.ELECTION_TASK

        def one(aid):
            cands = self.teams[self.agents[aid].team]

            def validate(o):
                reps = o.get("representatives")
                if not isinstance(reps, list):
                    return None, "'representatives' must be a list"
                reps = list(dict.fromkeys(str(x).strip() for x in reps))
                if len(reps) != k or any(x not in cands for x in reps):
                    return None, f"need exactly {k} distinct ids from {cands}"
                return reps, None

            reps, cid = self.ask(aid, "election", r, task.format(k=k, cands=", ".join(cands)), validate,
                                 {"candidates": cands, "k": k})
            return aid, reps, cid

        results = self.pmap(one, list(self.agents))
        for t, members in self.teams.items():
            tally = Counter(x for aid, reps, _ in results if self.agents[aid].team == t for x in (reps or []))
            order = sorted(members, key=lambda m: (-tally[m], self.rng.random()))
            winners = sorted(order[:k])
            if leader_vote:
                self.leaders[t] = winners[0]
            else:
                self.reps[t] = winners
            self.log.log("election", round=r, team=t, representatives=winners, tally=dict(tally),
                         ballots={aid: reps for aid, reps, _ in results if self.agents[aid].team == t},
                         call_ids={aid: cid for aid, _, cid in results if self.agents[aid].team == t})

    def decision_makers(self) -> set:
        if self.cfg.team.structure == "single":
            return set(self.agents)
        return set(self.leaders.values()) | {a for reps in self.reps.values() for a in reps}

    def phase_messages(self, r):
        cc = self.cfg.communication
        talkers = [a for a in self.agents if self.hub.channels_for(a)]
        done, last_turn = set(), {}
        for sub in range(1, cc.subrounds + 1):
            # an agent that said "done" is only asked again if someone else has written to it since
            active = [a for a in talkers if a not in done or any(
                m.sender != a and m.subround >= last_turn[a] for m in self.hub.visible(a, r))]
            if not active:
                break

            def one(aid):
                chans = [c.id for c in self.hub.channels_for(aid)]

                def validate(o):
                    msgs = o.get("messages", [])
                    if not isinstance(msgs, list):
                        return None, "'messages' must be a list"
                    out, seen = [], set()
                    for m in msgs:
                        if not isinstance(m, dict) or m.get("channel") not in chans or not str(m.get("text", "")).strip():
                            return None, f"each message needs a valid 'channel' (one of {chans}) and non-empty 'text'"
                        if m["channel"] not in seen:
                            seen.add(m["channel"])
                            out.append(m)
                    return {"messages": out, "done": o.get("done") is True}, None

                task = P.MESSAGE_TASK.format(sub=sub, n=cc.subrounds, channels=", ".join(chans))
                out, cid = self.ask(aid, "message", r, task, validate, {"channels": chans, "subround": sub},
                                    before=(r, sub))
                return aid, out or {"messages": [], "done": True}, cid

            # simultaneous: everyone writes, then all messages are delivered
            for aid, out, cid in self.pmap(one, active):
                last_turn[aid] = sub
                (done.add if out["done"] else done.discard)(aid)
                for m in out["messages"]:
                    posted = self.hub.post(Message(r, sub, m["channel"], aid, str(m["text"]).strip(), cid))
                    if posted:
                        ch = self.hub.channels[posted.channel]
                        self.log.log("message", round=r, subround=sub, channel=posted.channel, sender=aid,
                                     team=self.agents[aid].team, recipients=[x for x in ch.members if x != aid],
                                     kind=ch.kind, scope=ch.scope, text=posted.text, call_id=cid,
                                     n_chars=len(posted.text), n_words=len(posted.text.split()))

    def _price_validator(self):
        cap = self.cfg.agent.max_price * self.cfg.market.alpha

        def validate(o):
            try:
                p = float(str(o.get("price")).replace("$", "").strip())
            except (TypeError, ValueError):
                return None, "'price' must be a number"
            if not p == p or p <= 0:
                return None, "'price' must be a positive number"
            return {**o, "price": min(p, cap), "clipped": p > cap}, None
        return validate

    def propose_task(self):
        return P.PROPOSE_TASK_CONVERSATION if self.conversation else P.PROPOSE_TASK

    def notes_from(self, out):
        if self.conversation:  # the optional note stays in the chat; logged as "plans" for the judge and console
            return {"plans": str(out.get("note") or ""), "insights": ""}
        return {"plans": str(out.get("plans", "")), "insights": str(out.get("insights", ""))}

    def _propose_extra(self, leader_phase=False):
        s = self.cfg.team.structure
        if leader_phase or s in ("single", "average"):
            return ""
        if s in ("leader", "elected_leader"):
            return ', "advice": "<advice for your team leader>"'
        return ', "rationale": "<short justification shown to your teammates>"'

    def phase_proposals(self, r):
        proposers = [a for a in self.agents if a not in self.leaders.values()]
        verb = "propose a price for your firm" if self.cfg.team.size > 1 else "set your price"

        def one(aid):
            task = self.propose_task().format(verb=verb, extra=self._propose_extra())
            out, cid = self.ask(aid, "propose", r, task, self._price_validator())
            return aid, out, cid

        props = {}
        for aid, out, cid in self.pmap(one, proposers):
            a = self.agents[aid]
            if out:
                a.notes = self.notes_from(out)
                a.proposals[r] = out["price"]
            props[aid] = {"price": out["price"] if out else None,
                          "rationale": (out or {}).get("rationale") or (out or {}).get("advice"),
                          "clipped": (out or {}).get("clipped", False), "call_id": cid}
            self.log.log("proposal", round=r, agent=aid, team=a.team, **props[aid],
                         plans=a.notes["plans"], insights=a.notes["insights"], failed=out is None)
        return props

    def _fallback_price(self, team):
        if self.history:
            return self.history[-1]["firms"][team]["price"], "previous_price"
        return self.bench["nash_price"], "nash_price"  # only if round 1 fails entirely

    def _ballots(self, r, team, voters, valid):
        items = "\n".join(f"- {aid}: ${p['price']:.2f}" + (
            f" — {p['rationale']}" if self.cfg.team.ballot_shows_rationales and p.get("rationale") else "")
            for aid, p in valid.items())
        cands = list(valid)

        def validate(o):
            v = str(o.get("vote", "")).strip()
            return (v, None) if v in cands else (None, f"'vote' must be one of {cands}")

        def one(aid):
            v, cid = self.ask(aid, "ballot", r, P.BALLOT_TASK.format(items=items), validate, {"candidates": cands})
            return aid, v, cid
        return self.pmap(one, voters)

    def decide_team(self, r, team, props):
        s, members = self.cfg.team.structure, self.teams[team]
        valid = {a: props[a] for a in members if a in props and props[a]["price"] is not None}
        rec = {"round": r, "team": team, "structure": s,
               "proposals": {a: props.get(a, {}).get("price") for a in members}}
        if s in ("leader", "elected_leader"):
            leader = self.leaders[team]
            rec["leader"] = leader
            items = "\n".join(f"- {a}: ${p['price']:.2f} — {p.get('rationale') or '(no advice)'}"
                              for a, p in valid.items()) or "(no proposals received)"
            task = (P.LEADER_INPUT.format(items=items) + "\n\n"
                    + self.propose_task().format(verb="set your firm's price", extra=""))
            out, cid = self.ask(leader, "leader", r, task, self._price_validator())
            rec["leader_call_id"] = cid
            if out:
                self.agents[leader].notes = self.notes_from(out)
                self.agents[leader].proposals[r] = out["price"]
                rec["price"], rec["method"] = out["price"], "leader"
            else:
                rec["price"], rec["method"] = self._fallback_price(team)
            return rec
        if not valid:
            rec["price"], rec["method"] = self._fallback_price(team)
            return rec
        if s in ("single", "average") or len(valid) == 1:
            rec["price"] = sum(p["price"] for p in valid.values()) / len(valid)
            rec["method"] = "average" if s == "average" else "single"
            return rec
        voters = members if s == "vote" else self.reps[team]
        ballots = self._ballots(r, team, voters, valid)
        tally = Counter(v for _, v, _ in ballots if v)
        rec["ballots"] = {a: v for a, v, _ in ballots}
        rec["ballot_call_ids"] = {a: cid for a, _, cid in ballots}
        rec["voters"] = list(voters)
        if not tally:
            rec["price"] = sum(p["price"] for p in valid.values()) / len(valid)
            rec["method"] = "average_fallback"
            return rec
        top = max(tally.values())
        winners = sorted(a for a, n in tally.items() if n == top)
        win = self.rng.choice(winners)
        rec.update(price=valid[win]["price"], method="vote", winner=win, tally=dict(tally), tie=len(winners) > 1)
        return rec

    # ------------------------------------------------------------------ run
    def step(self, r):
        s = self.cfg.team.structure
        if s in ("representatives", "elected_leader"):
            k = self.cfg.team.reelect_every
            if r == 1 or (k and (r - 1) % k == 0):
                self.phase_election(r)
        self.hub.blocked = self.decision_makers()
        if self.hub.channels:
            self.phase_messages(r)
        props = self.phase_proposals(r)
        decisions = list(self.team_pool.map(lambda t: self.decide_team(r, t, props), list(self.teams)))
        for d in decisions:
            self.log.log("team_decision", **d)
        prices = [d["price"] for d in decisions]
        out = self.market.clear(prices)
        firms = {t: {"price": out["prices"][i], "quantity": out["quantities"][i], "profit": out["profits"][i]}
                 for i, t in enumerate(self.teams)}
        mean_p = sum(prices) / len(prices)
        rec = {"round": r, "firms": firms, "mean_price": mean_p,
               "collusion_index": (mean_p - self.bench["nash_price"]) /
                                  (self.bench["monopoly_price"] - self.bench["nash_price"])}
        self.history.append(rec)
        self.log.log("market", **rec)

    def run(self):
        from .metrics import write_outputs  # local import to avoid a cycle
        cfg = self.cfg
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
        self.log.log("run_start", config=cfg.to_dict(), benchmarks=self.bench, teams=self.teams,
                     leaders=self.leaders, channels={c.id: {"members": c.members, "kind": c.kind, "scope": c.scope, "leaderless": c.leaderless}
                                                     for c in self.hub.channels.values()})
        try:
            for r in range(1, cfg.rounds + 1):
                self.step(r)
            self.log.log("run_end", rounds=cfg.rounds)
        finally:
            self.pool.shutdown(wait=True)
            self.team_pool.shutdown(wait=True)
            self.log.close()
        summary = write_outputs(self.run_dir)
        return summary
