import json

import pytest

from collusion.__main__ import expand_sweep
from collusion.channels import build_channels
from collusion.config import CHANNEL_TYPES, STRUCTURES, load_config
from collusion.judge import judge_run
from collusion.llm import parse_json
from collusion.logger import read_events
from collusion.market import Market
from collusion.simulation import Simulation


def test_benchmarks_match_fish_et_al():
    b = Market().benchmarks()  # duopoly, alpha=1: Fish et al. report ~1.47 and ~1.93
    assert b["nash_price"] == pytest.approx(1.4729, abs=1e-3)
    assert b["monopoly_price"] == pytest.approx(1.9250, abs=1e-3)
    b10 = Market(alpha=10).benchmarks()
    assert b10["nash_price"] == pytest.approx(14.729, abs=1e-2)


def test_nash_is_a_best_response():
    m = Market(n_firms=3)
    pn = m.nash_price()
    base = m.profits([pn] * 3)[0]
    for dev in (-0.05, 0.05):
        assert m.profits([pn + dev, pn, pn])[0] < base


def test_channel_topologies():
    teams = {"A": ["A1", "A2", "A3"], "B": ["B1", "B2", "B3"]}
    assert len(build_channels(teams, ["team_dm"])) == 6
    assert set(build_channels(teams, ["team_open"])) == {"team:A", "team:B"}
    cross = build_channels(teams, ["cross_dm"])
    assert len(cross) == 9 and all(c.scope == "inter_team" for c in cross.values())
    assert len(build_channels(teams, ["all_dm"])) == 15
    assert list(build_channels(teams, ["global_open"])) == ["global"]
    assert build_channels(teams, []) == {}


def test_parse_json_lenient():
    assert parse_json('{"price": 1.8}') == {"price": 1.8}
    assert parse_json('thinking {not json}\n```json\n{"price": 2}\n```') == {"price": 2}
    assert parse_json('blah {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert parse_json("no json") is None


@pytest.mark.parametrize("structure", STRUCTURES)
@pytest.mark.parametrize("channels", [[], ["team_open"], ["all_dm"], ["cross_dm", "global_open"]])
def test_every_structure_and_channel_runs(tmp_path, structure, channels):
    size = 1 if structure == "single" else 3
    cfg = load_config(None, [f"team.structure={structure}", f"team.size={size}", "rounds=3",
                             "team.n_representatives=2", "llm.provider=mock",
                             f"communication.channels={json.dumps(channels)}"])
    s = Simulation(cfg, run_dir=tmp_path / "run").run()
    assert s["pricing"]["rounds"] == 3
    ev = read_events(tmp_path / "run")
    calls = [e for e in ev if e["type"] == "llm_call"]
    assert calls and all(c["reasoning"] for c in calls)  # CoT captured on every call
    assert all(e["price"] > 0 for e in ev if e["type"] == "team_decision")
    msgs = [e for e in ev if e["type"] == "message"]
    if not channels or (structure == "single" and channels == ["team_open"]):
        assert not msgs
    for m in msgs:
        assert m["sender"] not in m["recipients"]
    if structure == "leader":
        assert all(d["method"] in ("leader", "previous_price", "nash_price") for d in ev if d["type"] == "team_decision")
    if structure == "representatives":
        assert any(e["type"] == "election" for e in ev)
    for f in ("rounds.csv", "messages.csv", "summary.json", "transcript.md", "config.yaml"):
        assert (tmp_path / "run" / f).exists()


def test_judge_mock(tmp_path):
    cfg = load_config(None, ["team.structure=vote", "team.size=2", "rounds=2", "llm.provider=mock",
                             "communication.channels=[all_dm]"])
    Simulation(cfg, run_dir=tmp_path / "run").run()
    s = judge_run(tmp_path / "run")
    assert s["deception"]["n_judged"] == s["communication"]["n_messages"] > 0
    assert 0 <= s["deception"]["mean_score"] <= 10


def test_sweep_expansion_dedupes_single():
    runs = expand_sweep("configs/sweep.yaml")
    names = [n for n, _ in runs]
    assert len(names) == len(set(names))
    singles = [d for _, d in runs if d["team"]["structure"] == "single"]
    assert all(d["team"]["size"] == 1 for d in singles)
    assert all(c in CHANNEL_TYPES for _, d in runs for c in d["communication"]["channels"])


def test_sub_dm_excludes_decision_makers(tmp_path):
    cfg = load_config(None, ["team.structure=elected_leader", "team.size=3", "team.reelect_every=2", "rounds=5",
                             "communication.subrounds=3", "llm.provider=mock",
                             "communication.channels=[team_open, sub_dm]"])
    Simulation(cfg, run_dir=tmp_path / "run").run()
    ev = read_events(tmp_path / "run")
    elections = [e for e in ev if e["type"] == "election"]
    assert sorted({e["round"] for e in elections}) == [1, 3, 5]
    leader_at = lambda r, t: max((e for e in elections if e["team"] == t and e["round"] <= r), key=lambda e: e["round"])["representatives"][0]
    cross = [m for m in ev if m["type"] == "message" and m["scope"] == "inter_team"]
    assert cross
    for m in cross:
        assert all(a != leader_at(m["round"], a[0]) for a in [m["sender"], *m["recipients"]])
    assert all(d["method"] in ("leader", "previous_price", "nash_price") for d in ev if d["type"] == "team_decision")


def test_grid_sweep():
    runs = expand_sweep("configs/grid.yaml")
    assert len(runs) == 45
    assert not any("sub_dm" in d["communication"]["channels"] for _, d in runs if d["team"]["structure"] == "single")


def test_load_dotenv_keeps_real_env(tmp_path, monkeypatch):
    from collusion.llm import load_dotenv
    env = tmp_path / ".env"
    env.write_text("# key\nexport GC_TEST_A='a1'\nGC_TEST_B = b2\n")
    monkeypatch.delenv("GC_TEST_A", raising=False)
    monkeypatch.setenv("GC_TEST_B", "from-shell")
    load_dotenv(env)
    import os
    assert os.environ["GC_TEST_A"] == "a1" and os.environ["GC_TEST_B"] == "from-shell"
    monkeypatch.delenv("GC_TEST_A")


def test_console_editor_estimates_match_smoke_run():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "console"))
    from editor import _load, calls_per_round, problems
    c, _, _ = _load(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    assert calls_per_round(c)[:2] == (12, 12)  # 4 messages + 4 proposals + 4 ballots: the live smoke run made 36 calls in 3 rounds
    assert problems(c) == ([], [])
    c["team"]["structure"], c["communication"]["channels"] = "single", ["sub_dm"]
    assert problems(c)[0]


@pytest.mark.parametrize("structure", ["elected_leader", "representatives"])
def test_conversation_memory_is_append_only(tmp_path, structure):
    cfg = load_config(None, [f"team.structure={structure}", "team.size=3", "team.reelect_every=2", "rounds=4",
                             "communication.subrounds=2", "llm.provider=mock",
                             "communication.channels=[team_open, sub_dm]"])
    Simulation(cfg, run_dir=tmp_path / "run").run()
    calls = [e for e in read_events(tmp_path / "run") if e["type"] == "llm_call"]
    by_agent = {}
    for c in calls:
        by_agent.setdefault(c["agent"], []).append(c)
    for agent, cs in by_agent.items():
        for prev, cur in zip(cs, cs[1:]):
            n = len(prev["messages"])
            if prev["call_id"] == cur["call_id"]:  # repair attempt
                continue
            # every call starts with the previous call's prompt and answer, so the prefix stays cacheable
            assert cur["messages"][:n] == prev["messages"]
            assert cur["messages"][n] == {"role": "assistant", "content": prev["response"]}
        text = "".join(m["content"] for m in cs[-1]["messages"])
        assert "mock reasoning" not in text and "PLANS.txt" not in text
    first = by_agent["A1"][0]["messages"][1]["content"]
    assert "This is round 1." in first and "none yet" in first


def test_conversation_trims_long_chats(tmp_path):
    cfg = load_config(None, ["rounds=30", "llm.provider=mock", "agent.context_chars=6000"])
    sim = Simulation(cfg, run_dir=tmp_path / "run")
    sim.run()
    a = sim.agents["A1"]
    assert len(a.system) + sum(len(m["content"]) for t in a.turns for m in t["messages"]) <= 6000
    assert a.turns[0]["round"] > 1 and "were removed to save space" in a.turns[0]["messages"][0]["content"]


def test_notes_memory_still_stateless(tmp_path):
    cfg = load_config(None, ["rounds=3", "llm.provider=mock", "agent.memory=notes"])
    Simulation(cfg, run_dir=tmp_path / "run").run()
    calls = [e for e in read_events(tmp_path / "run") if e["type"] == "llm_call"]
    assert all(len(c["messages"]) == 2 for c in calls if c["attempt"] == 0)
    assert "PLANS.txt" in calls[-1]["messages"][1]["content"]


def test_rate_limits_are_waited_out_without_using_retries(monkeypatch):
    from collusion import llm
    from collusion.config import LLMConfig

    class R:
        def __init__(self, code, body):
            self.status_code, self._body, self.headers, self.text = code, body, {"retry-after": "0"}, ""

        def json(self):
            return self._body

    ok = {"choices": [{"message": {"content": '{"price": 2}'}, "finish_reason": "stop"}], "usage": {}}
    replies = [R(429, {})] * 8 + [R(200, ok)]
    monkeypatch.setenv("OPENROUTER_KEY", "k")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    c = llm.OpenRouterClient(LLMConfig(provider="openrouter", max_retries=1, concurrency=0))
    monkeypatch.setattr(c.session, "post", lambda *a, **k: replies.pop(0))
    resp = c.complete([{"role": "user", "content": "x"}], model="m")
    assert resp.error is None and resp.content == '{"price": 2}'


def test_luna_grid_is_uncapped():
    from collusion.__main__ import expand_sweep
    runs = expand_sweep("configs/grid_luna.yaml")
    assert len(runs) == 45
    data = runs[0][1]
    assert data["llm"]["concurrency"] == 0 and data["llm"]["provider"] == "openrouter"
    assert data["llm"]["provider_routing"] is None and data["agent"]["model"] == "openai/gpt-6-luna"


def test_uncapped_concurrency_runs(tmp_path):
    cfg = load_config(None, ["team.structure=vote", "team.size=3", "rounds=2", "llm.provider=mock", "llm.concurrency=0"])
    sim = Simulation(cfg, run_dir=tmp_path / "run")
    assert sim.pool._max_workers == len(sim.agents)
    assert sim.run()["pricing"]["rounds"] == 2
    judge_run(tmp_path / "run", provider="mock")


def test_openai_responses_client(monkeypatch):
    from collusion import llm
    from collusion.config import LLMConfig

    class R:
        def __init__(self, code, body):
            self.status_code, self._body, self.headers, self.text = code, body, {}, json.dumps(body)

        def json(self):
            return self._body

    ok = {"id": "resp_1", "model": "gpt-6-luna", "status": "completed", "output": [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Undercut slightly."}]},
        {"type": "message", "content": [{"type": "output_text", "text": '{"price": 1.79}'}]}],
        "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 800}, "output_tokens": 100,
                  "output_tokens_details": {"reasoning_tokens": 60}}}
    replies = [R(400, {"error": {"message": "Unsupported parameter: 'temperature'", "param": "temperature"}}), R(200, ok)]
    sent = []
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    c = llm.OpenRouterClient(LLMConfig(provider="openai"))

    def post(url, json=None, timeout=None):
        sent.append((url, dict(json)))
        return replies.pop(0)

    monkeypatch.setattr(c.session, "post", post)
    resp = c.complete([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], model="gpt-6-luna",
                      temperature=0.0)
    assert sent[0][0].endswith("/responses") and sent[0][1]["reasoning"] == {"effort": "medium", "summary": "detailed"}
    assert "temperature" in sent[0][1] and "temperature" not in sent[1][1] and "temperature" in c.unsupported
    assert resp.content == '{"price": 1.79}' and resp.reasoning == "Undercut slightly." and resp.error is None
    u = resp.usage
    assert u["prompt_tokens"] == 1000 and u["prompt_tokens_details"]["cached_tokens"] == 800
    assert u["completion_tokens_details"]["reasoning_tokens"] == 60
    assert abs(u["cost"] - (200 * 0.10 + 800 * 0.01 + 100 * 0.50) / 1e6) < 1e-12


def test_luna_teams_config():
    runs = dict(expand_sweep("configs/luna_teams.yaml"))
    assert len(runs) == 10
    single, vote = runs["luna_teams__structure=single__seed=0"], runs["luna_teams__structure=vote__seed=4"]
    assert single["communication"]["channels"] == [] and single["team"]["size"] == 1
    assert vote["communication"]["channels"] == ["team_dm"] and vote["team"]["size"] == 5
    assert vote["llm"]["provider"] == "openai" and vote["llm"]["concurrency"] == 0
