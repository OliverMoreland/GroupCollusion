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
