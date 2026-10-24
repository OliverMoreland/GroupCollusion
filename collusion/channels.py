"""Communication topologies and message store.

Agents are named `<team letter><member number>` (A1, A2, B1, ...). Channel ids:
    global          one open channel shared by every agent
    team:A          open channel for the members of team A
    dm:A1|B2        private 1-on-1 channel (ids sorted)

Channel *types* compose; a config lists any subset of:
    team_dm      1-on-1 between teammates
    team_open    one open channel per team
    cross_dm     1-on-1 between agents of different teams (not within a team)
    all_dm       1-on-1 between every pair (= team_dm + cross_dm)
    global_open  one channel shared by all agents of all teams
"""
from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field


@dataclass
class Channel:
    id: str
    members: tuple
    kind: str  # "dm" | "open"
    scope: str  # "intra_team" | "inter_team"


@dataclass
class Message:
    round: int
    subround: int
    channel: str
    sender: str
    text: str
    call_id: str | None = None


def team_of(agent_id: str) -> str:
    return agent_id.rstrip("0123456789")


def build_channels(teams: dict[str, list[str]], types: list[str]) -> dict[str, Channel]:
    chans: dict[str, Channel] = {}
    everyone = [a for members in teams.values() for a in members]
    types = set(types)
    if "all_dm" in types:
        types |= {"team_dm", "cross_dm"}

    def dm(x, y):
        a, b = sorted((x, y))
        scope = "intra_team" if team_of(a) == team_of(b) else "inter_team"
        chans[f"dm:{a}|{b}"] = Channel(f"dm:{a}|{b}", (a, b), "dm", scope)

    for x, y in itertools.combinations(everyone, 2):
        same = team_of(x) == team_of(y)
        if (same and "team_dm" in types) or (not same and "cross_dm" in types):
            dm(x, y)
    if "team_open" in types:
        for t, members in teams.items():
            if len(members) > 1:
                chans[f"team:{t}"] = Channel(f"team:{t}", tuple(members), "open", "intra_team")
    if "global_open" in types:
        chans["global"] = Channel("global", tuple(everyone), "open", "inter_team")
    return chans


class ChannelHub:
    def __init__(self, channels: dict[str, Channel], max_chars: int = 800):
        self.channels = channels
        self.max_chars = max_chars
        self.messages: list[Message] = []
        self._lock = threading.Lock()

    def channels_for(self, agent_id: str) -> list[Channel]:
        return [c for c in self.channels.values() if agent_id in c.members]

    def post(self, msg: Message) -> Message | None:
        ch = self.channels.get(msg.channel)
        if ch is None or msg.sender not in ch.members:
            return None
        msg.text = msg.text[: self.max_chars]
        with self._lock:
            self.messages.append(msg)
        return msg

    def visible(self, agent_id: str, since_round: int, before: tuple | None = None) -> list[Message]:
        """Messages in the agent's channels from `since_round` on, strictly before (round, subround)."""
        mine = {c.id for c in self.channels_for(agent_id)}
        out = [m for m in self.messages if m.channel in mine and m.round >= since_round]
        if before is not None:
            out = [m for m in out if (m.round, m.subround) < before]
        return out
