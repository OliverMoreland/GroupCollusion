"""Experiment configuration: nested dataclasses loaded from YAML, with dotted overrides."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

import yaml

STRUCTURES = ("single", "average", "vote", "leader", "elected_leader", "representatives")
CHANNEL_TYPES = ("team_dm", "team_open", "cross_dm", "sub_dm", "all_dm", "global_open")


@dataclass
class MarketConfig:
    n_firms: int = 2
    alpha: float = 1.0
    beta: float = 100.0
    a: float = 2.0
    a0: float = 0.0
    mu: float = 0.25
    cost: float = 1.0
    # Optional "no customer would pay more than $X" line in the prompt (Fish et al.
    # include a willingness-to-pay cap; set in units of alpha, null to omit).
    max_wtp: float | None = None


@dataclass
class TeamConfig:
    size: int = 1
    structure: str = "single"  # one of STRUCTURES
    leader: str = "first"  # "first" | "random"  (structure=leader)
    n_representatives: int = 1  # structure=representatives
    reelect_every: int = 0  # representatives / elected_leader: 0 = elect once at round 1; k = re-elect every k rounds
    ballot_shows_rationales: bool = True  # voters see the proposers' rationales


@dataclass
class CommConfig:
    # Composable list of channel types (see CHANNEL_TYPES); [] = no communication.
    channels: list = field(default_factory=list)
    subrounds: int = 1  # max messaging turns per agent per pricing round (agents can stop early with "done")
    max_chars: int = 800  # messages are truncated to this length
    history_rounds: int = 3  # how many past rounds of messages an agent sees


@dataclass
class AgentConfig:
    model: str = "deepseek/deepseek-v4-flash"
    reasoning: bool = True  # request (and log) chain-of-thought from the provider
    reasoning_effort: str | None = None  # e.g. "low" | "medium" | "high"; None = provider default
    temperature: float = 1.0
    max_tokens: int = 8000
    prompt_prefix: str = "P1"  # Fish et al. prefixes: P1 | P2
    history_window: int = 100  # rounds of market history shown
    show_competitor_prices: bool = True
    max_price: float = 10.0  # proposals are clipped to [0, max_price * alpha]


@dataclass
class JudgeConfig:
    model: str = "deepseek/deepseek-v4-flash"
    reasoning: bool = True
    reasoning_effort: str | None = None
    temperature: float = 0.0
    max_tokens: int = 8000


@dataclass
class LLMConfig:
    # "vllm" = our self-hosted DeepSeek on Modal (modal_app/deepseek.py), the default since 2026-10-24;
    # "openrouter" = OpenRouter's hosted models; "mock" = offline stand-in
    provider: str = "vllm"
    vllm_url_env: str = "MODAL_LLM_URL"  # e.g. https://<workspace>--groupcollusion-deepseek-serve.modal.run/v1
    vllm_key_env: str = "MODAL_LLM_KEY"
    api_key_env: str = "OPENROUTER_KEY"  # openrouter only; falls back to OPENROUTER_API_KEY
    base_url: str = "https://openrouter.ai/api/v1"  # openrouter only
    concurrency: int = 16
    timeout: float = 90.0  # most calls take < 30 s; a stuck call holds up its whole phase until retried
    max_retries: int = 5
    # OpenRouter provider preferences (openrouter only), sent as-is. Pinned by default: OpenRouter's own routing mixes hosts of very
    # different speed (and possibly quantization); on 2026-10-24 Baidu gave p50 8.5 s / max 18 s per call against
    # 12-38 s / 46-349 s unpinned. Without fallbacks, calls fail if both hosts are down. null = OpenRouter default.
    provider_routing: dict | None = field(default_factory=lambda: {"only": ["baidu", "gmicloud"], "allow_fallbacks": False})


@dataclass
class Config:
    name: str = "run"
    seed: int = 0
    rounds: int = 50
    output_dir: str = "runs"
    market: MarketConfig = field(default_factory=MarketConfig)
    team: TeamConfig = field(default_factory=TeamConfig)
    communication: CommConfig = field(default_factory=CommConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> "Config":
        t, c = self.team, self.communication
        if t.structure not in STRUCTURES:
            raise ValueError(f"team.structure must be one of {STRUCTURES}, got {t.structure!r}")
        if t.structure == "single" and t.size != 1:
            raise ValueError("team.structure=single requires team.size=1")
        if t.structure != "single" and t.size < 2:
            raise ValueError(f"team.structure={t.structure} needs team.size >= 2")
        if t.structure == "representatives" and not 1 <= t.n_representatives <= t.size:
            raise ValueError("need 1 <= team.n_representatives <= team.size")
        bad = [ch for ch in c.channels if ch not in CHANNEL_TYPES]
        if bad:
            raise ValueError(f"unknown channel types {bad}; choose from {CHANNEL_TYPES}")
        if self.market.n_firms < 2:
            raise ValueError("market.n_firms must be >= 2")
        return self


def _build(cls, data: dict):
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    for k, v in (data or {}).items():
        if k not in known:
            raise KeyError(f"unknown config key {cls.__name__}.{k}")
        default = known[k].default_factory() if callable(known[k].default_factory) else None
        kwargs[k] = _build(type(default), v) if is_dataclass(default) else v
    return cls(**kwargs)


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(data: dict, overrides: list[str] | dict | None) -> dict:
    """Apply `a.b.c=value` strings (value parsed as YAML) or a {dotted: value} dict."""
    if not overrides:
        return data
    items = overrides.items() if isinstance(overrides, dict) else (
        (o.split("=", 1)[0], yaml.safe_load(o.split("=", 1)[1])) for o in overrides)
    data = copy.deepcopy(data)
    for key, val in items:
        node = data
        *path, last = key.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[last] = val
    return data


def load_config(path: str | None = None, overrides=None) -> Config:
    data: dict[str, Any] = {}
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data.pop("sweep", None)
    return _build(Config, apply_overrides(data, overrides)).validate()


def config_from_dict(data: dict) -> Config:
    return _build(Config, data).validate()
