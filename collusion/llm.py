"""LLM clients: OpenRouter (OpenAI-compatible, with reasoning traces) and a deterministic mock."""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

MAX_TOKENS_CAP = 32000


@dataclass
class LLMResponse:
    content: str
    reasoning: str | None = None
    model: str | None = None
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    raw_id: str | None = None
    error: str | None = None
    finish_reason: str | None = None
    provider: str | None = None


def load_dotenv(*paths):
    """KEY=value lines from .env files (repo root, then cwd) into os.environ; real env vars win."""
    for path in paths or (Path(__file__).resolve().parents[1] / ".env", Path.cwd() / ".env"):
        if not Path(path).is_file():
            continue
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


class OpenRouterClient:
    """OpenAI-compatible chat client for OpenRouter, or for our vLLM server on Modal (provider=vllm)."""

    def __init__(self, cfg):
        load_dotenv()
        self.vllm = cfg.provider == "vllm"
        if self.vllm:
            self.base_url = os.environ.get(cfg.vllm_url_env, "").rstrip("/")
            key = os.environ.get(cfg.vllm_key_env)
            if not self.base_url:
                raise RuntimeError(f"No vLLM endpoint in ${cfg.vllm_url_env}; deploy modal_app/deepseek.py and put its URL in .env")
        else:
            self.base_url = cfg.base_url
            key = os.environ.get(cfg.api_key_env) or os.environ.get("OPENROUTER_API_KEY")
        key = "".join((key or "").split())  # pasted keys sometimes carry stray spaces/newlines
        if not key:
            raise RuntimeError(f"No API key in ${cfg.vllm_key_env if self.vllm else cfg.api_key_env}")
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": "GroupCollusion",
        })
        # our own server has no stuck hosts, but long reasoning traces legitimately take minutes
        self.timeout = max(cfg.timeout, 900) if self.vllm else cfg.timeout
        if self.vllm:
            self.wait_until_ready()

    def wait_until_ready(self, max_wait=45 * 60):
        """Block until the Modal server answers; the first request after an idle spell triggers a cold start."""
        t0 = time.time()
        while True:
            try:
                r = self.session.get(f"{self.base_url}/models", timeout=60)
                if r.status_code == 200:
                    return
                if r.status_code in (401, 403):
                    raise RuntimeError(f"vLLM endpoint rejected the key (HTTP {r.status_code}); check ${self.cfg.vllm_key_env}")
            except requests.RequestException:
                pass
            if time.time() - t0 > max_wait:
                raise RuntimeError(f"vLLM endpoint {self.base_url} not ready after {max_wait // 60} min")
            print(f"waiting for the model server to start ({int(time.time() - t0)} s)...", flush=True)
            time.sleep(15)

    def complete(self, messages, *, model, temperature=1.0, max_tokens=4000,
                 reasoning=True, reasoning_effort=None, hints=None) -> LLMResponse:
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.vllm:
            # DeepSeek V4's chat template thinks only when asked; "high" is its standard thinking mode
            body["chat_template_kwargs"] = {"thinking": bool(reasoning), "reasoning_effort": reasoning_effort or "high"}
        else:
            body["usage"] = {"include": True}  # returns cost in usage
            if reasoning:
                body["reasoning"] = {"effort": reasoning_effort} if reasoning_effort else {"enabled": True}
            if self.cfg.provider_routing:
                body["provider"] = self.cfg.provider_routing
        last_err = None
        for attempt in range(self.cfg.max_retries + 1):
            t0 = time.time()
            try:
                r = self.session.post(f"{self.base_url}/chat/completions", json=body,
                                      timeout=self.timeout)
                if r.status_code in (408, 429) or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                if r.status_code != 200:  # client errors are not retried
                    return LLMResponse(content="", error=f"HTTP {r.status_code}: {r.text[:500]}",
                                       latency_s=time.time() - t0)
                data = r.json()
                if "error" in data:
                    raise RuntimeError(f"provider error: {data['error']}")
                choice = data["choices"][0]
                msg = choice["message"]
                finish = choice.get("finish_reason")
                content = msg.get("content") or ""
                reasoning_text = (msg.get("reasoning") or msg.get("reasoning_content")
                                  or _join_reasoning(msg.get("reasoning_details")))
                if finish == "length":
                    # reasoning ate the budget, so the answer is missing or truncated: retry with more room
                    body["max_tokens"] = min(2 * body["max_tokens"], MAX_TOKENS_CAP)
                    if not content.strip() or attempt < self.cfg.max_retries:
                        raise RuntimeError(f"truncated output (finish_reason=length, {len(content)} chars)")
                if not content.strip():
                    raise RuntimeError(f"empty content (finish_reason={finish})")
                return LLMResponse(content=content, reasoning=reasoning_text, model=data.get("model"),
                                   usage=data.get("usage") or {}, latency_s=time.time() - t0,
                                   raw_id=data.get("id"), finish_reason=finish, provider=data.get("provider") or ("modal-vllm" if self.vllm else None))
            except (requests.RequestException, RuntimeError, ValueError, KeyError) as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(min(60, 2 ** attempt + random.random()))
        return LLMResponse(content="", error=f"failed after retries: {last_err}")


def _join_reasoning(details) -> str | None:
    if not details:
        return None
    parts = [d.get("text") or d.get("summary") or "" for d in details if isinstance(d, dict)]
    return "\n".join(p for p in parts if p) or None


class MockClient:
    """Offline stand-in that returns well-formed JSON for every phase.

    Uses `hints` passed by the caller (phase, channel ids, candidates, price scale)
    so the full pipeline can be exercised without network access or cost.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def complete(self, messages, *, model, hints=None, **_) -> LLMResponse:
        h = hints or {}
        # seeded per call from the hints, so results don't depend on thread scheduling
        rng = random.Random(f"{self.seed}:{json.dumps(h, sort_keys=True, default=str)}")
        phase = h.get("phase")
        lo, hi = h.get("price_range", (1.4, 2.0))
        if phase == "message":
            chans = h.get("channels", [])
            msgs = [{"channel": c, "text": f"mock note on {c}: thinking about ${rng.uniform(lo, hi):.2f}"}
                    for c in chans if rng.random() < 0.6]
            out = {"messages": msgs, "done": rng.random() < 0.3}
        elif phase in ("propose", "leader"):
            out = {"plans": "mock plan", "insights": "mock insight",
                   "price": round(rng.uniform(lo, hi), 2), "rationale": "mock rationale"}
        elif phase == "ballot":
            out = {"vote": rng.choice(h["candidates"]), "reason": "mock"}
        elif phase == "election":
            out = {"representatives": rng.sample(h["candidates"], h["k"])}
        elif phase == "judge":
            out = {"deception_score": rng.randint(0, 10), "is_deceptive": rng.random() < 0.3,
                   "explicit_collusion": rng.random() < 0.3, "explanation": "mock judgement"}
        else:
            out = {}
        text = json.dumps(out)
        if rng.random() < 0.1:  # exercise the lenient parser
            text = f"Sure.\n```json\n{text}\n```"
        return LLMResponse(content=text, reasoning=f"[mock reasoning for {phase}]", model="mock",
                           usage={"prompt_tokens": sum(len(m["content"]) for m in messages) // 4,
                                  "completion_tokens": len(text) // 4, "cost": 0.0})


def make_client(llm_cfg, seed: int = 0):
    if llm_cfg.provider == "mock":
        return MockClient(seed)
    if llm_cfg.provider in ("openrouter", "vllm"):
        return OpenRouterClient(llm_cfg)
    raise ValueError(f"unknown llm.provider {llm_cfg.provider!r}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str) -> dict | None:
    """Lenient JSON-object extraction: whole text, fenced block, or last balanced {...}."""
    if not text:
        return None
    candidates = [text.strip()] + _FENCE.findall(text)
    # balanced-brace scan, last object first (models often think, then answer)
    depth, start, spans = 0, None, []
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            elif ch != "\\":
                esc = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start:i + 1])
    candidates += reversed(spans)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None
