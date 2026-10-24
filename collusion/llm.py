"""LLM clients: OpenRouter (OpenAI-compatible, with reasoning traces) and a deterministic mock."""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field

import requests


@dataclass
class LLMResponse:
    content: str
    reasoning: str | None = None
    model: str | None = None
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    raw_id: str | None = None
    error: str | None = None


class OpenRouterClient:
    def __init__(self, cfg):
        key = os.environ.get(cfg.api_key_env) or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(f"No API key in ${cfg.api_key_env} or $OPENROUTER_API_KEY")
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": "GroupCollusion",
        })

    def complete(self, messages, *, model, temperature=1.0, max_tokens=4000,
                 reasoning=True, reasoning_effort=None, hints=None) -> LLMResponse:
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},  # returns cost in usage
        }
        if reasoning:
            body["reasoning"] = {"effort": reasoning_effort} if reasoning_effort else {"enabled": True}
        last_err = None
        for attempt in range(self.cfg.max_retries + 1):
            t0 = time.time()
            try:
                r = self.session.post(f"{self.cfg.base_url}/chat/completions", json=body,
                                      timeout=self.cfg.timeout)
                if r.status_code in (408, 429) or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                if r.status_code != 200:  # client errors are not retried
                    return LLMResponse(content="", error=f"HTTP {r.status_code}: {r.text[:500]}",
                                       latency_s=time.time() - t0)
                data = r.json()
                if "error" in data:
                    raise RuntimeError(f"provider error: {data['error']}")
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning_text = msg.get("reasoning") or _join_reasoning(msg.get("reasoning_details"))
                if not content.strip():
                    raise RuntimeError(f"empty content (finish_reason={data['choices'][0].get('finish_reason')})")
                return LLMResponse(content=content, reasoning=reasoning_text, model=data.get("model"),
                                   usage=data.get("usage") or {}, latency_s=time.time() - t0,
                                   raw_id=data.get("id"))
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
    if llm_cfg.provider == "openrouter":
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
