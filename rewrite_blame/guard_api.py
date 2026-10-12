"""Decisions-API guard: OpenAI's /v1/decisions endpoint (gpt-6-luna) asked the same typed question
as the other guards, zero-shot or with our labelled rows pasted in as context. External baseline
only: it needs the network, an API key in OPENAI_API_KEY, and hundreds of milliseconds per call.
"""

from __future__ import annotations

import json
import os
import urllib.request

from .guard_encoder import QUESTION, serialize

ENDPOINT = "https://api.openai.com/v1/decisions"


def load_api_key() -> str | None:
    """OPENAI_API_KEY from the environment, else the owner-only file ~/.openai_key (never in the repo)."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    p = os.path.expanduser("~/.openai_key")
    if os.path.exists(p):
        return open(p).read().strip() or None
    return None


def build_request(row: dict, family: str, context_rows: list[dict] | None = None, model: str = "gpt-6-luna") -> dict:
    """Request body: a predicate question (probability that the firing is harmful) about the state.
    ``context_rows`` are serialised as labelled examples so the model can condition on measurements."""
    state = serialize(row, family)
    if context_rows:
        examples = "\n".join(f"- {serialize(r, family)} -> {'slower' if r['label'] == 'harmful' else 'not slower'} ({(r['ratio']-1)*100:+.1f}%)"
                             for r in context_rows)
        state = "Measured examples on these machines:\n" + examples + "\n\nNow decide for: " + state
    return {"model": model, "input": state,
            "questions": [{"type": "predicate", "name": "harmful", "instructions": QUESTION[family]}]}


def ask(body: dict, api_key: str | None = None, timeout: float = 30.0) -> float:
    """Return the probability the API assigns to the predicate. Raises if no key or on HTTP errors."""
    key = api_key or load_api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set and ~/.openai_key does not exist")
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode())
    return parse_probability(out)


def parse_probability(out: dict) -> float | None:
    """Tolerant parsing of the decisions response: the first probability-like field, or None when
    the model returned a ``refusal`` answer (a documented answer type)."""
    for key in ("decisions", "answers", "results", "output"):
        items = out.get(key)
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    if it.get("type") == "refusal":
                        return None
                    for pk in ("probability", "p", "value", "score"):
                        if isinstance(it.get(pk), (int, float)):
                            return float(it[pk])
    if isinstance(out.get("probability"), (int, float)):
        return float(out["probability"])
    raise ValueError(f"no probability in response: {json.dumps(out)[:300]}")


_CACHE_PATH = os.path.join("results", "api_cache.json")
_cache: dict | None = None


def _cache_get(body: dict):
    global _cache
    import hashlib
    if _cache is None:
        try:
            _cache = json.load(open(_CACHE_PATH))
        except Exception:
            _cache = {}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(), _cache


def ask_cached(body: dict, api_key: str | None = None, timeout: float = 30.0):
    """``ask`` with an on-disk cache so re-runs are free and deterministic."""
    k, cache = _cache_get(body)
    if k in cache:
        return cache[k]
    p = ask(body, api_key, timeout)
    cache[k] = p
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    json.dump(cache, open(_CACHE_PATH, "w"))
    return p


def evaluate_api(rows: list[dict], family: str, with_context: bool = False, n_context: int = 40, threshold: float = 0.5,
                 ask_fn=None, log=print) -> dict:
    """Leave-one-out over configuration slices is unnecessary for a zero-shot model; with context we
    hold out the row's own slice from the examples."""
    from .guard_models import _groups
    ask_fn = ask_fn or ask_cached
    groups = _groups(rows, family)
    y = [1 if r["label"] == "harmful" else 0 for r in rows]
    pred, probs = [], []
    refusals = 0
    for i, r in enumerate(rows):
        ctx = None
        if with_context:
            ctx = [x for x, g in zip(rows, groups) if g != groups[i]][:n_context]
        p = ask_fn(build_request(r, family, ctx))
        if p is None:                       # refusal: no decision -> Inductor's default (fire)
            refusals += 1
            p = 0.0
        probs.append(p); pred.append(int(p >= threshold))
        if (i + 1) % 10 == 0:
            log(f"[api] {i+1}/{len(rows)}")
    n = len(rows)
    harmful = [i for i, t in enumerate(y) if t == 1]
    nonharm = [i for i, t in enumerate(y) if t == 0]
    cost = sum(abs(r["ratio"] - 1) * 100 for r, p, t in zip(rows, pred, y) if (t == 1 and p == 0) or (r["label"] == "helpful" and p == 1)) / n
    default_cost = sum(abs(r["ratio"] - 1) * 100 for r, t in zip(rows, y) if t == 1) / n
    return {"family": family, "model": "decisions-api" + ("+context" if with_context else ":zero-shot"), "n": n, "harmful": len(harmful),
            "folds": 1, "accuracy": sum(int(p == t) for p, t in zip(pred, y)) / n, "default_accuracy": len(nonharm) / n,
            "recall": (sum(pred[i] for i in harmful) / len(harmful)) if harmful else 1.0,
            "false_alarm_rate": (sum(pred[i] for i in nonharm) / len(nonharm)) if nonharm else 0.0,
            "cost_weighted": cost, "default_cost_weighted": default_cost, "per_fold": [], "probs": probs, "refusals": refusals}
