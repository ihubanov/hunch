"""Turns typed checks into constrained one-token calls and reads the answer off the logprobs.

Every check is one (or, for big picks, several) `structured_outputs` choice + `logprobs` call with
max_tokens=1 and temperature 0. Label probabilities are renormalised over the allowed labels. All calls
of a request run in parallel, limited by a backend-wide semaphore.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

import httpx

from . import prompts
from .config import ModelSpec, Settings

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
MAX_LEVELS = len(DIGITS)


class HunchError(Exception):
    """Returned to the client as {"error": {"code": code, "message": message}} with this HTTP status."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    backend_calls: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.backend_calls += other.backend_calls


def confidence(probs: list[float]) -> float:
    """1 - normalised Shannon entropy: 1.0 = all mass on one answer, 0.0 = uniform."""
    n = len(probs)
    if n <= 1:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, 1.0 - h / math.log(n))


def even_chunks(items: list, max_size: int) -> list[list]:
    n_groups = math.ceil(len(items) / max_size)
    base, extra = divmod(len(items), n_groups)
    out, i = [], 0
    for g in range(n_groups):
        size = base + (1 if g < extra else 0)
        out.append(items[i:i + size])
        i += size
    return out


class Engine:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.s = settings
        self.client = client
        self.sem = asyncio.Semaphore(settings.max_concurrency)

    @property
    def max_options(self) -> int:
        return self.s.group_size * min(self.s.max_top_logprobs, len(LETTERS))

    # ------------------------------------------------------------- backend call
    async def label_probs(self, spec: ModelSpec, msgs: list[dict], labels: str) -> tuple[dict[str, float], Usage]:
        body = {
            "model": spec.backend_model,
            "messages": msgs,
            # Use structured_outputs: some vLLM versions silently ignore the legacy `guided_choice` param.
            "structured_outputs": {"choice": list(labels)},
            "logprobs": True,
            "top_logprobs": min(self.s.max_top_logprobs, max(len(labels), 2)),
            "max_tokens": 1,
            "temperature": 0,
            **spec.extra_body,
        }
        headers = {"Authorization": f"Bearer {self.s.backend_api_key}"} if self.s.backend_api_key else {}
        last = "no attempt"
        for attempt in range(self.s.retries + 1):
            if attempt:
                await asyncio.sleep(min(0.5 * 2 ** (attempt - 1), 4.0))
            try:
                async with self.sem:
                    r = await self.client.post(f"{self.s.backend_url}/v1/chat/completions", json=body,
                                               headers=headers, timeout=self.s.timeout_s)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = f"{type(e).__name__}: {e}"
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"backend HTTP {r.status_code}"
                if r.status_code == 429:
                    try:
                        await asyncio.sleep(min(float(r.headers.get("retry-after", "1")), 10.0))
                    except ValueError:
                        await asyncio.sleep(1.0)
                continue
            if r.status_code >= 400:
                text = r.text[:400]
                if any(k in text.lower() for k in ("context length", "maximum context", "too long")):
                    raise HunchError(413, "context_too_long", "context plus question exceed the backend model's context window")
                raise HunchError(502, "backend_error", f"backend HTTP {r.status_code}: {text}")
            return self._parse(r.json(), labels)
        raise HunchError(503, "backend_unavailable", last)

    @staticmethod
    def _parse(data: dict, labels: str) -> tuple[dict[str, float], Usage]:
        try:
            tok = data["choices"][0]["logprobs"]["content"][0]
        except (KeyError, IndexError, TypeError):
            raise HunchError(502, "backend_error", "backend returned no logprobs")
        chosen = str(tok.get("token", "")).strip()
        if chosen not in labels:
            # The constraint did not reach the model: fail loudly, never derive a probability from an unrelated token.
            raise HunchError(502, "backend_error",
                             f"backend answered {chosen!r}, not one of {list(labels)}; it does not enforce structured_outputs")
        probs = {lab: 0.0 for lab in labels}
        for t in tok.get("top_logprobs") or []:
            s = str(t.get("token", "")).strip()
            if s in probs and t.get("logprob") is not None:
                probs[s] += math.exp(t["logprob"])
        if probs[chosen] == 0.0:
            probs[chosen] = math.exp(tok.get("logprob", 0.0))
        total = sum(probs.values())
        u = data.get("usage") or {}
        return {k: v / total for k, v in probs.items()}, Usage(u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0, 1)

    # --------------------------------------------------------------- check kinds
    async def yesno(self, spec, ctx, c) -> tuple[dict, Usage]:
        orders = [False, True] if spec.debias else [False]
        results = await asyncio.gather(*[
            self.label_probs(spec, prompts.messages(ctx, prompts.yesno(c.get("question"), c.get("yes_if"), c.get("no_if"), n_first)), "YN")
            for n_first in orders])
        usage = Usage()
        for _, u in results:
            usage.add(u)
        return {"kind": "yesno", "p_yes": round(sum(p["Y"] for p, _ in results) / len(results), 4)}, usage

    async def pick(self, spec, ctx, c) -> tuple[dict, Usage]:
        options = list(c["options"].items())
        keys = [k for k, _ in options]
        usage = Usage()
        if len(options) == 1:
            dist = {keys[0]: 1.0}
        elif len(options) <= self.s.group_size:
            labels = LETTERS[:len(options)]
            orders = [options, options[::-1]] if spec.debias else [options]
            results = await asyncio.gather(*[
                self.label_probs(spec, prompts.messages(ctx, prompts.pick(c.get("question"), opts, labels)), labels) for opts in orders])
            dist = {k: 0.0 for k in keys}
            for opts, (p, u) in zip(orders, results):
                usage.add(u)
                for lab, (key, _) in zip(labels, opts):
                    dist[key] += p[lab] / len(orders)
        else:
            dist, usage = await self._grouped_pick(spec, ctx, c.get("question"), options)
        probs = [dist[k] for k in keys]
        best = keys[max(range(len(keys)), key=probs.__getitem__)]
        return {"kind": "pick", "pick": best, "probs": {k: round(dist[k], 4) for k in keys},
                "confidence": round(confidence(probs), 4)}, usage

    async def _grouped_pick(self, spec, ctx, question, options):
        """P(option) = P(group) x P(option | group); all calls run in parallel."""
        groups = even_chunks(options, self.s.group_size)
        glabels = LETTERS[:len(groups)]
        calls = [self.label_probs(spec, prompts.messages(ctx, prompts.pick_groups(question, groups, glabels)), glabels)]
        calls += [self.label_probs(spec, prompts.messages(ctx, prompts.pick(question, g, LETTERS[:len(g)])), LETTERS[:len(g)]) for g in groups]
        results = await asyncio.gather(*calls)
        usage = Usage()
        for _, u in results:
            usage.add(u)
        pg = results[0][0]
        dist = {}
        for glab, g, (pw, _) in zip(glabels, groups, results[1:]):
            for lab, (key, _) in zip(LETTERS, g):
                dist[key] = pg[glab] * pw[lab]
        total = sum(dist.values()) or 1.0
        return {k: v / total for k, v in dist.items()}, usage

    async def scale(self, spec, ctx, c) -> tuple[dict, Usage]:
        levels = list(c["levels"])
        if len(levels) == 1:
            return {"kind": "scale", "value": 0.0, "probs": [1.0], "confidence": 1.0}, Usage()
        labels = DIGITS[:len(levels)]
        p, usage = await self.label_probs(spec, prompts.messages(ctx, prompts.scale(c.get("question"), levels)), labels)
        probs = [p[lab] for lab in labels]
        return {"kind": "scale", "value": round(sum(i * x for i, x in enumerate(probs)), 4),
                "probs": [round(x, 4) for x in probs], "confidence": round(confidence(probs), 4)}, usage

    # ------------------------------------------------------------------ request
    async def judge(self, spec: ModelSpec, context: Any, checks: dict[str, dict]) -> tuple[dict, Usage]:
        ctx = prompts.context_block(context)
        handlers = {"yesno": self.yesno, "pick": self.pick, "scale": self.scale}
        tasks = {cid: asyncio.ensure_future(handlers[c["kind"]](spec, ctx, c)) for cid, c in checks.items()}
        try:
            await asyncio.gather(*tasks.values())
        except BaseException:
            for t in tasks.values():
                t.cancel()
            raise
        usage, results = Usage(), {}
        for cid, t in tasks.items():
            result, used = t.result()
            results[cid] = result
            usage.add(used)
        return results, usage


def validate_checks(checks: dict[str, Any], max_options: int) -> dict[str, dict]:
    if not checks:
        raise HunchError(400, "invalid_request", "`checks` must contain at least one check")
    out = {}
    for cid, c in checks.items():
        if not isinstance(c, dict):
            raise HunchError(400, "invalid_request", f"check {cid!r} must be an object")
        kind = c.get("kind")
        if kind == "yesno":
            if c.get("question") in (None, "") and c.get("yes_if") in (None, "") and c.get("no_if") in (None, ""):
                raise HunchError(400, "invalid_request", f"check {cid!r} needs a question (or yes_if / no_if)")
        elif kind == "pick":
            opts = c.get("options")
            if not isinstance(opts, dict) or not opts:
                raise HunchError(400, "invalid_request", f"check {cid!r}: `options` must be a non-empty object of key -> description|null")
            if len(opts) > max_options:
                raise HunchError(400, "too_many_options", f"check {cid!r} has {len(opts)} options; the maximum is {max_options}")
        elif kind == "scale":
            levels = c.get("levels")
            if not isinstance(levels, list) or not levels:
                raise HunchError(400, "invalid_request", f"check {cid!r}: `levels` must be a non-empty list, lowest first")
            if len(levels) > MAX_LEVELS:
                raise HunchError(400, "too_many_levels", f"check {cid!r} has {len(levels)} levels; the maximum is {MAX_LEVELS}")
        else:
            raise HunchError(400, "invalid_request", f"check {cid!r}: `kind` must be one of yesno, pick, scale")
        out[cid] = c
    return out
