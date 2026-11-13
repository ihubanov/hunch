"""Turns typed checks into constrained one-token calls and reads the answer off the logprobs.

Every check is one (or, for big picks, several) `structured_outputs` choice + `logprobs` call with
max_tokens=1 and temperature 0. Label probabilities are renormalised over the allowed labels. All calls
of a request run in parallel, limited by a backend-wide semaphore.

Backends that return no token logprobs (hosted APIs) run in mode "sample": the same prompt is asked
`samples` times at temperature > 0 and the probability is the share of votes. Coarser (1/samples
steps) and samples-times the input tokens, but it works on any OpenAI-compatible chat endpoint.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, replace
from typing import Any

import httpx

from . import hosts, prompts
from .config import ModelSpec, Settings

MODES = ("one_token", "deliberate", "auto", "sample")
PROBE_MAX_TOKENS = 16   # enough to tell an answer from the start of a scratchpad
# Values of `reasoning_effort` that mean "don't think". Dropped in deliberate mode; real effort levels
# (low / high / max) are passed through, and "low" is often the best setting there — it can cut the
# thinking to a few tokens without losing accuracy.
THINKING_OFF_EFFORTS = hosts.THINKING_OFF_EFFORTS
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
MAX_LEVELS = len(DIGITS)
MAX_SORT_ITEMS = 100
MAX_TAGS = 50


def read_label(text: str, labels: str) -> str | None:
    """The label a free-text reply names (mode "sample"): its first word, or else its last line.

    Some hosted models work the answer out in plain text and give the label at the end
    ("... average 78, at least 75.\n\nY"), so a reply whose first word is not a label is read from
    its last line, exactly as deliberate mode reads the verdict that follows the thinking."""
    def head(t: str) -> str | None:
        t = t.strip().lstrip("*_`\"' (").upper()
        if not t:
            return None
        if set(labels) == {"Y", "N"}:
            if t.startswith("YES"):
                return "Y"
            if t.startswith("NO"):
                return "N"
        # a lone label, not the first letter of a word or sentence ("Scores: ..." is not S, "A customer" not A)
        if t[0] in labels and (len(t) == 1 or t[1] in ").]:*_`'\"-"):
            return t[0]
        return None
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    return head(lines[0]) or (head(lines[-1]) if len(lines) > 1 else None)


def effective_extra_body(spec: ModelSpec, deliberate: bool) -> dict:
    """spec.extra_body as sent. In deliberate mode the fields that switch thinking OFF are dropped;
    real effort levels (low / high / max) are kept, since they control how long the model thinks."""
    extra = dict(spec.extra_body)
    if not deliberate:
        return extra
    if str(extra.get("reasoning_effort", "")).lower() in THINKING_OFF_EFFORTS:
        extra.pop("reasoning_effort")
    kwargs = extra.get("chat_template_kwargs")
    # Qwen-style templates read `enable_thinking`, DeepSeek-style ones read `thinking`
    if isinstance(kwargs, dict) and any(kwargs.get(k) is False for k in ("enable_thinking", "thinking")):
        kwargs = {k: v for k, v in kwargs.items() if not (k in ("enable_thinking", "thinking") and v is False)}
        if kwargs:
            extra["chat_template_kwargs"] = kwargs
        else:
            extra.pop("chat_template_kwargs")
    return extra


# How vLLM (and some gateways) refuse image input on a text-only model, e.g. "At most 0 image(s) may be provided".
IMAGE_REFUSALS = ("at most 0 image", "not a multimodal", "does not support image", "image input is not supported",
                  "multimodal inputs are not supported", "images are not supported")


def _has_images(body: dict) -> bool:
    return any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
               for m in body.get("messages", []))


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
        self._detected: dict[str, str] = {}   # model name -> resolved mode, for mode="auto"
        # backend model -> request fields the backend refused by name (see hosts.rejected_field). Learnt on the
        # first refusal and not sent again by this engine.
        self.dropped: dict[str, set[str]] = {}
        self.calls = 0                          # backend POSTs made, retries included
        self.streaming: set[str] = set()        # backend models that only answer streamed calls
        self.min_tokens: dict[str, int] = {}    # backend model -> smallest max_tokens it accepts, when above 1

    def unconstrained(self, spec: ModelSpec) -> bool:
        """Answers are read from top_logprobs without a constraint: configured so, or the backend refused it."""
        return not spec.constrained or "structured_outputs" in self.dropped.get(spec.backend_model, ())

    async def send(self, spec: ModelSpec, body: dict, retry_if=None) -> dict:
        """POST one chat completion and return its JSON. Every backend call goes through here.

        - host quirks are applied (hosts.adapt) and fields this backend refused before are left out;
        - a 400/422 that names a droppable field drops it and resends at once (not counted as a retry);
        - 429 / 5xx / timeouts are retried with backoff; `retry_if(data)` returning a reason retries a 200 too;
        - other errors map to HunchError codes.
        A call that may think (max_tokens above PROBE_MAX_TOKENS) gets slow_timeout_s.
        """
        dropped = self.dropped.setdefault(spec.backend_model, set())
        url = self.s.completions_url
        long_call = (body.get("max_tokens") or 0) > PROBE_MAX_TOKENS
        timeout = max(self.s.timeout_s, self.s.slow_timeout_s) if long_call else self.s.timeout_s
        body = {k: v for k, v in hosts.adapt(url, body).items() if k not in dropped}
        if spec.backend_model in self.streaming:
            body.update(stream=True, stream_options={"include_usage": True})
        tok_key = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
        floor = self.min_tokens.get(spec.backend_model)
        if floor and (body.get(tok_key) or 0) < floor:
            body[tok_key] = floor
        headers = self.s.backend_headers
        last = "no attempt"
        attempt, drops = 0, 0
        while attempt <= self.s.retries:
            if attempt:
                await asyncio.sleep(min(0.5 * 2 ** (attempt - 1), 4.0))
            try:
                async with self.sem:
                    self.calls += 1
                    r = await self.client.post(url, json=body, headers=headers, timeout=timeout)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = f"{type(e).__name__}: {e}"
                attempt += 1
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last = f"backend HTTP {r.status_code}"
                if r.status_code == 429:
                    try:
                        await asyncio.sleep(min(float(r.headers.get("retry-after", "1")), 10.0))
                    except ValueError:
                        await asyncio.sleep(1.0)
                attempt += 1
                continue
            if r.status_code >= 400:
                text = r.text[:400]
                if hosts.requires_streaming(r.status_code, text) and not body.get("stream"):
                    self.streaming.add(spec.backend_model)
                    body.update(stream=True, stream_options={"include_usage": True})
                    drops += 1
                    continue
                if hosts.needs_more_tokens(r.status_code, text) and (body.get(tok_key) or 0) < hosts.MIN_TOKENS_RETRY:
                    self.min_tokens[spec.backend_model] = hosts.MIN_TOKENS_RETRY
                    body[tok_key] = hosts.MIN_TOKENS_RETRY
                    drops += 1
                    continue
                field = hosts.rejected_field(r.status_code, text, body)
                if field and drops < len(hosts.DROPPABLE):
                    dropped.add(field)
                    body.pop(field)
                    drops += 1
                    continue
                if _has_images(body) and any(k in text.lower() for k in IMAGE_REFUSALS):
                    raise HunchError(400, "images_not_supported",
                                     f"model {spec.name!r} ({spec.backend_model}) does not accept images: {text[:200]}")
                if any(k in text.lower() for k in ("context length", "maximum context", "too long")):
                    raise HunchError(413, "context_too_long", "context plus question exceed the backend model's context window")
                raise HunchError(502, "backend_error", f"backend HTTP {r.status_code}: {text}")
            data = hosts.collect_stream(r.text) if body.get("stream") else r.json()
            if body.get("stream") and data.get("error"):
                raise HunchError(502, "backend_error", f"backend stream error: {str(data['error'])[:300]}")
            why = retry_if(data) if retry_if else None
            if why:
                last = why
                attempt += 1
                continue
            return data
        raise HunchError(503, "backend_unavailable", last)

    @property
    def max_options(self) -> int:
        return self.s.group_size * min(self.s.max_top_logprobs, len(LETTERS))

    # ------------------------------------------------------------- backend call
    def _request(self, spec: ModelSpec, msgs: list[dict], labels: str, deliberate: bool) -> dict:
        extra = effective_extra_body(spec, deliberate)
        body = {
            "model": spec.backend_model,
            "messages": msgs,
            # Use structured_outputs: some vLLM versions silently ignore the legacy `guided_choice` param.
            "structured_outputs": {"choice": list(labels)},
            "logprobs": True,
            "top_logprobs": min(self.s.max_top_logprobs, max(len(labels), 2)),
            "max_tokens": spec.think_budget if deliberate else 1,
            "temperature": 0,
            **extra,
        }
        if not spec.constrained:
            del body["structured_outputs"]
        return body

    async def label_probs(self, spec: ModelSpec, msgs: list[dict], labels: str,
                          mode: str | None = None) -> tuple[dict[str, float], Usage]:
        mode = mode or await self.mode_for(spec)
        if mode == "sample":
            return await self._sample_probs(spec, msgs, labels)
        deliberate = mode == "deliberate"
        body = self._request(spec, msgs, labels, deliberate)
        if self.unconstrained(spec):
            body.pop("structured_outputs", None)

        def no_logprobs(data: dict) -> str | None:
            # A router in front of several providers occasionally serves a call from one that drops logprobs.
            # The model never answered here, so this is a retry, not a guess. Only for unconstrained backends:
            # on vLLM a missing logprob is a real error and is reported as one.
            if self.unconstrained(spec) and not ((data.get("choices") or [{}])[0].get("logprobs") or {}).get("content"):
                return "backend returned no logprobs"
            return None

        data = await self.send(spec, body, retry_if=no_logprobs)
        # unconstrained() is read again: the backend may have refused structured_outputs during this call
        try:
            return self._parse(data, labels, deliberate,
                               min_label_mass=spec.min_label_mass if self.unconstrained(spec) else None)
        except HunchError as e:
            if e.code != "off_label":
                raise
            # Hosted APIs are not repeatable even at temperature 0: Qwen3.8-Flash on Together once answered "," in
            # 720 calls and never again in the next 720. Ask once more; a model that really answers off the
            # options does it again and fails as before.
            data = await self.send(spec, body, retry_if=no_logprobs)
            return self._parse(data, labels, deliberate, min_label_mass=spec.min_label_mass)

    async def post_chat(self, spec: ModelSpec, body: dict) -> dict:
        """One chat completion with the retry / error mapping every Hunch call uses. Returns the JSON body."""
        return await self.send(spec, body)

    async def _sample_probs(self, spec: ModelSpec, msgs: list[dict], labels: str) -> tuple[dict[str, float], Usage]:
        """Mode "sample": no logprobs, so ask `samples` times at temperature > 0 and count the votes.

        Answers are read leniently (the first allowed label in the reply, or yes / no for Y / N), because
        there is no constraint to enforce the format. Replies with no label are discarded; if none of the
        samples answers, the call fails rather than guess.

        max_tokens is think_budget, not a handful: hosted models may think before answering (some do it only
        on harder questions), and a small cap cuts them off with an empty reply. A direct answer still stops
        after its one label, so the cap only costs when the model actually thinks.
        """
        body = {"model": spec.backend_model, "messages": msgs, "max_tokens": spec.think_budget,
                "temperature": spec.sample_temperature, **spec.extra_body}
        replies = await asyncio.gather(*[self.post_chat(spec, body) for _ in range(max(1, spec.samples))])
        votes = {lab: 0 for lab in labels}
        usage = Usage()
        unread: list[str] = []
        cut = 0                                  # replies stopped by max_tokens before any visible answer
        for data in replies:
            u = data.get("usage") or {}
            usage.add(Usage(u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0, 1))
            try:
                text = (data["choices"][0].get("message") or {}).get("content") or ""
            except (KeyError, IndexError, TypeError):
                continue
            lab = read_label(text, labels)
            if lab is not None:
                votes[lab] += 1
            else:
                unread.append(text)
                cut += not text.strip() and (data["choices"][0].get("finish_reason") == "length")
        n = sum(votes.values())
        if n == 0 and cut:
            raise HunchError(502, "no_label", f"{cut} of {spec.samples} sampled replies were cut off at "
                                              f"think_budget={spec.think_budget} tokens before answering (the model "
                                              "thinks first on some questions); raise think_budget")
        if n == 0:
            # say what the model wrote: an empty reply, a refusal and a label buried mid-sentence need different fixes
            seen = "; ".join(repr(t[:160]) for t in unread[:2]) or "no text"
            raise HunchError(502, "no_label", f"none of {spec.samples} sampled replies named one of {list(labels)}; "
                                              f"replies were e.g. {seen}")
        return {k: v / n for k, v in votes.items()}, usage

    async def mode_for(self, spec: ModelSpec) -> str:
        """The resolved mode. With mode="auto", probe the backend once and cache the answer."""
        if spec.mode != "auto":
            if spec.mode not in MODES:
                raise HunchError(400, "invalid_request", f"model {spec.name!r}: mode must be one of {MODES}")
            return spec.mode
        if spec.name not in self._detected:
            self._detected[spec.name] = "deliberate" if await self.opens_scratchpad(spec) else "one_token"
        return self._detected[spec.name]

    async def opens_scratchpad(self, spec: ModelSpec) -> bool:
        """True when the model starts thinking instead of answering, i.e. it has no non-thinking mode.

        Such models (GLM-5.3, for one) append a thinking block to every generation prompt, so constraining
        the FIRST token to a label reads a distribution that is not about the answer. The probe asks for a
        one-word answer without any constraint and checks whether one arrives:

          - thinking text in `reasoning` / `reasoning_content` -> it is thinking
          - the scratchpad leaking into `content` instead of the answer -> it is thinking
            (vLLM issue #54744: the parser is gated on kwargs some templates never read)
          - anything that is not the requested word -> treat as thinking

        A single token is not enough to tell: GLM returns just "The" there, which looks like an answer.
        """
        return await self._starts_thinking(spec, spec.extra_body)

    async def can_think(self, spec: ModelSpec) -> bool:
        """True when the model thinks once the switches that turn thinking off are dropped.

        Different from opens_scratchpad: DeepSeek-V4.1-Flash has a real non-thinking mode and answers
        in one token, but those answers follow the listed order of the options (ECE 0.19 with debias).
        Allowed to think, it qualifies with ECE 0.02. qualify uses this to try deliberate mode too.
        """
        return await self._starts_thinking(spec, effective_extra_body(spec, deliberate=True))

    async def _starts_thinking(self, spec: ModelSpec, extra: dict) -> bool:
        body = {"model": spec.backend_model,
                "messages": [{"role": "user", "content": "Reply with exactly one word: yes"}],
                "max_tokens": PROBE_MAX_TOKENS, "temperature": 0, **extra}
        try:
            msg = (await self.send(spec, body))["choices"][0].get("message") or {}
        except (HunchError, KeyError, IndexError, TypeError):
            return False            # can't tell; assume the cheap mode rather than force the expensive one
        if (msg.get("reasoning") or msg.get("reasoning_content") or "").strip():
            return True             # 0.29.0 uses `reasoning`; older builds use `reasoning_content`
        return not (msg.get("content") or "").strip().lower().lstrip('"\'*_ ').startswith("yes")

    @staticmethod
    def _parse(data: dict, labels: str, deliberate: bool = False,
               min_label_mass: float | None = None) -> tuple[dict[str, float], Usage]:
        """min_label_mass None: the backend constrained the answer. A number: it did not (constrained=False),
        so read only the allowed labels from the top_logprobs and require them to hold that much probability."""
        allowed = set(labels)   # NOT `in labels`: "" is a substring of every string
        try:
            toks = data["choices"][0]["logprobs"]["content"]
            # one_token: the answer IS the first token. deliberate: the model thinks first, so the
            # verdict is the last token that is one of the labels.
            tok = next((t for t in reversed(toks) if str(t.get("token", "")).strip() in allowed)) if deliberate else toks[0]
        except (KeyError, IndexError, TypeError, StopIteration):
            if deliberate:
                raise HunchError(502, "backend_error",
                                 "no constrained verdict token within think_budget; raise it or use mode=one_token")
            raise HunchError(502, "backend_error", "backend returned no logprobs")
        chosen = str(tok.get("token", "")).strip()
        if min_label_mass is not None and not deliberate:
            return Engine._parse_unconstrained(data, tok, labels, chosen, min_label_mass)
        if chosen not in allowed:
            # Never derive a probability from an unrelated token. Say WHY: a thinking model in one_token mode
            # spends its one token opening the reasoning (it lands in `reasoning`), which is a config problem,
            # not a backend that ignores the constraint.
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            if (msg.get("reasoning") or msg.get("reasoning_content") or "").strip():
                raise HunchError(502, "backend_error",
                                 f"the model is thinking in one_token mode (its one token {chosen!r} opened its reasoning): "
                                 "switch thinking off for this model (reasoning_effort \"none\", or its chat template's "
                                 "thinking flag in extra_body), or set mode = \"deliberate\"")
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

    @staticmethod
    def _parse_unconstrained(data: dict, tok: dict, labels: str, chosen: str,
                             min_label_mass: float) -> tuple[dict[str, float], Usage]:
        probs = {lab: 0.0 for lab in labels}
        for t in tok.get("top_logprobs") or []:
            s = str(t.get("token", "")).strip()
            if s in probs and t.get("logprob") is not None:
                probs[s] += math.exp(t["logprob"])
        if chosen in probs and probs[chosen] == 0.0 and tok.get("logprob") is not None:
            probs[chosen] = math.exp(tok["logprob"])
        mass = sum(probs.values())
        if mass < min_label_mass:
            raise HunchError(502, "off_label",
                             f"unconstrained backend answered {chosen!r}; the allowed labels {list(labels)} hold only "
                             f"{mass:.2f} of the probability (min_label_mass {min_label_mass:g})")
        u = data.get("usage") or {}
        return {k: v / mass for k, v in probs.items()}, Usage(u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0, 1)

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

    async def tags(self, spec, ctx, c) -> tuple[dict, Usage]:
        """Multi-label: every tag is its own yes/no check, all in parallel over the same cached context."""
        items = list(c["tags"].items())
        results = await asyncio.gather(*[
            self.yesno(spec, ctx, {"question": prompts.tag_question(c.get("question"), key, desc)}) for key, desc in items])
        usage = Usage()
        probs = {}
        for (key, _), (r, u) in zip(items, results):
            usage.add(u)
            probs[key] = r["p_yes"]
        return {"kind": "tags", "tags": [k for k, p in probs.items() if p >= 0.5], "probs": probs}, usage

    async def sort(self, spec, ctx, c) -> tuple[dict, Usage]:
        """Order items by a criterion. The model rates each item (0-9, in parallel); code sorts by the
        expected rating, so the order is as fine-grained as the probabilities, not just 10 buckets."""
        items = c["items"]
        keyed = list(items.items()) if isinstance(items, dict) else [(str(i), v) for i, v in enumerate(items)]
        levels = c.get("levels") or prompts.SORT_LEVELS
        labels = DIGITS[:len(levels)]
        results = await asyncio.gather(*[
            self.label_probs(spec, prompts.messages(ctx, prompts.sort_item(c.get("question"), item, levels)), labels)
            for _, item in keyed])
        usage = Usage()
        scores = {}
        for (key, _), (p, u) in zip(keyed, results):
            usage.add(u)
            scores[key] = round(sum(i * p[lab] for i, lab in enumerate(labels)), 4)
        desc = c.get("order", "desc") != "asc"
        order = sorted(scores, key=lambda k: scores[k], reverse=desc)   # stable: ties keep the given order
        return {"kind": "sort", "order": order, "scores": scores}, usage

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
        calls += [self._within_group(spec, ctx, question, g) for g in groups]
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

    async def _within_group(self, spec, ctx, question, group) -> tuple[dict[str, float], Usage]:
        """P(option | group). Unconstrained (mode "sample"), a model shown a group that lacks the answer may
        decline to pick from it ("none of these"). That is no preference within the group, not an error: the
        group call already gives such a group almost no weight."""
        labels = LETTERS[:len(group)]
        try:
            return await self.label_probs(spec, prompts.messages(ctx, prompts.pick(question, group, labels)), labels)
        except HunchError as e:
            if e.code != "no_label":
                raise
            return {lab: 1 / len(labels) for lab in labels}, Usage(backend_calls=spec.samples)

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
    async def judge(self, spec: ModelSpec, context: Any, checks: dict[str, dict],
                    images: list[str] | None = None, min_probability: float | None = None,
                    reasoning: str | None = None, escalate_below: float | None = None) -> tuple[dict, Usage]:
        """Run every check in parallel. `reasoning` (see REASONING): None keeps the model's own mode;
        "off" answers in one token; "on" thinks first; "auto" answers in one token and re-asks, with
        thinking, only the checks whose probability is under `escalate_below` (default 0.9) or whose
        answer a one-token pre-check says has to be worked out (arithmetic, counting, dates, rule chains)."""
        ctx = prompts.context_block(context, images or ())
        handlers = {"yesno": self.yesno, "pick": self.pick, "scale": self.scale, "tags": self.tags, "sort": self.sort}
        if reasoning is not None and reasoning not in REASONING:
            raise HunchError(400, "invalid_request", f"`reasoning` must be one of {REASONING}")
        if reasoning is not None and await self.mode_for(spec) == "sample":
            reasoning = None          # votes have no one-token / thinking split; run as configured
        floor = DEFAULT_ESCALATE_BELOW if escalate_below is None else escalate_below

        async def run(cid: str, c: dict) -> tuple[dict, Usage]:
            h = handlers[c["kind"]]
            if reasoning is None:
                r, u = await h(spec, ctx, c)
                return with_answer(r, c, min_probability), u
            if reasoning == "on":
                r, u = await h(thinking_spec(spec), ctx, c)
                return {**with_answer(r, c, min_probability), "reasoned": True}, u
            if reasoning == "off":
                r, u = await h(fast_spec(spec), ctx, c)
                return {**with_answer(r, c, min_probability), "reasoned": False}, u
            # auto: the cheap answer first, and, alongside it on the same cached context, a one-token
            # pre-check of whether the answer has to be worked out. Think when either says so: confidence
            # alone misses one-pass answers that are confidently wrong on a computed question.
            usage = Usage()
            fast_p, needs_steps = None, None
            fast_task = asyncio.ensure_future(h(fast_spec(spec), ctx, c))
            gate_task = asyncio.ensure_future(self.yesno(fast_spec(spec), ctx, prompts.needs_steps_check(c)))
            try:
                g, gu = await gate_task
                usage.add(gu)
                needs_steps = g["p_yes"] < 0.5          # the check asks "can it be read directly?"
            except HunchError as e:
                if e.code != "backend_error":
                    fast_task.cancel()
                    raise
            try:
                r, u = await fast_task
                usage.add(u)
                r = with_answer(r, c, min_probability)
                fast_p = r.get("probability")
                sure = fast_p is None or fast_p >= floor   # sort has no probability: one pass is the answer
                if sure and needs_steps is False:
                    return {**r, "reasoned": False}, usage
            except HunchError as e:
                if e.code != "backend_error":            # e.g. a model with no thinking-off mode: think instead
                    raise
            r, u = await h(thinking_spec(spec), ctx, c)
            usage.add(u)
            out = {**with_answer(r, c, min_probability), "reasoned": True}
            if fast_p is not None:
                out["fast_probability"] = round(fast_p, 4)
            if needs_steps is not None:
                out["needs_steps"] = needs_steps
            return out, usage

        tasks = {cid: asyncio.ensure_future(run(cid, c)) for cid, c in checks.items()}
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


REASONING = ("off", "on", "auto")
DEFAULT_ESCALATE_BELOW = 0.9


def fast_spec(spec: ModelSpec) -> ModelSpec:
    """The model answering in its first token, with thinking switched off."""
    extra = dict(spec.extra_body)
    if str(extra.get("reasoning_effort", "none")).lower() not in THINKING_OFF_EFFORTS:
        extra["reasoning_effort"] = "none"            # a configured effort would switch thinking on
    return replace(spec, mode="one_token", extra_body=extra)


def thinking_spec(spec: ModelSpec) -> ModelSpec:
    """The model thinking first, its final verdict token constrained (deliberate mode)."""
    return replace(spec, mode="deliberate")

def with_answer(result: dict, check: dict, min_probability: float | None = None) -> dict:
    """Add the uniform `answer` / `probability` pair to a result (additive: the kind's own fields stay).

    yesno -> "yes" | "no"; pick -> the key; scale -> the most likely level index; tags -> the list of
    tags at p >= 0.5; sort -> the order. `answer` is null when `probability` is under the check's
    `min_probability` (or the request's): the model was not sure, and code should not act on it.
    """
    kind = result.get("kind")
    if kind == "yesno":
        p = result["p_yes"]
        answer, prob = ("yes", p) if p >= 0.5 else ("no", 1 - p)
    elif kind == "pick":
        answer, prob = result["pick"], result["probs"][result["pick"]]
    elif kind == "scale":
        probs = result["probs"]
        answer = max(range(len(probs)), key=probs.__getitem__)
        prob = probs[answer]
    elif kind == "tags":
        # the least certain tag decides how far the whole set can be trusted
        answer = result["tags"]
        prob = min((max(p, 1 - p) for p in result["probs"].values()), default=1.0)
    else:
        return result
    floor = check.get("min_probability", min_probability)
    if floor is not None and prob < floor:
        answer = None
    return {**result, "answer": answer, "probability": round(prob, 4)}


MAX_IMAGES = 8


def validate_images(images: list[str] | None) -> None:
    if not images:
        return
    if len(images) > MAX_IMAGES:
        raise HunchError(400, "too_many_images", f"at most {MAX_IMAGES} images per request, got {len(images)}")
    for i, url in enumerate(images, 1):
        if not isinstance(url, str) or not url.startswith(("http://", "https://", "data:image/")):
            raise HunchError(400, "invalid_request", f"image {i} must be an http(s) URL or a data:image/... URL")


async def apply_effort(engine: "Engine", spec: ModelSpec, effort: str | None) -> ModelSpec:
    """A per-call thinking effort. Off-values are refused; on a one_token model it is ignored, because it
    would switch thinking on and break the one-token answer."""
    if effort is None:
        return spec
    if effort.lower() in THINKING_OFF_EFFORTS:
        raise HunchError(400, "invalid_request", f"effort {effort!r} would switch thinking off; use mode=one_token instead")
    if await engine.mode_for(spec) == "deliberate":
        return replace(spec, extra_body={**spec.extra_body, "reasoning_effort": effort})
    return spec


async def run_info(engine: "Engine", spec: ModelSpec, requested_effort: str | None) -> dict:
    """How a call actually ran, for the response: the resolved mode, the thinking effort really sent (None in
    one_token mode, where thinking is off), and a note whenever a requested effort was not applied, so a
    caller never mistakes an ignored effort for an applied one."""
    mode = await engine.mode_for(spec)
    if mode == "sample":
        return {"mode": mode, "effort": None, "samples": spec.samples}
    sent = effective_extra_body(spec, mode == "deliberate").get("reasoning_effort") if mode == "deliberate" else None
    info: dict = {"mode": mode, "effort": sent}
    if requested_effort is not None and sent != requested_effort:
        info["notes"] = [f"effort {requested_effort!r} not applied: model {spec.name!r} runs {mode}, with thinking off; "
                         "set mode = \"deliberate\" for this model in hunch.toml (or HUNCH_MODE=deliberate "
                         "when no config file declares it) to let it think"]
    return info


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
        elif kind == "tags":
            tags = c.get("tags")
            if not isinstance(tags, dict) or not tags:
                raise HunchError(400, "invalid_request", f"check {cid!r}: `tags` must be a non-empty object of key -> description|null")
            if len(tags) > MAX_TAGS:
                raise HunchError(400, "too_many_options", f"check {cid!r} has {len(tags)} tags; the maximum is {MAX_TAGS}")
        elif kind == "sort":
            items = c.get("items")
            if not isinstance(items, (dict, list)) or not items:
                raise HunchError(400, "invalid_request", f"check {cid!r}: `items` must be a non-empty list or object of key -> item")
            if len(items) > MAX_SORT_ITEMS:
                raise HunchError(400, "too_many_options", f"check {cid!r} has {len(items)} items; the maximum is {MAX_SORT_ITEMS}")
            if c.get("order", "desc") not in ("asc", "desc"):
                raise HunchError(400, "invalid_request", f"check {cid!r}: `order` must be asc or desc")
            levels = c.get("levels")
            if levels is not None and (not isinstance(levels, list) or not 2 <= len(levels) <= MAX_LEVELS):
                raise HunchError(400, "invalid_request", f"check {cid!r}: `levels` must list 2-{MAX_LEVELS} levels, lowest first")
        else:
            raise HunchError(400, "invalid_request", f"check {cid!r}: `kind` must be one of yesno, pick, scale, tags, sort")
        mp = c.get("min_probability")
        if mp is not None and not (isinstance(mp, (int, float)) and 0 <= mp <= 1):
            raise HunchError(400, "invalid_request", f"check {cid!r}: `min_probability` must be a number in [0, 1]")
        out[cid] = c
    return out
