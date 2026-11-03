"""Streaming decisions: keep appending text (a call transcript, a chat, a log) and get the checks re-judged
on everything so far, with the ids whose answer just changed.

Cheap by construction: the context is always the first thing in every prompt, so each re-judgment shares
its prefix with the previous one and the backend's prefix cache only has to read the new text. A check's
answer is the uniform `answer` field (see engine.with_answer); `changed` lists the checks whose answer is
different from the last update, so code can act the moment one flips.

    from hunch import astream
    async with astream({"cancel": {"kind": "yesno", "question": "Does the caller want to cancel?"}}) as s:
        for chunk in transcript_chunks:
            u = await s.append(chunk)
            if "cancel" in u["changed"] and u["results"]["cancel"]["answer"] == "yes":
                ...

Over HTTP the same session is a WebSocket on /v1/stream (see hunch.server).
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, AsyncIterator

import httpx

from .config import ModelSpec
from .engine import Engine, HunchError, Usage, apply_effort, validate_checks


class Session:
    """One growing text and a fixed set of checks."""

    def __init__(self, engine: Engine, spec: ModelSpec, checks: dict[str, dict], *,
                 min_probability: float | None = None, text: str = "",
                 reasoning: str | None = None, escalate_below: float | None = None):
        self.engine, self.spec = engine, spec
        self.checks = validate_checks(checks, engine.max_options)
        self.min_probability = min_probability
        self.reasoning, self.escalate_below = reasoning, escalate_below
        self.text = text
        self.answers: dict[str, Any] = {}
        self.updates = 0
        self.usage = Usage()

    async def append(self, text: str) -> dict:
        """Add `text` (sent as is: include your own spaces / newlines) and re-judge. Returns
        {"results", "changed", "update", "chars", "usage", "latency_ms"}."""
        self.text += text
        return await self.judge()

    async def judge(self) -> dict:
        """Re-judge the text as it stands."""
        t0 = time.perf_counter()
        results, usage = await self.engine.judge(self.spec, self.text, self.checks, min_probability=self.min_probability,
                                                 reasoning=self.reasoning, escalate_below=self.escalate_below)
        changed = [cid for cid, r in results.items() if self.updates == 0 or self.answers.get(cid) != r.get("answer")]
        self.answers = {cid: r.get("answer") for cid, r in results.items()}
        self.updates += 1
        self.usage.add(usage)
        return {"results": results, "changed": changed, "update": self.updates, "chars": len(self.text),
                "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                          "backend_calls": usage.backend_calls},
                "latency_ms": round((time.perf_counter() - t0) * 1000)}


@contextlib.asynccontextmanager
async def astream(checks: dict[str, Any], *, text: str = "", model: str | None = None, effort: str | None = None,
                  min_probability: float | None = None, reasoning: str | None = None,
                  escalate_below: float | None = None, base_url: str | None = None, api_key: str | None = None,
                  transport: httpx.AsyncBaseTransport | None = None) -> AsyncIterator[Session]:
    """A streaming session with the caller's LLM (settings found exactly as for judge())."""
    from .client import _MODES, _settings
    if not checks:
        raise HunchError(400, "invalid_request", "`checks` must contain at least one check")
    settings, spec = await _settings(base_url, api_key, model)
    async with httpx.AsyncClient(**({"transport": transport} if transport is not None else {})) as client:
        engine = Engine(settings, client)
        engine._detected = _MODES.setdefault((settings.backend_url, spec.backend_model), {})
        spec = await apply_effort(engine, spec, effort)
        yield Session(engine, spec, checks, min_probability=min_probability, text=text,
                      reasoning=reasoning, escalate_below=escalate_below)


async def run_socket(receive, send, engine: Engine, resolve_spec) -> None:
    """The /v1/stream protocol, independent of the web framework.

    client -> {"type": "start", "checks": {...}, "model"?, "effort"?, "min_probability"?, "reasoning"?, "escalate_below"?,
               "text"?, "emit"?: "all"|"changes"}
    client -> {"type": "append", "text": "..."}            (any number)
    client -> {"type": "end"}                              (judges what is pending, then closes)
    server -> {"type": "ready"} | {"type": "decision", ...Session.append result} | {"type": "error", "code", "message"}

    Appends that arrive while a judgment is running are coalesced into the next one, so a fast talker
    never queues up stale work: every decision is about the newest text.
    """
    try:
        start = await receive()
        if not isinstance(start, dict) or start.get("type") != "start":
            raise HunchError(400, "invalid_request", "the first message must be {\"type\": \"start\", \"checks\": {...}}")
        emit = start.get("emit", "changes")
        if emit not in ("all", "changes"):
            raise HunchError(400, "invalid_request", "`emit` must be all or changes")
        spec = await apply_effort(engine, resolve_spec(start.get("model")), start.get("effort"))
        session = Session(engine, spec, start.get("checks") or {}, min_probability=start.get("min_probability"),
                          text=str(start.get("text") or ""), reasoning=start.get("reasoning"),
                          escalate_below=start.get("escalate_below"))
    except HunchError as e:
        await send({"type": "error", "code": e.code, "message": e.message})
        return
    await send({"type": "ready"})

    pending = asyncio.Event()
    state = {"dirty": bool(session.text), "closing": False}
    if state["dirty"]:
        pending.set()

    async def reader():
        try:
            while True:
                msg = await receive()
                kind = msg.get("type") if isinstance(msg, dict) else None
                if kind == "append":
                    session.text += str(msg.get("text") or "")
                    state["dirty"] = True
                elif kind == "end":
                    state["closing"] = True
                    pending.set()
                    return
                else:
                    await send({"type": "error", "code": "invalid_request", "message": "expected append or end"})
                    continue
                pending.set()
        except Exception:  # noqa: BLE001 - the client went away
            state["closing"] = True
            pending.set()

    reader_task = asyncio.ensure_future(reader())
    try:
        while True:
            await pending.wait()
            pending.clear()
            if state["dirty"]:
                state["dirty"] = False
                try:
                    out = await session.judge()
                except HunchError as e:
                    await send({"type": "error", "code": e.code, "message": e.message})
                else:
                    if emit == "all" or out["changed"]:
                        if emit == "changes":
                            out = {**out, "results": {cid: out["results"][cid] for cid in out["changed"]}}
                        await send({"type": "decision", **out})
                if state["dirty"]:
                    pending.set()
                    continue
            if state["closing"]:
                return
    finally:
        reader_task.cancel()
