"""Hunch as a library: judge with the LLM the caller already uses, no server and no hunch.toml.

    from hunch import judge
    r = judge({"ticket": "Charged twice, fix it or I cancel."},
              {"refund": {"kind": "yesno", "question": "Is the customer asking for money back?"}})
    r["results"]["refund"]["p_yes"]

Settings, first match wins:
  1. arguments: judge(..., base_url=, api_key=, model=)
  2. a hunch.toml (HUNCH_CONFIG, or ./hunch.toml), exactly as the server reads it
  3. the environment an agent already has for its LLM (hunch.config.URL_VARS / KEY_VARS / MODEL_VARS):
       endpoint  HUNCH_BACKEND_URL, ANTHROPIC_BASE_URL, OPENAI_BASE_URL, OPENAI_API_BASE, LLM_BASE_URL
       key       HUNCH_BACKEND_KEY, ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, OPENAI_API_KEY, LLM_API_KEY
                 (none is fine: local gateways often take no key)
       model     HUNCH_BACKEND_MODEL, ANTHROPIC_MODEL, OPENAI_MODEL, LLM_MODEL; if none is set and the
                 endpoint serves exactly one model, that one
  mode defaults to "auto" (one probe, cached per process): thinking models are asked in deliberate
  mode without any setting. HUNCH_MODE / HUNCH_EFFORT override it.

The endpoint must be an OpenAI-compatible vLLM server: Hunch reads token logprobs under a constrained
choice, which hosted APIs without logprobs cannot provide. A missing capability is reported up front.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import pathlib
from typing import Any

import httpx

from .config import DiscoveryError, ModelSpec, Settings, discover_settings, load_settings
from .engine import Engine, HunchError, apply_effort, run_info, validate_checks, validate_images

_MODES: dict[tuple[str, str], dict[str, str]] = {}   # (endpoint, backend model) -> auto-mode answer, per process


def _has_toml() -> bool:
    return bool(os.environ.get("HUNCH_CONFIG")) or pathlib.Path("hunch.toml").exists()


async def _settings(base_url: str | None, api_key: str | None, model: str | None) -> tuple[Settings, ModelSpec]:
    if base_url is None and api_key is None and _has_toml():
        s = load_settings()
        spec = s.resolve(model)
        if spec is None:
            raise HunchError(400, "unknown_model", f"unknown model {model!r}; configured: {sorted(s.models)}")
        return s, spec
    try:
        # may list the endpoint's models (one blocking GET), so keep it off the event loop
        s = await asyncio.to_thread(discover_settings, base_url, api_key, model)
    except DiscoveryError as e:
        raise HunchError(400, "invalid_request", str(e)) from e
    return s, s.resolve(None)


async def ajudge(context: Any = "", checks: dict[str, Any] | None = None, *, images: list[str] | None = None,
                 model: str | None = None, effort: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    """Async version of judge(). Returns {"model", "results", "usage"}; raises HunchError."""
    if not checks:
        raise HunchError(400, "invalid_request", "`checks` must contain at least one check")
    validate_images(images)
    settings, spec = await _settings(base_url, api_key, model)
    async with httpx.AsyncClient(**({"transport": transport} if transport is not None else {})) as client:
        engine = Engine(settings, client)
        # the auto-mode probe costs a call: remember its answer for the life of the process
        engine._detected = _MODES.setdefault((settings.backend_url, spec.backend_model), {})
        checks = validate_checks(checks, engine.max_options)
        spec = await apply_effort(engine, spec, effort)
        results, usage = await engine.judge(spec, context, checks, images)
        info = await run_info(engine, spec, effort)
    return {"model": spec.backend_model, "results": results,
            "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                      "backend_calls": usage.backend_calls}, **info}


def judge(context: Any = "", checks: dict[str, Any] | None = None, **kwargs: Any) -> dict:
    """Judge `checks` about `context` (and optional `images`) with the caller's LLM. Blocking.

    Safe to call from code that already runs an event loop: it then runs on a worker thread.
    """
    coro = ajudge(context, checks, **kwargs)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
