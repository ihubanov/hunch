"""Hunch HTTP API: POST /v1/judge, GET /v1/models, GET /health."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from . import __version__
from .config import Settings, load_settings
from dataclasses import replace

from .engine import THINKING_OFF_EFFORTS, Engine, HunchError, validate_checks


class JudgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context: str | dict[str, Any] | list[Any] = ""
    checks: dict[str, Any]
    # Images for a vision model: http(s) URLs the backend can fetch, or data:image/...;base64 URLs.
    # They are part of the context, numbered IMAGE 1..n, so questions can refer to them.
    images: list[str] | None = None
    model: str | None = None
    # Per-request thinking effort for deliberate-mode models ("low" / "high" / "max"): trade cost for
    # stability on this call only. Ignored by backends that don't support it.
    effort: str | None = None


MAX_IMAGES = 8


def validate_images(images: list[str] | None) -> None:
    if not images:
        return
    if len(images) > MAX_IMAGES:
        raise HunchError(400, "too_many_images", f"at most {MAX_IMAGES} images per request, got {len(images)}")
    for i, url in enumerate(images, 1):
        if not isinstance(url, str) or not url.startswith(("http://", "https://", "data:image/")):
            raise HunchError(400, "invalid_request", f"image {i} must be an http(s) URL or a data:image/... URL")


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        limits = httpx.Limits(max_connections=settings.max_concurrency * 2, max_keepalive_connections=settings.max_concurrency)
        async with httpx.AsyncClient(transport=transport, limits=limits) as client:
            app.state.engine = Engine(settings, client)
            yield

    app = FastAPI(title="Hunch", version=__version__, lifespan=lifespan)

    @app.exception_handler(HunchError)
    async def _hunch_error(_: Request, exc: HunchError):
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        return _error(400, "invalid_request", f"{where}: {first.get('msg', 'invalid request')}")

    def auth(authorization: str | None = Header(default=None)):
        if settings.api_keys and (authorization or "").removeprefix("Bearer ").strip() not in settings.api_keys:
            raise HunchError(401, "unauthorized", "invalid or missing API key")

    @app.post("/v1/judge", dependencies=[Depends(auth)])
    async def judge(req: JudgeRequest, request: Request):
        engine: Engine = request.app.state.engine
        spec = settings.resolve(req.model)
        if spec is None:
            raise HunchError(400, "unknown_model", f"unknown model {req.model!r}; configured: {sorted(settings.models)}")
        if req.effort is not None:
            if req.effort.lower() in THINKING_OFF_EFFORTS:
                raise HunchError(400, "invalid_request",
                                 f"effort {req.effort!r} would switch thinking off; use mode=one_token instead")
            spec = replace(spec, extra_body={**spec.extra_body, "reasoning_effort": req.effort})
        checks = validate_checks(req.checks, engine.max_options)
        validate_images(req.images)
        t0 = time.perf_counter()
        results, usage = await engine.judge(spec, req.context, checks, req.images)
        return {"model": spec.name, "results": results,
                "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                          "backend_calls": usage.backend_calls},
                "latency_ms": round((time.perf_counter() - t0) * 1000)}

    @app.get("/v1/models", dependencies=[Depends(auth)])
    async def models(request: Request):
        engine: Engine = request.app.state.engine
        out = []
        for s in settings.models.values():
            # `mode` is what checks actually run in: "auto" is resolved (one probe, cached), because the
            # configured value alone can't tell a reader whether answers are one token or deliberate.
            try:
                mode = await engine.mode_for(s)
            except HunchError:
                mode = None
            out.append({"name": s.name, "backend_model": s.backend_model, "description": s.description,
                        "debias": s.debias, "mode": mode, "configured_mode": s.mode,
                        "default": s.name == settings.default_model})
        return {"models": out}

    @app.get("/health")
    async def health(request: Request):
        engine: Engine = request.app.state.engine
        try:
            headers = {"Authorization": f"Bearer {settings.backend_api_key}"} if settings.backend_api_key else {}
            r = await engine.client.get(f"{settings.backend_url}/v1/models", headers=headers, timeout=5)
            if r.status_code != 200:
                return JSONResponse({"ok": False, "error": f"backend /v1/models returned HTTP {r.status_code}"}, status_code=503)
            served = {m["id"] for m in r.json().get("data", [])}
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"ok": False, "error": repr(e)}, status_code=503)
        status = {s.name: s.backend_model in served for s in settings.models.values()}
        ok = bool(status) and all(status.values())
        return JSONResponse({"ok": ok, "models_served": status}, status_code=200 if ok else 503)

    return app
