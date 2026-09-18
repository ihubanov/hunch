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
from .engine import Engine, HunchError, validate_checks


class JudgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context: str | dict[str, Any] | list[Any]
    checks: dict[str, Any]
    model: str | None = None


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
        checks = validate_checks(req.checks, engine.max_options)
        t0 = time.perf_counter()
        results, usage = await engine.judge(spec, req.context, checks)
        return {"model": spec.name, "results": results,
                "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                          "backend_calls": usage.backend_calls},
                "latency_ms": round((time.perf_counter() - t0) * 1000)}

    @app.get("/v1/models", dependencies=[Depends(auth)])
    async def models():
        return {"models": [{"name": s.name, "backend_model": s.backend_model, "description": s.description,
                            "debias": s.debias, "default": s.name == settings.default_model}
                           for s in settings.models.values()]}

    @app.get("/health")
    async def health(request: Request):
        engine: Engine = request.app.state.engine
        try:
            r = await engine.client.get(f"{settings.backend_url}/v1/models", timeout=5)
            served = {m["id"] for m in r.json().get("data", [])}
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"ok": False, "error": repr(e)}, status_code=503)
        status = {s.name: s.backend_model in served for s in settings.models.values()}
        ok = bool(status) and all(status.values())
        return JSONResponse({"ok": ok, "models_served": status}, status_code=200 if ok else 503)

    return app
