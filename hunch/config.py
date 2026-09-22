"""Settings: the backend (any vLLM OpenAI-compatible server), the models to expose, and service limits.

Read from a TOML file (HUNCH_CONFIG, else ./hunch.toml if present), with environment overrides.
Quickest setup without a file: HUNCH_BACKEND_URL + HUNCH_BACKEND_MODEL (exposed as model "default").
"""
from __future__ import annotations

import os
import pathlib

from dataclasses import dataclass, field, replace

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    import tomli as tomllib


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend_model: str
    description: str = ""
    # Extra fields merged into every backend request (e.g. to disable reasoning on thinking models).
    extra_body: dict = field(default_factory=lambda: {"reasoning_effort": "none"})
    # Ask yes/no checks in both answer orders and small picks in both option orders, then average.
    # Cancels position bias for models that favour the first-listed answer (costs 2x calls).
    debias: bool = False
    # "one_token"  - the answer is the first generated token (default; one token per check)
    # "deliberate" - let the model think, constrain only its final verdict token, read that token's
    #                logprobs. For models with no non-thinking mode, whose first token opens a
    #                scratchpad instead of answering. Costs think_budget tokens and seconds per check.
    # "auto"       - probe the backend once and pick
    mode: str = "one_token"
    think_budget: int = 512   # max tokens the model may think for in deliberate mode
    # How long the model may think in deliberate mode, if the backend supports it ("low" / "high" /
    # "max" on GLM-style models). Sent as reasoning_effort. More thinking buys stability: on our
    # benchmark GLM went 94.2% / 4.6% flips at "low" to 97.5% / 0.8% flips at the default.
    effort: str | None = None


@dataclass(frozen=True)
class Settings:
    backend_url: str = "http://localhost:8000"
    backend_api_key: str | None = None
    models: dict[str, ModelSpec] = field(default_factory=dict)
    default_model: str | None = None
    max_concurrency: int = 16          # simultaneous calls to the backend
    timeout_s: float = 30.0            # per backend call
    retries: int = 4                   # on 429 / 5xx / timeouts
    group_size: int = 15               # options per call for big picks (<= backend top_logprobs cap)
    max_top_logprobs: int = 20         # vLLM's default --max-logprobs
    api_keys: frozenset[str] = frozenset()  # empty = no auth (only allowed on localhost)

    def resolve(self, name: str | None) -> ModelSpec | None:
        return self.models.get(name or self.default_model or "")


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    raw: dict = {}
    path = path or os.environ.get("HUNCH_CONFIG") or ("hunch.toml" if pathlib.Path("hunch.toml").exists() else None)
    if path:
        raw = tomllib.loads(pathlib.Path(path).read_text())

    models = {name: ModelSpec(name=name, **spec) for name, spec in raw.get("models", {}).items()}
    # `effort` is sugar for extra_body.reasoning_effort, so one place decides what is sent
    models = {n: (replace(m, extra_body={**m.extra_body, "reasoning_effort": m.effort}) if m.effort else m)
              for n, m in models.items()}
    if not models and os.environ.get("HUNCH_BACKEND_MODEL"):
        models = {"default": ModelSpec(name="default", backend_model=os.environ["HUNCH_BACKEND_MODEL"],
                                       debias=os.environ.get("HUNCH_DEBIAS", "0") == "1")}
    svc = raw.get("service", {})
    backend = raw.get("backend", {})
    default_model = os.environ.get("HUNCH_DEFAULT_MODEL", svc.get("default_model")) or (next(iter(models)) if models else None)
    if default_model and default_model not in models:
        raise ValueError(f"default_model {default_model!r} is not one of the configured models {sorted(models)}")
    keys = os.environ.get("HUNCH_API_KEYS", ",".join(svc.get("api_keys", [])))
    group_size = int(svc.get("group_size", 15))
    max_top = int(backend.get("max_top_logprobs", 20))
    return Settings(
        backend_url=os.environ.get("HUNCH_BACKEND_URL", backend.get("url", "http://localhost:8000")).rstrip("/"),
        backend_api_key=os.environ.get("HUNCH_BACKEND_KEY", backend.get("api_key")),
        models=models,
        default_model=default_model,
        max_concurrency=int(os.environ.get("HUNCH_CONCURRENCY", svc.get("max_concurrency", 16))),
        timeout_s=float(svc.get("timeout_s", 30.0)),
        retries=int(svc.get("retries", 4)),
        group_size=min(group_size, max_top),
        max_top_logprobs=max_top,
        api_keys=frozenset(k.strip() for k in keys.split(",") if k.strip()),
    )
