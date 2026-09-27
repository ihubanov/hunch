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
    # Hard cap, not a knob: it does not shorten thinking, it only cuts it off, and a cut-off call
    # raises rather than guessing. Measured on GLM at default effort: median 128, p95 671, max 3579.
    # With `effort` set the tail collapses (low: max 59, high: max 138), so 256 is plenty there.
    think_budget: int = 2048
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
    for n, m in models.items():
        if m.effort and m.mode == "one_token":
            # effort turns thinking ON, and a one_token model must answer in its first token
            raise ValueError(f"model {n!r}: effort={m.effort!r} needs thinking, but mode is one_token; "
                             "set mode = \"deliberate\" (or \"auto\"), or remove effort")
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
        # HUNCH_BACKEND_* > hunch.toml > the agent's own LLM variables > localhost
        backend_url=api_root(os.environ.get("HUNCH_BACKEND_URL") or backend.get("url")
                             or _first(URL_VARS) or "http://localhost:8000"),
        backend_api_key=os.environ.get("HUNCH_BACKEND_KEY") or backend.get("api_key") or _first(KEY_VARS),
        models=models,
        default_model=default_model,
        max_concurrency=int(os.environ.get("HUNCH_CONCURRENCY", svc.get("max_concurrency", 16))),
        timeout_s=float(svc.get("timeout_s", 30.0)),
        retries=int(svc.get("retries", 4)),
        group_size=min(group_size, max_top),
        max_top_logprobs=max_top,
        api_keys=frozenset(k.strip() for k in keys.split(",") if k.strip()),
    )


# ---------------------------------------------------------------- zero-config: the agent's own LLM settings
# First set variable wins. ANTHROPIC_* are what Claude Code (and forks pointed at a local gateway) use for
# their own LLM; ANTHROPIC_MODEL is the main model, never the small/fast one, which on our benchmark opened
# a scratchpad instead of answering.
URL_VARS = ("HUNCH_BACKEND_URL", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "LLM_BASE_URL")
KEY_VARS = ("HUNCH_BACKEND_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "LLM_API_KEY")
MODEL_VARS = ("HUNCH_BACKEND_MODEL", "ANTHROPIC_MODEL", "OPENAI_MODEL", "LLM_MODEL")
# Hosted APIs that give no token logprobs under a constrained choice: Hunch cannot read an answer there.
NO_LOGPROBS_HOSTS = ("api.anthropic.com",)


class DiscoveryError(ValueError):
    pass


def _first(names: tuple[str, ...]) -> str | None:
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def api_root(url: str) -> str:
    """OpenAI/Anthropic clients are often configured with .../v1; Hunch appends /v1/... itself."""
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def served_models(url: str, key: str | None, timeout: float = 15.0) -> list[str]:
    import httpx
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = httpx.get(f"{url}/v1/models", headers=headers, timeout=timeout)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


def discover_settings(base_url: str | None = None, api_key: str | None = None,
                      model: str | None = None) -> Settings:
    """Settings from the caller's own LLM configuration, for use without a hunch.toml.

    A missing key is normal (local gateways often take none). With no model named anywhere, the
    endpoint's only served model is used; several served models is an error that lists them.
    mode is "auto": one probe per model decides one-token or deliberate.
    """
    url = base_url or _first(URL_VARS)
    if not url:
        raise DiscoveryError("no LLM endpoint: set HUNCH_BACKEND_URL (or ANTHROPIC_BASE_URL / OPENAI_BASE_URL)")
    url = api_root(url)
    if any(h in url for h in NO_LOGPROBS_HOSTS):
        raise DiscoveryError(f"{url} returns no token logprobs, which Hunch reads its answers from; "
                             "point HUNCH_BACKEND_URL at an OpenAI-compatible vLLM server instead")
    key = api_key if api_key is not None else _first(KEY_VARS)
    backend_model = model or _first(MODEL_VARS)
    if not backend_model:
        try:
            served = served_models(url, key)
        except Exception as e:  # noqa: BLE001
            raise DiscoveryError(f"cannot list models at {url}: {e}") from e
        if len(served) != 1:
            raise DiscoveryError(f"{url} serves {len(served)} models; set HUNCH_BACKEND_MODEL "
                                 f"(or ANTHROPIC_MODEL / OPENAI_MODEL) to one of {served}")
        backend_model = served[0]
    effort = os.environ.get("HUNCH_EFFORT")
    spec = ModelSpec(name=os.environ.get("HUNCH_MODEL_NAME", "default"), backend_model=backend_model,
                     extra_body={"reasoning_effort": effort or "none"},
                     mode=os.environ.get("HUNCH_MODE", "auto"))
    return Settings(backend_url=url, backend_api_key=key, models={spec.name: spec}, default_model=spec.name,
                    max_concurrency=int(os.environ.get("HUNCH_CONCURRENCY", "16")),
                    api_keys=frozenset(k.strip() for k in os.environ.get("HUNCH_API_KEYS", "").split(",") if k.strip()))
