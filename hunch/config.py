"""Settings: the backend (any vLLM OpenAI-compatible server), the models to expose, and service limits.

Read from a TOML file (HUNCH_CONFIG, else ./hunch.toml if present), with environment overrides.
Quickest setup without a file: HUNCH_BACKEND_URL + HUNCH_BACKEND_MODEL (exposed as model "default").
"""
from __future__ import annotations

import json
import os
import sys
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
    # mode "sample" only (backends without token logprobs): votes per check, and their temperature.
    # The probability moves in 1/samples steps; each vote costs a full prompt.
    samples: int = 7
    sample_temperature: float = 1.0
    # False for backends that return logprobs but can't constrain the answer (hosted APIs that ignore
    # structured_outputs). The answer is then read from the first token's top_logprobs: only the allowed
    # labels count, renormalised. A call whose labels hold less than min_label_mass of the probability
    # raises `off_label` instead of guessing. one_token and deliberate modes only.
    constrained: bool = True
    min_label_mass: float = 0.5
    # extract: cap on generated tokens for the JSON answer (thinking tokens come on top, from think_budget)
    extract_max_tokens: int = 1024


@dataclass(frozen=True)
class Settings:
    backend_url: str = "http://localhost:8000"
    backend_api_key: str | None = None
    models: dict[str, ModelSpec] = field(default_factory=dict)
    default_model: str | None = None
    max_concurrency: int = 16          # simultaneous calls to the backend (4 by default on hosted APIs)
    timeout_s: float = 30.0            # per backend call
    slow_timeout_s: float = 90.0       # per call that may think first (deliberate, sample, extract)
    retries: int = 4                   # on 429 / 5xx / timeouts
    group_size: int = 15               # options per call for big picks (<= backend top_logprobs cap)
    max_top_logprobs: int = 20         # vLLM's default --max-logprobs
    api_keys: frozenset[str] = frozenset()  # empty = no auth (only allowed on localhost)
    chat_url: str | None = None        # full chat-completions URL, for servers not at {backend_url}/v1/...
    key_header: str = "Authorization"  # header for backend_api_key: Bearer, or the bare key in any other header

    @property
    def completions_url(self) -> str:
        from .hosts import chat_url
        return chat_url(self.backend_url, self.chat_url)

    @property
    def backend_headers(self) -> dict:
        from .hosts import headers
        return headers(self.backend_api_key, self.key_header, self.backend_url)

    def resolve(self, name: str | None) -> ModelSpec | None:
        return self.models.get(name or self.default_model or "")


def _warn_ignored_mode_env(models: dict[str, "ModelSpec"], path: str | os.PathLike | None) -> None:
    """HUNCH_MODE / HUNCH_EFFORT only apply when a model is DISCOVERED from the environment, because `mode`
    is per model in a config file. Say so out loud: silently ignoring them once cost a caller an hour, and our
    own error message recommends HUNCH_MODE."""
    setvars = [v for v in ("HUNCH_MODE", "HUNCH_EFFORT") if os.environ.get(v)]
    if setvars and models:
        print(f"hunch: ignoring {' and '.join(setvars)} because {path or 'a hunch.toml'} configures "
              f"{len(models)} model(s). `mode` and `effort` are per model there; set them in the file "
              f"(HUNCH_MODE/HUNCH_EFFORT apply only when no config file declares models).", file=sys.stderr)
    elif setvars:
        pass  # no models declared here; discover_settings() will use them


def default_concurrency(url: str, models: "dict[str, ModelSpec] | None" = None) -> int:
    """16 for a server on this machine or LAN. On a hosted API, 4 checks at a time, where 16 parallel calls mostly
    buy 429s. A check in mode "sample" is `samples` calls, and they all go out at once so that voting costs tokens
    but not latency: 4 checks x samples calls."""
    from .hosts import is_hosted
    if not is_hosted(url):
        return 16
    votes = max((m.samples for m in (models or {}).values() if m.mode == "sample"), default=1)
    return 4 * max(1, votes)


def _truthy(v: str) -> bool:
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def env_model_options() -> dict:
    """ModelSpec fields set from the environment, for a model that is not declared in a hunch.toml:
      HUNCH_CONSTRAINED   0 = the backend returns logprobs but ignores structured_outputs (hosted APIs)
      HUNCH_EXTRA_BODY    a JSON object sent with every request, replacing the default extra_body
      HUNCH_SAMPLES       votes per check in mode "sample"
    `python -m hunch probe` prints the values to use for a backend."""
    out: dict = {}
    if os.environ.get("HUNCH_CONSTRAINED"):
        out["constrained"] = _truthy(os.environ["HUNCH_CONSTRAINED"])
    if os.environ.get("HUNCH_EXTRA_BODY"):
        try:
            extra = json.loads(os.environ["HUNCH_EXTRA_BODY"])
        except json.JSONDecodeError as e:
            raise ValueError(f"HUNCH_EXTRA_BODY is not JSON: {e}") from e
        if not isinstance(extra, dict):
            raise ValueError("HUNCH_EXTRA_BODY must be a JSON object")
        out["extra_body"] = extra
    if os.environ.get("HUNCH_SAMPLES"):
        out["samples"] = int(os.environ["HUNCH_SAMPLES"])
    return out


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
    _warn_ignored_mode_env(models, path)
    if not models and os.environ.get("HUNCH_BACKEND_MODEL"):
        # a model declared by environment alone: HUNCH_MODE / HUNCH_EFFORT apply here, as in discover_settings
        opts = env_model_options()
        if os.environ.get("HUNCH_MODE"):
            opts["mode"] = os.environ["HUNCH_MODE"]
        if os.environ.get("HUNCH_EFFORT"):
            opts["extra_body"] = {**opts.get("extra_body", {"reasoning_effort": "none"}),
                                  "reasoning_effort": os.environ["HUNCH_EFFORT"]}
        models = {"default": ModelSpec(name="default", backend_model=os.environ["HUNCH_BACKEND_MODEL"],
                                       debias=os.environ.get("HUNCH_DEBIAS", "0") == "1", **opts)}
    svc = raw.get("service", {})
    backend = raw.get("backend", {})
    default_model = os.environ.get("HUNCH_DEFAULT_MODEL", svc.get("default_model")) or (next(iter(models)) if models else None)
    if default_model and default_model not in models:
        raise ValueError(f"default_model {default_model!r} is not one of the configured models {sorted(models)}")
    keys = os.environ.get("HUNCH_API_KEYS", ",".join(svc.get("api_keys", [])))
    group_size = int(svc.get("group_size", 15))
    max_top = int(os.environ.get("HUNCH_MAX_TOP_LOGPROBS") or backend.get("max_top_logprobs", 20))
    # HUNCH_BACKEND_* > hunch.toml > the agent's own LLM variables > localhost
    url = api_root(os.environ.get("HUNCH_BACKEND_URL") or backend.get("url") or _first(URL_VARS) or "http://localhost:8000")
    return Settings(
        backend_url=url,
        backend_api_key=os.environ.get("HUNCH_BACKEND_KEY") or backend.get("api_key") or _first(KEY_VARS),
        models=models,
        default_model=default_model,
        max_concurrency=int(os.environ.get("HUNCH_CONCURRENCY") or svc.get("max_concurrency") or default_concurrency(url, models)),
        timeout_s=float(os.environ.get("HUNCH_TIMEOUT") or svc.get("timeout_s", 30.0)),
        slow_timeout_s=float(os.environ.get("HUNCH_SLOW_TIMEOUT") or svc.get("slow_timeout_s", 90.0)),
        chat_url=os.environ.get("HUNCH_CHAT_URL") or backend.get("chat_url"),
        key_header=os.environ.get("HUNCH_KEY_HEADER") or backend.get("key_header", "Authorization"),
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
# Hosted APIs that give no token logprobs under a constrained choice. Hunch runs them in mode "sample":
# several votes per check instead of one read of the logprobs.
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

    from .hosts import headers, model_ids, models_url
    r = httpx.get(models_url(url, os.environ.get("HUNCH_CHAT_URL")),
                  headers=headers(key, os.environ.get("HUNCH_KEY_HEADER") or "Authorization", url), timeout=timeout)
    r.raise_for_status()
    return model_ids(r.json())


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
    no_logprobs = any(h in url for h in NO_LOGPROBS_HOSTS)
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
    mode = os.environ.get("HUNCH_MODE") or ("sample" if no_logprobs else "auto")
    # a hosted API may refuse vLLM's fields; in sample mode send only what the caller asked for
    extra = ({"reasoning_effort": effort} if effort else {}) if mode == "sample" else {"reasoning_effort": effort or "none"}
    opts = {"extra_body": extra, **env_model_options()}
    spec = ModelSpec(name=os.environ.get("HUNCH_MODEL_NAME", "default"), backend_model=backend_model, mode=mode, **opts)
    max_top = int(os.environ.get("HUNCH_MAX_TOP_LOGPROBS") or 20)
    return Settings(backend_url=url, backend_api_key=key, models={spec.name: spec}, default_model=spec.name,
                    max_concurrency=int(os.environ.get("HUNCH_CONCURRENCY") or default_concurrency(url, {spec.name: spec})),
                    timeout_s=float(os.environ.get("HUNCH_TIMEOUT") or 30.0),
                    slow_timeout_s=float(os.environ.get("HUNCH_SLOW_TIMEOUT") or 90.0),
                    group_size=min(15, max_top), max_top_logprobs=max_top,
                    chat_url=os.environ.get("HUNCH_CHAT_URL") or None,
                    key_header=os.environ.get("HUNCH_KEY_HEADER") or "Authorization",
                    api_keys=frozenset(k.strip() for k in os.environ.get("HUNCH_API_KEYS", "").split(",") if k.strip()))
