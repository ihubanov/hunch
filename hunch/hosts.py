"""What a hosted API needs that a self-hosted vLLM does not. This covers where to POST, how to send the key,
and which request fields each host accepts.

Every backend call goes through this module (checks, probes, selftest, qualify), so they all send the same
request. The quirks below exist only so that Hunch's own parameters are honoured. Anything else is the
caller's business (extra_body).
"""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

# Fields Hunch may drop when a backend rejects them by name (HTTP 400/422). Dropping one changes nothing a
# caller relies on, except structured_outputs: when that goes, answers are read the unconstrained way, with the
# min_label_mass guard. logprobs / top_logprobs / max_tokens are never dropped: without them there is no answer
# to read. Longest names first, so "reasoning_effort" in an error is not read as "reasoning".
DROPPABLE = ("structured_outputs", "reasoning_effort", "temperature", "reasoning", "top_p", "top_k")

# Values of reasoning_effort that switch thinking off (shared with the engine)
THINKING_OFF_EFFORTS = ("none", "off", "disable", "disabled")


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _is(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def is_hosted(url: str) -> bool:
    """A public API on the internet rather than a server on this machine or LAN. It only sets defaults
    (concurrency), so a wrong guess costs speed, never correctness."""
    h = host_of(url)
    if not h or h == "localhost" or "." not in h or h.endswith((".local", ".lan", ".internal", ".home", ".arpa")):
        return False
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return True
    return not (ip.is_private or ip.is_loopback or ip.is_link_local)


def chat_url(backend_url: str, override: str | None = None) -> str:
    """The chat-completions endpoint. Most servers serve {root}/v1/chat/completions, so that is the default.
    Google's OpenAI-compatible API lives under .../openai/chat/completions with no /v1, which is why a URL
    ending in /openai gets the bare path. Anything else unusual (such as Azure) is set with HUNCH_CHAT_URL."""
    if override:
        return override
    root = backend_url.rstrip("/")
    return f"{root}/chat/completions" if root.endswith("/openai") else f"{root}/v1/chat/completions"


def models_url(backend_url: str, override: str | None = None) -> str:
    """The model list that sits next to the chat endpoint."""
    url = chat_url(backend_url, override)
    return url[: -len("chat/completions")] + "models" if url.endswith("chat/completions") else f"{backend_url.rstrip('/')}/v1/models"


def headers(key: str | None, header: str = "Authorization", url: str = "") -> dict:
    """Bearer token by default. Any other header name (Azure's `api-key`) carries the bare key.
    Anthropic's chat-completions layer takes the Bearer token, but its native /v1/models wants x-api-key
    and a version header, so both are sent there."""
    if not key:
        return {}
    if header.lower() != "authorization":
        return {header: key}
    out = {"Authorization": f"Bearer {key}"}
    if _is(host_of(url), "anthropic.com"):
        out.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
    return out


def model_ids(payload) -> list[str]:
    """Model ids from a /models response: {"data": [...]} (OpenAI style) or a bare list (Together)."""
    items = payload.get("data", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    return [m["id"] for m in items if isinstance(m, dict) and "id" in m]


def adapt(url: str, body: dict) -> dict:
    """The request as this host needs it. A copy is returned; `body` is not changed."""
    body = dict(body)
    h = host_of(url)
    if _is(h, "openrouter.ai"):
        # Without require_parameters OpenRouter silently routes to providers that drop logprobs / top_logprobs.
        provider = dict(body.get("provider") or {})
        provider.setdefault("require_parameters", True)
        body["provider"] = provider
        # OpenRouter's own switch is `reasoning`. It maps reasoning_effort for some models but not to "off".
        effort = body.get("reasoning_effort")
        if isinstance(effort, str) and "reasoning" not in body:
            body.pop("reasoning_effort")
            body["reasoning"] = ({"enabled": False} if effort.lower() in THINKING_OFF_EFFORTS
                                 else {"effort": effort})
    elif _is(h, "anthropic.com"):
        # Current Claude API versions refuse sampling parameters on every model. Sample mode gets its spread
        # from the default temperature (1.0).
        for k in ("temperature", "top_p", "top_k"):
            body.pop(k, None)
    elif _is(h, "openai.com"):
        # max_tokens is refused by reasoning models and deprecated on the rest; max_completion_tokens works on all.
        if "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
    return body


def rejected_field(status: int, text: str, body: dict) -> str | None:
    """The droppable field a 400/422 refuses by name, if any."""
    if status not in (400, 422):
        return None
    t = text.lower()
    for f in DROPPABLE:
        if f in body and re.search(rf"\b{f}\b", t):
            return f
    return None


def requires_streaming(status: int, text: str) -> bool:
    """The backend refuses non-streaming calls (Together's passthrough models: code streaming_required)."""
    t = text.lower()
    return status in (400, 422) and ("streaming_required" in t or "only supports streaming" in t)


def collect_stream(text: str) -> dict:
    """An SSE chat-completions stream folded back into the non-streaming response shape Hunch reads:
    choices[0].message.{content, reasoning}, choices[0].logprobs.content, usage."""
    import json
    content, reasoning, tokens, usage, finish = [], [], [], None, None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if chunk.get("error"):
            return {"error": chunk["error"]}
        usage = chunk.get("usage") or usage
        for ch in chunk.get("choices") or []:
            delta = ch.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
            tokens.extend((ch.get("logprobs") or {}).get("content") or [])
            finish = ch.get("finish_reason") or finish
    message = {"role": "assistant", "content": "".join(content)}
    if any(reasoning):
        message["reasoning"] = "".join(reasoning)
    choice = {"index": 0, "message": message, "finish_reason": finish}
    if tokens:
        choice["logprobs"] = {"content": tokens}
    return {"choices": [choice], "usage": usage or {}}


def needs_more_tokens(status: int, text: str) -> bool:
    """The backend refuses a max_tokens this small outright (OpenAI's GPT-5 family at 1 token: "Could not finish the
    message because max_tokens or model output limit was reached")."""
    return status in (400, 422) and "could not finish the message because max_tokens" in text.lower()


MIN_TOKENS_RETRY = 16   # what a one-token call is raised to; Hunch still reads only the first token
