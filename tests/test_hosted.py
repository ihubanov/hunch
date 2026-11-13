"""Hosted APIs: host quirks, refused fields, env settings and the capability probe (fake backends only)."""
import asyncio
import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from hunch import hosts
from hunch.config import ModelSpec, Settings
from hunch.engine import Engine
from hunch.probe import SCHEMA, aprobe_settings
from hunch.server import create_app

from test_api import ENV_VARS, fake_backend, judge


def S(url="http://backend.test", **kw):
    spec_kw = kw.pop("spec", {})
    return Settings(backend_url=url, default_model="m", retries=kw.pop("retries", 1),
                    models={"m": ModelSpec(name="m", backend_model="org/m", **spec_kw)}, **kw)


def hosted_backend(*, logprobs=True, refuse=(), enforce=False, think=False, answer="A", seen=None):
    """A configurable OpenAI-style backend.
    logprobs: returns first-token logprobs. refuse: field names answered with a 400 naming them.
    enforce: honours structured_outputs. think: opens a scratchpad on the 'yes' probe."""
    seen = seen if seen is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append({"url": str(request.url), "body": body, "headers": dict(request.headers),
                     "timeout": request.extensions.get("timeout", {}).get("read")})
        for f in refuse:
            if f in body:
                return httpx.Response(400, json={"error": {"message": f"Unrecognized request argument supplied: {f}"}})
        prompt = body["messages"][-1]["content"]
        prompt = prompt if isinstance(prompt, str) else " ".join(p.get("text", "") for p in prompt)
        so = (body.get("structured_outputs") or {}).get("choice") if enforce else None
        if "one word: yes" in prompt:
            text = "Let me think about this" if think else "yes"
        elif "hello" in prompt:
            text = so[0] if so else "hello"
        elif so:
            text = so[0]
        else:
            text = answer
        out = {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
        if logprobs and body.get("logprobs"):
            labels = so or ["A", "B", "Y", "N"]
            top = [{"token": text, "logprob": math.log(0.8)}] + \
                  [{"token": lab, "logprob": math.log(0.05)} for lab in labels if lab != text][:3]
            out["choices"][0]["logprobs"] = {"content": [{"token": text, "logprob": math.log(0.8), "top_logprobs": top}]}
        return httpx.Response(200, json=out)
    return httpx.MockTransport(handler), seen


def probe(settings, transport):
    return asyncio.run(aprobe_settings(settings, settings.resolve(None), transport))


# ---------------------------------------------------------------- hosts
def test_openrouter_requires_parameters_and_switches_reasoning_off_its_own_way():
    b = hosts.adapt("https://openrouter.ai/api/v1/chat/completions",
                    {"model": "x", "reasoning_effort": "none", "provider": {"order": ["a"]}})
    assert b["provider"] == {"order": ["a"], "require_parameters": True}
    assert b["reasoning"] == {"enabled": False} and "reasoning_effort" not in b
    b = hosts.adapt("https://openrouter.ai/api/v1/chat/completions",
                    {"reasoning_effort": "high", "provider": {"require_parameters": False}})
    assert b["reasoning"] == {"effort": "high"} and b["provider"]["require_parameters"] is False   # caller wins


def test_anthropic_gets_no_sampling_params_and_openai_gets_max_completion_tokens():
    b = hosts.adapt("https://api.anthropic.com/v1/chat/completions", {"temperature": 1.0, "top_p": 1, "max_tokens": 9})
    assert b == {"max_tokens": 9}
    b = hosts.adapt("https://api.openai.com/v1/chat/completions", {"max_tokens": 1, "temperature": 0})
    assert b == {"max_completion_tokens": 1, "temperature": 0}
    body = {"max_tokens": 1}
    hosts.adapt("https://api.openai.com/v1/chat/completions", body)
    assert body == {"max_tokens": 1}                       # the caller's body is not changed
    assert hosts.adapt("http://localhost:8000/v1/chat/completions", {"max_tokens": 1, "temperature": 0}) == \
        {"max_tokens": 1, "temperature": 0}


def test_chat_url_models_url_and_key_header():
    assert hosts.chat_url("http://x:8000") == "http://x:8000/v1/chat/completions"
    g = "https://generativelanguage.googleapis.com/v1beta/openai"
    assert hosts.chat_url(g) == g + "/chat/completions" and hosts.models_url(g) == g + "/models"
    az = "https://r.openai.azure.com/openai/deployments/d/chat/completions?api-version=2024-10-21"
    assert hosts.chat_url("https://ignored", az) == az
    assert hosts.headers("k") == {"Authorization": "Bearer k"}
    assert hosts.headers("k", "api-key") == {"api-key": "k"} and hosts.headers(None) == {}


@pytest.mark.parametrize("url,hosted", [("https://openrouter.ai/api", True), ("http://localhost:8000", False),
                                        ("http://127.0.0.1:8000", False), ("http://192.168.1.20:8000", False),
                                        ("http://gpu-box:8000", False), ("http://llm.lan:80", False),
                                        ("https://api.together.xyz", True)])
def test_hosted_backends_get_a_lower_default_concurrency(url, hosted):
    from hunch.config import default_concurrency
    assert hosts.is_hosted(url) is hosted and default_concurrency(url) == (4 if hosted else 16)


def test_rejected_field_reads_the_name_not_a_prefix():
    body = {"reasoning": {}, "temperature": 0}
    assert hosts.rejected_field(400, "Unsupported parameter: 'reasoning_effort'", body) is None
    assert hosts.rejected_field(400, "'temperature' does not support 0 with this model", body) == "temperature"
    assert hosts.rejected_field(500, "temperature", body) is None
    assert hosts.rejected_field(400, "messages too long", body) is None


# ---------------------------------------------------------------- engine
def test_a_refused_constraint_falls_back_to_reading_top_logprobs_and_is_remembered():
    transport, seen = hosted_backend(refuse=("structured_outputs",), answer="Y")
    with TestClient(create_app(S(), transport=transport)) as c:
        r1 = judge(c, {"q": {"kind": "yesno", "question": "x"}})
        r2 = judge(c, {"q": {"kind": "yesno", "question": "x"}})
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["results"]["q"]["p_yes"] == pytest.approx(0.8 / 0.85, abs=1e-3)
    sent = [s["body"] for s in seen]
    assert "structured_outputs" in sent[0] and all("structured_outputs" not in b for b in sent[1:])
    assert len(seen) == 3                              # one refusal, then one call per judge


def test_a_refused_reasoning_effort_is_dropped_and_the_constraint_kept():
    transport, seen = hosted_backend(refuse=("reasoning_effort",), enforce=True)
    with TestClient(create_app(S(), transport=transport)) as c:
        r = judge(c, {"q": {"kind": "yesno", "question": "x"}})
    assert r.status_code == 200
    assert "structured_outputs" in seen[-1]["body"] and "reasoning_effort" not in seen[-1]["body"]


def test_calls_that_may_think_get_the_slow_timeout():
    transport, seen = hosted_backend(enforce=True)
    s = S(timeout_s=30, slow_timeout_s=90, spec={"mode": "deliberate"})
    with TestClient(create_app(s, transport=transport)) as c:
        judge(c, {"q": {"kind": "yesno", "question": "x"}})
    with TestClient(create_app(S(timeout_s=30, slow_timeout_s=90), transport=transport)) as c:
        judge(c, {"q": {"kind": "yesno", "question": "x"}})
    assert seen[0]["timeout"] == 90 and seen[-1]["timeout"] == 30


def test_custom_chat_url_and_key_header_are_used_for_every_call():
    transport, seen = hosted_backend(enforce=True)
    s = S(url="https://ignored.test", chat_url="https://az.test/openai/deployments/d/chat/completions?api-version=1",
          backend_api_key="sekret", key_header="api-key")
    with TestClient(create_app(s, transport=transport)) as c:
        judge(c, {"q": {"kind": "yesno", "question": "x"}})
    assert seen[0]["url"].startswith("https://az.test/openai/deployments/d/chat/completions")
    assert seen[0]["headers"].get("api-key") == "sekret" and "authorization" not in seen[0]["headers"]


# ---------------------------------------------------------------- env
@pytest.fixture
def env(monkeypatch, tmp_path):
    for v in ENV_VARS + ("HUNCH_CONSTRAINED", "HUNCH_EXTRA_BODY", "HUNCH_MAX_TOP_LOGPROBS", "HUNCH_CHAT_URL",
                         "HUNCH_KEY_HEADER", "HUNCH_CONCURRENCY", "HUNCH_TIMEOUT", "HUNCH_SLOW_TIMEOUT"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.chdir(tmp_path)
    return monkeypatch


def test_zero_config_reads_the_hosted_settings_from_env(env):
    from hunch.config import discover_settings, load_settings
    env.setenv("HUNCH_BACKEND_URL", "https://openrouter.ai/api/v1")
    env.setenv("HUNCH_BACKEND_MODEL", "google/gemma")
    env.setenv("HUNCH_CONSTRAINED", "0")
    env.setenv("HUNCH_EXTRA_BODY", '{"reasoning": {"enabled": false}}')
    env.setenv("HUNCH_MAX_TOP_LOGPROBS", "5")
    env.setenv("HUNCH_KEY_HEADER", "x-key")
    for s in (discover_settings(), load_settings()):
        spec = s.resolve(None)
        assert spec.constrained is False and spec.extra_body == {"reasoning": {"enabled": False}}
        assert (s.max_top_logprobs, s.group_size, s.max_concurrency, s.key_header) == (5, 5, 4, "x-key")
        assert s.backend_url == "https://openrouter.ai/api"
    env.setenv("HUNCH_EXTRA_BODY", "[1]")
    with pytest.raises(ValueError, match="JSON object"):
        discover_settings()


def test_local_zero_config_keeps_its_defaults(env):
    from hunch.config import discover_settings
    env.setenv("HUNCH_BACKEND_URL", "http://localhost:8000")
    env.setenv("HUNCH_BACKEND_MODEL", "org/m")
    s = discover_settings()
    spec = s.resolve(None)
    assert (spec.constrained, spec.extra_body, s.max_concurrency, s.max_top_logprobs) == \
        (True, {"reasoning_effort": "none"}, 16, 20)


# ---------------------------------------------------------------- probe
def test_probe_self_hosted_vllm_is_one_token_constrained():
    transport, _ = fake_backend()
    r = probe(S(retries=0), transport)
    assert r["schema"] == SCHEMA and r["mode"] == "one_token" and r["constrained"] is True
    assert r["env"]["HUNCH_MODE"] == "one_token" and r["env"]["HUNCH_CONSTRAINED"] == "1"
    assert r["capabilities"]["constraint_enforced"] is True and r["capabilities"]["thinks_first"] is False


def test_probe_hosted_logprobs_without_constraint_is_one_token_unconstrained():
    transport, seen = hosted_backend()
    r = probe(S(), transport)
    assert (r["mode"], r["constrained"], r["max_top_logprobs"]) == ("one_token", False, 20)
    assert r["env"] == {"HUNCH_MODE": "one_token", "HUNCH_CONSTRAINED": "0", "HUNCH_MAX_TOP_LOGPROBS": "20"}
    assert r["calls"] == len(seen) and r["calls"] <= 4


def test_probe_refused_constraint_and_effort_are_reported_and_left_out():
    transport, _ = hosted_backend(refuse=("structured_outputs", "reasoning_effort"))
    r = probe(S(), transport)
    assert r["mode"] == "one_token" and r["constrained"] is False
    assert r["capabilities"]["structured_outputs_refused"] is True
    assert r["dropped_fields"] == ["reasoning_effort", "structured_outputs"]
    assert json.loads(r["env"]["HUNCH_EXTRA_BODY"]) == {}


def test_probe_no_logprobs_is_sample():
    transport, _ = hosted_backend(logprobs=False)
    r = probe(S(retries=0), transport)
    assert r["mode"] == "sample" and r["capabilities"]["votes_answer"] is True
    assert r["env"] == {"HUNCH_MODE": "sample", "HUNCH_CONSTRAINED": "0"} and r["max_top_logprobs"] is None


def test_probe_thinking_model_with_logprobs_is_deliberate():
    transport, _ = hosted_backend(enforce=True, think=True)
    r = probe(S(), transport)
    assert (r["mode"], r["constrained"]) == ("deliberate", True)
    transport, _ = hosted_backend(think=True)
    r = probe(S(), transport)
    assert (r["mode"], r["constrained"]) == ("deliberate", False) and "qualify" in r["reason"]
    transport, _ = hosted_backend(think=True, logprobs=False)
    assert probe(S(retries=0), transport)["mode"] == "sample"


def test_probe_a_backend_that_never_names_a_label_is_unsupported():
    transport, _ = hosted_backend(logprobs=False, answer="I cannot help with that")
    r = probe(S(retries=0), transport)
    assert r["mode"] == "unsupported" and r["env"] == {}


def test_probe_lowers_top_logprobs_to_a_hosts_cap():
    transport, seen = hosted_backend()

    def handler(request):
        if json.loads(request.content).get("top_logprobs", 0) > 5:
            return httpx.Response(400, json={"error": {"message": "top_logprobs must be <= 5"}})
        return transport.handle_request(request)
    r = probe(S(), httpx.MockTransport(handler))
    assert r["max_top_logprobs"] == 5 and r["env"]["HUNCH_MAX_TOP_LOGPROBS"] == "5"


def test_probe_over_http(env):
    transport, _ = hosted_backend()
    with TestClient(create_app(S(), transport=transport)) as c:
        r = c.get("/v1/probe")
        assert r.status_code == 200 and r.json()["mode"] == "one_token"
        assert c.get("/v1/probe", params={"model": "nope"}).status_code == 400


def test_model_lists_in_both_shapes_and_anthropic_headers():
    assert hosts.model_ids({"data": [{"id": "a"}]}) == ["a"]
    assert hosts.model_ids([{"id": "b", "type": "chat"}]) == ["b"]          # Together returns a bare list
    assert hosts.model_ids({"error": "x"}) == [] and hosts.model_ids(None) == []
    h = hosts.headers("k", url="https://api.anthropic.com")
    assert h["x-api-key"] == "k" and h["anthropic-version"] and h["Authorization"] == "Bearer k"
    assert "x-api-key" not in hosts.headers("k", url="https://openrouter.ai/api")


def test_env_declared_model_honours_hunch_mode_for_server_and_qualify(env, capsys):
    from hunch.config import load_settings
    env.setenv("HUNCH_BACKEND_URL", "https://api.anthropic.com/v1")
    env.setenv("HUNCH_BACKEND_MODEL", "claude-x")
    env.setenv("HUNCH_MODE", "sample")
    env.setenv("HUNCH_EXTRA_BODY", "{}")
    spec = load_settings().resolve(None)
    assert (spec.mode, spec.extra_body) == ("sample", {})
    assert "ignoring" not in capsys.readouterr().err


def test_sample_mode_votes_go_out_together_on_a_hosted_api(env):
    from hunch.config import default_concurrency, discover_settings
    assert default_concurrency("https://api.anthropic.com", {"m": ModelSpec(name="m", backend_model="c", mode="sample")}) == 28
    env.setenv("HUNCH_BACKEND_URL", "https://api.anthropic.com/v1")
    env.setenv("HUNCH_BACKEND_MODEL", "claude-x")
    assert discover_settings().max_concurrency == 28            # 4 checks x 7 votes, all in flight at once
    env.setenv("HUNCH_SAMPLES", "5")
    assert discover_settings().max_concurrency == 20


def test_votes_run_in_parallel_not_in_waves():
    import time
    inflight = {"now": 0, "max": 0}

    async def handler(request):
        inflight["now"] += 1
        inflight["max"] = max(inflight["max"], inflight["now"])
        await asyncio.sleep(0.05)
        inflight["now"] -= 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "Y"}}], "usage": {}})
    s = S(url="https://api.anthropic.com", max_concurrency=28, spec={"mode": "sample", "extra_body": {}})
    with TestClient(create_app(s, transport=httpx.MockTransport(handler))) as c:
        t0 = time.perf_counter()
        r = judge(c, {"a": {"kind": "yesno", "question": "x"}, "b": {"kind": "yesno", "question": "y"}})
        took = time.perf_counter() - t0
    assert r.status_code == 200 and inflight["max"] == 14 and took < 0.5    # 2 checks x 7 votes, one wave


def streaming_only_backend(seen):
    """Answers only stream=true calls, as SSE with logprobs (Together's passthrough models)."""
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if not body.get("stream"):
            return httpx.Response(400, json={"error": {"message": 'This model only supports streaming. Set "stream": true.',
                                                       "code": "streaming_required"}})
        top = [{"token": "Y", "logprob": math.log(0.75)}, {"token": "N", "logprob": math.log(0.2)}]
        chunks = [{"choices": [{"delta": {"role": "assistant", "content": ""}, "logprobs": {"content": []}}]},
                  {"choices": [{"delta": {"content": "Y"},
                                "logprobs": {"content": [{"token": "Y", "logprob": math.log(0.75), "top_logprobs": top}]}}]},
                  {"choices": [{"delta": {"content": ""}, "finish_reason": "length"}]},
                  {"choices": [], "usage": {"prompt_tokens": 41, "completion_tokens": 1}}]
        sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
    return httpx.MockTransport(handler)


def test_a_streaming_only_model_is_called_streamed_and_read_the_same():
    seen = []
    with TestClient(create_app(S(spec={"constrained": False}), transport=streaming_only_backend(seen))) as c:
        r1 = judge(c, {"q": {"kind": "yesno", "question": "x"}})
        r2 = judge(c, {"q": {"kind": "yesno", "question": "x"}})
    assert r1.status_code == 200 and r1.json()["results"]["q"]["p_yes"] == pytest.approx(0.75 / 0.95, abs=1e-3)
    assert r1.json()["usage"]["prompt_tokens"] == 41
    assert [bool(b.get("stream")) for b in seen] == [False, True, True]      # learnt once, then always streamed
    assert r2.status_code == 200


def test_collect_stream_folds_content_reasoning_and_logprobs():
    sse = ('data: {"choices":[{"delta":{"reasoning":"hm "}}]}\n\n'
           'data: {"choices":[{"delta":{"content":"Ye"},"logprobs":{"content":[{"token":"Ye","logprob":-0.1}]}}]}\n\n'
           'data: {"choices":[{"delta":{"content":"s"},"logprobs":{"content":[{"token":"s","logprob":-0.2}]}}]}\n\n'
           'data: {"choices":[],"usage":{"prompt_tokens":3}}\n\ndata: [DONE]\n\n')
    d = hosts.collect_stream(sse)
    assert d["choices"][0]["message"] == {"role": "assistant", "content": "Yes", "reasoning": "hm "}
    assert [t["token"] for t in d["choices"][0]["logprobs"]["content"]] == ["Ye", "s"] and d["usage"] == {"prompt_tokens": 3}


def test_votes_cut_off_by_the_token_cap_say_so():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                                         "usage": {"completion_tokens": 256}})
    s = S(url="https://api.anthropic.com", spec={"mode": "sample", "extra_body": {}, "think_budget": 256})
    with TestClient(create_app(s, transport=httpx.MockTransport(handler))) as c:
        r = judge(c, {"q": {"kind": "yesno", "question": "x"}})
    assert r.status_code == 502 and "raise think_budget" in r.json()["error"]["message"]


def test_a_one_off_off_label_answer_is_asked_once_more():
    calls = {"n": 0}
    good, _ = hosted_backend(answer="Y")

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            tok = {"token": ",", "logprob": -0.01, "top_logprobs": [{"token": ",", "logprob": -0.01}]}
            return httpx.Response(200, json={"choices": [{"message": {"content": ","}, "logprobs": {"content": [tok]}}]})
        return good.handle_request(request)
    with TestClient(create_app(S(spec={"constrained": False}), transport=httpx.MockTransport(handler))) as c:
        r = judge(c, {"q": {"kind": "yesno", "question": "x"}})
    assert r.status_code == 200 and calls["n"] == 2


def test_a_backend_that_refuses_one_token_gets_a_few_more_and_the_first_is_read():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body.get("max_completion_tokens"))
        if body.get("max_completion_tokens", 0) < 8:
            return httpx.Response(400, json={"error": {"message": "Could not finish the message because max_tokens or "
                                                                  "model output limit was reached. Please try again with higher max_tokens."}})
        top = [{"token": "Y", "logprob": math.log(0.6)}, {"token": "N", "logprob": math.log(0.4)}]
        return httpx.Response(200, json={"choices": [{"message": {"content": "Y"}, "finish_reason": "stop",
                                                      "logprobs": {"content": [{"token": "Y", "logprob": math.log(0.6), "top_logprobs": top}]}}]})
    s = S(url="https://api.openai.com", spec={"constrained": False, "extra_body": {}})
    with TestClient(create_app(s, transport=httpx.MockTransport(handler))) as c:
        r1 = judge(c, {"q": {"kind": "yesno", "question": "x"}})
        r2 = judge(c, {"q": {"kind": "yesno", "question": "x"}})
    assert r1.status_code == 200 and r1.json()["results"]["q"]["p_yes"] == pytest.approx(0.6)
    assert r2.status_code == 200 and seen == [1, 16, 16]          # learnt once
