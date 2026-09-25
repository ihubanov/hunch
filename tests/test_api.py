"""Offline tests against a fake OpenAI-compatible backend (httpx.MockTransport)."""
import json
import math
import re
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from hunch.config import ModelSpec, Settings
from hunch.engine import confidence
from hunch.server import create_app

STAR = "★"  # the fake backend prefers the option / level / group whose line contains this

SETTINGS = Settings(
    backend_url="http://backend.test",
    models={"fast": ModelSpec(name="fast", backend_model="org/fast-model"),
            "biased": ModelSpec(name="biased", backend_model="org/biased-model", debias=True)},
    default_model="fast",
)


def fake_backend(prose=False, fail_first=0, status=504, scratchpad=False, think_tokens=0, thinks_when_allowed=False):
    """scratchpad: the model's first unconstrained token opens a thinking block (no non-thinking mode).
    think_tokens: in deliberate calls, emit this many thinking tokens before the verdict.
    thinks_when_allowed: answers directly with thinking off, thinks otherwise (DeepSeek-style)."""
    state = {"fails": fail_first, "calls": 0, "bodies": [], "probes": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "org/fast-model"}, {"id": "org/biased-model"}]})
        state["calls"] += 1
        if state["fails"] > 0:
            state["fails"] -= 1
            return httpx.Response(status, text="try later", headers={"retry-after": "0"})
        body = json.loads(request.content)
        if "structured_outputs" not in body:          # the unconstrained scratchpad probe
            state["probes"] += 1
            # a model with no non-thinking mode leaks its scratchpad into content (vLLM #54744)
            content = "The user is asking me to reply with exactly one word" if scratchpad else "yes"
            if thinks_when_allowed and body.get("reasoning_effort") != "none":
                return httpx.Response(200, json={"choices": [{"message": {"content": "", "reasoning": "We need answer."}}],
                                                 "usage": {"prompt_tokens": 10, "completion_tokens": 16}})
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                             "usage": {"prompt_tokens": 10, "completion_tokens": 1}})
        state["bodies"].append(body)
        labels = body["structured_outputs"]["choice"]
        user = body["messages"][-1]["content"]
        if labels == ["Y", "N"]:
            pref = "Y" if "yes-please" in user else "N"
        else:
            pref, current = labels[0], labels[0]
            for line in user.splitlines():
                m = re.match(r"^(?:GROUP )?([A-Z0-9])[):]", line.strip())
                if m and m.group(1) in labels:
                    current = m.group(1)
                    if STAR in line:
                        pref = current
                    continue
                if STAR in line and line.startswith("  - "):
                    pref = current
        others = [lab for lab in labels if lab != pref]
        top = [{"token": pref, "logprob": math.log(0.7)}]
        top += [{"token": lab, "logprob": math.log(0.2 / len(others))} for lab in others][:17]
        top += [{"token": "<eos>", "logprob": math.log(0.05)}, {"token": "The", "logprob": math.log(0.05)}]
        chosen = "The" if prose else pref
        tokens = [{"token": "thinking", "logprob": 0.0, "top_logprobs": []} for _ in range(think_tokens)]
        tokens.append({"token": chosen, "logprob": math.log(0.7), "top_logprobs": top})
        if think_tokens:  # a trailing stop token, as real backends emit
            tokens.append({"token": "<|endoftext|>", "logprob": 0.0, "top_logprobs": []})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": chosen}, "logprobs": {"content": tokens}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 1 + think_tokens}})

    return httpx.MockTransport(handler), state


def client(settings=SETTINGS, **kw):
    transport, state = fake_backend(**kw)
    return TestClient(create_app(settings, transport=transport)), state


def judge(c, checks, **extra):
    return c.post("/v1/judge", json={"context": "some context", "checks": checks, **extra})


def test_yesno_renormalised_over_labels():
    c, state = client()
    with c:
        r = judge(c, {"a": {"kind": "yesno", "question": "yes-please?"}, "b": {"kind": "yesno", "question": "nope"}})
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert res["a"] == {"kind": "yesno", "p_yes": pytest.approx(0.7 / 0.9, abs=1e-3)}  # junk tokens don't dilute
    assert res["b"]["p_yes"] == pytest.approx(0.2 / 0.9, abs=1e-3)
    assert r.json()["model"] == "fast"
    assert r.json()["usage"] == {"prompt_tokens": 200, "completion_tokens": 2, "backend_calls": 2}
    assert all("structured_outputs" in b and "guided_choice" not in b for b in state["bodies"])


def test_pick_small():
    c, _ = client()
    with c:
        r = judge(c, {"q": {"kind": "pick", "question": "team?", "options": {"billing": None, "technical": f"bugs {STAR}", "sales": None}}})
    res = r.json()["results"]["q"]
    assert res["pick"] == "technical"
    assert sum(res["probs"].values()) == pytest.approx(1.0, abs=1e-3)
    assert res["confidence"] == pytest.approx(confidence(list(res["probs"].values())), abs=1e-3)


@pytest.mark.parametrize("n", [16, 40, 300])
def test_pick_grouped(n):
    c, state = client()
    target = f"opt_{n - 3}"
    options = {f"opt_{i}": (f"desc {STAR}" if f"opt_{i}" == target else None) for i in range(n)}
    with c:
        r = judge(c, {"q": {"kind": "pick", "question": "which?", "options": options}})
    assert r.status_code == 200, r.text
    res = r.json()["results"]["q"]
    assert res["pick"] == target and len(res["probs"]) == n
    assert sum(res["probs"].values()) == pytest.approx(1.0, abs=1e-2)
    assert state["calls"] == 1 + math.ceil(n / 15)


def test_scale():
    c, _ = client()
    with c:
        r = judge(c, {"s": {"kind": "scale", "question": "how?", "levels": ["low", "mid", f"high {STAR}"]}})
    res = r.json()["results"]["s"]
    assert res["probs"][2] == pytest.approx(0.7 / 0.9, abs=1e-3)
    assert res["value"] == pytest.approx(sum(i * p for i, p in enumerate(res["probs"])), abs=1e-3)


def test_trivial_checks_need_no_backend_call():
    c, state = client()
    with c:
        r = judge(c, {"p": {"kind": "pick", "options": {"only": None}}, "s": {"kind": "scale", "levels": ["one"]}})
    assert r.json()["results"]["p"]["pick"] == "only" and state["calls"] == 0


@pytest.mark.parametrize("checks,code", [
    ({"q": {"kind": "pick", "options": {f"o{i}": None for i in range(301)}}}, "too_many_options"),
    ({"q": {"kind": "scale", "levels": [str(i) for i in range(11)]}}, "too_many_levels"),
    ({"q": {"kind": "yesno"}}, "invalid_request"),
    ({"q": {"kind": "banana"}}, "invalid_request"),
    ({}, "invalid_request"),
])
def test_validation(checks, code):
    c, state = client()
    with c:
        r = judge(c, checks)
    assert r.status_code == 400 and r.json()["error"]["code"] == code, r.text
    assert state["calls"] == 0


def test_unknown_model_and_unknown_fields():
    c, _ = client()
    with c:
        r1 = judge(c, {"q": {"kind": "yesno", "question": "x"}}, model="nope")
        r2 = judge(c, {"q": {"kind": "yesno", "question": "x"}}, temperature=0)
    assert r1.status_code == 400 and r1.json()["error"]["code"] == "unknown_model"
    assert r2.status_code == 400 and r2.json()["error"]["code"] == "invalid_request"


def test_unconstrained_backend_fails_loudly():
    c, _ = client(prose=True)
    with c:
        r = judge(c, {"q": {"kind": "yesno", "question": "x"}})
    assert r.status_code == 502 and "structured_outputs" in r.json()["error"]["message"]


@pytest.mark.parametrize("status", [429, 504])
def test_retries_transient_errors(status):
    c, state = client(fail_first=2, status=status)
    with c:
        r = judge(c, {"q": {"kind": "yesno", "question": "yes-please"}})
    assert r.status_code == 200 and state["calls"] == 3


def test_backend_down_is_503():
    c, _ = client(fail_first=99)
    with c:
        r = judge(c, {"q": {"kind": "yesno", "question": "x"}})
    assert r.status_code == 503 and r.json()["error"]["code"] == "backend_unavailable"


def test_debias_asks_both_orders():
    c, state = client()
    with c:
        judge(c, {"n": {"kind": "yesno", "question": "x"}, "p": {"kind": "pick", "options": {"a": None, "b": None}}}, model="biased")
    assert state["calls"] == 4


def test_models_health_and_auth():
    c, _ = client(settings=replace(SETTINGS, api_keys=frozenset({"k1"})))
    with c:
        assert c.get("/v1/models").status_code == 401
        r = c.get("/v1/models", headers={"Authorization": "Bearer k1"})
        h = c.get("/health")
    assert {m["name"] for m in r.json()["models"]} == {"fast", "biased"}
    assert h.status_code == 200 and h.json()["ok"] is True


def test_confidence():
    assert confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert confidence([1 / 3] * 3) == pytest.approx(0.0, abs=1e-9)
    assert 0 < confidence([0.7, 0.2, 0.1]) < 1


def test_health_sends_backend_key_and_reports_auth_failure():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        if seen["auth"] != "Bearer backend-secret":
            return httpx.Response(401, json={"error": "Unauthorized"})
        return httpx.Response(200, json={"data": [{"id": "org/fast-model"}, {"id": "org/biased-model"}]})

    ok = TestClient(create_app(replace(SETTINGS, backend_api_key="backend-secret"), transport=httpx.MockTransport(handler)))
    with ok:
        assert ok.get("/health").json()["ok"] is True
    assert seen["auth"] == "Bearer backend-secret"
    bad = TestClient(create_app(replace(SETTINGS, backend_api_key="wrong"), transport=httpx.MockTransport(handler)))
    with bad:
        r = bad.get("/health")
    assert r.status_code == 503 and "401" in r.json()["error"]


# ---------------------------------------------------------------- qualify
from hunch.lookalikes import build  # noqa: E402
from hunch.qualify import Criteria, Report, qualify_model, verdict  # noqa: E402


def test_lookalikes_dataset_is_balanced_and_fictional():
    items = build()
    assert len(items) == 240 and sum(it["label"] for it in items) == 80
    text = json.dumps(items)
    assert "192.0.2." in text and "10.20." not in text  # documentation IP range only
    vague = build(vague=True)
    assert all(set(it["check"]) == {"kind", "question"} for it in vague)


@pytest.mark.parametrize("fields,qualified,needle", [
    (dict(accuracy=99.0, accuracy_vague=83.0, ece=0.05, flip_rate=0.0), True, None),
    (dict(accuracy=85.0, accuracy_vague=80.0, ece=0.05, flip_rate=0.0), False, "accuracy"),
    (dict(accuracy=96.0, accuracy_vague=72.0, ece=0.18, flip_rate=0.0), False, "ECE"),
    (dict(accuracy=91.0, accuracy_vague=95.0, ece=0.05, flip_rate=0.0), False, "WORSE"),
    (dict(accuracy=96.0, accuracy_vague=80.0, ece=0.05, flip_rate=0.05), False, "flip"),
])
def test_verdict(fields, qualified, needle):
    r = verdict(Report(model="m", backend_model="b", **fields), Criteria())
    assert r.qualified is qualified
    if needle:
        assert any(needle in reason for reason in r.reasons)


def test_qualify_end_to_end_on_fake_backend():
    import asyncio

    async def go():
        transport, _ = fake_backend()
        async with httpx.AsyncClient(transport=transport) as client:
            from hunch.engine import Engine
            engine = Engine(SETTINGS, client)
            return await qualify_model(engine, client, "fast", {"org/fast-model", "org/biased-model"}, {}, Criteria(),
                                       quick=True, progress=lambda *_: None)

    r = asyncio.run(go())
    # the fake backend answers "no" to everything -> 160/240 = 66.7%, well below the bar
    assert r.accuracy == pytest.approx(66.7, abs=0.1) and r.qualified is False
    assert any("accuracy" in reason for reason in r.reasons)


def test_qualify_rejects_unconstrained_backend():
    import asyncio

    async def go():
        transport, _ = fake_backend(prose=True)
        async with httpx.AsyncClient(transport=transport) as client:
            from hunch.engine import Engine
            return await qualify_model(Engine(SETTINGS, client), client, "fast", {"org/fast-model"}, {}, Criteria(),
                                       quick=True, progress=lambda *_: None)

    r = asyncio.run(go())
    assert r.qualified is False and r.accuracy is None
    assert any("structured_outputs NOT enforced" in reason for reason in r.reasons)


def test_probe_retries_transient_errors():
    import asyncio
    from hunch.selftest import probe_constraint

    async def go():
        transport, state = fake_backend(fail_first=2, status=504)
        async with httpx.AsyncClient(transport=transport) as client:
            return await probe_constraint(client, SETTINGS, SETTINGS.models["fast"], {}), state

    problem, state = asyncio.run(go())
    assert problem is None and state["calls"] == 3


def test_qualify_backend_down_is_unavailable_not_a_verdict():
    import asyncio

    async def go():
        transport, _ = fake_backend(fail_first=99, status=504)
        async with httpx.AsyncClient(transport=transport) as client:
            from hunch.engine import Engine
            s = replace(SETTINGS, retries=1)
            return await qualify_model(Engine(s, client), client, "fast", {"org/fast-model"}, {}, Criteria(),
                                       quick=True, progress=lambda *_: None)

    r = asyncio.run(go())
    assert r.unavailable is True and r.qualified is False and r.accuracy is None


def test_package_version_matches_pyproject():
    import pathlib
    import tomllib
    import hunch
    pyproject = tomllib.loads((pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert pyproject["project"]["version"] == hunch.__version__
    assert pyproject["tool"]["setuptools"]["packages"] == ["hunch"]  # bench/ and deploy/ must not be packaged


# ---------------------------------------------------------------- modes
from hunch.engine import Engine  # noqa: E402


def _engine(settings=SETTINGS, **kw):
    transport, state = fake_backend(**kw)
    return Engine(settings, httpx.AsyncClient(transport=transport)), state


def test_detects_a_model_with_no_non_thinking_mode():
    import asyncio

    async def go(scratchpad):
        engine, state = _engine(scratchpad=scratchpad)
        async with engine.client:
            return await engine.opens_scratchpad(SETTINGS.models["fast"]), state["probes"], state["bodies"]

    leaked, probes, _ = asyncio.run(go(True))       # scratchpad leaked into content
    answers, _, _ = asyncio.run(go(False))          # model answers the question
    assert (leaked, answers, probes) == (True, False, 1)


@pytest.mark.parametrize("message,expected", [
    ({"content": "yes"}, False),
    ({"content": "Yes."}, False),
    ({"content": '"yes"'}, False),
    ({"content": "", "reasoning": "The user is asking..."}, True),          # vLLM 0.29.0 field
    ({"content": "", "reasoning_content": "The user is asking..."}, True),  # older builds
    ({"content": "The user is asking me to reply"}, True),                  # leaked scratchpad
    ({"content": ""}, True),
])
def test_scratchpad_probe_reads_both_reasoning_fields(message, expected):
    import asyncio

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", **message}}]})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await Engine(SETTINGS, client).opens_scratchpad(SETTINGS.models["fast"])

    assert asyncio.run(go()) is expected


def test_auto_mode_picks_deliberate_and_caches_the_probe():
    import asyncio

    async def go():
        settings = replace(SETTINGS, models={"m": replace(SETTINGS.models["fast"], mode="auto")})
        engine, state = _engine(settings, scratchpad=True, think_tokens=3)
        async with engine.client:
            first = await engine.mode_for(settings.models["m"])
            second = await engine.mode_for(settings.models["m"])
            results, _ = await engine.judge(settings.models["m"], "ctx", {"q": {"kind": "yesno", "question": "yes-please"}})
        return first, second, state["probes"], state["bodies"][-1], results["q"]

    first, second, probes, body, result = asyncio.run(go())
    assert (first, second) == ("deliberate", "deliberate")
    assert probes == 1                                   # probed once, then cached
    assert body["max_tokens"] == SETTINGS.models["fast"].think_budget
    assert "reasoning_effort" not in body                # "don't think" fields dropped in deliberate mode
    assert result["p_yes"] == pytest.approx(0.7 / 0.9, abs=1e-3)   # verdict read past the thinking tokens


def test_deliberate_mode_errors_when_no_verdict_is_reached():
    import asyncio
    from hunch.engine import HunchError

    async def go():
        spec = replace(SETTINGS.models["fast"], mode="deliberate")
        engine, _ = _engine(replace(SETTINGS, models={"fast": spec}), prose=True, think_tokens=2)
        async with engine.client:
            try:
                await engine.judge(spec, "ctx", {"q": {"kind": "yesno", "question": "x"}})
            except HunchError as e:
                return e.code, e.message
        return None, None

    code, message = asyncio.run(go())
    assert code == "backend_error" and "think_budget" in message


def test_one_token_mode_is_unchanged_and_sends_max_tokens_1():
    import asyncio

    async def go():
        engine, state = _engine()
        async with engine.client:
            await engine.judge(SETTINGS.models["fast"], "ctx", {"q": {"kind": "yesno", "question": "yes-please"}})
        return state["bodies"][-1], state["probes"]

    body, probes = asyncio.run(go())
    assert body["max_tokens"] == 1
    assert body["reasoning_effort"] == "none"            # kept in one-token mode
    assert probes == 0                                   # no probe unless mode="auto"


@pytest.mark.parametrize("token", ["", " ", "The"])
def test_empty_or_foreign_verdict_token_is_rejected_not_crashed(token):
    """'' in "YN" is True in Python (substring!), so an empty token must be rejected explicitly."""
    import asyncio
    from hunch.engine import HunchError

    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": token},
                         "logprobs": {"content": [{"token": token, "logprob": 0.0, "top_logprobs": []}]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            try:
                await Engine(SETTINGS, client).judge(SETTINGS.models["fast"], "ctx",
                                                     {"q": {"kind": "yesno", "question": "x"}})
            except HunchError as e:
                return e.code, e.status
        return None, None

    assert asyncio.run(go()) == ("backend_error", 502)


def test_qualify_records_metrics_for_the_vague_run_too():
    import asyncio

    async def go():
        transport, _ = fake_backend()
        async with httpx.AsyncClient(transport=transport) as client:
            return await qualify_model(Engine(SETTINGS, client), client, "fast", {"org/fast-model"}, {}, Criteria(),
                                       quick=True, progress=lambda *_: None)

    r = asyncio.run(go())
    assert r.accuracy_vague is not None
    assert r.auroc_vague is not None and r.brier_vague is not None and r.ece_vague is not None


@pytest.mark.parametrize("extra,expected", [
    ({"reasoning_effort": "none"}, {}),                                   # "off" value dropped
    ({"reasoning_effort": "low"}, {"reasoning_effort": "low"}),           # a real effort level survives
    ({"reasoning_effort": "max"}, {"reasoning_effort": "max"}),
    ({"chat_template_kwargs": {"enable_thinking": False}}, {}),           # the off switch dropped
    ({"chat_template_kwargs": {"enable_thinking": False, "keep": 1}}, {"chat_template_kwargs": {"keep": 1}}),
    ({"top_k": 5}, {"top_k": 5}),                                         # unrelated fields untouched
])
def test_deliberate_mode_keeps_real_effort_levels(extra, expected):
    import asyncio
    spec = replace(SETTINGS.models["fast"], mode="deliberate", extra_body=extra)

    async def go():
        transport, state = fake_backend(think_tokens=2)
        async with httpx.AsyncClient(transport=transport) as client:
            await Engine(replace(SETTINGS, models={"fast": spec}), client).judge(
                spec, "ctx", {"q": {"kind": "yesno", "question": "yes-please"}})
        return state["bodies"][-1]

    body = asyncio.run(go())
    sent = {k: v for k, v in body.items() if k in ("reasoning_effort", "chat_template_kwargs", "top_k")}
    assert sent == expected


def test_constraint_probe_gives_a_thinking_model_room():
    """With max_tokens=2 a deliberate model spends both tokens thinking and looks unconstrained."""
    import asyncio
    from hunch.selftest import probe_constraint
    seen = {}

    def handler(request):
        body = json.loads(request.content)
        seen["max_tokens"] = body["max_tokens"]
        seen["extra"] = {k: v for k, v in body.items() if k == "reasoning_effort"}
        content = "A" if body["max_tokens"] > 2 else ""       # thinking eats a 2-token budget
        return httpx.Response(200, json={"choices": [{"message": {"content": content},
                                                      "logprobs": {"content": [{"token": content or "x", "logprob": 0.0}]}}]})

    async def go(spec, deliberate):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await probe_constraint(client, SETTINGS, spec, {}, deliberate=deliberate)

    spec = replace(SETTINGS.models["fast"], mode="deliberate", extra_body={"reasoning_effort": "high"})
    assert asyncio.run(go(spec, True)) is None                # passes, with room to think
    assert seen["max_tokens"] == spec.think_budget
    assert seen["extra"] == {"reasoning_effort": "high"}      # a real effort level reaches the probe too
    assert "NOT enforced" in (asyncio.run(go(spec, False)) or "")   # the old behaviour would fail it


def test_effort_is_configurable_per_model_and_per_request():
    import asyncio
    import tomllib  # noqa: F401  (config path is exercised via ModelSpec directly)

    # per model: `effort` is sugar for extra_body.reasoning_effort
    from hunch.config import ModelSpec, load_settings
    import os
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write('[backend]\nurl = "http://b"\n[models.m]\nbackend_model = "x"\nmode = "deliberate"\neffort = "high"\n')
        path = f.name
    try:
        spec = load_settings(path).models["m"]
        assert spec.extra_body["reasoning_effort"] == "high"
    finally:
        os.unlink(path)

    # per request: `effort` overrides it for that call only
    transport, state = fake_backend(think_tokens=2)
    settings = replace(SETTINGS, models={"fast": replace(SETTINGS.models["fast"], mode="deliberate",
                                                         extra_body={"reasoning_effort": "max"})})
    with TestClient(create_app(settings, transport=transport)) as c:
        r = c.post("/v1/judge", json={"context": "x", "checks": {"q": {"kind": "yesno", "question": "yes-please"}},
                                      "effort": "low"})
        assert r.status_code == 200, r.text
        assert state["bodies"][-1]["reasoning_effort"] == "low"
        bad = c.post("/v1/judge", json={"context": "x", "checks": {"q": {"kind": "yesno", "question": "x"}},
                                        "effort": "none"})
    assert bad.status_code == 400 and "switch thinking off" in bad.json()["error"]["message"]


def test_detects_a_model_that_can_think_but_answers_directly():
    import asyncio

    async def go(**kw):
        engine, state = _engine(**kw)
        async with engine.client:
            spec = SETTINGS.models["fast"]
            return await engine.opens_scratchpad(spec), await engine.can_think(spec)

    assert asyncio.run(go(thinks_when_allowed=True)) == (False, True)   # DeepSeek-V4.1-Flash
    assert asyncio.run(go(scratchpad=True)) == (True, True)             # GLM-5.3
    assert asyncio.run(go()) == (False, False)                          # plain instruct model


def test_deliberate_drops_deepseek_style_thinking_switch():
    from hunch.engine import effective_extra_body
    spec = replace(SETTINGS.models["fast"], extra_body={"chat_template_kwargs": {"thinking": False, "x": 1}})
    assert effective_extra_body(spec, deliberate=True) == {"chat_template_kwargs": {"x": 1}}
    assert effective_extra_body(spec, deliberate=False) == spec.extra_body


def test_qualify_reruns_a_thinking_capable_model_in_deliberate_mode(monkeypatch, tmp_path):
    import asyncio
    import functools
    from hunch import qualify

    transport, _ = fake_backend(thinks_when_allowed=True)
    monkeypatch.setattr(qualify, "load_settings", lambda: replace(SETTINGS, models={"fast": SETTINGS.models["fast"]}))
    monkeypatch.setattr(qualify.httpx, "AsyncClient", functools.partial(httpx.AsyncClient, transport=transport))
    out = tmp_path / "q.json"
    code = asyncio.run(qualify.run([], qualify.Criteria(), quick=True, json_path=str(out)))
    reports = json.loads(out.read_text())["reports"]
    assert [r["model"] for r in reports] == ["fast", "fast (deliberate)"]
    assert reports[0]["can_think"] is True and reports[0]["opens_scratchpad"] is False
    assert reports[1]["mode"] == "deliberate"
    assert code == 1   # the fake answers "no" to everything, so neither mode qualifies
