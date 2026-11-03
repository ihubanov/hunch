"""Offline tests for the 2.0 surface: tags, sort, answer/min_probability, sample mode, extract, stream."""
import asyncio
import datetime as dt
import json
import math
import re
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from hunch import values
from hunch.config import ModelSpec, Settings
from hunch.engine import Engine
from hunch.extraction import TokenProbs, extract, locate, normalize_fields, value_spans
from hunch.server import create_app
from hunch.stream import Session

from test_api import SETTINGS, STAR, client, judge


# ------------------------------------------------------------------ uniform answer + abstain
def test_every_result_has_answer_and_probability():
    c, _ = client()
    with c:
        r = judge(c, {"y": {"kind": "yesno", "question": "yes-please?"},
                      "n": {"kind": "yesno", "question": "nope"},
                      "p": {"kind": "pick", "options": {"a": None, "b": f"x {STAR}"}},
                      "s": {"kind": "scale", "levels": ["lo", f"hi {STAR}"]}})
    res = r.json()["results"]
    assert (res["y"]["answer"], res["n"]["answer"], res["p"]["answer"], res["s"]["answer"]) == ("yes", "no", "b", 1)
    assert res["n"]["probability"] == pytest.approx(1 - res["n"]["p_yes"], abs=1e-3)
    assert res["p"]["probability"] == res["p"]["probs"]["b"]


def test_min_probability_turns_unsure_answers_into_null():
    c, _ = client()
    with c:
        r = judge(c, {"y": {"kind": "yesno", "question": "yes-please?"},                     # p ~0.78
                      "strict": {"kind": "yesno", "question": "yes-please?", "min_probability": 0.95}},
                  min_probability=0.5)
        bad = judge(c, {"y": {"kind": "yesno", "question": "x", "min_probability": 2}})
    res = r.json()["results"]
    assert res["y"]["answer"] == "yes" and res["strict"]["answer"] is None
    assert res["strict"]["p_yes"] == res["y"]["p_yes"]          # the probability is still reported
    assert bad.status_code == 400


# ------------------------------------------------------------------ tags / sort
def test_tags_are_independent_yesno_checks():
    c, state = client()
    with c:
        r = judge(c, {"t": {"kind": "tags", "question": "Topics?",
                            "tags": {"billing": "yes-please", "bug": None, "praise": "yes-please"}}})
    res = r.json()["results"]["t"]
    assert res["tags"] == ["billing", "praise"] and res["answer"] == ["billing", "praise"]
    assert set(res["probs"]) == {"billing", "bug", "praise"} and state["calls"] == 3
    assert res["probability"] == pytest.approx(min(max(p, 1 - p) for p in res["probs"].values()), abs=1e-3)


def sort_backend():
    """Rates an item by the digit written in it: 'item-7' -> level 7 (with some spread)."""
    def handler(request):
        body = json.loads(request.content)
        labels = body["structured_outputs"]["choice"]
        user = body["messages"][-1]["content"]
        item = re.search(r"ITEM:\n(.*)", user).group(1)
        want = re.search(r"\d", item).group(0)
        top = [{"token": lab, "logprob": math.log(0.82 if lab == want else 0.02)} for lab in labels]
        return httpx.Response(200, json={"choices": [{"message": {"content": want}, "logprobs": {"content": [
            {"token": want, "logprob": math.log(0.82), "top_logprobs": top}]}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 1}})
    return httpx.MockTransport(handler)


def test_sort_orders_by_expected_rating():
    with TestClient(create_app(SETTINGS, transport=sort_backend())) as c:
        r = judge(c, {"s": {"kind": "sort", "question": "How urgent?", "items": {"a": "item-2", "b": "item-9", "c": "item-5"}},
                      "asc": {"kind": "sort", "items": ["item-4", "item-1"], "order": "asc"}})
    res = r.json()["results"]
    assert res["s"]["order"] == ["b", "c", "a"] and res["s"]["scores"]["b"] > res["s"]["scores"]["c"]
    assert res["asc"]["order"] == ["1", "0"]


@pytest.mark.parametrize("check", [
    {"kind": "tags", "tags": {}},
    {"kind": "tags", "tags": {f"t{i}": None for i in range(51)}},
    {"kind": "sort", "items": []},
    {"kind": "sort", "items": ["a"], "order": "sideways"},
    {"kind": "sort", "items": ["a"], "levels": ["only one"]},
])
def test_v2_validation(check):
    c, state = client()
    with c:
        r = judge(c, {"q": check})
    assert r.status_code == 400 and state["calls"] == 0, r.text


# ------------------------------------------------------------------ sample mode (no logprobs)
def sampling_backend(replies):
    """Hands out the given free-text replies in turn; refuses vLLM-only fields like a hosted API would."""
    state = {"i": 0, "bodies": []}

    def handler(request):
        body = json.loads(request.content)
        state["bodies"].append(body)
        if "structured_outputs" in body or "logprobs" in body:
            return httpx.Response(400, json={"error": "unknown field"})
        text = replies[state["i"] % len(replies)]
        state["i"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}],
                                         "usage": {"prompt_tokens": 40, "completion_tokens": 2}})
    return httpx.MockTransport(handler), state


SAMPLED = Settings(backend_url="http://hosted.test", default_model="h",
                   models={"h": ModelSpec(name="h", backend_model="hosted-model", mode="sample", samples=4, extra_body={})})


def test_sample_mode_counts_votes():
    transport, state = sampling_backend(["Yes", "**Y**", "No.", "y"])
    with TestClient(create_app(SAMPLED, transport=transport)) as c:
        r = judge(c, {"q": {"kind": "yesno", "question": "x"}})
    out = r.json()
    assert r.status_code == 200, r.text
    assert out["results"]["q"]["p_yes"] == 0.75 and out["mode"] == "sample" and out["samples"] == 4
    assert out["usage"]["backend_calls"] == 4
    assert all(b["temperature"] == 1.0 and b["max_tokens"] == SAMPLED.models["h"].think_budget for b in state["bodies"])


def test_sample_mode_skips_replies_cut_off_while_thinking():
    """A hosted model that thinks first returns no content when cut off; those votes are dropped, not misread."""
    transport, _ = sampling_backend(["", "", "Y", "N"])
    with TestClient(create_app(SAMPLED, transport=transport)) as c:
        r = judge(c, {"q": {"kind": "yesno", "question": "x"}})
    assert r.json()["results"]["q"]["p_yes"] == 0.5


@pytest.mark.parametrize("text,labels,want", [
    ("Y", "YN", "Y"), ("Yes.", "YN", "Y"), ("no", "YN", "N"), ("**B**", "ABC", "B"), ("C) sales", "ABC", "C"),
    ("Scores: 60, 80. Average 78, at least 75.\n\nY", "YN", "Y"),     # worked out in text, label last
    ("Because the total is 12\nAnswer: B", "AB", None),               # last line isn't a lone label
    ("Absolutely", "AB", None), ("A customer is upset.", "AB", None), ("B.", "AB", "B"), ("", "YN", None),
])
def test_read_label(text, labels, want):
    from hunch.engine import read_label
    assert read_label(text, labels) == want


def test_sample_mode_fails_when_nobody_answers():
    transport, _ = sampling_backend(["I cannot say."])
    with TestClient(create_app(SAMPLED, transport=transport)) as c:
        r = judge(c, {"q": {"kind": "pick", "options": {"a": None, "b": None}}})
    assert r.status_code == 502 and r.json()["error"]["code"] == "no_label" and "sampled replies" in r.json()["error"]["message"]


def test_sample_mode_grouped_pick_tolerates_a_group_without_the_answer():
    """20 options -> a group call plus one call per group. The group lacking the answer gets 'none of these'."""
    def handler(request):
        body = json.loads(request.content)
        user = body["messages"][-1]["content"]
        if "GROUP A:" in user:
            text = "A"                                   # the group call: the answer is in group A
        elif "opt_3" in user:
            text = "D"                                   # group A holds opt_3 at D
        else:
            text = "None of these options fit."          # group B: declines
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": {}})
    opts = {f"opt_{i}": None for i in range(20)}
    with TestClient(create_app(SAMPLED, transport=httpx.MockTransport(handler))) as c:
        r = judge(c, {"q": {"kind": "pick", "options": opts}})
    assert r.status_code == 200, r.text
    assert r.json()["results"]["q"]["pick"] == "opt_3"


# ------------------------------------------------------------------ values
@pytest.mark.parametrize("kind,text,want", [
    ("number", "1,250.5 units", 1250.5), ("number", "12k", 12000), ("number", "3 million", 3000000),
    ("number", "seven", 7), ("number", "none", None),
    ("money", "$450 a month", {"amount": 450, "currency": "USD"}),
    ("money", "EUR 1.2m", {"amount": 1200000, "currency": "EUR"}),
    ("money", "450 dollars", {"amount": 450, "currency": "USD"}), ("money", "£20", {"amount": 20, "currency": "GBP"}),
    ("date", "2026-03-03", "2026-03-03"), ("date", "March 3, 2026", "2026-03-03"), ("date", "3rd of March 2026", "2026-03-03"),
    ("date", "25/12/2026", "2026-12-25"), ("date", "03/04/2026", None), ("date", "tomorrow", "2026-11-04"),
    ("date", "next Friday", "2026-11-06"), ("date", "in two weeks", "2026-11-17"),
    ("time", "5pm tomorrow", "17:00"), ("time", "5:30 p.m.", "17:30"), ("time", "17:45", "17:45"),
    ("time", "noon", "12:00"), ("time", "12am", "00:00"), ("time", "soon", None),
    ("duration", "2 hours", 7200), ("duration", "1h30m", 5400), ("duration", "half an hour", 1800),
    ("duration", "an hour and a half", 5400), ("duration", "90 min", 5400), ("duration", "a while", None),
])
def test_value_parsers(kind, text, want):
    assert values.parse(kind, text, dt.date(2026, 11, 3)) == want      # a Tuesday


# ------------------------------------------------------------------ extract
def test_value_spans_and_locate():
    text = '{"a": "x y", "b": null, "c": ["p", "q"]}'
    sp = value_spans(text)
    assert text[slice(*sp["a"][1])] == '"x y"' and text[slice(*sp["b"][1])] == "null"
    assert [text[a:b] for a, b in sp["c"][2]] == ['"p"', '"q"']
    assert locate("midsize  suv", "A Midsize SUV, please") == (2, 13)
    assert locate("tea", "tea and tea", taken={0}) == (8, 11)
    assert locate("coffee", "tea") is None


def test_token_probs_cover_a_span_and_skip_thinking():
    answer = '{"a": "xy"}'
    toks = [{"token": "think ", "logprob": math.log(0.1)}, {"token": '{"a": "', "logprob": 0.0},
            {"token": "x", "logprob": math.log(0.5)}, {"token": "y", "logprob": math.log(0.8)},
            {"token": '"}', "logprob": 0.0}]
    tp = TokenProbs(toks, answer)
    a, b = value_spans(answer)["a"][1]
    assert tp.prob(a, b) == pytest.approx(0.4)        # the thinking token is not in the span


def test_normalize_fields_shapes():
    f = normalize_fields({"a": "desc", "b": ["an item"], "c": {"type": "money", "min_probability": 0.9}})
    assert f["a"] == {"description": "desc", "type": "text", "many": False}
    assert f["b"]["many"] and f["c"]["type"] == "money" and f["c"]["min_probability"] == 0.9
    from hunch.engine import HunchError
    for bad in ({}, {"a": {"type": "colour"}}, {"a": 5}):
        with pytest.raises(HunchError):
            normalize_fields(bad)


SOURCE = "Hi, I'd like a midsize SUV picked up at 5pm tomorrow. Budget is $450 a month. Add a child seat and a GPS."


def extract_backend(answer: dict, token_p: dict | None = None, fail_first: int = 0, raw: str | None = None):
    """Answers with `answer` as JSON. Every character of a value listed in token_p costs that probability once."""
    state = {"calls": 0, "bodies": []}

    def handler(request):
        body = json.loads(request.content)
        state["calls"] += 1
        state["bodies"].append(body)
        if state["calls"] <= fail_first:
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"vehicle": "midsize'}, "logprobs": {"content": []}}]})
        text = raw if raw is not None else json.dumps(answer)
        toks = []
        # one token per character keeps the arithmetic obvious; the first char of a listed value carries its p
        marks = {}
        for k, p in (token_p or {}).items():
            i = text.find(f'"{k}": ') + len(f'"{k}": ')
            marks[i] = p
        for i, ch in enumerate(text):
            toks.append({"token": ch, "logprob": math.log(marks[i]) if i in marks else 0.0})
        toks.append({"token": "<eot>", "logprob": 0.0})
        return httpx.Response(200, json={"choices": [{"message": {"content": text}, "logprobs": {"content": toks},
                                                      "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 120, "completion_tokens": len(toks)}})
    return httpx.MockTransport(handler), state


def run_extract(transport, fields, settings=SETTINGS, **kw):
    async def go():
        async with httpx.AsyncClient(transport=transport) as cl:
            eng = Engine(settings, cl)
            return await extract(eng, settings.resolve(None), SOURCE, fields, now=dt.date(2026, 11, 3), **kw)
    return asyncio.run(go())


FIELDS = {"vehicle": "the vehicle asked for", "pickup": {"type": "time"}, "pickup_day": {"type": "date"},
          "budget": {"type": "money"}, "colour": "paint colour", "extras": ["an extra asked for"]}
ANSWER = {"vehicle": "midsize SUV", "pickup": "5pm", "pickup_day": "tomorrow", "budget": "$450 a month",
          "colour": None, "extras": ["a child seat", "a GPS"]}


def test_extract_grounds_types_and_scores_each_field():
    transport, state = extract_backend(ANSWER, token_p={"vehicle": 0.9, "colour": 0.6})
    res, usage = run_extract(transport, FIELDS)
    v = res["vehicle"]
    assert v["status"] == "found" and v["value"] == "midsize SUV" and SOURCE[v["start"]:v["end"]] == "midsize SUV"
    assert v["probability"] == pytest.approx(0.9)
    assert res["pickup"]["value"] == "17:00" and res["pickup_day"]["value"] == "2026-11-04"
    assert res["budget"]["value"] == {"amount": 450, "currency": "USD"} and res["budget"]["probability"] == 1.0
    assert res["colour"] == {"value": None, "text": None, "start": None, "end": None, "probability": 0.6, "status": "absent"}
    assert [i["value"] for i in res["extras"]["items"]] == ["a child seat", "a GPS"]
    body = state["bodies"][0]
    assert body["response_format"]["type"] == "json_schema" and body["logprobs"] is True and body["temperature"] == 0
    assert body["response_format"]["json_schema"]["schema"]["required"] == list(FIELDS)
    assert usage.backend_calls == 1


def test_extract_never_returns_an_ungrounded_value():
    transport, _ = extract_backend({**ANSWER, "vehicle": "a red pickup truck"})
    res, _ = run_extract(transport, FIELDS)
    assert res["vehicle"]["status"] == "ungrounded" and res["vehicle"]["value"] is None
    assert res["vehicle"]["text"] == "a red pickup truck"


def test_extract_unsure_and_unparsed():
    transport, _ = extract_backend({**ANSWER, "budget": "Budget"}, token_p={"vehicle": 0.4})
    res, _ = run_extract(transport, FIELDS, min_probability=0.5)
    assert res["vehicle"]["status"] == "unsure" and res["vehicle"]["value"] is None
    assert res["budget"]["status"] == "unparsed" and res["budget"]["text"] == "Budget"


def test_extract_retries_a_broken_json_answer_then_fails_loudly():
    transport, state = extract_backend(ANSWER, fail_first=1)
    res, usage = run_extract(transport, FIELDS)
    assert res["vehicle"]["status"] == "found" and usage.backend_calls == 2
    transport, _ = extract_backend(ANSWER, fail_first=99)
    from hunch.engine import HunchError
    with pytest.raises(HunchError, match="JSON object"):
        run_extract(transport, FIELDS)


def test_extract_sampled_uses_agreement():
    replies = [json.dumps({**ANSWER, "vehicle": v}) for v in ["midsize SUV", "Midsize SUV", "midsize SUV", "SUV"]]
    transport, _ = sampling_backend(["```json\n" + r + "\n```" for r in replies])
    res, usage = run_extract(transport, FIELDS, settings=SAMPLED)
    assert res["vehicle"]["probability"] == 0.75 and res["vehicle"]["status"] == "found"
    assert usage.backend_calls == 4


def test_extract_endpoint():
    transport, _ = extract_backend(ANSWER)
    with TestClient(create_app(SETTINGS, transport=transport)) as c:
        r = c.post("/v1/extract", json={"context": SOURCE, "fields": FIELDS, "now": "2026-11-03"})
        bad = c.post("/v1/extract", json={"context": SOURCE, "fields": FIELDS, "now": "someday"})
    assert r.status_code == 200, r.text
    assert r.json()["results"]["pickup_day"]["value"] == "2026-11-04" and r.json()["mode"] == "one_token"
    assert bad.status_code == 400


# ------------------------------------------------------------------ stream
def keyword_backend():
    """yes when the CONTEXT contains 'cancel', else no."""
    def handler(request):
        body = json.loads(request.content)
        user = body["messages"][-1]["content"]
        ctx = user.split("\n\nQUESTION")[0]
        want = "Y" if "cancel" in ctx else "N"
        top = [{"token": "Y", "logprob": math.log(0.9 if want == "Y" else 0.1)},
               {"token": "N", "logprob": math.log(0.1 if want == "Y" else 0.9)}]
        return httpx.Response(200, json={"choices": [{"message": {"content": want}, "logprobs": {"content": [
            {"token": want, "logprob": math.log(0.9), "top_logprobs": top}]}}],
            "usage": {"prompt_tokens": len(ctx), "completion_tokens": 1}})
    return httpx.MockTransport(handler)


CANCEL = {"cancel": {"kind": "yesno", "question": "Does the caller want to cancel?"}}


def test_session_reports_flips():
    async def go():
        async with httpx.AsyncClient(transport=keyword_backend()) as cl:
            s = Session(Engine(SETTINGS, cl), SETTINGS.resolve(None), CANCEL)
            a = await s.append("Hi, I'm calling about my plan.")
            b = await s.append(" It's fine I guess.")
            c = await s.append(" Actually, I want to cancel it.")
            return a, b, c
    a, b, c = asyncio.run(go())
    assert a["changed"] == ["cancel"] and a["results"]["cancel"]["answer"] == "no"
    assert b["changed"] == [] and c["changed"] == ["cancel"] and c["results"]["cancel"]["answer"] == "yes"
    assert c["update"] == 3 and c["chars"] > b["chars"]


def test_stream_websocket_emits_only_changes():
    with TestClient(create_app(SETTINGS, transport=keyword_backend())) as c:
        with c.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "start", "checks": CANCEL})
            assert ws.receive_json() == {"type": "ready"}
            ws.send_json({"type": "append", "text": "Hello there."})
            first = ws.receive_json()
            assert first["type"] == "decision" and first["results"]["cancel"]["answer"] == "no"
            ws.send_json({"type": "append", "text": " Still here."})          # no flip -> no message
            ws.send_json({"type": "append", "text": " Please cancel."})
            flip = ws.receive_json()
            assert flip["changed"] == ["cancel"] and flip["results"]["cancel"]["answer"] == "yes"
            ws.send_json({"type": "end"})


def test_stream_websocket_rejects_bad_start_and_bad_key():
    with TestClient(create_app(SETTINGS, transport=keyword_backend())) as c:
        with c.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "append", "text": "x"})
            assert ws.receive_json()["type"] == "error"
    keyed = replace(SETTINGS, api_keys=frozenset({"k1"}))
    with TestClient(create_app(keyed, transport=keyword_backend())) as c:
        with c.websocket_connect("/v1/stream") as ws:
            assert ws.receive_json()["code"] == "unauthorized"
        with c.websocket_connect("/v1/stream?api_key=k1") as ws:
            ws.send_json({"type": "start", "checks": CANCEL, "emit": "all"})
            assert ws.receive_json() == {"type": "ready"}
            ws.send_json({"type": "end"})


# ------------------------------------------------------------------ library + MCP
def test_library_extract_and_mcp_tool(monkeypatch, tmp_path):
    import hunch
    from hunch import mcp
    monkeypatch.chdir(tmp_path)
    for v in ("HUNCH_CONFIG", "HUNCH_MODE"):
        monkeypatch.delenv(v, raising=False)
    transport, _ = extract_backend(ANSWER)
    out = hunch.extract(SOURCE, FIELDS, base_url="http://b.test", model="org/m", transport=transport, now="2026-11-03")
    assert out["results"]["budget"]["value"]["amount"] == 450
    tools = mcp.handle({"method": "tools/list"})["tools"]
    assert [t["name"] for t in tools] == ["judge", "extract"]


# ------------------------------------------------------------------ reasoning: off / on / auto
def reasoning_backend(fast_p: dict, think_p: float = 0.99, fast_breaks: bool = False, gate_p: dict | None = None):
    """One-token calls (max_tokens 1) answer Y with fast_p[word in question]; thinking calls think, then Y with think_p.
    The auto pre-check ("can the answer be read directly?") answers Y with gate_p[word] (default 0.95: readable).
    fast_breaks: the one-token call opens a scratchpad instead of answering (a model with no thinking-off mode)."""
    state = {"fast": 0, "gate": 0, "think": 0, "bodies": []}

    def handler(request):
        body = json.loads(request.content)
        state["bodies"].append(body)
        user = body["messages"][-1]["content"]
        deliberate = body["max_tokens"] > 1
        if deliberate:
            state["think"] += 1
            p = think_p
        elif "read directly" in user and not fast_breaks:
            state["gate"] += 1
            p = next((v for k, v in (gate_p or {}).items() if k in user), 0.95)
        else:
            state["fast"] += 1
            if fast_breaks:
                return httpx.Response(200, json={"choices": [{"message": {"content": "<think>", "reasoning": "hmm"},
                    "logprobs": {"content": [{"token": "<think>", "logprob": 0.0, "top_logprobs": []}]}}]})
            p = next(v for k, v in fast_p.items() if k in user)
        top = [{"token": "Y", "logprob": math.log(p)}, {"token": "N", "logprob": math.log(1 - p)}]
        toks = ([{"token": "thinking", "logprob": 0.0, "top_logprobs": []}] * 3 if deliberate else [])
        toks.append({"token": "Y", "logprob": math.log(p), "top_logprobs": top})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Y"}, "logprobs": {"content": toks}}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": len(toks)}})
    return httpx.MockTransport(handler), state


EFFORTFUL = replace(SETTINGS, models={"fast": replace(SETTINGS.models["fast"], mode="deliberate",
                                                      extra_body={"reasoning_effort": "low"})})


def test_reasoning_auto_thinks_only_on_unsure_checks():
    transport, state = reasoning_backend({"easy": 0.97, "hard": 0.6})
    with TestClient(create_app(EFFORTFUL, transport=transport)) as c:
        r = judge(c, {"e": {"kind": "yesno", "question": "easy?"}, "h": {"kind": "yesno", "question": "hard?"}},
                  reasoning="auto")
    out = r.json()
    assert r.status_code == 200, r.text
    assert out["results"]["e"]["reasoned"] is False and out["results"]["e"]["p_yes"] == pytest.approx(0.97, abs=1e-3)
    assert out["results"]["h"]["reasoned"] is True and out["results"]["h"]["fast_probability"] == pytest.approx(0.6, abs=1e-3)
    assert out["results"]["h"]["p_yes"] == pytest.approx(0.99, abs=1e-3)
    assert out["reasoning"] == "auto" and out["escalated"] == ["h"]
    assert (state["fast"], state["gate"], state["think"]) == (2, 2, 1)
    fast_bodies = [b for b in state["bodies"] if b["max_tokens"] == 1]
    think_bodies = [b for b in state["bodies"] if b["max_tokens"] > 1]
    assert all(b["reasoning_effort"] == "none" for b in fast_bodies)       # the configured effort can't leak in
    assert all(b["reasoning_effort"] == "low" for b in think_bodies)


def test_reasoning_auto_thinks_when_the_answer_must_be_worked_out_even_if_confident():
    """A one-pass answer can be confidently wrong on a computed question; the pre-check catches it."""
    transport, state = reasoning_backend({"total": 0.97, "plain": 0.97}, gate_p={"total": 0.1})
    with TestClient(create_app(EFFORTFUL, transport=transport)) as c:
        r = judge(c, {"t": {"kind": "yesno", "question": "Is the total over 100?"},
                      "p": {"kind": "yesno", "question": "Is this plain?"}}, reasoning="auto")
    res = r.json()["results"]
    assert res["t"]["reasoned"] is True and res["t"]["needs_steps"] is True and res["t"]["fast_probability"] == pytest.approx(0.97, abs=1e-3)
    assert res["p"]["reasoned"] is False and r.json()["escalated"] == ["t"] and state["think"] == 1


def test_reasoning_auto_threshold_and_off_on():
    transport, state = reasoning_backend({"easy": 0.97, "hard": 0.6})
    with TestClient(create_app(EFFORTFUL, transport=transport)) as c:
        loose = judge(c, {"h": {"kind": "yesno", "question": "hard?"}}, reasoning="auto", escalate_below=0.5)
        off = judge(c, {"h": {"kind": "yesno", "question": "hard?"}}, reasoning="off")
        on = judge(c, {"e": {"kind": "yesno", "question": "easy?"}}, reasoning="on")
        bad = judge(c, {"h": {"kind": "yesno", "question": "hard?"}}, reasoning="sometimes")
        bad2 = judge(c, {"h": {"kind": "yesno", "question": "hard?"}}, reasoning="auto", escalate_below=3)
    assert loose.json()["results"]["h"]["reasoned"] is False
    assert off.json()["results"]["h"]["reasoned"] is False and off.json()["escalated"] == []
    assert on.json()["results"]["e"]["reasoned"] is True and on.json()["results"]["e"]["p_yes"] == pytest.approx(0.99, abs=1e-3)
    assert bad.status_code == 400 and bad2.status_code == 400


def test_reasoning_auto_thinks_when_the_model_cannot_answer_in_one_token():
    transport, state = reasoning_backend({}, fast_breaks=True)
    with TestClient(create_app(EFFORTFUL, transport=transport)) as c:
        r = judge(c, {"q": {"kind": "yesno", "question": "x?"}}, reasoning="auto")
    res = r.json()["results"]["q"]
    assert r.status_code == 200, r.text
    assert res["reasoned"] is True and "fast_probability" not in res and state["think"] == 1


def test_reasoning_is_ignored_in_sample_mode_and_without_it_nothing_changes():
    transport, _ = sampling_backend(["Y"])
    with TestClient(create_app(SAMPLED, transport=transport)) as c:
        r = judge(c, {"q": {"kind": "yesno", "question": "x"}}, reasoning="auto")
    assert r.status_code == 200 and "reasoned" not in r.json()["results"]["q"]
    c2, _ = client()
    with c2:
        plain = judge(c2, {"q": {"kind": "yesno", "question": "yes-please"}})
    assert "reasoning" not in plain.json() and "reasoned" not in plain.json()["results"]["q"]
