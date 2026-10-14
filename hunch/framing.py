"""Framing-invariance for yes/no judgments — flag the checks whose answer moves with the WORDING, not the facts.

A leading question (one whose phrasing signals the answer it wants) is how a judgment gets *inflicted* rather than
measured. Hunch already removes the conversational channel for that — you commit a fixed, neutral criterion instead of
asking a loaded question — and `debias` flips LABEL order. This adds the next axis: flip the QUESTION *framing*.

You give a yes/no check one or more alternate framings of the same underlying question (a polarity flip, and/or a
neutral rewording). Hunch judges all of them in one request (concurrent, shared context prefix), maps each to the same
P(true) axis, and returns:
  p_yes        : the consensus (mean of the framings' P(true)) — use THIS for the decision, it's the robust estimate
  views        : P(true) under each framing, so you can see the spread
  swing        : max P(true) − min P(true) across framings
  framing_flag : swing > swing_gate — the answer is being driven by framing, not content; trust p_yes less / neutralise

This is `debias` generalised from label-order to question-framing. It is ADDITIVE: checks with no `framings` key pass
straight through `judge`/`ajudge` unchanged.

A framing is {"question": str, "polarity": "same"|"flip", "yes_if"?: str, "no_if"?: str}. "flip" marks a COMPLEMENT
phrasing (e.g. "is X irrelevant?") whose p_yes is mapped to 1 − p_yes before comparison, so every framing lands on one
P(the thing is true) axis.

Scope — be honest about it: this catches FRAMING-driven answers (the question's wording inflicting the answer). It does
NOT catch a model that is stably wrong across every framing — that is the model believing the wrong thing, and it needs
evidence/grounding, not this. And it only guards Hunch's own (code-written) checks; a human's leading question in a chat
is a different channel this cannot touch.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any


def _plain(check: dict) -> dict:
    """The check as the engine wants it — our 'framings' key stripped."""
    return {k: v for k, v in check.items() if k != "framings"}


def _p_true(res: dict, sign: int) -> float:
    """Map a sub-check's p_yes onto the shared P(true) axis: +1 as-is, −1 for a polarity-flipped (complement) framing."""
    p = float(res["p_yes"])
    return p if sign > 0 else 1.0 - p


def _expand(checks: dict[str, Any]) -> tuple[dict[str, dict], dict[str, tuple[list[tuple[str, int]], bool]]]:
    """Turn framed yes/no checks into plain sibling checks for one judge() call.
    Returns (engine_checks, plan) where plan[parent] = ([(sub_name, sign), ...], framed?)."""
    engine: dict[str, dict] = {}
    plan: dict[str, tuple[list[tuple[str, int]], bool]] = {}
    for name, c in checks.items():
        framings = c.get("framings") if isinstance(c, dict) else None
        if not framings or not isinstance(c, dict) or c.get("kind") != "yesno":
            engine[name] = _plain(c) if isinstance(c, dict) else c
            plan[name] = ([(name, 1)], False)
            continue
        subs: list[tuple[str, int]] = [(name, 1)]
        engine[name] = _plain(c)                       # the primary (as-asked) framing keeps the parent's name
        for i, fr in enumerate(framings):
            sub = f"{name}__f{i}"
            spec: dict[str, Any] = {"kind": "yesno"}
            if isinstance(fr, dict):
                spec["question"] = fr.get("question", c.get("question"))
                for k in ("yes_if", "no_if"):
                    if fr.get(k):
                        spec[k] = fr[k]
                sign = -1 if str(fr.get("polarity", "same")).lower() == "flip" else 1
            else:                                      # a bare string = a same-polarity reword
                spec["question"], sign = str(fr), 1
            engine[sub] = spec
            subs.append((sub, sign))
        plan[name] = (subs, True)
    return engine, plan


def _combine(plan: dict[str, tuple[list[tuple[str, int]], bool]], raw: dict[str, dict], swing_gate: float) -> dict:
    """Fold sibling framings back into one result per parent: consensus p_yes, per-framing views, swing, flag."""
    out: dict[str, dict] = {}
    for name, (subs, framed) in plan.items():
        if not framed:
            out[name] = raw[name]                      # untouched: non-framed checks pass straight through
            continue
        views = {sub: _p_true(raw[sub], sign) for sub, sign in subs}
        vals = list(views.values())
        consensus = sum(vals) / len(vals)
        swing = max(vals) - min(vals) if len(vals) > 1 else 0.0
        out[name] = {"kind": "yesno", "p_yes": round(consensus, 4), "swing": round(swing, 4),
                     "framing_flag": swing > swing_gate, "views": {s: round(v, 4) for s, v in views.items()}}
    return out


async def ajudge_framed(context: Any = "", checks: dict[str, Any] | None = None, *,
                        swing_gate: float = 0.25, **kw: Any) -> dict:
    """ajudge(), but any yes/no check carrying a `framings` list is judged under every framing and folded into a
    consensus p_yes + a framing_flag. All other kwargs (model/effort/base_url/api_key/images/transport) pass through."""
    from .client import ajudge                         # lazy: keep this module importable (and _combine testable) w/o httpx
    if not checks:
        return await ajudge(context, checks, **kw)     # let ajudge raise the standard empty-checks error
    engine_checks, plan = _expand(checks)
    r = await ajudge(context, engine_checks, **kw)
    return {**{k: v for k, v in r.items() if k != "results"}, "results": _combine(plan, r["results"], swing_gate)}


def judge_framed(context: Any = "", checks: dict[str, Any] | None = None, *, swing_gate: float = 0.25, **kw: Any) -> dict:
    """Blocking framing-invariant judge. Safe to call from inside an event loop (runs on a worker thread), like judge()."""
    coro = ajudge_framed(context, checks, swing_gate=swing_gate, **kw)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _selftest() -> int:
    fails = 0

    def check(name: str, ok: bool) -> None:
        nonlocal fails
        fails += (not ok)
        print(f"  {'OK ' if ok else 'FAIL'} {name}")

    plan = {"n": ([("n", 1), ("n__f0", -1)], True)}    # as-asked + one polarity-flipped framing

    # framing-INVARIANT: "relevant?"=0.9 and "irrelevant?"=0.1 agree (1-0.1=0.9) -> swing~0, NOT flagged
    r = _combine(plan, {"n": {"kind": "yesno", "p_yes": 0.9}, "n__f0": {"kind": "yesno", "p_yes": 0.1}}, 0.25)
    check("invariant: consensus 0.9, swing~0, not flagged",
          abs(r["n"]["p_yes"] - 0.9) < 1e-9 and r["n"]["swing"] < 0.01 and r["n"]["framing_flag"] is False)

    # framing-SENSITIVE: "relevant?"=0.9 YES but "irrelevant?"=0.8 ALSO yes (1-0.8=0.2) -> views {0.9,0.2} swing 0.7 FLAG
    r2 = _combine(plan, {"n": {"kind": "yesno", "p_yes": 0.9}, "n__f0": {"kind": "yesno", "p_yes": 0.8}}, 0.25)
    check("sensitive: swing 0.7 -> framing_flag True", r2["n"]["framing_flag"] and abs(r2["n"]["swing"] - 0.7) < 1e-9)
    check("sensitive: consensus is the midpoint 0.55", abs(r2["n"]["p_yes"] - 0.55) < 1e-9)

    # same-polarity reword that disagrees also flags
    plan2 = {"n": ([("n", 1), ("n__f0", 1)], True)}
    r3 = _combine(plan2, {"n": {"kind": "yesno", "p_yes": 0.85}, "n__f0": {"kind": "yesno", "p_yes": 0.30}}, 0.25)
    check("reword disagreement 0.55 -> flagged", r3["n"]["framing_flag"] and abs(r3["n"]["swing"] - 0.55) < 1e-9)

    # non-framed checks pass straight through, untouched (yesno AND pick)
    plan3 = {"y": ([("y", 1)], False), "p": ([("p", 1)], False)}
    raw3 = {"y": {"kind": "yesno", "p_yes": 0.7}, "p": {"kind": "pick", "pick": "a", "probs": {"a": 0.7}}}
    r4 = _combine(plan3, raw3, 0.25)
    check("passthrough: non-framed yesno + pick unchanged", r4 == raw3)

    # _expand builds the right sibling checks and strips 'framings'
    eng, pl = _expand({"rel": {"kind": "yesno", "question": "Is this note relevant to the task?",
                               "framings": [{"question": "Is this note irrelevant / unrelated to the task?", "polarity": "flip"}]},
                       "plain": {"kind": "yesno", "question": "x?"}})
    check("_expand: primary + one sibling, framings stripped",
          set(eng) == {"rel", "rel__f0", "plain"} and "framings" not in eng["rel"] and eng["rel__f0"]["kind"] == "yesno")
    check("_expand: plan marks rel framed with flip sign, plain not framed",
          pl["rel"] == ([("rel", 1), ("rel__f0", -1)], True) and pl["plain"] == ([("plain", 1)], False))

    print(f"\nframing self-test: {'ALL PASS' if fails == 0 else str(fails) + ' FAIL'}")
    return fails


if __name__ == "__main__":
    import sys
    sys.exit(1 if _selftest() else 0)
