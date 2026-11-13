"""What a backend can do, and how Hunch should run on it. Probes are a handful of small calls.

  python -m hunch probe [model] [--json]

The probe answers four questions:
  1. constraint  - does the backend enforce structured_outputs? (a "write hello" prompt constrained to A/B)
  2. logprobs    - does it return top_logprobs on the first token, and how many? Do the allowed labels hold
                   most of the probability when nothing constrains the answer?
  3. thinking    - with thinking switched off, does the model still open a scratchpad before answering?
  4. votes       - when there are no usable logprobs: does a plain reply name one of the labels?

From those it picks the mode (see DECISION below). It prints the settings to use, as environment variables
for the zero-config path. Callers that probe once and cache the verdict should key it by
(backend URL, backend model, hunch version), and re-probe when judge calls start failing.

The JSON output is schema "hunch.probe/1":
  schema            "hunch.probe/1"
  hunch_version     the Hunch that probed
  backend_url       API root the probe used
  chat_url          the exact chat-completions URL
  model             backend model id
  mode              "one_token" | "deliberate" | "sample" | "unsupported"
  constrained       bool: true when the backend enforces structured_outputs (only meaningful for one_token / deliberate)
  max_top_logprobs  int | null: the largest top_logprobs the backend accepted (20 or 5), null without logprobs
  env               {HUNCH_MODE, HUNCH_CONSTRAINED, HUNCH_MAX_TOP_LOGPROBS[, HUNCH_EXTRA_BODY]}: strings to set
                    for this backend; {} when mode is "unsupported"
  capabilities      {constraint_enforced: bool|null, structured_outputs_refused: bool, logprobs: bool,
                     label_mass: float|null, thinks_first: bool, votes_answer: bool|null}
  dropped_fields    request fields the backend refused by name, which Hunch now leaves out
  calls             backend calls the probe made
  errors            up to 3 backend errors seen while probing ("code: message")
  reason            one sentence: why this mode
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import replace

import httpx

from . import __version__
from .engine import Engine, HunchError, fast_spec, read_label

SCHEMA = "hunch.probe/1"
_HELLO = [{"role": "user", "content": "Write the word hello and nothing else."}]
_LETTER = [{"role": "user", "content": "Reply with exactly one letter, A or B: which letter comes first in the alphabet?"}]

DECISION = """\
  constraint enforced, logprobs, answers first    -> one_token   (1 call per check; the best)
  logprobs, thinks first                          -> deliberate  (1 call, up to think_budget tokens; without an
                                                                  enforced answer the verdict is the last option token)
  no constraint, logprobs, answers first          -> one_token with constrained = false (1 call; read from top_logprobs)
  anything else, if plain replies name a label    -> sample      (samples calls per check; coarser, costlier)
  otherwise                                       -> unsupported"""


def _logprob_content(data: dict) -> list:
    return ((data.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []


def _content(data: dict) -> str:
    return (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()


async def probe(engine: Engine, spec) -> dict:
    """Probe `spec`'s backend model. Never raises for a backend that merely lacks a feature."""
    calls0 = engine.calls
    errors: list[str] = []
    spec = fast_spec(replace(spec, constrained=True))    # thinking off, constraint asked for
    caps: dict = {"constraint_enforced": None, "structured_outputs_refused": False, "logprobs": False,
                  "label_mass": None, "thinks_first": False, "votes_answer": None}

    # 1. constraint
    body = engine._request(spec, _HELLO, "AB", deliberate=False)
    try:
        d = await engine.send(spec, body)
        if engine.unconstrained(spec):
            caps["structured_outputs_refused"] = True
            caps["constraint_enforced"] = False
        else:
            caps["constraint_enforced"] = _content(d) in ("A", "B") and bool(_logprob_content(d))
        caps["logprobs"] = bool(_logprob_content(d))
    except HunchError as e:
        errors.append(f"{e.code}: {e.message[:200]}")
        caps["constraint_enforced"] = False

    # 2. logprobs, unconstrained; the widest top_logprobs the backend takes
    max_top = None
    for top in (engine.s.max_top_logprobs, 5):
        body = {"model": spec.backend_model, "messages": _LETTER, "logprobs": True, "top_logprobs": top,
                "max_tokens": 1, "temperature": 0, **spec.extra_body}
        try:
            # routers occasionally answer from a provider that drops logprobs: give it the usual retries
            d = await engine.send(spec, body, retry_if=lambda d: None if _logprob_content(d) else "no logprobs")
        except HunchError as e:
            if "top_logprobs" in e.message.lower() and top != 5:
                continue        # the cap is lower: try 5
            errors.append(f"{e.code}: {e.message[:200]}")
            break               # no logprobs without a constraint (step 1 may still have had them)
        toks = _logprob_content(d)
        if toks:
            caps["logprobs"] = True
            max_top = top
            mass = sum(math.exp(t["logprob"]) for t in toks[0].get("top_logprobs") or []
                       if str(t.get("token", "")).strip() in ("A", "B") and t.get("logprob") is not None)
            caps["label_mass"] = round(mass, 4)
        break

    # 3. thinking, with thinking switched off
    caps["thinks_first"] = await engine.opens_scratchpad(spec)

    # decide
    enforced, lp, thinks = caps["constraint_enforced"], caps["logprobs"], caps["thinks_first"]
    mass_ok = (caps["label_mass"] or 0) >= spec.min_label_mass
    if enforced and lp and not thinks:
        mode, constrained, reason = "one_token", True, "the backend enforces the answer and returns logprobs"
    elif lp and thinks:
        # deliberate reads the last option token after the thinking. With an enforced constraint that token is
        # guaranteed; without one it is whatever the model ends with, so a call that ends elsewhere fails loudly.
        mode, constrained = "deliberate", bool(enforced)
        reason = ("the model thinks before answering even with thinking off"
                  + ("" if enforced else "; nothing enforces its final answer, so the verdict is the last option "
                     "token it writes (run qualify)"))
    elif lp and not thinks and mass_ok:
        mode, constrained, reason = ("one_token", False, "logprobs come back but the answer is not enforced; "
                                     "answers are read from the allowed labels in top_logprobs")
    else:
        # 4. votes
        why = "no logprobs" if not lp else f"the allowed labels hold only {caps['label_mass']} of the probability"
        body = {"model": spec.backend_model, "messages": _LETTER, "max_tokens": spec.think_budget,
                "temperature": spec.sample_temperature, **spec.extra_body}
        try:
            caps["votes_answer"] = read_label(_content(await engine.send(spec, body)), "AB") == "A"
        except HunchError as e:
            errors.append(f"{e.code}: {e.message[:200]}")
            caps["votes_answer"] = False
        if caps["votes_answer"]:
            mode, constrained, reason = "sample", False, f"{why}; answers are votes over {spec.samples} replies"
        else:
            mode, constrained, reason = "unsupported", False, f"{why}, and a plain reply did not name the right label"
            if errors:
                reason += f" (backend said: {errors[-1]})"

    dropped = sorted(engine.dropped.get(spec.backend_model, ()))
    env: dict = {}
    if mode != "unsupported":
        env = {"HUNCH_MODE": mode, "HUNCH_CONSTRAINED": "1" if constrained else "0"}
        if max_top is not None:
            env["HUNCH_MAX_TOP_LOGPROBS"] = str(max_top)
        if any(f in spec.extra_body for f in dropped):
            env["HUNCH_EXTRA_BODY"] = json.dumps({k: v for k, v in spec.extra_body.items() if k not in dropped})
    return {"schema": SCHEMA, "hunch_version": __version__, "backend_url": engine.s.backend_url,
            "chat_url": engine.s.completions_url, "model": spec.backend_model, "mode": mode,
            "constrained": constrained, "max_top_logprobs": max_top, "env": env, "capabilities": caps,
            "dropped_fields": dropped, "calls": engine.calls - calls0, "errors": errors[:3], "reason": reason}


async def aprobe_settings(settings, spec, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    async with httpx.AsyncClient(**({"transport": transport} if transport is not None else {})) as client:
        return await probe(Engine(settings, client), spec)


def _print(r: dict) -> None:
    print(f"backend: {r['chat_url']}\nmodel:   {r['model']}\n")
    c = r["capabilities"]
    yn = lambda v: "-" if v is None else ("yes" if v else "no")  # noqa: E731
    print(f"  constraint enforced:  {yn(c['constraint_enforced'])}"
          + ("  (structured_outputs refused by name)" if c["structured_outputs_refused"] else ""))
    print(f"  logprobs:             {yn(c['logprobs'])}"
          + (f"  (top_logprobs up to {r['max_top_logprobs']}, label mass {c['label_mass']})" if c["logprobs"] else ""))
    print(f"  thinks first:         {yn(c['thinks_first'])}")
    if c["votes_answer"] is not None:
        print(f"  plain replies answer: {yn(c['votes_answer'])}")
    if r["dropped_fields"]:
        print(f"  refused fields:       {', '.join(r['dropped_fields'])} (left out from now on)")
    print(f"\nmode: {r['mode']}  - {r['reason']}  ({r['calls']} calls)")
    if r["env"]:
        print("\nsettings:\n" + "\n".join(f"  {k}={v if k != 'HUNCH_EXTRA_BODY' else repr(v)}" for k, v in r["env"].items()))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m hunch probe", description=__doc__.split("\n\n")[0],
                                 epilog="Decision:\n" + DECISION, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="configured model name, or with zero-config the backend model id")
    ap.add_argument("--json", action="store_true", help="print the hunch.probe/1 JSON only")
    a = ap.parse_args(argv)
    from .client import _settings
    try:
        settings, spec = asyncio.run(_settings(None, None, a.model))
        r = asyncio.run(aprobe_settings(settings, spec))
    except HunchError as e:
        print(json.dumps({"error": {"code": e.code, "message": e.message}}) if a.json else f"probe failed: {e.message}",
              file=sys.stdout if a.json else sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r))
    else:
        _print(r)
    return 0 if r["mode"] != "unsupported" else 1
