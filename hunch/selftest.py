"""Pre-flight check: run it against your backend before relying on Hunch.

  python -m hunch selftest [model ...]     (exit code 0 = all checks passed)

Per configured model:
  1. the backend lists the model on /v1/models
  2. the backend ENFORCES `structured_outputs` (a "write hello" prompt must come back as A or B).
     Backends that silently drop the constraint would give unconstrained answers, so they fail here.
  3. logprobs come back (probed together with 2)
  4. a 12-case labelled mini-benchmark (yesno / pick / scale) scores >= 11/12
  5. (informational) extraction: does the backend enforce a JSON schema, and are the fields right?
In mode "sample" (no logprobs) steps 2-3 are skipped: answers are votes, not a constrained token.
"""
from __future__ import annotations

import asyncio
import time

import httpx

from .config import load_settings
from .engine import Engine, HunchError, effective_extra_body

_REPLACE = {"kind": "yesno", "question": "Does `new` replace `old`'s value for the same thing?",
            "yes_if": "same thing, changed value", "no_if": "same value restated, or a different thing"}
_TEAM = {"kind": "pick", "question": "Which team should handle this?",
         "options": {"billing": "charges, invoices, refunds", "technical": "bugs, crashes, errors", "sales": "pricing, upgrades"}}
_ANGER = {"kind": "scale", "question": "How angry is the customer?", "levels": ["calm", "annoyed", "furious"]}
_COUNTRIES = ["Germany", "Spain", "Italy", "France", "Portugal", "Belgium", "Austria", "Poland", "Greece", "Sweden",
              "Norway", "Denmark", "Finland", "Ireland", "Netherlands", "Czechia", "Hungary", "Romania", "Slovakia", "Croatia"]

CASES = [  # (context, check, expected): bool for yesno (p >= 0.5), key for pick, rounded level for scale
    ({"old": "The staging DB listens on port 5432.", "new": "Staging DB moved to port 6432."}, _REPLACE, True),
    ({"old": "The staging DB listens on port 5432.", "new": "Port 5432 is where the staging DB accepts connections."}, _REPLACE, False),
    ({"old": "The staging DB listens on port 5432.", "new": "The production DB listens on port 5432."}, _REPLACE, False),
    ("I was charged twice for my order, please refund one of the charges.", {"kind": "yesno", "question": "Is the customer asking for money back?"}, True),
    ("How do I change my profile picture?", {"kind": "yesno", "question": "Is the customer asking for money back?"}, False),
    ("The app crashes every time I open the settings page since the last update.", _TEAM, "technical"),
    ("Can I get a discount if I upgrade 50 seats to the enterprise plan?", _TEAM, "sales"),
    ("Paris is the capital of France.", {"kind": "pick", "question": "Which country does the text mention?",
                                         "options": {c: None for c in _COUNTRIES}}, "France"),
    ("Thanks, everything works perfectly now!", _ANGER, 0),
    ("THIS IS THE THIRD TIME I'M WRITING. UNACCEPTABLE. I WANT A MANAGER NOW!!!", _ANGER, 2),
    ("The meeting is on Tuesday.", {"kind": "yesno", "question": "Does the text mention a day of the week?"}, True),
    ("The meeting is next month.", {"kind": "yesno", "question": "Does the text mention a specific day of the week?"}, False),
]


async def probe_constraint(engine: Engine, spec, deliberate: bool = False) -> str | None:
    """None if the backend enforces structured_outputs and returns logprobs for this model, else the problem.

    In deliberate mode the model thinks first, so the probe must allow it the same budget: with
    max_tokens=2 a thinking model spends both tokens thinking and looks unconstrained.
    """
    if engine.unconstrained(spec):
        return await _probe_logprobs(engine, spec, deliberate)
    body = {"model": spec.backend_model, "messages": [{"role": "user", "content": "Write the word hello and nothing else."}],
            "structured_outputs": {"choice": ["A", "B"]}, "logprobs": True, "top_logprobs": 2,
            "max_tokens": spec.think_budget if deliberate else 2, "temperature": 0,
            **effective_extra_body(spec, deliberate)}
    try:
        d = await engine.send(spec, body)
    except HunchError as e:
        return _probe_problem("constraint probe", e)
    if engine.unconstrained(spec):
        # the backend refused the constraint by name; the engine now reads answers from top_logprobs instead
        return await _probe_logprobs(engine, spec, deliberate)
    content = (d["choices"][0]["message"].get("content") or "").strip()
    if content not in ("A", "B"):
        return f"structured_outputs NOT enforced (got {content!r}); answers would be unconstrained"
    if not (d["choices"][0].get("logprobs") or {}).get("content"):
        return "no logprobs returned"
    return None


def _probe_problem(what: str, e: HunchError) -> str:
    if e.code == "backend_unavailable":
        return f"{what}: backend unreachable ({e.message})"
    return f"{what}: {e.message[:200]}"


async def _probe_logprobs(engine: Engine, spec, deliberate: bool) -> str | None:
    """constrained = false: the backend is not asked to constrain the answer, only to return logprobs.
    Whether the model keeps to the labels on its own is measured by the run itself (off_label errors)."""
    body = {"model": spec.backend_model,
            "messages": [{"role": "user", "content": "Reply with exactly one letter, A or B: which letter comes first in the alphabet?"}],
            "logprobs": True, "top_logprobs": 2, "max_tokens": spec.think_budget if deliberate else 1, "temperature": 0,
            **effective_extra_body(spec, deliberate)}
    try:
        d = await engine.send(spec, body, retry_if=lambda d: None if (d["choices"][0].get("logprobs") or {}).get("content")
                              else "backend returned no logprobs")
    except HunchError as e:
        if e.message == "backend returned no logprobs":
            return "no logprobs returned"
        return _probe_problem("logprobs probe", e)
    return None if (d["choices"][0].get("logprobs") or {}).get("content") else "no logprobs returned"


async def served_models(client: httpx.AsyncClient, s, headers: dict | None = None) -> set[str]:
    from .hosts import models_url
    r = await client.get(models_url(s.backend_url, s.chat_url), headers=s.backend_headers, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"backend /v1/models returned HTTP {r.status_code} (check HUNCH_BACKEND_KEY / [backend] api_key)")
    from .hosts import model_ids
    return set(model_ids(r.json()))


def _correct(check: dict, result: dict, expected) -> bool:
    if check["kind"] == "yesno":
        return (result["p_yes"] >= 0.5) == expected
    if check["kind"] == "pick":
        return result["pick"] == expected
    return round(result["value"]) == expected


def _png(size: int = 32, rgb: tuple[int, int, int] = (220, 30, 30)) -> str:
    """A solid-colour PNG as a data URL, built without an imaging library."""
    import base64
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


async def image_support(engine: Engine, spec) -> str:
    """Informational: can this model judge images? Text-only models are fine for text checks."""
    check = {"kind": "yesno", "question": "Is IMAGE 1 mostly red?"}
    try:
        results, _ = await engine.judge(spec, "", {"c": check}, [_png()])
    except HunchError as e:
        return "not supported (text-only model)" if e.code == "images_not_supported" else f"unknown ({e.code})"
    p = results["c"]["p_yes"]
    return f"accepted (a red square is red: p_yes {p:.2f})" if p >= 0.5 else f"accepted, but misjudged a red square (p_yes {p:.2f})"


_EXTRACT_TEXT = "Order for Dana Reyes: two large pizzas, pick up at 7:30 pm on 2026-05-14, total $38.50, no drinks."
_EXTRACT_FIELDS = {"customer": "the customer's name", "pickup": {"type": "time"}, "day": {"type": "date"},
                   "total": {"type": "money"}, "drink": "a drink ordered"}
_EXTRACT_WANT = {"customer": "Dana Reyes", "pickup": "19:30", "day": "2026-05-14",
                 "total": {"amount": 38.5, "currency": "USD"}, "drink": None}


async def extract_support(engine: Engine, spec) -> str:
    """Informational: extraction needs response_format json_schema; judge checks don't."""
    from .extraction import extract
    try:
        results, _ = await extract(engine, spec, _EXTRACT_TEXT, _EXTRACT_FIELDS)
    except HunchError as e:
        return f"not available ({e.code}: {e.message[:120]})"
    right = sum(results[k]["value"] == v for k, v in _EXTRACT_WANT.items())
    probs = [r["probability"] for r in results.values() if r.get("probability") is not None]
    p = f", min field p {min(probs):.2f}" if probs else ""
    return f"{right}/{len(_EXTRACT_WANT)} fields right{p}"


async def run(names: list[str]) -> int:
    s = load_settings()
    names = names or list(s.models)
    if not names:
        print("FAIL no models configured")
        return 1
    failures = 0
    async with httpx.AsyncClient() as client:
        engine = Engine(s, client)
        print(f"backend: {s.backend_url}")
        try:
            served = await served_models(client, s)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL backend /v1/models: {e}")
            return 1
        for name in names:
            spec = s.resolve(name)
            if spec is None:
                print(f"FAIL {name}: not configured")
                failures += 1
                continue
            print(f"\n== {name} -> {spec.backend_model}")
            if spec.backend_model not in served:
                print(f"FAIL model not served by the backend (served: {sorted(served)})")
                failures += 1
                continue
            print("ok   model is served")
            # the RESOLVED mode: with mode="auto" a thinking model needs the deliberate-sized probe
            mode = await engine.mode_for(spec)
            if mode == "sample":
                print(f"info mode sample: no logprobs; each answer is {spec.samples} votes")
            else:
                problem = await probe_constraint(engine, spec, deliberate=(mode == "deliberate"))
                if problem:
                    print(f"FAIL {problem}")
                    failures += 1
                    continue
                print("ok   structured_outputs enforced and logprobs returned" if not engine.unconstrained(spec)
                      else "ok   logprobs returned (constrained = false: answers read from top_logprobs)")
            t0 = time.perf_counter()
            correct = []
            for context, check, expected in CASES:
                try:
                    results, _ = await engine.judge(spec, context, {"c": check})
                    correct.append(_correct(check, results["c"], expected))
                except HunchError as e:
                    print(f"     error: {e.code}: {e.message}")
                    correct.append(False)
            passed = sum(correct)
            ok = passed >= len(CASES) - 1
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} mini-benchmark {passed}/{len(CASES)} correct, "
                  f"~{(time.perf_counter() - t0) * 1000 / len(CASES):.0f} ms per request")
            print(f"info images: {await image_support(engine, spec)}")
            print(f"info extract: {await extract_support(engine, spec)}")
    print("\nSELFTEST", "PASSED" if not failures else f"FAILED ({failures})")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    return asyncio.run(run(argv))
