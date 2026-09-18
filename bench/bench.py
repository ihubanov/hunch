"""Benchmarks against a running Hunch (default http://127.0.0.1:8791, override with HUNCH_URL).

  python bench/bench.py accuracy [model ...]   # 240 labelled look-alike pairs, 2 runs per model
  python bench/bench.py fanout   [model]       # latency vs number of checks and big picks

Accuracy is measured at the 0.9 gate (a yes/no answer counts as "yes" when p_yes >= 0.9), plus
AUROC, Brier score, expected calibration error (ECE), run-to-run drift and latency.
"""
import asyncio
import collections
import os
import pathlib
import statistics
import sys
import time

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset import build  # noqa: E402

URL = os.environ.get("HUNCH_URL", "http://127.0.0.1:8791")
HEADERS = {"Authorization": f"Bearer {os.environ['HUNCH_API_KEY']}"} if os.environ.get("HUNCH_API_KEY") else {}


async def call(c, model, context, checks):
    t = time.perf_counter()
    r = await c.post(f"{URL}/v1/judge", json={"model": model, "context": context, "checks": checks}, headers=HEADERS, timeout=120)
    return r, round((time.perf_counter() - t) * 1000)


def auroc(pairs):
    pos = [p for p, y in pairs if y]
    neg = [p for p, y in pairs if not y]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")


def ece(pairs, bins=10):
    buckets = collections.defaultdict(list)
    for p, y in pairs:
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    return sum(len(b) / len(pairs) * abs(statistics.mean(p for p, _ in b) - statistics.mean(y for _, y in b)) for b in buckets.values())


async def accuracy(models):
    items = build()
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient() as c:
        async def one(model, it):
            async with sem:
                r, ms = await call(c, model, it["context"], {"q": it["check"]})
            return (r.json()["results"]["q"]["p_yes"] if r.status_code == 200 else None), ms

        print(f"{'model':16} {'acc@.9':>7} {'AUROC':>6} {'Brier':>6} {'ECE':>6} {'drift max':>9} {'flips':>5} {'errors':>6} {'p50/p95 ms':>12}")
        for m in models:
            runs = [await asyncio.gather(*[one(m, it) for it in items]) for _ in range(2)]
            first = runs[0]
            scored = [(p, it["label"]) for (p, _), it in zip(first, items) if p is not None]
            if not scored:
                print(f"{(m or '(default)'):16} every request failed; check the model name, the backend and auth")
                continue
            acc = 100 * sum((p is not None and p >= 0.9) == bool(it["label"]) for (p, _), it in zip(first, items)) / len(items)
            drift = [abs(a[0] - b[0]) for a, b in zip(*runs) if a[0] is not None and b[0] is not None]
            flips = sum((a[0] >= 0.5) != (b[0] >= 0.5) for a, b in zip(*runs) if a[0] is not None and b[0] is not None)
            errors = sum(p is None for run in runs for p, _ in run)
            ms = sorted(t for run in runs for _, t in run)
            print(f"{(m or '(default)'):16} {acc:7.1f} {auroc(scored):6.3f} {statistics.mean((p - y) ** 2 for p, y in scored):6.3f} "
                  f"{ece(scored):6.3f} {max(drift):9.3f} {flips:5d} {errors:6d} {ms[len(ms) // 2]:>5}/{ms[int(len(ms) * .95)]:<6}")


async def fanout(model):
    context = {"ticket": "I was charged twice for order A-104 and nobody replies. Fix it today or I cancel.",
               "customer": {"plan": "Pro", "tenure_months": 14}}
    topics = "billing refund cancel urgency order support shipping login".split()
    cases = [(f"{n} yes/no checks", {f"q{i}": {"kind": "yesno", "question": f"Does the ticket mention {topics[i % 8]} (item {i})?"} for i in range(n)})
             for n in (1, 10, 50, 100)]
    cases.append(("pick, 100 options", {"p": {"kind": "pick", "question": "Which category fits?",
                                              "options": {f"cat_{i}": ("duplicate charge / billing" if i == 42 else f"unrelated topic {i}") for i in range(100)}}}))
    cases.append(("scale, 10 levels", {"s": {"kind": "scale", "question": "How angry is the customer?", "levels": [f"level {i}" for i in range(10)]}}))
    async with httpx.AsyncClient() as c:
        await call(c, model, context, {"warm": {"kind": "yesno", "question": "Is this a ticket?"}})
        print(f"{'case':22} {'ms (3 runs)':>22} {'backend calls':>14}")
        for name, checks in cases:
            runs = [await call(c, model, context, checks) for _ in range(3)]
            r = runs[-1][0].json()
            print(f"{name:22} {str([ms for _, ms in runs]):>22} {r['usage']['backend_calls']:>14}")


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "accuracy"), sys.argv[2:]
    if cmd == "accuracy":
        asyncio.run(accuracy(args or [None]))
    else:
        asyncio.run(fanout(args[0] if args else None))
