"""Run Hunch on the public multistep_decisions benchmark (MIT, github.com/levantolabs/multistep_decisions).

100 typed decisions whose answer is not written in the text: it has to be constructed (a running total,
a rule chain, a state after a sequence of events). A one-token answer has no scratchpad for that, so this
set separates "decide in one forward pass" from "think briefly, then decide".

    git clone https://github.com/levantolabs/multistep_decisions /tmp/msd
    python bench/multistep.py /tmp/msd/data/multistep_decisions.jsonl \\
        --base-url http://vllm:8000 --model google/gemma-4-31b-it --mode one_token --out preds.jsonl
    python /tmp/msd/evaluate.py preds.jsonl            # their scorer, on our predictions

`yesno` items ("noul", labels no/yes) become a yesno check with the item's criteria as yes_if / no_if;
`choice` items become a pick over the labels with the criteria as option descriptions. Every prediction
carries the full distribution, so the scorer also reports Brier and ECE.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hunch import HunchError, ajudge  # noqa: E402


def to_check(item: dict) -> dict:
    q = item["question"]
    crit = q.get("criteria") or {}
    if q["type"] == "noul":
        return {"kind": "yesno", "question": q["instructions"], "yes_if": crit.get("true"), "no_if": crit.get("false")}
    return {"kind": "pick", "question": q["instructions"], "options": {lab: crit.get(lab) for lab in item["labels"]}}


def to_prediction(item: dict, result: dict) -> dict:
    if result["kind"] == "yesno":
        probs = {"yes": result["p_yes"], "no": round(1 - result["p_yes"], 4)}
    else:
        probs = dict(result["probs"])
    return {"id": item["id"], "answer": max(probs, key=probs.get), "probs": probs}


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    total, out = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        cell = [(p, ok) for p, ok in pairs if lo < p <= hi or (b == 0 and p == 0)]
        if cell:
            out += len(cell) / total * abs(sum(p for p, _ in cell) / len(cell) - sum(ok for _, ok in cell) / len(cell))
    return out


async def run(args) -> int:
    items = [json.loads(line) for line in open(args.data) if line.strip()]
    if args.limit:
        items = items[:args.limit]
    if args.mode:
        os.environ["HUNCH_MODE"] = args.mode
    if args.effort:
        os.environ["HUNCH_EFFORT"] = args.effort
    sem = asyncio.Semaphore(args.concurrency)
    preds, lat, tokens, errors = {}, [], [0, 0], 0

    async def one(item):
        nonlocal errors
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await ajudge(item["state"], {"d": to_check(item)}, base_url=args.base_url, model=args.model,
                                 api_key=args.api_key)
            except HunchError as e:
                errors += 1
                print(f"  {item['id']}: {e.code}: {e.message[:120]}", file=sys.stderr)
                return
            lat.append(time.perf_counter() - t0)
            tokens[0] += r["usage"]["prompt_tokens"]
            tokens[1] += r["usage"]["completion_tokens"]
            preds[item["id"]] = to_prediction(item, r["results"]["d"])

    t0 = time.perf_counter()
    await asyncio.gather(*[one(it) for it in items])
    wall = time.perf_counter() - t0
    if args.out:
        with open(args.out, "w") as f:
            for it in items:
                if it["id"] in preds:
                    f.write(json.dumps(preds[it["id"]]) + "\n")
    right = sum(1 for it in items if it["id"] in preds and preds[it["id"]]["answer"] == it["expected"])
    calib = [(max(preds[it["id"]]["probs"].values()), preds[it["id"]]["answer"] == it["expected"])
             for it in items if it["id"] in preds]
    by_fam: dict[str, list[int]] = {}
    for it in items:
        ok = it["id"] in preds and preds[it["id"]]["answer"] == it["expected"]
        by_fam.setdefault(it["family"], []).append(ok)
    report = {
        "model": args.model, "mode": args.mode or "auto", "effort": args.effort, "n": len(items),
        "accuracy": round(right / len(items), 4), "correct": right, "errors": errors,
        "ece": round(ece(calib), 4) if calib else None,
        "p50_s": round(statistics.median(lat), 3) if lat else None,
        "p90_s": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 3) if lat else None,
        "prompt_tokens_per_item": round(tokens[0] / max(1, len(preds))),
        "completion_tokens_per_item": round(tokens[1] / max(1, len(preds)), 1),
        "wall_s": round(wall, 1),
        "by_family": {f: f"{sum(v)}/{len(v)}" for f, v in sorted(by_fam.items())},
    }
    print(json.dumps(report, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="multistep_decisions.jsonl")
    ap.add_argument("--base-url", default=os.environ.get("HUNCH_BACKEND_URL"))
    ap.add_argument("--api-key", default=os.environ.get("HUNCH_BACKEND_KEY"))
    ap.add_argument("--model", default=os.environ.get("HUNCH_BACKEND_MODEL"))
    ap.add_argument("--mode", choices=["one_token", "deliberate", "auto", "sample"])
    ap.add_argument("--effort", help="reasoning effort in deliberate mode, e.g. low")
    ap.add_argument("--concurrency", type=int, default=4, help="items in flight (keep modest on a shared server)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", help="write predictions here, in the benchmark's own format")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
