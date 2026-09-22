"""Score the upstream Laya typed-decision model on Hunch's look-alike benchmark.

Laya (https://github.com/NandhaKishorM/laya) is a purpose-trained decision model, not an
OpenAI-compatible server, so it can't be a Hunch backend today. This script runs it directly on the
same 240 labelled pairs, with the same metrics as `python -m hunch qualify`, so the numbers compare.

    pip install laya "hunch @ git+https://github.com/ihubanov/hunch"
    python bench/laya_compare.py [--model convaiinnovations/laya] [--device cuda] [--json out.json]

Each pair is one `noul` question. Two variants, as in qualify:
  named  - the question plus Laya's native criteria {"true": yes_if, "false": no_if}
  vague  - the bare question, no criteria
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from a checkout without installing
from hunch.lookalikes import build  # noqa: E402
from hunch.qualify import (  # noqa: E402
    GATE, Criteria, Report, accuracy_at_gate, auroc, ece, print_report, verdict)


def question_def(check: dict, vague: bool) -> dict:
    """A Laya noul question. Laya reads `criteria` for every type, so the definitions go there
    (its docstring lists criteria only for choice/score, but agent._to_internal passes it through)."""
    q = {"type": "noul", "instructions": check["question"]}
    if not vague and (check.get("yes_if") or check.get("no_if")):
        q["criteria"] = {"true": check.get("yes_if"), "false": check.get("no_if")}
    return q


def run(agent, items: list[dict], vague: bool) -> tuple[list[float | None], float]:
    out, t0 = [], time.perf_counter()
    for it in items:
        try:
            result = agent.predict(it["context"], {"q": question_def(it["check"], vague)})
            answer = result["answers"]["q"]
            out.append(float(answer["noul"] if isinstance(answer, dict) and "noul" in answer else answer))
        except Exception as e:  # noqa: BLE001
            print(f"  error on {it['id']}: {e!r}")
            out.append(None)
    return out, (time.perf_counter() - t0) / len(items) * 1000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--device", default=None, help="cuda / cpu / mps (default: the library's choice)")
    ap.add_argument("--runs", type=int, default=2, help="repeats of the named variant, for the stability check")
    ap.add_argument("--json", dest="json_path")
    a = ap.parse_args()

    import laya  # noqa: PLC0415  (imported here so --help works without the dependency)

    kwargs = {"device": a.device} if a.device else {}
    t0 = time.perf_counter()
    agent = laya.load(a.model, **kwargs)
    load_s = time.perf_counter() - t0
    items = build()
    labels = [it["label"] for it in items]
    print(f"{a.model} loaded in {load_s:.1f}s; {len(items)} labelled pairs")

    runs, per_item_ms = [], None
    for i in range(a.runs):
        ps, ms = run(agent, items, vague=False)
        runs.append(ps)
        per_item_ms = ms if per_item_ms is None else per_item_ms
        print(f"  named run {i + 1}/{a.runs}: {ms:.0f} ms per question")
    vague_ps, vague_ms = run(agent, items, vague=True)
    print(f"  vague run: {vague_ms:.0f} ms per question")

    scored = [(p, y) for p, y in zip(runs[0], labels) if p is not None]
    r = Report(model=f"laya:{a.model}", backend_model=a.model)
    r.accuracy = round(accuracy_at_gate(runs[0], labels), 1)
    r.accuracy_vague = round(accuracy_at_gate(vague_ps, labels), 1)
    r.auroc = round(auroc(scored), 3)
    r.brier = round(statistics.mean((p - y) ** 2 for p, y in scored), 3)
    r.ece = round(ece(scored), 3)
    r.errors = sum(p is None for run_ in runs for p in run_) + sum(p is None for p in vague_ps)
    if len(runs) > 1:
        both = [(x, y) for x, y in zip(runs[0], runs[1]) if x is not None and y is not None]
        r.flip_rate = round(sum((x >= 0.5) != (y >= 0.5) for x, y in both) / len(both), 4) if both else None
    criteria = Criteria()
    print_report(verdict(r, criteria), criteria)
    print(f"   median latency: {per_item_ms:.0f} ms per question (gate {GATE})")
    if a.json_path:
        with open(a.json_path, "w") as f:
            json.dump({"model": a.model, "device": a.device, "load_seconds": round(load_s, 1),
                       "ms_per_question": round(per_item_ms, 1), "report": r.__dict__}, f, indent=1)
    return 0 if r.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
