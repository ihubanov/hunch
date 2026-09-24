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


def question_def(check: dict, vague: bool, labels: dict | None = None) -> dict:
    """A Laya noul question. Laya reads `criteria` for every type, so the definitions go there
    (documented for noul since laya 0.3.7). `labels` (laya >= 0.3.11) replaces the option words
    `true:` / `false:` the model sees; the returned value is still P(true)."""
    q = {"type": "noul", "instructions": check["question"]}
    if not vague and (check.get("yes_if") or check.get("no_if")):
        q["criteria"] = {"true": check.get("yes_if"), "false": check.get("no_if")}
    if labels:
        q["labels"] = labels
    return q


# --noul-labels: neutral option words for noul (laya#163). "once" is one call per question;
# "both" also asks with the words swapped and averages, so a preference for either word cancels.
# The slot order stays false-then-true either way: labels change the words, not the positions.
NOUL_LABELS = {"once": [{"true": "A", "false": "B"}],
               "both": [{"true": "A", "false": "B"}, {"true": "B", "false": "A"}]}


SEMANTIC_KEYS = {"old": "old_assertion", "new": "new_assertion", "a": "assertion_a", "b": "assertion_b"}


def context_of(it: dict, semantic: bool) -> dict:
    ctx = it["context"]
    return {SEMANTIC_KEYS.get(k, k): v for k, v in ctx.items()} if semantic and isinstance(ctx, dict) else ctx


def choice_defs(check: dict, vague: bool) -> list[tuple[dict, str]]:
    """The same yes/no question as a two-option choice with neutral keys, in both orders.

    laya#156: `noul` hardcodes its option labels to `false:` / `true:`, and on the shipped checkpoints
    those label WORDS can decide the answer regardless of the state. Neutral A/B keys avoid that; running
    both orders and averaging also cancels any preference for the first-listed option.
    """
    yes_text = check.get("yes_if") or "the statement is true of the data"
    no_text = check.get("no_if") or "the statement is not true of the data"
    if vague:  # no definitions: the options only say yes / no
        yes_text, no_text = "yes", "no"
    q = check["question"]
    return [({"type": "choice", "instructions": q, "criteria": {"A": yes_text, "B": no_text}}, "A"),
            ({"type": "choice", "instructions": q, "criteria": {"A": no_text, "B": yes_text}}, "B")]


def _p_yes(answer, yes_key: str) -> float:
    if isinstance(answer, dict) and "noul" in answer:
        return float(answer["noul"])
    probs = answer.get("probabilities") or answer.get("probs") or {}
    if probs:
        total = sum(float(v) for v in probs.values()) or 1.0
        return float(probs.get(yes_key, 0.0)) / total
    return 1.0 if answer.get("choice") == yes_key else 0.0


def run(agent, items: list[dict], vague: bool, semantic: bool = False, as_choice: bool = False,
        noul_labels: str | None = None) -> tuple[list[float | None], float]:
    out, t0 = [], time.perf_counter()
    for it in items:
        try:
            context = context_of(it, semantic)
            if as_choice:
                ps = []
                for qdef, yes_key in choice_defs(it["check"], vague):
                    ps.append(_p_yes(agent.predict(context, {"q": qdef})["answers"]["q"], yes_key))
                out.append(sum(ps) / len(ps))
            elif noul_labels:
                ps = [_p_yes(agent.predict(context, {"q": question_def(it["check"], vague, lab)})["answers"]["q"], "A")
                      for lab in NOUL_LABELS[noul_labels]]
                out.append(sum(ps) / len(ps))
            else:
                answer = agent.predict(context, {"q": question_def(it["check"], vague)})["answers"]["q"]
                out.append(_p_yes(answer, "A"))
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
    ap.add_argument("--as-choice", action="store_true",
                    help="ask each pair as a two-option `choice` with neutral keys A/B instead of `noul`, "
                         "per NandhaKishorM/laya#156: noul hardcodes true:/false: option labels, and those "
                         "label words can decide the answer. Runs both key orders and averages, so a "
                         "position preference cancels out.")
    ap.add_argument("--noul-labels", choices=sorted(NOUL_LABELS),
                    help="ask as noul with neutral option words A/B instead of true/false (laya >= 0.3.11, "
                         "laya#163). once: one call per question; both: also swapped, averaged")
    ap.add_argument("--semantic-keys", action="store_true",
                    help="rename the context keys old/new -> old_assertion/new_assertion and a/b -> "
                         "assertion_a/assertion_b, as recommended in NandhaKishorM/laya#135")
    a = ap.parse_args()
    if a.as_choice and a.noul_labels:
        ap.error("--as-choice and --noul-labels are alternative framings; pick one")

    import laya  # noqa: PLC0415  (imported here so --help works without the dependency)

    kwargs = {"device": a.device} if a.device else {}
    t0 = time.perf_counter()
    agent = laya.load(a.model, **kwargs)
    load_s = time.perf_counter() - t0
    items = build()
    labels = [it["label"] for it in items]
    version = getattr(laya, "__version__", "unknown")
    print(f"{a.model} (laya {version}) loaded in {load_s:.1f}s; {len(items)} labelled pairs"
          f"{'; semantic context keys' if a.semantic_keys else ''}"
          f"{'; asked as a two-option choice, both orders' if a.as_choice else ''}"
          f"{f'; noul with neutral labels ({a.noul_labels})' if a.noul_labels else ''}")

    runs, per_item_ms = [], None
    for i in range(a.runs):
        ps, ms = run(agent, items, vague=False, semantic=a.semantic_keys, as_choice=a.as_choice, noul_labels=a.noul_labels)
        runs.append(ps)
        per_item_ms = ms if per_item_ms is None else per_item_ms
        print(f"  named run {i + 1}/{a.runs}: {ms:.0f} ms per question")
    vague_ps, vague_ms = run(agent, items, vague=True, semantic=a.semantic_keys, as_choice=a.as_choice, noul_labels=a.noul_labels)
    print(f"  vague run: {vague_ms:.0f} ms per question")

    scored = [(p, y) for p, y in zip(runs[0], labels) if p is not None]
    framing = " (as choice)" if a.as_choice else f" (noul labels {a.noul_labels})" if a.noul_labels else ""
    r = Report(model=f"laya:{a.model}{framing}", backend_model=a.model)
    r.accuracy = round(accuracy_at_gate(runs[0], labels), 1)
    r.accuracy_vague = round(accuracy_at_gate(vague_ps, labels), 1)
    scored_vague = [(p, y) for p, y in zip(vague_ps, labels) if p is not None]
    if scored_vague:
        r.auroc_vague = round(auroc(scored_vague), 3)
        r.brier_vague = round(statistics.mean((p - y) ** 2 for p, y in scored_vague), 3)
        r.ece_vague = round(ece(scored_vague), 3)
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
            json.dump({"model": a.model, "laya_version": version, "device": a.device, "load_seconds": round(load_s, 1),
                       "ms_per_question": round(per_item_ms, 1), "report": r.__dict__}, f, indent=1)
    return 0 if r.qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
