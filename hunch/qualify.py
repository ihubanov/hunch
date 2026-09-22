"""Qualify a model as a Hunch backend before relying on it.

  python -m hunch qualify [model ...] [--min-accuracy 90] [--max-ece 0.15] [--max-flip-rate 0.02]
                          [--quick] [--json FILE]

Runs directly against the configured backend (no Hunch server needed) and prints QUALIFIED / NOT QUALIFIED
per model, with the reasons. Exit code 0 only if every model qualifies.

Criteria, on the 240 fictional look-alike pairs in hunch/lookalikes.py:
  1. setup      - the backend serves the model, enforces structured_outputs and returns logprobs
  2. accuracy   - accuracy at the p_yes >= 0.9 gate, with the look-alike cases named in yes_if / no_if
  3. calibration - expected calibration error (ECE) of p_yes
  4. definitions - naming the look-alikes must not make the model worse than the bare question
                  (if it does, better checks won't fix it)
  5. stability  - share of answers that flip across 0.5 between two identical runs
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field, replace

import httpx

from .config import load_settings
from .engine import Engine, HunchError
from .lookalikes import build
from .selftest import probe_constraint, served_models

GATE = 0.9


@dataclass
class Criteria:
    min_accuracy: float = 90.0
    max_ece: float = 0.15
    max_flip_rate: float = 0.02


@dataclass
class Report:
    model: str
    backend_model: str
    qualified: bool = False
    unavailable: bool = False   # backend down: no verdict either way
    mode: str = "one_token"     # how the answer was read: one_token / deliberate
    opens_scratchpad: bool | None = None   # the model's first token starts thinking, not answering
    reasons: list[str] = field(default_factory=list)
    accuracy: float | None = None
    accuracy_vague: float | None = None
    auroc: float | None = None
    brier: float | None = None
    auroc_vague: float | None = None     # same metrics for the question-only run, for comparison
    brier_vague: float | None = None
    ece_vague: float | None = None
    ece: float | None = None
    flip_rate: float | None = None
    errors: int = 0
    seconds: float = 0.0


def auroc(pairs: list[tuple[float, int]]) -> float:
    pos = [p for p, y in pairs if y]
    neg = [p for p, y in pairs if not y]
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def ece(pairs: list[tuple[float, int]], bins: int = 10) -> float:
    buckets = collections.defaultdict(list)
    for p, y in pairs:
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    return sum(len(b) / len(pairs) * abs(statistics.mean(p for p, _ in b) - statistics.mean(y for _, y in b))
               for b in buckets.values())


def accuracy_at_gate(ps: list[float | None], labels: list[int]) -> float:
    return 100 * sum((p is not None and p >= GATE) == bool(y) for p, y in zip(ps, labels)) / len(labels)


def verdict(r: Report, c: Criteria) -> Report:
    """Fill r.qualified / r.reasons from the measured numbers (setup failures are added by the caller)."""
    if r.accuracy is None:
        r.qualified = False
        return r
    if r.accuracy < c.min_accuracy:
        r.reasons.append(f"accuracy {r.accuracy:.1f}% < {c.min_accuracy:g}% at the p_yes >= {GATE} gate")
    if r.ece is not None and r.ece > c.max_ece:
        r.reasons.append(f"calibration error (ECE) {r.ece:.3f} > {c.max_ece:g}: its probabilities can't be trusted as probabilities")
    if r.accuracy_vague is not None and r.accuracy < r.accuracy_vague:
        r.reasons.append(f"naming the look-alikes made it WORSE ({r.accuracy_vague:.1f}% -> {r.accuracy:.1f}%): "
                         "better-written checks won't fix this model")
    if r.flip_rate is not None and r.flip_rate > c.max_flip_rate:
        r.reasons.append(f"{100 * r.flip_rate:.1f}% of answers flip between identical runs (> {100 * c.max_flip_rate:g}%)")
    if r.errors:
        r.reasons.append(f"{r.errors} requests failed")
    r.qualified = not r.reasons
    return r


async def _run(engine: Engine, spec, items: list[dict]) -> list[float | None]:
    async def one(it):
        try:
            results, _ = await engine.judge(spec, it["context"], {"q": it["check"]})
            return results["q"]["p_yes"]
        except HunchError:
            return None
    return await asyncio.gather(*[one(it) for it in items])


async def qualify_model(engine: Engine, client: httpx.AsyncClient, name: str, served: set[str], headers: dict,
                        criteria: Criteria, quick: bool = False, progress=print,
                        spec_override=None, label: str | None = None) -> Report:
    s = engine.s
    spec = spec_override or s.resolve(name)
    if spec is None:
        return Report(model=name, backend_model="?", reasons=[f"model {name!r} is not configured"])
    r = Report(model=label or name, backend_model=spec.backend_model)
    t0 = time.perf_counter()
    if spec.backend_model not in served:
        r.reasons.append(f"backend does not serve {spec.backend_model!r}")
        return r
    r.mode = await engine.mode_for(spec)
    problem = await probe_constraint(client, s, spec, headers, deliberate=(r.mode == "deliberate"))
    if problem:
        r.reasons.append(problem)
        r.unavailable = problem.startswith("constraint probe: HTTP 5") or "unreachable" in problem
        return r

    if r.mode == "one_token":
        r.opens_scratchpad = await engine.opens_scratchpad(spec)
    named, vague = build(), build(vague=True)
    labels = [it["label"] for it in named]
    progress(f"   {r.model}: mode={r.mode}, named look-alikes, run 1/{1 if quick else 2} ...")
    run1 = await _run(engine, spec, named)
    run2 = None
    if not quick:
        progress(f"   {name}: named look-alikes, run 2/2 ...")
        run2 = await _run(engine, spec, named)
    progress(f"   {name}: question only (vague) ...")
    run_vague = await _run(engine, spec, vague)

    scored = [(p, y) for p, y in zip(run1, labels) if p is not None]
    r.errors = sum(p is None for p in run1) + sum(p is None for p in run_vague) + (sum(p is None for p in run2) if run2 else 0)
    if not scored:
        r.reasons.append("every request failed")
        return r
    r.accuracy = round(accuracy_at_gate(run1, labels), 1)
    r.accuracy_vague = round(accuracy_at_gate(run_vague, labels), 1)
    scored_vague = [(p, y) for p, y in zip(run_vague, labels) if p is not None]
    if scored_vague:
        r.auroc_vague = round(auroc(scored_vague), 3)
        r.brier_vague = round(statistics.mean((p - y) ** 2 for p, y in scored_vague), 3)
        r.ece_vague = round(ece(scored_vague), 3)
    r.auroc = round(auroc(scored), 3)
    r.brier = round(statistics.mean((p - y) ** 2 for p, y in scored), 3)
    r.ece = round(ece(scored), 3)
    if run2:
        both = [(a, b) for a, b in zip(run1, run2) if a is not None and b is not None]
        r.flip_rate = round(sum((a >= 0.5) != (b >= 0.5) for a, b in both) / len(both), 4) if both else None
    r.seconds = round(time.perf_counter() - t0, 1)
    return verdict(r, criteria)


def print_report(r: Report, c: Criteria) -> None:
    status = "QUALIFIED" if r.qualified else ("UNAVAILABLE (backend down, no verdict)" if r.unavailable else "NOT QUALIFIED")
    print(f"\n== {r.model} ({r.backend_model}, mode={r.mode}): {status}")
    if r.opens_scratchpad:
        print("   NOTE: this model's first generated token opens a thinking block rather than answering, so it has")
        print("         no non-thinking mode. One-token numbers below do not measure its ability; see mode=deliberate.")
    if r.accuracy is not None:
        fmt = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
        print(f"   accuracy @{GATE}: {r.accuracy:.1f}%   (min {c.min_accuracy:g}%)")
        print(f"   question only:    {fmt(r.accuracy_vague, '.1f')}%   (definitions must not make it worse)"
              f"   AUROC {fmt(r.auroc_vague, '.3f')}   Brier {fmt(r.brier_vague, '.3f')}   ECE {fmt(r.ece_vague, '.3f')}")
        print(f"   calibration ECE:  {fmt(r.ece, '.3f')}   (max {c.max_ece:g})   AUROC {fmt(r.auroc, '.3f')}   Brier {fmt(r.brier, '.3f')}")
        print(f"   flips between runs: {'-' if r.flip_rate is None else f'{100 * r.flip_rate:.1f}%'}   (max {100 * c.max_flip_rate:g}%)")
    for reason in r.reasons:
        print(f"   - {reason}")


async def run(names: list[str], criteria: Criteria, quick: bool, json_path: str | None) -> int:
    s = load_settings()
    names = names or list(s.models)
    if not names:
        print("no models configured: create hunch.toml or set HUNCH_BACKEND_MODEL")
        return 2
    headers = {"Authorization": f"Bearer {s.backend_api_key}"} if s.backend_api_key else {}
    reports = []
    async with httpx.AsyncClient() as client:
        engine = Engine(s, client)
        print(f"backend: {s.backend_url}   ({len(build())} labelled pairs, {'1 run' if quick else '2 runs'} + a vague run per model)")
        try:
            served = await served_models(client, s, headers)
        except Exception as e:  # noqa: BLE001
            print(f"cannot reach the backend: {e}")
            return 2
        for name in names:
            report = await qualify_model(engine, client, name, served, headers, criteria, quick)
            print_report(report, criteria)
            reports.append(report)
            # A model with no non-thinking mode can't be judged on its first token. Re-run it the way it
            # can actually answer, and report both, so the trade-off is visible.
            if report.opens_scratchpad and not report.qualified and not report.unavailable:
                spec = s.resolve(name)
                print(f"\n   re-running {name} in deliberate mode (it thinks first; ~{spec.think_budget} tokens per check) ...")
                deliberate = await qualify_model(engine, client, name, served, headers, criteria, quick,
                                                 spec_override=replace(spec, mode="deliberate"),
                                                 label=f"{name} (deliberate)")
                print_report(deliberate, criteria)
                reports.append(deliberate)
    if json_path:
        with open(json_path, "w") as f:
            json.dump({"criteria": asdict(criteria), "reports": [asdict(r) for r in reports]}, f, indent=1)
    ok = all(r.qualified for r in reports)
    print(f"\n{'ALL QUALIFIED' if ok else 'NOT ALL QUALIFIED'}: "
          + ", ".join(f"{r.model}={'yes' if r.qualified else ('unavailable' if r.unavailable else 'no')}" for r in reports))
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m hunch qualify", description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="*", help="configured model names (default: all)")
    ap.add_argument("--min-accuracy", type=float, default=Criteria.min_accuracy)
    ap.add_argument("--max-ece", type=float, default=Criteria.max_ece)
    ap.add_argument("--max-flip-rate", type=float, default=Criteria.max_flip_rate)
    ap.add_argument("--quick", action="store_true", help="one named run instead of two (no stability check)")
    ap.add_argument("--json", dest="json_path", help="also write the reports to this JSON file")
    a = ap.parse_args(argv)
    return asyncio.run(run(a.models, Criteria(a.min_accuracy, a.max_ece, a.max_flip_rate), a.quick, a.json_path))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
