"""Image look-alike benchmark: does a vision model judge photos as well as Hunch needs?

    python bench/images.py [model ...] [--quick] [--json out.json] [--pass-urls]

188 freely licensed Wikimedia Commons photos (bench/images_manifest.json: source page, licence and author
for each), labelled by hand for eight yes/no questions a news or monitoring pipeline asks. Every question
comes with its look-alike: a flooded street vs a wet one, a protest vs a concert crowd, a building on fire
vs one lit red by fireworks, earthquake rubble vs a demolition site, and so on.

Scored exactly like `python -m hunch qualify` (accuracy at p_yes >= 0.9, ECE, definitions cost at most 1 point,
flips between two identical runs), against the models configured in hunch.toml. Images are downloaded
once to ~/.cache/hunch-images and sent inline as data URLs, since backends often have no internet access;
--pass-urls sends the Commons URLs instead.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import json
import pathlib
import statistics
import sys
import time

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # run from a checkout without installing
from hunch.config import load_settings  # noqa: E402
from hunch.engine import Engine, HunchError  # noqa: E402
from hunch.qualify import Criteria, Report, accuracy_at_gate, auroc, ece, print_report, verdict  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
CACHE = pathlib.Path.home() / ".cache" / "hunch-images"
UA = {"User-Agent": "hunch-image-benchmark (https://github.com/ihubanov/hunch)"}

# question id -> (question, yes_if, no_if). The no_if names the look-alike, as in hunch/lookalikes.py.
QUESTIONS = {
    "flood": ("Does this photo show flooding?",
              "water covers land, streets or buildings that are normally dry",
              "rain, wet streets or puddles, a river or lake at its normal level, or a place or sign named 'flood' with no floodwater"),
    "protest": ("Does this photo show a protest or demonstration?",
                "people gathered in public to press a political or social demand, e.g. marching or holding placards",
                "a concert, festival or sports crowd; a memorial of flowers or candles; protest slogans painted or posted with no one gathered"),
    "wildfire": ("Does this photo show an active wildfire?",
                 "vegetation is burning: visible flames, or a smoke column rising from a fire in open country",
                 "a sunset or sunrise with an orange or red sky; land that has already burned, with no active fire"),
    "crash": ("Does this photo show a road traffic accident?",
              "a vehicle crashed, overturned, or visibly damaged in a collision",
              "a traffic jam, parked cars, or vehicles stopped on a road with no crash visible"),
    "building_fire": ("Does this photo show a building on fire?",
                      "flames or heavy smoke are coming from a building or structure",
                      "fire stations, firefighters or fire equipment with no fire; buildings lit red or orange by lights or fireworks"),
    "disaster_damage": ("Does this photo show buildings damaged or destroyed by a disaster such as an earthquake or storm?",
                        "collapsed, broken or wrecked buildings and rubble left by a disaster",
                        "planned demolition or construction: machines taking a building down, building sites, excavations"),
    "election": ("Does this photo show an election: people voting, ballots, a ballot box or a polling station?",
                 "voting, ballots, ballot boxes, voting booths or a polling station",
                 "people queuing or waiting for something else, e.g. at a shop, bus stop or ticket office"),
    "press_conference": ("Does this photo show a press conference or public statement by an official or politician?",
                         "a public figure speaking to the press or public, typically at a podium or lectern with microphones",
                         "a lecture, class, TED-style talk or conference presentation"),
}


# --task headline: does the photo show the event in a headline? Headlines name the KIND of event only.
# Specific ones ("Thousands march ... demanding new elections") make most real protest photos a correct
# "no", which is a labelling problem rather than a model error.
HEADLINES = {
    "flood": "Floods hit the area",
    "protest": "Protesters take to the streets",
    "wildfire": "Wildfire burns out of control",
    "crash": "Vehicles wrecked in road crash",
    "building_fire": "Building goes up in flames",
    "disaster_damage": "Disaster leaves buildings in ruins",
    "election": "Voting under way in election",
    "press_conference": "Official makes a statement to the press",
}
HEADLINE_CHECK = {
    "question": "Does the photo show the event described in the headline?",
    "yes_if": "the photo shows that kind of event",
    "no_if": "a similar-looking but different scene (e.g. a wet street for a flood, a concert crowd for a protest, "
             "a demolition for a disaster, a sunset for a wildfire), or a real photo of a different kind of event",
}
# Photos of land after a wildfire are a look-alike for "active wildfire" but arguably show the wildfire
# story; they are left out of the headline task rather than labelled either way.
# A real event photo paired with the headline of a clearly different event (a burning building and "disaster
# leaves buildings in ruins" are not clearly different, so neighbouring disasters are never paired).
WRONG_HEADLINE = {"flood": "press_conference", "protest": "crash", "wildfire": "election", "crash": "protest",
                  "building_fire": "election", "disaster_damage": "press_conference", "election": "wildfire",
                  "press_conference": "flood"}
HEADLINE_SKIP = {"wildfire_06", "wildfire_07", "wildfire_08", "wildfire_19", "wildfire_21"}


def load_items() -> list[dict]:
    return json.loads((HERE / "images_manifest.json").read_text())


def headline_items(items: list[dict]) -> list[dict]:
    """Every photo with its own question's headline (label as before: look-alikes are no), plus every
    yes-photo with a clearly different event's headline (a real event photo for the wrong story: no)."""
    items = [it for it in items if it["id"] not in HEADLINE_SKIP]
    out = [{**it, "headline": HEADLINES[it["question"]]} for it in items]
    for it in items:
        if it["label"] == 1:
            other = WRONG_HEADLINE[it["question"]]
            out.append({**it, "headline": HEADLINES[other], "label": 0, "question": f"{it['question']}->{other}"})
    return out


def fetch_images(items: list[dict]) -> dict[str, str]:
    """id -> data URL, downloading each thumbnail once."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    with httpx.Client(headers=UA, timeout=60, follow_redirects=True) as c:
        for it in items:
            path = CACHE / f"{it['id']}.jpg"
            if not path.exists():
                r = c.get(it["image"])
                r.raise_for_status()
                path.write_bytes(r.content)
                time.sleep(0.1)
            data = path.read_bytes()
            mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
            out[it["id"]] = f"data:{mime};base64," + base64.b64encode(data).decode()
    return out


def check(qid: str, vague: bool) -> dict:
    question, yes_if, no_if = QUESTIONS[qid]
    return {"kind": "yesno", "question": question} if vague else \
        {"kind": "yesno", "question": question, "yes_if": yes_if, "no_if": no_if}


async def run_once(engine: Engine, spec, items, images, vague: bool) -> list[float | None]:
    async def one(it):
        if "headline" in it:
            ctx = {"headline": it["headline"]}
            chk = {"kind": "yesno", "question": HEADLINE_CHECK["question"]} if vague else {"kind": "yesno", **HEADLINE_CHECK}
        else:
            ctx, chk = "", check(it["question"], vague)
        try:
            results, _ = await engine.judge(spec, ctx, {"q": chk}, [images[it["id"]]])
            return results["q"]["p_yes"]
        except HunchError as e:
            print(f"  {it['id']}: {e.code}: {e.message[:120]}")
            return None
    return await asyncio.gather(*[one(it) for it in items])


def score(r: Report, run1, run2, run_vague, labels) -> Report:
    scored = [(p, y) for p, y in zip(run1, labels) if p is not None]
    r.errors = sum(p is None for run in (run1, run2 or [], run_vague) for p in run)
    if not scored:
        r.reasons.append("every request failed")
        return r
    r.accuracy = round(accuracy_at_gate(run1, labels), 1)
    r.accuracy_vague = round(accuracy_at_gate(run_vague, labels), 1)
    r.auroc, r.ece = round(auroc(scored), 3), round(ece(scored), 3)
    r.brier = round(statistics.mean((p - y) ** 2 for p, y in scored), 3)
    sv = [(p, y) for p, y in zip(run_vague, labels) if p is not None]
    if sv:
        r.auroc_vague, r.ece_vague = round(auroc(sv), 3), round(ece(sv), 3)
        r.brier_vague = round(statistics.mean((p - y) ** 2 for p, y in sv), 3)
    if run2:
        both = [(a, b) for a, b in zip(run1, run2) if a is not None and b is not None]
        r.flip_rate = round(sum((a >= 0.5) != (b >= 0.5) for a, b in both) / len(both), 4) if both else None
    return r


def per_question(items, ps) -> str:
    wrong = collections.Counter()
    total = collections.Counter(it["question"] for it in items)
    for it, p in zip(items, ps):
        if p is None or (p >= 0.9) != (it["label"] == 1):
            wrong[(it["question"], "missed yes" if it["label"] else "false yes")] += 1
    lines = []
    for q in dict.fromkeys(it["question"] for it in items):
        miss, fy = wrong[(q, "missed yes")], wrong[(q, "false yes")]
        lines.append(f"     {q:33s} {total[q] - miss - fy:3d}/{total[q]:<3d}  missed yes {miss:2d}  false yes {fy:2d}")
    return "\n".join(lines)


async def main_async(a) -> int:
    s = load_settings()
    names = a.models or list(s.models)
    photos = load_items()
    images = {it["id"]: it["image"] for it in photos} if a.pass_urls else fetch_images(photos)
    items = headline_items(photos) if a.task == "headline" else photos
    labels = [it["label"] for it in items]
    print(f"backend: {s.backend_url}   ({len(photos)} labelled photos, task {a.task}: {len(items)} checks)")
    reports, rows = [], {}
    async with httpx.AsyncClient() as client:
        engine = Engine(s, client)
        for name in names:
            spec = s.resolve(name)
            r = Report(model=name, backend_model=spec.backend_model, mode=await engine.mode_for(spec))
            t0 = time.perf_counter()
            run1 = await run_once(engine, spec, items, images, vague=False)
            run2 = None if a.quick else await run_once(engine, spec, items, images, vague=False)
            run_vague = await run_once(engine, spec, items, images, vague=True)
            r = verdict(score(r, run1, run2, run_vague, labels), Criteria())
            r.seconds = round(time.perf_counter() - t0, 1)
            print_report(r, Criteria())
            print(f"   per question (named, at p_yes >= 0.9):\n{per_question(items, run1)}")
            reports.append(r)
            rows[name] = {"named": run1, "named_2": run2, "vague": run_vague}
    if a.json_path:
        pathlib.Path(a.json_path).write_text(json.dumps(
            {"reports": [r.__dict__ for r in reports], "ids": [it["id"] for it in items], "p_yes": rows}, indent=1))
    return 0 if all(r.qualified for r in reports) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="*", help="configured model names (default: all)")
    ap.add_argument("--task", choices=["photo", "headline"], default="photo",
                    help="photo: a yes/no question about the photo. headline: does the photo show the event in a "
                         "headline (look-alike photos, and real photos of a different event, are no)")
    ap.add_argument("--quick", action="store_true", help="one named run instead of two (no stability check)")
    ap.add_argument("--json", dest="json_path")
    ap.add_argument("--pass-urls", action="store_true", help="send Commons URLs instead of inline data URLs")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
