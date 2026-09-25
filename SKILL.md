---
name: hunch
description: >
  Use Hunch to get calibrated yes/no, pick-one and scale judgments from a local LLM, read from the
  logprobs of a single constrained token. Use when code needs a semantic decision that plain logic
  can't express - routing, filtering, ranking, verification, gating expensive work - and you want a
  probability to threshold on instead of text to parse. Read this before writing the checks.
---

# Building with Hunch

Hunch turns a typed question about some data into a probability your code can act on. One HTTP call,
one constrained output token per check, no text to parse. Repo: https://github.com/ihubanov/hunch

## The call

```http
POST {HUNCH_URL}/v1/judge
Content-Type: application/json
Authorization: Bearer {key}        # only if the deployment sets HUNCH_API_KEYS
```

```json
{
  "context": {"ticket": "Charged twice for order A-104."},
  "model": "qwen",
  "checks": {
    "refund":  {"kind": "yesno", "question": "Is the customer asking for money back?",
                "yes_if": "asks for a refund or reversal of a charge",
                "no_if": "anything else, including only reporting a problem"},
    "team":    {"kind": "pick", "question": "Which team handles this?",
                "options": {"billing": "charges, refunds", "technical": "bugs", "none": "none of these"}},
    "anger":   {"kind": "scale", "question": "How angry is the customer?",
                "levels": ["calm", "annoyed", "furious"]}
  }
}
```

Add `"images": ["https://...", "data:image/jpeg;base64,..."]` (up to 8) to ask about photos on a vision
model; they become part of the context as `IMAGE 1`…`IMAGE n`, and `context` may then be omitted.

Results come back under your own ids: `yesno` → `p_yes`; `pick` → `pick`, `probs`, `confidence`;
`scale` → `value` (expected level), `probs`, `confidence`. `model` is optional and defaults to the
server's configured model.

## Rules for writing checks

1. **Name the look-alike case.** For every check, ask: what looks like *yes* but is *no*? Put it in
   `no_if` (or in an option description). This is the single highest-value thing you do: vague checks
   cost 17-24 accuracy points on every current model, and "does NEW replace OLD?" without it read
   restatements as replacements 39 times out of 40 on one model. There is often more than one: for
   replace/duplicate checks also name *only adds detail* (a refinement is neither), or a model can call
   refinements replacements at p = 1.0, which no threshold catches.
2. **Definitions belong in `yes_if` / `no_if` and option descriptions,** not only in the question.
3. **One judgment per check.** Split "is this a good candidate?" into separate checks and combine them
   in your code, where you can change a weight without touching a prompt.
4. **Keep arithmetic, counting, dates, IDs and exact matching in code.** Give the model only the part
   that needs language understanding.
5. **Always give a `pick` an escape option** (`"none": "none of the above"`) when nothing may fit. A
   pick always returns something.
6. **Put the data in `context`, never instructions.** Use an object with named fields and refer to them
   in the question with backticks: `` `ticket.messages[0]` ``. Text inside `context` is data; it can
   still try to steer the model, so never let a single check be the only thing standing between
   untrusted input and a dangerous action.
7. **Ask everything about one context in one request.** Checks run in parallel over a shared cached
   context, so speculative checks your code may ignore are cheap.

## Thresholds

The probability is the point. Choose the threshold per decision, based on the cost of being wrong:

```python
if p >= 0.9:    act()            # high confidence
elif p >= 0.5:  queue_review()   # unsure
else:           skip()
```

Treat `p_yes` and `confidence` as uncalibrated for *your* task until you have measured them on 20-40
labelled cases of your own. `confidence` (pick/scale) is 1 - normalised entropy: 1.0 means all the
probability is on one answer, 0.0 means uniform.

## Before trusting a model

```bash
python -m hunch selftest          # 10 seconds: constraint enforced, logprobs, 12 cases
python -m hunch qualify <model>   # ~1 minute: verdict with reasons, on 240 labelled look-alike pairs
```

`qualify` checks accuracy at the 0.9 gate (≥ 90%), calibration (ECE ≤ 0.15), that naming the
look-alikes does not make the model *worse*, and stability between identical runs. Models that pass on
the bundled benchmark include Qwen3.5-397B, Qwen3.8-27B, Qwen3.6-35B-A3B, Gemma-4-31B, and GLM-5.3 and
DeepSeek-V4.1-Flash in deliberate mode. For images, `python bench/images.py` runs the same criteria on
188 labelled look-alike photos; Gemma-4-31B, Qwen3.5-397B and Qwen3.8-Flash-Next pass.

## What not to do

- **Don't judge a thinking model on one token.** Some have no thinking-off mode (GLM-5.3: 66.7% on one
  token, 97.5% when it thinks first), and some have one that answers worse than it thinks
  (DeepSeek-V4.1-Flash: fails calibration on one token, 98.8% with ~60 thinking tokens). Set
  `mode = "deliberate"` for those; `qualify` tries it automatically and tells you which mode passes.
- **Don't ask Hunch to generate text,** do maths, compare dates, or reason in several steps. Find
  candidates in code and let a `pick` choose between them.
- **Don't use multi-token labels.** Option keys and levels map to single tokens (letters and digits);
  Hunch handles that for you, so don't try to bypass it.
- **Don't read a wrong answer as an unlucky one.** If a check fails often, first check whether the
  look-alike case is named, then whether the model qualifies.
