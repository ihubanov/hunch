# Hunch

**Calibrated yes/no, pick-one and scale judgments from your own LLMs, read straight off the logprobs.**

A lot of code needs a small judgment that plain logic can't express: *is this message asking for a refund?
Which team should handle this ticket? How severe is this alert? Does this new fact replace the old one?*
The usual answer is a pile of `if/else` branches and regexes, or a chat prompt whose free-text answer you then
parse. Hunch replaces both with one typed call. You describe the check, and it returns a probability your code
can threshold, not a paragraph.

- **Typed checks:** `yesno` → `p_yes`, `pick` (one of up to 300 options) → the best option and a full
  distribution, `scale` (up to 10 ordered levels) → an expected value and a distribution. Every answer carries a
  `confidence`.
- **No text generation.** Each check is a single constrained output token (`max_tokens: 1`), and the answer
  is read from the token logprobs and renormalised over the allowed labels. The model can't ramble, and a
  malformed answer can't make it into your code.
- **Parallel by default.** All checks in a request run concurrently, with the context first in every prompt so the
  server's prefix cache is reused.
- **Your models, your hardware.** Hunch runs against any [vLLM](https://github.com/vllm-project/vllm)
  OpenAI-compatible server. No data leaves your network.

## Quick start

```bash
pip install .            # or: docker build -t hunch .
export HUNCH_BACKEND_URL=http://localhost:8000          # your vLLM server
export HUNCH_BACKEND_MODEL=Qwen/Qwen3.5-397B-A17B        # a model it serves
python -m hunch selftest                                  # must print SELFTEST PASSED
python -m hunch                                           # serves http://127.0.0.1:8791
```

```bash
curl -s localhost:8791/v1/judge -H 'content-type: application/json' -d '{
  "context": {"ticket": "Charged twice for order A-104. Fix it today or I cancel."},
  "checks": {
    "refund": {"kind": "yesno", "question": "Is the customer asking for money back?",
               "yes_if": "asks for a refund or reversal of a charge",
               "no_if": "anything else, including only reporting a problem"},
    "team":   {"kind": "pick", "question": "Which team handles this?",
               "options": {"billing": "charges, refunds", "technical": "bugs", "sales": "pricing"}},
    "anger":  {"kind": "scale", "question": "How angry is the customer?",
               "levels": ["calm", "annoyed", "furious"]}
  }
}'
```

```json
{
  "model": "qwen3.5",
  "results": {
    "refund": {"kind": "yesno", "p_yes": 0.5927},
    "team":   {"kind": "pick", "pick": "billing", "probs": {"billing": 0.9997, "technical": 0.0002, "sales": 0.0001}, "confidence": 0.9975},
    "anger":  {"kind": "scale", "value": 1.647, "probs": [0.0027, 0.3477, 0.6496], "confidence": 0.3963}
  },
  "usage": {"prompt_tokens": 465, "completion_tokens": 3, "backend_calls": 3}
}
```

That's a real response from Qwen3.5-397B (lightly trimmed). `p_yes` is 0.59 because the customer reports a double charge but never
explicitly asks for money back. Hunch reports that uncertainty instead of guessing, and your code decides what 0.59 means.

## Qualify your model first

Not every model makes a good Hunch backend. In our tests the older Qwen3 generation was **confidently wrong**, and
spelling out definitions made it *worse*. Check before you rely on a model:

```bash
python -m hunch qualify qwen            # one or more model names from hunch.toml (default: all)
```

It runs straight against your backend (no Hunch server needed) and uses the 240 fictional look-alike pairs that ship
with Hunch ([`hunch/lookalikes.py`](hunch/lookalikes.py)): two runs with the look-alikes named, plus one with the
bare question. That's about 720 one-token calls, roughly a minute at the default concurrency. A model is
**QUALIFIED** when:

| Criterion | Default | Why |
| --- | --- | --- |
| Backend setup | constraint enforced, logprobs returned | otherwise answers are unconstrained |
| Accuracy at `p_yes ≥ 0.9` | ≥ 90% (`--min-accuracy`) | the gate you'd actually use |
| Calibration error (ECE) | ≤ 0.15 (`--max-ece`) | a wrong answer mustn't look as certain as a right one |
| Definitions help | named ≥ question-only | if better-written checks make it worse, you can't fix it by writing better checks |
| Stability | ≤ 2% of answers flip between identical runs (`--max-flip-rate`) | the same input should get the same decision |

A backend that is down is reported as **UNAVAILABLE**, not as a failed model. The exit code is 0 only if every
model qualifies, and `--json FILE` writes the full report. `--quick` does one named run (no stability check).

`python -m hunch selftest` is the 10-second version (setup plus 12 cases). Run it after every deployment. Use
`qualify` when choosing or upgrading a model.

## API

### `POST /v1/judge`

| Field | Type | |
| --- | --- | --- |
| `context` | string, object or array | The data to judge. Use an object with named fields when there are several parts, and refer to them in questions with backticks, e.g. `` `ticket.messages[0]` `` |
| `checks` | object: id → check | Your own ids; results come back under the same ids |
| `model` | string, optional | One of the configured models; defaults to `service.default_model` |

Check kinds:

| `kind` | Fields | Result |
| --- | --- | --- |
| `yesno` | `question`, optional `yes_if`, `no_if` | `p_yes` (0–1) |
| `pick` | `question`, `options`: key → description or `null` (1–300 options) | `pick` (most likely key), `probs` (key → p), `confidence` |
| `scale` | `question`, `levels`: list of descriptions, lowest first (1–10) | `value` (expected level, can fall between levels), `probs` (list), `confidence` |

`confidence` = 1 − normalised entropy of the distribution: 1.0 means all probability is on one answer, 0.0 means uniform.

Errors come back as `{"error": {"code": "...", "message": "..."}}`:

| HTTP | `code` | When |
| --- | --- | --- |
| 400 | `invalid_request`, `unknown_model`, `too_many_options`, `too_many_levels` | Bad input (unknown fields are rejected) |
| 401 | `unauthorized` | `HUNCH_API_KEYS` is set and the bearer key is missing or wrong |
| 413 | `context_too_long` | The context plus the question exceed the model's context window |
| 502 | `backend_error` | The backend returned an error, or did **not** enforce the constraint (never turned into a made-up probability) |
| 503 | `backend_unavailable` | Still failing after retries (429 / 5xx / timeouts, with `Retry-After` honoured) |

`GET /v1/models` lists the configured models. `GET /health` checks that the backend serves them.

## Why logprobs instead of asking for JSON

The common way to get a decision out of an LLM is to ask for JSON (`{"is_refund": true}`) and parse it. That works,
but you only get the model's *top* answer, with no sense of how close the call was. You pay for the output tokens,
reasoning models may think for hundreds of tokens first, and every so often the output isn't valid.

Hunch asks for exactly **one constrained token** and reads the probability the model put on each allowed answer:

| | Ask for JSON | Hunch |
| --- | --- | --- |
| What you get | one answer (`true`) | a probability (`p_yes: 0.93`) plus a distribution for picks and scales |
| Close calls | invisible: 51% and 99% look the same | visible, so you can route 0.4–0.6 to a human or a bigger model |
| Output tokens | tens to hundreds (more with thinking) | 1 |
| Malformed output | possible, needs retries or repair | impossible: the grammar only allows the labels |
| Thresholds | fixed by the model's wording | yours: act at 0.9, review at 0.5–0.9, drop below 0.5 |
| Many questions | one long prompt, and answers influence each other | independent parallel checks over the same cached context |

The probability is what makes this useful in code: you pick the threshold per decision, based on how costly a
mistake is. That's a plain `if p_yes >= 0.9`, not a prompt tweak.

## Using it from Python

No client library is needed; it's one POST:

```python
import httpx

HUNCH = "http://127.0.0.1:8791"

def judge(context, checks, model=None):
    r = httpx.post(f"{HUNCH}/v1/judge", json={"context": context, "checks": checks, "model": model}, timeout=60)
    r.raise_for_status()
    return r.json()["results"]

res = judge(
    {"old": "The staging DB listens on port 5432.", "new": "Staging DB moved to port 6432."},
    {"replaces": {"kind": "yesno",
                  "question": "Does `new` replace `old`'s value for the same thing?",
                  "yes_if": "same thing, changed value",
                  "no_if": "the same value restated, or a different thing"}},
)
p = res["replaces"]["p_yes"]
if p >= 0.9:
    ...        # act automatically
elif p >= 0.5:
    ...        # queue for review
else:
    ...        # leave as is
```

Patterns that work well:

- **Fan out.** Put every question you *might* need about one context in a single request. They run in parallel and
  share the cached context, so the questions your code ends up ignoring cost little.
- **Choose from candidates, don't generate.** Find candidates with code (regex, search, a list of IDs) and ask a
  `pick` which one fits. The answer is always one of your candidates.
- **Decompose.** Rather than one "how good is this?" `scale`, ask three narrow `scale`s and weight them in code.
  When your priorities change, you change a weight, not a prompt.
- **Gate expensive work.** Use a cheap `yesno` ("does this passage answer the question at all?") before sending
  anything to a big generative model.

## Choosing a model

From the bundled benchmark (see [Measured](#measured)):

- **Most accurate:** a large instruct model (Qwen3.5-397B: 100%). Use it for decisions where errors are costly.
- **Best mid-size:** Qwen3.8-27B (99.6%, calibration error 0.058). It qualifies comfortably on a single GPU.
- **Why not a small purpose-trained decision model?** We measured one (Laya, 322M-421M, Apache-2.0):
  on this benchmark its best checkpoint reaches 68.8% and is ~2.3x faster than a 9B LLM on the same GPU.
  Method, numbers and caveats: [docs/COMPARISON-laya.md](docs/COMPARISON-laya.md).
- **Generation beats size.** Every Qwen3-generation model failed (Qwen3-14B scored *below* Qwen3-8B), while the newer
  Qwen3.5+ models pass. The small Qwen3.5-9B ranks almost perfectly (AUROC 0.996) but hedges on look-alikes, so use it
  only with a strict act threshold (≥ 0.97) and human review below that.
- **Best calibrated per GPU:** a mid-size dense model (Gemma-4-31B: 98.8%, calibration error 0.013). Its
  probabilities are the most trustworthy as probabilities, and it's light enough to run next to other workloads.
- **Edge:** a small MoE (Qwen3.6-35B-A3B, 4-bit: 99.6%). It fits on edge hardware and
  suits requests that ask only a few checks.
- **Position-biased models** (DeepSeek-V4-Flash here) work with `debias = true`, at twice the calls.
- **Check the model has a real non-thinking mode.** GLM-5.3 has none: its template always opens a thinking block, so
  its first token is never an answer and it never clears a 0.9 gate — though it reaches 96.7% with ECE 0.033 when
  allowed to think (see [Models with no non-thinking mode](#models-with-no-non-thinking-mode)). Run `python -m hunch selftest` and `python bench/bench.py accuracy <model>` before adopting a model.
- **Thinking models must have thinking switched off per request.** Otherwise the one allowed token is spent on
  reasoning. Hunch sends `reasoning_effort: "none"` by default. Some chat templates need
  `chat_template_kwargs = { enable_thinking = false }` in the model's `extra_body` instead (see `hunch.toml.example`).

## When not to use Hunch

- **Generating text** (summaries, replies, extraction of free-form values). Use a generative model, or have code
  find candidates and let Hunch `pick`.
- **Arithmetic, counting, date comparison, exact matching.** Do it in code.
- **Multi-step reasoning.** A check is a snap judgment over the context you give it. Split the problem, or use a
  reasoning model.
- **Security boundaries.** Context is data, but a model can still be steered by adversarial text inside it. Don't make
  a Hunch check the only thing standing between untrusted input and a dangerous action.

## How it works

1. **Labels, not text.** Answers map to single-token labels: `Y`/`N`, letters `A`… for options, and digits `0`–`9`
   for levels. Each check is sent as a chat completion with
   `structured_outputs: {"choice": [labels]}`, `logprobs: true`, `max_tokens: 1`, `temperature: 0` and
   `reasoning_effort: "none"`.
2. **Probabilities from logprobs.** The label logprobs are exponentiated and renormalised over the allowed labels.
   `scale.value` is the expectation Σ i·p(i).
3. **Big picks.** vLLM returns at most `--max-logprobs` alternatives (20 by default), so a pick with more than 15
   options is split into groups. One call picks the group and one call per group picks within it, all in parallel:
   p(option) = p(group) × p(option | group). 300 options take 21 calls, 100 options take 8.
4. **Position-bias correction** (`debias = true` per model). Some models lean toward whichever answer is listed
   first. With debias on, yes/no checks are asked in both answer orders and small picks in both option orders, and
   the results are averaged. That doubles the calls, so only enable it for models that need it.

**vLLM gotcha:** use `structured_outputs`. Some vLLM versions silently **ignore** the legacy `guided_choice`
parameter, so the model answers unconstrained and nothing tells you. `python -m hunch selftest` checks that the
constraint is really enforced (it asks the model to write "hello" while restricted to `A`/`B`).

## Measured

Hunch ships 240 **fictional, labelled look-alike pairs** ([`hunch/lookalikes.py`](hunch/lookalikes.py)): "does NEW replace OLD?" and
"do A and B say the same thing?". It deliberately includes restatements and same-value-different-thing traps,
two runs per model, with yes counted at `p_yes ≥ 0.9`. All results below use the checks in
[`hunch/lookalikes.py`](hunch/lookalikes.py), which **name the look-alike cases** in `no_if` (e.g. *"`new` restates the same
value, or is about a different thing"*). Results on vLLM (NVFP4 checkpoints unless noted), measured 2026-09:

| Model | Verdict | Accuracy | AUROC | Brier | ECE | Max drift between runs |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-397B-A17B | ✅ Qualified | **100.0** | 1.000 | 0.017 | 0.049 | 0.150 |
| Qwen3.8-27B (bf16) | ✅ Qualified | 99.6 | 1.000 | 0.030 | 0.058 | – (0 flips) |
| Qwen3.6-35B-A3B (AWQ int4) | ✅ Qualified* | 99.6 | 1.000 | 0.055 | 0.116 | 0.263 |
| Gemma-4-31B-IT | ✅ Qualified | 98.8 | 0.991 | **0.013** | **0.013** | **0.011** |
| DeepSeek-V4-Flash (debias on) | ❌ Not qualified | 96.2 | 0.996 | 0.118 | 0.179 | 0.369 |
| Qwen3.5-9B (bf16) | ❌ Not qualified | 90.8 | 0.996 | 0.192 | 0.266 | – (0 flips) |
| Qwen3-8B (bf16) | ❌ Not qualified | 76.7 | 0.838 | 0.238 | 0.242 | 0.169 |
| Qwen3-14B (bf16) | ❌ Not qualified | 71.7 | 0.806 | 0.286 | 0.289 | 0.195 |
| GLM-5.3 | ❌ Not qualified | 66.7 | – | – | – | – |
| Qwen3-0.6B (bf16) | ❌ Not qualified | 35.8 | 0.758 | 0.611 | 0.630 | 0.024 |

Verdicts use `python -m hunch qualify`'s default criteria (accuracy ≥ 90%, ECE ≤ 0.15, definitions must not hurt,
≤ 2% flips). Qwen3.5-397B, Qwen3.8-27B, Gemma-4-31B and Qwen3.5-9B were run through `qualify` itself; for the others the
same criteria are applied to their benchmark numbers. \* Qwen3.6-35B-A3B passes on accuracy, calibration and stability;
its question-only run wasn't measured.

Why the others fail:
- **DeepSeek-V4-Flash:** calibration.
- **Qwen3.5-9B:** calibration. Real positives sit at p≈0.99, but the look-alike traps land at a median of 0.42–0.65, so
  it hedges rather than being confidently wrong.
- **The Qwen3 generation:** confidently wrong, and definitions make it *worse*.
- **GLM-5.3:** has no non-thinking mode, so its first token opens a scratchpad rather than answering. It reaches
  96.7% (ECE 0.033) when allowed to think. See [Models with no non-thinking mode](#models-with-no-non-thinking-mode).

### Same benchmark, vague checks

The same 240 pairs with only the `question` ("Does `new` replace `old`'s value for the SAME thing?"), with no `yes_if` /
`no_if` (`python bench/bench.py accuracy --vague <model>`):

| Model | Verdict | Accuracy, look-alikes named | Accuracy, question only | AUROC, question only | Brier, question only |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-397B-A17B | ✅ Qualified | 100.0 | **83.3** | 0.884 | 0.182 |
| Qwen3.8-27B | ✅ Qualified | 99.6 | **80.0** | – | – |
| Gemma-4-31B-IT | ✅ Qualified | 98.8 | **80.4** | 0.863 | 0.196 |
| DeepSeek-V4-Flash (debias on) | ❌ Not qualified | 96.2 | **72.1** | 0.932 | 0.329 |
| Qwen3.5-9B | ❌ Not qualified | 90.8 | **81.7** | – | – |
| Qwen3-8B | ❌ Not qualified | 76.7 | **81.2** (definitions hurt) | – | – |
| Qwen3-14B | ❌ Not qualified | 71.7 | **74.6** (definitions hurt) | – | – |

The model and the service are the same; only the definitions changed. On every current-generation model, two short
definitions (`yes_if` / `no_if`) are worth 9–24 accuracy points. The older Qwen3 models go the other way, and that is
exactly what `qualify`'s "definitions help" check catches. Write the definitions first, then choose the model.

These are results for one synthetic task on our hardware. **Measure on your own labelled cases** before a decision depends on
Hunch: `python bench/bench.py accuracy <model>` shows how.

### Models with no non-thinking mode

GLM-5.3 scores 66.7% here, which looks like a weak model. It isn't — we were measuring the wrong thing.

**GLM-5.3 has no non-thinking mode.** Its chat template always appends a thinking block to the generation prompt, so
the first generated token is the model *opening a scratchpad*, never an answer. Constraining that position to `Y`/`N`
projects a distribution that isn't about the verdict onto two tokens. The result looks exactly like what we measured:
probabilities pinned near 0.5, unstable between identical runs (52 of 240 answers flipped), and very sensitive to how
the prompt is built. There is no flag that avoids this: the GLM-4.x thinking toggle was removed, and
`reasoning_effort: "none"` is not a recognised value — it falls through to *maximum* reasoning while also switching
the server's reasoning parser off.

Let the same model think, and constrain only its final verdict token, and it is excellent:

| How GLM-5.3 is asked | Accuracy | AUROC | ECE |
| --- | --- | --- | --- |
| One constrained token (what Hunch does today) | 68.3% | 0.736 | – |
| Thinking, then the probability read from its constrained verdict token | **96.7%** | **0.975** | **0.033** |

That second row would qualify comfortably. It costs about 122 thinking tokens and a few seconds per decision instead
of one token in well under a second — roughly 100× the tokens for +28 points on this task.

**What to take from this:**

- **Hunch measures something narrow:** does the model have an answer *at the first generated token*? A model whose
  template always starts with a scratchpad cannot, whatever its ability.
- **The symptom is recognisable.** Probabilities clustered near 0.5, unstable between identical runs and moving a lot
  when you reshape the prompt, mean you are reading a position that isn't the answer — not that the model is weak.
  Check whether your model has a real non-thinking mode before blaming it.
- **A reasoning model can still give you a calibrated probability.** Let it think, constrain the final answer to your
  labels, and read the logprobs of that token. Hunch does not do this today (see
  [Deliberate mode](#deliberate-mode-not-implemented)), but nothing about the approach forbids it.

### Deliberate mode (not implemented)

The measurement above suggests an obvious second mode: let the backend think, constrain the verdict, and read the
probability from that token. Same API, same thresholds, same `qualify` criteria — but hundreds of tokens and seconds
per decision instead of one token in well under a second. It would turn models that have no non-thinking mode from
unusable into the most accurate option available. It is not built; if you want it, open an issue.

## Writing good checks

- **Name the look-alike.** Say what counts as *no* for the case that looks most like *yes*. On the bundled
  benchmark, dropping `yes_if` / `no_if` costs 17–24 accuracy points on every model tested
  (see [vague checks](#same-benchmark-vague-checks)). One sentence in `no_if` ("restating the same value, or a
  different thing") is most of the fix.
- **Definitions go in `yes_if` / `no_if` and option descriptions,** not only in the question.
- **One judgment per check.** Split "is this a good candidate?" into several checks and combine them in code.
- **Keep arithmetic, counting, dates and IDs in code.** Give the model only the judgment.
- **Add an escape option** (`"none": "none of the above"`) to picks when nothing may fit. A pick always
  returns *something*.
- **Treat `p_yes` and `confidence` as uncalibrated for your task** until you've checked them against 20–40
  labelled cases.

## Configuration

`hunch.toml` (path in `HUNCH_CONFIG`, or `./hunch.toml`). See [`hunch.toml.example`](hunch.toml.example):

```toml
[backend]
url = "http://localhost:8000"

[service]
default_model = "qwen"
max_concurrency = 16        # simultaneous backend calls
# api_keys = ["change-me"]  # required when listening beyond localhost

[models.qwen]
backend_model = "Qwen/Qwen3.5-397B-A17B"

[models.deepseek]
backend_model = "deepseek-ai/DeepSeek-V4-Flash"
debias = true
```

| Environment variable | Meaning |
| --- | --- |
| `HUNCH_HOST` / `HUNCH_PORT` | Listen address (default `127.0.0.1:8791`) |
| `HUNCH_BACKEND_URL`, `HUNCH_BACKEND_KEY` | vLLM server and optional bearer key |
| `HUNCH_BACKEND_MODEL` | Quick single-model setup without a TOML file (exposed as `default`) |
| `HUNCH_DEFAULT_MODEL` | Overrides `service.default_model` |
| `HUNCH_CONCURRENCY` | Simultaneous backend calls. Keep it modest on a shared server |
| `SSL_CERT_FILE` | CA bundle for a backend behind a private or self-signed CA, e.g. `/etc/ssl/certs/ca-certificates.crt` (Python doesn't use the system store by default) |
| `HUNCH_API_KEYS` | Comma-separated bearer keys. Hunch **refuses** to listen beyond localhost without them (or `HUNCH_ALLOW_NOAUTH=1` behind an authenticating proxy) |

## Deploying

- **Docker:** `docker build -t hunch .`, then run it with `HUNCH_BACKEND_URL`, `HUNCH_API_KEYS` and a mounted `hunch.toml`
  (see [`deploy/docker-compose.yml`](deploy/docker-compose.yml)). The image runs as non-root under `tini`, with a health check.
- **systemd:** [`deploy/hunch.service`](deploy/hunch.service).
- After deploying, run `python -m hunch selftest` (or `docker compose exec hunch python -m hunch selftest`).

## For coding agents

[`SKILL.md`](SKILL.md) is a drop-in guide for an agent writing a Hunch integration: the call, the rules for writing
checks, thresholds, qualifying a model, and what not to do. [`llms.txt`](llms.txt) indexes it for agents that look for
one. Point your agent at the raw URLs:

```
https://raw.githubusercontent.com/ihubanov/hunch/main/llms.txt
https://raw.githubusercontent.com/ihubanov/hunch/main/SKILL.md
```

## Development

```bash
pip install -e '.[dev]'
pytest -q                                   # offline tests against a fake backend
python -m hunch selftest                    # quick live pre-flight against your backend
python -m hunch qualify qwen                 # full qualification of a model (verdict + reasons)
python bench/bench.py accuracy qwen gemma   # live accuracy benchmark
python bench/bench.py fanout qwen           # latency vs number of checks
```

## License

MIT. See [LICENSE](LICENSE).
