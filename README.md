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
  "usage": {"prompt_tokens": 465, "completion_tokens": 3, "backend_calls": 3},
  "latency_ms": 621
}
```

That's a real response from Qwen3.5-397B. `p_yes` is 0.59 because the customer reports a double charge but never
explicitly asks for money back. Hunch reports that uncertainty instead of guessing, and your code decides what 0.59 means.

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

Hunch ships a benchmark (`bench/`) of 240 **fictional, labelled look-alike pairs**: "does NEW replace OLD?" and
"do A and B say the same thing?". It deliberately includes restatements and same-value-different-thing traps,
two runs per model, with yes counted at `p_yes ≥ 0.9`. Results on vLLM with NVFP4 checkpoints, measured 2026-09:

| Model | Accuracy | AUROC | Brier | ECE | Max drift between runs | p50 / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-397B-A17B | **100.0** | 1.000 | 0.017 | 0.049 | 0.150 | 646 / 719 ms |
| Gemma-4-31B-IT | 98.8 | 0.991 | **0.013** | **0.013** | **0.011** | 546 / 615 ms |
| DeepSeek-V4-Flash (debias on) | 96.2 | 0.996 | 0.118 | 0.179 | 0.369 | 734 / 7300 ms |
| Qwen3.6-35B-A3B (AWQ int4, on a Jetson AGX Orin) | 99.6 | 1.000 | 0.055 | 0.116 | 0.263 | 790 / 821 ms |
| GLM-5.3 | 66.7: constrained but indecisive (p_yes stuck at 0.1–0.6) | | | | | |

Fan-out on Qwen3.5 at the default concurrency of 16: 1 check 570 ms, 10 checks 740 ms, 50 checks 2.4 s,
100 checks 4.0 s, a 100-option pick 730 ms.

On the Jetson, Qwen3.6-35B-A3B answers a single check in about **230 ms**, but it has less parallel throughput
(concurrency 8): 10 checks take 1.2 s, 100 checks 9.2 s, and a 100-option pick 1.6 s. It's a good fit for
few-checks-per-request gates on edge hardware.

These are one synthetic task on one cluster. **Measure on your own labelled cases** before a decision depends on
Hunch: `python bench/bench.py accuracy <model>` shows how.

## Writing good checks

- **Name the look-alike.** Say what counts as *no* for the case that looks most like *yes*. "Does NEW
  replace OLD?" with no definition of *no* got 39 of 40 restatements wrong on one model. One sentence in `no_if`
  ("restating the same value, or a different thing") fixed it.
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
| `HUNCH_API_KEYS` | Comma-separated bearer keys. Hunch **refuses** to listen beyond localhost without them (or `HUNCH_ALLOW_NOAUTH=1` behind an authenticating proxy) |

## Deploying

- **Docker:** `docker build -t hunch .`, then run it with `HUNCH_BACKEND_URL`, `HUNCH_API_KEYS` and a mounted `hunch.toml`
  (see [`deploy/docker-compose.yml`](deploy/docker-compose.yml)). The image runs as non-root under `tini`, with a health check.
- **systemd:** [`deploy/hunch.service`](deploy/hunch.service).
- After deploying, run `python -m hunch selftest` (or `docker compose exec hunch python -m hunch selftest`).

## Development

```bash
pip install -e '.[dev]'
pytest -q                                   # offline tests against a fake backend
python -m hunch selftest                    # live pre-flight against your backend
python bench/bench.py accuracy qwen gemma   # live accuracy benchmark
python bench/bench.py fanout qwen           # latency vs number of checks
```

## License

MIT. See [LICENSE](LICENSE).
