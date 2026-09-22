# Why not a small, purpose-trained decision model?

Hunch borrows a general LLM you already serve and reads a decision out of one constrained token. The
obvious alternative is a model *trained* to answer typed questions, which would be far smaller and
faster. [Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations, Apache-2.0) is exactly
that: 322M–421M encoder models with decision heads, the same three question types Hunch uses
(`noul` / `choice` / `score`), a single forward pass, no generated text. Its authors report 33 ms per
question on a T4 and 7.2 ms batched, and an [MLX port](https://github.com/mizorewww/laya-mlx) reports
7–13 ms on an M3 Max.

So we measured it on Hunch's own benchmark. Results are reported **per checkpoint**: Laya ships three,
they are trained for different things, and a verdict on one is not a verdict on the project. The claim
here is narrow — *this checkpoint, on these 240 pairs, through the corrected criteria path* — and the
caveats at the end matter.

## What was measured

The 240 fictional labelled look-alike pairs that ship with Hunch
([`hunch/lookalikes.py`](../hunch/lookalikes.py)): "does NEW replace OLD's value for the same thing?"
and "do A and B assert the same value?", deliberately including restatements and
same-value-different-thing traps. Each pair is one `noul` question. Two variants, exactly as
`python -m hunch qualify` runs them:

- **named** — the question plus definitions of what counts as yes and as no
- **vague** — the bare question

Reproduce with [`bench/laya_compare.py`](../bench/laya_compare.py), which calls Laya's Python API
directly and scores it with Hunch's own metric code:

```bash
pip install laya "hunch @ git+https://github.com/ihubanov/hunch"
python bench/laya_compare.py --model convaiinnovations/laya-typed-decisions --device cuda --json laya.json
```

## Results

Laya 0.3.5 on a single edge-class GPU, deterministic (0 flips between runs, 0 errors). Hunch's criteria:
accuracy at the `p_yes ≥ 0.9` gate ≥ 90%, ECE ≤ 0.15, definitions must not make it worse, ≤ 2% flips.
The same runs on CPU gave the same verdicts and the same numbers to within 0.4 points, so the device
changes nothing material.

| Checkpoint | Verdict | Accuracy (named) | Accuracy (vague) | AUROC | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| `laya-typed-decisions` (421M) | ❌ Not qualified | **68.8** | 66.7 | 0.759 | 0.224 | **0.207** |
| `laya` (421M, English) | ❌ Not qualified | 72.1 | 72.5 | 0.757 | 0.315 | 0.332 |
| `laya-multilingual` (322M) | ❌ Not qualified | 63.3 | 63.8 | 0.696 | 0.371 | 0.375 |
| *For reference*, Hunch on Qwen3.8-27B | ✅ Qualified | 99.6 | 80.0 | 1.000 | 0.030 | 0.058 |
| *For reference*, Hunch on Qwen3.5-9B | ❌ Not qualified | 90.8 | 81.7 | 0.996 | 0.192 | 0.266 |

Qwen3.5-9B is in that table because it *also* fails, on calibration. The ladder on this benchmark runs
68.8% (Laya's best) → 90.8% (a 9B LLM, still short) → 99.6% (a 27B LLM, qualified).

**Purpose-training beats raw generality at small size**, which is the clearest point in Laya's favour
here. The smallest general LLM we have measured, Qwen3-0.6B, scored 35.8% on these same 240 pairs
against `laya-typed-decisions`'s 68.8%, and 66.7% on the bare question — exactly the score you get by
answering "no" to everything. That measurement predates this harness, though: it has no ECE, AUROC,
Brier or stability figures and no concurrency-1 latency, so it is quoted here as context rather than
given a row in the table above.

**Checkpoint choice matters, and the purpose-built one is the best of the three.**
`laya-typed-decisions` has by far the best calibration of the three (ECE 0.207 against 0.332 and
0.377) and is the only one where naming the look-alikes *helped* (66.7% → 68.8%) rather than hurt. So
the "definitions make it worse" signature seen on the other two is a property of those checkpoints,
not of Laya. It still misses the accuracy gate by 21 points and the calibration bar by 0.057.

What no checkpoint manages is **separating the traps**: the best AUROC is 0.759, against 0.996 for the
smallest LLM in the table above (Qwen3.5-9B) and 1.000 for Qwen3.8-27B. That is not a threshold
problem — no gate placement rescues a ranking that weak.

## Speed, measured on the same GPU

Speed is Laya's main claim, so it deserves a like-for-like number rather than its authors' hardware
against ours. Both of these ran on the same machine and the same GPU (one edge-class board), one request at a time,
nothing else using the GPU, first call discarded:

| | Per question | Accuracy | AUROC | What it is |
| --- | --- | --- | --- | --- |
| `laya-typed-decisions` | **62 ms** | 68.8% | 0.759 | 421M encoder, one forward pass, PyTorch loop |
| `laya` (English) | 60 ms | 72.1% | 0.757 | as above |
| `laya-multilingual` | **50 ms** | 63.3% | 0.696 | 322M encoder |
| Hunch on Qwen3.5-9B | 141 ms | 90.8% | 0.996 | 9B LLM, one constrained decode step, vLLM over HTTP |

So on identical hardware the purpose-built model is about **2.3× faster** than a 9B LLM, not the order
of magnitude its 33 ms T4 figure might suggest next to a server-side number — and it is 22 accuracy
points short, with far weaker ranking. Read those timings with care:

- **Different stacks.** Laya is an in-process PyTorch loop; Hunch's figure includes a vLLM server, HTTP
  and a prefix-cache hit. Neither is tuned for the other's shape.
- **Different work per call.** One encoder forward pass versus one constrained decode step on a model
  20× larger.
- **Concurrency 1 only.** Batched throughput is a separate question: the same 9B measured 790 ms per
  request at 8 in flight, which is worse per request but far better per second.
- The Laya figures are medians of three runs (59–62 ms, 58–60 ms, 49–50 ms); the script reports a median
  per run rather than a distribution, so there is no p95 for them. The 9B's p95 was 154 ms over 34 calls.

## A methodology bug we made, and what it cost

Our first run appended the yes/no definitions to the instructions as prose, because Laya's docstring
lists `criteria` only for `choice` and `score`. That was wrong: `agent._to_internal` passes `criteria`
through for **every** question type, and Laya's own presets use `noul` criteria. Same checkpoint, same
pairs, same machine:

| | Definitions as prose | Laya's native `criteria` |
| --- | --- | --- |
| Accuracy (named) | 68.8 | **72.1** |
| AUROC | 0.696 | **0.757** |
| ECE | 0.368 | **0.332** |
| Accuracy (vague, control) | 72.5 | 72.5 |

The handicap was real and worth about 3 points; the control variant was unchanged, as it should be.
Correcting it did not change any verdict. The current script uses the native field.

## A suggestion from the project, tested

A reply on [our issue](https://github.com/NandhaKishorM/laya/issues/135) suggested our context keys were
suboptimal: we used bare `{"old": ..., "new": ...}`, and descriptive keys such as
`{"old_assertion": ..., "new_assertion": ...}` were recommended so the tokenizer keeps field boundaries.
Same 240 pairs, same machine, `bench/laya_compare.py --semantic-keys`:

| Checkpoint | Accuracy (named) | Accuracy (vague) | AUROC | ECE |
| --- | --- | --- | --- | --- |
| `laya-typed-decisions`, bare keys | 68.8 | 66.7 | 0.759 | 0.207 |
| `laya-typed-decisions`, descriptive keys | 68.8 | 66.7 | 0.734 | 0.204 |
| `laya`, bare keys | 72.1 | 72.5 | 0.757 | 0.331 |
| `laya`, descriptive keys | 67.5 | 72.1 | 0.691 | 0.304 |

It does not help here. On the purpose-built checkpoint the verdict-relevant numbers are unchanged and ranking is
slightly worse; on the English checkpoint accuracy and ranking are clearly worse. The largest movement any key
naming produced is about 4.6 points, against a 21-point gap to the qualification gate. Both runs were deterministic
with no errors, and the flag is in the repo so either variant can be reproduced.

## Caveats

- **One benchmark, and it's ours.** These pairs are deliberately adversarial about near-identical
  meaning. Laya's published results emphasise triage, routing, spam and phishing, which we did not
  test. "Fails our look-alike set" is not "is a weak model".
- **Our timings are ours.** Measured on one edge-class GPU, not the T4 or M3 Max their figures use, so
  they are not a check of their published numbers — only of how the two approaches compare on one
  machine we control.
- **Defaults only.** No router, presets, fine-tuning or prompt tuning of the kind the upstream project
  offers. A practitioner who knows the model would likely do better.
- **Context is not a factor.** Both encoders are ModernBERT with `max_position_embeddings` 8192; the
  512 / 1,024 figures quoted for these checkpoints are Laya's per-request truncation. None of the 240
  pairs comes near either bound.
- **A calibration warning appears on load** for `laya` and `laya-typed-decisions` (identical text and
  value: clamping `choice:11+`), though not for `laya-multilingual`. Those checkpoints ship temperature
  values outside the expected range, which the library clamps, and it says confidence from the affected
  bucket is uncalibrated. The named bucket is not the `noul` path used here, but since our finding is
  about calibration, it is worth recording.
- **Different category.** Laya is a Python library, not an OpenAI-compatible server, so it cannot be a
  Hunch backend today. This is a comparison of approaches, not of two interchangeable parts.

## What we conclude

Per checkpoint, on these pairs: all three fall short, and the purpose-built `laya-typed-decisions` is
the closest. For **this** workload — subtle same-or-different judgments, where the whole point is a
probability you can threshold — borrowing a modern general LLM still wins by a wide margin, and the
cost is latency and a GPU you were already running.

A small purpose-trained model that passed `qualify` would be strictly better than what Hunch does
today: far cheaper, far faster, runnable on a laptop or a single small board. We would happily add a
backend for one. On this benchmark, these checkpoints are not it.

If we have measured Laya unfairly, we would rather fix it than leave it standing: the script and the
data are in this repo, and corrections are welcome as issues or pull requests.
