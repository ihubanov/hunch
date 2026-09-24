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

Laya 0.3.5, deterministic (0 flips between runs, 0 errors). Hunch's criteria: accuracy at the
`p_yes ≥ 0.9` gate ≥ 90%, ECE ≤ 0.15, definitions must not make it worse, ≤ 2% flips. GPU and CPU runs
agreed to within 0.4 points, so the device changes nothing material.

**How the question is asked matters more than anything else here.** Laya's `noul` primitive hardcodes its
option labels to `false:` / `true:`, and [laya#156](https://github.com/NandhaKishorM/laya/issues/156)
reports that on the shipped checkpoints those label *words* can decide the answer regardless of the
state. Our first published numbers were all `noul`. The maintainer suggested the control: ask the same
pairs as a two-option `choice` with neutral `A` / `B` keys and the same definitions as the option
descriptions. We ran it in both key orders and averaged (`--as-choice`), so a preference for the
first-listed option cancels:

| Checkpoint | Asked as | Verdict | Accuracy (named) | Accuracy (vague) | AUROC | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `laya` (421M, English) | **neutral choice** | ❌ Not qualified | **81.2** | 66.7 | **0.953** | 0.215 | 0.301 |
| `laya-typed-decisions` (421M) | **neutral choice** | ❌ Not qualified | **73.8** | 69.2 | **0.904** | 0.189 | 0.236 |
| `laya` (421M, English) | `noul` | ❌ Not qualified | 72.1 | 72.5 | 0.757 | 0.315 | 0.332 |
| `laya-typed-decisions` (421M) | `noul` | ❌ Not qualified | 68.8 | 66.7 | 0.759 | 0.224 | 0.207 |
| `laya-multilingual` (322M) | `noul` | ❌ Not qualified | 63.3 | 63.8 | 0.696 | 0.371 | 0.375 |
| *For reference*, Hunch on Qwen3.8-27B | one token | ✅ Qualified | 99.6 | 80.0 | 1.000 | 0.030 | 0.058 |
| *For reference*, Hunch on Qwen3.5-9B | one token | ❌ Not qualified | 90.8 | 81.7 | 0.996 | 0.192 | 0.266 |

**We got this wrong the first time, and the correction is large.** An earlier version of this document
said Laya sat "barely above a constant-no baseline" with "only weak signal" (AUROC 0.759). With neutral
labels the English checkpoint reaches **AUROC 0.953** and the purpose-built one 0.904. That is a model
that ranks these look-alikes well. Asking through `noul` was suppressing its discrimination, not merely
shaving points off it. Thanks to the Laya maintainer for pointing at the right control rather than
arguing with the numbers.

**What survives the correction: good ranker, unusable probabilities.** Neither checkpoint qualifies,
but the reason has changed. Accuracy at the 0.9 gate is 73.8% / 81.2% against a 90% bar, and ECE is
0.236 / 0.301 against 0.15 — **calibration did not improve at all**, and on `laya-typed-decisions` it
got slightly worse. So the ordering these models produce is informative; the numbers attached to it are
not something you can threshold at 0.9 without calibrating them yourself first.

**Definitions help, once the question is asked in a way the model can answer.** For the English
checkpoint, naming the look-alikes is worth 14.5 points as a neutral choice (66.7% → 81.2%), where under
`noul` it *hurt*. An earlier version of this document said "better-written checks won't fix this model"
about that checkpoint; through a neutral-label choice, better-written checks help it a lot.

**Purpose-training still beats raw generality at small size.** The smallest general LLM we have measured,
Qwen3-0.6B, scored 35.8% on these same pairs. That measurement predates this harness (no ECE, AUROC,
Brier, stability or concurrency-1 latency), so it is context rather than a row in the table.

One limit on the claim: switching to neutral labels in both orders moves accuracy and ranking by this
much on our set. We did not verify that laya#156's label bug is the *mechanism*, only that the control it
implies changes the result.

**Upstream since these runs (laya 0.3.7 → 0.3.20, 23–24 September).** The API half of laya#156 is fixed:
`noul` now rejects criteria keys other than `true` / `false` instead of silently replacing them with
default text ([#249](https://github.com/NandhaKishorM/laya/pull/249)). Our runs always used
`true` / `false`, so they were not affected. A new opt-in `labels` field
([#163](https://github.com/NandhaKishorM/laya/pull/163)) replaces the `true:` / `false:` option words while still
returning P(true), which makes a neutral-word `noul` possible in a single call. The checkpoint bias itself
is unchanged: the maintainer says it needs a retrained checkpoint. The numbers above are 0.3.5 and
stand. A re-run on 0.3.20 with neutral `labels` (`bench/laya_compare.py --noul-labels`) is in progress
and will be added here.

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
of magnitude its 33 ms T4 figure might suggest next to a server-side number. Two caveats on that
comparison now: those timings were measured through `noul`, and the neutral-choice framing that the
accuracy figures above use **doubles the calls** (both key orders), so the like-for-like speed advantage
is roughly halved unless you ask in one order only. Read the timings with care:

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
listed `criteria` only for `choice` and `score` (fixed upstream in 0.3.7 after our report,
[#146](https://github.com/NandhaKishorM/laya/pull/146)). That was wrong: `agent._to_internal` passes `criteria`
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

Per checkpoint, on these pairs, asked the way each model can actually answer: none qualifies, and the
reason is calibration rather than blindness. Asked as a neutral two-option choice, `laya` ranks these
look-alikes at AUROC 0.953 — comparable to a 9B LLM — in a 421M model that runs about 2.3× faster per
call. What it cannot do is hand you a probability you can put behind a 0.9 gate without calibrating it
yourself: ECE 0.236-0.301 against our 0.15 bar, and accuracy at that gate of 73.8-81.2% against 90%.

For **this** workload — subtle same-or-different judgments where the whole point is a thresholdable
probability — borrowing a modern general LLM still wins, and the cost is latency and a GPU you were
already running. But the margin is much smaller than our first numbers suggested, and a calibration
layer on top of Laya's ordering might close it. A small purpose-trained model that passed `qualify`
would be strictly better than what Hunch does today: far cheaper, far faster, runnable on a laptop or a
single small board. We would happily add a backend for one.

If we have measured Laya unfairly, we would rather fix it than leave it standing: the script and the
data are in this repo, and corrections are welcome as issues or pull requests. That is not a slogan —
this document has already been corrected twice that way, once for a bug of our own (prose instead of
native `criteria`) and once for the `noul` label bias, which the maintainer pointed us at.
