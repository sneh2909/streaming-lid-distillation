# Padding-aware short-prefix evaluation

Research cutoff: **2026-09-26**. Local artifacts, primary documentation,
web sources, and Hugging Face Hub metadata were checked on that date. This
note addresses backlog **POLISH-10** only.

## Bottom line

Right-padding a short recording to a requested deadline and then declaring the
padding valid is not a harmless batching operation. It changes the evaluated
waveform:

- the frozen ECAPA teacher is told that every appended zero is real input;
- the student frontend converts the zero tail into log-mel floor frames, and
  the experiment averages the resulting logits as evidence;
- a recording that ended at 3.10 s is consequently reported as a 4 s prefix,
  even though it supplied neither 4 s of speech nor a recorded 0.90 s pause.

The repository should use two distinct protocols:

1. A **fixed evidence deadline** is evaluable only when the source contains at
   least that many samples. Take the exact leading samples and never synthesize
   the missing interval. Report per-language eligible denominators.
2. A short recording is still valid for an **end-of-utterance/full** slice.
   Run its natural waveform, and label it by its actual duration. Do not mix
   that result into the fixed-deadline cell.

Zero-padding remains appropriate for a variable-length *batch* only when the
true sample lengths are retained and consumed by the model. SpeechBrain 1.0.3
explicitly defines `wav_lens` for ignoring padding, and ECAPA's attentive
statistics pool masks padded frames. The current experiment first expands each
short item to four seconds, then records that expanded length, defeating the
mask.

This is primarily an evaluation and reporting correction. Expected model
gain: **none**. The large paired channel effects and the voice experiment's
full-duration primary result are not disproved, but the padded absolute prefix
scores are not release evidence and must be rescored before serving as gates.

## 1. Exact local audit

The audit is bound to these unchanged result files:

| Artifact | SHA-256 |
|---|---|
| `experiments/telephony-robustness/results.json` | `93712aeb7086bf8e712f89441af12dffa0eee08c5b6db5bcdf4db7129d0940b3` |
| `experiments/cross-view-kd/results.json` | `35307a1846ac5c86db2e91d4baf4b949aae2c6ce78966a3f2f5397cf189a2d11` |
| `experiments/voice-text-factorial/results.json` | `f12b5045dfc3816affcf837f9948cdab8321d953b68d6b1d301f8cfb89daaccb` |
| `data/generated/manifest.jsonl` | `7df6fc5d94db4c88cac9df6b2de8bc9dc5fcacf1bf36853f839bf494ad59314c` |

### 1.1 Padding prevalence

| Experiment | Affected cells | Consequence |
|---|---:|---|
| Telephony robustness | 15/21 four-second clips in each of 11 conditions; 165/693 stored monolingual requests | Every advertised 63-request `1/2/4 s` condition aggregate contains 15 padded rows (23.81%). |
| Cross-view KD | 60/84 four-second condition rows; 60/252 prefix rows | Each of four evaluation conditions has 15/21 padded four-second rows, scored under both student arms. |
| Voice/text factorial, fresh | 52/84 four-second clips; 52/420 fresh requests | 61.90% of the four-second curve cell is padded. |
| Voice/text factorial, exact controls | 26/42 four-second clips; 26/210 exact requests | 61.90% of the four-second control cell is padded. |

For the 21 main held-out clips, 15 are shorter than four seconds. The appended
tail totals **6.994 s**, with positive median **0.4282 s** and maximum
**0.9032 s**. Only six clips naturally reach four seconds, and they cover only
English, Hindi, Marathi, and Gujarati. Bengali, Tamil, and Telugu have no
eligible row, so simply dropping the 15 rows does not yield a valid seven-way
four-second release metric.

For the 84 fresh factorial clips, the appended tail totals **22.761875 s**;
the positive median is **0.4096875 s** and the maximum is **1.0443125 s**.
Padding prevalence also differs by voice profile in the four-second cell
(28/42 seen-profile versus 24/42 unseen-profile rows), so the duration/padding
mixture is another component of that curve's renderer-profile bundle.

### 1.2 What the implementations do

`experiments/telephony-robustness/run.py::prefix_audio` and
`experiments/voice-text-factorial/run.py::prefix_waveform` append zeros to the
requested sample count. Teacher batching then computes `lengths` *after* that
operation. Every four-second tensor therefore has relative length `1.0`, even
when the manifest says the original recording ended earlier.

The student path has no length mask. At 16 kHz with a 400-sample window,
160-sample hop, `center=False`, four future feature frames, and 21-frame label
delay, a four-second tensor produces:

```text
feature frames                 = 1 + floor((64000 - 400) / 160) = 398
stable streaming logits        = 398 - 4                         = 394
delay-aligned averaged logits  = 394 - 21                        = 373
```

Every stored four-second row in both telephony robustness and cross-view KD
reports exactly **373** averaged student frames, including the short clips.
Thus the zero tail contributes up to 103–104 additional delay-aligned frames
in the audited data.

Digital zero is not an abstract padding token for this frontend. It becomes
`log(1e-5) = -11.5129...` in each floored mel bin, after which learned biases
and temporal context can produce non-uniform language posteriors. Even a
literal silent tail must be masked unless the experiment is explicitly about
recorded post-speech silence and router behavior.

### 1.3 Stored full-versus-padded comparisons

The voice/text artifact contains both natural `full` and padded `4` requests
under the same scoring code, allowing a read-only paired check.

For the **52 fresh padded clips**:

- ECAPA has 0/52 selected-seven top-1 flips, but posterior L1 distance has
  median `0.0001870` and maximum `0.4560391`;
- the student has 1/52 majority-label flips, posterior L1 median `0.0740151`,
  and maximum `0.3600691`;
- padding adds 2,276 delay-aligned student frames in total, median 41 per clip
  and range 0–104. A clip can be marked padded yet add zero complete feature
  frames when it is only a few samples short.

For the **26 padded exact controls**, ECAPA again has no top-1 flip and the
student has one. The flipped Telugu item overlaps the fresh set, so these are
not two independent failures. Top-1 stability on this small, synthetic,
poorly generalizing checkpoint does not validate padding: the probability
shifts are real, and a future model or decision threshold can cross a boundary.

The full-duration voice-profile endpoint is unaffected: the stored primary
student result remains 28/42 (66.67%) for seen profiles versus 7/42 (16.67%)
for unseen profiles. The four-second row (64.29% versus 19.05%) should be
labelled padded/mixed until rescored; it is not needed to support the 50-point
full-duration finding.

### 1.4 Direct ECAPA length-mask probe

A six-thread, read-only probe loaded the pinned teacher and the 15 main
held-out clips shorter than four seconds. It compared:

1. natural variable-length batching, padded only to the longest natural clip
   (3.9511875 s) with correct relative lengths;
2. a fixed four-second backing tensor with correct original relative lengths;
3. the same four-second tensor with every length falsely set to one, matching
   the current experiment.

Restricted-seven posterior differences versus natural batching were:

| Four-second handling | Top-1 flips | Median L1 | Maximum L1 |
|---|---:|---:|---:|
| Correct `wav_lens` | 0/15 | `1.04e-13` | `0.0016905` |
| All zeros declared valid | 0/15 | `0.0001085` | `0.2159494` |

The small nonzero masked maximum is compatible with feature-length rounding
between different backing widths. This one-batch probe is not an accuracy
study, but it verifies that the documented mask closes almost all of the local
teacher discrepancy. Student masked-rescore outcomes and every downstream
decision change remain **unverified**.

## 2. Correct fixed-deadline contract

Let `n` be the number of source samples, `r` the sample rate, and `d` the
requested evidence deadline in seconds. Define `wanted = round(d*r)`.

```text
eligible_fixed_deadline = (n >= wanted)
evaluated_samples       = waveform[:wanted]        if eligible
                         = no fixed-deadline row    otherwise
```

An ineligible clip should still receive a natural full-duration row with
`actual_audio_seconds=n/r`. It must not be copied into the `d`-second cell.
This keeps two questions separate:

- "How accurate is the system after exactly `d` seconds of available source
  audio?"
- "How accurate is it when this shorter utterance ends?"

If the product question is instead wall-clock behavior after VAD onset, the
input must include the *recorded* gap/noise following speech, and the causal VAD
and router must be replayed. Appending mathematical zeros is neither recorded
silence nor evidence that the live policy would continue averaging language
frames. That separate protocol is currently **unverified**.

## 3. Model-specific padding rules

### 3.1 ECAPA teacher

For an eligible prefix, batch tensors may be padded to the longest item in the
batch. Preserve `true_samples` before padding and pass:

```python
relative_lengths = true_samples / padded_batch_width
teacher.classify_batch(padded_waveforms, relative_lengths)
```

The installed SpeechBrain 1.0.3
[`EncoderClassifier`](https://github.com/speechbrain/speechbrain/blob/v1.0.3/speechbrain/inference/classifiers.py)
passes `wav_lens` to mean/variance normalization and the embedding model. Its
documented contract says those relative lengths are used to ignore padding.
The matching
[`ECAPA_TDNN`](https://github.com/speechbrain/speechbrain/blob/v1.0.3/speechbrain/lobes/models/ECAPA_TDNN.py)
constructs a length mask, excludes padded positions from global statistics,
sets their attention logits to negative infinity, and then pools.

Assert both that the batch tail is exactly zero and that each original length
is in `[1, batch_width]`. `wav_lens=torch.ones(...)` is valid only if every
item truly has the full width.

### 3.2 Causal student

Prefer running the unpadded eligible waveform. If batching requires a padded
backing tensor, create output validity from original samples, not tensor width.
For output index `o`, lookahead `L`, hop `H`, and window `W`, the latest raw
sample it can require is exclusive index:

```text
latest_sample_exclusive(o) = (o + L) * H + W
valid(o)                    = latest_sample_exclusive(o) <= true_samples
```

Apply the 21-frame semantic delay only after this evidence check. Equivalently,
for a clip with `F=true feature frames`, the number of delay-aligned stable
logits is `max(F - L - D, 0)`. Store the true sample count, feature-frame count,
stable-logit count, delay-aligned count, and latest-sample range. Never recover
these quantities later from padded tensor shape.

## 4. Metric and release rules

Every prefix table should publish, per deadline and language:

- selected, eligible, and excluded-short counts;
- minimum/median/maximum actual duration;
- exact requested and evaluated sample counts;
- padding policy and model-specific mask version;
- accuracy/F1/recall denominators and independent clip/speaker counts.

A seven-language macro score is `not_evaluable` when an eligibility filter
leaves any language empty. Do not silently assign the absent classes zero F1,
and do not use a duration-skewed six-row slice as a release gate. Acquire longer
balanced recordings or use a shorter deadline. Natural full-duration metrics
remain valid but answer a different question.

Paired channel comparisons may retain the same source eligibility set across
all conditions and arms. Padding being identical within a pair makes the old
deltas diagnostically useful, but nonlinear pooling means it does **not** prove
that a masked rescore will preserve their magnitude or ordering. Report both
paired deltas and absolute masked quality; gate only on the latter after all
required languages have support.

## 5. Minimal machine-readable schema

```json
{
  "clip_id": "en_heldout_10",
  "requested_deadline_samples": 64000,
  "source_samples": 49548,
  "eligible_fixed_deadline": false,
  "outcome": "ineligible_short",
  "batch_width_samples": null,
  "model_valid_samples": 49548,
  "padding_samples": 0,
  "padding_consumed_as_evidence": false,
  "valid_feature_frames": 308,
  "valid_stable_logits": 304,
  "valid_delay_aligned_logits": 283,
  "summary_membership": ["natural_full"],
  "mask_contract": "sample_exact_v1"
}
```

The frame counts above are illustrative and must be recomputed from the exact
stored sample count; do not copy them as fixture truth. Bind the schema to
frontend window/hop/centering, lookahead, delay, source audio hash, transform
identity, model hash, and scorer source.

## 6. Acceptance tests

1. A source with exactly `wanted` samples is eligible and consumes no pad; a
   source with `wanted-1` is `ineligible_short` and contributes only to its
   natural-full cell.
2. For ECAPA, variable batching and zero-padded batching with correct
   `wav_lens` retain identical top-1 on the 15-clip fixture and maximum
   restricted-posterior L1 no larger than `0.005`. The historical all-valid arm
   must exceed that tolerance on the current fixture.
3. For the student, changing arbitrary samples strictly after
   `true_samples` cannot change any output whose
   `latest_sample_exclusive <= true_samples`; an output one sample beyond the
   boundary must be masked.
4. The four-second historical audit must reproduce 15/21 per telephony
   condition, 60/84 cross-view condition rows, 52/84 fresh factorial clips,
   and 26/42 exact controls.
5. A prefix summary with zero eligible Bengali/Tamil/Telugu rows must return
   `not_evaluable` for seven-language release gating while still publishing
   raw per-language denominators.
6. Row order, teacher batch size, and amount of zero batch padding cannot
   change membership or declared source duration.
7. A paired rescore must use the same eligible clip IDs for clean and every
   degraded condition; any missing transform row fails closed rather than
   changing the cohort.

The `0.005` ECAPA parity tolerance is a local regression threshold derived
from one pinned artifact, not a general standard.

## 7. Hugging Face Hub and external-source audit

The Hub audit on **2026-09-26** was targeted and non-exhaustive.

- [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9)
  still resolves to revision `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`,
  last modified **2024-11-27**, with Apache-2.0 metadata. Its card describes
  16 kHz utterance classification; it does not require or endorse a fixed
  four-second zero tail.
- [`google/fleurs`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)
  resolves to revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`,
  last modified **2026-05-15**, with CC BY 4.0 metadata. The in-progress
  external-validation harness already excludes rows shorter than a requested
  prefix rather than padding them, but no completed report exists at this
  cutoff; its result and implementation validity are **unverified**.
- Hugging Face's official
  [Wav2Vec2 documentation](https://huggingface.co/docs/transformers/main/en/model_doc/wav2vec2)
  distinguishes models that require attention masks from models trained
  without them and warns that mask behavior is model-specific. Wav2Vec2 is not
  used in these three experiments; the relevant lesson is procedural only:
  padding semantics must follow the exact model contract, not a universal
  “zeros are ignored” assumption.

No reviewed Hub card in this targeted audit was treated as evidence that
synthetic right-padding estimates Hindi-English streaming quality.

## 8. Concrete next steps

1. **Add one sample-exact prefix request/scoring utility and adversarial
   fixtures.** It should separate fixed-deadline eligibility from natural-full
   evaluation, pass original lengths to ECAPA, mask student logits by latest
   raw sample, and serialize all counts above. Expected model gain: **none**.
   Acceptance is the seven-test suite in section 6.
2. **Publish versioned masked rescoring for the three affected experiments.**
   Preserve the historical JSON and add new result/report names bound to the
   same source audio, transforms, checkpoints, and cohorts. Report old versus
   corrected accuracy/F1, posterior L1, prediction flips, eligible counts, and
   whether each old verdict changes. Expected numerical changes are
   **unverified**; the stored voice comparison suggests few top-1 flips but
   cannot predict channel-arm or future-checkpoint behavior.
3. **Use natural full plus supported fixed deadlines as release inputs.** Keep
   the voice experiment's full-duration primary endpoint, use 1/2 s deadlines
   on the present 21 clips, and mark the present seven-way 4 s cell
   `not_evaluable` until every language has a predeclared minimum of genuine
   four-second examples. For pinned FLEURS validation, report excluded-short
   counts by language and never let duration eligibility silently rebalance a
   selector. Expected accuracy gain: **none**; this prevents a contaminated or
   under-supported metric from authorizing promotion.

## Sources and dates

- SpeechBrain 1.0.3
  [`EncoderClassifier`](https://github.com/speechbrain/speechbrain/blob/v1.0.3/speechbrain/inference/classifiers.py)
  and [`ECAPA_TDNN`](https://github.com/speechbrain/speechbrain/blob/v1.0.3/speechbrain/lobes/models/ECAPA_TDNN.py),
  accessed **2026-09-26**.
- SpeechBrain
  [`classify_batch` API documentation](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.inference.classifiers.html),
  accessed **2026-09-26**.
- Pinned Hugging Face ECAPA model card/tree and FLEURS dataset tree linked in
  section 7; Hub API revision/license/modification metadata queried
  **2026-09-26**.
- Hugging Face Transformers
  [Wav2Vec2 model documentation](https://huggingface.co/docs/transformers/main/en/model_doc/wav2vec2),
  accessed **2026-09-26**.
- Local code, manifests, reports, and SHA-256-bound result artifacts listed in
  section 1, audited **2026-09-26**.
