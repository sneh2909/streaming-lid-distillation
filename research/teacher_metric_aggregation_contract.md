# Teacher metric aggregation contract (METHOD-3)

Research cutoff: **2026-09-26**. Web sources and Hugging Face Hub/API
metadata were checked on that date. Local numbers are bound to the exact
artifact hashes listed below.

## Bottom line

The current `mean_anchor_label_agreement` is not an anchor-pooled metric. It
first computes accuracy inside each clip and then gives every clip equal
weight. That makes a one-anchor monolingual utterance worth as much as a
33-anchor switch trajectory. On the current cache this yields **94.81%**, while
the actual pooled count is **157/190 = 82.63%**.

Neither number is a sufficient headline:

- clip-macro accuracy almost discards the switch trajectories;
- anchor pooling gives each switch clip about 33 times the weight of a
  monolingual clip and depends on the arbitrary teacher sampling interval;
- the two regimes do not even run the same inference view: monolingual clips
  receive one full-utterance posterior, while switch clips receive local
  1.75 s-past/0.25 s-future windows;
- both are **restricted seven-way conditional top-1 accuracy**, not full
  107-way ECAPA accuracy.

The defensible fix is to publish integer numerators and denominators separately
for monolingual utterances, sparse local-window anchors, and duration-weighted
expanded switch trajectories. Keep split, language, and switch direction
slices. Transition recall/lag remains a separate event metric. There is no
model gain from this change; it repairs the evidence contract.

## 1. Exact audit of the current artifacts

The read-only audit used:

| Artifact | SHA-256 |
|---|---|
| `data/generated/manifest.jsonl` | `7df6fc5d94db4c88cac9df6b2de8bc9dc5fcacf1bf36853f839bf494ad59314c` |
| `data/generated/targets/metadata.json` | `d87fe4714accd7c4883b5197e478399c6eb3d2f578b98a752caf92b72a589513` |
| `results/teacher_metrics.json` | `70f75fa24a6adb4f5862b0aae57ca6f4a7aea65e4856749dae7f3dc45ebed3f8` |

This is the schema-3, `previous_anchor_hold` target set generated on
2026-09-26. The backlog still cites **94.78%** and **156/190 = 82.11%** from an
older snapshot. Those figures must not be silently combined with the current
cache; the current exact values are below.

### 1.1 Why the two aggregates disagree

For clip `j`, let `c_j` be correct anchors and `n_j` be total anchors. The
current field is

```text
clip_macro = (1 / number_of_clips) * sum_j(c_j / n_j).
```

An anchor-pooled result is

```text
anchor_pooled = sum_j(c_j) / sum_j(n_j).
```

The current cache gives:

| Slice | Clips | Correct / anchors | Anchor-pooled | Clip-macro |
|---|---:|---:|---:|---:|
| All target-generation clips | 94 | 157 / 190 | 82.63% | **94.81%** |
| Converged full-utterance targets | 91 | 87 / 91 | 95.60% | 95.60% |
| Local-window switch targets | 3 | 70 / 99 | 70.71% | 70.71% |
| Held-out monolingual only | 21 | 21 / 21 | 100.00% | 100.00% |
| Training monolingual only | 70 | 66 / 70 | 94.29% | 94.29% |
| Training switch only | 1 | 23 / 33 | 69.70% | 69.70% |
| Evaluation switches only | 2 | 47 / 66 | 71.21% | 71.21% |

The local-window clip-macro and pooled values happen to match only because all
three current switch clips have exactly 33 anchors. They will diverge as soon
as clip lengths or anchor counts differ.

Of 94 clips, 91 are monolingual; they therefore receive **96.81%** of the
current clip-macro weight. Of 190 anchors, 99 are local switch anchors; they
receive **52.11%** of the pooled weight. The 12.18-point headline gap is not a
rounding issue. It is two different estimands created by the mixed target
recipe.

### 1.2 Regime and class slices reveal different failures

The 91 monolingual calls contain 13 clips per selected language. Restricted
seven-way top-1 counts are:

| Language | Correct / clips | Accuracy |
|---|---:|---:|
| English | 9 / 13 | 69.23% |
| Hindi | 13 / 13 | 100.00% |
| Marathi | 13 / 13 | 100.00% |
| Bengali | 13 / 13 | 100.00% |
| Tamil | 13 / 13 | 100.00% |
| Telugu | 13 / 13 | 100.00% |
| Gujarati | 13 / 13 | 100.00% |

All four monolingual errors are English training clips: three are conditionally
classified as Hindi and one as Gujarati. The 21 held-out monolingual clips are
21/21, including 3/3 English. Thus “teacher accuracy = 100%” is correct only
for the held-out monolingual slice, not for the target-generation corpus as a
whole.

For the local targets, the current per-clip sparse-anchor counts are:

| Clip | Role/direction | Correct / anchors | Accuracy |
|---|---|---:|---:|
| `switch_hi_en_train` | train, Hindi -> English | 23 / 33 | 69.70% |
| `switch_hi_en_eval` | eval, Hindi -> English | 23 / 33 | 69.70% |
| `switch_en_hi_eval` | eval, English -> Hindi | 24 / 33 | 72.73% |

Training and evaluation rows must remain separate. Combining all three would
let a training fixture affect an evaluation claim.

### 1.3 Anchor pooling is not duration weighting

Sparse anchors are approximately 250 ms apart, but the final interval can be
shorter and every anchor still gets equal weight. After causal previous-anchor
hold, a second useful view is agreement on the 10 ms semantic-frame grid:

| Clip | Correct / dense frames | Dense accuracy | Outside nominal +/-250 ms boundary collar |
|---|---:|---:|---:|
| `switch_hi_en_train` | 546 / 798 | 68.42% | 521 / 748 = 69.65% |
| `switch_hi_en_eval` | 546 / 798 | 68.42% | 521 / 748 = 69.65% |
| `switch_en_hi_eval` | 571 / 798 | 71.55% | 546 / 748 = 72.99% |
| Evaluation switches pooled | 1,117 / 1,596 | 69.99% | 1,067 / 1,496 = 71.32% |

This dense result answers “for how much nominal switch time does the deployed
target trajectory's top class agree?” It does **not** create 1,596 independent
observations: held frames are repeated and neighboring posteriors share almost
all their waveform. Confidence intervals must resample independent calls or
source/speaker clusters, not frames. With two mirrored evaluation clips made
from one utterance pair, no population interval is justified.

The reference is also only the manifest's nominal 4.000 s synthetic split.
Earlier waveform audit found target activity begins around 4.080 s and the
joins contain substantial nonspeech. Therefore the dense/collar values above
are deterministic fixture diagnostics, **not human-verified phonetic
accuracy**.

## 2. The prediction space must be named

`scripts/teacher_targets.py` selects seven ECAPA log-probability columns and
softmaxes those selected values before computing `anchor_label_accuracy`.
Consequently, every number above is:

```text
restricted_7way_conditional_top1_accuracy_at_T1
```

It is not full-107-way ECAPA accuracy. A selected language can win after
renormalization even when almost all original teacher mass belongs outside the
seven-language deployment set. The current cache records selected-language
mass, but not full-teacher top-1, so full-107 accuracy is **unverified and not
recoverable from this cache**. METHOD-22's proposed full-label/top-probability
fields are required before that metric can be reported.

Use “agreement” only when comparing two model outputs. When a synthesis or
human label is the reference, use “restricted top-1 accuracy” and name the
reference source. This prevents a reader from mistaking conditional teacher
agreement for accuracy in the teacher's native label space.

## 3. What external evaluations support

### 3.1 Balance languages explicitly

The [NIST LRE22 evaluation plan](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=935161)
(**published 2022-08**) computes miss and false-alarm rates per target language
and then averages across the `N_L` target languages in `C_avg`. That metric is
not the right objective for this seven-class KD experiment, but it is strong
precedent that the language weighting must be explicit rather than inherited
from sample counts.

The [TitaNet-LID paper](https://www.isca-archive.org/interspeech_2023/jia23b_interspeech.html)
(**Interspeech 2023**) labels its FLEURS result “macro accuracy.” It also
reports validation macro accuracy separately from VoxLingua107 evaluation
error because the latter evaluation set is small and imbalanced. This directly
supports a per-language macro result plus per-language counts, not one unnamed
accuracy.

The current Hugging Face [`google/fleurs`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)
snapshot is pinned at `70bb2e84b976b7e960aa89f1c648e09c59f894dd`
(**last modified 2026-05-15**). Its seven relevant configurations contain:

| Config | Validation | Test |
|---|---:|---:|
| `bn_in` | 402 | 920 |
| `en_us` | 394 | 647 |
| `gu_in` | 432 | 1,000 |
| `hi_in` | 239 | 418 |
| `mr_in` | 443 | 1,015 |
| `ta_in` | 377 | 591 |
| `te_in` | 311 | 472 |

The selected test counts vary by **2.43x**. A pooled FLEURS accuracy would
therefore weight Marathi more than twice as heavily as Hindi. For the planned
external check, publish language-macro accuracy as the fairness-oriented
primary score and pooled `correct/N` as a secondary workload score, with the
complete per-language table.

### 3.2 Pair duration and segment/event views

The [DISPLACE 2023 evaluation plan](https://displace2023.github.io/docs/DISPLACE_Evaluation_Plan_v1.pdf)
(**2023**) defines language diarization evaluation over human segmentation with
duration-pooled DER, overlap included, and no forgiveness collar. That supports
a duration-weighted view of local language trajectories.

Duration weighting has a known blind spot. Liu and Yu's
[*Balanced Error Rate for Speaker Diarization*](https://arxiv.org/abs/2211.04304)
(**submitted 2022-11-08**) shows that long segments can overwhelm errors on
short but meaningful segments and argues for duration, segment, and
speaker-balanced views together. This is speaker-diarization evidence, not a
validated LID metric, but the aggregation failure transfers: time accuracy can
hide a missed short language insertion. Keep switch-event recall/lag and
short-span bins beside dense accuracy.

For uncertainty, Liu and Peng's
[*Modeling Dependent Structure for Utterances in ASR Evaluation*](https://www.isca-archive.org/interspeech_2023/liu23_interspeech.html)
(**Interspeech 2023**) explicitly treats speech observations as dependent and
uses blockwise rather than observation-wise bootstrap. The exact method is not
prescribed here; the relevant rule is to resample at the independent call or
source/speaker unit. Any gain from a particular interval procedure for this
project is **unverified**, and the current effective switch sample is too small
for one.

### 3.3 Hugging Face cards do not resolve the local ambiguity

The pinned [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/blob/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9/README.md)
card at `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`
(**last modified 2024-11-27**) declares an `Accuracy` metric and reports 6.7%
error on 1,609 VoxLingua107 development segments from 33 languages. It does not
state in the card whether the 6.7% is utterance-micro, language-macro, or
another weighting, nor does it provide numerators or per-language results.

That omission is not evidence that the model result is wrong. It means the
card's scalar cannot define this repository's aggregation contract. The local
result must state its units, prediction space, reference, sampling policy, and
counts explicitly.

## 4. Proposed machine-readable contract

Replace the single ambiguous headline with a versioned object. Retain the old
field for one compatibility cycle only, renamed or documented as
`deprecated_clip_macro_anchor_accuracy`.

```json
{
  "teacher_quality_schema_version": 1,
  "prediction_space": "selected_7way_conditional_T1",
  "reference_source": "manifest_segments_nominal",
  "monolingual_utterance": {
    "unit": "clip",
    "correct": 87,
    "total": 91,
    "accuracy": 0.9560439560,
    "by_split": {},
    "by_language": {}
  },
  "local_window_anchors": {
    "unit": "teacher_call",
    "correct": 70,
    "total": 99,
    "accuracy": 0.7070707071,
    "by_split": {},
    "by_direction": {}
  },
  "expanded_local_trajectory": {
    "unit": "10ms_semantic_frame",
    "expansion": "previous_anchor_hold",
    "by_split": {},
    "by_direction": {},
    "outside_boundary_collar": {"collar_ms": 250}
  },
  "combined_diagnostic_only": {
    "correct_anchors": 157,
    "total_anchors": 190,
    "accuracy": 0.8263157895
  }
}
```

Every leaf should carry integer `correct` and `total`; never serialize only a
rounded rate. Add `n_clips`, `n_independent_source_groups`, and exact artifact
identity beside each slice. Soft-target quality should additionally report
proper scores such as mean negative log probability/Brier score and retained
mass, but those do not replace top-1 counts or transition metrics.

Do not publish an “overall primary teacher accuracy” across full-utterance and
local-window regimes. If product planning later requires one scalar, freeze an
application mixture first (for example, monolingual calls versus switch time)
and label its weights. The current corpus composition and anchor hop are not a
deployment prior.

## 5. Deterministic acceptance tests

1. Reconstruct every aggregate from per-clip integer counts and assert that
   disjoint children sum exactly to their parent.
2. On the artifact hashes above, assert 87/91 monolingual, 70/99 local,
   157/190 combined, 21/21 held-out monolingual, and 47/66 switch-evaluation
   anchors. This is a snapshot test, not a universal expected model score.
3. Assert the existing clip-macro reconstruction is
   `0.9480980012894908` and differs from pooled
   `0.8263157894736842`; a refactor must not silently relabel one as the other.
4. Duplicate frames inside one held interval: the duration diagnostic may
   change according to duration, but clip count, independent-trial count, and
   sparse teacher-call count must not.
5. Change the anchor hop while holding the underlying trajectory fixed: mark
   the sparse-anchor score as a different sampling contract and never compare
   it as if only model quality changed.
6. Reject aggregation across train and evaluation unless the output is
   explicitly named `all_cache_diagnostic`.
7. Require `prediction_space`; a restricted-seven metric must fail validation
   if named `full_teacher_accuracy`.
8. Require switch event recall/lag/miss outputs beside dense local accuracy so
   a long correct segment cannot hide a missed boundary or short insertion.
9. Bootstrap only independent call/source groups when enough exist. With the
   current two mirrored evaluation files, publish the two rows and no
   inferential percentile or confidence interval.

## 6. Concrete next steps

1. **Publish structured teacher counts without rerunning ECAPA.** Recompute the
   tables from the already bound anchor arrays and manifest, then propagate the
   structured object to teacher metrics, summary, and documentation. Expected
   model gain: **none**. Acceptance is exact count reconciliation and removal
   of the ambiguous scalar as a headline.
2. **Add duration-weighted switch diagnostics from the cached expanded
   trajectory.** Report eval directions individually and pooled only within
   the eval switch regime, both no-collar and nominal +/-250 ms sensitivity.
   Expected gain: **none**; this is reporting. Do not describe nominal joins as
   human speech boundaries.
3. **Make external metrics language-balanced and count-preserving.** On pinned
   FLEURS validation/test manifests, report per-language `correct/N`, macro
   accuracy, and pooled `correct/N`; select on validation only. Expected score
   direction is **unverified**. A result fails the evidence gate if it reports
   only a pooled scalar.
4. **Extend the cache before claiming native-teacher accuracy.** Store full
   ECAPA top label/probability and selected absolute probabilities under
   METHOD-22, then report full-107 and restricted-seven metrics side by side.
   The full-space result and its relationship to student quality are
   **unverified** until that rerun.

## 7. Scope and unverified points

- The exact local tables describe one tiny synthetic target cache, not teacher
  population accuracy.
- Full-107 accuracy is unavailable from the selected-column cache.
- Dense switch labels use nominal synthetic segments, not human-timed speech.
- Neighboring anchors/frames are dependent; `N=190` is not an independent
  sample size.
- The two evaluation directions reuse one unordered utterance pair.
- Macro versus pooled reporting exposes different failure modes; neither
  predicts that changing the teacher or training objective will improve the
  student.
- FLEURS aggregation advice applies to external monolingual evaluation; it
  does not validate FLEURS as a natural code-switch benchmark.

## Sources and access record

- NIST, [*2022 Language Recognition Evaluation Plan*](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=935161),
  **published 2022-08**; accessed 2026-09-26.
- Jia et al., [*A Compact End-to-End Model with Local and Global Context for
  Spoken Language Identification*](https://www.isca-archive.org/interspeech_2023/jia23b_interspeech.html),
  **Interspeech 2023**; accessed 2026-09-26.
- DISPLACE, [*DISPLACE 2023 Evaluation Plan*](https://displace2023.github.io/docs/DISPLACE_Evaluation_Plan_v1.pdf),
  **2023**; accessed 2026-09-26.
- Liu and Yu, [*BER: Balanced Error Rate for Speaker Diarization*](https://arxiv.org/abs/2211.04304),
  **submitted 2022-11-08**; accessed 2026-09-26.
- Liu and Peng, [*Modeling Dependent Structure for Utterances in ASR
  Evaluation*](https://www.isca-archive.org/interspeech_2023/liu23_interspeech.html),
  **Interspeech 2023**; accessed 2026-09-26.
- Hugging Face, [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/blob/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9/README.md)
  at `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`, **last modified
  2024-11-27**; card and [API metadata](https://huggingface.co/api/models/speechbrain/lang-id-voxlingua107-ecapa)
  checked 2026-09-26.
- Hugging Face, [`google/fleurs`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)
  at `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, **last modified
  2026-05-15**; card and [revision API](https://huggingface.co/api/datasets/google/fleurs/revision/70bb2e84b976b7e960aa89f1c648e09c59f894dd)
  checked 2026-09-26.

