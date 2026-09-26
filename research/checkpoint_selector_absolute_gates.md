# Absolute and globally balanced checkpoint-selection gates

Research cutoff: **2026-09-26**. Web sources and Hugging Face Hub/API metadata
were checked on that date. This note addresses backlog **METHOD-43** only. It
is a focused amendment to
[`checkpoint_selection_validation_protocol.md`](checkpoint_selection_validation_protocol.md):
the trajectory infrastructure remains useful, but its relative eligibility
rules and shortlist construction are not safe when every snapshot is
collapsed.

## Bottom line

The current 161-state trajectory contains **no checkpoint that should advance
to external validation**.

- Every state has zero minimum-language recall on the frozen 1 s / 2 s / full
  composite. English recall is exactly zero at all 161 states; no state covers
  more than five of seven languages.
- Every state has 0/2 raw switch detections by 1 s and 0/2 policy detections.
  Under the stored legacy scorer, no state has both raw source preconditions;
  under the frozen policy, no state has even one source precondition.
- No state reaches the previously predeclared 80% exact-familiar-control fit
  floor. The maximum is 15/21 = 71.43%; only five states reach 80% on all 70
  monolingual training clips.
- Step 970 passes the existing relative gate because its 30.16% label
  composite is three decisions better than the already weak final state and
  its 14 committed false switches merely equal the final state's 14. That is
  638.22 false switches/hour on 78.97 s of monolingual development audio, not
  an acceptable absolute stability result.

The corrected outcome is therefore:

```text
selector_status = no_eligible_checkpoint
selected_step = null
external_validation_authorized = false
trajectory_infrastructure_status = retain
model_candidate_status = reject
```

This is a useful negative result. It prevents a locked external validation set
from being spent to choose among checkpoints that already fail class coverage,
fit, stability, and switch evaluability.

## 1. Exact local audit

The audit is bound to these unchanged artifacts:

- [`experiments/checkpoint-trajectory/results.json`](../experiments/checkpoint-trajectory/results.json):
  SHA-256 `c4533c0b4aecd667289dff1df38f0a7b79cd98df359450da817b47706c4a4379`
- [`experiments/checkpoint-trajectory/run.py`](../experiments/checkpoint-trajectory/run.py):
  SHA-256 `119a4e39325d7e5e801e51be3f88c1dc68a25475a439f0fa729a4fee72697cbc`
- retained 161-state trajectory: 29,759,358 bytes, SHA-256
  `f87f2aa46d7de7992a53c8d01135602a41485aae5bc203ae3a6f480a32468eac`

The result's own source-execution caveat remains open under BUG-15. The
numbers below are a read-only analysis of the retained artifact, not proof of
which source bytes historically executed.

### 1.1 Global trajectory ranges

| Quantity across all 161 states | Exact result |
|---|---:|
| Best 1 s / 2 s / full label composite | 30.1587%, at steps 970, 1440, 1490 |
| Best minimum-language recall | 0%, at all 161 states |
| Maximum languages with nonzero recall | 5/7, at steps 290, 910, 1010 |
| English composite recall | 0/9 at every state |
| Telugu states with any composite recall | 6/161; maximum 2/9 |
| Tamil states with any composite recall | 13/161; maximum 3/9 |
| Raw switch recall by 1 s | 0/2 at every state |
| Policy switch detections | 0/2 at every state |
| Stored raw source preconditions | 0 or 1 of 2; never 2/2 |
| Policy source preconditions | 0/2 at every state |
| Committed monolingual false switches | 0--19; 44 states have zero |
| Best exact Edge-control fit | 15/21 = 71.4286% |
| Best all-training fit | 59/70 = 84.2857%, at step 1510 |

The raw source-precondition counts use the legacy chronology-unsafe helper and
are diagnostic only. METHOD-41's maximal-run, boundary-adjacent scorer must
produce the gate used by a repaired selector. The policy result is already
decisive here: every state has zero source commits at both reference
boundaries.

### 1.2 Why step 970 is not a useful candidate

| Metric | Step 970 | Step 1440 | Final step 1600 |
|---|---:|---:|---:|
| Label composite | 30.16% | 30.16% | 25.40% |
| Languages with nonzero recall | 3/7 | 3/7 | 3/7 |
| Minimum-language recall | 0% | 0% | 0% |
| Exact Edge controls | 8/21 | 15/21 | 15/21 |
| All 70 training clips | 45/70 | 58/70 | 55/70 |
| Committed false switches | 14 | 14 | 14 |
| False switches/hour | 638.22 | 638.22 | 638.22 |
| Raw switch detections by 1 s | 0/2 | 0/2 | 0/2 |

Steps 970 and 1440 tie on the current label maximum and false-switch count,
but the shortlist keeps only the earliest label champion. Step 1440 has seven
more correct exact controls and thirteen more correct training clips, and a
slightly lower raw-change rate (101.25 versus 105.00/min). It is excluded
before the selector sees those facts. Step 1440 still fails class coverage,
the 80% exact-control floor, absolute stability, and switch evaluability, so
this observation does **not** nominate it. It shows why “earliest step” must be
the final deterministic tie-break, after fit and stability, rather than the
rule that decides which tied champion enters the shortlist.

### 1.3 Why adding only a global worst-language champion is insufficient

The existing code computes `best_minimum_recall` inside its five-category
shortlist. That is incorrect; the comparator must be global. But in this run
the global maximum is also zero. A rule such as “within five points of the
global best” would still admit every zero-recall model.

A repaired selector needs both:

1. a positive **absolute** class-coverage floor; and
2. comparison with the best value across all 161 states.

If the global optimum itself misses the absolute floor, selection stops. A
relative tolerance is never allowed to turn zero into eligibility.

## 2. External evidence and what it does not establish

### 2.1 Selection over many checkpoints can overfit validation noise

[Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)
(JMLR, **2010-07**) show that optimizing a noisy finite-sample model-selection
criterion can overfit that criterion and create selection bias of a magnitude
comparable to reported algorithm differences. The local procedure chooses
among 161 states using only 21 clips, with the same clips reused at three
prefixes. The 63 decisions are correlated, and the step-970 gain comes from
only three full-clip outcomes. The paper does not quantify this repository's
bias; the transfer is qualitative.

A new 2026 study,
[Apicella et al.](https://arxiv.org/abs/2602.22107)
(arXiv v1, **2026-02-25**), reports that checkpoint choice depends materially
on the validation criterion and that loss-based rules were more stable than
accuracy-based early stopping in its tested neural classifiers. It is an
**unreviewed preprint**, does not study speech, class collapse, switching, or
knowledge distillation, and does not justify selecting this student by KD.
Its useful warning is narrower: no single noisy scalar should silently stand
in for the actual deployment constraints.

### 2.2 Average accuracy can coexist with complete class failure

[Sagawa et al.](https://proceedings.mlr.press/v119/sagawa20a.html)
(ICML, **2020-07**) demonstrate settings where average test performance
improves while minority-group performance worsens.
[Yang et al.](https://proceedings.mlr.press/v202/yang23s.html)
(ICML, **2023-07**) find worst-class validation accuracy to be a useful model
selection criterion across their subpopulation-shift benchmark, while also
warning that worst-group and average metrics trade off. These are not spoken
LID experiments, so their numerical results do not transfer. They support the
guardrail that a pooled or macro improvement cannot erase a zero-language
recall.

Language-recognition evaluation also treats language conditions explicitly.
The [NIST LRE17 evaluation plan](https://www.nist.gov/document/lre17-evaluation-plan)
(**2017**) computes miss/false-alarm quantities per target language and
equalizes language/data-source partitions before its primary average. It also
compares information against a do-nothing default. NIST does not prescribe a
neural checkpoint selector, a worst-recall floor, or a streaming stability
gate; the exact contract below remains a project proposal.

## 3. Proposed fail-closed selector

The rule is an epsilon-constraint design: hard requirements first, then a
small, inspectable ranking among survivors. It deliberately does not combine
accuracy, switching, fit, and churn into one tunable weighted score.

### Stage 0 — identity and scorer validity

Before reading metrics, require:

- a BUG-15-compliant executed-source snapshot;
- one immutable trajectory, validation manifest, model-state table, and
  selector configuration;
- finite outputs and stable-emission equivalence;
- the METHOD-41 chronology-safe switch scorer; and
- exact integer numerators/denominators behind every rate.

Any failure yields `invalid_artifact`, not `no_eligible_checkpoint`.

### Stage 1 — absolute local smoke gates over all states

Apply every gate to every one of the 161 rows **before** constructing a
shortlist.

| Gate | Proposed current threshold | Provenance and interpretation | States passing alone |
|---|---:|---|---:|
| Class coverage | minimum recall `> 0` across all seven languages on the frozen composite | Collapse smoke test; one success is not evidence of quality | 0/161 |
| Macro floor | label composite `>= 2/7 = 28.5714%` | **Unverified project heuristic:** twice uniform seven-way chance | 6/161 |
| Familiar fit | exact Edge controls `>= 80%` | Reuse the predeclared voice-factorial ceiling; do not lower it after seeing results | 0/161 |
| Training fit | all 70 monolingual clips `>= 80%` | Diagnostic that the tiny model/recipe can fit known classes | 5/161 |
| Stability | zero committed false switches on the 21 monolingual clips | Fail-closed smoke gate while no product SLO exists; 1 event is already 45.59/hour here | 44/161 |
| Raw switch evaluability | chronology-safe source precondition on 2/2 directions | Makes a raw switch metric defined; does not make two mirrored events representative | 0/161 under stored legacy counts |
| Policy switch evaluability | source committed at 2/2 boundaries | Makes the frozen-policy comparison defined | 0/161 |

The combined pass count is **0/161**. The 2/7 macro threshold, 80% fit
thresholds, and zero-event stability threshold are not published universal
standards. They are explicit, testable project guardrails. A future product SLO
may replace them, but only prospectively and with its source recorded. The
current outcome does not depend on the debatable macro threshold: class
coverage, exact-control fit, and switch evaluability independently reject all
states.

### Stage 2 — global balance, then shortlist

Only if Stage 1 has survivors:

1. Compute the global best minimum-language recall over **all** valid states.
   Require each survivor to satisfy both the positive absolute floor and the
   frozen tolerance from that global maximum.
2. Remove any state strictly dominated on the predeclared vector
   `(minimum recall, macro label score, exact-control fit, training fit,
   switch evaluability/recall, -false switches, -raw churn)`.
3. From the remaining Pareto set, retain at most five unique category
   champions: best minimum recall, best macro label score, best
   source-conditioned switch result, best stability, and final state **only if
   it is eligible**. Lowest KD can be recorded as a diagnostic but cannot
   replace a deployment category.
4. Within a tied category, order by larger hard-gate slack, then the frozen
   remaining metrics, then optimizer step, then model-state hash. Do not let
   the earliest step discard a tied checkpoint with materially stronger fit.
5. Publish ineligible final/midpoint/KD states as references outside the
   candidate list. A reference is not automatically entitled to consume
   external validation.

Pareto filtering and this category set are **unverified project choices**,
not a claim of optimal multi-objective selection. Their value is auditability:
no globally balanced state can be omitted because an unrelated category built
the shortlist first.

### Stage 3 — authorize external validation once

If and only if at least one local candidate survives, freeze its step/hash and
the full selector before loading external validation. If no candidate
survives, the runner must prove that the external loader was never called.

The pinned Hugging Face [`google/fleurs`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)
revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd` is CC BY 4.0 and was last
modified **2026-05-15**. Hub dataset-server metadata queried **2026-09-26**
gives these seven validation partitions:

| Config | Validation rows |
|---|---:|
| `en_us` | 394 |
| `hi_in` | 239 |
| `mr_in` | 443 |
| `bn_in` | 402 |
| `ta_in` | 377 |
| `te_in` | 311 |
| `gu_in` | 432 |
| **Total** | **2,598** |

Use per-language counts, language-macro score, and minimum-language recall;
do not pool the 1.85x-imbalanced validation counts as the sole result. Compare
checkpoints on paired utterances, and resample utterance IDs—not prefix
frames—if uncertainty is reported. Keep the 5,063 test rows locked until the
model and policy are frozen.

FLEURS is monolingual read speech. It can confirm seven-language transfer, but
it cannot repair or validate a failed local switch-evaluability gate. Its
[primary paper](https://simran-khanuja.github.io/assets/fleurs_slt.pdf)
(IEEE SLT, **2023**) presents it as a 102-language Speech-LangID benchmark;
there are no code-switch boundary annotations.

## 4. Result schema and deterministic tests

The selection artifact should make refusal a first-class valid result:

```json
{
  "selector_schema": 2,
  "absolute_gate_configuration": {},
  "global_metric_extrema": {},
  "per_state_gate_ledger": [],
  "n_valid_states": 161,
  "n_locally_eligible_states": 0,
  "shortlist": [],
  "selected_step": null,
  "selector_status": "no_eligible_checkpoint",
  "external_validation_loaded": false
}
```

Minimum regression fixtures:

1. all states have minimum recall zero: result is `no_eligible_checkpoint`,
   even though all are within zero points of the global best;
2. the global best-balanced state is absent from the old category shortlist:
   repaired selection still sees it;
3. a high-macro state with one zero-recall language is rejected;
4. a high-accuracy state with 0/2 source preconditions is `not_evaluable`, not
   a switch non-inferiority pass;
5. a tied label champion with stronger fit/stability is considered before the
   earlier-step tie-break;
6. a strictly dominated state never enters the external shortlist;
7. permuting trajectory-row order leaves candidates and result hashes
   unchanged;
8. an empty survivor set never imports or reads FLEURS validation/test;
9. changing one gate, tolerance, scorer version, state hash, or manifest hash
   invalidates resume/reuse; and
10. locked test rows are rejected from every selector input.

After implementing the pure selector, replaying the retained metric table
should produce the exact negative result above without model inference. Only a
separately staged current-source trajectory rerun can become new official
checkpoint evidence.

## 5. Hugging Face model-card audit

A dated, non-exhaustive Hub audit found no released card that supplies this
project's absolute class-coverage + fit + switch-evaluability + stability
checkpoint contract:

- [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9),
  revision `0253049a...e8e9`, last modified **2024-11-27**, reports one 6.7%
  VoxLingua107 development error and explicitly warns that smaller-language
  accuracy may be limited; it does not document per-language checkpoint gates.
- [`tflite-hub/conformer-lang-id`](https://huggingface.co/tflite-hub/conformer-lang-id),
  revision `213db93a...f4739`, last modified **2024-09-19**, identifies a
  streaming Conformer with attentive pooling but publishes no validation or
  checkpoint-selection table on its card.
- [`surogate/ambernet-langid`](https://huggingface.co/surogate/ambernet-langid/tree/3b7b3fbfcc51753745f1d9abd107c9530c391d5b),
  revision `3b7b3fbf...1d5b`, last modified **2026-08-13**, warns about short
  utterances and code-switching but likewise gives no auditable selector.

This absence claim is limited to those cards as observed on **2026-09-26**;
it is not proof that no other Hub repository describes such a protocol.

## 6. Claims that remain unverified

- The proposed 2/7 macro floor, 80% fit floors, and zero local false-switch
  threshold are screening choices, not calibrated production operating points.
- Passing the smoke gates would not establish real-call, 8 kHz, natural
  Hinglish, or population performance.
- FLEURS-validation ranking may not predict synthetic-call or natural-switch
  ranking.
- Two mirrored switch files can establish only mechanical evaluability, not a
  switch-recall distribution or confidence interval.
- Pareto/category shortlisting may or may not select a better external model
  than another frozen multi-objective rule.
- The 2026 validation-criterion preprint's results may not transfer from its
  classifiers to causal KD LID.
- Expected model/accuracy gain from this proposal is **none**. The expected
  gain is evidentiary: reject a collapsed candidate and preserve locked
  validation for a model that clears local requirements.

## Sources checked

- Local trajectory driver, 161-state result, report, reviewer finding, and
  prior selection memo, audited **2026-09-26**.
- [Cawley and Talbot, model-selection overfitting](https://www.jmlr.org/papers/v11/cawley10a.html),
  JMLR **2010-07**.
- [Sagawa et al., average versus minority-group performance](https://proceedings.mlr.press/v119/sagawa20a.html),
  ICML **2020-07**.
- [Yang et al., worst-class model selection](https://proceedings.mlr.press/v202/yang23s.html),
  ICML **2023-07**.
- [NIST LRE17 evaluation plan](https://www.nist.gov/document/lre17-evaluation-plan),
  **2017**; [NIST LRE22 plan](https://www.nist.gov/publications/nist-2022-language-recognition-evaluation-plan),
  published **2022-08-31**.
- [Apicella et al., validation criteria](https://arxiv.org/abs/2602.22107),
  unreviewed arXiv v1 posted **2026-02-25**.
- [FLEURS paper](https://simran-khanuja.github.io/assets/fleurs_slt.pdf),
  IEEE SLT **2023**; [pinned Hub revision](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)
  and [dataset-server size endpoint](https://datasets-server.huggingface.co/size?dataset=google%2Ffleurs),
  queried **2026-09-26**.
- Pinned Hub cards for
  [ECAPA](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9),
  [streaming Conformer](https://huggingface.co/tflite-hub/conformer-lang-id),
  and [AmberNet port](https://huggingface.co/surogate/ambernet-langid/tree/3b7b3fbfcc51753745f1d9abd107c9530c391d5b),
  cards/API queried **2026-09-26**.
