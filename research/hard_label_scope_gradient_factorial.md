# Hard-label scope under common gradient pressure (METHOD-51)

Research cutoff: **2026-09-26 (IST)**. Web sources and Hugging Face Hub/API
metadata were checked on that date. This note researches one question only:
how to test whether the earlier loss-factorial result came from broad labelled
regularization, targeted correction of four conditional teacher errors, or the
different KD/CE recipes and gradient scales used by those two arms.

Evidence labels used below:

- **Observed** means rederived from a retained project artifact or from a
  read-only, six-thread autograd probe on its exact seed-7 initial state.
- **Publisher-reported** means stated by a paper or model publisher and not
  reproduced here.
- **Proposed / unverified** means a future experimental rule. No student was
  trained and no model inference was run for this note.

## Bottom line

The existing experiment cannot support the sentence “broad hard-label
regularization helped more than correcting the four teacher errors.” It
compared:

- labels on all 70 monolingual clips versus four English clips;
- a per-clip `0.5 KD + 0.5 CE` blend versus complete KD replacement; and
- objectives with different temperature-induced and realized gradient scales.

The later external run has already rejected the exact 50:50 hybrid recipe.
METHOD-51 must not reopen it by changing the explanation of the old local fit.

The smallest useful causal diagnostic is a **two-factor crossed experiment**:

1. hard-label scope: all 70 monolingual clips versus the frozen four
   conditional-seven teacher errors; and
2. teacher treatment on the labelled scope: retain KD and add CE versus remove
   KD and use CE.

Use one equal-clip KD control and one production frame-KD parity sentinel. For
each seed, give the hard-label component the same initial full-parameter
gradient budget in both scopes, then scale every treatment's complete initial
gradient to the equal-clip control norm. Freeze those coefficients before the
first update. This produces four crossed cells plus two controls, rather than
comparing two confounded recipes.

This normalization is a project-specific diagnostic, not a published LID
standard. It equalizes one full-corpus norm at initialization; it does **not**
equalize direction, minibatch variance, clipping, AdamW updates, or later
training pressure. Those quantities must be measured rather than assumed.

## 1. Exact claim and artifact boundary

The local comparison is bound to:

- [`experiments/loss-factorial/results.json`](../experiments/loss-factorial/results.json),
  SHA-256 `b22125b174c76d073c7569f44cb9ebd0ef2562fba9f98fa42d767a1366ab91ea`;
- [`experiments/loss-factorial/run.py`](../experiments/loss-factorial/run.py),
  SHA-256 `d2f7cb03fabfb93c676830b8f8d5585d57c1be63e1a22b18fa409aa836b0e322`;
- [`experiments/loss-factorial/REPORT.md`](../experiments/loss-factorial/REPORT.md),
  SHA-256 `94aaf23e733080b20ec1fa6c8cb17715a7e4a94fd089ce44ae5f6267970e2b38`;
- run ID
  `044ef348f33663f014b6d5b868ea23e969758438f70e5cfd500439d4e7cdbdf5`;
- initial model-state digest
  `3ab2b8147b21371c2550bd90e6f581215e444ab5a13f4bc391a532183410a059`;
- manifest SHA-256
  `7df6fc5d94db4c88cac9df6b2de8bc9dc5fcacf1bf36853f839bf494ad59314c`;
  and
- [`results/teacher_metrics.json`](../results/teacher_metrics.json), SHA-256
  `ee00866264539e66a3204deca793ce7395f276a2e962ad53e8f677fcc466e327`.

The historical result's broad hybrid was the only arm with locally eligible
states, but its selected step still had zero held-out English/Hindi recall,
more false commits than the matched control, and no evaluable switch. The
subsequent frozen external comparison scored that hybrid below both controls.
The no-promotion decision is therefore unchanged.

Open BUG-19 and BUG-20 also mean the old experiment does not prove which
source and separately read reference bytes executed. A METHOD-51 run must use
the staged-source and immutable-reference repairs before its evidence can be
called provenance-safe.

### “Four errors” needs a prediction-space qualifier

The frozen scope is exactly:

```text
en_train_06, en_train_07, en_train_08, en_train_09
```

These are the four clips whose **renormalized selected-seven** ECAPA top-1
disagrees with the manifest label. That is the prediction space used by the
seven-output student, so it is the right scope for reproducing the review's
question. It is not the complete set of native-teacher errors: the current
107-way audit finds only 1/10 native-English top-1 decisions correct. The new
artifact must call the scope `conditional7_error`, not the unqualified
`teacher_error`. Expanding it to the nine native errors after seeing results
would be a different, post-selected experiment.

## 2. Why a crossed design is necessary

Let each of the 71 training clips have a ramp-weighted, per-clip-normalized KD
loss `K_i`. For monolingual clips, let `C_i` be hard-label CE under the same
causal-valid frame mask and ramp. The mixed Hindi-to-English clip has no hard
utterance label and always keeps local KD.

The old broad and targeted arms were:

```text
broad:    mean[0.5 K_i + 0.5 C_i] on all 70 monolingual clips; K_switch
targeted: C_i on four conditional errors; K_i everywhere else
```

A difference between them could be caused by scope, retention of soft teacher
structure, CE dose, `T^2` KD scale, or an interaction. A two-factor full
factorial estimates both main contrasts and their interaction; NIST's
engineering handbook notes that full factorial models include main effects
and interactions, and its two-way formulation explicitly includes the
interaction term ([NIST handbook, accessed 2026-09-26](https://www.itl.nist.gov/div898/handbook/pri/section4/pri43.htm),
[two-way model](https://itl.nist.gov/div898/handbook/prc/section4/prc437.htm)).
This statistical structure does not make five neural-network seeds
confirmatory; it simply prevents the known recipe variables from remaining
aliased.

Selective-KD evidence supports studying sample partitions, but not a specific
partition or coefficient here. Wang et al. report that distilling every NMT
sample is not always beneficial and explicitly compare partitions
([ACL-IJCNLP 2021, published 2021-08](https://aclanthology.org/2021.acl-long.504/)).
That is machine translation rather than speech LID, so transfer is
**unverified**.

The closest LID-specific results also stop short of this design:

- Dey et al. retain soft labels only from correctly classified short LID
  segments and use dynamic weighting
  ([Interspeech 2025, 2025-08-17 to 2025-08-21](https://www.isca-archive.org/interspeech_2025/dey25_interspeech.html)).
- Liu et al. jointly weight full-mode CE, short-mode CE, and KD for short LID
  ([Odyssey 2022](https://www.isca-archive.org/odyssey_2022/liu22c_odyssey.html)).

Neither paper crosses known-label scope with teacher retention, uses this
teacher/student timing contract, or establishes a gradient coefficient for
Hindi-English streaming LID.

## 3. Proposed common-pressure objective

For `N=71`, define the equal-clip KD control

```text
K(theta) = (1/N) sum_i K_i(theta).
```

For a frozen hard-label scope `S`, define

```text
H_S(theta)      = (1/N) sum_{i in S} C_i(theta)
K_not_S(theta)  = (1/N) sum_{i not in S} K_i(theta).
```

The denominator stays `N` in all three expressions. That makes minibatch
indicator-weighted losses unbiased for the declared corpus objective and
makes every coefficient auditable. The two scopes are:

- `all_mono`: all 70 monolingual clips; and
- `conditional7_error`: the four frozen English IDs above.

For seed block `r` at its frozen initial state `theta_r`, compute full-corpus,
full-parameter gradients before clipping or optimizer construction:

```text
lambda[S,r] = ||grad K(theta_r)||_2 / ||grad H_S(theta_r)||_2.

L_raw[retain,S,r]  = K + lambda[S,r] H_S
L_raw[replace,S,r] = K_not_S + lambda[S,r] H_S.

c[q,S,r] = ||grad K(theta_r)||_2 / ||grad L_raw[q,S,r](theta_r)||_2
L_train[q,S,r] = c[q,S,r] L_raw[q,S,r].
```

`retain` asks whether adding a fixed hard-label gradient budget helps while
preserving all teacher information. `replace` asks whether the same labelled
scope benefits from removing its teacher loss. `lambda` holds the initial
hard-label gradient budget constant across the 70-clip and four-clip scopes;
`c` holds the complete initial gradient norm constant across cells. Both are
computed separately inside each seed block and then frozen for all 1,600
updates.

This is intentionally **not** dynamic GradNorm or PCGrad. GradNorm shows that
component gradient magnitudes affect multi-objective training
([Chen et al., ICML 2018](https://proceedings.mlr.press/v80/chen18a.html));
PCGrad defines a conflict by a negative gradient cosine
([Yu et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)).
Those papers motivate measuring magnitudes and directions. They do not
validate this fixed normalization for LID, and their adaptive gradient
algorithms must not be silently introduced.

Hinton et al. support combining hard and soft objectives when labels are
available, but they also derive the temperature dependence that makes scalar
loss weights different from target-mixture weights
([submitted 2015-03-09](https://arxiv.org/abs/1503.02531)). The exact
`lambda/c` rule above is a **proposed, unverified diagnostic**, not a result of
that paper.

### Required arms

Run the following six arms inside every seed block:

| Role | Scope | Teacher on labelled clips | Purpose |
|---|---|---|---|
| `frame_kd_sentinel` | none | KD | Reproduce the production objective; not part of the 2x2 contrasts. |
| `equal_clip_kd_control` | none | KD | `K`, the factorial control and norm reference. |
| `all_mono_retain` | 70 | retained | Broad labels with fixed hard-gradient budget. |
| `all_mono_replace` | 70 | removed | Broad hard supervision; only the mixed clip keeps KD. |
| `conditional7_error_retain` | 4 | retained | Targeted correction without discarding teacher structure. |
| `conditional7_error_replace` | 4 | removed | Targeted replacement under the same calibration rule. |

No arm may apply an utterance label to the mixed clip. All arms retain the
same model, local switch targets, timing, valid-frame mask, ramp, optimizer,
learning rate, clipping threshold, update budget, and minibatch IDs.

## 4. Seed-7 component-gradient probe

The following is **observed diagnostic evidence**, not training. A read-only
probe used six Torch CPU threads, the exact 71 cached training examples, and
initial state `3ab2b814…e059`. It evaluated each component with the existing
per-clip loss code and accumulated full-parameter norms in float64.

| Component at seed 7 | Full-parameter L2 | Cosine with `grad K` | English output-bias gradient |
|---|---:|---:|---:|
| `K` | 1.5151880104 | 1.000000 | +0.1682970077 |
| `K_not_all_mono` | 0.1202062823 | +0.215856 | -0.0012065750 |
| `K_not_conditional7_error` | 1.3281971183 | +0.963699 | +0.1614176482 |
| `H_all_mono` | 1.0942263893 | +0.084642 | +0.0119423121 |
| `H_conditional7_error` | 0.4043129931 | **-0.613666** | **-0.0476298854** |

The four-error hard-label gradient conflicts with equal-clip KD at this one
state, while the all-label hard gradient is almost orthogonal. That is exactly
why target counts or nominal 50:50 weights cannot identify optimization
pressure. It does not predict which trained model will transfer.

The proposed calibration gives:

| Cell | `lambda` | Whole-loss `c` | Cosine of raw cell with `grad K` | Scaled English bias gradient |
|---|---:|---:|---:|---:|
| `all_mono_retain` | 1.3847116330 | 0.6789562918 | +0.736424 | +0.1254939806 |
| `all_mono_replace` | 1.3847116330 | 1.0621990818 | +0.108096 | +0.0162836007 |
| `conditional7_error_retain` | 3.7475620032 | 1.1376358305 | +0.439508 | -0.0116026807 |
| `conditional7_error_replace` | 3.7475620032 | 1.2233906390 | +0.282729 | -0.0208934329 |

After multiplying by `c`, every cell's initial full-parameter norm is
1.5151880104 by construction. Their directions and English components remain
very different. The error-scope multiplier is also large because only 4/71
clips contribute CE. In minibatch training that can create rare, bursty
gradients and differential clipping. Every update must therefore store the
selected-scope count, unscaled/scaled component norms, pre-clip total norm,
and whether the 5.0 clip threshold activated. Full-corpus equality must never
be described as per-update equality.

These numbers are exact only for seed 7 and the bound artifacts. Coefficients
for the other seed blocks are **unverified** and must be computed rather than
copied.

## 5. Repeated, paired seed blocks

One seed cannot distinguish a recipe effect from initialization/order luck.
Bouthillier et al. show that initialization and data order materially affect
model comparisons and recommend pairing algorithms by reusing each sampled
seed within a pair
([MLSys 2021](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf)).

Use five predeclared master seeds:

```text
7, 762827487, 930234305, 1171872382, 1524105651
```

The first preserves the historical block. The other four are the first
32-bit hexadecimal word of
`SHA256(run_id + "|METHOD-51|" + i)`, reduced modulo `2^31-1` for `i=1..4`.
This deterministic derivation is fixed before any new outcome exists.

Within each seed block:

- construct one initial state and copy it to all six arms;
- generate one ordered minibatch ledger and replay it for every arm;
- compute all `lambda/c` values from that state and exact full corpus before
  optimizer creation;
- keep optimizer settings and update count identical; and
- bind initial-state, batch-order, calibration, per-step example IDs, and
  every retained state hash.

Across blocks, initialization and data order vary together. The unit of
replication is the seed block, not a clip, frame, checkpoint, or 7x3 prefix
table. Report every paired contrast plus median and range. With only five
blocks, do not attach a confirmatory p-value or population 95% interval.

The historical four-arm run took 266.88 seconds. A linear estimate for 30
arm-seeds is about 33.4 minutes, but shared-host contention, extra gradient
audits, and the repaired staged runner make the real runtime **unverified**.
Publish seed blocks atomically so a safe resume does not mix identities.

## 6. Analysis contract

### Fixed-step primary analysis

Use step 1,600 for the primary factorial contrasts. Do not choose each cell's
best checkpoint and then compare those adaptive maxima. Retain ten-step curves
only as descriptive training-dynamics evidence and report how many seed-arm
trajectories ever pass METHOD-43's absolute gates.

For each response `Y` and seed `r`, compute:

```text
scope effect under retain  = Y[all,retain,r]  - Y[error,retain,r]
scope effect under replace = Y[all,replace,r] - Y[error,replace,r]

retention effect for all   = Y[all,retain,r]  - Y[all,replace,r]
retention effect for error = Y[error,retain,r]- Y[error,replace,r]

interaction = (Y[all,retain,r] - Y[error,retain,r])
            - (Y[all,replace,r]- Y[error,replace,r]).
```

Preserve the sign convention beside every table. If the interaction is
material relative to the two scope contrasts, do not summarize the result as
one “scope main effect.” NIST's factorial model supports estimating this
interaction; it does not supply a project-specific materiality threshold.

### Responses

Publish integer denominators and per-language values before aggregates:

1. the four frozen error clips, ten English clips, all 70 monolingual train
   clips, and exact Edge controls;
2. fixed-step 1 s, 2 s, and natural-full held-out per-language recall,
   language-macro recall, and minimum-language recall;
3. monolingual false commits and flip-flop/churn under the frozen router;
4. chronology-safe raw/policy source preconditions and detections for both
   switch directions; and
5. loss, full and component gradients, component cosine, clipping rate, and
   selected-scope exposure per seed/arm.

Training fit is a mechanism diagnostic, not the promotion outcome. A recipe
that fits the four selected clips but retains zero held-out minimum recall or
unevaluable switches is still a failed transfer recipe.

### Decision boundary

This five-seed factorial may support only a scoped statement such as “under
this fixed calibration, broader labels changed fixed-step local fit more than
the four-clip scope.” It cannot establish a population effect or authorize a
release model.

- Any cell failing absolute class-coverage, held-out, stability, or switch
  gates is `diagnostic_follow_up`, never `adopt`.
- Do not reopen the already rejected exact 50:50 hybrid based on a new
  gradient-normalized objective.
- Do not read the locked 5,063 selected-language FLEURS test rows.
- At most one legally eligible recipe may be frozen for later untouched
  natural/external confirmation; expected transfer gain is **unverified**.

## 7. External evidence and its limits

### Teacher errors and selective supervision

Distil-Whisper uses a weighted KL plus pseudo-label loss and filters teacher
transcripts whose normalized WER against references is too high
([paper submitted 2023-11-01](https://arxiv.org/abs/2311.00430)). This is a
large English ASR system with sequence pseudo-labels, not frame-level LID. It
supports auditing/filtering bad teacher supervision, not this four-clip scope,
the proposed fixed gradient budget, or a Hindi-English accuracy forecast.

The current project's teacher-error labels are known synthetic language
labels. Production unlabelled calls cannot use this gate directly; they still
need valid teacher targets, human labels, or a separately validated
reliability mechanism.

### Gradient magnitude and conflict

GradNorm and PCGrad show why magnitude and cosine deserve explicit reporting,
but both are multi-task methods with adaptive training rules. This proposal
uses their diagnostics only. Applying either algorithm would add another
factor and invalidate the clean 2x2 attribution.

### Multiple runs

MLSys 2021 supports varying initialization/order and pairing treatment runs.
It does not say five blocks are sufficient for a confirmatory conclusion.
Five is a CPU-budgeted project diagnostic; every outcome direction and its
uncertainty remain **unverified**.

## 8. Dated Hugging Face Hub audit

On **2026-09-26**, exact Hub API searches for `language identification
distillation`, `spoken language identification knowledge distillation`, `lid
knowledge distillation`, `distilled language identification`, and `short
utterance language identification` returned zero models. Hub search is
lexical and non-exhaustive, so this is not evidence that no such model exists.

Pinned cards/API records checked directly:

| Hub artifact | Revision / last modified | Relevant evidence | Gap for METHOD-51 |
|---|---|---|---|
| [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9) | `0253049a…e8e9`; 2024-11-27 | Apache-2.0, 107-way ECAPA LID teacher used here. | No student KD/CE scope or gradient-balancing recipe. |
| [`distil-whisper/distil-large-v3`](https://huggingface.co/distil-whisper/distil-large-v3/tree/8031d2e6ce6631b7fc45629dddfc00271116d981) | `8031d2e6…d981`; 2026-04-21 | MIT; card describes weighted KL/pseudo-label training and WER filtering. | English ASR, not LID; no crossed scope/retention or component-gradient audit. |
| [`ntu-spml/distilhubert`](https://huggingface.co/ntu-spml/distilhubert/tree/fa87d96265d6b7af66e112faff6ff44df419cec9) | `fa87d962…cec9`; 2023-07-24 | Apache-2.0 distilled English speech representation model. | Not an LID head and no known-label scope factorial in the card. |
| [`tiantiaf/voxlect-indic-lid-whisper-small`](https://huggingface.co/tiantiaf/voxlect-indic-lid-whisper-small/tree/1792f34d4a5c71d9a61583aaf98b7226eaaaa916) | `1792f34d…a916`; 2025-08-10 | Indic audio classifier; structured API license `openrail`. | Card documents no KD, hard/soft mixture, scope crossing, or gradient normalization. |

No reviewed Hub card supplies a drop-in METHOD-51 objective or checkpoint.
That negative result is **non-exhaustive**. The proposed coefficients remain
specific to this bound corpus and initial state.

## 9. Acceptance tests

1. **Scope fixture.** Require 70 `all_mono` IDs and exactly the four ordered
   `conditional7_error` IDs above, bound to selected-seven T=1 decisions,
   manifest, target audit, and teacher revision. Reject an unqualified
   `teacher_error` name or native/conditional substitution.
2. **Objective algebra.** On scalar toy losses, prove `retain = K + lambda H`
   and `replace = K_not_S + lambda H`; prove the switch contributes identical
   KD and zero CE in all six arms.
3. **Component denominator.** Recompute `K`, `H_S`, and `K_not_S` with the
   common `N=71` denominator and reject scope-mean or batch-presence-dependent
   renormalization.
4. **Calibration.** Require finite nonzero component norms and reproduce each
   `lambda/c`; after `c`, every treatment's initial full-parameter norm must
   equal its seed's `K` norm within a declared float64 accumulation tolerance.
5. **Direction audit.** Store complete classifier-bias components plus shared,
   classifier, and full-parameter norms/cosines. A forged norm-preserving
   direction must fail.
6. **Freeze rule.** Hash coefficients before optimizer creation and reject any
   per-step recalibration, GradNorm, PCGrad, or outcome-dependent change.
7. **Paired seeds.** Within each seed, require identical initial-state and
   ordered-batch hashes across arms; across the five seed blocks, preserve the
   explicit master seeds and all per-block hashes.
8. **Control parity.** Seed 7 frame KD must reproduce the bound production
   loss/gradient/model trace when the source/reference repairs are in place;
   equal-clip KD must reproduce its historical objective before calibration.
9. **Optimizer exposure.** Store selected-scope counts, pre/post-scale and
   pre/post-clip norms, clipping flags, and actual parameter-update L2 for
   every step. Never infer minibatch balance from the full-corpus calibration.
10. **Fixed-step analysis.** Recompute every step-1,600 cell contrast and
    interaction from integer metric tables; row order, arm aliases, or a
    per-arm best checkpoint must not change them.
11. **Fail-closed outcome.** Zero-recall classes, missing switch source
    preconditions, NaN/Inf, incomplete seed blocks, or failed absolute gates
    must produce `diagnostic_follow_up`/`not_evaluable`, not `adopt`.
12. **Provenance and data-role guard.** Apply BUG-19/20 staging and immutable
    reference bytes, preserve the historical result, publish a separately
    named artifact, and prove that no locked FLEURS test row or later
    confirmation outcome was read.

## Recommendation

Run one provenance-safe, five-seed, six-arm experiment implementing the
two-scope by retain/replace factorial above. Use the fixed initial-gradient
calibration only as a controlled diagnostic, not as a deployment recipe.
Make fixed step 1,600 and seed-paired factorial contrasts primary, retain
trajectories as descriptive evidence, and refuse promotion whenever absolute
transfer, stability, or switch gates fail.

Expected model gain: **unverified**. The only new numerical evidence in this
note is an initialization-gradient audit. It shows that the four-error hard
component is directionally different and initially conflicts with KD; it does
not show that targeted correction, broad labels, retention, or replacement
will improve accuracy.
