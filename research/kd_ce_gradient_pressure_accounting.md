# Gradient-pressure accounting for mixed KD and hard-label CE (METHOD-50)

Research cutoff: **2026-09-26 (IST)**. Web sources and Hugging Face Hub/API
metadata were checked on that date. This note researches one question only:
what the loss-factorial experiment's `0.5*T^2*KL + 0.5*CE` and
teacher-error-gated objectives actually optimize, and how their class pressure
should be reported.

Evidence labels used below:

- **Observed** means rederived from a retained project artifact or verified by
  an exact analytic/autograd fixture.
- **Publisher-reported** means stated by a paper, official documentation, or a
  model publisher but not reproduced here.
- **Proposed / unverified** means a future reporting or experimental rule. No
  student was trained and no model inference was run for this note.

## Bottom line

The published `10.4861%` hybrid and `11.7477%` gated English values are valid
arithmetic descriptions of constructed targets, but they are **not effective
optimization priors** for the implemented losses.

1. For `L_KD = T^2 KL(q_T || softmax(z/T))`, the gradient with respect to the
   unscaled student logit is `T * (p_T - q_T)`. Hard CE contributes
   `p_1 - y`. At `T=2`, equal scalar coefficients on KD and CE therefore do
   not give equal logit-gradient scale.
2. At zero logits, one monolingual `0.5 KD + 0.5 CE` item is equivalent, up to
   a scalar, to **2/3 soft target + 1/3 hard target**, not a 50:50 target.
   Accounting also for the mixed clip's unhalved pure KD, the exact
   zero-logit gradient-equivalent English target mass is **9.2432368%**.
3. The teacher-error gate uses full KD on 66 correct monolingual clips plus the
   mixed clip and unscaled CE on four erroneous English clips. Its exact
   zero-logit gradient-equivalent English mass is **9.1896651%**, not
   11.7477%.
4. At the retained real initialization, both treatments attenuate but do not
   reverse pressure against the English output bias. The full-corpus English
   bias gradient is `+0.0895164` for hybrid and `+0.1137878` for the gate;
   gradient descent therefore initially moves that bias downward. This is an
   output-layer diagnostic, not the realized AdamW parameter step or an
   accuracy prediction.
5. Away from uniform logits, `softmax(z/T)` and `softmax(z)` differ. No one
   fixed target used with an ordinary CE/KD at a single temperature reproduces
   the mixed gradient for every student state. Report actual component
   gradients, norms, and cosine, not a target average presented as training
   pressure.
6. The exact 50:50 hybrid has already failed external validation. Correcting
   its description needs **no retraining** and does not reopen that rejected
   recipe. Any gain from a future gradient-normalized mixture is
   **unverified**.

## 1. Artifact and claim boundary

The read-only audit is bound to:

- `experiments/loss-factorial/results.json`, SHA-256
  `b22125b174c76d073c7569f44cb9ebd0ef2562fba9f98fa42d767a1366ab91ea`;
- `experiments/loss-factorial/run.py`, SHA-256
  `d2f7cb03fabfb93c676830b8f8d5585d57c1be63e1a22b18fa409aa836b0e322`;
- `experiments/loss-factorial/REPORT.md`, SHA-256
  `94aaf23e733080b20ec1fa6c8cb17715a7e4a94fd089ce44ae5f6267970e2b38`;
- run ID `044ef348f33663f014b6d5b868ea23e969758438f70e5cfd500439d4e7cdbdf5`;
- initial-state digest
  `3ab2b8147b21371c2550bd90e6f581215e444ab5a13f4bc391a532183410a059`;
  and
- batch-order digest
  `694ceb055b96aa1b817272a4d21ab0f9238252385a19a17d2d88fb21c8c95c15`.

The retained source table names the exact current `run.py` hash above, but
open BUG-19/20 mean that the historical experiment still lacks pre-import
executed-source proof and immutable reference-byte consumption. The arithmetic
below is an exact analysis of the retained target and gradient fields; it does
not repair those provenance defects or prove which historical code executed.

The later frozen external run rejected this hybrid recipe: its FLEURS
validation composite was 15.29%, versus 18.00% for the matched control and
17.67% for the current final checkpoint. METHOD-50 changes the interpretation
of the loss audit, not that no-promotion decision.

## 2. Exact gradient derivation

For teacher target `q_T`, unscaled student logits `z`,
`p_T = softmax(z/T)`, and hard one-hot label `y`:

```text
L_KD(z) = T^2 KL(q_T || p_T)
dL_KD/dz = T (p_T - q_T)

L_CE(z) = -log softmax(z)[y]
dL_CE/dz = p_1 - y
```

The `T` in the first gradient remains after the `T^2` loss factor cancels one
`1/T` from differentiating `softmax(z/T)`. This is why a loss coefficient is
not a target-mixture coefficient.

The original distillation paper derives the temperature dependence and notes
the approximately `1/T^2` gradient shrinkage before compensating the soft loss
([Hinton, Vinyals, and Dean, submitted 2015-03-09](https://arxiv.org/html/1503.02531)).
The current official PyTorch tutorial likewise computes `T^2 * KL` and then
applies separate KD and CE coefficients; it calls those coefficients tunable
objective weights, not class-prior proportions
([PyTorch tutorial, updated 2025-01-24](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html#knowledge-distillation-run)).

### One monolingual hybrid item

The implemented hybrid gradient is

```text
g_h(z) = 0.5*T*(p_T - q_T) + 0.5*(p_1 - y).
```

At zero logits, `p_T = p_1 = u`, the seven-class uniform distribution. With
`T=2`:

```text
g_h(0) = 1.5 * (u - ((2/3)*q_T + (1/3)*y)).
```

Thus `0.5` and `0.5` are equal **loss coefficients**, while the target-side
zero-logit logit-gradient weights are `2/3` and `1/3`. If equal soft/hard
scalar sensitivity at uniform logits were the only goal, coefficients that
sum to one would be `1/(T+1)=1/3` on KD and `T/(T+1)=2/3` on CE. That rule does
not equalize actual parameter-gradient norms or resolve conflicting directions;
using it as a new training prescription is **unverified**.

### The complete 71-clip hybrid objective

There are 70 monolingual hybrid items and one mixed item that keeps full KD.
For the ramp-weighted per-clip soft means `q_i` and mixed target `q_s`, the
zero-logit target-side scale and normalized target are

```text
S_h = 70*(0.5*T + 0.5) + T = 107

q_h,0 = [sum_mono(0.5*T*q_i + 0.5*y_i) + T*q_s] / S_h.
```

The mean objective scale is `S_h / 71 = 1.507042253521`. Recomputing from all
71 retained clip means gives

```text
q_h,0(en) = 0.092432367941
g_h,0(en) = (S_h/71) * (1/7 - q_h,0(en)) = +0.0759922665.
```

The report instead averaged `0.5*q_i + 0.5*y_i` for monolingual clips and
`q_s` for the mixed clip. That construction produces `0.104861009645` English
mass, but omits the remaining `T` in the implemented KD gradient.

### The complete teacher-error gate

The gate has 66 correct monolingual KD clips, four erroneous English clips
using CE, and one mixed KD clip. Therefore 67 clips have KD gradient scale `T`
and four have CE scale one:

```text
S_g = 67*T + 4 = 138

q_g,0 = [T*sum_KD(q_i) + sum_error(y_i)] / S_g.
```

The mean objective scale is `138/71 = 1.943661971831`. The retained targets
give

```text
q_g,0(en) = 0.091896650709
g_g,0(en) = (S_g/71) * (1/7 - q_g,0(en)) = +0.099049971.
```

The report's `0.117477026746` construction weights every clip target equally
after choosing `q` or `y`; it does not account for KD clips having twice the
zero-logit logit-gradient scale of CE clips.

### Why the zero-logit target is only a declared reference

At nonuniform `z`, the hybrid prediction side is a weighted combination of
`p_T` and `p_1`. It is generally neither `softmax(z/T)` nor `softmax(z)`.
The normalized target-side vector above remains useful for an explicitly named
zero-logit diagnostic, but it is not a fixed single-temperature target that
fully describes optimization pressure throughout training.

A six-thread float64 autograd fixture used the fixed nonuniform vector
`[2,-1,0.5,-0.5,1.2,-2,0.2]`. Replacing the true hybrid gradient with the
zero-reference target under a single `T=1` or `T=2` softmax produced L2 errors
of `0.2151603` and `0.1045918`. For the gate the errors were `0.4004372` and
`0.0119533`. All four are nonzero; the same fixture matched the analytic
zero-logit gradients within `2e-9`. This is a mathematical regression, not a
student-quality result.

## 3. Exact retained-artifact reanalysis

The table separates three different quantities that the report currently
conflates. `Nominal target` is the stored target construction.
`Zero-logit target` is the normalized target-side vector after the implemented
loss/temperature scales and mixed-clip rule. `Actual initial gradient` is the
retained full-corpus gradient of the real initialized model's final classifier
bias; its logits vary by clip and frame.

| Arm | Nominal English target | Zero-logit gradient-equivalent English target | Mean zero-logit scale | Zero-logit English bias gradient | Actual initial English bias gradient | Actual full bias-gradient L2 |
|---|---:|---:|---:|---:|---:|---:|
| Equal-clip pure KD | 6.6150532% | 6.6150532% | 2.0000000 | +0.1534132 | +0.1682970 | 0.2273949 |
| Hybrid KD+CE | 10.4861010% | **9.2432368%** | 1.5070423 | +0.0759923 | **+0.0895164** | 0.1409280 |
| Teacher-error gate | 11.7477027% | **9.1896651%** | 1.9436620 | +0.0990500 | **+0.1137878** | 0.1669032 |

For gradient descent, a positive English-bias gradient means a downward bias
update. Both hard-label treatments reduce that initial anti-English component
relative to equal-clip pure KD, but neither makes it negative. This does not
show that the hard-label component is harmful: the full gradient also changes
the other six output biases and every hidden parameter, and AdamW applies
moments, weight decay, and clipping during training.

The full actual initial bias vectors are:

```text
hybrid:
  en +0.089516371  hi +0.002471507  mr +0.005072125  bn +0.036456808
  ta -0.004556405  te -0.031689692  gu -0.097270727

gate:
  en +0.113787763  hi -0.003187607  mr +0.016780205  bn +0.032634087
  ta -0.010905148  te -0.040520884  gu -0.108588390
```

The actual-versus-zero-reference gradient cosine is `0.6660162` for hybrid
and `0.7680197` for the gate; actual L2 is respectively `1.3494x` and
`1.2927x` the zero-logit L2. The reference is informative but plainly not the
real initialization.

The current artifact records only the **combined** gradient for each arm. It
does not retain separate `g_KD` and `g_CE` tensors, their weighted norms, or
their cosine. Consequently it cannot determine whether the two components
reinforce or conflict in the shared trunk. Work on multi-objective learning
supports inspecting those quantities: GradNorm balances objectives through
gradient magnitudes ([Chen et al., ICML 2018](https://proceedings.mlr.press/v80/chen18a.html)),
while PCGrad defines conflicting gradients by negative cosine
([Yu et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)).
Applying either adaptive algorithm here would be a new method and is
**unverified**; they are cited only to justify measuring norm and direction
rather than inferring them from scalar loss weights.

## 4. What a defensible gradient audit should publish

### Preserve target arithmetic, but name it honestly

The current target vectors remain useful data diagnostics. Rename
`effective_target_prior_by_arm` to `nominal_target_construction_by_arm` for
the mixed arms, or add a `meaning` field that forbids optimization-pressure
claims. Pure-KD target mass can remain a target-mass statistic, provided it is
not called the realized minibatch gradient distribution.

### Add two explicit reference layers

1. **Analytic zero-logit reference:** store `T`, loss coefficients, clip-scope
   counts, per-clip normalization, exceptional mixed-clip handling, total
   scale `S`, normalized target vector, and exact bias-gradient vector. Name it
   `zero_logit_gradient_equivalent`, not `effective_prior`.
2. **Real-state autograd audit:** on the bound initial state and exact ordered
   full-corpus batch, compute KD and CE separately before combining them.
   Store unweighted and weighted gradients for:
   - the seven classifier biases;
   - the classifier weight tensor; and
   - all shared trainable parameters.

For each parameter scope, report `||g_KD||_2`, `||g_CE||_2`,
`cos(g_KD,g_CE)`, `||a*g_KD + b*g_CE||_2`, and each component's projection on
the combined direction. Norm percentages alone must not be forced to sum to
one when gradients oppose. Keep per-language classifier-bias components for
interpretability.

The same diagnostic at predeclared retained checkpoints can show drift, but
reconstructing it requires loadable states or a provenance-safe deterministic
rerun. Later-state component gradients are currently **unverified**.

### Do not confuse three separate questions

- **Target composition:** what probability mass appears in `q` or `y`?
- **Instantaneous gradient pressure:** what direction and magnitude does each
  loss contribute at a declared model state and batch?
- **Causal effect on the trained model:** what changes under a controlled arm
  with matched scope, scaling, initialization, order, and selection?

METHOD-50 can answer the first two. It cannot identify why the hybrid fit
better locally; METHOD-51's crossed design is still required for the third.

## 5. LID-specific evidence and its limit

Recent short-utterance LID work supports conditional and dynamic weighting,
but not this project's numerical rule. Dey et al. update soft labels only from
correctly classified segments and add dynamic weighting
([Interspeech 2025](https://www.isca-archive.org/interspeech_2025/dey25_interspeech.html)).
Liu et al. jointly optimize full- and short-input LID modes with KD
([Odyssey 2022](https://www.isca-archive.org/odyssey_2022/liu22c_odyssey.html)).
Neither source establishes that a nominal 50:50 target mix equals a 50:50
gradient mix, nor does either validate a gradient-normalized recipe for this
42,567-parameter causal Hindi-English student. Any transfer is **unverified**.

## 6. Dated Hugging Face Hub audit

On **2026-09-26**, exact Hub API searches for `language identification
distillation`, `spoken language identification knowledge distillation`, and
`lid knowledge distillation` returned zero models. Hub search is lexical and
non-exhaustive, so this is not proof that no such artifact exists.

Three pinned cards were inspected directly:

| Hub artifact | Revision / API date | Relevant card evidence | Gradient-accounting gap |
|---|---|---|---|
| [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9) | `0253049…e8e9`; last modified 2024-11-27 | Apache-2.0; card says the LID model used cross-entropy | No KD/CE mixture or gradient audit. |
| [`ntu-spml/distilhubert`](https://huggingface.co/ntu-spml/distilhubert/tree/fa87d96265d6b7af66e112faff6ff44df419cec9) | `fa87d96…cec9`; last modified 2023-07-24 | Apache-2.0, 23.5M-parameter distilled speech representation model | Card names layer-wise/multi-task distillation but gives no temperature, component coefficients, or gradient balance. It is English representation pretraining, not LID. |
| [`tiantiaf/voxlect-indic-lid-whisper-small`](https://huggingface.co/tiantiaf/voxlect-indic-lid-whisper-small/tree/1792f34d4a5c71d9a61583aaf98b7226eaaaa916) | `1792f34…a916`; last modified 2025-08-10 | Indic audio-classification card; structured API license `openrail` | Card contains no KD, temperature, loss-mixture, or gradient details. |

No reviewed card supplies a drop-in, LID-specific gradient-normalization
contract. That negative finding is **non-exhaustive** and the proposed audit
below remains project-specific.

## 7. Concrete acceptance tests

1. **Analytic/autograd zero fixture.** With the exact 71 retained clip means,
   assert hybrid `S=107`, gate `S=138`, English target masses
   `0.092432367941` and `0.091896650709`, and analytic/autograd gradient
   equality within a declared float64 tolerance.
2. **Nominal-label guard.** Preserve `0.104861009645` and `0.117477026746`
   only under a field containing `nominal`; reject `effective_prior`,
   `gradient_share`, or equivalent prose for those numbers.
3. **Mixed-clip regression.** Removing, halving, or treating the mixed clip as
   labelled must change `S` and fail the bound fixture.
4. **Nonuniform-logit regression.** Use a fixed nonuniform seven-logit vector
   and prove that neither a `T=1` nor `T=2` single-softmax proxy using the
   zero-reference target matches the true combined gradient.
5. **Real-state parity.** Recompute the retained combined initial output-bias
   vectors from the exact initial state and full-corpus order before adding
   component fields; require byte/content identity and existing BUG-19/20
   source/reference gates.
6. **Component additivity.** Assert `g_total = a*g_KD + b*g_CE` per declared
   parameter scope, finite norms, cosine in `[-1,1]`, and class-bias sum near
   zero. A forged component that preserves only the total must fail.
7. **No-decision-change guard.** A documentation/analysis repair must retain
   the historical external rejection and must not launch training, consume
   locked test data, or relabel the old result as provenance-safe.

## Recommendation

Publish a small versioned analysis sidecar for the historical loss factorial,
rename the two mixed-arm values as nominal target constructions, and expose the
already-retained actual initial combined bias gradients. For any future loss
comparison, add separately differentiable KD/CE component gradients at one
frozen real reference before choosing weights. If METHOD-51 needs a common
scale, compare a clearly declared theoretical zero-logit rule and a fixed
initial-gradient-norm rule; do not silently introduce adaptive GradNorm/PCGrad.

Expected model gain: **none for the correction; unverified for every future
reweighted objective**. The most important immediate outcome is narrower: the
current artifact does not show that hybrid supervision supplied 10.4861%
effective English pressure, and the actual initial bias diagnostic still
points away from English.

