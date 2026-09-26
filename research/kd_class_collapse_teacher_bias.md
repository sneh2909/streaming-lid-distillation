# Teacher-target bias and class collapse in the seven-language student

Research cutoff: **2026-09-26**. Web sources and Hugging Face Hub/API
metadata were checked on that date. This note answers one post-factorial
question only: why the balanced 70-clip monolingual corpus still produces
zero-recall languages, and what is the smallest controlled KD experiment that
can distinguish teacher-target bias from optimisation or representation
failure.

## Bottom line

The training manifest is clip-balanced, but the **supervision is not**.

- There are ten monolingual training clips per language, yet the current
  loss-weighted, temperature-2 teacher target contains only **6.529% English
  probability mass**, versus **20.649% Hindi** and **17.781% Gujarati**. This
  is a full-corpus diagnostic under the declared frame mask/ramp; mini-batch
  normalisation means it is not claimed as the exact 1,600-step gradient
  share.
- The selected-seven ECAPA teacher is top-1 correct on only **6/10 English
  training clips** and 10/10 for every other monolingual class. All four
  English errors are from `en-IN-PrabhatNeural`: three are labelled Hindi and
  one Gujarati. The five gTTS English clips are 5/5 correct, while the five
  Edge English clips are 1/5.
- At temperature 2, the mean true-English target on English clips falls from
  55.61% to **48.00%**. The four bad Edge targets are therefore not harmless
  “dark knowledge”; they explicitly train the student away from English.
- Equalising clip weights alone barely changes the aggregate English target
  mass (6.529% to 6.615%). It cannot repair a teacher that gives the wrong
  class most of the probability.
- This explains the English failure, but **not all collapse**. Marathi's
  teacher is 10/10 correct with mean temperature-2 true-class probability
  87.77%, yet exact-Edge Marathi recall is nonzero at only 1/161 stored
  checkpoints. Marathi still requires an optimisation, representation, or
  profile-transfer diagnosis.

The next experiment should therefore retain KD but add known-label
supervision on the labelled monolingual synthetic clips. The simplest
factorial is current pure KD, equal-clip pure KD, a 50:50 hard/soft hybrid on
monolingual clips, and a teacher-error-gated arm that uses hard CE when the
teacher is wrong. The switch clip keeps its availability-valid local KD
targets in every arm. Expected English improvement is **unverified**; this is
first a causal diagnosis, not a proposed final production recipe.

## 1. Artifact-bound local audit

The calculations below are bound to:

- `data/generated/manifest.jsonl`: 164,407 bytes, SHA-256
  `7df6fc5d94db4c88cac9df6b2de8bc9dc5fcacf1bf36853f839bf494ad59314c`
- `data/generated/targets/metadata.json`: 79,644 bytes, SHA-256
  `bcacc0d668d0fea6bc172ed0f5ef24e1db5745eaa9e009d6c2cffd3ed55da53a`
- `experiments/checkpoint-trajectory/results.json`: 4,292,312 bytes,
  SHA-256
  `c4533c0b4aecd667289dff1df38f0a7b79cd98df359450da817b47706c4a4379`

The trajectory has the open BUG-15 executed-source caveat. Its stored numbers
are useful read-only diagnostics, not proof of which worktree bytes ran.

### 1.1 What the current objective actually weights

For each training clip, this audit reproduced the current valid-target count
`length - D - L`, the first-100-frame ramp, and the cached dense targets. The
table's “frame-loss share” is the fraction of full-corpus ramp weight attached
to that manifest label. “Target mass” is the corresponding share of the
seven-way temperature-2 teacher distribution, including the one mixed
Hindi-to-English training clip. Teacher correctness and true-class
probabilities use monolingual clips only.

| Manifest language | Clips | Frame-loss share | Teacher selected-seven top-1 | Mean true-class probability, T=1 | Mean true-class probability, T=2 | Full-corpus T=2 target mass |
|---|---:|---:|---:|---:|---:|---:|
| English | 10 | 12.497% | **6/10** | 55.6085% | **48.0021%** | **6.529%** |
| Hindi | 10 | 14.625% | 10/10 | 98.2843% | 96.3713% | 20.649% |
| Marathi | 10 | 13.914% | 10/10 | 94.6961% | 87.7732% | 12.912% |
| Bengali | 10 | 13.083% | 10/10 | 99.7979% | 96.7251% | 13.279% |
| Tamil | 10 | 13.419% | 10/10 | 99.9905% | 99.3689% | 13.545% |
| Telugu | 10 | 14.776% | 10/10 | 99.9880% | 99.0964% | 15.306% |
| Gujarati | 10 | 14.950% | 10/10 | 99.6386% | 95.5046% | 17.781% |
| mixed Hindi->English | 1 | 2.735% | not one-label data | — | — | included above |

The mixed clip's own loss-weighted T=2 distribution is 59.168% Hindi and
19.358% English. It contributes only 2.735% of corpus ramp weight, so it
aggravates the Hindi/English difference but does not create the much larger
monolingual English defect.

The aggregate target distributions are:

| Target temperature | en | hi | mr | bn | ta | te | gu |
|---|---:|---:|---:|---:|---:|---:|---:|
| T=1 | 7.581% | 20.089% | 13.424% | 13.225% | 13.444% | 14.953% | 17.284% |
| T=2 | **6.529%** | **20.649%** | 12.912% | 13.279% | 13.545% | 15.306% | 17.781% |

Temperature 2 is not generally wrong. Here it further reduces the already
weak English target and spreads mass toward the teacher's Hindi/Gujarati
confusions. A temperature sweep cannot make the four incorrect top-1 targets
correct by itself.

### 1.2 The error is concentrated in the Indian-English Edge profile

| English training profile | Clips | Teacher top-1 correct | Mean T=1 `P(en)` | Mean T=2 `P(en)` | Mean T=1 `P(hi)` | Mean T=1 `P(gu)` |
|---|---:|---:|---:|---:|---:|---:|
| `gtts:en:default` | 5 | 5/5 | 89.64% | 74.97% | 8.18% | 0.34% |
| `en-IN-PrabhatNeural` | 5 | **1/5** | **14.22%** | **15.09%** | **54.10%** | **29.77%** |

The exact failed selected-seven targets are:

| Clip | Teacher top-1 | T=1 `P(en)` | T=2 `P(en)` | Native selected-language mass |
|---|---|---:|---:|---:|
| `en_train_06` | Hindi | 0.2895% | 3.3747% | 89.4445% |
| `en_train_07` | Hindi | 18.5330% | 27.3978% | 66.0490% |
| `en_train_08` | Hindi | 0.1886% | 2.8550% | 72.4089% |
| `en_train_09` | Gujarati | 0.0812% | 1.9544% | 76.7952% |

These are not merely low-retained-mass cases: most teacher probability is
inside the seven selected languages and points to the wrong selected class.
The common voice/provider/text bundle is an association, not a causal
explanation. Whether ECAPA is reacting to Indian accent, this synthetic voice,
renderer artifacts, the text, or their interaction remains **unverified**.

### 1.3 Why a balanced manifest can still start in one class

Ignoring the constant `T^2`, the current objective is

```text
L = sum_i w_i KL(q_i || p_i) / sum_i w_i .
```

If a student can emit only one constant distribution, the optimum is the
weighted mean teacher target:

```text
p* = sum_i w_i q_i / sum_i w_i .
```

The largest component of the measured mean target is Hindi (20.649%), not
English (6.529%). This does not prove that the trained TCN must collapse to
Hindi, but it proves that equal clip counts do not imply equal class pressure
under pure KD.

The stored trajectory is consistent with that risk:

- at steps 0, 10, and 100, the student predicts Hindi for all 70 monolingual
  training clips and scores 10/70;
- all-training English recall is nonzero at only 17/161 stored states, peaks
  at 5/10 at steps 1400 and 1410, and returns to 0/10 at step 1600;
- the three exact Edge-English controls are 0/3 at **all 161 states**;
- exact Edge-Marathi recall is nonzero at only one state (1/3 at step 1540),
  despite the clean Marathi teacher targets.

The first bullet is an observed association between initialization,
optimisation, and target prior; it is not a proof that target prior alone
caused the initial Hindi prediction.

### 1.4 Clip normalisation alone is not the repair

Recomputing one mean target per clip and weighting all 71 training clips
equally gives this deterministic diagnostic:

| Proposed target construction | en | hi | mr | bn | ta | te | gu |
|---|---:|---:|---:|---:|---:|---:|---:|
| equal-clip pure KD | 6.615% | 19.924% | 12.865% | 14.330% | 14.193% | 14.546% | 17.527% |
| equal-clip 50:50 hard/soft on monolingual clips | 10.486% | 17.421% | 13.517% | 14.235% | 14.148% | 14.363% | 15.829% |
| equal-clip, replace teacher-wrong monolingual targets with hard labels | 11.748% | 17.237% | 12.583% | 14.052% | 14.121% | 14.416% | 15.843% |

These are target-prior calculations, **not trained results or accuracy
forecasts**. They show why duration/clip balancing is a useful control but not
a sufficient English repair.

## 2. What primary evidence supports

### 2.1 Hard plus soft targets are standard KD, not an abandonment of KD

[Hinton, Vinyals, and Dean](https://arxiv.org/html/1503.02531v1)
(submitted **2015-03-09**) explicitly recommend a weighted average of soft
teacher cross-entropy and correct-label cross-entropy when labels are known,
with `T^2` compensation for the soft loss. Their speech experiment used
temperature 2 and a 0.5 relative weight on hard-target CE. That experiment was
large-vocabulary ASR on roughly 2,000 hours, not seven-language streaming LID;
the exact coefficient does not transfer. It does establish that a hybrid loss
still qualifies as distillation.

The current project deliberately uses pure KD. If a hybrid arm is adopted,
README/DESIGN must stop claiming that synthesis labels never enter the
optimisation loss.

### 2.2 Spoken-LID evidence supports withholding bad soft labels

[Dey, Mondal, and Kurmi](https://www.isca-archive.org/interspeech_2025/dey25_interspeech.html)
(Interspeech **2025**, DOI `10.21437/Interspeech.2025-2120`) update their
teacher-free short-utterance LID soft labels only from correctly classified
training segments, and add conditional updates and entropy-based dynamic
weights. This is the closest task evidence for not learning indiscriminately
from a wrong language target. It is self-distillation on their LID corpora,
not frozen-ECAPA-to-causal-TCN distillation, and it does not validate a gain on
Indian English or code-switch boundaries here.

### 2.3 Worst-class KD depends on teacher calibration

[Wang et al.](https://proceedings.mlr.press/v216/wang23e.html)
(UAI **2023-07/08**) show that average KD gains can coexist with worse rare
classes and identify **per-class calibration of teacher scores** as important
for robust students. Their image-classification experiments support measuring
teacher quality and student recall per class rather than relying on aggregate
teacher accuracy. They do not prescribe this project's class gate or loss
weight.

[Sun et al.](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Knowledge_Distillation_with_Refined_Logits_ICCV_2025_paper.html)
(ICCV **2025-10**) address incorrect teacher predictions by refining logits
with known labels while retaining some non-target class relationships. That is
a relevant second-stage option if simple hard/soft mixing works but loses
useful confusions. Its gains are on CIFAR-100 and ImageNet; transfer to speech
and frame-aligned KD is **unverified**.

The newest directly relevant bias work found,
[Kim's LTKD](https://arxiv.org/abs/2506.18496)
(CVPR **2026-06**), rebalances cross-group teacher mass and within-group
contributions for long-tailed image datasets. The conceptual warning applies:
a biased teacher can pass its class bias to the student. The current corpus is
not class-count long-tailed, however; its imbalance is produced by duration
and erroneous soft targets. Importing LTKD's head/medium/tail machinery before
the simple labelled controls would be unjustified.

[Menon et al.](https://research.google/pubs/long-tail-learning-via-logit-adjustment/)
(ICLR **2021**) derive logit adjustment for unequal label priors. Here the
known manifest label prior is uniform and the skewed “prior” is partly a
teacher error. Naively treating teacher target mass as the true prior risks
calibrating the student toward the very bias under investigation. Logit
adjustment should be deferred unless a labelled development set establishes a
real deployment-prior mismatch.

## 3. Hugging Face Hub audit

The audit was performed on **2026-09-26** and is non-exhaustive.

- The current
  [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9)
  still resolves to `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`
  (last modified **2024-11-27**). Its card reports only a 6.7% aggregate
  VoxLingua107 development error and explicitly warns that foreign-accented
  speech works poorly. It publishes no per-language Indian-English
  calibration, confusion table, or hard/soft target policy.
- The convenient AmberNet challenger
  [`surogate/ambernet-langid`](https://huggingface.co/surogate/ambernet-langid/tree/3b7b3fbfcc51753745f1d9abd107c9530c391d5b)
  resolves to `3b7b3fbf...1d5b` (last modified **2026-08-13**). Its own card
  warns about under-five-second, heavily accented, and code-switched speech;
  Hub metadata says `license:other`. It is not a validated drop-in correction
  for these four clips.
- A current Hub API search for audio-classification models named
  “language-identification” surfaced three March-2026 community Indic uploads:
  [`ansc00053/indic-language-identification`](https://huggingface.co/ansc00053/indic-language-identification/tree/ca41635de96f4a5fc843631566559a8178e8ea65),
  [`radioduran/indic-language-identification`](https://huggingface.co/radioduran/indic-language-identification/tree/4eb05c92649c7923b4c09b65e065f662a7213d82),
  and
  [`PremKumar12/indic-language-identification`](https://huggingface.co/PremKumar12/indic-language-identification/tree/08e9cf8c846b9bf7e737c5dd3290efaa1066700e).
  Their config label maps cover 22 Indian languages but omit English, and
  their cards are unfilled templates with no dataset, metrics, license, or
  imbalance/KD documentation. They cannot test the headline English class.
- Exact Hub searches for “teacher-free language identification” and
  “knowledge distillation language identification” returned no model results.
  Search indexing and naming are incomplete, so this is **not proof that no
  such checkpoint exists**. No reviewed Hub card found here documents a
  Hindi-English causal LID student trained with teacher-error correction.

## 4. Proposed diagnostic experiment

Use one frozen source/data/target snapshot, identical initialization, identical
clip order, 1,600 updates, and the existing availability-valid switch targets.
Checkpoint every ten updates, but apply METHOD-43's absolute gates; no arm may
reach external validation merely by beating the collapsed baseline.

### Arm A — current pure frame KD

Retain the submitted loss exactly. This is the reference.

### Arm B — equal-clip pure KD

Normalise the ramped frame loss inside each clip, then average the seven clip
losses in the batch. This isolates duration/batch weighting without changing
any target probability. The deterministic full-corpus English target mass is
only 6.615%, so a large English recovery is not expected; actual behaviour is
**unverified**.

### Arm C — equal-clip hybrid KD + hard CE

For monolingual clips with known synthesis language, use

```text
L_mono = 0.5 * T^2 * KL(q_T || p_T) + 0.5 * CE(y, p_1).
```

For the mixed clip, retain only the current local KD loss. This keeps boundary
training teacher-derived while testing whether known monolingual labels repair
the effective prior. The 0.5/0.5 split is a single predeclared diagnostic arm,
not a tuned optimum or expected production setting.

### Arm D — teacher-error-gated KD

On a monolingual clip, keep pure KD only if the teacher's selected-seven T=1
top-1 equals the known language; otherwise use hard CE for that clip. Keep the
mixed clip unchanged. Record the exact gate decision in the target lineage so
future teacher/cache changes cannot silently change which loss was used.

This is deliberately simpler than refined-logit or long-tail KD. If Arm D
helps English without damaging other classes, a later experiment may compare
hard replacement with RLD-style target-logit refinement. If it does not help,
there is no reason to add that complexity.

### Required measurements and interpretation

For every arm and checkpoint, publish:

- integer per-language teacher top-1 counts, mean `q_y` at T=1/T=2, selected
  mass, and total effective target mass;
- per-language output-bias gradient sum/norm on the first batch and one full
  deterministic pass;
- all-70 and exact-Edge per-language recall/confusion, teacher agreement, KD,
  CE, and total loss separately;
- the existing 1 s/2 s/full synthetic-development table, monolingual churn,
  and both switch directions under the chronology-safe scorer; and
- unchanged parameter count, causality, alignment, and stable-emission tests.

Interpret the factorial before looking at external validation:

| Result | Supported conclusion |
|---|---|
| B recovers English over A | frame/clip weighting is causally material |
| C or D recovers English but B does not | erroneous teacher supervision is causally material |
| D beats C | concentrating label correction on teacher errors is useful |
| none recover English | optimisation/capacity/frontend or stronger profile confounding dominates |
| English recovers but Marathi does not | confirms at least two distinct collapse mechanisms |

The local diagnostic gate should require all seven languages to have nonzero
recall on all 70 training clips, at least 80% exact-Edge and all-training
accuracy, and no regression in the already-published causality/availability
contracts. These are project smoke gates, not literature-derived production
thresholds. A successful arm is still not promotable without speaker-disjoint
natural validation and viable switch behaviour.

## 5. Targeted teacher follow-up

Before changing the global teacher, score the four bad English clips plus all
70 monolingual training clips with the already-pinned ECAPA, Whisper-small,
MMS-LID-126, and a parity-verified AmberNet artifact. Add pinned FLEURS English
validation and Svarah as natural Indian-English controls; do not tune on their
locked test splits.

Report full-space top-1, selected-seven top-1, `P(en)`, retained mass,
short-prefix accuracy, CPU time, and license. A challenger should not replace
ECAPA merely for fixing four synthetic clips: it must remove zero-recall
classes on labelled development data without materially worsening the other
six languages, switch-target timing, or CPU cost. The existence and magnitude
of such a gain are **unverified**.

## 6. Claims not supported by this audit

- It does not prove that Indian accent caused ECAPA's English errors; voice,
  renderer, text, and accent move together.
- It does not prove that hard-label mixing improves held-out or natural speech.
- It does not explain Marathi collapse.
- It does not validate LTKD, RLD, logit adjustment, or any other vision method
  for streaming LID.
- It does not justify using synthesis labels on future unlabelled call audio.
  The hybrid is available only where trusted labels exist; unlabelled data
  still needs teacher reliability controls.
- It does not authorize an external-validation run for the current 161
  collapsed checkpoints.

## 7. Reproduction notes

Read-only checks used NumPy over all 71 training target files, the exact
manifest and target metadata, and all 161 stored trajectory rows. The audit
recomputed valid frame/ramp weights, T=1/T=2 target mass, teacher top-1 counts,
voice-stratified English probabilities, equal-clip/hybrid/gated target priors,
and checkpoint recall extrema. Web research used primary proceedings/papers;
Hub checks used live model APIs, exact revision trees, cards, and config label
maps. No model was trained or run for inference, no target/cache/result was
changed, and all expected gains are marked unverified.
