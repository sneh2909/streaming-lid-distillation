# Fixed validation and checkpoint selection for the streaming LID student

Research cutoff: **2026-09-26**. Web sources and Hugging Face Hub cards/API metadata were checked on that date. This note addresses builder backlog **METHOD-4** only: selecting a checkpoint on fixed, speaker-disjoint validation evidence and deciding whether switch examples need balanced sampling.

## Bottom line

The submitted run has a valid optimization trace, but **step 1600 is not a selected checkpoint**. It is merely the last checkpoint. Training KD fell from a first-ten mean of 6.8704 to a last-ten mean of 1.4403, while the final student reached only 18.80% held-out frame-macro label accuracy and missed the scored switch. That is direct local evidence that falling training KD cannot stand in for generalization.

The smallest defensible fix is:

1. Treat the repeatedly consulted 21-clip synthetic “heldout” set as **development**, not untouched test.
2. Save a minimal model snapshot once per loader pass (every 10 optimizer steps in the current run).
3. Score every snapshot on one frozen, cheap development manifest; keep the routing policy fixed while choosing the model.
4. Shortlist at most five checkpoints and choose among them once on pinned external **validation** data.
5. Bind the selected step, selector version, validation-manifest hashes, metric table, and trajectory identity into the published checkpoint.
6. Score locked FLEURS/HiACC **test** data only after the model and policy are frozen.

Do not oversample the existing switch item. It is one boundary, not a switch distribution. First create a causally valid, source-disjoint switch pool; then compare modest, explicitly balanced sampling schedules.

All proposed gains below are **unverified** until this repository runs the experiment.

## 1. What the repository currently does

Read-only audit on 2026-09-26:

| Item | Verified current state | Consequence |
|---|---:|---|
| Training set | 71 clips / 318.8606 s: 70 balanced monolingual clips and one Hindi→English splice | Only one training boundary and one direction exist |
| Training voices | 14 synthetic voice IDs | Voice-disjoint evaluation is possible, but this is not population diversity |
| Optimizer schedule | 1,600 steps, batch 7, 11,200 examples | 157.7465 effective corpus passes |
| Loader passes | 10 full batches per iterator because `71 // 7 = 10` with `drop_last=True`; 160 iterators in the run | A natural snapshot cadence is every 10 steps |
| Training-switch exposure | Exact replay of the seeded sampler included item index 70 in 156/160 loader passes | The same acoustic boundary was repeated 156 times; this is memorisation pressure, not boundary diversity |
| Checkpointing | [`scripts/train.py`](../scripts/train.py) saves only after all requested updates | There is no earlier model to compare or restore |
| Validation in training | None; the dataset is created with `splits=("train",)` and only training loss is recorded | No generalization metric can select a step |
| Synthetic heldout | 21 clips / 78.9702 s, three clips per language, but exactly one held-out voice ID per language | Frame/clip counts overstate the seven independent voice clusters |
| Switch evaluation | Two 8 s files reverse the same held-out Hindi and English utterances | The two directions are not independent trials |
| Final student | frame micro 18.51%, frame macro 18.80%, clip accuracy 4/21; teacher label accuracy 100% | The final model transfers poorly despite lower training KD |

The builder and experimenter logs have repeatedly used the 21 clips and two reversed switches to make design choices. Therefore, **inference:** those files now function as development data even though their manifest split is named `heldout`/`switch`. Renaming their experimental role does not make the samples worse; continuing to call later measurements on them untouched test evidence would be misleading.

The exact switch-exposure replay used a dummy 71-item `DataLoader` with the repository's batch size, `shuffle=True`, seed 7, `drop_last=True`, and 160 iterator constructions. It did not load audio or update a model.

## 2. Why training KD is not a checkpoint-selection metric

Three evidence streams agree on the need for separate validation measurements:

- The local run is already a counterexample: training KD decreases substantially while unseen-voice label accuracy is poor. KD asks whether the student matches cached teacher distributions on repeated training audio; it does not ask whether the student recognizes an unseen speaker, a short prefix, or a new switch.
- [Wu et al., *What Mechanisms Does Knowledge Distillation Distill?*](https://proceedings.mlr.press/v243/wu24a.html) (**PMLR, 2024-12-15**) measure teacher/student KL separately from out-of-distribution accuracy and show that fidelity does not imply transfer of the teacher's invariances, especially with spurious correlations. This is non-speech evidence; application to this LID student is **unverified**, but it supports keeping validation label/OOD metrics distinct from KD.
- [Wang et al., *Robust Distillation for Worst-Class Performance*](https://proceedings.mlr.press/v216/wang23e.html) (**UAI, 2023-07-31 to 2023-08-04**) show that average KD gains can come at the expense of rare classes and that teacher-score calibration matters to worst-class student performance. Their tasks are not speech LID, so numerical transfer is **unverified**. The relevant guardrail is to inspect per-language and worst-language results rather than optimize pooled frame loss.

Speech training practice also saves and evaluates intermediate models. [Wang et al., *ApproBiVT*](https://arxiv.org/abs/2308.02870) (**posted 2023-08-05**) describe held-out validation and checkpoint averaging as the conventional ASR recipe and report 2.5–4.6% relative CER improvements from their proposed train/validation-loss selection method on AISHELL-1/2. Those gains do **not** establish a benefit for a 42,567-parameter LID TCN. They establish that final-step selection is not the only defensible speech recipe.

Finally, repeatedly reading a nominal test result while changing models makes that set development feedback. [Dwork et al., *Generalization in Adaptive Data Analysis and Holdout Reuse*](https://arxiv.org/abs/1506.02629) (**posted 2015-06-08; NeurIPS 2015**) formalize how adaptive reuse can overfit a holdout. This tiny repository does not need their privacy-based reusable-holdout mechanism; it needs the simpler remedy of explicitly separating development from a once-only final test.

## 3. Validation must cover the failure modes, not just average frames

[Styles et al., *Investigating Model Performance in Language Identification: Beyond Simple Error Statistics*](https://www.isca-archive.org/interspeech_2023/styles23_interspeech.html) (**Interspeech, 2023-08**) analyze eight systems on spontaneous English–Mandarin speech. They find that most errors concentrate in 0.5–2 s segments, error rises with switch rate, and corpus-wide EER hides recording- and language-specific disparities. They explicitly recommend multiple metrics and individual-recording analysis. This is not Hindi–English evidence, but the evaluated failure modes match this project's short-prefix and switching requirements.

Accordingly, each candidate snapshot should produce four separate validation blocks:

### 3.1 Teacher fidelity

- monolingual KD, macro-averaged first over clips and then languages;
- switch KD reported separately, never pooled with the much larger monolingual frame count;
- student↔teacher top-1 agreement;
- teacher known-label accuracy beside student known-label accuracy.

KD is a diagnostic and final tie-breaker. A checkpoint must not win solely because it imitates a teacher error more closely.

### 3.2 Known-label prefix generalization

At fixed available-audio deadlines `{0.5, 1, 2, 4, full}` seconds, report:

- macro accuracy and macro-F1 over the seven languages;
- per-language recall and the minimum language recall;
- clip-macro NLL/Brier score if probabilities are compared;
- number of clips and independent source/voice clusters behind every aggregate.

Do not choose from frame micro accuracy: long clips and repeated frames would dominate it.

### 3.3 Boundary behavior

With model architecture and one routing policy frozen across checkpoints, report:

- raw stable-switch recall and fixed-policy switch recall at 0.5/1/2/3 s;
- miss count, with a miss never converted to a latency;
- median/p95 availability-time lag with `n/N` matched events;
- false/premature switches, wrong-language time, and UNKNOWN time;
- results by direction and target-span duration.

METHOD-0 must fix future-leaking target interpolation before switch-KD or switch-lag selection can be described as latency-valid. The selection machinery can be implemented earlier, but an invalid target trajectory remains invalid at every checkpoint.

### 3.4 Stability

- raw and committed changes per voiced minute;
- unmatched known-language switches per voiced hour;
- inverse `A→B→A` reversals;
- no-collar language diarization error or its miss/false/confusion components.

This prevents a checkpoint from winning on an early crossing created by severe posterior churn.

## 4. A practical data split for this repository

### Tier A: fast local development

Use the current 21 monolingual held-out clips and two switch files for a cheap metric trace at every snapshot. Label all results `synthetic_dev`. Because there is one voice per language and only one underlying Hindi/English pair, this tier may rank snapshots but cannot support a final generalization claim or a confidence interval.

### Tier B: pinned external model-selection data

Use the **validation**, never test, split of Hugging Face [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) for the seven monolingual classes. The official card states that train speakers differ from dev/test speakers and that the corpus is CC BY 4.0. Hub API metadata queried 2026-09-26 resolved revision [`70bb2e84b976b7e960aa89f1c648e09c59f894dd`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd), last modified 2026-05-15.

| Config | Validation rows | Locked test rows |
|---|---:|---:|
| `en_us` | 394 | 647 |
| `hi_in` | 239 | 418 |
| `mr_in` | 443 | 1,015 |
| `bn_in` | 402 | 920 |
| `ta_in` | 377 | 591 |
| `te_in` | 311 | 472 |
| `gu_in` | 432 | 1,000 |
| **Total** | **2,598** | **5,063** |

The published Hub schema does not expose a speaker-ID column. Therefore the card's train-versus-dev/test statement can be recorded, but speaker-macro confidence intervals or a validation-versus-test speaker-disjoint claim require upstream metadata/path audit and are otherwise **unverified**.

For a switch-development surface, use two complementary sources:

1. Create a deterministic, source-disjoint diagnostic from FLEURS `validation` Hindi and English clips after METHOD-0: both directions, multiple independent source pairs, and target-span bins `{0.5, 1, 2, 4}` s with sample-exact joins. This is deliberately synthetic and must be labelled as such.
2. Use the canonical HiACC **validation** directory for natural Hinglish utterance/mixed-span diagnostics after verifying speaker-ID intersections. HiACC lacks token times, so it cannot supply switch-lag truth without manual alignment.

Do not use the community Hub mirrors [`Trelis/hiacc-adult-test-eval`](https://huggingface.co/datasets/Trelis/hiacc-adult-test-eval) or [`Trelis/hiacc-child-test-eval`](https://huggingface.co/datasets/Trelis/hiacc-child-test-eval) for checkpoint selection. Their cards explicitly say eval-only. Hub API/dataset-server queries on 2026-09-26 found only `adult_test` (664 rows; revision `45782b4752f1e353173cd0b940332a984fc0a249`) and `child_test` (372 rows; revision `19f4fda91183b889f642e1e16c88c7bbeda64dad`) splits. The canonical [HiACC paper](https://doi.org/10.1016/j.dib.2025.111886) was published **2025-07-17** and reports speaker-independent train/validation/test partitions, but the archive's inconsistent counts and speaker intersections still need the audit documented in the dataset research note.

### Tier C: locked final reporting

After selecting the model and routing policy, score once on:

- pinned FLEURS `test` for seven-way monolingual/prefix results;
- untouched canonical HiACC adult/child `test` for natural Hinglish slices;
- the separately frozen 8 kHz and accented-English tests already proposed in the research inbox.

If a Tier C result changes the architecture, target, checkpoint rule, threshold, or dwell, it has become development feedback. Start a new locked test revision rather than continuing to call it untouched.

## 5. Deterministic checkpoint-selection contract

The first implementation should complete all 1,600 updates, save snapshots, and select retrospectively. The student is small; stopping early is not needed to save compute, and a full curve is more informative than choosing a patience value from the same tiny data.

### Snapshot cadence and shortlist

1. Save a minimal model snapshot at step 0 and every 10 steps through 1,600. Ten steps are one current loader pass, so this produces 161 comparable points.
2. Evaluate every snapshot on Tier A using cached features/targets and deterministic order.
3. Form a predeclared shortlist of at most five unique steps: best monolingual label score, best raw switch score, lowest development KD, final step, and the fixed midpoint step 800. Do not replace duplicates or add another candidate after seeing Tier B.
4. Evaluate those candidates once on Tier B and apply the selector below.

This limits external-validation compute and reduces analyst freedom while retaining the full local learning curve.

### Eligibility gates

For each Tier B candidate:

- all identity, finiteness, causal-availability, and stable-emission checks must pass;
- let `M*` be the best candidate's mean of language-macro accuracy at 1 s, 2 s, and full clip; require `M >= M* - 0.02`;
- require minimum-language recall no more than 0.05 below the best candidate's minimum-language recall;
- require monolingual committed false-switch rate no worse than the final-step baseline;
- require teacher target accuracy and coverage to be reported, so a poor teacher slice cannot silently decide the student.

The 2- and 5-point tolerances are **project proposals, not published standards**. Freeze them before running Tier B.

### Selection order

Among eligible candidates, choose lexicographically:

1. higher raw stable-switch recall at 1 s;
2. lower miss-penalized switch score, assigning every miss the 3 s deadline plus a fixed 1 s penalty rather than dropping it;
3. lower language-diarization error / wrong-language time;
4. lower raw flip/reversal rate;
5. lower macro development KD;
6. earlier optimizer step.

Publish the entire candidate table, not just the winner. The lexicographic rule reflects this assignment's headline switch case while hard-gating monolingual and worst-language quality. Its superiority over a scalar weighted score is **unverified**; its advantage is that every trade-off is inspectable and precommitted.

Keep the current router fixed during checkpoint selection. If threshold/EMA/dwell are also tuned, use a distinct policy-development partition or nested grouped folds. Jointly trying every checkpoint×policy combination on the same 23 synthetic files would multiply adaptive overfitting.

### Identity and reproducibility

The selected artifact should bind:

- trajectory identity: initialization, exact minibatch order, optimizer/settings, source, corpus, targets, and requested/completed steps;
- `selected_step` and selected model-state hash;
- every snapshot step/hash considered;
- Tier A and Tier B manifest IDs, source revisions, waveform hashes, and role (`model_dev`, never `test`);
- selector source/version, metric schema, gates, shortlist rule, full candidate metrics, and winning comparison path;
- an explicit flag that Tier C was not read during selection.

Record `trajectory_completed_steps=1600` separately from `selected_step`; otherwise an earlier winner can falsely look like a run that stopped at that step.

## 6. Balanced switch sampling: only after boundary diversity exists

The current switch clip accounts for 1/71 examples and one unique boundary. Its longer duration gives it more training frames than an ordinary clip, but it still teaches only the same Hindi speaker/text to the same English speaker/text transition. Repeating it more often would strengthen that exact shortcut.

After METHOD-0 and after constructing a source-disjoint switch-training pool, compare:

- `natural_clip_rate`: ordinary shuffled sampling;
- `one_per_4_batches`: one switch-containing example in every fourth batch;
- `one_per_2_batches`: one switch-containing example in every second batch.

Hold total optimizer steps, initialization, monolingual examples seen, target construction, checkpoint selector, and validation manifests fixed. Alternate Hindi→English/English→Hindi and span-duration bins. Compute per-example loss before regime averaging so an 8 s switch does not win solely by contributing more frames.

**Unverified hypothesis:** diversified balanced sampling will improve switch recall, whereas oversampling the current single splice will improve its training fit without external boundary benefit. Adopt a balanced arm only for at least +10 absolute points raw switch recall@1 s or at least 100 ms lower miss-penalized median lag, with monolingual macro score down no more than 2 points, worst-language recall down no more than 5 points, and no increase in false switches. If all arms miss, report the sampling ablation as negative and fix targets/data rather than increasing the rate again.

## 7. Minimal implementation order

1. Add snapshot saving and deterministic Tier A scoring without changing training or sampling.
2. Run the current seed once and determine whether any earlier snapshot beats step 1600. This is the direct METHOD-4 test.
3. Pin FLEURS validation/test manifests and add the Tier B shortlist selector.
4. Fix METHOD-0 and expand independent switch development/training examples.
5. Only then run the switch-sampling ablation.

The first run should not checkpoint-average. ASR literature supports averaging as a possible later ablation, but it adds a second model-combination choice before this repository has established that basic validation selection works.

## 8. Claims that remain unverified

- That an earlier checkpoint will outperform step 1600. The current artifacts contain no intermediate model states.
- That FLEURS-validation ranking predicts synthetic-call, natural Hinglish, or 8 kHz ranking.
- That HiACC's published partitions have zero speaker overlap until the canonical archive is audited locally.
- That the proposed lexicographic selector is better than a scalar score or checkpoint averaging.
- That balanced switch sampling helps this model; the existing one-boundary corpus cannot test the hypothesis fairly.
- Any confidence interval treating frames, repeated clips from one voice, or the two reversed switch files as independent observations.

## Sources checked

- Local [`scripts/train.py`](../scripts/train.py), [`data/generated/manifest.jsonl`](../data/generated/manifest.jsonl), [`results/train_metrics.json`](../results/train_metrics.json), and [`results/eval_metrics.json`](../results/eval_metrics.json), audited **2026-09-26**.
- [FLEURS paper](https://arxiv.org/abs/2205.12446), posted **2022-05-25**; [Hugging Face dataset card](https://huggingface.co/datasets/google/fleurs) and [Hub API](https://huggingface.co/api/datasets/google/fleurs), queried **2026-09-26**.
- [Styles et al., Interspeech 2023](https://www.isca-archive.org/interspeech_2023/styles23_interspeech.html), published **2023-08**.
- [Wang et al., ApproBiVT](https://arxiv.org/abs/2308.02870), posted **2023-08-05**.
- [Wang et al., Robust Distillation](https://proceedings.mlr.press/v216/wang23e.html), UAI **2023-07-31 to 2023-08-04**.
- [Wu et al., KD mechanisms](https://proceedings.mlr.press/v243/wu24a.html), PMLR **2024-12-15**.
- [Dwork et al., adaptive holdout reuse](https://arxiv.org/abs/1506.02629), posted **2015-06-08**.
- [HiACC primary article](https://doi.org/10.1016/j.dib.2025.111886), online **2025-07-17**; [adult](https://huggingface.co/datasets/Trelis/hiacc-adult-test-eval) and [child](https://huggingface.co/datasets/Trelis/hiacc-child-test-eval) Hub test mirrors, queried **2026-09-26**.
