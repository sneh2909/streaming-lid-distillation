# Restricted-label-space distillation: retained teacher mass and UNKNOWN routing

Research cutoff: **2026-09-26**. Web sources, Hugging Face model cards, Hub API metadata, and the repository's current cached targets were checked on that date. Any expected model gain below is explicitly marked **unverified**.

## Bottom line

The current 107-to-7 projection computes a valid probability distribution, but it answers a narrower question than the full teacher:

> Assuming the utterance is one of `en hi mr bn ta te gu`, which of those seven is most likely?

It does **not** preserve how much probability the teacher assigned to that assumption. That distinction is material in the current cache: among the 190 teacher anchors belonging to the current 94-record manifest, 61 (32.1%) retain less than 50% of the teacher's original probability mass on the seven deployment languages. Yet the conditional seven-way maximum averages 0.872 on those 61 anchors, and 41/61 are at least 0.9 after renormalization.

This does not mean all 61 anchors should become `UNKNOWN`. The training corpus is deliberately known to contain one of the seven languages, and the full teacher is sometimes simply wrong. One known-English training clip has only 5.13% full-teacher mass on the seven classes and is classified as Welsh with 93.55% probability, while the conditional seven-way target is 99.94% English. Blindly converting the discarded mass into an `OTHER` label would turn that teacher error into stronger wrong supervision.

The defensible sequence is therefore:

1. Keep conditional seven-way targets for the current **known-in-set** clips.
2. First test retained mass as a bounded reliability weight and report its behavior around switches; do not call a non-normalized seven-vector a KL target.
3. Add an explicit in-set/other gate only with separately sourced out-of-set speech. A factorized binary gate plus the existing seven-way conditional head is the cleanest version and adds only 33 parameters to the 42,567-parameter student.
4. Calibrate the gate on speaker- and language-disjoint development data. The ECAPA card reports classification error, not posterior calibration, so retained mass is a score until validated.
5. Keep `OTHER_LANGUAGE`, non-speech/blank, and temporary uncertainty conceptually separate. A VAD or explicit blank state should handle silence rather than teaching that silence is another language.

This refines backlog METHOD-1 and corrects the earlier research-inbox suggestion to compare seven-way targets "with/without renormalization." An unnormalized vector is not a categorical target for the existing KL implementation. The meaningful alternatives are **conditional KD**, **mass-weighted conditional KD**, or a **normalized eight-way/factorized model that represents the discarded mass**.

## 1. What the current artifact and code actually produce

The current teacher is Hugging Face [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa). The Hub card says it is a 16 kHz, utterance-level ECAPA-TDNN trained on VoxLingua107, covers 107 language classes, is Apache-2.0, and reports 6.7% error on the VoxLingua107 development set. The card does **not** publish ECE, Brier score, OOS rejection, or any other calibration result. The [Hub API metadata](https://huggingface.co/api/models/speechbrain/lang-id-voxlingua107-ecapa), queried 2026-09-26, resolved `main` to `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`, last modified 2024-11-27.

The pinned project's SpeechBrain 1.0.3 [`EncoderClassifier.classify_batch`](https://github.com/speechbrain/speechbrain/blob/v1.0.3/speechbrain/inference/classifiers.py) documents and returns a vector of **log posterior probabilities**. The model's [pinned Hub hyperparameters](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/blob/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9/hyperparams.yaml) set `out_n_neurons: 107`, and SpeechBrain's X-vector classifier ends in log-softmax.

In `scripts/teacher_targets.py`, let `ell_k` be the returned log posterior for teacher class `k`, and let `S` be the seven deployment classes. The code does this:

```text
p_k = exp(ell_k)                         # full 107-way posterior
s   = sum(k in S) p_k                   # retained in-set mass, cached separately
r_c = softmax(ell_c over c in S)        # p_c / s at T=1
```

At the configured distillation temperature `T=2`, it uses:

```text
r_T(c) = softmax(ell_c / T over c in S).
```

This is exactly the full softened teacher distribution **conditioned on membership in `S`**. It is not an implementation bug. The information loss is that `s_T`, the softened probability of membership in `S`, is discarded from the KD target.

The same closed-set issue survives a teacher replacement. Meta's official [`facebook/mms-lid-126`](https://huggingface.co/facebook/mms-lid-126) card defines 126 language outputs and no unknown output. The community [`surogate/ambernet-langid`](https://huggingface.co/surogate/ambernet-langid) card explicitly warns that it always returns one of 107 languages and has no unknown, silence, or non-speech class. A larger label inventory reduces, but does not eliminate, the need for a deployment reject option.

## 2. The probability-space choices

Let the full teacher distribution after applying the chosen temperature be `pi_T(k)`. Define:

```text
s_T       = sum(k in S) pi_T(k)
r_T(c)    = pi_T(c) / s_T, c in S
q_T       = [pi_T(c) for c in S, 1 - s_T]
```

These give four distinct designs.

### A. Conditional seven-way KD: the current design

```text
L_cond = T^2 KL(r_T || softmax(z / T)).
```

This is appropriate when an external fact says every training example belongs to `S`. That fact is true for the monolingual synthetic clips and, outside silence/padding, for their Hindi-English splices. It deliberately asks the teacher only for within-set confusions.

Its limitation is reliability: a teacher that assigns 0.5% total mass to `S` can still supply a nearly one-hot conditional target. The student has no way to distinguish that anchor from one where the teacher assigns 99.9% to `S`.

### B. Mass-weighted conditional KD: minimal change

```text
L_weighted = w(s_1) T^2 KL(r_T || softmax(z / T)).
```

Using `w(s_1)=clip(s_1, floor, 1)` preserves the current seven-class inference interface while reducing the gradient from anchors on which the full teacher prefers excluded languages. Here `s_1` should be named a **reliability weight**, not silently treated as a calibrated probability.

This is the safest first experiment for the current known-in-set corpus. Its main risk is that low-mass frames cluster near padded edges and language boundaries, so it may obtain a smoother loss simply by ignoring the examples needed for fast switching. Boundary recall and lag must gate adoption.

### C. One eight-way coarse target: exact mass preservation

For a student whose eighth class is `OTHER_LANGUAGE`, the normalized target is:

```text
q_T(c)     = pi_T(c), c in S
q_T(other) = sum(k not in S) pi_T(k) = 1 - s_T.
```

This is the probability push-forward under the mapping that leaves each deployment class alone and merges the other 100 teacher classes. It is the correct categorical target if the desired task really is seven languages versus all other teacher languages.

Do **not** take the seven `T=1` probabilities, leave their sum below one, and pass them to `F.kl_div`. That is neither a normalized seven-way distribution nor the eight-way coarse distribution; it changes gradient magnitude with `s` and no longer represents a KL between categorical distributions.

Temperature order matters. For a `T=2` coarse target, first compute `softmax(ell / 2)` over all 107 classes and then sum the excluded 100 entries. `sum(exp(ell))` cached at `T=1` cannot reconstruct `s_2`, because it does not retain the shape of the excluded distribution.

### D. Factorized in-set gate plus conditional language head: recommended UNKNOWN design

Let the student emit:

- `g = P_student(Y in S | x)` from one binary logit;
- `u(c) = P_student(Y=c | Y in S, x)` from the existing seven-way head.

The implied eight-way distribution is `[g u(c), 1-g]`. At one common temperature, the coarse KL decomposes exactly:

```text
KL(q_T || [g u, 1-g])
  = KL(Bernoulli(s_T) || Bernoulli(g))
    + s_T KL(r_T || u).
```

This makes the design intent explicit: learn whether routing is in scope, then identify the language conditionally. It also lets the gate and language head be calibrated and ablated independently. Adding `Linear(32, 1)` beside the current final `Linear(32, 7)` adds 33 parameters, taking the student from 42,567 to **42,600 parameters**, with no added acoustic lookahead.

For the present labeled clips, however, the gate target should be the known fact `in_set=1`, not the fallible teacher's `s`. For separately collected out-of-set-language clips it should be `in_set=0`. A soft teacher gate is most defensible on genuinely unlabeled transfer traffic and must still be calibrated.

## 3. Local audit of the current cached targets

### Method

The audit joined `.npz` targets only to IDs in the current `data/generated/manifest.jsonl`; 34 stale target files were present and intentionally excluded. This is another reason to complete backlog BUG-3, but it does not affect the statistics below. The 94 matched records contain:

- 70 monolingual training anchors plus 33 anchors from the training switch;
- 21 held-out monolingual anchors;
- 66 anchors from the two switch-evaluation clips.

The cached `in_set_mass` is available at the 250 ms teacher anchors, not at every interpolated 10 ms target frame. Consequently these are **anchor-level diagnostics**, not frame-weighted training statistics.

### Retained mass

| Current split | Clips | Anchors | Mean `s_1` | `s_1 < .10` | `s_1 < .25` | `s_1 < .50` |
|---|---:|---:|---:|---:|---:|---:|
| train, including one switch | 71 | 103 | 0.7778 | 10 (9.7%) | 15 (14.6%) | 25 (24.3%) |
| held-out monolingual | 21 | 21 | 0.9476 | 0 | 0 | 0 |
| switch evaluation | 2 | 66 | 0.5317 | 5 (7.6%) | 21 (31.8%) | 36 (54.5%) |
| all current records | 94 | 190 | 0.7111 | 15 (7.9%) | 36 (18.9%) | 61 (32.1%) |

Across all anchors, the minimum is 0.00588 and the median is 0.98018. The bimodality matters more than the mean: many monolingual examples are essentially fully in-set, while edge/boundary windows and several English examples carry much less retained mass.

For the 61 anchors with `s_1 < 0.5`, mean conditional seven-way maximum is 0.87198, median is 0.98007, and 41/61 have conditional maximum at least 0.9. For the 15 anchors below 0.1, 8/15 still have conditional maximum at least 0.9. The renormalization factor is `1/s`; at the minimum-mass anchor it multiplies selected probabilities by about 170.

### Full-output spot checks

The full 107-way teacher was rerun on eight low-mass examples at the locally resolved Hub SHA. Four representative results are below. `r_1 max` is the current unsoftened conditional target; `s_2` and `r_2 max` show the corresponding values after full `T=2` softening.

| Record / teacher anchor | `s_1` | `r_1 max` | Full-teacher top class at `T=1` | `s_2` | `r_2 max` |
|---|---:|---:|---|---:|---:|
| `en_train_02`, full clip | 0.051326 | en 0.999389 | Welsh 0.935458 | 0.156634 | en 0.964575 |
| `switch_en_hi_eval`, 1.00 s | 0.013241 | en 0.979615 | Slovenian 0.855678 | 0.068195 | en 0.793620 |
| `switch_hi_en_train`, 4.25 s | 0.048022 | hi 0.997966 | Esperanto 0.927690 | 0.139768 | hi 0.920757 |
| `switch_hi_en_train`, 0.00 s | 0.066183 | ta 0.944973 | Kannada 0.904887 | 0.161356 | ta 0.656878 |

These examples establish three things locally:

1. The mass loss is real and sometimes extreme; it is not just numeric roundoff.
2. Conditional projection can recover the correct deployment label from a full-teacher error on a clip independently known to be in-set.
3. Retained mass changes with temperature and cannot be derived from the cached conditional target alone.

They do **not** establish that `s` is calibrated or that a particular threshold generalizes. Eight manually inspected examples are diagnostics, not an OOS benchmark.

## 4. What the literature supports

### Open-set LID requires a non-exhaustive decision model

[McLaren et al., *Calibration Approaches for Language Detection*](https://www.isca-archive.org/interspeech_2017/mclaren17_interspeech.pdf) (Interspeech 2017, 2017-08; accessed 2026-09-26) point out that ordinary multi-class calibration assumes mutually exclusive **and exhaustive** language hypotheses. That assumption fails when deployment contains unseen OOS languages. Their language-detection experiments favor explicitly detection-oriented binary calibration under several mismatched conditions. The models predate neural ECAPA, but the decision-theory issue applies directly to a seven-language router.

[Wang et al., *The OLR 2021 Challenge: Datasets, Rules and Baselines*](https://www.isca-archive.org/interspeech_2022/wang22ga_interspeech.pdf) (Interspeech 2022; accessed 2026-09-26) treat all interfering languages as one unknown language when computing open-set `Cavg`. This supports measuring known-language misses and OOS false accepts, not only closed-set top-1 accuracy.

[Zhang and Hansen, *Training candidate selection for effective out-of-set rejection in robust open-set language identification*](https://doi.org/10.1121/1.5017608) (JASA, published 2018-01; [abstract/metadata](https://pubmed.ncbi.nlm.nih.gov/29390735/), accessed 2026-09-26) show that the composition of OOS training languages matters. Their selected representatives reduced OOS training-language diversity by 86% while maintaining performance similar to using all probe OOS languages, and beat random candidate selection. This is older i-vector evidence, not a neural-KD result, but it argues for deliberate near/far and language-held-out OOS splits rather than one arbitrary `OTHER` corpus.

### Short windows contain ambiguous and non-speech material

[Dey, Mondal, and Kurmi, *Teacher-Free Knowledge Distillation for Improving Short-Utterance Spoken Language Identification*](https://www.isca-archive.org/interspeech_2025/dey25_interspeech.html) (Interspeech 2025, 2025-08; accessed 2026-09-26) attribute 36.94% of analyzed 2 s English misclassifications to non-speech, named entities, fillers, or overlapping speech. Their best short-utterance methods condition updates on correct predictions and use entropy weighting. This supports reliability-aware supervision and a VAD/blank path, but their entropy scheme is utterance-level and their gains do not verify retained-mass weighting for Hindi-English switch frames.

### Softmax mass is not automatically calibrated confidence

[Guo et al., *On Calibration of Modern Neural Networks*](https://proceedings.mlr.press/v70/guo17a.html) (ICML 2017, 2017-08; accessed 2026-09-26) show that modern neural-network probabilities are often miscalibrated and that temperature scaling can help on held-out calibration data. This is general classification evidence, not spoken-LID evidence. It is enough to reject the unsupported assumption that ECAPA's raw `s` already equals the true probability of being in the seven-language set.

## 5. Hugging Face data for a valid gate experiment

Hugging Face [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) is CC BY 4.0 and exposes 103 configurations (102 languages plus the aggregate configuration). The Hub datasets-server [split endpoint](https://datasets-server.huggingface.co/splits?dataset=google%2Ffleurs), queried 2026-09-26, confirmed the exact configs below and their train/validation/test splits.

A useful student-OOS pool is:

- close Indic hard negatives already known to ECAPA: `as_in`, `kn_in`, `ml_in`, `ne_np`, `pa_in`, `sd_in`, `ur_pk`;
- a close Indic language absent from ECAPA's published 107 labels: `or_in` (Odia), useful as a true teacher-OOD test;
- far controls: `de_de`, `fr_fr`, `es_419`.

Do not put every OOS language into training. Rotate language-held-out folds so that each final test contains OOS languages unseen by the gate during fitting and threshold selection. Use FLEURS `train` only for model fitting, `validation` for gate calibration/policy selection, and `test` once for reporting. Keep the seven supported FLEURS configs and the existing Svarah accented-English slice as in-set controls.

This tests two distinct questions:

1. **Student-OOS but teacher-known:** can the 107-way teacher's discarded mass supervise a seven-language router?
2. **Teacher-OOD:** can the learned gate reject a language the teacher itself never modeled?

Success on the first does not imply the second. It also does not imply rejection of music, silence, overlapping speakers, or channel corruption; those need separate slices.

## 6. Recommended experiment, in order

### Phase 0 — cache and score before training

For every teacher anchor, cache with artifact provenance:

- full-teacher top-1 label and probability;
- `s_1`, conditional `r_1`, full-teacher entropy;
- `s_T` and conditional `r_T` at the actual KD temperature;
- the seven selected absolute probabilities, not only their conditional version.

Interpolate `s` to the same 10 ms grid as the target if it will weight frame losses. Report distributions separately for monolingual speech, switch interiors, a ±500 ms boundary region, padded edges, and VAD non-speech. This phase should not change a checkpoint.

On frozen FLEURS development data, compare these teacher-side OOS scores before building a student gate:

1. retained mass `s_1`;
2. absolute selected maximum `s_1 * max(r_1)`;
3. current conditional maximum `max(r_1)`;
4. conditional entropy.

Report AUROC, AUPR-out, OOS false-accept rate at 95% in-set acceptance, in-set false-reject rate, and results by language/duration. Fit any scalar calibration only on validation. **Expected result is unverified**; the hypothesis is that `s_1` or absolute selected maximum will dominate conditional maximum for student-OOS detection.

### Phase 1 — minimal mass-weighted KD

Keep the seven-way model, target type, delay, seed, steps, and policy fixed. Compare:

1. current early ramp;
2. ramp × `clip(s_1, 0.10, 1)`;
3. ramp × `clip(s_1, 0.25, 1)`;
4. ramp × binary gate `1[s_1 >= tau]`, with `tau` selected only on development data.

The clipped arms are deliberately bounded so one bad teacher window cannot erase all supervision. Slice results by mass decile and boundary distance. The existing research hypothesis of a 1–2 point short-clip macro gain or 10–20% fewer flips remains **unverified**. Adopt only if known-label macro accuracy improves by at least 2 points or flips fall at least 10%, while median switch lag rises no more than 100 ms and 0.5–1 s embedded-span recall falls no more than 2 points.

### Phase 2 — factorized UNKNOWN gate

Only after Phase 0 establishes a usable score, add the one-logit gate:

- supported-language labeled audio: hard gate target `in_set=1`, plus existing conditional KD;
- FLEURS OOS-language audio: hard gate target `in_set=0`; no seven-way language loss;
- optional unlabeled production audio: teacher soft gate target only after calibration and privacy/provenance review.

Keep class-balanced batches and language-held-out OOS test folds. At inference, first apply VAD; when voiced, route to `UNKNOWN` if the calibrated gate fails, otherwise apply the existing seven-way posterior policy. Do not let a transient boundary dip trigger an ASR handoff without a separately tuned gate dwell/hysteresis rule.

Report:

- supported-language macro accuracy/F1, NLL, and coverage-risk curve;
- gate AUROC, AUPR-out, Brier score/reliability plot, FPR@95 in-set acceptance, and per-OOS-language false accepts;
- wrong-route seconds, `UNKNOWN` voiced time, UNKNOWN churn, initial decision coverage, switch recall/lag, and false switches;
- `s` and gate performance by duration, channel, boundary distance, and seen versus unseen OOS language.

Engineering acceptance target (**not a literature-backed expected result**): at least 95% in-set acceptance, no more than 2 points absolute loss in accepted-speech seven-way macro accuracy, at least 25% relative reduction in OOS false accepts versus a calibrated conditional-max rejector, and no more than 100 ms added median switch lag. Failure to meet this target means retain the seven-way model and use bounded loss weighting only.

## 7. What remains unverified

- Whether retained mass predicts correctness or OOS status on natural Indian speech; the current audit is mostly synthetic TTS.
- Whether mass weighting improves this student. No new student was trained in this research iteration.
- Whether an UNKNOWN gate generalizes to languages held out at gate-training time.
- Whether `s_1`, `s_2`, absolute selected maximum, entropy, or a calibrated combination is the best gate score.
- Whether low switch-window mass comes mostly from padding/silence, mixed phonetics, synthetic voices, teacher calibration, or the 1.75 s past-context window. The examples show all are plausible, not which dominates.
- Whether the FLEURS read-speech operating point transfers to real 8 kHz calls.
- No reviewed Hugging Face spoken-LID checkpoint supplied a documented, calibrated UNKNOWN output suitable as a drop-in streaming gate. This is a dated, non-exhaustive Hub search finding, not proof that none exists.

## Source and command record

Primary sources used, all accessed **2026-09-26**:

- [SpeechBrain VoxLingua107 ECAPA Hub card](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) and [Hub API metadata](https://huggingface.co/api/models/speechbrain/lang-id-voxlingua107-ecapa) — frozen teacher output space, license, published evaluation, and resolved revision.
- [SpeechBrain 1.0.3 `EncoderClassifier`](https://github.com/speechbrain/speechbrain/blob/v1.0.3/speechbrain/inference/classifiers.py) and [pinned model hyperparameters](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/blob/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9/hyperparams.yaml) — log-posterior return contract and 107 outputs.
- [Meta MMS-LID-126 Hub card/config](https://huggingface.co/facebook/mms-lid-126) — closed 126-language output space, 1B parameters, CC BY-NC 4.0.
- [AmberNet community Hub card](https://huggingface.co/surogate/ambernet-langid) — explicit absence of unknown/silence/non-speech outputs.
- [McLaren et al., language-detection calibration](https://www.isca-archive.org/interspeech_2017/mclaren17_interspeech.pdf) — Interspeech 2017; open-set/exhaustive-hypothesis problem and binary calibration.
- [Zhang and Hansen, OOS candidate selection](https://doi.org/10.1121/1.5017608) — JASA **2018-01**; representative OOS-language training evidence.
- [Wang et al., OLR 2021 challenge](https://www.isca-archive.org/interspeech_2022/wang22ga_interspeech.pdf) — Interspeech 2022; unknown aggregation in open-set `Cavg`.
- [Dey et al., short-utterance LID KD](https://www.isca-archive.org/interspeech_2025/dey25_interspeech.html) — Interspeech **2025-08**; out-of-scope short-window analysis and reliability-aware soft labels.
- [Guo et al., neural-network calibration](https://proceedings.mlr.press/v70/guo17a.html) — ICML **2017-08**; temperature scaling and the need to validate confidence calibration.
- [`google/fleurs` Hub dataset](https://huggingface.co/datasets/google/fleurs) and [live split/config endpoint](https://datasets-server.huggingface.co/splits?dataset=google%2Ffleurs) — CC BY 4.0 multilingual in-set/OOS evaluation source.

Local diagnostics used only current-manifest target files and six-thread CPU execution. The two read-only audit commands computed mass quantiles/conditional confidence from cached `.npz` files and reran the pinned teacher on eight selected windows. Key reproducible counts are 94 matched target files, 190 anchors, 61 anchors below `s_1=0.5`, and 41/61 of those with conditional maximum at least 0.9.
