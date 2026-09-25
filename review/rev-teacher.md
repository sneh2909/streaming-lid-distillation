# Teacher target, audio, and dataset review

## Summary

- The sample/frame bookkeeping is correct. A teacher target at feature frame `i` is anchored at that frame's right edge, and student frame `i+21` with four feature frames of lookahead consumes audio through exactly `t_i+250 ms`; the tail validity inequality is also off-by-one correct.
- The main methodological problem is target quality, not tensor alignment. Seven-way renormalization turns very low supported-language mass into confident supervision, and the current 2 s local ECAPA windows are poor switch labelers even after that renormalization.
- `mean_anchor_label_agreement` materially overstates anchor performance: the current artifact reports 91.23%, whereas pooling its 190 anchors gives 74.74%. It is a macro average over clips, even though 91 clips have one anchor and each of the three switch clips has 33.
- The cache contains its language-column order but the dataset ignores it, so a stale cache with a reordered seven-language configuration can silently train every class against the wrong column.
- The current 94-example cache loads with finite, normalized targets, the frontend frame counts match the cached lengths, the speaker audit is disjoint, and the focused tests pass.

## Findings

### [METHOD] scripts/teacher_targets.py:147-156 — Closed-set renormalization turns teacher rejection into confident supervision

`chosen` retains only seven of the ECAPA teacher's 107 log posteriors. Both `raw_anchors` and `soft_anchors` are then normalized only across those seven classes. Although line 149 measures the retained mass, neither the target nor `DistillationDataset` uses that mass to mask or weight training. The result is not merely a seven-class version of the teacher posterior: it can be highly confident precisely when the frozen teacher prefers an unsupported language.

I reran the cached teacher read-only on representative clips and compared its 107-way top class with the saved conditional target. For `en_train_02`, the actual top class was `cy` (Welsh), total probability on all seven supported classes was only `0.051326`, but the saved seven-way raw target was `0.9994` English and the `T=2` training target was `0.9646` English. Across the three switch clips, actual 107-way anchor accuracy against the known segments was respectively `12/33`, `11/33`, and `12/33` (33.3–36.4%); after forced seven-way conditioning it became `23/33`, `17/33`, and `18/33`. The current cache also contains an anchor with only `0.005357` supported-language mass that becomes a `0.9847` conditional-English target before temperature softening.

That is unsuitable as unweighted soft supervision: it teaches the student an arbitrary best-of-seven answer when the teacher is effectively rejecting the window. Add an explicit `other/unknown` class, or interpolate the retained mass and mask/down-weight low-mass frames. Report full-107 top-1/coverage separately from conditional seven-way accuracy. If closed-set conditioning is retained for this demo, set and justify a mass threshold using a development split rather than merely storing `in_set_mass` for inspection.

### [METHOD] scripts/teacher_targets.py:49-65 — The 1.75 s past / 0.25 s future window bakes large switch lag into the labels

ECAPA produces one utterance-level decision for the entire two-second window. With seven times as much past as future, that decision estimates the dominant language in a backward-heavy window, not the instantaneous language at `t_i`. This is causally alignable, but it is not empirically a useful local labeler in the current data.

On the stable 94-clip target snapshot, conditional anchor accuracy is only `23/33` for the sole training switch, `17/33` for held-out Hindi→English, and `18/33` for held-out English→Hindi. A read-only scan of `anchor_probs` showed:

- training Hindi→English: first English anchor at 5.275 s and first permanently English anchor at 6.275 s, versus the 4.000 s boundary;
- held-out Hindi→English: first and permanently English anchor at 6.775 s;
- held-out English→Hindi: a false Hindi prediction at 2.025 s, but no permanent Hindi run until 5.275 s.

Thus the teacher targets alone contain stable switch lags of 1.275–2.775 s at 250 ms anchor resolution. This verifies the stated concern that measured student lag can be dominated by the teacher construction. The one training switch also supplies wrong conditional labels on 30.3% of its anchors.

Before treating these as switch targets, sweep window history/duration and teacher choice against the known synthetic boundaries, establish an acceptance criterion for local accuracy and teacher-only lag, and report teacher and student lag separately. A teacher with locally meaningful frame/chunk outputs is preferable. At minimum, do not attribute the end-to-end lag to the student when the target trajectory itself transitions this late.

### [METHOD] scripts/teacher_targets.py:169-221 — “Mean anchor agreement” is a macro clip average that hides the hard regime

Line 178 computes an anchor mean within each clip, but lines 207-221 then average those 94 per-clip numbers with equal weight. In the current snapshot, 91 monolingual clips contribute one anchor each while three switch clips contribute 33 anchors each. The resulting `mean_anchor_label_agreement` is `0.9123146`; pooling the actual correct counts gives `142/190 = 0.7473684`. Each switch anchor is effectively weighted 33 times less than each monolingual anchor, even though local switch targets are the part that needs validation.

The metric is additionally conditional on the seven selected classes, as described above, so it is neither global teacher accuracy nor pooled anchor accuracy. Accumulate integer `correct` and `total` counts, report their ratio, split results by `converged_utterance` versus `local_windows`, and retain a separately named `macro_clip_agreement` only if that statistic is wanted.

### [BUG] src/streaming_lid/data.py:101-105 — Cached target column order is never validated

The generator deliberately stores `language_codes` in every `.npz` at `scripts/teacher_targets.py:166`, but the loader reads only `teacher_soft_targets`. Length equality cannot catch a reorder: a cache produced as `[en, hi, ...]` still has shape `[T, 7]` after configuration changes to `[hi, en, ...]`, and the loss will silently train the current class 0 (`hi`) against the cached English column.

Read and compare `tuple(target_file["language_codes"])` with the configured `LANGUAGE_CODES` before accepting the target. Also validate class count, finiteness, non-negativity, and row sums; ideally bind the cache to teacher ID, temperature, frontend timing, and window settings through versioned metadata or a configuration hash.

### [DEFEND] scripts/teacher_targets.py:49-65 — Matching the right edge does not match the teacher's past evidence

The right-context argument is exact, but the evidence histories differ. With a 127-frame student receptive field, prediction `i+21` reaches back to raw feature `i-105`; relative to the end of teacher frame `i`, that is 1,075 ms of raw history. The teacher gets 1,750 ms, a 675 ms excess. At clip start, `extract_window` places the first sample after 27,600 zero samples (1.725 s), and the teacher receives no length mask. Yet the loss ramp reaches full weight at target frame 99 (anchor end about 1.015 s), when roughly 735 ms of the teacher window is still artificial leading silence.

This does not create future leakage, and compressing a stronger teacher into a smaller-history student can be intentional. It should nevertheless be stated as a capacity/evidence mismatch rather than claiming that the contexts match merely because their latest sample does. Either shorten the teacher history, enlarge the student history, or validate the mismatch and mask/ramp the edge region through the point where a full teacher window exists.

### [MINOR] src/streaming_lid/data.py:44-54 — Unknown split names are silently omitted from the speaker audit

Records with nonempty speaker IDs but a split other than `train`, `heldout`, or `switch` pass through the loop without entering either side of the overlap check. A misspelled `heldout` or a future `test` split can therefore make `require_speaker_disjoint` report success without auditing those speakers. The generated manifest currently uses only the three recognized values. Reject unknown split values, or accept an explicit set of training/evaluation split names.

## Verified correct

- Frozen inference is real: the teacher is put in evaluation mode and every call is under `torch.inference_mode()`. Inspection of the installed SpeechBrain `EncoderClassifier.classify_batch` confirms that its first result is a log posterior, so exponentiating it for retained mass and applying `softmax(log_p/T)` are mathematically correct before the closed-set caveat above.
- Monolingual and mixed clips are separated correctly: known monolingual records receive one repeated full-utterance posterior, while every `language == "mixed"` record uses local windows. A switch is not accidentally assigned one whole-clip label.
- A boundary probe of `extract_window` at frame 200 returned exactly 32,000 samples, source samples `[4400, 36400)`, and latest time 2,275 ms. The corresponding student latest feature end for `i+21+4` was also 2,275 ms. There is no one-frame or one-window error here.
- `center=False`, no utterance CMVN, and right-padding only for audio shorter than one analysis window keep the log-mel frontend causal. `feature_frame_count` matched the actual frontend for sample lengths `0, 1, 399, 400, 401, 559, 560, 561, 16000` and for all current cached examples.
- Anchor arrays are sorted, include frame zero and the final feature frame, and interpolation yields normalized probabilities. Loading the current 94 examples found finite feature/target tensors and maximum target row-sum error `2.38e-7`; `DistillationDataset` rejected no frame-length mismatch.
- `collate_distillation_batch` records unpadded lengths and pads features and targets consistently. In the delayed loss, `i+D+L < length` is the correct condition for requiring real right context; padded target rows do not contribute.
- The speaker audit correctly includes component speakers of switch clips and detects train/evaluation intersections for the supported split schema. The current manifest reports no overlap.
- `UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch .venv/bin/pytest -q tests/test_alignment.py tests/test_causality.py tests/test_data_split.py` passed all 7 tests in 29.74 s.

## Interview questions

1. **Why is the label delay 21 frames rather than 25?** The output at `i+21` explicitly stacks feature frames through `i+25`; the four-frame model lookahead supplies the remaining 40 ms. Since frames share a 10 ms hop, the latest feature ends exactly 250 ms after teacher frame `i` ends.
2. **What probability distribution is actually distilled?** It is the ECAPA 107-way log posterior restricted to seven configured classes, renormalized, temperature-softened at `T=2`, and linearly interpolated. It is not the original teacher posterior, and it currently loses the teacher's out-of-set uncertainty.
3. **Why repeat one utterance posterior over a monolingual clip?** The manifest asserts a stationary language, so the converged decision is a low-variance semantic target; the early ramp acknowledges that the causal student cannot reproduce full-utterance confidence immediately. The same construction would be invalid for a switch clip.
4. **How much switch lag belongs to the teacher?** In the current anchor targets, the first permanent target-language prediction occurs 1.275–2.775 s after the known boundary, depending on the switch clip. Teacher-only lag must be reported before interpreting student/detector lag.
5. **Does exact future alignment mean teacher and student see identical evidence?** No. Their latest sample matches, but the teacher has 1.750 s of past while this student has about 1.075 s relative to the target-frame end. That is a deliberate-or-accidental compression problem, not a causality violation.
6. **How is batch padding kept out of the objective?** For target `i`, the mask requires the latest raw feature index used by prediction `i+D`, namely `i+D+L`, to be strictly less than that item's true length. Targets and logits in padded or unavailable tail positions therefore receive zero weight.
