# Causally valid expansion of sparse teacher posteriors

Research cutoff: **2026-09-26**. Web sources, primary papers, Hugging Face model cards/API metadata, repository code, and the current cached targets were checked on that date.

This memo addresses backlog **METHOD-0** only: how to turn an expensive offline teacher evaluated every 250 ms into 10 ms training targets without silently extending the teacher's future-information horizon. It does not choose a new teacher, redesign the student, or claim that a causally valid target is an accurate language-boundary target.

Evidence labels used below:

- **Local fact:** derived from the checked-in code or current content-bound target cache.
- **Direct evidence:** a spoken-LID/language-diarization result.
- **Transfer evidence:** a result from streaming ASR, speaker diarization, or another causal sequence task. Its LID benefit is **unverified**.
- **Proposal:** a concrete repository change whose model gain is **unverified until rerun**.

## Bottom line

The current switch target is causally invalid at 24 of every 25 ordinary frames. `numpy.interp` constructs a piecewise-linear value between the previous and next teacher anchors. The next anchor's ECAPA window includes audio up to 250 ms after that next anchor. Consequently, a target between anchors depends on audio **260–490 ms after its own frame**, while the aligned student can use only 250 ms. Computing the target offline does not make that evidence available to the streaming student.

The safest immediate correction is **previous-anchor hold**, with an explicit per-frame availability assertion. It needs no extra deployment latency and is already implemented in the isolated target-ablation driver. It is not a free accuracy fix: it makes each target up to 240 ms old and, on the present cache, moves the intended teacher transition 140–220 ms later than linear interpolation.

The preferred follow-up is therefore not a fancier interpolator. Run the teacher more densely offline and retain previous-anchor hold:

- 50 ms teacher hop: at most 40 ms target age, about 4.88 times the current switch-window calls, no inference-time cost or latency change;
- 10 ms teacher hop: no expansion at all, about 24.18 times the current switch-window calls, no inference-time cost or latency change;
- current 250 ms hop plus linear interpolation: keep only as an explicitly invalid diagnostic, or pay another 240 ms of student evidence delay.

No reviewed paper or Hugging Face card validates probability interpolation between sparse offline LID windows as a frame-level KD target. The closest evidence instead favors context matching, explicit paid delay, and native fine-rate outputs. All claimed downstream gains remain **unverified**.

## 1. Exact timing contract

Let the 10 ms feature-hop index be `i`, with feature-frame end

```text
e(i) = 25 ms + 10 i ms.
```

The current teacher anchors are `a_k = 0, 25, 50, ...` plus the final frame, so their nominal hop is `H=25` frames or 250 ms. For an anchor `a_k`, the cached posterior is

```text
q_k = ECAPA(x[e(a_k) - 1750 ms : e(a_k) + F]),  F = 250 ms.
```

The student target for semantic frame `i` is compared with student output `i+D`; the model also stacks `L` future feature frames. Its latest available feature ends at

```text
e(i) + (D + L) * 10 ms.
```

With `D=21` and `L=4`, this is exactly `e(i)+250 ms`. At a true teacher anchor, teacher and student therefore have the same right edge. This is the valid part of the current contract.

For an interior frame `a_k < i < a_(k+1)`, the code calls [`numpy.interp`](https://numpy.org/doc/stable/reference/generated/numpy.interp.html), which the NumPy 2.5 manual describes as the piecewise-linear interpolant of discrete sample points (documentation current in 2026; accessed 2026-09-26). Thus

```text
q_linear(i) = (1-alpha) q_k + alpha q_(k+1),  0 < alpha < 1.
```

Any nonzero `alpha` makes the target depend on `q_(k+1)`. The latest waveform sample used by that posterior is

```text
e(a_(k+1)) + F
= e(i) + F + (a_(k+1) - i) * 10 ms.
```

For the 24 interior frames this horizon is 490, 480, ..., 260 ms. The current student horizon is 250 ms, so every interior frame fails the information-availability condition. The failure is binary, not proportional to interpolation weight: even a 4% contribution from the next anchor still depends on unavailable audio.

An auditable per-frame invariant is:

```text
latest_teacher_audio_sample(i) <= latest_student_audio_sample(i + D, L).
```

This should be asserted from recorded sample indices, not inferred from names such as `centred`, `prefix`, or `causal`.

## 2. Read-only audit of the current target cache

I compared each current dense probability array with a previous-anchor hold reconstructed from its stored `anchor_frames` and `anchor_probs`. No teacher or student was run and no cache was changed.

Command shape:

```bash
UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf uv run python - <<'PY'
# Load data/generated/targets/switch*.npz; reconstruct the previous-anchor
# hold with np.searchsorted; compare probabilities, argmaxes, and transitions.
PY
```

The three files are `switch_hi_en_train`, `switch_hi_en_eval`, and `switch_en_hi_eval`. Each contains 798 target frames and 33 anchors.

| Cache comparison | Result |
|---|---:|
| Dense frames / stored anchors | 2,394 / 99 |
| Non-anchor frames that consult a later anchor | 2,295 / 2,394 (**95.8647%**) |
| Linear-vs-hold argmax disagreements | 180 / 2,394 (**7.5188%**) |
| Mean `KL(linear || hold)` | 0.12255 |
| Median / p95 `KL(linear || hold)` | 0.00243 / **0.74941** |
| Maximum absolute class-probability difference | **0.84107** |

Argmax agreement on 92.5% of frames does not make the issue negligible because training consumes the complete soft vector. The p95 and maximum differences show a concentrated but large effect near changing teacher trajectories.

Using frame-end timestamps, interpolation also anticipates the intended transition relative to an actually evaluated teacher anchor:

| Clip and intended target | Linear transition | Anchor / previous-hold transition | Artificial advance |
|---|---:|---:|---:|
| held-out Hindi to English: English | 5.385 s | 5.525 s | 140 ms |
| held-out English to Hindi: Hindi | 5.135 s | 5.275 s | 140 ms |
| training Hindi to English: English | 5.305 s | 5.525 s | 220 ms |

These are not good absolute teacher lags: the reference splice is at 4.0 s, and the trajectories also contain incorrect Gujarati, Tamil, Telugu, or Marathi runs. This audit proves only that interpolation can make a cached boundary look 140–220 ms earlier using later audio. It does not prove that hold, dense windows, or the current ECAPA window yields a better model. The sample consists of three synthetic splices, so no population claim is warranted.

## 3. Causally valid alternatives

[`scipy.interpolate.interp1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html) explicitly distinguishes `previous`, `next`, `nearest`, and `linear` expansion modes (SciPy 1.18 documentation; accessed 2026-09-26). Only the choice and its timestamp contract—not the API name—determine causal validity.

| Construction at frame `i` | Valid with current 250 ms evidence? | Target age / extra future | Offline teacher calls | Main trade-off |
|---|---:|---:|---:|---|
| Loss only at exact 250 ms anchors | Yes | 0 | 1.00x | Exact and cheap, but only 33 switch loss points per 8 s clip |
| Previous-anchor hold, 250 ms hop | Yes | 0–240 ms old | 1.00x | Simplest correctness fix; discontinuous and can delay boundaries |
| Nearest anchor, 250 ms hop | No | up to 120 ms future-anchor distance | 1.00x | Smaller age, but still leaks on roughly half the interval |
| Next-anchor hold, 250 ms hop | No | 10–240 ms extra future | 1.00x | Maximum anticipation; invalid at current latency |
| Linear interpolation, 250 ms hop | No | 10–240 ms extra future | 1.00x | Smooth but uses both endpoints and invents an unobserved curve |
| Previous hold, 100 ms hop | Yes | 0–90 ms old | about 2.45x | Moderate generation cost |
| Previous hold, 50 ms hop | Yes | 0–40 ms old | about 4.88x | Recommended compute/resolution candidate |
| Native teacher call every 10 ms | Yes | 0 | about 24.18x | Cleanest target grid; expensive offline generation |
| Current linear target with paid delay | Yes only if `D+L >= 49` frames | 240 ms more evidence | 1.00x | Preserves interpolation but raises model bound from 435 to 675 ms |

The call ratios use the current 798-frame clips including their appended final anchor: 33, 81, 161, and 798 calls per clip for 250, 100, 50, and 10 ms grids. ECAPA target generation is offline, so denser anchors increase only preprocessing time and cache provenance, not deployed student parameters, RTF, or algorithmic latency.

### Why previous hold is valid but not equivalent to dense inference

At a frame 240 ms after anchor `a_k`, previous hold still assigns `q_k`. That posterior's window ends only 10 ms after the later frame, rather than 250 ms after it, and its nominal semantic time is 240 ms old. Hold therefore respects the information boundary by using **less recent** evidence; it does not estimate what ECAPA would have produced at the later frame.

### Why reserving one hop is a poor default

Linear expansion can be made information-valid by delaying every supervised student output until the next anchor is available. The exact worst case is `F + (H-1)*10 ms = 490 ms`, so with `L=4`, `D` must be at least 45 frames rather than 21. The repository's conservative bound then becomes

```text
25 ms analysis + 490 ms evidence + 160 ms chunk = 675 ms.
```

That pays 240 ms of permanent model latency to retain a curve that ECAPA never actually evaluated. It is defensible as a negative control, not as the first production choice.

### Why extrapolation and backward smoothing are not recommended

Linear extrapolation from the two preceding anchors can be causal, but it may leave the probability simplex, amplify teacher jitter, and anticipate a switch without acoustic evidence. A causal EWMA over anchors stays normalized but adds another smoothing lag already present in the routing policy. Neither method has direct LID-KD evidence in the reviewed sources.

## 4. What the literature does and does not support

### Native fine-rate outputs are preferable when available

[González-Domínguez et al., *Frame-by-Frame Language Identification*](https://research.google/pubs/frame-by-frame-language-identification-in-short-utterances-using-deep-neural-networks/) (Neural Networks, 2015; page accessed 2026-09-26) emit a posterior at each newly observed acoustic frame and then study ways to combine those posteriors. This is direct real-time LID evidence for producing predictions on their native time grid, not evidence for interpolating utterance classifiers.

[SAGE-LD](https://arxiv.org/abs/2510.00582) (arXiv v1, **2025-10-01**; accessed 2026-09-26) is offline, non-causal language diarization, but its frame-rate ablation is relevant. Keeping 25 ms features obtains practical DER 21.37/18.03/16.92 on DISPLACE-D/DISPLACE-E/SA Soap Opera, versus 21.97/18.95/17.65 at 105 ms and 21.91/19.09/18.25 at 205 ms. The authors attribute the loss to discarded temporal cues and enlarged receptive field. This supports testing denser targets; it does **not** establish that 10 or 50 ms ECAPA windows will improve this causal student.

### Streaming KD pays for alignment and context

[Delayed-KD](https://arxiv.org/abs/2505.22069) (submitted **2025-05-28**, Interspeech 2025; accessed 2026-09-26) introduces an explicit temporal alignment buffer and reports the associated accuracy/latency trade-off for CTC. It supports treating delay as paid evidence rather than relabelling future-dependent supervision as free. CTC spike alignment is different from smooth LID state, so its gains do not validate elastic per-frame matching here.

[Shim et al.](https://www.isca-archive.org/interspeech_2023/shim23_interspeech.html) (Interspeech 2023; accessed 2026-09-26) add a training-only non-streaming branch specifically so teacher and distilled student features have the same context. [Future-aware Transformer](https://www.isca-archive.org/interspeech_2024/yang24m_interspeech.html) (Interspeech 2024; accessed 2026-09-26) transfers later-chunk information through an inference-time lookahead window. Both are transfer evidence for matching the information set; neither makes inaccessible future audio causally predictable.

The newest cross-domain statement is [Context-Matched Distillation](https://arxiv.org/abs/2608.13391) (video, submitted **2026-08-13**; accessed 2026-09-26), which explicitly replaces full-clip teacher scoring when it depends on frames unavailable to a causal student. This is conceptually aligned but **not speech/LID evidence**.

[Alignment-Path Distillation](https://arxiv.org/abs/2609.20121) (arXiv v1, **2026-09-17**; accessed 2026-09-26) reports that teacher-derived ASR alignment paths can improve error rate while increasing flicker. It reinforces the need to score target stability and boundary timing, not just KD loss. It does not propose sparse-posterior interpolation.

### Evidence gap

I found no spoken-LID or streaming-ASR paper that establishes any of the following:

- linear interpolation of sparse full-window class probabilities is a calibrated estimate of intermediate-frame probabilities;
- a smoother target is preferable to a piecewise-constant target for switch detection;
- reducing anchor hop from 250 to 50 or 10 ms improves Hindi-English student accuracy or lag;
- the current 1.75 s past / 0.25 s future ECAPA window is semantically well aligned to a language boundary.

All four remain **unverified**. METHOD-0 can enforce availability; the separate teacher-window audit is still needed to solve semantic lag.

## 5. Hugging Face Hub audit

The Hub was queried through its model API on **2026-09-26**, including `audio-classification` searches for `language`, `lid`, `langid`, `code-switching`, `streaming`, and `diarization`, plus direct repository inspection. This is a dated, non-exhaustive search—not proof that no unpublished or poorly tagged model exists.

- The current [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) at revision `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9` is Apache-2.0. Its card describes classifying a **speech utterance** and returning one 107-way score vector; it exposes no native frame posterior. Sliding windows are therefore a project-defined temporal wrapper.
- [`tflite-hub/conformer-lang-id`](https://huggingface.co/tflite-hub/conformer-lang-id) at revision `213db93aebf552836020c029a405b355569f4739` is Apache-2.0 and genuinely streaming, but its attentive temporal pooling accumulates prefix history. It is a monolingual-prefix reference, not a local offline teacher or a solution to sparse-target expansion.
- A recent code-switch-labelled artifact, [`yyhenggg/CS-LID-Model`](https://huggingface.co/yyhenggg/CS-LID-Model/tree/dde99d4de24abafeb02cbe15be6e6018a75c42a4), revision `dde99d4de24abafeb02cbe15be6e6018a75c42a4` (last modified **2026-07-22**), mean-pools real Whisper-large-v3 encoder frames and produces utterance-level `P(code-switch)` plus a dominant language among English, Mandarin, Indonesian, and Malay. It has no Hindi, no frame trajectory, no reported metrics in the card, manual-gated weights, and no declared license. It cannot replace the target wrapper.
- The newly uploaded [`shun3232/mms1b-lid-sigmoid-fleurs-csfleurs-csyodas`](https://huggingface.co/shun3232/mms1b-lid-sigmoid-fleurs-csfleurs-csyodas/tree/d66af871161f08265ed586f78cdab301805f3147), revision `d66af871161f08265ed586f78cdab301805f3147` (last modified **2026-09-17**), uses MMS-1B, ECAPA-TDNN, attentive statistics pooling, and a 102-language multilabel head. Its card asserts no evaluation metrics, exposes no local timestamps, and is CC BY-NC 4.0. Code-switch training does not make its output frame-level.
- [`espnet/lid_voxlingua107_mms_ecapa`](https://huggingface.co/espnet/lid_voxlingua107_mms_ecapa/tree/ca4b2dcb318a0cc9f48898e2b7357808e184621c), revision `ca4b2dcb318a0cc9f48898e2b7357808e184621c` (last modified **2026-09-18**), is CC BY 4.0 and reports strong utterance LID across several datasets, but its MMS-1B plus ECAPA attentive-statistics architecture again pools time. It is not a dense boundary teacher.
- [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0) at revision `e66f3d3b9eb0873085418a7b813d3b369bf160bb` is an MIT-licensed example of a checkpoint that natively returns a `(num_frames, num_classes)` matrix. Its classes are speaker/VAD states from a full 10 s chunk, not languages, and it is not causal. It illustrates the interface distinction only.

No reviewed Hub card documents a Hindi-English local LID checkpoint with native dense outputs and a bounded right-context contract. The project must therefore make its window, anchor, expansion, and availability semantics first-class provenance.

## 6. Recommended experiment order

### A. Close the correctness bug before interpreting another run

Promote a small `previous_anchor_hold` implementation, preferably from the existing isolated driver, and record for every dense target frame:

```text
semantic_frame
left_source_anchor
right_source_anchor (null for hold)
latest_teacher_audio_sample
latest_student_audio_sample
availability_valid
```

Fail target generation or loading if any supervised frame violates the sample-level invariant. Required fixtures:

1. current 250 ms linear expansion with `D=21,L=4` fails on all non-anchor frames;
2. exact anchors pass and reproduce stored probabilities bit-for-bit;
3. previous hold passes at every frame;
4. paid linear expansion passes only at `D>=45` with `L=4`;
5. final irregular anchor intervals and padded clip edges are checked explicitly.

Expected accuracy gain: **none claimed**. This is a correctness/provenance fix. The expected observable change is that the three current intended teacher transitions become 140–220 ms later than the invalid linear trace.

### B. Choose an anchor hop with a teacher-only audit

On development switch clips, generate true ECAPA windows at hops `{250,100,50,10}` ms. Treat the 10 ms calls as a dense reference, not ground truth. For each coarser grid compare anchor-only and previous-hold expansion against dense calls using probability KL, argmax disagreement, stable boundary timing, flips, full-teacher retained mass, and known-language accuracy outside a fixed collar.

Proposed shortlist gate (**unverified**): choose the coarsest hop with

- p95 `KL(q_dense || q_expanded) <= 0.10`;
- argmax disagreement `<=2%` outside the boundary collar;
- every intended stable crossing within 50 ms of the dense trajectory;
- no more than one extra raw flip per clip.

The expected compute result is about 4.88x current switch-window calls at 50 ms versus 24.18x at 10 ms, with unchanged deployment latency. The expected accuracy/lag gain is **unverified**.

### C. Train only causally valid arms

With one frozen initialization, batch order, and run identity, compare:

1. 250 ms hop + previous hold;
2. selected denser hop + previous hold;
3. 10 ms native targets if affordable;
4. exact-anchor-only loss as a low-density diagnostic.

Keep the current linear arm only as `invalid_future_diagnostic`; do not place its lag on the deployable Pareto frontier. Optionally include linear plus `D=45,L=4` as a paid-latency negative control with a 675 ms architecture bound.

Report target-only boundary lag first, then student excess lag, deadline recall/misses, frame and clip macro accuracy, flips, and CPU target-generation time. A denser arm should be adopted only if it improves stable switch lag by at least 100 ms or switch recall@1 s by at least 10 points, with monolingual macro accuracy loss at most 2 points and no added inference latency. These gates and gains are **proposals, not predictions**.

## Conclusion

`np.interp` is functioning as documented; the bug is the semantic use of its future endpoint under a tighter evidence budget. Previous-anchor hold closes that bug immediately, but a 250 ms hold exchanges future leakage for up to 240 ms of stale supervision. A 50 ms held grid is the most defensible next candidate because it keeps the current 435 ms deployment bound while sharply reducing target age at modest offline cost. Dense 10 ms teacher calls are the clean reference. None of these changes fixes ECAPA's long 1.75 s past window or the synthetic corpus weakness, so availability, target semantic lag, student error, and routing-policy lag must continue to be reported separately.

