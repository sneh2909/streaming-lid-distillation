# Distilling a non-causal speech teacher into a causal student

Research cutoff: **2026-09-26**. Web sources and Hugging Face Hub model cards were checked on that date.

This review distinguishes three evidence levels:

- **Direct LID evidence:** the experiment is spoken language identification, although generally utterance-level and monolingual.
- **Transfer evidence:** the experiment is streaming ASR or keyword spotting (KWS); the mechanism is relevant, but its benefit for frame-level Hindi–English LID is **unverified**.
- **Project inference:** a proposed adaptation to this repository. Every expected gain stated for it is **unverified until measured locally**.

Interspeech 2026 is scheduled for **2026-09-27 to 2026-10-01**, one day after this cutoff. Its proceedings and DOI pages are already online. Results from those papers are described as forthcoming-conference evidence, not as locally reproduced facts.

## Bottom line

The repository already satisfies the narrow information-availability condition for its switch targets: a teacher target centered at acoustic time `t` uses audio through `t + 250 ms`, and the student prediction trained against it consumes audio through `t + 210 ms + 40 ms`. That equality prevents literal future leakage.

It does **not** establish that the target is the best local label for time `t`. The current teacher window contains **1.75 s of past speech and 0.25 s of future speech**. Its posterior can therefore cross a real language boundary late even before student error, EMA, or dwell is added. A delayed student can imitate that late target perfectly and still be a poor switch detector.

The highest-value sequence is therefore:

1. Measure the teacher trajectory's own boundary lag and flip rate for every target construction.
2. Compare full-utterance, cumulative-prefix, finite rolling-causal, and bounded-lookahead local targets.
3. Sweep only delays that make the teacher's right context physically available to the student.
4. Add bounded confidence/evidence weighting after context and alignment are correct.
5. Consider hidden-feature or temporal-relation KD only if output-posterior KD then plateaus.

Direct LID work supports pairing a long-context teacher view with the **corresponding** short view. Streaming-ASR work strongly supports explicit temporal alignment, but also shows that blindly copying a full-context distribution at the same index can hurt. No reviewed work directly validates any of these recipes for causal Hindi–English code-switch LID, so the proposed order is deliberately conservative.

## 1. The causal contract: availability is necessary, not sufficient

Define a local teacher posterior

```text
q_t^(P,F) = Teacher(x[t-P : t+F]),
```

where `P` is past context and `F` is teacher right context. Let the student target for time `t` be emitted at index `t + D`, while its architecture uses explicit lookahead `L`:

```text
p_(t+D) = Student(x[... : t+D+L]).
```

In common time units, a necessary no-leakage condition is

```text
D + L >= F.
```

For the current configuration, `D=210 ms`, `L=40 ms`, and `F=250 ms`, so the condition holds at equality. Reducing `D` below 210 ms while keeping the same teacher target and lookahead would ask the student to reproduce information that has not arrived. A valid delay sweep must either keep `D >= 210 ms`, reduce teacher right context, or change student lookahead.

Four different problems can otherwise be conflated:

| Problem | Diagnostic | Corrective action |
|---|---|---|
| **Information mismatch** | Teacher target uses audio later than `t+D+L` | Increase paid delay/lookahead or shorten the teacher's future context |
| **Alignment mismatch** | Teacher and student represent the same event at different indices | Validate a fixed shift or bounded alignment buffer |
| **Semantic/window mismatch** | The teacher posterior at `t` is dominated by the previous language or by unrelated future speech | Change the teacher window/prefix construction; delay alone cannot repair it |
| **Decision-policy lag** | Raw student posterior changes promptly, but EMA/dwell changes late | Tune smoothing and dwell separately from model/KD latency |

This order matters. A lower KD loss after adding delay is not evidence of a better switch detector if the target itself changes late.

The monolingual full-utterance target is a special case. Repeating one converged posterior is semantically defensible only under the stationary-language assumption. It knowingly gives the early student a confidence level that a prefix cannot reproduce; the current first-second ramp is a pragmatic mitigation. The same target is invalid on a code-switched clip because no single stationary label exists.

## 2. Direct spoken-LID evidence

### 2.1 Matched long/short views work, but the studies are not streaming

[Liu et al., *Enhancing Language Identification Using Dual-Mode Model with Knowledge Distillation*](https://www.isca-archive.org/odyssey_2022/liu22c_odyssey.html) (Odyssey 2022, DOI published 2022; accessed 2026-09-26) jointly train an XSA-LID model on a full utterance and a corresponding masked 3 s clip. The two modes share weights, use temperature 2, and add KD between their distributions. On NIST LRE17 MLS14 narrowband speech, the random-location 3 s dual-mode model improves `Cavg` over the single-mode XSA baseline as follows:

| Test duration | XSA-LID `Cavg` | Dual-mode + KD `Cavg` | Relative reduction |
|---:|---:|---:|---:|
| 3 s | 0.1685 | 0.1361 | 19.23% |
| 10 s | 0.0739 | 0.0580 | 21.52% |
| 30 s | 0.0406 | 0.0372 | 8.37% |

This is the closest direct support for paired full/short LID views. It also limits the conclusion: the output is one label per monolingual trial, not a causal frame trajectory, and the long view can safely supervise the short view because both contain the same language. Across a switch, a long window can contain a different semantic state from its short sub-window.

[Shen et al., *Feature Representation of Short Utterances Based on Knowledge Distillation for Spoken Language Identification*](https://www.isca-archive.org/interspeech_2018/shen18b_interspeech.html) (Interspeech 2018, 2018-09; accessed 2026-09-26) give a 4 s teacher the corresponding longer segment and a student its contained 0.5–2 s segment. Output KD and L1 hidden-representation KD are combined. On their 10-language task, utterance error at 2 s falls from 6.87% to 4.70%; at 0.5 s it falls from 24.01% to 21.57%. The relative benefit shrinks as evidence becomes extremely short. Again, this is utterance LID with hard-label supervision as well as KD, not code-switch or label-only distillation.

**Transfer to this project:** generate teacher and student views from the same acoustic neighborhood and keep their right edge explicit. Do not infer that a full-utterance target is safe at a language boundary.

### 2.2 Confidence weighting has LID evidence, but confidence is not correctness

[Dey, Mondal, and Kurmi, *Teacher-Free Knowledge Distillation for Improving Short-Utterance Spoken Language Identification*](https://www.isca-archive.org/interspeech_2025/dey25_interspeech.html) (Interspeech 2025, 2025-08; accessed 2026-09-26) find that 36.94% of misclassified 2 s English segments in their analysis contain one of four out-of-scope factors: non-speech, named entities, filler words, or overlap. Their best method retains correctly classified segment predictions, updates soft labels conditionally, and weights them by inverse entropy. On a VoxLingua-derived ten-language set containing English, Hindi, Bengali, and Urdu, 2 s `Cavg` improves from 11.20 to 8.24; in VoxLingua-to-Common-Voice transfer it improves from 20.40 to 16.22.

Important limitations are:

- The method uses ground-truth correctness to select soft labels and mixes hard-label loss with KD. This repository intentionally does not use human/TTS language labels in optimization, so the method cannot be copied literally.
- `1 / entropy` is unbounded as entropy approaches zero and assumes confidence is calibrated. A teacher may be confidently wrong on accented English, unsupported languages, noise, or mixed windows.
- Down-weighting high-entropy boundary frames may remove precisely the supervision needed to learn a switch.

A safer label-free adaptation is a bounded ablation, not an assumed improvement. With the original unsoftened seven-class posterior `q` and the pre-renormalization probability mass `s` assigned to the seven supported languages, compare:

```text
u(q) = 1 - H(q) / log(7)

w_entropy = clip(u(q), 0.10, 1.00)
w_mass    = clip(s,    0.10, 1.00)
w_both    = w_entropy * w_mass
```

Apply the existing validity mask and early-evidence ramp separately. Compute confidence from the **temperature-1** teacher posterior; computing it after the target is softened at `T=2` would manufacture extra uncertainty. The exact formulas above are a project proposal and are **unverified**.

Confidence weighting does not solve future-information mismatch. A full-context teacher can be highly confident only because it heard speech the student has not yet received.

## 3. Temporal alignment and delayed targets

### 3.1 A fixed shift is a baseline; a validated buffer is the next step

[Li et al., *Delayed-KD: Delayed Knowledge Distillation based CTC for Low-Latency Streaming ASR*](https://www.isca-archive.org/interspeech_2025/li25u_interspeech.html) (Interspeech 2025, 2025-08; accessed 2026-09-26) explicitly address the fact that a streaming CTC student's spikes lag a non-streaming teacher's. Their Temporal Alignment Buffer (TAB) tests student delays in `D={0,...,d}` and takes the minimum KL at each teacher time. At a fixed 40 ms decoding chunk on AISHELL-1, rescoring CER changes from 5.77 with a 0 ms TAB to 5.65/5.42/5.47/5.48 with 40/80/120/160 ms buffers. First- and last-token delays rise as the buffer grows. The best 80 ms setting reaches 5.42 CER, versus 5.98 for U2++ at the same 40 ms chunk and 5.44 for U2++ at 320 ms.

The current repository loss is a single-point version of this idea: `q_t` is compared with `p_(t+D)` for one fixed `D`. Delayed-KD supports testing rather than assuming that shift.

The exact TAB objective should **not** be imported blindly. CTC has sparse token spikes whose locations genuinely differ. LID posteriors should usually evolve smoothly. Selecting a different delay independently at every LID frame can hide boundary error, create an unrealistically elastic target, and lower KD loss without lowering detection latency. If a buffer is tried here, first compare:

1. a fixed global delay;
2. one delay selected per complete training clip by mean KL; and only then
3. a soft minimum over delays.

Never report only the minimized KD loss. Report the selected-delay distribution, raw switch lag, flip rate, short-span misses, and paid inference latency. This LID adaptation is **unverified**.

### 3.2 Streaming and offline outputs can have different alignments even on the same speech

[Kurata and Saon, *Knowledge Distillation from Offline to Streaming RNN Transducer*](https://www.isca-archive.org/interspeech_2020/kurata20_interspeech.html) (Interspeech 2020, 2020-10; accessed 2026-09-26) show that independently trained offline and streaming RNN-T posterior lattices differ substantially because the streaming recognizer emits later. They first align the teacher behavior before distillation. This is ASR-specific, but the general warning transfers: equal array indices do not guarantee equal semantic time.

[Yu et al., *Dual-mode ASR*](https://arxiv.org/abs/2010.06030) (ICLR 2021; arXiv v2 2021-01-27; accessed 2026-09-26) jointly run a shared-weight model in full-context and streaming modes, stop gradient through the full-context prediction, and apply in-place KD. They explicitly sweep teacher shifts from -2 to +2 frames. In the LibriSpeech test-other ContextNet ablation, shared-weight joint training with KD obtains 8.5 WER and 40/160 ms median/p90 latency; removing KD gives 10.2 WER and 120/310 ms. Separate weights with joint training and KD give 9.9 WER and 50/210 ms. The evidence favors coupled modes and validated timing, but the common architecture, shared tokenizer, and RNN-T alignment make this easier than distilling a frozen ECAPA teacher into a tiny TCN.

[Yoon, *Heuristic-free Knowledge Distillation for Streaming ASR via Multi-modal Training*](https://ojs.aaai.org/index.php/AAAI/article/view/34765) (AAAI 2025, published 2025-04-11; accessed 2026-09-26) is an explicit criticism of choosing an offline-to-streaming shift `tau` heuristically. It avoids the shift by self-distilling within one streaming backbone and giving the training-only teacher full transcript context. LID has no corresponding transcript input, so this is not a directly usable recipe; it is further evidence that `D` must be treated as an empirical latency/accuracy parameter rather than a universal constant.

### 3.3 The newest alignment work also exposes a stability trade-off

[Jia et al., *Alignment-Path Distillation from Non-streaming ASR-LLMs for Streaming Speech Recognition*](https://arxiv.org/abs/2609.20121) (arXiv v1, submitted **2026-09-17**; accessed 2026-09-26) extracts a global monotonic path from a non-streaming teacher's soft text–audio attention and falls back to a forced aligner when teacher attention confidence is low. Teacher-derived paths improve error rate, but the paper reports similar mean emission latency and **higher flicker** than forced alignments; the complete path/logit/hidden-state recipe gives a 16.6% relative error reduction over its undistilled forced-alignment baseline.

This is a very recent, non-peer-reviewed preprint and depends on text tokens and attention paths. Its transferable lesson is narrower: use globally coherent or segment-coherent alignment, confidence-gated fallback, and an explicit stability metric. Independent per-frame delay choices are risky for a language-state trajectory.

## 4. Prefix teachers, lookahead KD, and multi-context training

### 4.1 “Prefix teacher” has three materially different meanings

| Teacher view for target time `t` | Future context | Switch behavior | Recommended role |
|---|---:|---|---|
| Full utterance `x[0:T]` | Unbounded | Invalid local target when a switch exists | Low-variance monolingual control only |
| Cumulative prefix `x[0:t]` | 0 | Can become sticky because all old-language speech remains | Required backlog control, not the preferred switch target |
| Finite rolling prefix `x[max(0,t-P):t]` | 0 | Can forget the old language; responsiveness depends on `P` | Best causal-teacher candidate |
| Bounded local window `x[t-P:t+F]` | `F` | Can use modest future evidence if latency pays for it | Accuracy/latency challenger |

The direct LID studies above justify matched long and short views, but they do not establish that a cumulative prefix is good for code-switching. The deployed student has about 1.27 s of finite past receptive field. A teacher that pools from utterance start can encode evidence the student has already forgotten and can recreate the all-history stickiness found in streaming utterance-LID models. A finite rolling prefix is the closer information set.

For early windows, do not pad missing speech and pretend the target is reliable. Keep the evidence ramp or mask until a declared amount of voiced audio is present, then separately report time to first decision.

### 4.2 Dynamic chunk training is a context-distribution method, not KD

[Zhang et al., *Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition (U2)*](https://arxiv.org/abs/2012.05481) (arXiv v2, 2021-12-29; accessed 2026-09-26) train one encoder with dynamic chunk masks so it supports full context and multiple streaming chunk sizes. In the published recipe, batches sample full context or chunked context, with chunk sizes varied during training. This reduces train/inference context mismatch and makes latency selectable.

For this repository, changing only the external inference chunk size is not analogous: the finite convolutional student is already tested so chunked and whole-sequence logits are identical. The useful analogue is to vary the **available feature lookahead or teacher target horizon** during training, not to expect a different answer from mechanically re-chunking the same receptive field.

### 4.3 Lookahead KD can transfer only through context that inference really provides

[Yang et al., *Learning from Back Chunks: Acquiring More Future Knowledge for Streaming ASR Models via Self Distillation*](https://www.isca-archive.org/interspeech_2024/yang24m_interspeech.html) (Interspeech 2024, 2024-09; accessed 2026-09-26) use a Future-aware Transformer: representations in an earlier chunk learn from the same frames when those frames appear in the next chunk with more context. Prediction-KL or feature-MSE self-distillation is carried through an inference-time lookahead window. On AISHELL-1 at their 12-frame chunk setting, rescoring CER moves from 6.4 for the baseline to 6.1 with lookahead and 5.9 with future-aware distillation.

The important constraint is that the bridge exists at inference. This method does not make inaccessible future speech causally predictable. The current `L=40 ms` plus paid target delay is already a simpler, auditable bridge. Increasing it should be justified by a measured Pareto improvement, not by describing future information as free.

### 4.4 New 2026 mode-consistency evidence favors symmetric/coupled training

[Andrusenko et al., *Reducing the Offline-Streaming Gap for Unified ASR Transducer with Consistency Regularization*](https://www.isca-archive.org/interspeech_2026/andrusenko26_interspeech.html) (Interspeech 2026 proceedings online; conference begins 2026-09-27; accessed 2026-09-26) train shared offline and streaming FastConformer-RNNT modes and regularize their full joint distributions. On the 128M model ablation, using the offline mode as the one-way KL teacher is poor at the tight 0.32 s setting (15.86 average WER); a streaming teacher gives 8.56 and symmetric consistency gives 8.34. Their full unified system with symmetric mode consistency improves the streaming/offline Pareto frontier.

The associated Hugging Face checkpoint, [`nvidia/parakeet-unified-en-0.6b`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b), was released 2026-04-07. Its card documents 600M parameters, shared offline/streaming weights, dynamic chunked convolution, latency settings from 160 ms upward, and the NVIDIA Open Model License. It is English ASR rather than LID and is far outside this CPU budget. More importantly, the card says only inference is currently supported and that the unified training pipeline will be released later, so exact external reproduction is **unverified as of the cutoff**.

The project-level implication is not to adopt Parakeet. It is that a strong offline distribution should not automatically dominate a severely context-limited mode. If multi-context self-consistency is tried later, a symmetric loss or a context-aware teacher choice deserves comparison with one-way full-context KD.

## 5. Feature KD and causal temporal relations

### 5.1 Point-wise hidden-feature matching is particularly fragile

[Shim et al., *Knowledge Distillation from Non-streaming to Streaming ASR Encoder Using Auxiliary Non-streaming Layer*](https://www.isca-archive.org/interspeech_2023/shim23_interspeech.html) (Interspeech 2023, 2023-08; accessed 2026-09-26) argue that direct teacher/student frame features have different context and alignment. They attach a training-only non-streaming auxiliary layer to the student, so teacher and auxiliary features have matched context, and add future-prediction training. On LibriSpeech test-other, token-probability KD is worse than the 12.8 WER scratch baseline (13.3); their auxiliary approach reaches 10.7. This is transfer evidence against naïve same-index feature regression.

Adding such branches to the 42,567-parameter TCN would substantially complicate the take-home system, and the current ECAPA teacher does not expose an already aligned frame sequence. Output-level targets with explicit windows should be exhausted first.

### 5.2 Causal temporal relation KD is a promising second-line method

[Zhang, Zhu, and Qin, *Mitigating Causality Mismatch with Causal Temporal Relation Distillation for Streaming Keyword Spotting*](https://www.isca-archive.org/interspeech_2026/zhang26ia_interspeech.html) (Interspeech 2026 proceedings online; conference begins 2026-09-27; accessed 2026-09-26) is unusually close to this project's deployment shape. A non-causal AST teacher supervises a strictly causal 150K-parameter 1D-CNN student. Rather than regress absolute teacher features, the method forms within-utterance temporal similarity matrices and uses a lower-triangular mask as a realizable historical anchor. A training-only bidirectional relation term is added more weakly.

On Google Speech Commands V2, the causal student's accuracy rises from 95.12% without KD to 95.74% with logit KD, 96.53% with causal temporal-relation KD, and 96.91% with the bidirectional auxiliary relation. In a constructed continuous stream, false rejection at 1 false alarm/hour falls from 8.52% to 4.15%. The authors explicitly limit the conclusion to GSC v2, AST-to-1D-CNN, and zero lookahead.

This method adds no inference cost, but a LID adaptation remains **unverified**. It would require a meaningful temporal feature sequence from the windowed ECAPA teacher or an experiment using posterior-relation matrices. Seven-dimensional posterior relations may be too impoverished; ECAPA embedding trajectories are more plausible but cost more to cache. Treat this as a later ablation, not a replacement for correct target windows.

## 6. Emformer is an architecture reference, not a KD recipe

[Shi et al., *Emformer*](https://arxiv.org/abs/2010.10759) (arXiv v4 2020-12-30; ICASSP 2021; accessed 2026-09-26) use center blocks, explicit right context, cached left context, and a compressed memory bank. In its low-latency LibriSpeech setting, center context is 80 ms, right context is 40 ms, and left context is 1,280 ms; the 120M model reports 3.01/7.09 WER on test-clean/test-other at 80 ms average encoder-induced latency.

Two clarifications prevent common misapplication:

- “Distilled into memory” in the Emformer abstract means summarizing history, not teacher–student knowledge distillation.
- The paper warns that attention masks alone can leak right context across stacked layers, causing effective lookahead to grow with depth. It duplicates each block's right-context inputs during parallel training to prevent that leak.

The current convolutional student and causality/chunk-equivalence tests are simpler and more auditable. Replacing it with Emformer is unjustified unless a longer-memory requirement appears. The transferable requirement is to test effective lookahead end-to-end, not just inspect one layer's mask.

## 7. Recommended experiment for this repository

### Phase A — teacher-only target audit

Cache teacher trajectories at the existing 250 ms teacher hop for these views:

1. `full_utterance`: existing monolingual control; never use it on switch clips.
2. `prefix_all`: `x[0:t]`, the causal-prefix backlog control.
3. `rolling_0p5`, `rolling_1p0`, `rolling_2p0`: finite causal windows ending at `t`.
4. `local_0p75_0p25`: 750 ms past plus 250 ms future, a more responsive bounded-lookahead challenger.
5. `local_1p75_0p25`: the existing target.

Before training a student, report for every switch clip:

- teacher first crossing of the new language;
- first crossing that remains correct for 250 and 500 ms;
- teacher flip rate and missed 0.5/1/2 s embedded spans;
- frame accuracy outside a declared boundary collar;
- retained seven-language mass before renormalization;
- entropy by time from the boundary.

This isolates the best possible student that perfectly copies each target. If a target's stable crossing is already 1.3 s late, no loss alignment can produce a 0.75 s detector while continuing to imitate it.

### Phase B — context-valid delay grid

Keep `L=4` feature frames and train the same student/seed/steps. For causal teacher targets (`F=0`), use `D={0,10,21}` frames. For 250 ms-right-context targets, use `D={21,25,32}`. These are all causally valid under `D+L>=F`.

With the repository's conservative latency accounting, those settings imply:

| Delay `D` | Evidence delay `D+L` | Total model latency (`25 ms` frame + evidence + `160 ms` chunk) |
|---:|---:|---:|
| 0 | 40 ms | 225 ms |
| 10 | 140 ms | 325 ms |
| 21 | 250 ms | 435 ms |
| 25 | 290 ms | 475 ms |
| 32 | 360 ms | 545 ms |

Do not compare models only at their target index. Report wall-clock decision time from the physical switch, so paid delay cannot masquerade as better alignment. Use the speaker-disjoint split requested in the backlog and keep policy smoothing fixed while comparing models.

Primary metrics should be known-label macro accuracy, student/teacher agreement, median and p95 raw switch lag, post-policy switch lag, flips/minute, and recall for 0.5/1/2 s embedded spans. Select from the validation split, then report the locked choice once on test.

### Phase C — bounded reliability weighting

On the best target/delay pair, compare exactly four weights:

1. existing early ramp only;
2. ramp × bounded normalized confidence;
3. ramp × retained seven-language mass;
4. ramp × both.

Keep the KD target and temperature unchanged. Cache the unsoftened posterior and selected-class mass separately. Report metrics by teacher-confidence decile and around switch boundaries. Reject weighting if it obtains a cleaner KD curve by ignoring hard boundary frames while increasing switch lag or short-span misses.

### Phase D — optional methods only after the simple grid

- Test a clip-global delayed buffer over the best three valid delays; do not begin with per-frame hard minima.
- If posterior KD saturates, add causal temporal-relation KD with a small loss-weight sweep and no inference branch.
- If multiple deployment latency modes are genuinely required, train with randomly masked student lookahead or a shared multi-context mode. Merely varying the external chunk size will not alter this finite TCN's exact outputs.

## 8. Hugging Face Hub audit

The Hub was searched on 2026-09-26 for dual-mode LID, delayed KD, streaming-ASR distillation, Emformer, and dynamic-chunk models. The negative result below is a **non-exhaustive search result**, not proof that no upload exists.

| Hub ID | What it verifies | Why it is not a drop-in method/checkpoint |
|---|---|---|
| [`nvidia/parakeet-unified-en-0.6b`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b) | A released shared offline/streaming model with mode-consistency training; 600M; NVIDIA Open Model License; released 2026-04-07 | English RNNT ASR, GPU-oriented, no LID head, and training pipeline not yet released on the card |
| [`speechbrain/asr-streaming-conformer-librispeech`](https://huggingface.co/speechbrain/asr-streaming-conformer-librispeech) | Apache-2.0 runnable Dynamic Chunk Training reference; its card reports full-context and 320–1,280 ms chunk WER | English ASR; card does not claim offline-teacher KD or LID |
| [`nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms`](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms) | Public cache-aware bounded-lookahead FastConformer reference, about 32M parameters, CC BY 4.0 | English ASR; not a distillation checkpoint and still much larger than the student |
| [`anton-l/emformer-base-librispeech`](https://huggingface.co/anton-l/emformer-base-librispeech) | Apache-2.0 community Emformer artifact compatible with the Transformers architecture | Sparse/empty model documentation and no LID/KD evidence; provenance is weaker than the paper |
| [`tflite-hub/conformer-lang-id`](https://huggingface.co/tflite-hub/conformer-lang-id) | Apache-2.0 recurrent streaming LID artifact covering the seven project languages | It was not trained with the reviewed KD methods and pools the whole prefix, so code-switch recovery remains unverified |

No reviewed Hub card explicitly documented a released **LID** checkpoint trained with Delayed-KD, Future-aware Transformer distillation, causal temporal-relation KD, or the Odyssey dual-mode XSA-LID recipe. Paper results must not be assigned to similarly named community checkpoints.

## 9. What is still unverified

- No cited study evaluates these distillation methods on Hindi–English frame-level code-switch LID.
- Direct LID studies use utterance labels and generally include hard-label loss; their gains do not predict this label-free KD setting.
- ASR delay buffers solve token-spike alignment, which is not identical to a slowly varying language posterior.
- The proposed finite rolling-prefix windows, delay grid, bounded confidence weights, and relation loss have no local result yet.
- The current teacher's raw switch-boundary lag has not been separated from student and policy lag in the published repository results.
- The two Interspeech 2026 papers are in online proceedings, but their conference presentation is still forthcoming at this cutoff.
- The 2026 alignment-path work is an arXiv v1 preprint and has not been peer reviewed.

## Source record

Primary sources used, with publication/version dates and all accessed **2026-09-26**:

- [Liu et al., dual-mode XSA-LID](https://www.isca-archive.org/odyssey_2022/liu22c_odyssey.html) — Odyssey 2022; full/short paired LID modes and duration-binned results.
- [Shen et al., short-utterance LID representation KD](https://www.isca-archive.org/interspeech_2018/shen18b_interspeech.html) — Interspeech 2018; matched 4 s teacher and 0.5–2 s student views.
- [Dey et al., teacher-free short-utterance LID KD](https://www.isca-archive.org/interspeech_2025/dey25_interspeech.html) — Interspeech 2025; entropy weighting, OOS analysis, same/cross-corpus results.
- [Li et al., Delayed-KD](https://www.isca-archive.org/interspeech_2025/li25u_interspeech.html) — Interspeech 2025; Temporal Alignment Buffer, CER and emission-delay trade-offs.
- [Yu et al., Dual-mode ASR](https://arxiv.org/abs/2010.06030) — ICLR 2021, arXiv v2 **2021-01-27**; shared-mode in-place KD and explicit shift sweep.
- [Kurata and Saon, offline-to-streaming RNN-T KD](https://www.isca-archive.org/interspeech_2020/kurata20_interspeech.html) — Interspeech 2020; posterior-lattice alignment mismatch.
- [Yoon, heuristic-free streaming-ASR KD](https://ojs.aaai.org/index.php/AAAI/article/view/34765) — AAAI 2025, published **2025-04-11**; critique of heuristic teacher shifts.
- [Zhang et al., U2 dynamic chunk training](https://arxiv.org/abs/2012.05481) — arXiv v2 **2021-12-29**; variable-context streaming/non-streaming training.
- [Yang et al., Future-aware Transformer](https://www.isca-archive.org/interspeech_2024/yang24m_interspeech.html) — Interspeech 2024; lookahead-mediated self-distillation from later chunks.
- [Shim et al., auxiliary non-streaming student layers](https://www.isca-archive.org/interspeech_2023/shim23_interspeech.html) — Interspeech 2023; context-matched feature KD.
- [Shi et al., Emformer](https://arxiv.org/abs/2010.10759) — arXiv v4 **2020-12-30**, ICASSP 2021; bounded block context, memory, and lookahead-leak prevention.
- [Andrusenko et al., mode-consistency RNNT](https://www.isca-archive.org/interspeech_2026/andrusenko26_interspeech.html) — Interspeech 2026 online proceedings; conference **2026-09-27 to 2026-10-01**. [Associated HF checkpoint](https://huggingface.co/nvidia/parakeet-unified-en-0.6b), released **2026-04-07**.
- [Zhang et al., causal temporal-relation distillation](https://www.isca-archive.org/interspeech_2026/zhang26ia_interspeech.html) — Interspeech 2026 online proceedings; conference **2026-09-27 to 2026-10-01**.
- [Jia et al., alignment-path distillation](https://arxiv.org/abs/2609.20121) — arXiv v1 submitted **2026-09-17**; preprint, peer-review status unverified.
