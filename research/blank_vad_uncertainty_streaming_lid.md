# Blank, VAD, and uncertainty for streaming LID (METHOD-11)

Research cutoff: **2026-09-26**. Web sources, primary papers/code, Hugging Face model cards/API metadata, and the repository's current audio/target artifacts were checked on that date. Expected downstream gains are marked **unverified**.

## Bottom line

Do **not** make one eighth class mean all of silence, unsupported language, and low confidence. They require different actions:

- **non-speech / blank:** suppress language evidence and do not advance a language dwell counter;
- **other language:** speech is present, but it is outside the seven supported languages;
- **uncertain:** abstain because the available evidence is weak, without asserting either of the above.

The smallest defensible first experiment is therefore an external **causal VAD mask plus a silence-frozen routing policy**, not a new language class. Keep the seven-way teacher posterior normalized conditional on speech, multiply its KL weight by a causal speech mask/probability, and leave wall-clock time intact. If that helps, a factorized speech head costs only **33 parameters** (`Linear(32,1)`), taking the student from 42,567 to 42,600 parameters. It should sit beside—not compete in one softmax with—the seven conditional language logits.

The strongest new downloadable candidate is the causal checkpoint inside [`FireRedTeam/FireRedVAD`](https://huggingface.co/FireRedTeam/FireRedVAD) (official Hub revision `7990aaccc6b7aec1e527743bd30201f2c4a03b8c`, Apache-2.0). Its streaming model is a 567,937-parameter, zero-lookahead DFSMN. However, the published 97.57% F1 is explicitly attached to the **non-streaming** system on an unreleased aggregate benchmark; short Hindi/English, native 8 kHz, causal-checkpoint accuracy, and CPU cost remain **unverified**. Pin and compare it with Silero VAD rather than adopting the headline number.

## 1. Keep four states separate

| State | Acoustic fact | Training treatment | Router treatment |
|---|---|---|---|
| `SPEECH_IN_SET` | Speech from one of the seven supported languages | Seven-way conditional KD | Update language posterior and dwell |
| `NON_SPEECH` / blank | Silence, background noise, or no usable voice | Mask language KD; optionally train a separate speech gate | Freeze language EMA/dwell; do not declare a language switch |
| `OTHER_LANGUAGE` | Speech, but not one of the seven | Separate in-set/OOS supervision | Route to multilingual fallback / `OTHER_LANGUAGE` |
| `UNCERTAIN` | Insufficient or conflicting evidence | Down-weight or abstain; it is not a ground-truth class by itself | Stay `UNKNOWN` before commit or retain the active route temporarily |

This distinction matters mathematically. If `s_i` is a speech probability and `r_i(c)` is a normalized seven-way language distribution conditional on speech, the factorized joint output is

```text
P_i(blank) = 1 - s_i
P_i(language=c) = s_i * r_i(c).
```

It sums to one without asking the language softmax to learn silence as a language. A future OOS gate `g_i` can remain separate:

```text
P_i(blank) = 1 - s_i
P_i(other_language) = s_i * (1 - g_i)
P_i(language=c) = s_i * g_i * r_i(c).
```

Teacher entropy or low retained seven-language mass must **not** become a blank label: an uncertain frame may contain real in-set speech, and low retained mass may mean an unsupported language rather than silence.

The “blank” in a CTC router also needs care. [SC-MoE](https://www.isca-archive.org/interspeech_2024/ye24_interspeech.html) (Interspeech 2024) gives its streaming Mandarin/English router a CTC blank expert, which usefully prevents every encoder frame from voting for a language. But CTC blank is also an alignment state during speech; it is not proof that every blank is acoustic silence. [SAGE-LD](https://arxiv.org/abs/2510.00582) (submitted 2025-10-01) instead uses a distinct VAD query beside language queries. Both support an explicit no-language-evidence path; neither validates one class that merges silence, OOS speech, and uncertainty for Hindi-English.

## 2. Current repository audit

### 2.1 The current objective forces a language on every valid frame

The student has seven outputs and no speech gate. `delayed_distillation_loss` applies the seven-way KL to every causally valid frame, apart from padding and the first-second ramp. The ECAPA teacher itself has no non-speech output, and restricting/renormalizing it to seven languages guarantees a language distribution even when the semantic frame contains digital silence.

The log-mel frontend is suitable for a later blank head: it is causal (`center=False`) and does not apply whole-utterance normalization. Exact zero audio becomes a stable floor-valued feature rather than being normalized into clip-dependent evidence.

### 2.2 Fixed four-second splices create labelled silence

`prepare_data.py` truncates or right-pads each switch side to exactly four seconds, then assigns the whole interval a language. A read-only audit of the PCM files found these exact-zero runs:

| Clip | Exact-zero run at/around nominal 4.000 s join | Exact-zero tail |
|---|---:|---:|
| `switch_hi_en_train` | 4.0000–4.0233 s (23.3 ms, on the second side) | none material |
| `switch_hi_en_eval` | 3.8423–4.0000 s (157.7 ms) | 7.0966–8.0000 s (903.4 ms) |
| `switch_en_hi_eval` | 3.0966–4.0000 s (903.4 ms) | 7.8423–8.0000 s (157.7 ms) |

These are lower bounds on non-speech: the synthesis code deliberately retains an 80 ms low-amplitude margin, and exact PCM zero is not a VAD. In particular, the reverse-direction evaluation clip has no target-language speech at the nominal 4.000 s join until after a long padded gap. Switch lag must be measured from annotated target-speech onset, not the concatenation timestamp.

### 2.3 Frame audit

For this diagnostic only, an “energy proxy” frame means all samples in the 25 ms analysis frame have `max(abs(PCM)) <= 0.003`, the same amplitude threshold used when trimming the TTS files. This is **not** ground-truth VAD.

| Split | Clips | Causally valid frames | Exact-zero frames | Energy-proxy frames | Ramp-weighted exact-zero share | Ramp-weighted proxy share |
|---|---:|---:|---:|---:|---:|---:|
| train | 71 | 29,969 | 64 (0.21%) | 2,098 (7.00%) | 0.004% | 6.06% |
| held-out monolingual | 21 | 7,330 | 0 | 521 (7.11%) | 0% | 5.83% |
| held-out switches | 2 | 1,546 | 164 (10.61%) | 266 (17.21%) | 11.33% | 17.43% |

Thus the two switch tests contain a large amount of labelled non-speech that the training switch scarcely contains as exact silence. This does not explain the student's overall 18.51% held-out accuracy, but it invalidates silence-region language metrics and can distort route stability around the only two switch trials.

The existing dense teacher targets are particularly misleading on those frames. Across the **164 causally valid exact-zero frames** in the two evaluation switches, the seven-way conditional teacher maximum averages **0.98276**, and **154/164 (93.90%)** are at least 0.9. This is expected from a teacher window dominated by past language context plus seven-way renormalization; it is not evidence that silence has a language.

Teacher-window padding adds a related edge issue. Across all 99 switch anchors, 13 windows contain at least 50% exact-zero samples; their mean retained seven-language mass is 0.2901, yet the mean conditional maximum is still 0.6666. The 51 anchors with under 10% exact zeros have means 0.5273 and 0.9352. This small descriptive audit mixes clip-edge padding and real pauses and is **not** a calibrated VAD study, but it shows why language confidence alone cannot define speech activity.

## 3. What published evidence supports

### Short-utterance LID benefits from removing non-speech, but VAD does not solve every ambiguity

[Dey, Mondal, and Kurmi](https://www.isca-archive.org/interspeech_2025/dey25_interspeech.html) (Interspeech 2025) report that 689 of 1,865 misclassified two-second English trials—36.94%—contained at least one analyzed OOS factor among non-speech, named entities, fillers, and overlapping speech. Their pipeline applies VAD before log-mel extraction. This supports excluding non-speech from language supervision, but the 36.94% is a union of four factors on one English/VoxLingua setup; it is not an expected gain for this Hindi-English student.

The DISPLACE 2024 language-diarization system of [Kalda et al.](https://www.isca-archive.org/interspeech_2024/kalda24_interspeech.html) (Interspeech 2024) first applies Silero VAD, then segments detected speech for language embedding/clustering. That establishes a conventional separation between speech detection and language identity. It does not establish online boundary latency because their downstream five-second overlapping windows are offline and much longer than this project's budget.

SC-MoE and SAGE-LD provide architecture-transfer evidence for a no-language-evidence path, not direct Hindi-English results. SC-MoE reports code-switch ASR mixed error rate, not VAD/LID boundary accuracy. SAGE-LD is an offline language diarizer and reports no causal timing contract. Therefore the gain of a blank mask/head here remains **unverified**.

## 4. Hugging Face and release audit

The Hub/API audit is dated 2026-09-26. “Coverage” means evidence relevant to this project, not merely that a model can accept arbitrary audio.

| Candidate | Streaming contract and size | Indic / 8 kHz evidence | Cost and license | Decision |
|---|---|---|---|---|
| [`FireRedTeam/FireRedVAD`](https://huggingface.co/FireRedTeam/FireRedVAD/tree/7990aaccc6b7aec1e527743bd30201f2c4a03b8c) | Official checkpoint; Hub created 2026-02-11, modified 2026-03-13. Streaming file SHA-256/LFS OID `ec88a8ae…fb59c9`, 2,283,513 bytes. Causal 8-block DFSMN, 567,937 parameters, 25 ms analysis / 10 ms shift, zero lookahead, fixed caches. Raw posterior plus trailing five-frame mean and duration state machine in the reference code. | The paper's FLEURS-VAD-102 covers 102 languages, so it should include all seven FLEURS configs. But the card explicitly assigns 97.57% F1 to **non-streaming VAD**, gives no language slices, and still says annotations are “coming soon.” Input is 16 kHz only; 8 kHz is unverified. | CPU RTF/RSS are unreported. Apache-2.0. | **First 16 kHz causal challenger**, but require local parity, CPU, Hindi/English, and 8 kHz tests. Do not inherit the offline score. |
| [`tphakala/silero-vad`](https://huggingface.co/tphakala/silero-vad/tree/394d7e6b8c193baf42e965c97b13894e9de4afd8) | Audited community mirror created/updated 2026-08-24 of upstream v6.2.1. Stateful 32 ms windows, LSTM state `[2,1,128]`; 2,327,524-byte ONNX supports 8/16 kHz. Upstream latest is [v6.2.3, released 2026-09-23](https://github.com/snakers4/silero-vad/releases/tag/v6.2.3); the three relevant model blobs are unchanged from v6.2.1, while v6.2.3 changes packaging/dependencies. | Upstream claims training spanning 6,000+ languages and native 8/16 kHz, but publishes no training data/paper and no Hindi/English slice. Its public quality pages mix chunk- and utterance-level protocols. | Upstream reports under 1 ms per 30+ ms chunk on one CPU thread; local cost is unverified. MIT. | **First 8 kHz/portable challenger**; pin weight hashes and validate rather than trusting broad-language claims. |
| [`nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0`](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0/tree/9d25eb877d9edab5101ccc843edd04787a6c72c4) | Official Hub model created 2025-05-08, modified 2025-05-14. 91.5K-parameter MarbleNet CNN, 20 ms posterior frames at 16 kHz. The card does not document a causal state/lookahead contract. | Training support listed only for Chinese, English, French, German, Russian, and Spanish—no Indian language. Its aggregate test sets do not verify Hindi. No 8 kHz result. | Card emphasizes NVIDIA GPU systems and gives no CPU timing. NVIDIA Open Model License. | Lightweight offline/reference candidate only until causality and Indic transfer are proven. |
| [`TEN-framework/ten-vad`](https://huggingface.co/TEN-framework/ten-vad/tree/bda8ffc78b1846c5c7cbd38f04e52deff49de707) | Hub created 2025-05-14, modified 2026-03-13. Real-time 16 kHz API with 10/16 ms optimized hops; repository reports x86 RTF 0.0086–0.0150 and a 306 KB Linux library. | No Indian-language breakdown or native 8 kHz path. The released test set is small and not an Indic benchmark. | The license says Apache-2.0 **plus a non-compete restriction**, so it is not plain Apache-2.0 despite community quantization tags. | Useful latency baseline, but license and evidence make it a poor default dependency. |

The [FireRedASR2S paper](https://arxiv.org/abs/2603.10420) was submitted 2026-03-11. It documents the streaming DFSMN precisely: eight blocks, hidden/projection sizes 256/128, look-back order 20, streaming look-ahead order zero, roughly 0.6M parameters, and incremental fixed caches. The [reference postprocessor](https://github.com/FireRedTeam/FireRedVAD/blob/c30ec49e8cc69642b0ee65362eba11b9d11c6e54/fireredvad/core/stream_vad_postprocessor.py) uses a **trailing** mean, so it does not peek ahead, but default `min_speech_frame=8` and `min_silence_frame=20` delay event confirmation by roughly 80 and 200 ms at the 10 ms grid. Backdated segment timestamps must not be mistaken for online availability. For LID training masks, cache raw causal probabilities before this duration postprocessor.

Silero's upstream [README](https://github.com/snakers4/silero-vad/blob/v6.2.3/README.md) documents MIT licensing, native 8/16 kHz, a roughly 2 MB JIT model, and the CPU claim. Its [FAQ](https://github.com/snakers4/silero-vad/wiki/FAQ) explicitly says training code/data/papers have not been published. That makes it convenient but less auditable than FireRed; neither candidate has verified short Hindi-English telephony boundary performance.

## 5. Recommended causal target and routing contract

### 5.1 First ablation: mask language KD, do not add a class

Let `v_i` be a causal speech probability aligned to semantic LID frame `i`, `q_i` the normalized seven-way teacher target, `z_{i+D}` the delayed student logits, `m_i` the existing causal validity mask, and `a_i` the early-evidence ramp. Compare:

```text
baseline:       w_i = m_i * a_i
hard VAD mask:  w_i = m_i * a_i * 1[v_i >= tau]
soft VAD mask:  w_i = m_i * a_i * clip(v_i, v_floor, 1)

L_lang = T^2 * sum_i w_i KL(q_i || softmax(z_{i+D}/T)) / sum_i w_i.
```

Thresholds/floors must be selected on a speaker-disjoint development set. Preserve time: never concatenate detected speech frames, because that would erase pause duration and make switch latency uninterpretable. Expand a coarser VAD grid only by previous-value hold, and record each VAD output's latest audio sample and availability time exactly as for teacher targets.

Do not crop the offline teacher window independently from the student waveform in the first experiment. That would change both the target and mask at once. First test whether excluding non-speech loss/router votes helps while keeping the selected causal teacher target fixed.

### 5.2 Router policy

Use raw or minimally smoothed causal VAD with calibrated onset/offset hysteresis:

```text
if speech probability >= tau_on:
    update LID EMA and language dwell
elif speech probability <= tau_off:
    freeze LID EMA and language dwell
else:
    emit no vote
```

Before the first language commit, no speech leaves the route at `UNKNOWN`. After a commit, a short pause preserves the active ASR route but contributes no language evidence and no flip. A true VAD end-of-turn may reset **candidate/dwell state**; whether it resets the active language is a product prior, not an acoustic language decision. Silence can never satisfy a challenger dwell.

Measure VAD onset/offset in availability time. FireRed's start/end timestamps are retrospectively backdated after minimum-duration confirmation; they are useful segment coordinates, not the time at which an online router knew the event.

### 5.3 Optional factorized student speech head

Only if the external-mask experiment helps, expose the 32-dimensional penultimate classifier state and add `Linear(32,1)`: **+33 parameters, 42,600 total**. Train it with human VAD labels where available and frozen pseudo-labels elsewhere:

```text
L = L_language_on_speech + beta * BCEWithLogits(speech_logit, speech_target).
```

Keep the seven-way language KD conditional on speech. Compare this factorized head with the external VAD under identical route policy. Do not call it `UNKNOWN`: name it `speech`/`non_speech`, and keep the separately proposed in-set/OOS gate distinct.

### 5.4 Metrics and data gates

Report two layers, because a VAD can improve conditional LID by simply dropping hard speech:

1. **VAD:** frame AUC/PR-AUC, speech miss rate, non-speech false-alarm rate, speech coverage, onset/offset availability lag, and CPU p50/p95/RTF at 16 and 8 kHz.
2. **End-to-end LID/router:** speech-only language macro-F1, no-collar LD-DER, first-decision and switch recall/lag from target-speech onset, false known-language transitions per non-speech hour, wrong-route speech time, and abstention/coverage.

Use synthetic inserted gaps/noise for exact timing unit tests, but calibrate/report on human frame annotations. IIT Bombay's requested TextGrids contain `S` regions and are useful for a two-speaker boundary test; they are not a population benchmark. The current 4 s splice metadata and the amplitude proxy above are not acceptable final VAD truth.

## 6. Testable experiment order

### A. Frozen-model VAD bake-off

Pin FireRed Stream-VAD at `7990aac…a03b8c` and Silero at the checked Hub revision `394d7e6…4afd8`/upstream weight hashes. Without retraining the LID student, cache raw frame probabilities and exact availability ledgers on development-only human-labelled speech/non-speech, the current clips, and paired 8 kHz transforms. Include a trivial causal energy baseline. Sweep thresholds on development only.

Expected model gain: **none**; this selects a gate. The hypothesis that FireRed wins at 16 kHz and Silero is safer at native 8 kHz is **unverified**. Shortlist a model only if speech recall is at least 98% at no more than 10% non-speech false alarms, p95 onset availability lag is at most 100 ms before LID policy delay, 8 kHz speech recall loses at most 2 points, and six-thread CPU RTF is at most 0.02. Report failures rather than retuning on the two switch clips.

### B. Mask and policy ablation

After METHOD-0 produces a causally valid target, compare the unchanged baseline, hard VAD loss mask, soft VAD loss weight, and route-only VAD freeze with identical initialization, batch order, targets, steps, and language policy. Add 0.1/0.25/0.5/1.0 s inserted silence and noise gaps to development traces, not by compressing time.

Expected direction (**unverified**): at least 50% fewer false known-language transitions during non-speech, with no intrinsic gain promised for speech-only language accuracy. Accept an arm only if speech-only macro accuracy falls at most 1 point, initial/switch recall does not fall, median switch lag from target-speech onset rises at most 50 ms, and non-speech false transitions fall at least 50%.

### C. Factorized speech head only after B succeeds

Compare the winning external gate with the +33-parameter internal speech head. Keep external VAD labels frozen and provenance-bound; evaluate clean, noise, codec, and OOS-speech slices separately. Expected benefit (**unverified**): remove the external runtime at nearly unchanged student cost. Adopt only if the internal head is within 1 point VAD F1 and 25 ms p95 onset lag of the external gate, retains B's language/stability gates, and does not reinterpret unsupported-language speech as non-speech.

## 7. What remains unverified

- No reviewed source reports Hindi-English switch-LID gain from adding this exact VAD mask or factorized speech head.
- FireRed's public aggregate score is not a released causal-checkpoint, per-language, CPU, or 8 kHz result; FLEURS-VAD-102 annotations were still unavailable on the cutoff date.
- Silero's broad-language and CPU claims are upstream self-reports; its training corpus and training code are not public.
- The local amplitude audit identifies digital/low-amplitude frames, not human-labelled speech activity. TTS, codec noise, breath, overlap, music, and far-field speech will violate a fixed energy rule.
- A VAD can add onset delay or suppress low-energy phones. Any improvement reported only on VAD-selected frames is subject to selection bias; end-to-end deadline recall and speech coverage are mandatory.
- A pause adjacent to a code switch is not itself evidence for either language. Reference boundaries must identify the onset of target-language speech and retain silence as a separate interval.

## Primary sources and dated artifacts

- Xu et al., [FireRedASR2S](https://arxiv.org/abs/2603.10420), arXiv v1 submitted **2026-03-11**.
- FireRed Team, [`FireRedTeam/FireRedVAD`](https://huggingface.co/FireRedTeam/FireRedVAD), official Hub model created **2026-02-11**, last modified **2026-03-13**, API/revision checked **2026-09-26**.
- FireRed Team, [causal streaming implementation](https://github.com/FireRedTeam/FireRedVAD/blob/c30ec49e8cc69642b0ee65362eba11b9d11c6e54/fireredvad/stream_vad.py) and [trailing postprocessor](https://github.com/FireRedTeam/FireRedVAD/blob/c30ec49e8cc69642b0ee65362eba11b9d11c6e54/fireredvad/core/stream_vad_postprocessor.py), repository commit dated **2026-05-06**, checked **2026-09-26**.
- Silero Team, [Silero VAD v6.2.3](https://github.com/snakers4/silero-vad/releases/tag/v6.2.3), released **2026-09-23**; [`tphakala/silero-vad`](https://huggingface.co/tphakala/silero-vad) audited v6.2.1 Hub mirror created **2026-08-24**.
- NVIDIA, [`Frame_VAD_Multilingual_MarbleNet_v2.0`](https://huggingface.co/nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0), Hub card last modified **2025-05-14**, checked **2026-09-26**.
- TEN Framework, [`TEN-framework/ten-vad`](https://huggingface.co/TEN-framework/ten-vad), Hub revision last modified **2026-03-13**, license/source checked **2026-09-26**.
- Dey, Mondal, and Kurmi, [Teacher-Free Knowledge Distillation for Improving Short-Utterance Spoken Language Identification](https://www.isca-archive.org/interspeech_2025/dey25_interspeech.html), Interspeech **2025**.
- Kalda et al., [TalTech-1 at DISPLACE 2024](https://www.isca-archive.org/interspeech_2024/kalda24_interspeech.html), Interspeech **2024**.
- Ye et al., [SC-MoE](https://www.isca-archive.org/interspeech_2024/ye24_interspeech.html), Interspeech **2024**.
- Lee et al., [SAGE-LD](https://arxiv.org/abs/2510.00582), submitted **2025-10-01**.
