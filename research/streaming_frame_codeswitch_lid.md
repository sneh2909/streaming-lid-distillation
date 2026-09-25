# Streaming, frame-level, and code-switch spoken LID

Research cutoff: **2026-09-25**. Web and Hugging Face Hub results were checked on that date.

## Bottom line

No reviewed public model simultaneously provides all four properties this project needs: **Hindi–English coverage, local frame/segment labels, causal or explicitly bounded-lookahead inference, and reusable weights with a clear license**.

The literature separates into three different tasks that are often all called “streaming LID”:

1. **Prefix or utterance LID:** emit a better estimate of one utterance-level language as more audio arrives. This can be truly online, but cumulative history makes it unsuitable for rapid mid-utterance switches unless state is reset or forgotten.
2. **Language diarization:** emit a local language label for each frame or short segment. The strongest published Indic systems are frame-aligned, but their Wav2vec/WavLM, BLSTM, Conformer, or self-attention context is normally non-causal.
3. **ASR-coupled LID:** use language tags or a framewise router inside a streaming recognizer. This is the closest published match to causal code-switch routing, but released standalone LID weights and boundary metrics are absent.

The only directly downloadable general-purpose streaming checkpoint found on the Hub is [tflite-hub/conformer-lang-id](https://huggingface.co/tflite-hub/conformer-lang-id) (created 2024-09-15, last modified 2024-09-19, Apache-2.0). It is worth benchmarking as a streaming reference, not adopting as a switch teacher without measurement: its attentive pool preserves **all** preceding frames.

For this repository, the most transferable ideas are smaller than the published models:

- supervise a **local** language state rather than a prefix-level language;
- represent silence/uncertain evidence explicitly, as SC-MoE represents CTC blank;
- train on short embedded-language spans and weight the minority/embedded language;
- keep smoothing memory finite and report its contribution to switch latency separately;
- include Hindi-accented English, since 2026 evidence shows it is a major Hindi–English failure mode.

## Comparison at a glance

“Frame output” below means an output is aligned to time. It does **not** imply causal inference.

| Work | Output and temporal context | Code-switch / Indic evidence | Reported result | Public status and fit |
|---|---|---|---|---|
| González-Domínguez et al., *Frame-by-frame LID* (2015) | DNN posterior for each newly observed frame, then accumulated for a short utterance | No switch segmentation | 40% relative improvement over i-vector on NIST LRE09 3 s trials | Historical causal scoring method; no current checkpoint |
| Chandak et al., *AcousText streaming LID* (2020) | Causal 3×768 LSTM acoustic state plus partial hypotheses from two ASRs every 600 ms; commits one utterance label | en-IN/hi-IN data are mostly code-switched, but receive one label | AcousText relative error reduction of 70.6% en-IN and 46.8% hi-IN versus its acoustic baseline | Private data/model; paper explicitly does not early-stop code-switch pairs |
| Červa et al., *off-line to on-line SLI* (2021) | Frame labels from a DNN classifier, smoothed by a WFST decoder | Real Czech–Slovak broadcasts, so it genuinely tracks changes | About 2.5 s average detection latency; ASR WER increase of 2.9% over the 18.1% manual-label reference | True online change tracking, but unrelated languages/data and no reviewed release |
| Wang et al., *Attentive Temporal Pooling Conformer* (2022) | Streaming output about every 60 ms; recurrent weighted mean/std over every frame since stream start | 65-language paper includes en, hi, bn, gu, mr, ta, te; no code-switch experiment | Medium: 88.24% voice-query and 77.26% long-form macro accuracy; 1.91 GFLOP/audio-s | Apache public TFLite weights exist, but public checkpoint accuracy is unverified |
| Zhang et al., *joint streaming RNN-T LID* (2022) | Frame-synchronous prefix LID; cumulative mean/std; fixed 0.9 s second-pass right context | Nine private locales, no Indic and no code-switch evaluation | 92.9% nine-locale accuracy averaged over frames; 96.2% after merging en/es locales | 0.7–1M LID head, but depends on a roughly 226M ASR stack; no release |
| Punjabi et al., *streaming bilingual RNN-T* (2020) | Framewise acoustic embeddings, but joint language tag/posterior is utterance-final | Direct en-IN/hi-IN experiment with within-utterance switching | English-Hindi degrades relative to acoustic LID/monolingual ASR | Useful negative result; private model, not a local diarizer |
| Ye et al., *SC-MoE* (2024) | Chunk-streaming Conformer; CTC router emits Mandarin, English, or blank at encoder frames | Real Mandarin–English code-switch ASR | Streaming mixed error rate 9.42% vs 10.40% baseline; 83.8M total / 50.2M active | Strong routing design, but **no standalone LID/boundary score or released weights found** |
| Liu et al., *E2E language diarization* (2021) | 200 ms labels; BLSTM or full-sequence self-attention over x-vectors | Gujarati/Tamil/Telugu–English MSCS | XSA-E2E average 82.6% accuracy, 5.8% EER | Frame-aligned but offline; MIT code, no checkpoint |
| Mishra et al., *Wav2vec E2E diarization* (2023) | 20 ms Wav2vec features pooled to 200 ms; self-attention label sequence | Same three Indic–English MSCS pairs | Best average: 90.1% ID accuracy, 3.3% EER, 11.2 DER, 21.8 JER | Strongest directly relevant older Indic result; offline architecture; code has no declared license/weights |
| Frost et al., *fine-tuned WavLM diarization* (2022/2023) | 20 ms logits from bidirectional WavLM | English plus four South African languages | Large gains over BLSTM/x-vector baselines; not comparable to Indic sets | Torch Hub checkpoints exist; about 315M parameters and no repository license found |
| Lee et al., *SAGE-LD* (2025) | 25 ms masks from MMS conv frontend, Conformer, and query decoder | Pretrained on simulated Indic–English; adapted to DISPLACE | Practical DER 21.37 on DISPLACE-D, 18.03 on DISPLACE-E, 16.92 on SA Soap Opera | Best recent Indic diarization architecture reviewed; 72M and not described as causal; no licensed weights found |
| Patra et al., *accented-English adaptation* (2026) | Full-utterance multi-label ranking from MMS-LID; no boundaries | Hindi–English and Bengali–English, plus two other pairs | Hindi–English exact match: 509/3136 baseline, 915 LoRA, 2120 adapters, 997 Whisper | Not streaming/framewise; important evidence about accented English and false positives |

## Systems that are genuinely online

### 1. Attentive Temporal Pooling Conformer

[Wang et al.](https://arxiv.org/html/2202.12163) (arXiv v4, 2022-05-01; Odyssey 2022) use 128-bin filterbanks from 32 ms windows at a 10 ms step, stack/subsample them to 30 ms, and apply twelve Conformer layers. A further factor-of-two reduction makes each output step cover about 60 ms. Small, medium, and large encoders are approximately 7M, 30M, and 120M parameters and require 0.45, 1.91, and 7.56 GFLOP per second of audio in the paper.

The key online operation maintains cumulative weighted count, sum, and squared-sum states. It can therefore emit a weighted mean and standard deviation after every step with constant state size. This solves long-recording memory use, but its semantics remain **one language for the entire prefix**. The paper trains on a single label per utterance and evaluates 3.3 s voice queries and roughly 20.7 minute monolingual long-form items.

**Implication for switching (inference, not measured by the paper):** after a long Hindi prefix, every English frame competes with the accumulated Hindi sufficient statistics. Recovery time can grow with pre-switch history. A rolling window, decay, or an externally triggered reset is required before this becomes a local language tracker.

The Hub card says the released model was trained only on public data, while the paper used 1M–20M utterances per language from private Google Assistant/YouTube sets. Therefore the paper’s accuracy numbers must not be assigned to the public checkpoint. The public [language map](https://github.com/google/speaker-id/blob/master/lingvo/sidlingvo/language_map.py) currently has 71 entries and includes en-US, hi-IN, bn-BD, gu-IN, mr-IN, ta-IN, te-IN, kn-IN, and ml-IN.

The release contains medium (28,940,456 bytes), small (7,543,784), smaller (4,266,864), and tiny (2,535,664) TFLite files. The high-level [wav_to_lang helper](https://github.com/google/speaker-id/blob/master/lingvo/sidlingvo/wav_to_lang.py) resets state, runs the whole file, and returns only the last output frame. Its lower-level runner does expose the frame sequence, but a real chunked test must persist explicit TFLite states across two-feature-frame steps. VAD removes frames, so outputs also need to be mapped back to wall-clock time before measuring switch latency.

**Unverified:** public-checkpoint accuracy, 8 kHz behavior, CPU real-time factor, and code-switch recovery. The “smaller” and “tiny” architecture/quality trade-offs are not documented on the card.

### 2. Cascaded RNN-T with a frame-synchronous LID head

[Zhang et al.](https://www.isca-archive.org/interspeech_2022/zhang22da_interspeech.pdf) (Interspeech 2022) attach two 512-unit fully connected layers, only 0.7–1M parameters, to a large cascaded RNN-T. A 110M causal Conformer encoder has no right context; a 50M second encoder has 15 decoding frames, or 0.9 s, of right context; the two 33M decoders bring the total system to roughly 226M parameters.

The LID input joins lower-level causal acoustics with higher-level right-context features. Streaming mean/std over all frames from 1 through t then produces a distribution at every 60 ms decoding step. On nine private Voice Search locales, mean accuracy over all time steps is 92.9%; when English and Spanish locale pairs are merged, it is 96.2%. With the built-in 0.9 s delay, frame indices 0, 15, 30, and final obtain 88.1%, 94.7%, 97.6%, and 98.8% clustered accuracy.

This establishes a useful design point—**small bounded lookahead plus cumulative prefix evidence**—but not switch tracking. The test utterances are at most 5.5 s, carry one language, and contain no Hindi or other Indian language. No model/code release was found.

### 3. Acoustic plus partial-ASR LID

[Chandak et al.](https://arxiv.org/html/2006.00703) (arXiv, 2020-06-01; CC BY 4.0 paper) use a three-layer unidirectional LSTM with 768 cells per layer over 64-bin filterbanks. Every 600 ms, its acoustic embedding is fused with character-LSTM embeddings from the partial 1-best hypotheses of two competing recognizers. A high-confidence decision terminates the nonmatching ASR.

The experiment directly includes Indian English versus Hindi. Most examples in that bucket are code-switched, but every frame is trained with the same utterance label. The authors explicitly disable early stopping for this pair because a late word may determine the intended utterance language. Thus its 70.6%/46.8% relative error reduction for en-IN/hi-IN is evidence that lexical hypotheses help **global intent LID**, not evidence of boundary detection. Running two ASRs is also inconsistent with this project’s small CPU student goal.

The paper’s semi-supervised result is still relevant for later distillation work: a high-confidence text-based teacher labels 1M streams (1,100 h), improving the causal acoustic LID by 20.0% and 17.6% relative error for en-IN and hi-IN. That method belongs primarily to Topic 3.

### 4. The rare online system that actually follows language changes

[Červa et al.](https://doi.org/10.1016/j.csl.2020.101180) (available online 2020-12-15; *Computer Speech & Language* 68, 2021) feed multilingual/augmented bottleneck features to a frame classifier and smooth its output with a weighted finite-state transducer. It processes Czech–Slovak broadcast streams and reports about 2.5 s mean language-change latency.

This paper is important because it measures the right event: a label stream and change latency, not merely final utterance accuracy. It also shows that decoder smoothing is a first-class latency/accuracy trade-off. Its language pair, long-context ASR features, and resource footprint do not transfer directly, and no reusable checkpoint was found in this review.

### 5. Streaming code-switch routers inside ASR

[SC-MoE](https://www.isca-archive.org/interspeech_2024/ye24_interspeech.pdf) (Ye et al., Interspeech 2024) is the closest architecture match to a causal local router. It uses dynamic chunks in a U2++ Conformer and trains CTC language routers at several encoder depths. Crucially, blank is a third expert beside Mandarin and English rather than being replaced by a prior language decision. Later routers can correct earlier routing errors.

At streaming configuration [16, 8], the full model has 83.8M parameters but activates 50.2M and reduces Mandarin–English mixed error rate from 10.40% to 9.42%. Those are **ASR** metrics. The paper does not report router frame accuracy, false switches, boundary lag, or Hindi–English results, and no public checkpoint/repository was found in the reviewed sources. The transferable idea is the explicit blank/insufficient-evidence state, not the MoE scale.

[Punjabi et al.](https://arxiv.org/html/2007.03900) (arXiv, 2020-07-08) provide the most direct Hindi–English negative result. Their acoustic LID supplies updated framewise embeddings to a streaming RNN-T, but joint language tags are appended at utterance end and final LID uses posteriors after the last audio frame. English-Hindi performance degrades because of within-utterance switching and Latin/Devanagari ambiguity. Framewise embeddings alone did not turn utterance supervision into reliable local labels.

## Frame-aligned code-switch diarization, mostly offline

### 1. X-vector/self-attention E2E diarization

[Liu et al.](https://www.isca-archive.org/interspeech_2021/liu21d_interspeech.pdf) (Interspeech 2021) use the Microsoft/WSTCSMC Gujarati–English, Tamil–English, and Telugu–English data, whose ground truth is assigned every 200 ms. The BLSTM version has five bidirectional 256-unit layers. The stronger XSA-E2E version extracts a 256-D x-vector from each 200 ms segment and passes the sequence through four full self-attention encoder blocks.

XSA-E2E averages 82.6% accuracy and 5.8% EER across the three language pairs; on simulated Mandarin–English SEAME it reaches 89.84% accuracy. Both architectures see future context, so these are offline diarizers despite their segment-level outputs. The [MIT-licensed code](https://github.com/Lhx94As/E2E-language-diarization) is public, but its README says data preprocessing was incomplete and the repository contains no checkpoint as checked on 2026-09-25.

### 2. Indic Wav2vec E2E diarization

[Mishra et al.](https://www.isca-archive.org/interspeech_2023/mishra23_interspeech.pdf) (Interspeech 2023) replace x-vectors with a Wav2vec representation pretrained on roughly 1,000 h across 23 Indian languages. Features occur every 20 ms and are pooled to the dataset’s 200 ms labels, then passed through self-attention.

The best fine-tuned Wav2vec plus attention-pooling system averages 90.1% identification accuracy, 3.3% EER, 11.2 DER, and 21.8 JER. JER matters because primary-language duration is about four times embedded-language duration. Embedded-language recall rises to 79.8%, versus 0% for the x-vector baseline, while embedded-to-primary confusion falls from 65.2% to 7.7%.

This is strong evidence for an Indic-pretrained acoustic representation and an imbalance-aware evaluation, but it is not a streaming result: the Wav2vec Transformer and sequence self-attention are not described with causal masks. The [paper’s GitHub repository](https://github.com/jagabandhumishra/W2V-E2E-Language-Diarization) has code, but no license declaration or weights were found on 2026-09-25.

### 3. Fine-tuned WavLM frame labels

[Frost et al.](https://arxiv.org/abs/2312.09645) (SACAIR 2022; arXiv 2023-12-15) place a linear language head on WavLM and return logits corresponding to non-overlapping 20 ms spans. They evaluate English against isiZulu, isiXhosa, Setswana, and Sesotho. The [repository](https://github.com/GeoffreyFrost/code-switched-language-diarization) releases three WavLM-large Torch Hub checkpoints.

The time grid is fine, but the WavLM transformer is bidirectional, so each 20 ms label has full-utterance context. WavLM-large is also roughly 315–317M parameters. The weights are useful for reproducibility outside this language set, not for the target CPU runtime. No license file or SPDX license was present in the repository as checked on 2026-09-25, so reuse rights are **unverified**.

### 4. SAGE-LD

[SAGE-LD](https://arxiv.org/html/2510.00582) (Lee et al., arXiv 2025-10-01) is the strongest recent architecture in this review. It keeps only the MMS convolutional frontend, avoiding the pretrained Transformer’s tendency to mix languages, then applies a Conformer contextual encoder and an iterative masked-attention decoder with five learnable queries, including one VAD slot. It has 72M parameters.

The authors simulate 100 h each of Indic–English and Bantu–English code-switch data, use voice conversion to avoid learning that every speaker change is a language change, and add noise/reverberation. Embedded-language frames receive focal-loss weight 3, with focal Tversky loss added for sparse segments. On DISPLACE-D/DISPLACE-E/SA Soap Opera, practical DER is 21.37/18.03/16.92. Keeping 25 ms frames beats pooling to 105 or 205 ms on all three sets.

This supports fine-resolution local targets, embedded-language weighting, explicit VAD, and same-speaker simulation. It does **not** establish deployable streaming inference: the paper gives no causal attention mask, chunking protocol, lookahead, or online latency, and the query decoder aggregates contextual embeddings. Treat it as offline unless the authors document otherwise. The [implementation repository](https://github.com/sanghyang00/sage-ld) had no declared license or downloadable checkpoint in the reviewed tree on 2026-09-25.

### 5. Boundary-only systems

[Spoken Language Change Detection Inspired by Speaker Change Detection](https://link.springer.com/article/10.1007/s00034-024-02743-w) (Mishra and Prasanna, 2024) evaluates synthetic Hindi–English and the same 200 ms MSCS Indic–English pairs. It finds that detecting language changes needs a larger neighborhood than detecting speaker changes and benefits from language-specific prior exposure. This supports evaluating multiple embedded-span lengths. Its window around the candidate change uses both sides, so it is not evidence of causal latency.

[Generative attention based implicit language change detection](https://doi.org/10.1016/j.dsp.2024.104678) (Mishra and Prasanna, *Digital Signal Processing*, 2024-11) reports 50.7% relative improvement over an unsupervised distance baseline on MSCS. It detects a change point rather than assigning a durable language identity and does not report an online protocol, so it is complementary to—not a replacement for—a frame LID student.

## The 2026 Hindi–English result to carry into training

[Patra, Sah, and Jyothi](https://aclanthology.org/2026.findings-eacl.242/) (Findings of EACL, 2026-03) evaluate an MMS-LID model as an utterance-level, multi-label ranker. Baseline MMS nearly misses embedded English in Hindi–English and Bengali–English. They adapt it with only 50–500 matrix-language-accented English samples; their standard matched setup uses 80.

Adapters maximize Hindi–English exact match (2,120 of 3,136 versus 509 for baseline MMS), but overpredict English on monolingual speech. LoRA yields the best balance between detecting English in code-switch speech and suppressing it in monolingual Hindi. Hindi-accented English is taken from NISP; Hindi–English evaluation uses MUCS. Closely related Urdu and Punjabi frequently outrank English, especially at low code-mixing density.

This is not a frame model and should not be quoted as a switch-latency result. It does justify an accented-English slice and an out-of-set confusion audit for any Hindi–English teacher/student.

## Hugging Face Hub and release audit

The Hub was queried for streaming, diarization, codeswitch, and code-switching on 2026-09-25.

- [tflite-hub/conformer-lang-id](https://huggingface.co/tflite-hub/conformer-lang-id): actual audio streaming LID artifact; Apache-2.0; four quantized TFLite sizes. Public-checkpoint metrics are unverified.
- [TalTechNLP/voxlingua107-xls-r-300m-wav2vec](https://huggingface.co/TalTechNLP/voxlingua107-xls-r-300m-wav2vec): audio classification, CC BY 4.0, but full-utterance/non-causal—not a streaming diarizer.
- [sagorsarker/codeswitch-hineng-lid-lince](https://huggingface.co/sagorsarker/codeswitch-hineng-lid-lince): despite the name, its pipeline is **text token classification**, not speech.
- Search results for “streaming” and “diarization” were dominated by ASR and **speaker** diarization. No Hub-hosted Hindi–English causal audio diarizer surfaced. This is a dated search observation, **not proof that none exists**.

Release status outside the Hub:

| Project | Code | Weights | License checked 2026-09-25 |
|---|---|---|---|
| Google SidLingvo / Conformer LID | Yes | HF TFLite | Apache-2.0 |
| Liu E2E language diarization | Yes | No checkpoint found | MIT |
| Mishra W2V-E2E | Yes | No checkpoint found | No license declared |
| Frost WavLM LD | Yes | Three GitHub-release checkpoints | No license declared |
| SAGE-LD | Yes | No checkpoint found | No license declared |
| SC-MoE / Amazon / Google RNN-T systems | No linked reusable implementation found | No | **Unverified** |

## Consequences for this project

1. **Do not equate “per-frame” with “local.”** The Google prefix models emit every frame but encode all prior frames; WavLM/Wav2vec diarizers align every frame but see future frames.
2. **Do not use cumulative pooling for switch targets.** It is appropriate for “what language has this stream mostly been?” and structurally mismatched to “what language is active now?”
3. **Preserve the causal TCN student.** At 42,567 parameters it is already much closer to the CPU requirement than the 72M–315M diarizers. The literature supports changing targets, class balance, and decoding before replacing it.
4. **Separate model delay from decision delay.** The repository currently has 435 ms architectural worst-case latency but 1,655 ms observed switch lag. The gap is likely dominated by evidence window, EMA, and dwell behavior; each must be measured independently.
5. **Add silence/uncertain evidence.** Both SAGE-LD’s VAD slot and SC-MoE’s blank expert avoid forcing a language decision on acoustically weak frames.
6. **Evaluate embedded-language recall, not only average agreement.** MSCS has an approximately 4:1 primary/secondary duration ratio; Mishra et al. show that respectable average accuracy can coexist with zero embedded-language recall.
7. **Use same-speaker or speaker-controlled switch construction.** Otherwise the student can learn speaker boundaries as language boundaries. SAGE-LD’s voice-conversion simulation is evidence for this concern; a lightweight first test can at least match channel/noise and keep speakers disjoint across splits.

## Unverified items and limits

- Published Google Conformer results are from private training/evaluation and are not validation of the public-data HF files.
- No reviewed paper reports 8 kHz Hindi–English causal boundary accuracy or CPU RTF for its code-switch diarizer.
- “No checkpoint found” means none appeared in the cited repository/Hub search on 2026-09-25; it is not a permanent or exhaustive claim.
- SAGE-LD, Mishra W2V-E2E, and Frost WavLM are classified as offline from their unmasked contextual architectures and absence of an online protocol. If unpublished causal configurations exist, they were not verified.
- Cross-paper accuracy, DER, JER, and mixed error rate are not directly comparable because labels, collar rules, silence handling, and language sets differ.
- The posted Interspeech 2026 program was searched before the conference start (2026-09-27); no additional dedicated streaming language-diarization model surfaced, but proceedings indexing may still change.

## Primary sources

- González-Domínguez et al., [Frame by Frame Language Identification in Short Utterances using Deep Neural Networks](https://research.google/pubs/frame-by-frame-language-identification-in-short-utterances-using-deep-neural-networks/), *Neural Networks* 64, 2015.
- Chandak et al., [Streaming Language Identification using Combination of Acoustic Representations and ASR Hypotheses](https://arxiv.org/html/2006.00703), 2020-06-01.
- Punjabi et al., [Streaming End-to-End Bilingual ASR Systems with Joint Language Identification](https://arxiv.org/html/2007.03900), 2020-07-08.
- Červa et al., [Identification of related languages from spoken data: Moving from off-line to on-line scenario](https://doi.org/10.1016/j.csl.2020.101180), online 2020-12-15 / volume 2021.
- Liu et al., [End-to-End Language Diarization for Bilingual Code-Switching Speech](https://www.isca-archive.org/interspeech_2021/liu21d_interspeech.html), Interspeech 2021.
- Wang et al., [Attentive Temporal Pooling for Conformer-based Streaming Language Identification in Long-form Speech](https://arxiv.org/html/2202.12163), arXiv v4 2022-05-01.
- Zhang et al., [Streaming End-to-End Multilingual Speech Recognition with Joint Language Identification](https://www.isca-archive.org/interspeech_2022/zhang22da_interspeech.html), Interspeech 2022.
- Frost et al., [Fine-Tuned Self-Supervised Speech Representations for Language Diarization in Multilingual Code-Switched Speech](https://arxiv.org/abs/2312.09645), SACAIR 2022 / arXiv 2023-12-15.
- Mishra et al., [End to End Spoken Language Diarization with Wav2vec Embeddings](https://www.isca-archive.org/interspeech_2023/mishra23_interspeech.html), Interspeech 2023.
- Mishra and Prasanna, [Spoken Language Change Detection Inspired by Speaker Change Detection](https://link.springer.com/article/10.1007/s00034-024-02743-w), 2024.
- Mishra and Prasanna, [Generative attention based framework for implicit language change detection](https://doi.org/10.1016/j.dsp.2024.104678), 2024-11.
- Ye et al., [SC-MoE: Switch Conformer Mixture of Experts for Unified Streaming and Non-streaming Code-Switching ASR](https://www.isca-archive.org/interspeech_2024/ye24_interspeech.html), Interspeech 2024.
- Lee et al., [SAGE-LD](https://arxiv.org/abs/2510.00582), 2025-10-01.
- Patra et al., [Improving Language Identification for Code-Switched Speech: The Pivotal Role of Accented English](https://aclanthology.org/2026.findings-eacl.242/), Findings of EACL, 2026-03.
- Hugging Face, [tflite-hub/conformer-lang-id model card](https://huggingface.co/tflite-hub/conformer-lang-id), created 2024-09-15 / checked 2026-09-25.
