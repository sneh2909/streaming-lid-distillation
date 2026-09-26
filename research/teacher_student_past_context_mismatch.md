# Teacher/student past-context mismatch

Research cutoff: **2026-09-26 02:30 IST**. Web pages and Hugging Face Hub metadata were checked on **2026-09-26**. Local evidence is read-only and tied to the hashes stated below.

## Bottom line

The current alignment matches the teacher's **latest** audio sample, but not its complete evidence set. For teacher target frame `i`, the frozen ECAPA sees 1,750 ms of past audio and 250 ms of future audio. The aligned student output sees exactly the same 250 ms future endpoint but only **1,075 ms of past audio**. The missing 675 ms is not future leakage; it is a teacher/student capacity and evidence mismatch.

The smallest exact architectural repair is one additional 64-channel residual block with dilation 34:

- theoretical receptive field: 127 -> 195 feature contexts;
- past evidence at the aligned target: 1,075 -> **1,755 ms**;
- parameters: 42,567 -> **47,111** (+4,544, +10.68%);
- future context and conservative algorithmic latency: unchanged.

This is worth one controlled ablation, but **not as a switch-latency fix**. The experimenter's teacher-window audit rejected every shorter window, and the new dense-hop audit still places the current ECAPA target's median stable semantic transition at **1,260 ms**. A longer-history student may imitate that late target more faithfully; it cannot make the target itself transition earlier. Expected accuracy and fidelity gains are **unverified**, and the expected teacher-floor switch-lag gain is **zero**.

## 1. Exact evidence ledger

The frontend uses a 25 ms uncentred analysis window and a 10 ms hop. `extract_window` defines the teacher anchor as the **end** of feature frame `i`:

```text
anchor_sample(i) = i * 160 + 400
teacher samples  = [anchor - 1,750 ms, anchor + 250 ms)
```

For a stack of kernel-3 causal blocks with dilations `d_k`, the student's theoretical receptive field in stacked-feature contexts is

```text
R = 1 + 2 * sum_k d_k.
```

Teacher frame `i` is matched to student output `j=i+D`, with `D=21`, and each context explicitly stacks `L=4` future feature frames. The union of raw feature indices used by that output is therefore

```text
[j-(R-1), j+L] = [i+D-(R-1), i+D+L].
```

For the default `R=127`, that is `i-105 ... i+25`. Measured from the end of feature `i`, the earliest feature begins `105*10+25 = 1,075 ms` in the past and the latest feature ends exactly 250 ms in the future. This independently reproduces the reviewer calculation in [`review/rev-model.md`](../review/rev-model.md) and [`review/rev-teacher.md`](../review/rev-teacher.md) (2026-09-26).

To cover 1,750 ms of teacher past while preserving `D=21,L=4`, the student needs

```text
25 + (R - 1 - D) * 10 >= 1,750 ms,
```

so the minimum integer receptive field is `R=195`. Appending dilation 34 gives `127 + 2*34 = 195`. It does not create holes in the theoretical history: the existing stack covers lags `0...126`, while the new block combines the overlapping intervals `0...126`, `34...160`, and `68...194`.

### Candidate sizes and measured replay cost

| Arm | Dilations | `R` | Feature span for target `i` | Past / future | Params | Steady replay feature buffer | Median model-only replay, 8 s |
|---|---|---:|---|---:|---:|---:|---:|
| current | `1,2,4,8,16,32` | 127 | `i-105 ... i+25` | 1,075 / 250 ms | 42,567 | 130 frames / 20.31 KiB | 54.72 ms |
| exact-history | `1,2,4,8,16,32,34` | 195 | `i-173 ... i+25` | 1,755 / 250 ms | 47,111 | 198 frames / 30.94 KiB | 73.22 ms |
| power-of-two control | `1,2,4,8,16,32,64` | 255 | `i-233 ... i+25` | 2,355 / 250 ms | 47,111 | 258 frames / 40.31 KiB | 85.60 ms |

The local timing used one random 800-frame input, 15 repeats after warm-up, six Torch CPU threads, and the repository's deliberately uncached-overlap `streaming_forward`. It excludes the frontend and does not represent a production cached-convolution kernel. The dilation-34 arm was 33.8% slower than the current arm in this microbenchmark, but its model-only replay RTF was still about 0.0092. Absolute timing and any production speedup from per-layer state caching are **unverified**.

Increasing past history adds no wait for future audio, so it does not change the current analysis/evidence/chunk latency formula. It does increase retained state and replay work.

## 2. Why the missing history is real, not merely nominal

The pinned [`speechbrain/lang-id-voxlingua107-ecapa` configuration](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/blob/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9/hyperparams.yaml) (model revision last modified 2024-11-27; accessed 2026-09-26) uses sentence-level input normalization, ECAPA-TDNN layers, and an utterance embedding. SpeechBrain 1.0.3's [ECAPA source](https://github.com/speechbrain/speechbrain/blob/v1.0.3/speechbrain/lobes/models/ECAPA_TDNN.py) computes temporal means in squeeze-and-excitation blocks and global mean/std statistics in attentive statistics pooling. Thus every supplied part of the two-second teacher window can affect its one posterior; the extra 675 ms cannot be dismissed as unused from architecture inspection.

The direct LID duration evidence points the same way. TitaNet-LID reports VoxLingua107 error of 7.5% on 0–5 s inputs versus 5.2% on 5–20 s inputs for its 29M model, and states that performance generally improves with input duration. Its 29M global-context model also beats the cited ECAPA checkpoint's 11.9% error in the 0–5 s bin. These are utterance-level results, not code-switch or receptive-field evidence, but they warn against assuming that a shorter teacher view preserves language evidence ([Jia et al., Interspeech 2023](https://www.isca-archive.org/interspeech_2023/jia23b_interspeech.html), published 2023-08; accessed 2026-09-26).

The repository's own teacher-only tests are more directly relevant:

- The [teacher-window audit](../experiments/teacher-window-audit/REPORT.md) (2026-09-26) shortened the current 1,750/250 ms view to 750/250 ms. Median stable onset improved by only 125 ms, outside-collar accuracy fell 7.95 absolute points, and unmatched changes rose from 8 to 20. Causal 0.5/1/2 s and cumulative-prefix views also failed the gate.
- The [teacher anchor-hop audit](../experiments/teacher-anchor-hop-audit/REPORT.md) (2026-09-26) called the current window every 10 ms. Dense calls reduced median stable onset from 1,400 to 1,260 ms, only 140 ms, while producing 40 unmatched changes and 24.18x the current calls. Anchor density therefore does not explain the dominant target delay.

Together these results reject “shorten the ECAPA history” and “sample it more often” as sufficient repairs on the present two synthetic switches. They do **not** prove that extending the student history helps.

### Descriptive effective-history check

Theoretical coverage is not effective use. A read-only gradient audit used the checkpoint bytes present at 2026-09-26 02:29 IST (`sha256:bb2a7e2aee1796845c1880f618b0b465e025ace3912d860e02b2a63ac6c962e8`), one held-out clip per project language, and each clip's midpoint. For the known-class log posterior, it summed per-frame L2 norms of mel-feature gradients and divided by the total norm over the used context.

The oldest available 750–1,075 ms region contributed a median **5.97%** of that sensitivity across seven clips, but ranged from **5.45% to 41.96%**. This is a local gradient heuristic on a poorly generalising checkpoint, not a causal attribution or an accuracy result. It only shows that the oldest currently available history is not uniformly ignored; it says nothing about the inaccessible 675 ms.

That caveat matters because the NeurIPS study introducing the distinction between theoretical and effective receptive fields shows that effective influence generally occupies only a fraction of a convolutional network's theoretical field ([Luo et al., NeurIPS 2016](https://papers.nips.cc/paper/6203-understanding-the-effective-receptive-field-in-deep-convolutional-neural-networks), published 2016; accessed 2026-09-26). Its experiments are on vision CNNs, so transfer to this residual audio TCN is **unverified**. A larger theoretical field must be followed by a trained sensitivity/occlusion audit.

## 3. Clip-start padding is a second past-evidence mismatch

For local switch targets, the first teacher call has 27,600 leading zero samples, or 1.725 s. The current loss ramp reaches full weight at target frame 99, but that teacher window still has 11,760 zero samples, or 735 ms, on its left. A full real 1.75 s teacher history first exists at frame 173.

This edge effect does not explain the approximately four-second switch boundary, but it can corrupt early source-language supervision and initial-language acquisition. The target cache should record real versus padded teacher coverage, and a coverage-weighted or full-window mask should be compared with the current generic one-second ramp. Whether masking improves the initial commit is **unverified**; it could also delay first decisions by removing early supervision.

## 4. What the literature supports—and what it does not

Dilated convolution is an efficient way to enlarge a finite field without temporal downsampling. The original multi-scale dilated-convolution work shows exponential field expansion without losing resolution ([Yu and Koltun, ICLR 2016](https://arxiv.org/abs/1511.07122), arXiv 2015-11-23; accessed 2026-09-26), and the original dilated TCN work defines frame predictions as functions of a fixed-length temporal field ([Lea et al., CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.html), published 2017-07; accessed 2026-09-26). Neither paper is LID evidence, so neither predicts a gain here.

Streaming-ASR work supplies two narrower lessons:

- Shim et al. add a training-only branch so teacher and student feature KD operate with the same context, explicitly treating accessible-context mismatch as a distillation problem ([Interspeech 2023](https://www.isca-archive.org/interspeech_2023/shim23_interspeech.html), published 2023-08; accessed 2026-09-26). Their task, feature loss, and full-context auxiliary branch differ from this output-posterior LID setup.
- Emformer caches left-context keys/values and compresses longer history into memory while keeping low right-context latency ([Shi et al., ICASSP 2021](https://arxiv.org/abs/2010.10759), arXiv v4 2020-12-30; accessed 2026-09-26). This supports separating past-state cost from lookahead latency, not replacing the 42K TCN with a 120M ASR encoder.

No reviewed source found here measures the exact intervention “extend a 42K causal LID student's past from 1.075 to 1.755 s while distilling a backward-heavy ECAPA trajectory.” Any gain remains **unverified**.

## 5. Dated Hugging Face Hub audit

The following revisions were resolved through the Hub API on 2026-09-26. This is a non-exhaustive audit, not proof that no other model exists.

| Hub ID / pinned revision | History semantics | Why it is not a drop-in answer |
|---|---|---|
| [`speechbrain/lang-id-voxlingua107-ecapa@0253049`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9) | Whole supplied window via sentence normalization, SE temporal means, and attentive statistics pooling | Current offline teacher; Apache-2.0; utterance posterior, not a finite-history frame classifier |
| [`tflite-hub/conformer-lang-id@213db93`](https://huggingface.co/tflite-hub/conformer-lang-id/tree/213db93aebf552836020c029a405b355569f4739) | Recurrent weighted count/sum/sum-of-squares from stream start through `t` | Apache-2.0 and covers all seven project languages, but its history is cumulative rather than finite; published evaluation is monolingual, with no switch recovery result ([paper](https://arxiv.org/html/2202.12163v4), revised 2022-05-01) |
| [`surogate/ambernet-langid@3b7b3fb`](https://huggingface.co/surogate/ambernet-langid/tree/3b7b3fbfcc51753745f1d9abd107c9530c391d5b) | Entire input through SE and x-vector statistics pooling | 29M utterance classifier; card warns that code-switching and <5 s inputs degrade; Hub license metadata is `other`/NVIDIA NGC terms; no finite local output |
| [`Vengadanathan/hindi-hinglish-zipformer-ctc-streaming@c511762`](https://huggingface.co/Vengadanathan/hindi-hinglish-zipformer-ctc-streaming/tree/c5117627f436570b2d83d12a51fac307913174e0) | Card exposes 64/128/256-frame finite left-context settings | Useful recent Hindi/Hinglish streaming-ASR reference, but approximately 65M parameters, character CTC rather than LID, community-only results, and no language-boundary metric; transfer is **unverified** |

The public streaming Conformer is structurally too sticky for local switching unless its sufficient statistics are reset, decayed, or windowed. AmberNet and ECAPA remain utterance poolers. The Hindi/Hinglish Zipformer demonstrates that finite left context is a normal streaming-ASR control, but it cannot supervise seven-way local LID without a new head and labels. The existing finite TCN is still the simplest auditable student.

## 6. Recommended controlled ablation

Run this only after METHOD-0 replaces future-leaking interpolation with an availability-valid expansion. It is independent of finding a faster teacher trajectory and must not unblock the target-by-delay grid.

### Arms

1. `rf127`: current `(1,2,4,8,16,32)` baseline.
2. `rf195`: append dilation 34; exact teacher-history coverage.
3. `rf255`: append dilation 64; same parameter count as `rf195`, testing whether extra history helps or simply retains the old language longer.

Keep seed, shared-layer initialization, minibatch order, targets, update count, checkpoint selector, router, and evaluation manifests fixed. Record the exact theoretical feature/sample interval in the checkpoint identity. The new block necessarily introduces new parameters; copy the six shared blocks exactly and publish the new-block initialization hash rather than claiming identical complete initial states.

### Measurements

- optimization: train KD, valid-frame counts, gradient finiteness, exact-train teacher agreement;
- generalisation: fixed external-validation language-macro prefix/full accuracy and teacher agreement, not only the current synthetic voices;
- target fidelity: raw student excess lag relative to the same teacher trajectory, misses, and frame KL by distance from a boundary;
- actual routing: onset-anchored deadline recall, miss-aware lag, flip-flops, and wrong/UNKNOWN time;
- deployment: parameter/state bytes, model-only and frontend-inclusive p50/p95 chunk compute, replay RTF, and unchanged latest-sample/algorithmic-latency assertions;
- effective history: known-class and predicted-class gradient-by-age plus at least one segment-occlusion measure on held-out clips. Treat these as diagnostics, not accuracy.

### Decision rule

The proposed hypothesis is that `rf195` reduces the student's fidelity/capacity gap and may improve monolingual or prefix generalisation. A numeric gain cannot be responsibly predicted from existing evidence. Adopt it only if it improves fixed external-validation language-macro accuracy by at least 2 absolute points **or** teacher agreement by at least 5 points, with no new switch miss, no more than 100 ms worse student-excess stable lag, no higher unmatched churn, and six-thread frontend-inclusive RTF at most 0.02. These thresholds are proposed and **unverified**.

Do not call lower absolute switch lag an expected result. The current teacher's dense target transition is already around 1.26 s late on the two mirrored synthetic clips. A successful `rf195` arm may only reduce student excess error above that floor. If it produces no gated gain, retain `rf127` and document the 675 ms compression as intentional.

## 7. Concrete clip-start test

Alongside the architecture arms, replay one target-weighting diagnostic without changing the model:

1. current one-second evidence ramp;
2. ramp multiplied by the fraction of the requested 1.75 s teacher past that contains real samples;
3. hard mask until the teacher has its full requested past.

Store `teacher_requested_start_sample`, `teacher_real_start_sample`, `left_pad_samples`, and `real_past_fraction` per anchor. Report first correct/stable source-language decision, coverage-weighted KD, speech-only accuracy, and later switch metrics separately. Expected gain is **unverified**; the hard mask may worsen time to first decision and is a diagnostic, not the recommended default.

## 8. Source and command record

Primary external sources, all accessed 2026-09-26:

- [SpeechBrain ECAPA model card and pinned files](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9) and [SpeechBrain 1.0.3 ECAPA implementation](https://github.com/speechbrain/speechbrain/blob/v1.0.3/speechbrain/lobes/models/ECAPA_TDNN.py).
- [Jia et al., *A Compact End-to-End Model with Local and Global Context for Spoken Language Identification*](https://www.isca-archive.org/interspeech_2023/jia23b_interspeech.html), Interspeech 2023.
- [Yu and Koltun, *Multi-Scale Context Aggregation by Dilated Convolutions*](https://arxiv.org/abs/1511.07122), ICLR 2016.
- [Lea et al., *Temporal Convolutional Networks for Action Segmentation and Detection*](https://openaccess.thecvf.com/content_cvpr_2017/html/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.html), CVPR 2017.
- [Luo et al., *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks*](https://papers.nips.cc/paper/6203-understanding-the-effective-receptive-field-in-deep-convolutional-neural-networks), NeurIPS 2016.
- [Shim et al., *Knowledge Distillation from Non-streaming to Streaming ASR Encoder using Auxiliary Non-streaming Layer*](https://www.isca-archive.org/interspeech_2023/shim23_interspeech.html), Interspeech 2023.
- [Shi et al., *Emformer*](https://arxiv.org/abs/2010.10759), ICASSP 2021.
- [Wang et al., *Attentive Temporal Pooling for Conformer-based Streaming Language Identification*](https://arxiv.org/html/2202.12163v4), Odyssey 2022.

Local checks used six Torch threads and changed no model, cache, result, or pipeline file:

```text
rg/sed audit of src/streaming_lid/{model,config,audio}.py,
scripts/teacher_targets.py, reviewer findings, and experiment reports

instantiate R={127,195,255} variants; compute exact parameters/state;
15-repeat random 800-frame streaming_forward microbenchmark

read-only known-class input-gradient audit on seven held-out clips;
checkpoint SHA-256 bb2a7e2a...6c962e8

Hugging Face model API revision/license/date queries for the four Hub IDs
```

No student was trained, and no accuracy, agreement, or latency gain is claimed.
