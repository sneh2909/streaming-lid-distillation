# Clean-teacher / degraded-student KD for telephony robustness

**Research date:** 2026-09-26

**Scope:** Topic 6, METHOD-28 only

**Question:** after the failed 8 kHz/channel matrix, how should a frozen clean
ECAPA teacher supervise the causal student on paired narrowband/G.711 views?

## Bottom line

Clean-teacher/degraded-student knowledge distillation is a legitimate paired
domain-adaptation method, but the supporting literature is mostly large-scale
ASR. I found no reviewed Hugging Face spoken-LID artifact that documents this
training scheme with causal/frame-level output, Hindi and English, and 8 kHz or
G.711 evaluation. Its benefit here is therefore **unverified**.

Run METHOD-28 as a controlled diagnostic, not as a robustness claim:

1. Start with only the validated `narrowband_pcm`, `pcma`, and `pcmu` views.
   Keep a substantial clean fraction; defer additive-noise and noise-plus-codec
   arms until channel-only training works.
2. Compare three arms from identical initialization and minibatch/view order:
   clean input + clean target, degraded input + same-degraded-view target, and
   degraded input + clean target. The last comparison is required to isolate
   the value of the *clean target*, rather than merely the value of channel
   augmentation.
3. Bind source-sample time to received-sample time. The validated causal FIR
   delays content by exactly 128 samples at 16 kHz (8 ms); equal tensor lengths
   do not make clean targets and degraded features simultaneous.
4. Treat the current 71-clip, 318.8755-second (5.315-minute) synthetic corpus as
   a plumbing/pilot dataset. A positive result still needs natural,
   speaker-disjoint development data and the locked real-call benchmark.
5. Do not evaluate the switch gate when the clean control detects 0/2 switches.
   `no additional miss` is not a pass when there is no valid baseline event.

The current experiment harness under `experiments/cross-view-kd/` is a useful
two-arm pilot. At research time it samples four views uniformly, so its expected
mix is 25% clean and 75% degraded, retains clean targets, and compares only a
clean-input control with the cross-view treatment. If it has already launched,
preserve it as an immutable pilot; add the matched-view teacher arm in a new
run rather than rewriting its provenance.

## 1. Why METHOD-28 is triggered, and why the result can still mislead

The completed [telephony robustness report](../experiments/telephony-robustness/REPORT.md)
uses 21 held-out clips at leading 1/2/4-second prefixes. Those are 63 correlated
requests, not 63 independent speakers.

| Condition | ECAPA macro-F1 | Student macro-F1 | Student paired flips vs clean |
|---|---:|---:|---:|
| Clean | 90.36% | 9.63% | 0/63 |
| Resample only | 77.87% | 10.12% | 30/63 |
| Narrowband PCM | 80.83% | 3.96% | 28/63 |
| G.711 A-law | 78.10% | 3.96% | 29/63 |
| G.711 mu-law | 79.40% | 4.02% | 29/63 |

This is a strong trigger for a paired adaptation experiment, but not evidence
that the student is almost robust. The clean student is already at a severe
absolute floor and fails to establish the source language on both switch
clips. Small relative recoveries can therefore reward a collapsed predictor.
The original METHOD-28 gate—recover at least half the paired channel loss,
lose at most two clean macro-F1 points, and add no switch miss—must be reported,
but it is insufficient by itself for model promotion.

The training manifest contains 71 examples and 318.8755 seconds of audio. Its
speech was synthesized through remote TTS and decoded from MP3 to PCM. Adding
G.711 creates a compound lossy path and does not reproduce microphone,
telephone-network, speaker, or room variation. This pilot can answer whether
the optimization path is wired correctly; it cannot estimate real-call
robustness.

## 2. What the method is

For a paired source utterance `x_clean` and deterministic channel view `g_c`,
let the frozen teacher produce `q_i` from clean audio and let the causal student
produce `p_j` from `x_c = g_c(x_clean)`. The initial treatment should retain the
existing temperature, masks, label delay, architecture, and output-only KL:

```text
L_cross = T^2 * sum_i m_i w_i KL(q_i(clean; T) || p_j(i,c)(degraded; T))
          / sum_i m_i w_i
```

Here `j(i,c)` is not merely `i + D`: it is the earliest student output whose
received degraded audio contains at least the clean source samples used by the
teacher target. `m_i` must enforce that availability relation. This is a
paired-view objective, not ordinary noisy-data augmentation.

[Li et al., submitted 2017-08-17](https://arxiv.org/abs/1708.05466), formulate
teacher/student domain adaptation with synchronized clean/noisy pairs: the
teacher consumes the source-domain sample, the student consumes the paired
target-domain sample, and framewise KL transfers teacher posteriors without
target-domain transcripts. Their paper reports gains with 375 and 3,400 hours
of parallel data and emphasizes synchronization. It is ASR evidence, not LID,
and even the smaller corpus is roughly 4,200 times the duration available here.

[Mošner et al., submitted 2019-01-05, revised 2019-03-15](https://arxiv.org/abs/1901.02348)
apply the same clean-teacher/noisy-student idea to ASR with up to 8,000 hours of
untranscribed parallel data. Their best sequence model reports relative WER
reductions of 10.1%, 28.7%, and 19.6% on clean, simulated-noisy, and real test
sets. Those values are author-reported ASR results at a scale about 90,000
times this project's pilot data; they are not expected LID gains.

A recent multilingual example points in the same direction. [*Listen Like a
Teacher*, published 2026-03-14](https://ojs.aaai.org/index.php/AAAI/article/view/40614)
uses clean-teacher/noisy-student Whisper distillation for multilingual ASR,
including Hindi and English, and reports a noisy teacher as inferior in its
setup. It uses representation, attention, and token losses on large Whisper
models and GPU training, so transfer to a 42,567-parameter causal LID student
is **unverified**.

## 3. Direct Indic-LID evidence supports codec diversification, not this KD claim

[Dey, Sahidullah, and Saha, submitted 2023-02-10](https://arxiv.org/abs/2302.05110)
study cross-corpus Indian LID with Bengali, Hindi, Punjabi, Tamil, and Urdu.
Their 16 kHz IIITH source and 8 kHz LDC/KGP targets make the domain question
relevant. Audio-encoding augmentation randomly uses PCM, A-law, mu-law, or
ADPCM. In their ECAPA-TDNN experiment, the encoding arm changes EER/Cavg from
9.74/11.51 to 8.93/9.01 in-domain, from 42.43/44.82 to 39.32/36.44 on LDC,
and from 34.83/32.62 to 32.36/29.98 on KGP.

This is the closest task evidence found, but it is supervised utterance-level
LID, not frame-level KD or switching. Their augmentation fold-factor sweep also
shows that more augmented data is not monotonically better; performance often
degrades beyond a factor of three, and they choose factor two as a
performance/compute compromise. The existing pilot's expected degraded/clean
ratio is exactly three. A conservative first confirmatory run should use 50%
clean plus 1/6 each narrowband PCM, A-law, and mu-law (augmented/clean ratio
one). That schedule is a proposed engineering control and its advantage is
**unverified**.

There is also contrary evidence against assuming that a clean target is always
best. [Park et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/park21_interspeech.html)
find that, for streaming keyword spotting with aggressive spectral
augmentation, applying the augmentation to both teacher and student supports
large gains on difficult conditions. KWS and SpecAugment differ from
language-preserving telephony codecs, but this result is why the same-view
teacher control is necessary.

## 4. Exact alignment contract

The validated narrowband transform uses a causal 129-tap FIR at 8 kHz. Its
group delay is 64 samples at 8 kHz, or 128 samples / 8 ms on the 16 kHz project
clock. The transformed array is kept at the original length, which pads the
beginning and cuts an equal amount of late content. A frame-count equality
assertion cannot detect this semantic shift.

For every target/output pair, record:

- clean source interval used by the teacher target, including its window and
  right context;
- degraded received interval used by the frontend/student output, including
  FIR delay, frontend frame width, model lookahead, and any chunk wait;
- `target_latest_source_sample`, `student_latest_received_sample`, and the
  received-to-source mapping for the selected view;
- target expansion method and exact clean/degraded waveform, transform, target,
  and source-code hashes.

Supervision is valid only when the student evidence includes all clean source
samples used by the target. With a 160-sample/10 ms hop, the 128-sample FIR
delay generally moves the corresponding output to the next grid point; this
one-frame rule is only a conservative approximation, with a 2 ms overshoot.
The sample ledger is authoritative at padded starts, clipped ends, and switch
boundaries.

Do not fix alignment by deleting the first 128 output samples or advancing the
degraded waveform. That would give the offline synthetic path information
earlier than a live causal channel. Report both clocks instead:

- **received-audio latency:** model/policy delay after the degraded sample is
  available; the upstream FIR delay is excluded;
- **clean-source-origin latency:** received-audio latency plus the channel's
  8 ms content delay.

The existing main target cache still needs METHOD-0's availability proof.
Until that ledger passes, restrict the pilot's claim to monolingual
classification, or use a separately identity-bound previous-anchor-hold cache
and label switch findings experiment-only.

## 5. Controlled experiment

### Arms

| Arm | Student input | Teacher target | What it answers |
|---|---|---|---|
| A: clean control | clean | clean | exact baseline trajectory |
| B: matched-view control | sampled clean/channel view | same sampled view | benefit of ordinary view augmentation/KD |
| C: cross-view treatment | same sampled clean/channel view as B | clean | incremental value of a clean target |

All three arms must share initial state, source minibatches and order, selected
view per example, total examples, optimizer, steps, temperature, masks, target
type, target hop, delay, router, and checkpoint selector. B and C must consume
byte-identical feature tensors; only their target tensors differ. Generate B's
teacher posteriors once and cache them under the same transform identities.

The key contrasts are:

- `C - A`: total effect of paired channel exposure while retaining clean
  targets;
- `C - B`: effect attributable specifically to the clean target;
- `B - A`: value of same-view channel augmentation under the frozen teacher.

Do not add hidden-state, attention, hard-label, confidence, or temperature
changes to the first confirmatory run. If C beats B, a later confidence-weighted
arm can downweight clean-teacher errors; its gain is **unverified**. If B beats
C, first stratify by transform severity and clean/degraded teacher agreement
before concluding that clean targets are harmful.

### View schedule

Use one precomputed deterministic schedule for B and C. Recommended first
confirmatory allocation:

```text
P(clean) = 0.50
P(narrowband_pcm) = P(pcma) = P(pcmu) = 1/6
```

Balance view counts within short blocks rather than relying only on an
asymptotic random draw, and record the ordered schedule hash. Keep
`resample_only` as an evaluation diagnostic, not a telephony training view.
Exclude additive noise initially. After a positive channel-only result, run a
separate severity study with 20/10 dB noise and noise-before-codec; do not
silently enlarge the first treatment.

### Data roles

1. **Local synthetic development:** retain the current 21 clips and paired
   1/2/4-second requests for regression diagnosis. Cluster any interval by
   source clip/voice, because prefixes are correlated.
2. **Natural development:** pin [`google/fleurs`](https://huggingface.co/datasets/google/fleurs)
   at revision [`70bb2e84b976b7e960aa89f1c648e09c59f894dd`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd),
   whose card/API reported CC BY 4.0 on 2026-09-26. Use only the seven project
   configurations' train data for any expanded training and validation data
   for model selection; lock test. Audit the exact decoded audio and speaker
   fields before calling it lossless or speaker-disjoint.
3. **Real calls:** keep pinned IndicTelephony-Bench test-only and outside
   tuning. Its single-language turns can test actual 8 kHz performance after
   the model and policy are frozen; its set-valued Hindi-English turns do not
   provide switch timestamps.

## 6. Predeclared reporting and gates

Report each arm on clean, `resample_only`, narrowband PCM, A-law, and mu-law.
For each view publish accuracy, language-macro F1, every class recall,
clip-clustered paired bootstrap intervals, clean-to-view KL, correct-to-wrong
and wrong-to-correct flips, target coverage, and per-view counts. A pooled
number must never replace the condition table.

For the existing METHOD-28 diagnostic, define the pooled recovery fraction as:

```text
R = (F1_C,degraded - F1_A,degraded)
    / (F1_A,clean - F1_A,degraded)
```

If the denominator is non-positive, record `recovery_gate_evaluable=false`;
do not call it a pass. Retain the backlog thresholds `R >= 0.5` and clean
macro-F1 loss `F1_A,clean - F1_C,clean <= 2 pp`. Additionally:

- publish `C - B`; a cross-view attribution requires C to beat B under the
  same schedule, with its interval and all per-language effects shown;
- do not promote an arm while any language has zero recall on natural
  validation or while clean natural performance fails to beat the exact clean
  control; these are diagnostic gates, not a claim that the resulting model is
  deployment-ready;
- set `switch_gate_evaluable=false` unless the clean control establishes the
  source state and detects the target transition on every required baseline
  trial;
- when evaluable, require no new miss, no more than 100 ms extra actionable
  lag on either clock, and no increased monolingual false-switch rate;
- do not treat the two mirrored synthetic switches as independent evidence.

Passing the relative recovery gate on the current 9.63%-macro-F1 clean student
is evidence that the intervention moved the chosen metric, not evidence of a
usable LID model. Model promotion remains blocked by absolute natural-data
quality and the independent-switch protocol.

## 7. Hugging Face Hub audit, 2026-09-26

The Hub API query for the `spoken-language-identification` task tag returned 12
models on the research date. This was a non-exhaustive tag/card audit; untagged
repositories may exist.

- [`ajju3513/whisper-medium-ala-hinglish-noise-robust`](https://huggingface.co/ajju3513/whisper-medium-ala-hinglish-noise-robust),
  revision `4c607a64850b78994810f51e8bd91acfb05a9592`, was created and
  last modified 2026-09-06. Its card describes parallel clean/noisy KD for
  Hindi-English code-switched ASR and self-reports noisy WER, making it the
  closest procedural artifact found. It is Apache-2.0, has 769,110,016
  parameters, requires custom code, uses Whisper sequence losses, and does not
  publish an identifiable evaluation dataset, 8 kHz/G.711 metrics, causal LID
  output, or CPU cost. Treat its card metrics as **unverified self-reports**, not
  evidence for this student.
- [`williamhtan/cld-whisper-small-5lang`](https://huggingface.co/williamhtan/cld-whisper-small-5lang),
  revision `15e1468106ced97f00230885f008c43d89b44f74`, last modified
  2026-06-02, is MIT-licensed and covers Hindi/English. It mean-pools frozen
  Whisper-small encoder states and trains an utterance classifier on Common
  Voice. Its card documents no codec/noise cross-view KD or streaming output.
- [`surogate/ambernet-langid`](https://huggingface.co/surogate/ambernet-langid),
  revision `3b7b3fbfcc51753745f1d9abd107c9530c391d5b`, last modified
  2026-08-13, packages a 28.9M-parameter utterance-level AmberNet and explicitly
  calls telephone-band 8 kHz audio out of domain. Its license is NVIDIA NGC
  terms rather than a standard open-source license.
- [`desert-ant-labs/ear`](https://huggingface.co/desert-ant-labs/ear), revision
  `77d6f5c173dc70c3b1bd35769409901f469b9e44`, last modified
  2026-09-24, covers the project languages but selects 30-second windows and
  reports one dominant language for multilingual audio. It has a custom
  source-available license and no documented 8 kHz/cross-view result.

No reviewed card supplied all of: Hindi and English, causal or frame-level LID,
paired clean/degraded distillation, G.711 or true 8 kHz evaluation, short-clip
results, and CPU cost. That negative finding is bounded by the query and cards
reviewed; it is not proof that no such model exists.

## 8. What remains unverified

- Any accuracy, calibration, switch-latency, or robustness gain from cross-view
  KD for this causal student.
- Whether clean ECAPA posteriors are better targets than same-view posteriors
  for each codec and each language.
- Whether the proposed 50/50 schedule is better than the pilot's 25/75 mix.
- Whether the 8 ms sample correction changes monolingual metrics; it matters
  conceptually and at boundaries even if aggregate classification is unchanged.
- Generalization from MP3-derived synthetic TTS to real 8 kHz calls.
- The transfer of large ASR/KWS results to tiny causal LID.
- Any switch conclusion before METHOD-0, chronology-safe scoring, and
  independent source/speaker boundary trials are complete.

## Source and command record

Sources were checked on 2026-09-26. Primary paper pages/PDFs were used for the
method claims; Hugging Face API metadata and revision-pinned cards were used for
Hub claims. Relevant endpoints include the [Hub task query](https://huggingface.co/api/models?filter=spoken-language-identification&sort=lastModified&direction=-1&limit=100)
and the [FLEURS dataset API](https://huggingface.co/api/datasets/google/fleurs).
Local read-only checks covered `.loop/BACKLOG.md`, the telephony report/results,
the current cross-view harness, the generated manifest, target/loss code, and
the researcher/experimenter logs. No model was downloaded, trained, or changed
for this research iteration.
