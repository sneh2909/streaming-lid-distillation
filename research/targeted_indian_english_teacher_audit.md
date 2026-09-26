# Focused teacher audit for the Edge Indian-English failures (METHOD-47)

Research cutoff: **2026-09-26 (IST)**. Web sources and Hugging Face Hub/API
metadata were checked on that date. This note researches one question only:
how to compare replacement teachers on the four English training utterances
that the current ECAPA teacher labels as Hindi or Gujarati without selecting a
new teacher on four already-known errors.

Evidence labels used below:

- **Observed** means rederived from a retained project artifact or pinned source.
- **Publisher-reported** means stated by a model or dataset publisher but not
  reproduced here.
- **Proposed / unverified** means a future experiment, threshold, or expected
  direction. No teacher inference was run for this note.

## Bottom line

Run the comparison, but treat the four failures as a **screen**, not an
estimate of teacher quality.

1. The best production-eligible challenger to test first is pinned
   `openai/whisper-small` (Apache-2.0). The community AmberNet port is more
   attractive computationally, but it is eligible only after numerical parity
   with NVIDIA's official artifact and legal review of the NVIDIA NGC terms.
2. Pinned MMS-LID-126 and the newer Indic-specific Voxlect Whisper-small model
   are useful diagnostic arms. Both are unsuitable production recommendations
   for this project as currently licensed: MMS is CC-BY-NC-4.0 and Voxlect's
   card explicitly excludes commercial use.
3. Voxlect is particularly relevant because its 23 labels include Indian
   English and all seven project languages. It is not a drop-in CPU model: its
   pinned external loader calls `.cuda()` unconditionally, its Hub repository
   depends on separately cloned code and a mutable base-model resolution, and
   its training recipe discarded audio under three seconds. A patched,
   source-pinned run can be a research diagnostic only.
4. The current ECAPA result is not a general English failure. On the pinned
   700-row FLEURS validation cohort it gets 100/100 natural-full English and
   96/100 at two seconds, yet it gets only 1/5 selected-seven correct for the
   `en-IN-PrabhatNeural` training voice. Svarah is therefore the necessary
   Indian-accented natural-speech control.
5. All four selected failures are shorter than four seconds. Their fixed 4 s
   score is **not evaluable** without waveform padding. Compare true leading
   1 s and 2 s prefixes and natural full audio; never turn padding into evidence
   of short-utterance accuracy.
6. Expected teacher or student gain is **unverified**. A challenger that fixes
   only the synthetic voice should trigger a data/label repair, not a global
   teacher change. A global change additionally needs all-language, natural
   Indian-English, narrowband, switch-target, CPU, and license gates.

## 1. Exact failure being investigated

The local observations are bound to:

- `data/generated/manifest.jsonl`, SHA-256
  `7df6fc5d94db4c88cac9df6b2de8bc9dc5fcacf1bf36853f839bf494ad59314c`;
- `results/teacher_metrics.json`, SHA-256
  `ee00866264539e66a3204deca793ce7395f276a2e962ad53e8f677fcc466e327`;
- pinned ECAPA revision
  [`0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9),
  last modified **2024-11-27** and rechecked **2026-09-26**.

The five Edge English training clips use the same
`en-IN-PrabhatNeural` voice. The first is the sole selected-seven success; the
other four are the post-selected failures.

| Clip | Duration | ECAPA native top-1 | T=1 selected-seven `P(en)` | T=2 selected-seven `P(en)` | Selected-seven retained mass | WAV SHA-256 |
|---|---:|---|---:|---:|---:|---|
| `en_train_05` | 4.7072 s | Punjabi | 51.9920% | 39.8440% | 57.5988% | `886f709e…6feaf` |
| `en_train_06` | 3.9992 s | Hindi | 0.2895% | 3.3747% | 89.4445% | `cfebde98…72eab` |
| `en_train_07` | 3.9506 s | Hindi | 18.5330% | 27.3978% | 66.0490% | `6e2e775e…23baf` |
| `en_train_08` | 3.7146 s | Hindi | 0.1886% | 2.8550% | 72.4089% | `b0dbb50a…02c7` |
| `en_train_09` | 3.0391 s | Gujarati | 0.0812% | 1.9544% | 76.7952% | `abd32e1c…c0e1` |

The retained mass rules out a simple conditionalisation artifact for the four
errors: 66.0%--89.4% of ECAPA's native probability already lies inside the
seven project languages and points to Hindi or Gujarati. Conversely,
`en_train_05` shows why both views are mandatory: English wins after restriction
even though Punjabi wins in ECAPA's native 107-way space.

Across all ten English training clips ECAPA is selected-seven correct on 6/10;
the five gTTS English clips are 5/5 and these five Edge clips are 1/5. All other
six languages are 10/10 selected-seven correct, so the current 70-clip baseline
is 66/70 with a 60/60 non-English control. Association with provider/voice is
observed; whether the cause is accent, renderer acoustics, text, or their
interaction is **unverified**.

### Existing external counter-evidence

The retained external-validation result (SHA-256
`90d511775dcc6ff0ae49b7ef6e380eb4d6d573d57f0affc269f812beee1bb397`)
scores the same ECAPA revision on 100 pinned FLEURS validation utterances per
language:

| View | English recall | Hindi recall | Seven-language macro accuracy | Notes |
|---|---:|---:|---:|---|
| 0.5 s prefix | 66/100 | 24/100 | 20.14% | extremely low selected-space retained mass overall |
| 1 s prefix | 69/100 | 40/100 | 37.71% | true prefixes, no short-clip padding |
| 2 s prefix | 96/100 | 77/100 | 75.29% | all 700 eligible |
| 4 s prefix | 99/99 | 85/99 | 87.40% | four short rows excluded globally |
| Natural full | **100/100** | 95/100 | **97.71%** | 684/700 overall |

The English 1 s/2 s/full composite is 88.33%. These values are **observed
diagnostics**, not a clean confirmation result: FLEURS validation has already
been examined, and the cached ECAPA predictions have the open BUG-18 producer
provenance defect. FLEURS `en_us` is also not an Indian-English accent test.
Fresh contender scoring must use a new producer-bound cache; the old result
must not be silently mixed with new candidate rows.

## 2. Candidate set and evidence boundary

Every candidate below produces an utterance-level distribution after temporal
pooling or language-token scoring. None exposes a validated frame-level LID
trajectory. Prefix/window targets would be obtained by repeatedly applying the
utterance classifier to independently defined audio views.

| Candidate and exact Hub revision | Architecture / parameters / native output | Project-language coverage and short-clip evidence | CPU and 8 kHz evidence | License and disposition |
|---|---|---|---|---|
| [`speechbrain/lang-id-voxlingua107-ecapa@0253049…e8e9`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9) | ECAPA-TDNN with attentive statistics pooling; **21.25 M** locally measured parameters; 107-way utterance output | Contains all seven. Historical current-corpus baseline is 66/70; old held-out bake-off was 18/21 at 1 s and 19/21 at 2 s. | Historical project wall RTF **0.0258**. Its 8 kHz resample score was 54/63 across three durations. | Apache-2.0. Baseline and production-eligible. |
| [`openai/whisper-small@973afd…6b4d`](https://huggingface.co/openai/whisper-small/tree/973afd24965f72e36ca33b3055d56a652f456b4d) | Whisper encoder-decoder; language-token distribution from `detect_language`; **241.73 M** locally measured parameters; 99 language tokens | Contains all seven. Old held-out bake-off was 12/21 at 1 s and 13/21 at 2 s, on different audio. OpenAI's implementation masks non-language tokens before normalising ([source inspected 2026-09-26](https://github.com/openai/whisper/blob/main/whisper/decoding.py)). | Historical project wall RTF **0.4088** (15.9x ECAPA). Old 8 kHz resample total was 39/63. | Apache-2.0. Main production-eligible challenger. |
| [`facebook/mms-lid-126@53da6e…09a4`](https://huggingface.co/facebook/mms-lid-126/tree/53da6e311f3ce48f324fe335924216187e3109a4) | wav2vec 2.0/MMS encoder plus 126-way LID head; **966.09 M** locally measured parameters | Contains all seven. Old held-out bake-off was 16/21 at 1 s and 17/21 at 2 s. The official [MMS paper](https://www.jmlr.org/papers/v25/23-1318.html) was published in **JMLR 2024**. | Historical project wall RTF **0.2175** (8.4x ECAPA). Old 8 kHz resample total was 50/63. | CC-BY-NC-4.0. Research diagnostic only; commercial teacher use is not assumed permissible. |
| [`surogate/ambernet-langid@3b7b3f…1d5b`](https://huggingface.co/surogate/ambernet-langid/tree/3b7b3fbfcc51753745f1d9abd107c9530c391d5b) | Community conversion of AmberNet: depthwise-separable convolutions, squeeze-and-excitation, x-vector statistics pooling; **28.9 M** publisher-reported parameters; 107-way logits | Same VoxLingua107 label set, including all seven. The official [NVIDIA NGC entry](https://catalog.ngc.nvidia.com/orgs/nvidia/nemo/models/langid_ambernet) reports 5.22% error on 1,609 verified utterances over 33 languages; transfer to these clips is **unverified**. Its card warns that accuracy degrades below about 5 s and for heavy accents/code-switching. | Community card reports ONNX RTF 0.014 on four CPU threads (10 s in 138 ms), **not locally verified**. Card declares 8 kHz/telephone-band speech out of domain. | HF metadata says `license: other`; card points to NVIDIA NGC terms. Candidate only after official-artifact parity and legal clearance. |
| [`tiantiaf/voxlect-indic-lid-whisper-small@1792f3…a916`](https://huggingface.co/tiantiaf/voxlect-indic-lid-whisper-small/tree/1792f34d4a5c71d9a61583aaf98b7226eaaaa916) | Whisper-small encoder representations plus pointwise convolution/pooling/classifier; **90.44 M** parameters in the Hub weights; 23-way utterance output | Explicit Indian-English class plus all seven project languages. [Voxlect](https://arxiv.org/html/2508.01691), posted **2025-08-03**, reports 75.0% accuracy / 0.748 F1 for its 23-way Whisper-small condition. This is not comparable to the local cohorts. Training filtered clips under 3 s; 1 s/2 s results are out-of-recipe diagnostics. | CPU cost and 8 kHz behavior are **unverified**. The pinned [loader source](https://github.com/tiantiaf0627/voxlect/blob/271ae21e8d54947941874a172d32ed2ea2ebf479/src/model/dialect/whisper_dialect.py#L214-L239) calls `.cuda()` unconditionally, contradicting its advertised CPU selection. | Structured metadata says OpenRAIL; the card explicitly says **no commercial use**. Patched, source-pinned research arm only. |

Parameter counts and RTFs for the first three are local historical
measurements, not claims from model cards. That bake-off timed preprocessing
and inference over 360 s of requested audio with model-specific batch sizes,
excluded checkpoint loading, and used six Torch threads. It remains useful for
budgeting but must not be presented as single-request latency.

The exact Whisper and MMS repositories were last modified **2024-02-29** and
**2023-06-13** respectively; the AmberNet port was last modified
**2026-08-13**; Voxlect was last modified **2025-08-10**. These live Hub values
were rechecked on **2026-09-26**. The three historical loaders did not enforce
their recorded revisions; the future audit must use verified local snapshots,
as specified in `research/teacher_bakeoff_artifact_pinning.md`.

### Why Voxlect is informative but not yet runnable as advertised

The exact Hub card names IndicVoices and Common Voice 11 as training datasets,
lists all 23 labels, requires 16 kHz mono audio, and says the training pipeline
discarded audio under 3 s and truncated audio over 15 s. The [pinned GitHub
source](https://github.com/tiantiaf0627/voxlect/tree/271ae21e8d54947941874a172d32ed2ea2ebf479)
must be part of the artifact identity because the Hub model repository does not
contain a standalone Transformers `auto_map` implementation. The wrapper also
resolves `openai/whisper-small` separately without an immutable revision and
forces extracted features to CUDA at lines 230 and 239. A valid experiment
would need to:

1. pin and hash both repositories and every loaded file;
2. replace device-forcing with explicit device propagation and bind the patch;
3. demonstrate offline load and deterministic logits on a fixture;
4. mark 1 s/2 s results out of the model card's training-duration support; and
5. keep every result `research_only=true` because license eligibility fails.

All four failed local clips are at least 3.0391 s, so their natural-full view is
barely inside the card's advertised duration filter. That does not validate the
model on them; it merely avoids the known under-3-second exclusion.

### Newer Indic model screened out

[`panchajanya-ai/lid_Indic_Vaani@338cd6…5dcf`](https://huggingface.co/panchajanya-ai/lid_Indic_Vaani/tree/338cd6ab84ec3bcc94ed5afe2ea8998b48525dcf)
(last modified **2025-05-22**, checked **2026-09-26**) is an Apache-2.0
five-class ECAPA model for Marathi, Hindi, English, Tamil, and Telugu. It omits
Bengali and Gujarati and its card supplies neither evaluation metrics nor a
reconstructable dataset/split provenance. It can answer neither the seven-way
regression question nor the Gujarati confusion safely, so it should not consume
the main audit budget. Its accuracy on the four clips is **unverified**.

## 3. Why the old aggregate bake-off cannot decide this question

The historical report ranked ECAPA first: clean selected-seven totals were
58/63 for ECAPA, 54/63 for MMS, and 42/63 for Whisper. That is insufficient for
METHOD-47 for four independent reasons:

1. It used 21 held-out synthetic clips rather than today's four failed training
   clips or a natural Indian-English corpus.
2. Its input fingerprint no longer matches the current corpus. A rerun is a new
   experiment, not a missing row appended to the old result.
3. It symmetrically zero-padded 15/21 clips for the 4 s condition. The four
   selected failures have **zero** legitimate 4 s observations.
4. The failures were selected because ECAPA is wrong. A contender's 4/4 score
   would be useful debugging evidence but a biased estimate of relative
   accuracy. Candidate selection needs prespecified controls and a later
   untouched Indian-English check.

The old 8 kHz condition is also only a 16 kHz -> 8 kHz -> 16 kHz bandwidth
bottleneck, not a telephone codec/noise/channel study. It is a paired
narrowband diagnostic, not production-call evidence.

## 4. A staged experiment that fits the project constraints

### Stage 0 — eligibility before inference

For each model, bind the full revision, exact file table and hashes, loader and
preprocessor source, dependency versions, native label order, selected-seven
mapping, sample-rate conversion, batching/masking recipe, thread settings, and
output digest. Load only the verified local snapshot after prefetch.

- AmberNet must match the official NVIDIA artifact on a deterministic fixture:
  exact top-1 plus maximum absolute posterior difference at most `1e-4` is a
  **proposed, unverified project tolerance**, not a publisher guarantee.
- Voxlect must pass a CPU fixture after the pinned device patch and must not
  resolve GitHub or the base Whisper model at runtime.
- Record `commercial_use_status` as `allowed`, `prohibited`, or
  `legal_review_required`. Do not infer that distilling a non-commercial
  teacher makes the resulting student commercially unrestricted.

### Stage 1 — candidate screen on already-consumed development data

Run all eligible candidates on:

| Cohort | Purpose | Required rows |
|---|---|---:|
| Four ECAPA failures | direct rescue screen | 4 |
| All Edge English training | expose tradeoff against the one success | 5 |
| All monolingual training | seven-language regression control | 70 |
| Pinned FLEURS validation English | generic-English duration control | all 100 frozen rows |
| Pinned FLEURS non-English sentinel | detect an English-everywhere challenger | 100 rows, outcome-blind hash-ranked and balanced 16/17 across the other six languages |

Use FLEURS revision
[`70bb2e84b976b7e960aa89f1c648e09c59f894dd`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd),
last modified **2026-05-15** and checked **2026-09-26**. This is validation,
not the locked test set. BUG-16/17/18 source, release, and prediction-cache
repairs should precede the scored run; otherwise label every output
`diagnostic_provenance_incomplete=true`.

For every row, score the first 1 s, first 2 s, and natural full utterance.
Create a paired bandwidth diagnostic by resampling each exact view
16 kHz -> 8 kHz -> 16 kHz. Do not extend the waveform to a requested duration;
short rows are excluded with explicit denominators. Canonical model-internal
frontend padding is allowed only when it is part of the pinned implementation
and true audio length/mask semantics are preserved and recorded.

Fixed 4 s may be reported for other eligible rows, but for the four selected
failures it must be `n_eligible=0`, `metric=null`, `reason=all_sources_shorter`.

### Stage 2 — one frozen natural Indian-English confirmation

Before seeing Svarah outcomes, freeze **at most one commercially eligible**
challenger from Stage 1, its thresholds, and ECAPA as comparator. Then evaluate
both on an outcome-blind 200-utterance Svarah slice at 1 s, 2 s, and natural
full, clean and resampled narrowband.

[`ai4bharat/Svarah`](https://huggingface.co/datasets/ai4bharat/Svarah) is an
Indian-accented English ASR evaluation corpus. Its [Interspeech paper](https://www.isca-archive.org/interspeech_2023/javed23_interspeech.html)
was published in **2023**; the Hub revision observed on **2026-09-26** was
`ebbf7777fe771490696a3f7b007097606fa8c924`, last modified **2025-03-10**,
gated, with 6,656 examples, approximately 9.6 hours, 117 speakers, 65 districts,
and 19 states reported by the card/paper.

Freeze the 200 rows by stable source identity plus SHA-256 rank and bind audio
hashes after access. Stratify outcome-blind by published geography/primary
language and duration where fields are available. The live Hub schema inspected
for this note did not expose a stable speaker identifier, so speaker-disjoint
sampling and speaker-clustered uncertainty are **unverified**. Do not claim
`n=200` independent speakers or use an utterance-only interval as
speaker-generalisation evidence. The card prose states CC BY 4.0, while the
live structured dataset API returned no machine-readable license; retain the
exact card/license evidence in the release lock before use.

If no Stage-1 candidate is frozen, do not spend Svarah. If the frozen candidate
fails, do not try successive candidates on the same 200 rows while continuing
to call the slice confirmation data. The remaining Svarah utterances are not
automatically independent confirmation data because speaker identities and
overlap cannot currently be enforced.

### Stage 3 — only after the teacher and training design are frozen

Generate versioned alternative target caches, train a matched student, and run
the full local causal/switch gates. Keep the 5,063 selected-language FLEURS test
rows locked until the teacher, target recipe, student objective, checkpoint
selector, and analysis are frozen. FLEURS test is useful monolingual
confirmation, not a substitute for speaker-identified Indian-English and
natural code-switch evaluation.

## 5. Required row-level outputs

Store enough information to rederive every aggregate:

- cohort/source revision, clip ID, audio SHA-256, true duration, requested view,
  actual sample count, eligibility, and transformation identity;
- native label list/hash, native T=1 posterior vector, native top-1, and
  true-label probability;
- absolute probabilities for the seven project languages, their retained mass,
  conditional seven-way T=1 posterior/top-1, and the exactly derived T=2
  posterior that target generation would consume;
- entropy, maximum probability, model/preprocessor/artifact identity, batch
  recipe, and output-payload digest;
- load time, warm and cold wall time, process CPU time, peak RSS, thread and
  affinity settings, processed audio seconds, batch size, and wall RTF.

Native and conditional winners answer different questions. The primary rescue
screen must require native English, not merely English after deleting most of a
candidate's probability mass. Probability magnitudes across 23-, 99-, 107-,
and 126-class heads are not directly calibrated; report them, but do not rank
teachers solely by raw `P(en)`.

## 6. Predeclared gates and decision rules

The numerical margins below are **proposed project smoke gates**, not published
standards, and their usefulness is unverified until applied.

1. **Artifact and rights:** fail closed on a missing pin/hash/map, AmberNet
   parity failure, mutable Voxlect dependency, or non-production license.
   MMS/Voxlect may remain clearly separated diagnostic rows but cannot win the
   production selector.
2. **Direct rescue screen:** require native-English natural-full predictions on
   at least 3/4 selected failures and selected-seven correctness on at least 4/5
   Edge English clips. Report exact counts; do not attach a population-accuracy
   interpretation to these post-selected rows.
3. **Local regression:** require at least the ECAPA baseline 66/70
   selected-seven natural-full results and no new error among the 60/60
   non-English training controls. A challenger that trades another project
   language for English does not solve the target-quality problem.
4. **FLEURS development:** against freshly rescored ECAPA on identical rows,
   require English natural-full change at least `-2 pp`, 1 s and 2 s changes at
   least `-5 pp`, and no more than `+3 pp` extra false-English predictions on
   the non-English sentinel. Report paired discordances, not only accuracy.
5. **Svarah confirmation:** require an observed paired gain of at least `+5 pp`
   for both natural full and 2 s English recall, with exact discordant counts.
   Because speaker clustering is unavailable, this is a project gate rather
   than a confidence claim. Its expected direction is **unverified**.
6. **Narrowband and cost:** require no more than an additional `5 pp`
   degradation versus ECAPA on Stage-2 natural-full/2 s views. Run each
   experiment in under 45 minutes, without swap/OOM, and publish both throughput
   and single-item latency. Historical RTFs are budgeting priors only. Teacher
   cost affects cache iteration, not deployed student latency.
7. **Switch and student:** do not replace the local-window switch teacher from
   monolingual results. First train a matched candidate that changes only the
   monolingual teacher cache and leaves the existing switch trajectory fixed;
   then require the repository's causality, availability, held-out,
   switch-detection, flip-flop, and CPU gates. A later global teacher change
   needs a separate window-level switch audit.

Do not choose the teacher independently per labelled clip or substitute a hard
label only when a teacher is known to be wrong while still calling the result
pure KD. That is an oracle-labelled objective. If known synthetic labels are to
correct wrong teacher outputs, test the already-defined hybrid/gated-loss design
and name it accordingly.

### Interpretation matrix

| Outcome | Interpretation and next action |
|---|---|
| Challenger fixes the four clips but not Svarah | Likely synthetic voice/text interaction; improve synthesis diversity or use an explicitly labelled loss. No teacher switch. |
| Challenger improves Svarah but regresses another language | Teacher tradeoff, not a seven-language solution. Investigate calibrated ensemble/distillation only with a new prespecified experiment. |
| Only MMS or Voxlect passes quality gates | Evidence that model/domain choice matters, but not a commercially usable teacher. Use it to guide architecture/data research; do not adopt. |
| Whisper or legally cleared/parity-verified AmberNet passes every teacher gate | Build a versioned alternative monolingual target cache and matched student; adoption still depends on downstream causal, switch, external, and CPU results. |
| No challenger passes | Retain ECAPA and treat the four known labels through data/hybrid-loss work. |

## 7. Reproducible acceptance tests

- **Duration boundary:** the exact four durations above produce 4/4 eligible at
  1 s and 2 s, 4/4 at natural full, and 0/4 at fixed 4 s. Rounding 3.9992 to 4
  or zero-padding must fail.
- **Native versus restricted:** a fixture whose native winner is outside the
  seven project labels must preserve that winner while independently deriving
  the seven-way conditional winner and retained mass.
- **Pinning:** changing any model, processor, loader, label-map, source-patch,
  or audio hash invalidates reuse before inference.
- **Mapping:** assert exact unique mappings for all seven languages and reject a
  missing Bengali/Gujarati class; this intentionally screens out the
  Panchajanya five-class model.
- **Amber parity:** perturb one official/community logit enough to change the
  top-1 or exceed the declared tolerance and reject the port.
- **Voxlect CPU:** monkeypatch CUDA availability false; the patched pinned
  loader must complete an offline fixture without `.cuda()` or network calls.
- **No summary trust:** alter a stored row posterior and self-rehash only the
  summary; pure rederivation must change the payload identity and aggregates.
- **Cohort role:** attempting a second contender after reading Svarah outcomes
  must relabel that slice development data and keep confirmation false.
- **Rights:** an otherwise perfect MMS/Voxlect row remains
  `production_eligible=false`; accuracy cannot override the license gate.
- **Timing:** a timing row missing batch size, audio seconds, clock scope,
  threads, or peak RSS cannot satisfy the CPU gate.

## Conclusion

METHOD-47 should answer a diagnostic hierarchy, not “which model gets four
clips right?” First determine whether a challenger repairs the full Edge voice
without damaging the other 60 training examples. Then freeze one eligible
challenger and ask whether the gain appears on untouched natural Indian English.
Only after that should a matched student be trained. On current evidence ECAPA
remains the production baseline; every claimed replacement gain is
**unverified**.
