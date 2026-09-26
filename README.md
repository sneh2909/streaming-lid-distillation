# Offline-to-streaming language-ID distillation

This repository distills SpeechBrain's offline VoxLingua107 ECAPA language-ID model into a 42,567-parameter causal temporal-convolutional student. It is a deliberately small CPU experiment, not an accuracy claim: it performs real optimization on real audio waveforms, exercises a Hindi→English boundary, and makes the teacher/student timing contract explicit.

The selected languages are English (`en`), Hindi (`hi`), Marathi (`mr`), Bengali (`bn`), Tamil (`ta`), Telugu (`te`), and Gujarati (`gu`). Hindi↔English is the priority switch case.

## Reproduce from a clean checkout

Python 3.10 and network access are required for the first run. `gTTS` and `edge-tts` fetch the synthetic audio, and SpeechBrain downloads its checkpoint; no ffmpeg or GPU is used.

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export HF_HOME="$PWD/.cache/hf"
export TORCH_HOME="$PWD/.cache/torch"

uv sync --frozen
uv run python scripts/prepare_data.py
uv run python scripts/teacher_targets.py
uv run python scripts/train.py
uv run python scripts/eval.py
uv run pytest -q
```

The defaults reproduce the submitted run: 1,600 optimizer steps, batch size 7, learning rate `1e-3`, and CPU only. Steps, batch size, and thread count must be positive integers; the learning rate must be positive and finite. Incomplete final batches are dropped, so every update has the same batch size, and training rejects a batch size larger than the split instead of looping over an empty loader. On this 20-core/13 GB host the full cached run is comfortably under 30 minutes; network speed dominates a first run. `prepare_data.py --force` refreshes TTS audio. Each manifest row binds its text, language, provider, exact voice, synthesis options, client versions, audio-processing recipe, generator-source hash, and resulting WAV SHA-256. The script builds and validates a complete sibling staging corpus, publishes byte-addressed WAVs, and atomically replaces the manifest last; a failed refresh cannot leave the live manifest pointing at a partially updated corpus. Audio, teacher targets, caches, checkpoints, and `.venv` are ignored by Git.

The public train/eval commands are standard-library launchers before they import Torch or any project module. Each captures all `scripts/*.py`, the complete `streaming_lid` package, `pyproject.toml`, and `uv.lock` into a verified content-addressed tree under ignored data, then starts a fresh child with the staged `src` first and the editable live workspace removed from local import paths. Checkpoints bind that tree, the complete package-origin ledger, Python/dependency versions, and repeated pre-import/publication verification; evaluation runs from an independently captured byte-identical stage. A dirty run is labelled `captured_worktree`, not silently presented as a committed Git release.

Important outputs are:

- `results/summary.json`: required machine-readable summary, including every optimizer-step loss.
- `results/train_metrics.json`: loss and gradient history plus requested/completed update counts, post-update finiteness assertions, run identity, and checkpoint hash.
- `results/eval_metrics.json`: per-clip teacher agreement, known-label accuracy, speaker audit, exact per-call stable-emission ranges/clocks, policy settings, RTF, switch outcome, and the validated training/evaluation identity.
- `results/switch_plot.png`: offline teacher posteriors and emitted student chunk-EMA posteriors.
- `results/teacher_metrics.json`: target-generation and 94-file provenance audit, including the complete 70-clip per-class training-target ledger.

## Data and split

`scripts/prepare_data.py` synthesizes PCM WAVs at 16 kHz through gTTS and Microsoft Edge TTS, then decodes MP3 responses in-process with `miniaudio`. Cache reuse requires both an exact canonical recipe and matching WAV bytes; changing text, voice, options, client/generator identity, or audio bytes causes a miss. Remote provider model revisions are not exposed by either client, so that limitation is recorded explicitly and `--force` is the refresh mechanism. The manifest contains 70 balanced monolingual training clips (10 per language), 21 held-out monolingual clips (3 per language), one 4 s Hindi + 4 s English training concatenation, and two held-out 8 s switch clips in opposite directions. Thus `n_train_clips=71`. The training switch uses training voices; both evaluation switches use held-out voices.

For each language, training uses five utterances from the locale's gTTS voice and five from a named male Edge voice. Held-out text is spoken by a named female Edge voice that occurs nowhere in training. The manifest records these voice IDs, and every pipeline stage rejects any overlap between training and evaluation IDs. There are 14 training and 7 held-out synthetic voice IDs with zero overlap. The 1,600 updates consume 11,200 examples, or 157.7465 effective passes over the 71-clip training split, versus about 267 passes in the previous 21-clip run.

This is voice-disjoint only at the provider's synthetic voice-ID level; it is not a human speaker study. Each language still has just one held-out voice, and there is no telephony codec, room noise, or natural within-speaker variation. A frozen-checkpoint 2 × 2 Edge voice/text familiarity factorial found a material renderer/profile effect: full-clip student accuracy was 66.67% for seen profiles and 16.67% for unseen profiles, the largest tested factor. That effect is confounded with male-seen/female-unseen profiles and is not a sufficient sole fix: exact familiar training controls reached only 71.43%, with English and Marathi recall both 0/3, and Marathi reversed the aggregate effect. The held-out result below is therefore reported as failed transfer with both profile sensitivity and optimisation/target/class collapse, not as a human-speaker shortcut or population estimate.

## Teacher choice and targets

The frozen teacher is [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa), pinned to revision `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`. Before inference, target generation verifies SHA-256 and byte size for both weight files, `hyperparams.yaml`, and `label_encoder.txt` (combined artifact hash `f193a054…c0f706`), overrides SpeechBrain's pretrainer path to that resolved snapshot, and asserts the seven selected label indices. The teacher is purpose-built for LID, uses full-window attentive statistics, and is cheap enough to run repeatedly on CPU. The isolated [`teacher-bakeoff`](experiments/teacher-bakeoff/REPORT.md) retained it: across the same 21 held-out clips at 1/2/4 s, restricted seven-way accuracy was 58/63 for ECAPA, 54/63 for MMS-LID-126, and 42/63 for Whisper-small; ECAPA was also 8.4× and 15.9× cheaper by wall time. This clean synthetic comparison is not a production teacher benchmark, but neither challenger earned a switch.

The teacher's 107-way log posterior is restricted to the seven supported deployment languages, renormalized, and softened at temperature `T=2`. Production would retain out-of-set mass as an unknown-language signal; discarding it here keeps the student head and objective focused.

There are two target regimes:

1. For a known monolingual clip, one converged full-utterance teacher posterior is repeated over time. Language is stationary, so this is a low-variance semantic target. The first second is down-weighted because a causal student has little evidence there while the teacher has heard the whole clip.
2. A concatenated/code-switched clip **never** receives one utterance label. At acoustic frame `i`, the teacher sees a local full-context window `[t_i - 1.75 s, t_i + 0.25 s]`. It runs every 250 ms, and each 10 ms target holds the most recent anchor at or before that frame. This causal hold is up to 240 ms stale, but it never consults the next teacher anchor. It retains a temporal boundary without calling the entire 8 s clip Hindi or English.

This mixed strategy is intentional: a converged target is useful only under the stationary-language assumption; local targets are mandatory once that assumption is false.

Equal clip counts did not produce balanced supervision. The schema-6 audit applies the actual `D=21`, `L=4`, and 100-frame ramp to all 71 training targets; the per-language probabilities below are ramp/loss-weighted means over the ten monolingual source clips, while target mass is the share of all T=2 target probability mass, including the mixed clip.

| Manifest language | Clips | Seven-way teacher top-1 correct | Loss-weighted T=1 P(label) | Loss-weighted T=2 P(label) | T=2 target mass |
|---|---:|---:|---:|---:|---:|
| English | 10 | 6/10 | 55.6085% | 48.0021% | 6.5290% |
| Hindi | 10 | 10/10 | 98.2843% | 96.3713% | 20.6489% |
| Marathi | 10 | 10/10 | 94.6961% | 87.7732% | 12.9120% |
| Bengali | 10 | 10/10 | 99.7979% | 96.7251% | 13.2789% |
| Tamil | 10 | 10/10 | 99.9905% | 99.3689% | 13.5451% |
| Telugu | 10 | 10/10 | 99.9880% | 99.0964% | 15.3056% |
| Gujarati | 10 | 10/10 | 99.6386% | 95.5046% | 17.7806% |

The audit binds integer correctness/confusions, both selected seven-way and native 107-way top-1 evidence, clip-mean and loss-weighted T=1/T=2 manifest-label probability, retained selected-language mass, provider/voice counts, valid frames, ramp weight, and aggregate class contributions. For clarity, English's unweighted ten-clip means are 51.9271% at T=1 and 45.0260% at T=2; the 55.6085%/48.0021% table entries apply the actual frame/ramp loss weights. Its explicit floor requires exactly 10 monolingual clips and at least 8 selected-top-1-correct targets per language; only English fails. Missing, inconsistent, or even self-rehashed forged audit evidence is rejected. The failed floor is intentionally diagnostic rather than a training gate so this flawed baseline remains exactly reproducible, but it prevents describing the targets as balanced or treating another retrain as promotion evidence before the English supervision failure is controlled. The bound audit is `fd691497…38ca`.

Target caches are fail-closed rather than trusted by filename. Every `.npz` records the exact ordered language codes, clip/target kind, frame count, pinned teacher revision/artifact hash, target-generator source hash, exact manifest-file hash, canonical manifest-record hash, source-WAV SHA-256, and complete target-configuration hash. Schema 6 stores both raw and temperature-softened anchors, compact native-teacher top-1 evidence, and the recomputable training-target audit, and binds the directory to one exact manifest snapshot. Every strict load reconstructs the softened anchors from `normalize(anchor_probs ** (1/T))`, reconstructs both dense tensors under the declared previous-anchor-hold or constant-repeat policy, recomputes the audit from the bound targets, and rejects a mismatch above `rtol=1e-6, atol=2e-7`. It also binds each local semantic frame to its bracketing and selected source anchors, teacher/student latest-audio samples, and availability assertion. The generator identity covers its four source files and exact Python/library versions; the directory index additionally hashes the complete manifest and target-file set. Target generation, training, and evaluation validate the complete 94-file release, including 41,195 dense probability frames and the 2,394-frame switch availability ledger; training then reloads its 71 selected inputs through the same checks. Shape, finiteness, non-negativity, normalization, frame counts, expansion lineage, native-teacher evidence, audit contents, and evidence clocks are checked before use.

The student checkpoint is also the authoritative run bundle. Source schema 1 is captured before project imports; run-identity schema 4 then captures `manifest.jsonl` once before any data consumer or tensor preload. Its immutable value retains the exact bytes (`7df6fc5d…9314c`), canonical 94-record digest (`d47a8b62…ecc01`), and base directory; speaker audit, target cache, corpus identity, dataset, training, and evaluation receive defensive records from that same value. Root, corpus, and target identities must carry equal exact-file and canonical-record digests. Live pathname reads remain drift alarms, but cannot change the bytes already consumed. The run identity also covers the executed tree, all 94 audio hashes, target metadata/index, teacher, complete frontend/model/loss/timing configuration, environment, and training settings. Training revalidates the stage and live dependencies before preload, before optimization, and immediately before publication. The final content-derived run ID adds the model-state hash, and `train_metrics.json` is copied into the checkpoint as immutable training evidence before publication with the checkpoint hash. Evaluation captures one generation independently and refuses to score unless it equals the checkpoint-bound snapshot. The submitted run is `lidrun-25084aed…2db870`, with source root `54e098d6…d682`; `results/summary.json` records five training and two evaluation source-stage verifications, all three dependency equality checks, `manifest_bindings_equal=true`, and `evaluation_run_identity_validated=true`.

## Future-information asymmetry and objective

Log-mels use a 25 ms window, 10 ms hop, `center=False`, and no whole-utterance normalization. The student stacks four future feature frames (`L=4`, 40 ms) and emits the label for teacher frame `i` at student frame `i+D`, where `D=21` (210 ms). Therefore its latest raw feature is

```text
i + D + L = i + 25 frames = i + 250 ms,
```

exactly the end of a switch-teacher window evaluated at frame `i`. The padding mask additionally requires `i+D+L < sequence_length`, so padded right context cannot enter the loss.

Sparse switch anchors are expanded by previous-anchor hold. Schema-6 caches record an exclusive, unclipped sample clock so right padding at a clip edge cannot disguise unavailable evidence, and the validator proves that the stored dense tensors actually equal the declared expansion. For dense semantic frame `i` and its held source anchor `a<=i`, the teacher's latest requested sample is `a*H + W + F`; the aligned student's is `i*H + W + (D+L)*H`. Because `D+L=25` frames equals the teacher's 250 ms future, all 2,394 dense frames across the three switch clips satisfy `teacher_latest <= student_latest`. A regression proves that the old linear expansion fails this contract at all 2,295 non-anchor frames with `D=21,L=4`; another supplies a valid hold ledger plus forged linear dense tensors and requires rejection. Paying `D>=45` would make the interpolation evidence available only at a much larger delay.

This closes the future-information bug, not the target-quality problem. Holding a 250 ms grid can delay a target change by up to 240 ms, and the backward-heavy ECAPA window itself still reaches stable switches too late in the teacher-only audits. The corrected run is latency-valid with respect to future evidence, but it is not evidence that this target is a good low-latency switch teacher.

The isolated [`target-type-ablation`](experiments/target-type-ablation/REPORT.md) uses the same previous-anchor hold for a fair pure-target comparison. It rejects full-utterance, centred-2 s, and cumulative-prefix targets as replacements: all three students missed both switches, and the context-valid prefix arm gained only 0.47 frame-macro points over naive full targets while doubling unmatched EMA churn to 118.11 changes/min. That is a rejection guardrail, not validation of the main mixed targets.

The subsequent [`teacher-window-audit`](experiments/teacher-window-audit/REPORT.md) also rejected every tested shorter bounded, causal-rolling, and cumulative-prefix teacher window: early crossings were unstable or the accuracy/churn cost was too large. The [`teacher-anchor-hop-audit`](experiments/teacher-anchor-hop-audit/REPORT.md) then rejected every tested sparse grid against true 10 ms calls and showed that density alone does not repair the much larger stable-transition delay. These guardrails defer the target-by-delay sweep. They still do not validate the current mixed target; a different availability-valid trajectory is needed.

For batch item `b`, teacher posterior `q`, student logits `z`, temperature `T`, validity mask `m`, and early evidence ramp `w`, training minimizes

```text
                 sum(b,i) m[b,i] w[i] KL(q_T[b,i] || softmax(z[b,i+D] / T))
L_KD = T^2 * ------------------------------------------------------------------ .
                                  sum(b,i) m[b,i] w[i]

w[i] = min((i + 1) / 100, 1).
```

The `T²` term preserves gradient scale under softening. For switch clips, `q[b,i]` is the most recent available local-window posterior. For monolingual clips it is the converged utterance posterior; the ramp acknowledges that the teacher's confidence is not reproducible at the very start. This is knowledge distillation only—human/TTS language labels are used to build and audit the split, not in the optimization loss.

A target contributes only when `i+D+L < sequence_length`. If that condition is false for every item in a batch, the loss raises a clear error instead of returning zero and allowing a fake zero-gradient optimizer step. At the default `D=21`, `L=4` boundary, length 25 is rejected and length 26 supplies exactly the first valid target frame; the regression test checks that this frame backpropagates a nonzero gradient.

## Student and latency budget

The student projects 40 log-mels plus four explicit right-context frames into 64 channels, then applies six causal depthwise-separable residual blocks with dilations `(1, 2, 4, 8, 16, 32)` and a 7-class head. Its finite receptive field is 127 frames (about 1.27 s of past), so it cannot retain an old language forever as an unconstrained recurrent state can. It has **42,567 parameters**. The live `streaming_step` retains only finite left history and the four pending lookahead frames. It emits each logit once, after all four future feature frames really exist; it never publishes the zero-padded end tail as stable. Every call returns its absolute output range, received-feature boundary, and optional post-inference monotonic timestamp. Policy replay preserves those actual call groups and derives audio availability from the exclusive end sample of the latest feature received; it does not concatenate and re-slice outputs on a new phase. Chunked replay recomputes the small left overlap for clarity, while a production kernel would cache convolution state. Tests compare every emitted logit with the stable prefix of whole-sequence inference and exercise a growing prefix plus continuation.

The conservative worst-case algorithmic model latency is **435 ms**:

```text
25 ms analysis frame + (210 ms label delay + 40 ms lookahead) + 160 ms chunk = 435 ms.
```

Past context adds compute but no algorithmic latency. On six Torch CPU threads, including log-mel extraction and the deliberately uncached overlap, this recorded evaluation process's five repeats span **0.0064–0.0070 RTF** with median **0.0066**. Equivalent processes have produced materially different medians from `0.0058` to `0.0883` (also including `0.0171`), so this is a transient offline-replay observation, not a reproducible speed factor, incremental-frontend result, or end-to-end service SLO. Routing policy smoothing/dwell is separate from model latency.

## Submitted sanity results

These values are from the included `results/` artifacts, not aspirational numbers:

| Check | Result |
|---|---:|
| Monolingual train / held-out clips | 70 / 21 |
| Train / held-out synthetic voice IDs | 14 / 7 (no overlap) |
| Provenance-validated target cache | 94/94 files; schema 6 |
| Training-target audit / quality floor | 71/71 clips recomputed; valid ledger; failed for English (6/10 < 8/10) |
| Immutable manifest generation | one 94-record snapshot; exact/canonical child bindings equal |
| Dense target-to-anchor reconstruction | 41,195/41,195 frames valid; raw + `T=2` soft |
| Switch-target availability ledger | 2,394/2,394 frames valid; previous-anchor hold |
| Launch dependency snapshot / equality checks | before preload; 3/3 passed |
| Executed source snapshot / origin isolation | source schema 1; 5 train + 2 eval checks passed |
| Checkpoint/corpus/target/config/training identity validated | yes; schema 4 |
| Requested / successful / post-update-checked steps | 1,600 / 1,600 / 1,600 |
| Effective epochs | 157.7465 |
| First 10-step mean KD loss | 6.8697 |
| Last 10-step mean KD loss | 1.7415 |
| Finite loss/gradient histories and post-update model/optimizer state | yes |
| Teacher held-out known-label frame accuracy | 1.0000 |
| Student held-out frame agreement with teacher | 0.1823 |
| Student held-out known-label frame accuracy | 0.1823 |
| Student held-out clip accuracy | 0.1905 (4/21) |
| Hindi→English switch outcome | missed; lag `null` |
| Student parameters | 42,567 |
| Provisional end-tail outputs withheld | 4 frames |
| Actual switch emission schedule | 798 input / 794 stable / 773 aligned frames; 50 calls / 49 policy groups |
| First policy-group audio availability | 0.335 s actual call; 0.375 s legacy reconstruction |
| Actual vs legacy group-posterior L1 | 0.1864 mean; 0.4707 max |
| Six-thread offline-replay compute RTF | 0.0066 median (process-local) |

Agreement is not called accuracy: `eval_metrics.json` reports both student↔teacher agreement and student/teacher accuracy against the known synthesis language. The frozen teacher is correct on all 21 held-out monolingual clips, while the student generalises poorly to their unseen voices. On the held-out Hindi→English switch, the illustrative policy never establishes even its initial Hindi commit, so no English commit exists and lag is `null`; a miss is not assigned a flattering latency. The policy averages the delay-valid logits from each actual streaming call (the first group has 7 frames, later full groups have 16), applies an EMA with new weight 0.30, and requires posterior ≥0.60, a 0.10 margin, and three consecutive groups. The detailed 50-call schedule and offline monotonic completion offsets are in `eval_metrics.json`; `incremental_frontend_included=false` prevents treating them as an online SLO. This operating point is not calibrated on the tiny dataset; `DESIGN.md` describes how to set it properly.

## Tests and repository map

- `tests/test_alignment.py` proves `teacher[i] ↔ student[i+D]`, validates the valid-tail mask, rejects an all-invalid `length=D+L` batch, and proves `D+L+1` backpropagates.
- `tests/test_causality.py` changes every feature after `t+L` and requires outputs through `t` to be bit-identical; it also checks stable chunk/full equivalence, growing-prefix emission without duplicates, and the exact 798-frame call schedule with a fake clock.
- `tests/test_data_split.py` checks the manifest speaker audit, including speakers inherited by switch clips, and proves overlap is rejected.
- `tests/test_corpus_publication.py` checks recipe invalidation, byte-addressed cache reuse, checksum validation, and failure before the final atomic manifest swap.
- `tests/test_target_availability.py` proves previous-anchor hold is available at `D=21,L=4`, linear expansion is not available off-anchor unless `D>=45`, and end padding cannot hide an irregular final-anchor violation.
- `tests/test_target_cache.py` rejects reordered class columns, changed waveform/manifest/config content, modified target files, forged availability ledgers, dense tensors that disagree with their declared hold expansion, soft anchors that disagree with `T=2`, and a self-rehashed forged training-target audit; it also proves exact-byte versus semantic manifest identity, defensive copies, cache/corpus/dataset construction without reopening the manifest, and the equal-count/unequal-supervision case.
- `tests/test_training_contract.py` rejects zero/non-finite run settings and an empty full-batch loader, injects model/optimizer corruption after an update, and proves success flags require completed checked work.
- `tests/test_run_identity.py` proves launch snapshots are copied by value, rejects deterministic A/B/A and forged-child manifest mixes, rejects live audio/manifest/target-metadata changes before publication, and rejects changed timing, target identity, model state, or unrelated training metrics at evaluation while accepting one fully bound run.
- `tests/test_source_stage.py` proves an imported/captured source A still executes after the live path becomes B, includes package initializers, isolates editable paths and lazy live-only helpers, and rejects content, deletion, extra-file, symlink, mode, and manifest tampering.
- `scripts/prepare_data.py`, `teacher_targets.py`, `train.py`, and `eval.py` are the single entry points for each stage.
- `src/streaming_lid/` holds configuration, frontend, model, loss, and dataset code.
- `DESIGN.md` is the Part 2 live-ASR design.

## Implemented versus intentionally out of scope

Implemented: reproducible multi-voice audio acquisition with retry-safe, recipe- and byte-addressed caching plus staged atomic manifest publication; enforced voice-disjoint train/evaluation manifests; pinned and artifact-hashed frozen teacher inference; one immutable exact-byte manifest generation shared by every data consumer; content-bound target caches with generator-source, exact/canonical manifest, ordered-class, native-teacher summary, raw/soft anchor lineage, dense expansion, probability, sample-level availability, and recomputed per-class training-target validation; a pre-import content-addressed executed-source stage with origin/environment binding; an immutable launch dependency snapshot with three equality gates and a checkpoint-bound run identity over model/data/targets/config/source/training evidence; soft temporal targets with availability-valid previous-anchor hold on switches; special handling of switch clips; streaming-safe features; bounded-lookahead causal model with stable stateful emissions and exact per-call ranges/clocks; delayed KL with an explicit no-valid-frame guard; real backward/optimizer steps with finite model and optimizer state checked after every update; separate agreement and known-label metrics; stable chunk-equivalence and causality tests; RTF; hysteretic switch measurement; and the requested plot/JSON outputs.

Intentionally not implemented: a real telephony/VAD frontend, an ASR server/router, probability calibration on representative calls, an unknown-language head, checkpoint export/quantization, or convergence training. With more compute/data I would train on speaker-disjoint FLEURS/Common Voice plus anonymized 8 kHz call audio, add codec/noise/reverb augmentation and an `other` class, tune thresholds on a cost-weighted dev set, and report confidence intervals, false switches/hour, miss rate, and lag percentiles.

## Final summary and open issues

The CPU path works and the optimization plumbing is sound: it completes all 1,600 requested real-audio updates, passes 1,600 post-update model/optimizer checks, validates one exact manifest generation across the complete evaluation identity, reduces the 10-step mean KD loss from 6.8697 to 1.7415, and preserves causal/chunk-equivalent behavior. The schema-6 cache proves the manifest binding, switch evidence clocks, all 41,195 stored dense expansions, and a recomputed 71-clip target audit. That audit invalidates any “balanced supervision” claim: English is only 6/10 teacher-correct, fails the 8/10 diagnostic floor, and receives 6.5290% of T=2 mass versus Hindi's 20.6489%. The speaker-disjoint result remains poor: despite a perfect held-out teacher, the student achieves only 18.23% frame accuracy and misses the Hindi→English switch. None of the tested shorter teacher windows qualifies as a replacement, so a different target trajectory is required before latency tuning. The frozen voice/text factorial confirms a material 50-point synthetic-profile effect, but familiar audio reaches only 71.43% with two zero-recall classes; voice diversification and checkpoint/target/class-collapse diagnosis are both needed. The isolated cross-view-KD pilot recommends channel augmentation for a future controlled run, but its weights are not used here and it also missed both switches. The later [frozen-checkpoint external-validation diagnostic](experiments/checkpoint-external-validation/REPORT.md) also leaves the authoritative step-1,600 checkpoint unchanged: selected step 800 gains only 0.38 points on the pinned FLEURS-validation composite, the stored unadjusted descriptive interval is [-0.48, +1.24] points, every candidate has zero worst-language composite, and all miss both local switches. The interval treats repeated prefix views as independent and follows adaptive five-state selection, so it is not a confirmatory 95% coverage statement; independently recomputed clustered sensitivities also span zero. The run additionally reused an ECAPA cache that is internally coherent but does not bind its producing source, runtime, or ordered language mapping, so its exact teacher-control values have incomplete producer provenance. The no-promotion decision does not rely on either weakness: the 0.38-point gain independently misses the predeclared +2-point gate. These synthetic and read-speech results are sanity/failure evidence only; the policy is uncalibrated, and seven-way renormalization cannot reject an unsupported language.
