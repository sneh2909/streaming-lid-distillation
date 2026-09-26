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
- `results/teacher_metrics.json`: target-generation and 94-file provenance audit.

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

Target caches are fail-closed rather than trusted by filename. Every `.npz` records the exact ordered language codes, clip/target kind, frame count, pinned teacher revision/artifact hash, target-generator source hash, canonical manifest-record hash, source-WAV SHA-256, and complete target-configuration hash. Schema 4 stores both raw and temperature-softened anchors. Every strict load reconstructs the softened anchors from `normalize(anchor_probs ** (1/T))`, reconstructs both dense tensors under the declared previous-anchor-hold or constant-repeat policy, and rejects a mismatch above `rtol=1e-6, atol=2e-7`. It also binds each local semantic frame to its bracketing and selected source anchors, teacher/student latest-audio samples, and availability assertion. The generator identity covers its four source files and exact Python/library versions; the directory index additionally hashes the complete manifest and target-file set. Target generation and evaluation validate all 94 files, including 41,195 dense probability frames and the 2,394-frame switch availability ledger; training validates all 71 inputs. Shape, finiteness, non-negativity, normalization, frame counts, expansion lineage, and evidence clocks are checked before use.

The student checkpoint is also the authoritative run bundle. Source schema 1 is captured before project imports; run-identity schema 3 then takes one immutable dependency snapshot before any feature/target tensor is preloaded, covering that executed tree, the exact manifest bytes and all 94 audio hashes, target metadata/index, teacher, complete frontend/model/loss/timing configuration, environment, and training settings. The dataset consumes the captured manifest record set rather than reopening a mutable selection. Training revalidates the staged bytes and equality-checks data/target/settings before preload, before optimization, and immediately before publication; a concurrent stage, manifest/audio, target-metadata, or settings change aborts instead of being attributed to the run. The final content-derived run ID adds the model-state hash, and `train_metrics.json` is copied into the checkpoint as immutable training evidence before publication with the checkpoint hash. Evaluation refuses to score unless its independently staged source digest and current dependencies equal that checkpoint-bound payload exactly. The submitted run is `lidrun-b0dbf74a…06b15f`, with source root `99d4728a…8a6ee`; `results/summary.json` records five training and two evaluation source-stage verifications, all three dependency equality checks, and `evaluation_run_identity_validated=true`.

## Future-information asymmetry and objective

Log-mels use a 25 ms window, 10 ms hop, `center=False`, and no whole-utterance normalization. The student stacks four future feature frames (`L=4`, 40 ms) and emits the label for teacher frame `i` at student frame `i+D`, where `D=21` (210 ms). Therefore its latest raw feature is

```text
i + D + L = i + 25 frames = i + 250 ms,
```

exactly the end of a switch-teacher window evaluated at frame `i`. The padding mask additionally requires `i+D+L < sequence_length`, so padded right context cannot enter the loss.

Sparse switch anchors are expanded by previous-anchor hold. Schema-4 caches record an exclusive, unclipped sample clock so right padding at a clip edge cannot disguise unavailable evidence, and the validator proves that the stored dense tensors actually equal the declared expansion. For dense semantic frame `i` and its held source anchor `a<=i`, the teacher's latest requested sample is `a*H + W + F`; the aligned student's is `i*H + W + (D+L)*H`. Because `D+L=25` frames equals the teacher's 250 ms future, all 2,394 dense frames across the three switch clips satisfy `teacher_latest <= student_latest`. A regression proves that the old linear expansion fails this contract at all 2,295 non-anchor frames with `D=21,L=4`; another supplies a valid hold ledger plus forged linear dense tensors and requires rejection. Paying `D>=45` would make the interpolation evidence available only at a much larger delay.

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

Past context adds compute but no algorithmic latency. On six Torch CPU threads, including log-mel extraction and the deliberately uncached overlap, this recorded evaluation's median offline-replay compute RTF is **0.0171** (about 58.3× faster than real time). This is not incremental-frontend or end-to-end service timing. Routing policy smoothing/dwell is separate from model latency.

## Submitted sanity results

These values are from the included `results/` artifacts, not aspirational numbers:

| Check | Result |
|---|---:|
| Monolingual train / held-out clips | 70 / 21 |
| Train / held-out synthetic voice IDs | 14 / 7 (no overlap) |
| Provenance-validated target cache | 94/94 files; schema 4 |
| Dense target-to-anchor reconstruction | 41,195/41,195 frames valid; raw + `T=2` soft |
| Switch-target availability ledger | 2,394/2,394 frames valid; previous-anchor hold |
| Launch dependency snapshot / equality checks | before preload; 3/3 passed |
| Executed source snapshot / origin isolation | source schema 1; 5 train + 2 eval checks passed |
| Checkpoint/corpus/target/config/training identity validated | yes; schema 3 |
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
| Six-thread offline-replay compute RTF | 0.0171 |

Agreement is not called accuracy: `eval_metrics.json` reports both student↔teacher agreement and student/teacher accuracy against the known synthesis language. The frozen teacher is correct on all 21 held-out monolingual clips, while the student generalises poorly to their unseen voices. On the held-out Hindi→English switch, the illustrative policy never establishes even its initial Hindi commit, so no English commit exists and lag is `null`; a miss is not assigned a flattering latency. The policy averages the delay-valid logits from each actual streaming call (the first group has 7 frames, later full groups have 16), applies an EMA with new weight 0.30, and requires posterior ≥0.60, a 0.10 margin, and three consecutive groups. The detailed 50-call schedule and offline monotonic completion offsets are in `eval_metrics.json`; `incremental_frontend_included=false` prevents treating them as an online SLO. This operating point is not calibrated on the tiny dataset; `DESIGN.md` describes how to set it properly.

## Tests and repository map

- `tests/test_alignment.py` proves `teacher[i] ↔ student[i+D]`, validates the valid-tail mask, rejects an all-invalid `length=D+L` batch, and proves `D+L+1` backpropagates.
- `tests/test_causality.py` changes every feature after `t+L` and requires outputs through `t` to be bit-identical; it also checks stable chunk/full equivalence, growing-prefix emission without duplicates, and the exact 798-frame call schedule with a fake clock.
- `tests/test_data_split.py` checks the manifest speaker audit, including speakers inherited by switch clips, and proves overlap is rejected.
- `tests/test_corpus_publication.py` checks recipe invalidation, byte-addressed cache reuse, checksum validation, and failure before the final atomic manifest swap.
- `tests/test_target_availability.py` proves previous-anchor hold is available at `D=21,L=4`, linear expansion is not available off-anchor unless `D>=45`, and end padding cannot hide an irregular final-anchor violation.
- `tests/test_target_cache.py` rejects reordered class columns, changed waveform/manifest/config content, modified target files, forged availability ledgers, dense tensors that disagree with their declared hold expansion, and soft anchors that disagree with `T=2` while accepting a fully content-bound cache.
- `tests/test_training_contract.py` rejects zero/non-finite run settings and an empty full-batch loader, injects model/optimizer corruption after an update, and proves success flags require completed checked work.
- `tests/test_run_identity.py` proves launch snapshots are copied by value, rejects live audio/manifest/target-metadata changes before publication, and rejects changed timing, target identity, model state, or unrelated training metrics at evaluation while accepting one fully bound run.
- `tests/test_source_stage.py` proves an imported/captured source A still executes after the live path becomes B, includes package initializers, isolates editable paths and lazy live-only helpers, and rejects content, deletion, extra-file, symlink, mode, and manifest tampering.
- `scripts/prepare_data.py`, `teacher_targets.py`, `train.py`, and `eval.py` are the single entry points for each stage.
- `src/streaming_lid/` holds configuration, frontend, model, loss, and dataset code.
- `DESIGN.md` is the Part 2 live-ASR design.

## Implemented versus intentionally out of scope

Implemented: reproducible multi-voice audio acquisition with retry-safe, recipe- and byte-addressed caching plus staged atomic manifest publication; enforced voice-disjoint train/evaluation manifests; pinned and artifact-hashed frozen teacher inference; content-bound target caches with generator-source, ordered-class, raw/soft anchor lineage, dense expansion, probability, and sample-level availability validation; a pre-import content-addressed executed-source stage with origin/environment binding; an immutable launch dependency snapshot with three equality gates and a checkpoint-bound run identity over model/data/targets/config/source/training evidence; soft temporal targets with availability-valid previous-anchor hold on switches; special handling of switch clips; streaming-safe features; bounded-lookahead causal model with stable stateful emissions and exact per-call ranges/clocks; delayed KL with an explicit no-valid-frame guard; real backward/optimizer steps with finite model and optimizer state checked after every update; separate agreement and known-label metrics; stable chunk-equivalence and causality tests; RTF; hysteretic switch measurement; and the requested plot/JSON outputs.

Intentionally not implemented: a real telephony/VAD frontend, an ASR server/router, probability calibration on representative calls, an unknown-language head, checkpoint export/quantization, or convergence training. With more compute/data I would train on speaker-disjoint FLEURS/Common Voice plus anonymized 8 kHz call audio, add codec/noise/reverb augmentation and an `other` class, tune thresholds on a cost-weighted dev set, and report confidence intervals, false switches/hour, miss rate, and lag percentiles.

## Final summary and open issues

The CPU path works and the optimization plumbing is sound: it completes all 1,600 requested real-audio updates, passes 1,600 post-update model/optimizer checks, validates the complete evaluation run identity, reduces the 10-step mean KD loss from 6.8697 to 1.7415, and preserves causal/chunk-equivalent behavior. The schema-4 cache proves both the switch evidence clocks and all 41,195 stored dense expansions, closing the gap where a valid ledger could certify unrelated tensors. The speaker-disjoint result remains poor: despite a perfect held-out teacher, the student achieves only 18.23% frame accuracy and misses the Hindi→English switch. None of the tested shorter teacher windows qualifies as a replacement, so a different target trajectory is required before latency tuning. The frozen voice/text factorial confirms a material 50-point synthetic-profile effect, but familiar audio reaches only 71.43% with two zero-recall classes; voice diversification and checkpoint/target/class-collapse diagnosis are both needed. The isolated cross-view-KD pilot recommends channel augmentation for a future controlled run, but its weights are not used here and it also missed both switches. These synthetic results are sanity/failure evidence only; the policy is uncalibrated, and seven-way renormalization cannot reject an unsupported language.
