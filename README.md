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

The defaults reproduce the submitted run: 1,600 optimizer steps, batch size 7, learning rate `1e-3`, and CPU only. Steps, batch size, and thread count must be positive integers; the learning rate must be positive and finite. Incomplete final batches are dropped, so every update has the same batch size, and training rejects a batch size larger than the split instead of looping over an empty loader. On this 20-core/13 GB host the full cached run is comfortably under 30 minutes; network speed dominates a first run. `prepare_data.py --force` refreshes TTS audio. Audio filenames include the voice ID, so a changed split cannot reuse a stale recording. Audio, teacher targets, caches, checkpoints, and `.venv` are ignored by Git.

Important outputs are:

- `results/summary.json`: required machine-readable summary, including every optimizer-step loss.
- `results/train_metrics.json`: loss and gradient history plus requested/completed update counts, post-update finiteness assertions, run identity, and checkpoint hash.
- `results/eval_metrics.json`: per-clip teacher agreement, known-label accuracy, speaker audit, stable-emission audit, policy settings, RTF, switch outcome, and the validated training/evaluation identity.
- `results/switch_plot.png`: offline teacher posteriors and emitted student chunk-EMA posteriors.
- `results/teacher_metrics.json`: target-generation and 94-file provenance audit.

## Data and split

`scripts/prepare_data.py` synthesizes PCM WAVs at 16 kHz through gTTS and Microsoft Edge TTS, then decodes MP3 responses in-process with `miniaudio`. The manifest contains 70 balanced monolingual training clips (10 per language), 21 held-out monolingual clips (3 per language), one 4 s Hindi + 4 s English training concatenation, and two held-out 8 s switch clips in opposite directions. Thus `n_train_clips=71`. The training switch uses training voices; both evaluation switches use held-out voices.

For each language, training uses five utterances from the locale's gTTS voice and five from a named male Edge voice. Held-out text is spoken by a named female Edge voice that occurs nowhere in training. The manifest records these voice IDs, and every pipeline stage rejects any overlap between training and evaluation IDs. There are 14 training and 7 held-out synthetic voice IDs with zero overlap. The 1,600 updates consume 11,200 examples, or 157.7465 effective passes over the 71-clip training split, versus about 267 passes in the previous 21-clip run.

This is voice-disjoint only at the provider's synthetic voice-ID level; it is not a human speaker study. Each language still has just one held-out voice, and there is no telephony codec, room noise, or natural within-speaker variation. The poor held-out result below proves failed transfer to these unseen voices. It is consistent with a voice/provider shortcut, but without seen-voice and provider-controlled slices it does not isolate the cause. It is reported as a failure, not a population estimate.

## Teacher choice and targets

The frozen teacher is [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa), pinned to revision `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`. Before inference, target generation verifies SHA-256 and byte size for both weight files, `hyperparams.yaml`, and `label_encoder.txt` (combined artifact hash `f193a054…c0f706`), overrides SpeechBrain's pretrainer path to that resolved snapshot, and asserts the seven selected label indices. The teacher is purpose-built for LID, uses full-window attentive statistics, and is cheap enough to run repeatedly on CPU. The isolated [`teacher-bakeoff`](experiments/teacher-bakeoff/REPORT.md) retained it: across the same 21 held-out clips at 1/2/4 s, restricted seven-way accuracy was 58/63 for ECAPA, 54/63 for MMS-LID-126, and 42/63 for Whisper-small; ECAPA was also 8.4× and 15.9× cheaper by wall time. This clean synthetic comparison is not a production teacher benchmark, but neither challenger earned a switch.

The teacher's 107-way log posterior is restricted to the seven supported deployment languages, renormalized, and softened at temperature `T=2`. Production would retain out-of-set mass as an unknown-language signal; discarding it here keeps the student head and objective focused.

There are two target regimes:

1. For a known monolingual clip, one converged full-utterance teacher posterior is repeated over time. Language is stationary, so this is a low-variance semantic target. The first second is down-weighted because a causal student has little evidence there while the teacher has heard the whole clip.
2. A concatenated/code-switched clip **never** receives one utterance label. At acoustic frame `i`, the teacher sees a local full-context window `[t_i - 1.75 s, t_i + 0.25 s]`. It runs every 250 ms and its posterior is linearly interpolated to the student's 10 ms grid. This retains a temporal boundary and avoids calling the entire 8 s clip Hindi or English.

This mixed strategy is intentional: a converged target is useful only under the stationary-language assumption; local targets are mandatory once that assumption is false.

Target caches are fail-closed rather than trusted by filename. Every `.npz` records the exact ordered language codes, clip/target kind, frame count, pinned teacher revision/artifact hash, target-generator source hash, canonical manifest-record hash, source-WAV SHA-256, and complete target-configuration hash. The generator identity covers its four source files and exact Python/library versions; the directory index additionally hashes the complete manifest and target-file set. Target generation reopens and validates all 94 files; training validates its 71 clips, and evaluation revalidates all 94 as a release gate. Probability shape, finiteness, non-negativity, normalization, and hard/soft frame counts are checked before use.

The student checkpoint is also the authoritative run bundle. Its content-derived run ID binds the final model-state hash, all 94 current audio hashes, exact manifest and target index, teacher, complete frontend/model/loss/timing configuration, pipeline source, and training settings. `train_metrics.json` is copied into the checkpoint as immutable training evidence, then published with the checkpoint file hash. Evaluation refuses to score unless the current corpus/targets/source still match and the external training metrics equal that checkpoint-bound payload exactly. The submitted run is `lidrun-1c3d0003…370d7`; `results/summary.json` records the full identities and `evaluation_run_identity_validated=true`.

## Future-information asymmetry and objective

Log-mels use a 25 ms window, 10 ms hop, `center=False`, and no whole-utterance normalization. The student stacks four future feature frames (`L=4`, 40 ms) and emits the label for teacher frame `i` at student frame `i+D`, where `D=21` (210 ms). Therefore its latest raw feature is

```text
i + D + L = i + 25 frames = i + 250 ms,
```

exactly the end of a switch-teacher window evaluated at frame `i`. The padding mask additionally requires `i+D+L < sequence_length`, so padded right context cannot enter the loss.

There is one important residual mismatch: switch posteriors are evaluated only every 25 frames and linearly interpolated. A target between anchors uses the *next* anchor, whose own window ends 250 ms later; its effective future horizon is therefore 250–490 ms, while the aligned student has 250 ms. Exact anchor frames are context-matched, but 24/25 interpolated frames are not. The submitted run is retained as honest plumbing evidence, not as a latency-valid switch result. The next method fix is a previous-anchor hold, per-frame teacher inference, or reserving the anchor hop inside the evidence budget before any delay sweep.

The isolated [`target-type-ablation`](experiments/target-type-ablation/REPORT.md) already uses previous-anchor hold for a fair pure-target comparison. It rejects full-utterance, centred-2 s, and cumulative-prefix targets as replacements: all three students missed both switches, and the context-valid prefix arm gained only 0.47 frame-macro points over naive full targets while doubling unmatched EMA churn to 118.11 changes/min. That is a rejection guardrail, not validation of the main mixed targets; removing linear-interpolation leakage from the main cache remains open.

For batch item `b`, teacher posterior `q`, student logits `z`, temperature `T`, validity mask `m`, and early evidence ramp `w`, training minimizes

```text
                 sum(b,i) m[b,i] w[i] KL(q_T[b,i] || softmax(z[b,i+D] / T))
L_KD = T^2 * ------------------------------------------------------------------ .
                                  sum(b,i) m[b,i] w[i]

w[i] = min((i + 1) / 100, 1).
```

The `T²` term preserves gradient scale under softening. For switch clips, `q[b,i]` is the interpolated local-window posterior, with the horizon caveat above. For monolingual clips it is the converged utterance posterior; the ramp acknowledges that the teacher's confidence is not reproducible at the very start. This is knowledge distillation only—human/TTS language labels are used to build and audit the split, not in the optimization loss.

A target contributes only when `i+D+L < sequence_length`. If that condition is false for every item in a batch, the loss raises a clear error instead of returning zero and allowing a fake zero-gradient optimizer step. At the default `D=21`, `L=4` boundary, length 25 is rejected and length 26 supplies exactly the first valid target frame; the regression test checks that this frame backpropagates a nonzero gradient.

## Student and latency budget

The student projects 40 log-mels plus four explicit right-context frames into 64 channels, then applies six causal depthwise-separable residual blocks with dilations `(1, 2, 4, 8, 16, 32)` and a 7-class head. Its finite receptive field is 127 frames (about 1.27 s of past), so it cannot retain an old language forever as an unconstrained recurrent state can. It has **42,567 parameters**. The live `streaming_step` retains only finite left history and the four pending lookahead frames. It emits each logit once, after all four future feature frames really exist; it never publishes the zero-padded end tail as stable. Chunked replay recomputes the small left overlap for clarity, while a production kernel would cache convolution state. Tests compare every emitted logit with the stable prefix of whole-sequence inference and exercise a growing prefix plus continuation.

The conservative worst-case algorithmic model latency is **435 ms**:

```text
25 ms analysis frame + (210 ms label delay + 40 ms lookahead) + 160 ms chunk = 435 ms.
```

Past context adds compute but no algorithmic latency. On six Torch CPU threads, including log-mel extraction and the deliberately uncached overlap, measured median replay RTF is **0.0084** (about 120× real time). Routing policy smoothing/dwell is separate from model latency.

## Submitted sanity results

These values are from the included `results/` artifacts, not aspirational numbers:

| Check | Result |
|---|---:|
| Monolingual train / held-out clips | 70 / 21 |
| Train / held-out synthetic voice IDs | 14 / 7 (no overlap) |
| Provenance-validated target cache | 94/94 files; schema 2 |
| Checkpoint/corpus/target/config/training identity validated | yes; schema 1 |
| Requested / successful / post-update-checked steps | 1,600 / 1,600 / 1,600 |
| Effective epochs | 157.7465 |
| First 10-step mean KD loss | 6.8704 |
| Last 10-step mean KD loss | 1.4403 |
| Finite loss/gradient histories and post-update model/optimizer state | yes |
| Teacher held-out known-label frame accuracy | 1.0000 |
| Student held-out frame agreement with teacher | 0.1851 |
| Student held-out known-label frame accuracy | 0.1851 |
| Student held-out clip accuracy | 0.1905 (4/21) |
| Hindi→English switch outcome | missed; lag `null` |
| Student parameters | 42,567 |
| Provisional end-tail outputs withheld | 4 frames |
| Six-thread CPU replay RTF | 0.0084 |

Agreement is not called accuracy: `eval_metrics.json` reports both student↔teacher agreement and student/teacher accuracy against the known synthesis language. The frozen teacher is correct on all 21 held-out monolingual clips, while the student generalises poorly to their unseen voices. On the held-out Hindi→English switch, the illustrative policy never establishes even its initial Hindi commit, so no English commit exists and lag is `null`; a miss is not assigned a flattering latency. The policy averages each 160 ms chunk, applies an EMA with new weight 0.30, and requires posterior ≥0.60, a 0.10 margin, and three consecutive chunks. This operating point is not calibrated on the tiny dataset; `DESIGN.md` describes how to set it properly.

## Tests and repository map

- `tests/test_alignment.py` proves `teacher[i] ↔ student[i+D]`, validates the valid-tail mask, rejects an all-invalid `length=D+L` batch, and proves `D+L+1` backpropagates.
- `tests/test_causality.py` changes every feature after `t+L` and requires outputs through `t` to be bit-identical; it also checks stable chunk/full equivalence and growing-prefix emission without duplicates.
- `tests/test_data_split.py` checks the manifest speaker audit, including speakers inherited by switch clips, and proves overlap is rejected.
- `tests/test_target_cache.py` rejects reordered class columns, changed waveform/manifest/config content, and modified target files while accepting a fully content-bound cache.
- `tests/test_training_contract.py` rejects zero/non-finite run settings and an empty full-batch loader, injects model/optimizer corruption after an update, and proves success flags require completed checked work.
- `tests/test_run_identity.py` rejects changed audio, timing, target identity, model state, and unrelated training metrics while accepting one fully bound run.
- `scripts/prepare_data.py`, `teacher_targets.py`, `train.py`, and `eval.py` are the single entry points for each stage.
- `src/streaming_lid/` holds configuration, frontend, model, loss, and dataset code.
- `DESIGN.md` is the Part 2 live-ASR design.

## Implemented versus intentionally out of scope

Implemented: reproducible multi-voice audio acquisition with retry-safe, voice-qualified caching; enforced voice-disjoint train/evaluation manifests; pinned and artifact-hashed frozen teacher inference; content-bound target caches with generator-source, ordered-class, and probability validation; a checkpoint-bound run identity over model/data/targets/config/source/training evidence; soft temporal targets; special handling of switch clips; streaming-safe features; bounded-lookahead causal model with stable stateful emissions; delayed KL with an explicit no-valid-frame guard; real backward/optimizer steps with finite model and optimizer state checked after every update; separate agreement and known-label metrics; stable chunk-equivalence and causality tests; RTF; hysteretic switch measurement; and the requested plot/JSON outputs.

Intentionally not implemented: a real telephony/VAD frontend, an ASR server/router, probability calibration on representative calls, an unknown-language head, checkpoint export/quantization, or convergence training. With more compute/data I would train on speaker-disjoint FLEURS/Common Voice plus anonymized 8 kHz call audio, add codec/noise/reverb augmentation and an `other` class, tune thresholds on a cost-weighted dev set, and report confidence intervals, false switches/hour, miss rate, and lag percentiles.

## Final summary and open issues

The CPU path works and the optimization plumbing is sound: it completes all 1,600 requested real-audio updates, passes 1,600 post-update model/optimizer checks, validates the complete evaluation run identity, reduces the 10-step mean KD loss from 6.8704 to 1.4403, and preserves causal/chunk-equivalent behavior. The speaker-disjoint evaluation also overturns the earlier apparent success: despite a perfect held-out teacher, the student achieves only 18.51% frame accuracy and misses the Hindi→English switch. Before latency tuning, the interpolated switch targets need a causally valid availability contract; independently, a seen-voice/provider ablation is needed to attribute the poor unseen-voice transfer. These synthetic results are sanity/failure evidence only; the policy is uncalibrated, and seven-way renormalization cannot reject an unsupported language.
