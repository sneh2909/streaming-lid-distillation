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

The defaults reproduce the submitted run: 1,600 optimizer steps, batch size 7, learning rate `1e-3`, and CPU only. Incomplete final batches are dropped, so every update has the same batch size. On this 20-core/13 GB host the full cached run is comfortably under 30 minutes; network speed dominates a first run. `prepare_data.py --force` refreshes TTS audio. Audio filenames include the voice ID, so a changed split cannot reuse a stale recording. Audio, teacher targets, caches, checkpoints, and `.venv` are ignored by Git.

Important outputs are:

- `results/summary.json`: required machine-readable summary, including every optimizer-step loss.
- `results/train_metrics.json`: loss and gradient history plus the real-audio/no-NaN assertions.
- `results/eval_metrics.json`: per-clip teacher agreement, known-label accuracy, speaker audit, stable-emission audit, policy settings, RTF, and switch outcome.
- `results/switch_plot.png`: offline teacher posteriors and emitted student chunk-EMA posteriors.
- `results/teacher_metrics.json`: target-generation audit.

## Data and split

`scripts/prepare_data.py` synthesizes PCM WAVs at 16 kHz through gTTS and Microsoft Edge TTS, then decodes MP3 responses in-process with `miniaudio`. The manifest contains 70 balanced monolingual training clips (10 per language), 21 held-out monolingual clips (3 per language), one 4 s Hindi + 4 s English training concatenation, and two held-out 8 s switch clips in opposite directions. Thus `n_train_clips=71`. The training switch uses training voices; both evaluation switches use held-out voices.

For each language, training uses five utterances from the locale's gTTS voice and five from a named male Edge voice. Held-out text is spoken by a named female Edge voice that occurs nowhere in training. The manifest records these voice IDs, and every pipeline stage rejects any overlap between training and evaluation IDs. There are 14 training and 7 held-out synthetic voice IDs with zero overlap. The 1,600 updates consume 11,200 examples, or 157.7465 effective passes over the 71-clip training split, versus about 267 passes in the previous 21-clip run.

This is voice-disjoint only at the provider's synthetic voice-ID level; it is not a human speaker study. Each language still has just one held-out voice, and there is no telephony codec, room noise, or natural within-speaker variation. The poor held-out result below proves failed transfer to these unseen voices. It is consistent with a voice/provider shortcut, but without seen-voice and provider-controlled slices it does not isolate the cause. It is reported as a failure, not a population estimate.

## Teacher choice and targets

The frozen teacher is [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa). It is purpose-built for LID, directly exposes normalized scores for all seven chosen languages, uses full-window attentive statistics, and is cheap enough to run repeatedly on CPU. The isolated [`teacher-bakeoff`](experiments/teacher-bakeoff/REPORT.md) retained it: across the same 21 held-out clips at 1/2/4 s, restricted seven-way accuracy was 58/63 for ECAPA, 54/63 for MMS-LID-126, and 42/63 for Whisper-small; ECAPA was also 8.4× and 15.9× cheaper by wall time. This clean synthetic comparison is not a production teacher benchmark, but neither challenger earned a switch.

The teacher's 107-way log posterior is restricted to the seven supported deployment languages, renormalized, and softened at temperature `T=2`. Production would retain out-of-set mass as an unknown-language signal; discarding it here keeps the student head and objective focused.

There are two target regimes:

1. For a known monolingual clip, one converged full-utterance teacher posterior is repeated over time. Language is stationary, so this is a low-variance semantic target. The first second is down-weighted because a causal student has little evidence there while the teacher has heard the whole clip.
2. A concatenated/code-switched clip **never** receives one utterance label. At acoustic frame `i`, the teacher sees a local full-context window `[t_i - 1.75 s, t_i + 0.25 s]`. It runs every 250 ms and its posterior is linearly interpolated to the student's 10 ms grid. This retains a temporal boundary and avoids calling the entire 8 s clip Hindi or English.

This mixed strategy is intentional: a converged target is useful only under the stationary-language assumption; local targets are mandatory once that assumption is false.

## Future-information asymmetry and objective

Log-mels use a 25 ms window, 10 ms hop, `center=False`, and no whole-utterance normalization. The student stacks four future feature frames (`L=4`, 40 ms) and emits the label for teacher frame `i` at student frame `i+D`, where `D=21` (210 ms). Therefore its latest raw feature is

```text
i + D + L = i + 25 frames = i + 250 ms,
```

exactly the end of the switch teacher's local window. It is not asked to imitate information that will be unavailable online. The padding mask additionally requires `i+D+L < sequence_length`, so padded right context cannot enter the loss.

For batch item `b`, teacher posterior `q`, student logits `z`, temperature `T`, validity mask `m`, and early evidence ramp `w`, training minimizes

```text
                 sum(b,i) m[b,i] w[i] KL(q_T[b,i] || softmax(z[b,i+D] / T))
L_KD = T^2 * ------------------------------------------------------------------ .
                                  sum(b,i) m[b,i] w[i]

w[i] = min((i + 1) / 100, 1).
```

The `T²` term preserves gradient scale under softening. For switch clips, `q[b,i]` is the local window posterior above. For monolingual clips it is the converged utterance posterior; the ramp acknowledges that the teacher's confidence is not reproducible at the very start. This is knowledge distillation only—human/TTS language labels are used to build and audit the split, not in the optimization loss.

A target contributes only when `i+D+L < sequence_length`. If that condition is false for every item in a batch, the loss raises a clear error instead of returning zero and allowing a fake zero-gradient optimizer step. At the default `D=21`, `L=4` boundary, length 25 is rejected and length 26 supplies exactly the first valid target frame; the regression test checks that this frame backpropagates a nonzero gradient.

## Student and latency budget

The student projects 40 log-mels plus four explicit right-context frames into 64 channels, then applies six causal depthwise-separable residual blocks with dilations `(1, 2, 4, 8, 16, 32)` and a 7-class head. Its finite receptive field is 127 frames (about 1.27 s of past), so it cannot retain an old language forever as an unconstrained recurrent state can. It has **42,567 parameters**. The live `streaming_step` retains only finite left history and the four pending lookahead frames. It emits each logit once, after all four future feature frames really exist; it never publishes the zero-padded end tail as stable. Chunked replay recomputes the small left overlap for clarity, while a production kernel would cache convolution state. Tests compare every emitted logit with the stable prefix of whole-sequence inference and exercise a growing prefix plus continuation.

The conservative worst-case algorithmic model latency is **435 ms**:

```text
25 ms analysis frame + (210 ms label delay + 40 ms lookahead) + 160 ms chunk = 435 ms.
```

Past context adds compute but no algorithmic latency. On one CPU thread, including log-mel extraction and the deliberately uncached overlap, measured median RTF is **0.0057** (about 174× real time). Routing policy smoothing/dwell is separate from model latency.

## Submitted sanity results

These values are from the included `results/` artifacts, not aspirational numbers:

| Check | Result |
|---|---:|
| Monolingual train / held-out clips | 70 / 21 |
| Train / held-out synthetic voice IDs | 14 / 7 (no overlap) |
| Optimizer steps; effective epochs | 1,600; 157.7465 |
| First 10-step mean KD loss | 6.8704 |
| Last 10-step mean KD loss | 1.4403 |
| Finite losses and gradients | yes |
| Teacher held-out known-label frame accuracy | 1.0000 |
| Student held-out frame agreement with teacher | 0.1851 |
| Student held-out known-label frame accuracy | 0.1851 |
| Student held-out clip accuracy | 0.1905 (4/21) |
| Hindi→English switch outcome | missed; lag `null` |
| Student parameters | 42,567 |
| Provisional end-tail outputs withheld | 4 frames |
| Single-thread CPU RTF | 0.0057 |

Agreement is not called accuracy: `eval_metrics.json` reports both student↔teacher agreement and student/teacher accuracy against the known synthesis language. The frozen teacher is correct on all 21 held-out monolingual clips, while the student generalises poorly to their unseen voices. On the held-out Hindi→English switch, the illustrative policy never establishes even its initial Hindi commit, so no English commit exists and lag is `null`; a miss is not assigned a flattering latency. The policy averages each 160 ms chunk, applies an EMA with new weight 0.30, and requires posterior ≥0.60, a 0.10 margin, and three consecutive chunks. This operating point is not calibrated on the tiny dataset; `DESIGN.md` describes how to set it properly.

## Tests and repository map

- `tests/test_alignment.py` proves `teacher[i] ↔ student[i+D]`, validates the valid-tail mask, rejects an all-invalid `length=D+L` batch, and proves `D+L+1` backpropagates.
- `tests/test_causality.py` changes every feature after `t+L` and requires outputs through `t` to be bit-identical; it also checks stable chunk/full equivalence and growing-prefix emission without duplicates.
- `tests/test_data_split.py` checks the manifest speaker audit, including speakers inherited by switch clips, and proves overlap is rejected.
- `scripts/prepare_data.py`, `teacher_targets.py`, `train.py`, and `eval.py` are the single entry points for each stage.
- `src/streaming_lid/` holds configuration, frontend, model, loss, and dataset code.
- `DESIGN.md` is the Part 2 live-ASR design.

## Implemented versus intentionally out of scope

Implemented: reproducible multi-voice audio acquisition with retry-safe, voice-qualified caching; enforced voice-disjoint train/evaluation manifests; frozen real teacher inference; soft temporal targets; special handling of switch clips; streaming-safe features; bounded-lookahead causal model with stable stateful emissions; delayed KL with an explicit no-valid-frame guard; real backward/optimizer steps; separate agreement and known-label metrics; stable chunk-equivalence and causality tests; RTF; hysteretic switch measurement; and the requested plot/JSON outputs.

Intentionally not implemented: a real telephony/VAD frontend, an ASR server/router, probability calibration on representative calls, an unknown-language head, checkpoint export/quantization, or convergence training. With more compute/data I would train on speaker-disjoint FLEURS/Common Voice plus anonymized 8 kHz call audio, add codec/noise/reverb augmentation and an `other` class, tune thresholds on a cost-weighted dev set, and report confidence intervals, false switches/hour, miss rate, and lag percentiles.

## Final summary and open issues

The CPU path works and the optimization plumbing is sound: it executes 1,600 real updates without NaNs, reduces the 10-step mean KD loss from 6.8704 to 1.4403, and preserves causal/chunk-equivalent behavior. The speaker-disjoint evaluation also overturns the earlier apparent success: despite a perfect held-out teacher, the student achieves only 18.51% frame accuracy and misses the Hindi→English switch. The principal open issue is unseen-voice generalisation, before latency tuning; a seen-voice/provider ablation is still needed to attribute the cause. These synthetic results are sanity/failure evidence only; the policy is uncalibrated, and seven-way renormalization cannot reject an unsupported language.
