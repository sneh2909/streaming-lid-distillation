# Offline-to-streaming language-ID distillation

This repository distills SpeechBrain's offline VoxLingua107 ECAPA language-ID model into a 42,567-parameter causal temporal-convolutional student. It is a deliberately small CPU experiment, not an accuracy claim: it performs real optimization on real audio waveforms, exercises a Hindi→English boundary, and makes the teacher/student timing contract explicit.

The selected languages are English (`en`), Hindi (`hi`), Marathi (`mr`), Bengali (`bn`), Tamil (`ta`), Telugu (`te`), and Gujarati (`gu`). Hindi↔English is the priority switch case.

## Reproduce from a clean checkout

Python 3.10 and network access are required for the first run. `gTTS` downloads the audio and SpeechBrain downloads its checkpoint; no ffmpeg or GPU is used.

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

The defaults reproduce the submitted run: 800 optimizer steps, batch size 7, learning rate `1e-3`, and CPU only. On this 20-core/13 GB host the full cached run is comfortably under 30 minutes; network speed dominates a first run. `prepare_data.py --force` refreshes TTS audio. Audio, teacher targets, caches, checkpoints, and `.venv` are ignored by Git.

Important outputs are:

- `results/summary.json`: required machine-readable summary, including every optimizer-step loss.
- `results/train_metrics.json`: loss and gradient history plus the real-audio/no-NaN assertions.
- `results/eval_metrics.json`: per-clip agreement, policy settings, RTF, and switch lag.
- `results/switch_plot.png`: offline teacher posteriors and emitted student chunk-EMA posteriors.
- `results/teacher_metrics.json`: target-generation audit.

## Data and split

`scripts/prepare_data.py` synthesizes PCM WAVs at 16 kHz through gTTS and decodes MP3 responses in-process with `miniaudio`. The manifest contains 20 monolingual training clips (five each for English/Hindi and two each for the other five languages), seven held-out monolingual clips, one 4 s Hindi + 4 s English training concatenation, and two held-out 8 s switch clips in opposite directions. Thus `n_train_clips=21`; the training switch uses train utterances, while evaluation switches use held-out utterances.

TTS is permitted by the brief and makes the download deterministic in structure, but it is a severe domain limitation: there is effectively one synthetic voice per locale, no telephony codec, no room noise, and only one evaluated Hindi→English boundary. Results are plumbing checks, not production estimates.

## Teacher choice and targets

The frozen teacher is [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa). It is purpose-built for LID, directly exposes normalized scores for all seven chosen languages, uses full-window attentive statistics, and is cheap enough to run repeatedly on CPU. Whisper was the main alternative: it is robust and familiar, but language ID requires its encoder/decoder path and commonly pads to a 30 s input, making hundreds of temporal windows needlessly expensive. MMS-LID has broader coverage, but coverage is not the bottleneck in this seven-language demo and its stack is heavier.

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

## Student and latency budget

The student projects 40 log-mels plus four explicit right-context frames into 64 channels, then applies six causal depthwise-separable residual blocks with dilations `(1, 2, 4, 8, 16, 32)` and a 7-class head. Its finite receptive field is 127 frames (about 1.27 s of past), so it cannot retain an old language forever as an unconstrained recurrent state can. It has **42,567 parameters**. Chunked inference recomputes the small left overlap for clarity; a production kernel would cache convolution state. Whole-sequence and chunked logits are tested for equality.

The conservative worst-case algorithmic model latency is **435 ms**:

```text
25 ms analysis frame + (210 ms label delay + 40 ms lookahead) + 160 ms chunk = 435 ms.
```

Past context adds compute but no algorithmic latency. On one CPU thread, including log-mel extraction and the deliberately uncached overlap, measured RTF is **0.0064** (about 156× real time). Routing policy smoothing/dwell is separate from model latency.

## Submitted sanity results

These values are from the included `results/` artifacts, not aspirational numbers:

| Check | Result |
|---|---:|
| Optimizer steps on real audio | 800 |
| First 10-step mean KD loss | 6.5777 |
| Last 10-step mean KD loss | 0.8701 |
| Finite losses and gradients | yes |
| Held-out frame agreement with teacher | 0.8602 |
| Hindi→English detection lag | 1,655 ms |
| Student parameters | 42,567 |
| Single-thread CPU RTF | 0.0064 |

The lag uses the true 4.0 s join and the first committed English decision at 5.655 s. The illustrative policy averages each 160 ms chunk, applies an EMA with new weight 0.30, and requires posterior ≥0.60, a 0.10 margin, and three consecutive chunks. That operating point has not been calibrated on this tiny dataset; `DESIGN.md` describes how to set it properly.

## Tests and repository map

- `tests/test_alignment.py` proves `teacher[i] ↔ student[i+D]`, validates the valid-tail mask, and fails for naïve undelayed alignment.
- `tests/test_causality.py` changes every feature after `t+L` and requires outputs through `t` to be bit-identical; it also checks chunk/full equivalence.
- `scripts/prepare_data.py`, `teacher_targets.py`, `train.py`, and `eval.py` are the single entry points for each stage.
- `src/streaming_lid/` holds configuration, frontend, model, loss, and dataset code.
- `DESIGN.md` is the Part 2 live-ASR design.

## Implemented versus intentionally out of scope

Implemented: reproducible audio acquisition; ffmpeg-free decode; frozen real teacher inference; soft temporal targets; special handling of switch clips; streaming-safe features; bounded-lookahead causal model; delayed KL; real backward/optimizer steps; held-out agreement; exact chunk-equivalence and causality tests; RTF; hysteretic switch measurement; and the requested plot/JSON outputs.

Intentionally not implemented: a real telephony/VAD frontend, an ASR server/router, probability calibration on representative calls, an unknown-language head, checkpoint export/quantization, or convergence training. With more compute/data I would train on speaker-disjoint FLEURS/Common Voice plus anonymized 8 kHz call audio, add codec/noise/reverb augmentation and an `other` class, tune thresholds on a cost-weighted dev set, and report confidence intervals, false switches/hour, miss rate, and lag percentiles.

## Final summary and open issues

The complete CPU path works: it downloads audio, obtains frozen offline teacher targets, executes 800 real optimizer steps without NaNs, preserves causal/bounded-lookahead behavior, reaches 86.0% held-out teacher agreement, and detects the held-out Hindi→English switch. The current 1.655 s switch lag and synthetic single-voice data are the main open issues; the metrics are sanity checks only, the policy is uncalibrated, and seven-way renormalization cannot reject an unsupported language. Git commits could not be created in this execution sandbox because this worktree's writable directory points its Git metadata to a read-only parent worktree; no commit or push was made.
