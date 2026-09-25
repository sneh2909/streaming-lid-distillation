# Teacher bake-off

## Question

Should the main distillation pipeline replace its VoxLingua107 ECAPA teacher
with Whisper-small or MMS-LID-126 on the seven-language Hindi/English-focused
deployment set?

## Setup

- Input set: the main manifest's 21 current held-out monolingual clips (three
  clips for each of `en`, `hi`, `mr`, `bn`, `ta`, `te`, and `gu`) and its
  `switch_hi_en_eval` clip. The exact manifest records and WAV bytes hash to
  `52bca81e32db34edb0a9ad4529b01bd6ddd36aad6f587733a01ace669538d6fc`.
- Models: SpeechBrain
  [`lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
  at revision `0253049a`; OpenAI
  [`whisper-small`](https://huggingface.co/openai/whisper-small) at revision
  `973afd24`; and Meta
  [`mms-lid-126`](https://huggingface.co/facebook/mms-lid-126) at revision
  `53da6e31`.
- Monolingual examples: one centred 1 s, 2 s, and 4 s crop per held-out clip.
  A short utterance is symmetrically zero-padded to the requested duration; 15
  of 21 clips required padding at 4 s. Every crop was also passed through a
  16 kHz -> 8 kHz -> 16 kHz resampling bottleneck. This tests bandwidth loss,
  not a telephony codec.
- Scoring: the primary result restricts each native posterior to the main
  pipeline's seven languages before taking the argmax, matching target
  generation. Native 99/107/126-way argmax accuracy is retained as a
  diagnostic in `results.json`.
- Switch metric: the script imports the main pipeline's
  `scripts.teacher_targets.extract_window`, giving every teacher the same
  `[t-1.75 s, t+0.25 s]` window every 250 ms. "First persistent" is the first
  anchor in a run of three English anchors after an armed Hindi run;
  "confirmed" is the third anchor. Availability-adjusted lag adds the
  teacher window's 250 ms right context.
- Runtime: CPU only with `torch.set_num_threads(6)`. Timing includes model
  preprocessing and inference but excludes checkpoint load/download. Safe
  throughput batches were ECAPA 24, Whisper 2, and MMS 4. Across 159 requests
  per model, the timed audio duration was 360 s.
- MMS emitted Transformers' legacy `weight_g`/`weight_v` naming warning. The
  harness compares both loaded tensors with `model.safetensors`; both matched
  exactly, so no randomly initialized MMS weights entered the measurements.

Reproduction:

```bash
UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
  uv run --with transformers==4.44.2 \
  python experiments/teacher-bakeoff/run.py --fresh
```

## Numbers

Primary seven-way accuracy (correct clips out of 21):

| Teacher | Clean 1 s | Clean 2 s | Clean 4 s | 8 kHz 1 s | 8 kHz 2 s | 8 kHz 4 s |
|---|---:|---:|---:|---:|---:|---:|
| ECAPA | 18/21 (85.7%) | 19/21 (90.5%) | 21/21 (100%) | 18/21 (85.7%) | 18/21 (85.7%) | 18/21 (85.7%) |
| Whisper-small | 12/21 (57.1%) | 13/21 (61.9%) | 17/21 (81.0%) | 10/21 (47.6%) | 13/21 (61.9%) | 16/21 (76.2%) |
| MMS-LID-126 | 16/21 (76.2%) | 17/21 (81.0%) | 21/21 (100%) | 14/21 (66.7%) | 15/21 (71.4%) | 21/21 (100%) |

Across all three durations, ECAPA scored 58/63 clean and 54/63 after the 8
kHz bottleneck. Whisper scored 42/63 and 39/63; MMS scored 54/63 and 50/63.
MMS's notable cell-level win was 4 s at 8 kHz (21/21 versus ECAPA's 18/21),
but it lost to ECAPA at both shorter 8 kHz windows.

Restricted seven-way Hindi -> English switch behavior:

| Teacher | First persistent lag | Confirmation lag | Availability-adjusted lag | Anchor accuracy |
|---|---:|---:|---:|---:|
| ECAPA | 1,525 ms | 2,025 ms | 1,775 ms | 23/33 (69.7%) |
| Whisper-small | 1,525 ms | 2,025 ms | 1,775 ms | 26/33 (78.8%) |
| MMS-LID-126 | 1,525 ms | 2,025 ms | 1,775 ms | 23/33 (69.7%) |

The single clip and 250 ms anchor grid are too coarse to rank the tied switch
lags. Native full-label-space behavior was much less stable for ECAPA and MMS:
their first persistent English run moved to 3,025 ms, while Whisper remained
at 1,525 ms. That difference is relevant for unknown-language handling but not
for the current seven-way distilled target.

CPU throughput:

| Teacher | Parameters | Wall time | Process CPU time | Wall RTF | Relative wall time |
|---|---:|---:|---:|---:|---:|
| ECAPA | 21.25 M | 9.27 s | 54.83 s | 0.0258 | 1.0x |
| Whisper-small | 241.73 M | 147.15 s | 871.82 s | 0.4088 | 15.9x |
| MMS-LID-126 | 966.09 M | 78.28 s | 468.93 s | 0.2175 | 8.4x |

These are throughput measurements with model-specific batch sizes, not
single-request latency measurements. Full per-request predictions, selected
posterior mass, native-space results, model revisions, load validation, and
unrounded timings are in `results.json`.

## Verdict: reject

Reject replacing ECAPA with Whisper-small or MMS-LID-126 in the main pipeline.
ECAPA has the highest aggregate clean and 8 kHz seven-way accuracy, ties MMS at
4 s clean, and is substantially cheaper. Whisper has no accuracy or switch-lag
gain that justifies its 15.9x runtime. MMS's perfect 4 s 8 kHz cell is worth
remembering for a larger real-telephony evaluation, but it does not offset its
short-window losses and 8.4x runtime here.

Adopt the measurement, not a new checkpoint: report the teacher's own
restricted-space switch floor (1,525 ms first-persistent; 2,025 ms confirmed)
separately from student lag. Do not treat these 21 synthetic clips and one
switch as a production accuracy estimate; a teacher replacement would require
speaker-rich real 8 kHz calls, codec/noise conditions, multiple switches, and
confidence intervals.
