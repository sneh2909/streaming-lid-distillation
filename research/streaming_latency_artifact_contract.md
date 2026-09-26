# Precise streaming-latency artifact and scheduler contract (POLISH-7)

Research cutoff: **2026-09-26**. Web sources, Hugging Face Hub cards/API
metadata, and the local repository were checked on that date. Publication and
artifact dates are stated below. Any production claim not measured by this
repository is marked **unverified**.

## Bottom line

The serialized `algorithmic_latency_ms=435` is not an exact property of the
student. It is a **conservative, zero-compute, frame-start input-buffer bound
under one assumed 16-frame scheduler**:

```text
25 ms analysis window + 250 ms aligned evidence + 160 ms whole chunk = 435 ms
```

There are three more precise facts:

1. The model's intrinsic evidence delay is exactly **250 ms from the teacher
   frame-end anchor**, or **275 ms from that frame's start**.
2. A fixed 16-output-frame schedule adds **0--150 ms**, not 0--160 ms. Its
   exact range is therefore **250--400 ms from the frame-end anchor** or
   **275--425 ms from frame start**; the phase-uniform means are 325 and 350
   ms. The current 435 ms number is safe but 10 ms loose.
3. No live waveform scheduler enforces that range. `streaming_forward` first
   computes the complete waveform frontend, then feeds 16 new feature frames
   per `streaming_step`; `chunk_availability_times` later reconstructs a
   different, output-aligned chunk phase. On the current 8 s Hindi-to-English
   fixture, the two interpretations assign different availability times to
   **763/773 (98.71%)** aligned outputs and average different logits into the
   policy chunks.

The current switch remains a precondition failure/miss under both chunk phases.
Nothing here improves model accuracy or rescues that result. The immediate fix
is to rename the scalar, serialize the exact range and reference clock, and
derive event times from actual emissions rather than reconstructing them after
concatenation.

## Scope and terminology

This memo addresses the builder backlog's `POLISH-7` request and the closely
coupled `POLISH-3` timing name. It does not change the target trajectory,
student delay, router threshold, or teacher-floor findings.

The word “latency” should always identify both its endpoints:

| Quantity | Start | End | Includes |
|---|---|---|---|
| Evidence delay | semantic frame start or frame-end anchor | arrival of the newest required audio sample | analysis window, label alignment, explicit lookahead |
| Scheduled input-buffer delay | semantic frame reference | scheduler invocation | evidence delay plus chunk phase |
| Compute response | latest required sample at ingress | emitted logit at a monotonic wall clock | frontend/model queue and compute |
| Policy decision lag | speech onset or language boundary | stable router commit | acoustic evidence, model, smoothing, threshold, dwell |
| Route handoff lag | speech onset or boundary | downstream ASR acceptance | all prior stages plus IPC/startup/queue |
| RTF | total audio duration | aggregate processing duration ratio | throughput only; it is not a response-time clock |

The repository already has a broader four-clock metric proposal in
[`streaming_lid_metrics.md`](streaming_lid_metrics.md). This note supplies the
missing exact student/scheduler arithmetic and exposes a phase mismatch in the
current evaluator.

## Exact sample ledger

### Frontend convention

The pinned local frontend uses 16 kHz audio, `center=False`, a 400-sample
(25 ms) window, and a 160-sample (10 ms) hop. PyTorch's official `torch.stft`
documentation states that without centering, frame `k` begins at
`k * hop_length`, and gives the corresponding sample-indexed transform
([PyTorch 2.14 documentation, accessed 2026-09-26](https://docs.pytorch.org/docs/2.14/generated/torch.stft.html)).
The repository is pinned to PyTorch/torchaudio 2.6.0; the local causality test
and code use this same convention.

Let:

```text
Fs = 16,000 samples/s
H  = 160 samples       # 10 ms hop
W  = 400 samples       # 25 ms window
D  = 21 frames         # delayed target alignment
L  = 4 frames          # explicit model lookahead
C  = 16 frames         # assumed scheduler chunk
```

Feature `k` uses waveform samples `[kH, kH + W)` and can first exist when the
exclusive sample `kH + W` has arrived. `extract_window` defines the teacher
anchor identically as `iH + W`.

Teacher target frame `i` is matched to student output `j=i+D`, and output `j`
needs feature `j+L`. Therefore:

| Event | Exact sample/time |
|---|---:|
| Teacher frame start | `iH` |
| Teacher frame-end anchor | `iH + W` |
| Aligned student index | `j = i + D` |
| Latest required feature | `j + L = i + 25` |
| Latest required sample, exclusive | `(i+D+L)H + W` |
| Evidence delay from frame-end anchor | `(D+L)H/Fs = 250 ms` |
| Evidence delay from frame start | `((D+L)H+W)/Fs = 275 ms` |

These are exact availability times under punctual incremental feature
extraction and zero compute. The 25 ms analysis window must not be added when
using the frame-end anchor because it is already part of that reference time.

### Fixed output-chunk phase

If output indices `qC ... qC+C-1` are released together only after the entire
chunk's right context exists, the chunk becomes available at:

```text
t_chunk(q) = ((qC + C - 1 + L)H + W) / Fs
```

For an aligned output `j`, chunk scheduling adds:

```text
(C - 1 - (j mod C))H / Fs = 0, 10, ..., 150 ms.
```

Thus the exact steady-state range is:

| Reference | Minimum | Mean over 16 phases | Maximum |
|---|---:|---:|---:|
| Teacher frame-end anchor | 250 ms | 325 ms | 400 ms |
| Teacher frame start | 275 ms | 350 ms | 425 ms |

The existing constant adds `C*H=160 ms`; the exact worst chunk phase is
`(C-1)*H=150 ms`. Consequently, **435 ms is a conservative frame-start bound,
not an exact worst case**. Calling it simply “algorithmic latency” also hides
the chosen reference clock.

If the model is invoked as soon as each feature makes an output stable, there
is no extra chunk wait and the delays are 250/275 ms. Conversely, because the
public feature-level API accepts arbitrary call sizes and deadlines, a delayed
caller can add unbounded wait. The architecture alone therefore cannot enforce
a finite 425 ms service guarantee.

## The current replay and timestamp reconstruction use different phases

### What the stateful helper does

`streaming_forward(features, chunk_frames=16)` calls `streaming_step` with 16
new feature frames at a time. After receiving `r` frames, `streaming_step`
emits every output index below `r-L`. It therefore emits:

```text
first call: C-L = 12 logits (indices 0..11)
later full calls: C = 16 logits (12..27, 28..43, ...)
```

If each call occurs as soon as its last input feature is available, output `j`
is emitted at the first received-frame boundary `r` satisfying `j+L < r`.
For full chunks:

```text
r = C * ceil((j + L + 1) / C)
t_emit_audio(j) = ((r-1)H + W) / Fs.
```

This schedule has the same 275--425 ms frame-start range, but its phase is
shifted by four frames relative to output-aligned chunks.

### What the evaluator reconstructs

`chunk_availability_times` receives only the already concatenated stable-logit
count. It groups output indices `0..15`, `16..31`, and so on, then timestamps
each group from output-chunk end plus `L`. That corresponds to initially
buffering `C+L` feature frames and then advancing by `C`; it is not the call
sequence executed by `streaming_forward`.

Away from end-of-stream, the per-output difference is exact:

| `j mod 16` | Reconstructed minus direct `streaming_step` availability |
|---|---:|
| `0..11` | `+40 ms` |
| `12..15` | `-120 ms` |

More importantly, the policy averages different logits. Under the direct call
sequence, its first trained group contains outputs `21..27`; the evaluator's
first reconstructed group contains `21..31`. Subsequent groups remain shifted
by four output frames. This can change EMA/threshold/dwell behavior, not merely
move an otherwise identical event by 40 ms.

### Frozen-fixture audit

The audit used checkpoint SHA-256
`bb2a7e2aee1796845c1880f618b0b465e025ace3912d860e02b2a63ac6c962e8`
and the current 8.0 s `switch_hi_en_eval` waveform. It was read-only, with six
Torch CPU threads.

| Observation | Direct 16-feature calls | Current reconstructed output chunks |
|---|---:|---:|
| Feature frames | 798 | 798 |
| Stable logits | 794 | 794 |
| Trained/aligned outputs (`j>=21`) | 773 | 773 |
| Policy chunks | 49 | 49 |
| First policy-chunk availability | 335 ms | 375 ms |
| Initial Hindi commit | none | none |
| Hindi-to-English commit | none | none |

Across the 49 paired policy chunks, mean posterior L1 distance was **0.150856**,
maximum L1 distance was **0.452587**, and maximum single-class absolute change
was **0.224690**. At output level, the two schedules timestamped 763/773
aligned outputs differently: 571 were reconstructed 40 ms late, 188 were 120
ms early, four end-region outputs were 100 ms early, and ten final outputs
matched.

This is deterministic evidence about the current fixture and code, not a
population result. Because both exact detector replays still fail to arm the
initial Hindi state, the persisted `switch_lag_ms=null` remains the correct
outcome. The effect on a future successful switch is **unverified** and could
include a changed decision outcome because the posterior chunks differ.

## Throughput is also named too broadly

`benchmark_rtf` processes the whole waveform through `LogMelFrontend` before
calling feature-level `streaming_forward`. It times useful CPU work, including
the intentionally recomputed TCN overlap, but it does not exercise:

- an incremental waveform/STFT buffer;
- real chunk arrival or scheduler deadlines;
- queueing, policy compute, or route handoff;
- a production convolution cache.

The current `cpu_rtf=0.0086737` is therefore an
`offline_replay_compute_rtf`, not online p50/p95 response latency and not a
bound on it. Production per-chunk CPU timing is **unverified**.

The detector's `dwell_ms=3*160=480` has a similar naming issue. Three
qualifying chunks span 480 ms of chunk evidence, but after the first qualifying
chunk only two further chunk intervals—nominally 320 ms—are required. EMA may
delay the first qualifier by additional, data-dependent chunks. Call 480 ms a
`qualifying_span_ms`, not a guaranteed policy-delay increment.

## External evidence and Hugging Face Hub audit

These sources support the reporting contract; none validates this LID model's
accuracy, Hindi-English switch behavior, or CPU service latency.

### Primary and official sources

- [PyTorch `torch.stft` documentation](https://docs.pytorch.org/docs/2.14/generated/torch.stft.html)
  (living documentation, accessed 2026-09-26) fixes the sample convention used
  above: with `center=False`, frame `k` starts at `k*hop_length`.
- [NVIDIA NeMo 26.02 cache-aware streaming documentation](https://docs.nvidia.com/nemo-framework/user-guide/26.02/nemotoolkit/asr/models.html#cache-aware-streaming-conformer)
  (version 26.02, accessed 2026-09-26) explicitly notes that positions within a
  chunk receive different amounts of future context and that its latency
  figures assume zero network forward time. This directly supports reporting a
  phase range and compute separately.
- [Noroozi et al., *Stateful Conformer with Cache-based Inference for Streaming
  ASR*](https://arxiv.org/abs/2312.17279) (submitted 2023-12-27; v3
  2024-05-02) separates constrained context from cache-based compute and trains
  streaming inference to match full inference. It supports a real cache and
  emission contract, not any LID performance claim.
- [Shi et al., *Emformer*](https://arxiv.org/abs/2010.10759) (submitted
  2020-10-21; v4 2020-12-30; ICASSP 2021) reports encoder-induced latency and
  decoding RTF separately. That distinction transfers; its ASR numbers do not.
- [Lin et al., *X2Streaming-ASR*](https://arxiv.org/abs/2609.08672) (v1
  2026-09-08; v2 2026-09-15) defines commit latency relative to forced-aligned
  acoustic endpoints. It reinforces that buffer availability and semantic
  commit lag need separate endpoints. Transfer to LID is **unverified**.

### Dated Hub artifacts

| Hub artifact | Snapshot/date/license metadata | Relevant reporting behavior | Limitation |
|---|---|---|---|
| [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md) | revision `f667ed73...`; last modified 2026-09-24; Hub license `openmdw-1.1` | Names **Input Buffer Latency**, defines it as `(CHUNK_LEN + RIGHT_CONTEXT)*80 ms`, and explicitly excludes compute; reports speed separately. | Speaker diarization, not LID; no evidence for this CPU student. |
| [`nvidia/parakeet-unified-en-0.6b`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b/blob/fe53cd885760c96b6a5f51a0bfd362cb4584a98b/README.md) | revision `fe53cd88...`; released 2026-04-07; last modified 2026-06-09; card governed by NVIDIA Open Model License (API license field was `other`) | Defines streaming latency as chunk plus right context and exposes 160--2,080 ms buffered configurations. | English ASR; current inference is buffered and recomputes left context. Reporting precedent only. |
| [`nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi/blob/ae98143333690bd7ced4bc8ec16769bcb8918374/README.md) | revision `ae981433...`; last modified 2025-02-18; CC BY 4.0 | Distinguishes worst from average latency for multiple lookaheads and points to cache-aware inference. | English ASR; its card contains a likely `480s` typo, so use the linked paper/config rather than copying labels. |

The strongest transferable pattern is simple: name input buffering, reference
clock, compute, and task-level commit separately. Calling all four “latency”
prevents meaningful comparison.

## Concrete, testable proposals

### 1. Replace the scalar with a sample-exact structured artifact

Keep `algorithmic_latency_ms=435` only as a deprecated compatibility field,
renamed in documentation to `conservative_frame_start_buffer_bound_ms`. Bind
these values into the run identity and results:

```json
{
  "semantic_reference": "uncentered_feature_frame_start",
  "frame_window_samples": 400,
  "frame_hop_samples": 160,
  "label_delay_frames": 21,
  "model_lookahead_frames": 4,
  "assumed_scheduler": "fixed_16_output_frame_chunks",
  "evidence_delay_from_frame_end_ms": 250,
  "evidence_delay_from_frame_start_ms": 275,
  "scheduler_wait_min_ms": 0,
  "scheduler_wait_max_ms": 150,
  "input_buffer_delay_from_frame_end_ms": [250, 400],
  "input_buffer_delay_from_frame_start_ms": [275, 425],
  "conservative_frame_start_buffer_bound_ms": 435,
  "compute_included": false
}
```

Expected model/accuracy gain: **none**. Expected reporting correction: exact
worst-case bound drops 10 ms, and frame-start versus frame-end ambiguity is
removed. A parameterized test should enumerate at least two full chunk cycles,
derive every value from `Fs/H/W/D/L/C`, assert all 16 phase delays, and fail if
the semantic reference or compute inclusion is absent.

### 2. Choose one scheduler and timestamp the actual emissions

The smallest change is to adopt the already executed **16-new-feature-frame
input cadence**. Have `streaming_step` (or its wrapper) return, for every call:

```text
output_start_frame, output_stop_frame,
latest_received_feature, latest_received_sample_exclusive,
audio_availability_seconds, emit_monotonic_seconds
```

The evaluator must average exactly the logits returned by each call and use
that call's captured audio timestamp. Do not concatenate logits and infer a
new output-chunk phase afterward.

A fake-clock regression with `C=16,L=4` should prove emission counts
`12,16,16,...`, no duplicate/provisional output, and exact sample closure. The
current 8 s fixture should then produce 798 features, 794 stable logits, 773
aligned outputs, 49 policy calls, and a first trained policy call at 335 ms;
its exact detector outcome must remain `precondition_failure`/miss. Retain the
old output-aligned schedule as a separate named test only if it is deliberately
supported.

Expected accuracy gain: **none guaranteed**. The change to future successful
event times and policy outcomes is **unverified**; the current trace's maximum
chunk-posterior L1 difference is 0.452587, so this cannot be treated as a
cosmetic timestamp rename.

### 3. Add real incremental timing before claiming an online SLO

Implement a stateful waveform frontend that accepts capture-timestamped PCM,
releases each uncentered frame only after its last sample arrives, and passes
those frames through the chosen scheduler. Record:

```text
t_latest_audio_sample, t_frontend_done, t_model_done,
t_policy_done, t_route_accepted
```

Report per-call p50/p95/p99/max frontend, model, policy, queue, and total
response, plus deadline misses against the exact input-buffer deadline. Until
that exists, rename `cpu_rtf` to `offline_replay_compute_rtf` and state that it
uses a whole-waveform frontend and uncached-overlap reference kernel. Rename
the detector's 480 ms field to `qualifying_span_ms=480` and report the nominal
two-interval post-first-qualifier value (320 ms) separately.

Expected model/accuracy gain: **none**. Production p95/p99 CPU response and
deadline-miss rates are **unverified**. Acceptance requires timestamp closure
within one audio sample plus 1 ms monotonic-clock tolerance on a fake-clock
test, no missed/duplicated waveform samples across arbitrary packet splits,
and a named CPU/hardware/software environment for live timings.

## Evidence and limitations

Local formulas and the frozen replay were derived from exact file contents:

```text
src/streaming_lid/config.py  292846ed...900894
src/streaming_lid/model.py   6e11216a...d9e01d
src/streaming_lid/audio.py   4cce5b81...6260cd
scripts/eval.py              dbf985e9...6357c0
checkpoint                   bb2a7e2a...6c962e8
```

Commands included repository `rg`/`sed`, exact integer sample arithmetic,
two read-only six-thread frozen-checkpoint replays, Hugging Face model API and
pinned-card queries, and primary/official web-source inspection. No model was
trained, no result artifact was overwritten, and no builder- or
experimenter-owned file was edited.

Remaining limitations:

- The exact ranges assume punctual feature production and zero compute; live
  ingress, OS scheduling, and CPU response are **unverified**.
- The current frontend is not stateful, so the proposed sample closure is not
  implemented.
- Current interpolated switch targets have their separate 260--490 ms
  future-evidence defect. Correct latency naming does not fix METHOD-0.
- The one frozen switch is synthetic and the student generalizes poorly. Its
  phase sensitivity is a deterministic regression case, not an estimate of
  production impact.
- All external artifacts are ASR or speaker-diarization precedents. Their
  latency terminology transfers; their quality and compute results do not.
