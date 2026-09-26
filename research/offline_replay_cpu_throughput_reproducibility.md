# Reproducible offline-replay CPU throughput (METHOD-48)

Research and Hugging Face Hub audit: **2026-09-26 (Asia/Calcutta)**.

## Decision

Do not present `1 / 0.0171435 = 58.3x real time` as a reproducible speed
factor. It is the correct reciprocal of one five-repeat process median, but an
otherwise equivalent process on the same named machine recorded a median RTF
of `0.0883288`, or 11.3x real time. The two medians differ by **5.1523x** and
their individual-repeat ranges do not overlap.

The current benchmark is still useful as a scoped compute smoke test. Rename
its primary quantity to `offline_replay_frontend_student_wall_rtf`, run it in
multiple fresh, sequential processes under a recorded affinity/threading
contract, and report the distribution of **process medians**. Keep RTF primary;
derive an x-real-time number only from the aggregate process median and always
show its dispersion.

This change has **zero expected model or accuracy gain**. Its expected gain is
honest, repeatable measurement. The controlled median, dispersion, and cause
of the existing 5.15x gap are **unverified**.

## What the two artifacts establish

The comparison is between autosave commits
`ea58c4aeda6b960ef4d72a7a1380b2211294b25b` (2026-09-26 14:23:30 IST) and
`80c65e3c2a8d7b7c830fe520e96754b836682546` (2026-09-26 14:43:41 IST).
Read-only reconstruction found:

| Quantity | Earlier process | Later process |
|---|---:|---:|
| Stored RTF repeats | 0.088329, 0.069447, 0.115597, 0.139778, 0.086933 | 0.023586, 0.017836, 0.017143, 0.012417, 0.014229 |
| Median wall RTF | 0.088328768 | 0.017143464 |
| Median x real time | 11.321x | 58.331x |
| Min--max RTF | 0.069447--0.139778 | 0.012417--0.023586 |
| Within-process max/min | 2.0127x | 1.8995x |
| Population CV of five repeats | 24.76% | 22.38% |
| Median wall time over 78.97025 s audio | 6.9753 s | 1.3538 s |

The later process is not merely different from the earlier one; each process
is itself unstable. In the later run, the median of the last two repeats is
35.67% below the median of the first two. In the earlier run it is 43.69%
above. Five same-process samples after one partial warm-up do not demonstrate
a steady state.

Several important identities are held fixed:

- `scripts/eval.py` is byte-identical at both commits (whole-file SHA-256
  `147b8819...d7715`), including `benchmark_rtf()`.
- The student state is identical (`b96a856c...b42273`).
- The 94-file audio identity is identical (`f0946a45...e589a67`), as are the
  manifest and target identities. The timed subset is the same 21 held-out
  monolingual waveforms, totalling exactly 1,263,524 samples / 78.97025 s at
  16 kHz.
- Both artifacts declare six Torch intra-op threads and five repeats.

The checkpoint file hashes differ because the later release adds executed
source/environment evidence around the same model state. That does not explain
the timed-body result. Conversely, byte-equivalent code and data do not make
wall time deterministic.

## What is actually timed

[`benchmark_rtf()`](../scripts/eval.py#L184-L205) does the following:

1. all WAVs and the checkpoint are loaded before timing;
2. one held-out waveform passes through the frontend and student as a warm-up;
3. each of five timed repeats processes all 21 in-memory waveforms in fixed
   order through whole-waveform log-mel extraction and
   `model.streaming_forward()`;
4. elapsed `time.perf_counter()` seconds are divided by 78.97025 audio seconds;
5. the within-process median is stored as `cpu_rtf`.

Thus the timed scope includes frontend compute, Python scheduling, tensor
allocation, and the intentionally uncached overlap in streaming replay. It
excludes WAV I/O, checkpoint/model construction, the offline ECAPA teacher,
accuracy scoring, the router, and a real incremental audio frontend. It is an
offline throughput workload, not first-decision latency or a service SLO.

The name `cpu_rtf` is also imprecise. Python defines `perf_counter()` as a
monotonic **wall** clock that includes time elapsed while the process is not
running, whereas `process_time()` accumulates the process's user and system CPU
time and excludes sleep. Both are useful, but they answer different questions
([Python 3.10 `time` documentation, accessed 2026-09-26](https://docs.python.org/3.10/library/time.html#time.perf_counter)).
The stored value is CPU-device wall RTF, not process-CPU RTF.

## Missing controls in the current artifact

The run identity captures Python and package versions, project files, model,
audio, targets, and `training.threads=6`. It does **not** capture:

- CPU affinity, topology, NUMA placement, or whether six distinct physical
  cores were available;
- Torch inter-op threads or the effective OpenMP/MKL thread and binding
  settings;
- host load, run queue, competing work, cgroup quota/throttling, memory/swap
  pressure, or thermal/power/frequency state;
- the clock implementation/resolution, process CPU time, context switches, or
  per-phase timing;
- process start time, process ID, warm-up measurements, or an independent
  process/run identifier.

Only one waveform is warmed, although the timed unit is the complete
variable-duration corpus. PyTorch's benchmark utility explicitly performs
warm-ups because kernels may be lazily initialized, fixes the threadpool size,
and retains multiple measurements because run-to-run variation is a confounder
([`torch.utils.benchmark`, updated 2026-05-07](https://docs.pytorch.org/docs/stable/benchmark_utils.html)).
`blocked_autorange` is useful for the frontend/model sub-benchmarks, but it
does not replace fresh-process replication of this end-to-end corpus sweep.

PyTorch's CPU tuning guidance identifies OpenMP thread count, affinity, thread
migration, cache locality, and NUMA placement as performance-relevant controls
([Performance Tuning Guide, updated 2025-07-09](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)).
The evaluator calls `torch.set_num_threads(6)` after imports but never sets or
records `torch.set_num_interop_threads`; the latter must be set before inter-op
parallel work starts
([PyTorch API, accessed 2026-09-26](https://docs.pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html)).
Whether inter-op parallelism materially affects this eager student is
**unverified**; it should still be fixed and recorded.

## Current-host audit, not historical attribution

A read-only audit at 2026-09-26 15:05--15:08 IST observed:

- WSL2 kernel `6.18.33.2-microsoft-standard-WSL2` on an Intel Core i5-13500HX;
- 20 Linux-visible logical CPUs, 10 visible core IDs, SMT width 2, one NUMA
  node, and process affinity `0-19`;
- CPython 3.10.12; PyTorch 2.6.0+cpu built with MKL 2024.2, oneDNN 3.5.3,
  OpenMP, and AVX2;
- before the evaluator's override, Torch reported 10 intra-op and 10 inter-op
  threads; no `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, or OpenMP binding variable
  was present in this diagnostic shell;
- guest load averages `8.26 / 9.41 / 14.68` and unrestricted `0-19` affinity;
- no guest `cpufreq` driver/governor/current-frequency interface.

This proves that the current development environment is virtualized and was
not idle during the audit. It does **not** reconstruct the load, affinity,
power mode, temperature, or frequency during either stored evaluator process.
Microsoft documents WSL2 as a managed lightweight VM and exposes VM-level CPU
allocation through `.wslconfig`
([WSL settings, accessed 2026-09-26](https://learn.microsoft.com/en-us/windows/wsl/wsl-config));
its FAQ also warns that WSL differs from a traditional production environment
([WSL FAQ, accessed 2026-09-26](https://learn.microsoft.com/en-us/windows/wsl/faq)).
The Linux guest cannot establish the host's power/frequency state when those
interfaces are absent.

Linux itself documents dynamic frequency selection as a function of policy and
load, and notes that thermal or power capacity can limit sustainable maximum
frequency
([CPUFreq documentation, accessed 2026-09-26](https://www.kernel.org/doc/html/latest/admin-guide/pm/cpufreq.html)).
Linux affinity constrains where a task may run and can reduce migration/cache
cost, but it cannot make competing work or host power state disappear
([`sched_setaffinity(2)`, Linux man-pages, accessed 2026-09-26](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html)).

Plausible contributors therefore include scheduler contention/pre-emption,
thread placement and migration, frequency/power/thermal state, incomplete
warm-up, allocator/cache state, and unrecorded thread-runtime settings. Their
individual contributions—and any claim that the new source-stage launcher
caused the speedup—are **unverified**.

## Proposed measurement contract

### 1. Freeze the workload and the claim

Run a standalone benchmark child from the same verified source stage as the
evaluator. Bind:

- executed-source root, entry point, checkpoint and model-state hashes;
- ordered clip IDs, every WAV hash, aggregate audio hash, sample rate, exact
  sample counts, `total_audio_seconds=78.97025`, and fixed traversal order;
- frontend config, `chunk_frames=16`, dtype, device, inference/eval mode, and
  student output digest;
- `scope="whole_waveform_frontend_plus_uncached_streaming_replay"`, plus
  explicit `includes` and `excludes` lists.

The teacher must not be loaded by this child. Store
`teacher_in_timed_region=false`. Keep separate optional measurements for
frontend-only, model-only, cold model load, and a future truly incremental
frontend; do not add unlike scopes into one RTF.

### 2. Make a process the outer replicate

Proposed smoke protocol, deliberately feasible on this CPU host:

- **7 fresh child processes**, sequential rather than concurrent;
- in each child, **2 complete untimed corpus sweeps**, then **10 complete timed
  sweeps**;
- fixed six-CPU affinity selected from six distinct Linux-visible core IDs,
  derived from topology rather than hard-coded;
- before any parallel operation, set and verify
  `torch.set_num_threads(6)`, `torch.set_num_interop_threads(1)`,
  `OMP_NUM_THREADS=6`, `MKL_NUM_THREADS=6`, and one documented OpenMP binding
  policy;
- run with other team training/evaluation jobs stopped. Record but do not
  silently "correct" load, cgroup, memory/swap, and available
  frequency/governor/thermal fields. Missing WSL host power data must remain
  `null` with `environment_class="WSL2_observational"`.

Seven processes and ten sweeps are project smoke-test choices, not a published
universal minimum. Performance-methodology work recommends repeating at the
level where uncertainty is largest—build, execution, or iteration—rather than
spending every repeat inside one execution
([Kalibera and Jones, *Rigorous Benchmarking in Reasonable Time*, 2013](https://kar.kent.ac.uk/33611/)).
Here the observed discontinuity makes fresh execution the necessary outer
level.

For comparing two implementations, launch them from separate clean children
in a predeclared randomized or balanced `AB/BA` order. Do not run every A
process first and every B process later; innocuous setup/order changes can
create measurement bias
([Mytkowicz et al., ASPLOS 2009](https://sape.inf.usi.ch/publications/asplos09.html)).

### 3. Record dual clocks and enough diagnostics

For every measured sweep record:

- `wall_ns = perf_counter_ns()` delta;
- `process_cpu_ns = process_time_ns()` delta;
- wall RTF and process-CPU-seconds per audio second;
- start/end monotonic timestamps, affinity, effective Torch thread counts,
  load averages, voluntary/involuntary context-switch deltas where available,
  and output digest.

Wall RTF remains the deployment-relevant throughput observation. Process CPU
time is a diagnostic, not a replacement: roughly stable process CPU with
inflated wall time would support a scheduling/wait hypothesis, while both
moving together would redirect investigation toward placement, frequency,
warm state, or backend behavior. Neither pattern by itself proves a cause.

Use `torch.utils.benchmark.Timer(num_threads=6).blocked_autorange(...)` only
for the separately labelled frontend/model micro-breakdown. Its documented
warm-up and repeated-measurement behavior is preferable to a one-shot timer,
but the outer seven-process structure must remain.

### 4. Aggregate at the process level

Store all ten sweeps, but compute one median per child. The primary summary is
then the seven process medians:

- median wall RTF;
- Q1/Q3 and IQR;
- min/max and max/min ratio;
- median absolute deviation;
- all seven values in launch order.

Do not pool the 70 correlated within-process sweeps and pretend `n=70`
independent runs. If a confidence interval is added, resample **processes**, not
sweeps, and disclose that seven outer replicates give a coarse interval.
Report `x_realtime = 1 / median_process_rtf` only beside the RTF dispersion;
transform interval endpoints in reverse order (`[1/upper, 1/lower]`).

The following is a proposed release smoke gate, not a literature standard:

```text
all identities and output digests equal
AND max(process_median_rtf) / min(process_median_rtf) <= 1.20
AND for every process,
    abs(median(last_5) / median(first_5) - 1) <= 0.10
```

If it fails, retain every observation, set
`throughput_stability_passed=false`, suppress a scalar "Nx faster" headline,
and report the process-median range plus the uncontrolled fields. A later
bare-metal run may supersede a WSL observation only as a new environment, not
by overwriting it. MLPerf's much larger protocol is not required here, but its
core rule that the system under test includes performance-relevant hardware
and software and that unreplicable results are invalid is a useful precedent
([MLPerf Inference rules, accessed 2026-09-26](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)).

## Suggested result schema

```json
{
  "schema_version": 1,
  "metric": "offline_replay_frontend_student_wall_rtf",
  "scope": "whole_waveform_frontend_plus_uncached_streaming_replay",
  "teacher_in_timed_region": false,
  "input": {
    "n_clips": 21,
    "total_samples": 1263524,
    "sample_rate": 16000,
    "total_audio_seconds": 78.97025,
    "ordered_clip_ids_sha256": "...",
    "audio_files_sha256": "f0946a45...e589a67"
  },
  "identity": {
    "executed_source_root_sha256": "...",
    "checkpoint_sha256": "...",
    "model_state_sha256": "b96a856c...b42273",
    "output_digest_sha256": "..."
  },
  "system": {
    "environment_class": "WSL2_observational",
    "platform": "...",
    "cpu_model": "...",
    "logical_cpus": 20,
    "visible_core_ids": 10,
    "affinity_cpu_ids": [0, 2, 4, 6, 8, 10],
    "governor": null,
    "host_power_mode": null
  },
  "threading": {
    "torch_intraop": 6,
    "torch_interop": 1,
    "omp_num_threads": 6,
    "mkl_num_threads": 6,
    "binding_policy": "..."
  },
  "protocol": {
    "fresh_processes": 7,
    "full_corpus_warmups_per_process": 2,
    "measured_sweeps_per_process": 10
  },
  "processes": [
    {
      "process_index": 0,
      "wall_rtf_runs": ["..."],
      "process_cpu_seconds_per_audio_second_runs": ["..."],
      "median_wall_rtf": "...",
      "first_half_median_wall_rtf": "...",
      "last_half_median_wall_rtf": "...",
      "diagnostics": {"...": "..."}
    }
  ],
  "aggregate": {
    "process_median_wall_rtfs": ["..."],
    "median_wall_rtf": "...",
    "q1_wall_rtf": "...",
    "q3_wall_rtf": "...",
    "min_wall_rtf": "...",
    "max_wall_rtf": "...",
    "max_min_ratio": "...",
    "x_realtime": "...",
    "throughput_stability_passed": "..."
  }
}
```

The example affinity list is illustrative; the harness must derive and verify
a valid six-core set on the actual host. Strings shown as `"..."` are schema
placeholders, not measured values.

## Acceptance tests

1. A pure aggregator fixture treats seven process medians as `n=7`, is
   invariant to the order of sweeps within a process, and does not reproduce a
   falsely narrow pooled-70 interval.
2. Exact boundary fixtures pass at max/min `1.20` and half-drift `0.10`; a
   one-ULP larger value fails. Non-positive, non-finite, or empty timing arrays
   fail.
3. The current two historical process medians fail the dispersion gate by
   construction (`0.0883288 / 0.0171435 = 5.1523`).
4. Changed source, checkpoint, model state, clip order, WAV, sample count,
   frontend, chunk size, dtype, output digest, affinity, or thread setting
   prevents aggregation.
5. A worker whose observed affinity is wider/narrower than declared or whose
   Torch thread counts differ fails before timing.
6. Missing WSL governor/power data remains explicit `null`; it cannot be
   serialized as `performance` or `fixed`.
7. A fake clock proves wall and process clocks are stored separately and that
   reciprocal interval endpoints are reversed correctly.
8. Teacher construction inside the timed child fails the scope test. Frontend,
   model-only, cold-start, and incremental measurements require distinct
   metric names.
9. The result preserves every failed/noisy process row and never replaces an
   older artifact in place.

## Hugging Face Hub audit

The live Hub API on 2026-09-26 still resolves
[`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
to full revision
[`0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/commit/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9),
last modified 2024-11-27, with nine repository siblings, SpeechBrain/audio-
classification metadata, and Apache-2.0 license. The card describes a 16 kHz,
utterance-level ECAPA classifier and publishes no CPU RTF for this student's
replay workload.

This pin remains important to teacher-target provenance, but it cannot explain
or validate the submitted student RTF because the teacher is absent from the
timed region. No newer Hub artifact changes the METHOD-48 conclusion.

## Bottom line

The current `0.0171` number is a valid record of one opportunistic process, not
a stable host capability. Preserve it and the `0.0883` predecessor as evidence
of measurement variance. The next defensible artifact is a separately named,
source/input-bound seven-process distribution with fixed/verified threading
and affinity, dual clocks, full-corpus warm-up, and an explicit WSL limitation.
Until that exists, the honest public statement is: **offline replay was faster
than real time in both observed processes, with process medians spanning
0.0171--0.0883 RTF; reproducible throughput is unverified**.
