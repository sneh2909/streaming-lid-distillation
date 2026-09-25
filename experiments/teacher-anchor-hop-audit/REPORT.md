# Teacher anchor-hop audit

## Question

What is the coarsest true-call ECAPA anchor hop whose availability-valid
previous-anchor-hold trajectory is a faithful substitute for teacher calls at
every 10 ms feature frame?

## Setup

The experiment imports the main pipeline's exact
`[t-1.75 s, t+0.25 s]` window extractor, audio/manifest utilities, timing
constants, selected-language mapping, and pinned teacher artifact checks. It
does not edit main-pipeline code. The teacher is
`speechbrain/lang-id-voxlingua107-ecapa` at revision
`0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`, artifact SHA-256
`f193a054...c0f706`, run with six Torch CPU threads.

Inputs are the main pipeline's same 21 speaker-disjoint held-out monolingual
clips and both held-out switch clips, `switch_hi_en_eval` and
`switch_en_hi_eval`. Their manifest-record-plus-WAV fingerprint is
`b4e2916d7585c25c380dcaaddd582274a570a326da754962593a8b9bfb11c45d`.
Fresh full-utterance inference remained 21/21, and the fresh 250 ms anchor
frames, posteriors, and retained masses matched the current main cache exactly.

The current teacher window was called at true 250, 100, 50, and 10 ms grids.
Every sparse trajectory was expanded to the 10 ms feature grid by
previous-anchor hold; the 10 ms calls are a dense numerical reference, not
ground-truth language labels. Comparisons pool both switch clips and report
`KL(q_10ms || q_held)`, top-1 disagreement, retained seven-language mass,
500 ms-stable direction-aware crossings, top-1 changes, and generation time.
Known-label accuracy excludes a +/-250 ms collar around the nominal join.

The predeclared shortlist gate required pooled p95 KL at most 0.10, pooled
non-collar top-1 disagreement at most 2%, both stable events within 50 ms of
the dense trajectory, and no more than one extra top-1 change per clip. The
coarsest sparse grid passing every gate would be adopted. Model loading is
excluded from timing; timings are one warm run and therefore descriptive.

## Numbers

| True-call hop | Calls / request multiplier | Wall time / multiplier | Non-collar label accuracy | p95 KL to 10 ms | Non-collar top-1 disagreement | Stable-onset median | Max per-clip onset delta | Top-1 changes | Mean retained mass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 ms | 66 / 1.00x | 4.45 s / 1.00x | 71.32% | 0.9843 | 11.90% | 1,400 ms | 190 ms | 10 | 0.5341 |
| 100 ms | 162 / 2.45x | 10.28 s / 2.31x | 72.99% | 0.3326 | 6.75% | 1,325 ms | 90 ms | 18 | 0.5649 |
| 50 ms | 322 / 4.88x | 20.18 s / 4.54x | 72.99% | 0.1189 | 4.14% | 1,300 ms | 40 ms | 23 | 0.5601 |
| 10 ms reference | 1,596 / 24.18x | 101.69 s / 22.87x | 73.80% | 0 | 0% | 1,260 ms | 0 ms | 42 | 0.5622 |

All four grids detected both 500 ms-stable direction-aware events. The dense
semantic-onset lags were 1,335 ms for Hindi-to-English and 1,185 ms for
English-to-Hindi. Relative to those values, the per-clip onset delays were
`+190/+90 ms` at 250 ms, `+90/+40 ms` at 100 ms, and `+40/+40 ms` at
50 ms.

The 50 ms arm was closest, but it still missed both posterior-fidelity gates:
pooled p95 KL was 0.1189 rather than at most 0.10, and non-collar top-1
disagreement was 4.14% rather than at most 2%. Its per-clip p95 KL was 0.0987
on Hindi-to-English and 0.1374 on English-to-Hindi. Sparse grids had fewer
top-1 changes than the dense reference, so they passed the one-extra-change
gate, but this is smoothing by omission rather than proof of better targets.

Denser calls improve the current grid's median stable onset by only 140 ms
(1,400 to 1,260 ms). Even the 10 ms reference therefore remains far outside
the earlier target-window audit's 750 ms stable-onset requirement, while its
generation workload is 24.18x the current request count.

## Verdict: reject

Reject all three sparse grids under the predeclared dense-reference fidelity
gate; none is a defensible shortlist winner for a student rerun. In
particular, do not adopt 50 ms merely because it reproduces the two crossing
times: it still changes more than 4% of non-boundary top-1 frames and misses
the pooled KL threshold.

The result also rejects the hypothesis that anchor density alone rescues the
teacher's stable switch delay on these clips. It does **not** establish that
10 ms targets are intrinsically better: the dense arm is the reference by
construction, has 40 unmatched top-1 changes across two clips, and costs
24.18x as many teacher calls. A student target-by-delay grid should remain
blocked until a different availability-valid target trajectory meets the
latency and stability gates.

Only two reversed synthetic switches are available, and their nominal 4 s
joins contain source-audio silence. Treat this as a rejection guardrail, not a
population estimate. The canonical run identity is
`9b3e47e3c7cba82a4a8c4de8a9325dd1fb59c1ce8eff61ffece47ff8275dfd44`;
full traces, per-clip comparisons, artifact identities, and validation checks
are in `results.json`.
