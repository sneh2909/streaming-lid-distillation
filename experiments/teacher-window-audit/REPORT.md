# Teacher target-window audit

## Question

Can a shorter, availability-valid ECAPA evidence window produce a stable
Hindi/English target transition within 750 ms, while retaining the current
window's non-boundary accuracy closely enough to justify a student
target-by-delay sweep?

## Setup

The experiment imports the main pipeline's audio/manifest utilities, timing
constants, label mapping, exact current-window extractor, and pinned teacher
artifact validation. It changes no main-pipeline code. The teacher is
`speechbrain/lang-id-voxlingua107-ecapa` at revision
`0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`, artifact SHA-256
`f193a054...c0f706`, on six Torch CPU threads.

Inputs are the main pipeline's same 21 speaker-disjoint held-out monolingual
clips and both held-out switch clips, `switch_hi_en_eval` and
`switch_en_hi_eval`. Their manifest-record-plus-WAV fingerprint is
`c0a4dc40eac162301d1ab08bf20644f053ea1d5e6fa2b777fe25a36b6db2f3c1`.
Fresh full-utterance inference reproduced 21/21 monolingual accuracy and the
main cache within `2.38e-7`; the current-window switch anchor frames and
posteriors matched the main cache exactly.

Six target windows were compared at the same true 250 ms ECAPA anchor grid:

- current `[t-1.75 s, t+0.25 s]`;
- bounded `[t-0.75 s, t+0.25 s]`;
- causal rolling `[t-0.5 s,t]`, `[t-1 s,t]`, and `[t-2 s,t]`;
- cumulative causal prefix `[0,t]` (left-zero-padded only while shorter than
  the teacher's 500 ms minimum input).

Every sparse trajectory is expanded with previous-anchor hold, never the
future-leaking linear interpolation in the current main cache. Accuracy is
computed on the two switch clips outside a +/-250 ms boundary collar. A
500 ms-stable event is the first direction-aware target run that remains the
target class for at least 500 ms after a correctly acquired source. Semantic
onset and online confirmation are reported separately: confirmation includes
the stability horizon and the window's future-context availability.

The predeclared shortlist gate required both switches, median stable semantic
onset at most 750 ms, descriptive p95 at most 1,250 ms, no more than a
2-point non-boundary accuracy loss, no more than one extra unmatched flip per
clip, and no more than 250 ms future context. Adoption additionally required
at least 250 ms improvement over the current window. With only two mirrored
switch clips, the p95 values are descriptive, not population estimates.

## Numbers

| Window | Outside-collar accuracy | First crossing median (ms) | 500 ms-stable onset median / p95 (ms) | Stable confirmation available median (ms) | Stable switches | Unmatched changes, total | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current 1,750 past + 250 future | 71.32% | 1,400 | 1,400 / 1,513 | 2,150 | 2/2 | 8 | 4.12 s |
| Bounded 750 past + 250 future | 63.37% | 525 | 1,275 / 1,500 | 2,025 | 2/2 | 20 | 2.03 s |
| Causal 500 past | 43.45% | 25 | 2,400 / 2,963 | 2,900 | 2/2 | 37 | 1.07 s |
| Causal 1,000 past | 61.43% | 775 | 1,525 / 1,750 | 2,025 | 2/2 | 21 | 2.01 s |
| Causal 2,000 past | 64.64% | 1,650 | 1,650 / 1,763 | 2,150 | 2/2 | 10 | 3.91 s |
| Cumulative prefix | 51.40% | 1,525 detected-only | 1,525 / 1,525 detected-only | 2,025 detected-only | 1/2 | 7 | 17.04 s |

The early target-language frame recall shows why a first crossing alone is
misleading:

| Window | Recall in first 0.5 s | First 1 s | First 2 s |
|---|---:|---:|---:|
| Current | 0.0% | 0.0% | 30.2% |
| Bounded 750 + 250 | 25.0% | 12.5% | 42.8% |
| Causal 500 | 73.0% | 37.5% | 49.0% |
| Causal 1,000 | 23.0% | 12.5% | 30.2% |
| Causal 2,000 | 0.0% | 0.0% | 17.8% |
| Cumulative prefix | 0.0% | 0.0% | 12.0% |

The causal 500 ms window crossed almost immediately and reached 73% target
recall in the first half-second, but it repeatedly left the target class: its
stable onset moved to 2,400 ms and unmatched changes rose from 8 to 37. The
short bounded window was the least-bad latency challenger, but gained only
125 ms in stable median onset, lost 7.95 accuracy points, and more than doubled
unmatched changes. The cumulative prefix never produced a 500 ms-stable
Hindi-to-English event.

## Verdict: reject

Reject all five alternative windows as inputs to the next student
target-by-delay grid. None passed the predeclared gate. In particular, the
apparently fast crossings from short causal context are transient confusions,
not usable stable targets; trading the current window for them would reduce
target accuracy and increase churn without meeting the latency goal.

This does not validate the current window: its own median stable semantic
onset is 1,400 ms, and stable confirmation is not online-available until
2,150 ms. It only means these exact shorter rolling/prefix alternatives are
not a defensible replacement. Do not run the delay sweep with them. A separate
true-call anchor-hop audit can still quantify 250 ms grid error, but denser
sampling cannot by itself erase the much larger stable-transition deficit.

The canonical run identity is
`609e6b9b4cd8b8ef6855cf6be87c02c3a349fe55a9ec27c1945c6434d092890d`.
All probability, input, artifact, source-snapshot, and current-cache parity
checks passed. Full traces and machine-readable metrics are in
`results.json`.
