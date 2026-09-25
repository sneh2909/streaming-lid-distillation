# Target-type ablation

## Question

With the student architecture and 250 ms evidence budget fixed, should its ECAPA
targets be one repeated full-utterance posterior, a local centred window, or a
growing prefix ending at the emission horizon?

## Setup

The experiment used the main manifest without adding or replacing clips: 71
training clips, 21 speaker-disjoint held-out monolingual clips, and both held-out
switch clips (`switch_hi_en_eval` and `switch_en_hi_eval`). The evaluation-input
SHA-256 is `7b78eb2981dacc8a01b1e1fad6b98da1fbceebdb0aba4a2597751f36ebe94a79`.
For direct comparison with the teacher bake-off, the 21 monolingual clips plus
the Hindi-to-English switch still hash to
`52bca81e32db34edb0a9ad4529b01bd6ddd36aad6f587733a01ace669538d6fc`.

All three arms imported the main frontend, dataset/collator, TCN, delayed-KL
loss, teacher interpolation/label mapping, and chunk timing/smoothing code. Each
used the same 42,567-parameter TCN, seed 7, initial parameters, shuffled batch
order, 1,600 AdamW updates, batch size 7, learning rate `1e-3`, temperature 2,
210 ms label delay, 40 ms model lookahead, and one-second early-evidence ramp.
Six Torch threads were used throughout.

The only changed variable was the teacher target:

- `full_utterance`: classify the whole clip once and repeat its posterior. Its
  future context is unbounded, and a mixed clip receives one constant target.
- `centred_2s`: classify `[t-1 s, t+1 s]` every 250 ms and interpolate. This is
  a diagnostic, not a valid target at the tested latency: it uses 750 ms more
  future audio than the student's 250 ms evidence budget.
- `prefix_delta250`: classify the growing prefix `[0, t+250 ms]` every 250 ms.
  Its right edge exactly matches the latest evidence available to the delayed
  student, so this is the only context-valid arm.

Held-out student accuracy is against known synthesis language. “Own agreement”
is agreement with that arm's teacher trajectory. The common main teacher is
also reported in `results.json`; on the monolingual held-out set it is 100%
correct, so common-teacher agreement equals known-label accuracy. Switch
accuracy excludes no frames in one column and excludes a +/-250 ms boundary
collar in the table below. A miss remains a miss rather than receiving a
fabricated lag. Excess flip-flops are chunk-top-1 changes beyond the one intended
source-to-target transition.

Command:

```bash
UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
  uv run python experiments/target-type-ablation/run.py --fresh
```

The completed measurements were restarted from validated target caches after
the main pipeline added stable-tail withholding and the all-invalid-loss guard;
`results.json` binds the final run to pipeline-source SHA-256
`83937a4c48b1167afb7257fd6c569587c902e424f33c07f6ca2a8bae8aa14ce9`.

## Numbers

| Target | Context-valid | Teacher held-out label macro | Student own agreement macro | Student held-out label macro | Student clip accuracy | Teacher switch label macro | Student switch accuracy outside collar | Policy switches | EMA excess flips/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full utterance | no | 100.00% | 17.76% | 17.78% | 5/21 (23.81%) | 50.00% | 0.00% | 0/2 | 51.18 |
| Centred 2 s | no | 90.83% | 15.76% | 17.93% | 4/21 (19.05%) | 88.16% | 0.21% | 0/2 | 39.37 |
| Prefix through `t+250 ms` | yes | 86.19% | 13.64% | 18.21% | 3/21 (14.29%) | 58.80% | 0.28% | 0/2 | 39.37 |

The prefix arm's 18.21% frame macro is only **0.43 percentage points** above
full utterance and accompanies a two-clip drop in clip accuracy. All six
student/held-out-switch combinations miss both the raw persistent transition
and the threshold/EMA/dwell policy transition. Their switch accuracy is near
zero even away from the boundary, so this is not merely a lag-scoring effect.

The teacher trajectories explain why none of the pure targets is suitable:

| Target | Hindi -> English teacher transition | English -> Hindi teacher transition |
|---|---:|---:|
| Full utterance | missed (constant target) | missed (constant target) |
| Centred 2 s | +775 ms semantic; +1,775 ms when its 1 s future is available | +525 ms semantic; +1,525 ms available |
| Prefix through `t+250 ms` | missed | +1,275 ms semantic; +1,525 ms available |

The cumulative prefix retains too much pre-switch history: its teacher frame
accuracy is 38.68% on Hindi-to-English versus 78.91% on the mirrored
English-to-Hindi clip. The centred teacher represents both transitions better,
but only by using unavailable future context and still does not produce a
student that transfers to the held-out voices.

Optimization itself worked in every arm:

| Target | First-10 mean KD | Last-10 mean KD | Training wall time |
|---|---:|---:|---:|
| Full utterance | 6.8951 | 1.8425 | 29.38 s |
| Centred 2 s | 6.0588 | 2.0996 | 28.20 s |
| Prefix through `t+250 ms` | 5.9523 | 1.7328 | 29.30 s |

All losses, gradients, and post-update parameters were finite. The full-target
held-out arrays matched the main cache exactly (`max_abs_difference=0` for raw
and softened posteriors), and the run verified identical initialization and
minibatch order across arms plus unchanged inputs and pipeline source.

## Verdict: reject

Reject all three **pure** target regimes as replacements for the current mixed
strategy at this latency and data scale.

- Full-utterance targets are structurally wrong for code-switching: they cannot
  contain a boundary.
- Centred two-second targets violate the evidence budget by 750 ms and yielded
  no meaningful held-out or switch gain even with that extra information.
- The only deployable arm, cumulative prefix through `t+250 ms`, provides no
  decision-level improvement, misses the Hindi-to-English teacher transition
  itself, misses both student switches, and raises held-out frame macro by only
  0.43 points while reducing clip accuracy from 5/21 to 3/21.

Keep the main regime provisionally—full-utterance targets only for known
stationary clips and a bounded local target for switch clips—but do not read
this as validation of the current local window: the submitted hybrid student
also generalises poorly and misses its switch. The next useful experiment is
the research proposal's teacher-only boundary audit of shorter fixed bounded
and causal rolling windows. Only after finding a target that transitions within
the latency budget should a target-by-delay student grid be trained.

This is one seed on a tiny synthetic set with one training switch and two
mirrored held-out switches that reuse the same component audio. The policy is
uncalibrated. The result is strong enough to reject adoption of these exact
arms, but it is not a population claim about prefix or windowed distillation.
