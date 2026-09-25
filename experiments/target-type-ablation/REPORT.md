# Target-type ablation

## Question

With the student architecture and 250 ms evidence budget fixed, should ECAPA
supervision be a repeated full-utterance posterior, a centred two-second
window, or a growing prefix ending at `t+250 ms`?

## Setup

The experiment reused the main manifest exactly: 71 training clips, the same 21
speaker-disjoint held-out monolingual clips, and both held-out switch clips
(`switch_hi_en_eval` and `switch_en_hi_eval`). The evaluation-input SHA-256 is
`7b78eb2981dacc8a01b1e1fad6b98da1fbceebdb0aba4a2597751f36ebe94a79`.

All arms imported the main frontend, manifest/dataset code, 42,567-parameter
TCN, delayed-KL loss, teacher label mapping, and chunk timing/smoothing helpers.
They used seed 7, identical initial weights and shuffled minibatch order, 1,600
AdamW updates, batch size 7, learning rate `1e-3`, temperature 2, 210 ms label
delay, 40 ms model lookahead, and six Torch threads. The only changed variable
was the target trajectory:

- `full_utterance`: classify the whole clip once and repeat that posterior.
  Future context is unbounded and a switch clip receives one constant target.
- `centred_2s`: classify `[t-1 s,t+1 s]` at 250 ms anchors. This diagnostic
  uses 1,000 ms of future audio, 750 ms beyond the tested evidence budget.
- `prefix_delta250`: classify the growing prefix `[0,t+250 ms]` at 250 ms
  anchors. This is the only arm whose teacher evidence respects the tested
  250 ms future-context budget.

Centred and prefix anchors were expanded to the 10 ms target grid with a
previous-anchor hold. The harness asserts that the selected anchor is never
later than its target frame. This is important: linear interpolation would
consult the next 250 ms anchor and silently add up to 240 ms of future audio.

Held-out accuracy is against the known synthesis language. Switch accuracy is
also shown outside a +/-250 ms boundary collar. Transition misses receive no
fabricated lag. Churn is the rate of unmatched chunk-top-1 changes: the one
intended change is discounted only when a persistent source-to-target event
actually matches the annotated boundary.

Command:

```bash
MPLCONFIGDIR=/tmp UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf \
  TORCH_HOME=./.cache/torch \
  .venv/bin/python experiments/target-type-ablation/run.py --fresh
```

The accepted arms share run identity
`24052c0e7baccd40c1289f70ca59fa71b3489d4146ae669fb76917aa7fe2d417`,
driver SHA-256
`d04982511e70d63f9df9f5651d3d6666d479aa5eaddabcf05c055d00dd5c1af0`,
and launch pipeline SHA-256
`2ea3a73a7107da56e769927afe1ca59167b0e2f35bde111e098824a35ee90a63`.
ECAPA was pinned to revision
`0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`; its four-file aggregate artifact
hash is
`f193a0548951e8fbd6ca438a492b98b5c48bf7d98c86451ca3878dd4a7c0f706`.

## Numbers

| Target | Context-valid | Teacher held-out label macro | Student own-target agreement | Student held-out label macro | Student clip accuracy | Teacher switch label macro | Student switch accuracy outside collar | Policy switches | EMA unmatched changes/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full utterance | no | 100.00% | 17.78% | 17.78% | 5/21 (23.81%) | 50.00% | 0.000% | 0/2 | 59.06 |
| Centred 2 s | no | 88.42% | 15.23% | 17.36% | 4/21 (19.05%) | 85.32% | 0.069% | 0/2 | 78.74 |
| Prefix through `t+250 ms` | yes | 84.23% | 13.96% | 18.25% | 4/21 (19.05%) | 56.21% | 0.692% | 0/2 | 118.11 |

Prefix was the numerical frame-macro winner, but only by **0.47 percentage
points** over full utterance. It lost one clip-level decision, detected neither
policy switch, detected neither raw nor EMA persistent student transition, and
had twice the EMA unmatched-change rate. Its near-zero switch score remains
near zero after removing the boundary collar, so this is not a lag-scoring
artifact.

Teacher transition timing further limits the target choices:

| Target | Hindi -> English teacher transition | English -> Hindi teacher transition |
|---|---|---|
| Full utterance | missed (constant target) | missed (constant target) |
| Centred 2 s | start +775 ms; three-anchor confirmation online at +2,275 ms | start +525 ms; confirmation online at +2,025 ms |
| Prefix through `t+250 ms` | missed | start +1,275 ms; confirmation online at +2,025 ms |

For centred targets, the “online” confirmation adds the full 1,000 ms future
window to the semantic confirmation time; that arm is therefore not deployable
at this latency. The cumulative prefix is context-valid but retains too much
pre-switch history: teacher frame accuracy was 38.55% on Hindi-to-English and
73.87% on the mirrored English-to-Hindi clip, and it never formed a persistent
Hindi-to-English target transition.

Optimization converged in every arm:

| Target | First-10 mean KD | Last-10 mean KD | Training wall time |
|---|---:|---:|---:|
| Full utterance | 6.8951 | 1.8425 | 51.23 s |
| Centred 2 s | 6.0874 | 1.5751 | 44.61 s |
| Prefix through `t+250 ms` | 5.9194 | 1.6284 | 49.09 s |

All losses, gradients, and post-update parameters were finite. Full-utterance
held-out targets matched the main target cache exactly (maximum raw and softened
posterior difference `0.0`). Exact local rechecks also confirmed unchanged
inputs, driver, and pinned teacher bytes.

One provenance qualification remains: `scripts/train.py` and `scripts/eval.py`
were edited concurrently after this process imported them, so
`pipeline_source_matches_launch_snapshot_at_end=false`. Every target,
checkpoint, and arm is bound to the one launch snapshot above, and a preceding
corrected run under the prior snapshot produced the same reported metrics.
These numbers are therefore internally controlled and reproducible by identity,
but they are not certified as a comparison to the moving repository head. The
remote Hub recheck also stalled after all arms were atomically saved; it was
interrupted and replaced by direct hashes of the already-resolved local pinned
artifact, recorded transparently in `results.json`.

## Verdict: reject

Reject these three **exact pure target regimes** as replacements for the mixed
main strategy at this latency and data scale.

- Full-utterance targets are structurally incapable of representing a boundary.
- Centred two-second targets violate the evidence budget and still produced no
  useful student switch behavior or held-out gain.
- The latency-valid cumulative prefix offered only a 0.47-point frame-macro
  change, regressed clip accuracy and stability, missed both student switches,
  and its teacher itself missed Hindi-to-English.

Retain the mixed regime only provisionally. The next target experiment should
compare availability-valid bounded or causal rolling windows after a
teacher-only boundary audit identifies a trajectory that reliably transitions
within budget; only then is a target-by-delay student grid worth training.

This is one seed on a tiny synthetic corpus with one training switch and two
mirrored evaluation switches built from reused component audio. The policy is
uncalibrated. The result supports rejecting these exact arms, not a population
claim that prefix or windowed distillation can never work.
