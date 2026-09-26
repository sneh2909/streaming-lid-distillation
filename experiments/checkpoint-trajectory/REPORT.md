# Checkpoint-trajectory selection

## Question

Does an earlier state in the exact 1,600-step baseline TCN trajectory improve
fixed synthetic-development label or switch performance over the unselected
final iterate, without increasing committed false switches on monolingual
audio?

## Setup

- Replayed the main training recipe exactly: the same 71 training clips,
  schema-4 ECAPA targets, seed 7, batch size 7, AdamW at `1e-3`, and 1,600
  optimizer updates. The experiment imports the main dataset, frontend, model,
  loss, training-safety, timing, and run-identity code; no main-pipeline source
  was edited.
- Captured the initial state and every 10th post-update state through step
  1,600 (161 snapshots). The final replay exactly reproduced the submitted
  loss trace, gradient-norm trace, and model-state SHA-256
  `b96a856c4d77f0a70be10421fa362d838be4b7c047596fb1049b977629b42273`.
- Scored every snapshot on the same 21 speaker-disjoint held-out clips used by
  the main pipeline, at leading 0.5, 1, 2, and 4 s and full length. These clips
  are repeatedly consulted synthetic development data, not a locked test set.
- Also scored the same two held-out Hindi/English switch clips, all 70
  monolingual training clips, and the 21 exact Edge-training controls at text
  indices 5/6/7. No external validation or locked-test data were read.
- The label-selection score was the mean held-out accuracy at 1 s, 2 s, and
  full length. The frozen router used threshold 0.60, margin 0.10, EMA new
  weight 0.30, and three-call dwell. Raw switch detection required a 500 ms
  stable run. Streaming scores use the actual scheduler emission groups and
  exclude frames before the 21-frame label delay.
- The frozen shortlist contained the best label composite, best raw switch
  result, lowest development KD, final step, and fixed midpoint, with duplicate
  steps removed. Eligibility required being within 2 points of the best label
  composite, within 5 points of the best worst-language recall, and no more
  committed monolingual false switches than the final state. The local selector
  then used switch recall, miss-penalized lag, no-collar switch error, switch
  changes/min, development KD, and earlier step, in that order.
- The predeclared usefulness gate required an earlier state, either at least a
  2-point label-composite gain or a 10-point switch-recall@1 s gain, and no
  increase in committed monolingual false switches.

Command:

```bash
UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -u experiments/checkpoint-trajectory/run.py --fresh
```

The run used six CPU threads and completed in 144.77 s.

## Numbers

The frozen selector chose step 970. Its model-state SHA-256 is
`ed50eb4e38c055666e176a2976b4134b1c3cb900f59a926fc3e3e64e7453830b`.

| Metric | Step 970 | Step 1600 | Step 970 minus final |
|---|---:|---:|---:|
| Label composite: mean accuracy at 1 s / 2 s / full | 30.16% | 25.40% | +4.76 pp |
| 0.5 s accuracy | 28.57% (6/21) | 28.57% (6/21) | 0.00 pp |
| 1 s accuracy | 28.57% (6/21) | 28.57% (6/21) | 0.00 pp |
| 2 s accuracy | 28.57% (6/21) | 28.57% (6/21) | 0.00 pp |
| 4 s accuracy | 28.57% (6/21) | 19.05% (4/21) | +9.52 pp |
| Full accuracy | 33.33% (7/21) | 19.05% (4/21) | +14.29 pp |
| Worst-language recall composite | 0.00% | 0.00% | 0.00 pp |
| Full-frame known-label accuracy, language macro | 18.13% | 18.47% | -0.34 pp |
| Development distillation loss, language macro | 12.3749 | 15.3727 | -2.9978 |
| Raw switch recall by 1 s | 0/2 | 0/2 | 0 |
| Switch no-collar accuracy | 0.00% | 0.00% | 0.00 pp |
| Switch raw changes/min | 105.00 | 153.75 | -48.75 |
| Monolingual committed false switches | 14 | 14 | 0 |
| Monolingual committed false switches/hour | 638.22 | 638.22 | 0.00 |
| Monolingual raw top-1 changes | 112 | 96 | +16 |
| All 70 monolingual training clips, full accuracy | 64.29% | 78.57% | -14.29 pp |
| Exact Edge controls, full accuracy | 38.10% | 71.43% | -33.33 pp |

The label maximum was 30.16% at steps 970, 1440, and 1490; the frozen
tie-breaking/shortlist procedure retained the earliest, step 970. The five
shortlist entries were steps 970 (best label), 0 (best raw switch, tied at
zero), 10 (lowest development KD), 1600 (final), and 800 (midpoint). Only step
970 passed the local eligibility filters.

The 4.76-point composite gain clears the 2-point usefulness gate, and the 14
committed false switches equal the final state's count. The gain is entirely
late-utterance: 1 s and 2 s accuracy did not move, while full accuracy changed
by three of 21 clips. Both snapshots had zero source-language preconditions
across the two switch clips and detected neither switch, so there is no
positive switch-latency evidence. English, Hindi, Tamil, and Telugu recall were
zero in step 970's 1 s/2 s/full composite; the worst-language metric therefore
remained zero.

Training itself was finite for all 1,600 checked updates. Mean loss moved from
6.8697 over the first ten steps to 1.7415 over the last ten. The early selected
state's substantially weaker familiar-audio accuracy shows that its local
held-out gain is not a general claim of a better-fit model.

## Verdict: adopt

Adopt **trajectory checkpointing and fixed validation-based checkpoint
selection**, and advance step 970 only as a provisional Tier-A candidate. The
experiment establishes that the final optimizer iterate is not automatically
the best point on the existing synthetic-development label criterion: the
earlier state clears the predeclared material-gain gate without adding
committed false switches.

Do **not** replace the main checkpoint from this result. Step 970 still misses
both switches, has zero worst-language recall, produces 14 committed false
switches in only 78.97 s of monolingual audio, and loses 14.29 points on all
training clips and 33.33 points on exact Edge controls. Rank the frozen
at-most-five shortlist on a pinned external validation set before any
checkpoint replacement; retain the current final checkpoint if that gate does
not confirm the local result. No additional tuning should use these 21
synthetic-development clips.

## Reproducibility and limits

- Run identity:
  `a1397e936c7df6dea5d7753c22619cd3822ffe484397ad70ae532621ff9a827b`
- Results SHA-256:
  `c4533c0b4aecd667289dff1df38f0a7b79cd98df359450da817b47706c4a4379`
- Trajectory SHA-256:
  `f87f2aa46d7de7992a53c8d01135602a41485aae5bc203ae3a6f480a32468eac`
- The ignored weight artifact is
  `data/experiments/checkpoint-trajectory/trajectory.pt`; it contains all 161
  model states and no optimizer state.
- Independent validation recomputed the canonical run identity, trajectory
  hash, current source/reference hashes, and every stored model-state hash;
  checked all JSON numerics as finite; and passed `py_compile` and scoped
  whitespace checks.
- Evidence is limited to one deterministic seed, 71 synthetic training clips,
  21 repeatedly consulted synthetic-development clips, and two mirrored
  switch splices. It neither resolves the slow teacher trajectory nor provides
  external generalization, natural-call, or release-gate evidence.
