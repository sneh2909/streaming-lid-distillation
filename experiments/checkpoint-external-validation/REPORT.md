# Frozen-checkpoint external validation

## Question

Does a pinned external validation set confirm promotion of one of the four
earlier states in the frozen five-state checkpoint shortlist over the current
step-1600 student?

## Setup

- Evaluated only the shortlist frozen by `checkpoint-trajectory`: steps 970
  (best local label composite), 0 (best local raw switch result), 10 (lowest
  local development KD), 1600 (current final), and 800 (fixed midpoint). No
  state was trained, added, or removed after inspecting external results.
- Imported the main audio loader, log-mel frontend, causal TCN student,
  configuration, run-identity code, and ECAPA artifact loader. The main
  pipeline was not edited. The exact five stored model-state hashes were
  checked before scoring.
- Used only the `google/fleurs` **validation** split at revision
  `70bb2e84b976b7e960aa89f1c648e09c59f894dd` (CC BY 4.0). From the 2,598
  available validation rows for the seven supported languages, selected 100
  per language by the lowest SHA-256 rank of the pinned revision, config, row,
  ID, and basename. The 700-clip manifest and every materialized waveform were
  hash-checked. No FLEURS test row was read.
- Scored leading 0.5, 1, 2, and 4 s prefixes plus each full utterance. A clip
  shorter than a requested prefix was excluded from that prefix, never padded;
  this leaves 696 external clips at 4 s and all 700 at the other durations.
  Selection used only language-macro recall at 1 s, 2 s, and full length.
- Re-scored the same 21 speaker-disjoint synthetic held-out clips as the main
  pipeline. Their 1 s, 2 s, and full results matched the prior trajectory
  experiment exactly for every shortlisted state. Only 6/21 local clips are at
  least 4 s, so the local 4 s row is diagnostic and was not used for selection.
- Ran the pinned VoxLingua107 ECAPA teacher as an external control, reporting
  the conditional prediction among the same seven languages rather than
  claiming native 107-way accuracy. Full utterances were processed first and
  longest first with bounded batches to keep CPU memory bounded.
- The frozen external selector first required a state to be within 2 points of
  the best external composite, within 5 points of the best worst-language
  recall, and no worse than the final state's local committed false-switch
  count. Promotion then required an earlier state, at least +2 points of
  external composite, a positive lower bound from a paired 10,000-replicate
  language/prefix-stratified clip bootstrap, worst-language recall no more than
  5 points below final, and no extra local committed false switches.

Command:

```bash
UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
  .venv/bin/python -u \
  experiments/checkpoint-external-validation/run.py --fresh
```

The identity-bound ECAPA cache took 364.20 s to compute. The publication run
revalidated and reused that cache, used six CPU threads, and completed in
71.23 s; frontend extraction took 0.31 s and five-state student scoring took
5.65 s.

## Numbers

The five-state comparison was:

| Step | Frozen shortlist reason | External 1 s/2 s/full composite | Gain vs step 1600 | Worst-language composite | Local composite | Local committed false switches | Externally eligible |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 970 | Best local label composite | 18.00% | +0.33 pp | 0.00% | 30.16% | 14 | yes |
| 0 | Best local raw switch result | 14.29% | -3.38 pp | 0.00% | 14.29% | 0 | no |
| 10 | Lowest local development KD | 14.29% | -3.38 pp | 0.00% | 14.29% | 0 | no |
| 1600 | Current final | 17.67% | 0.00 pp | 0.00% | 25.40% | 14 | yes |
| 800 | Fixed midpoint | **18.05%** | **+0.38 pp** | 0.00% | 19.05% | 9 | yes |

The selector provisionally chose step 800 from the eligible states. All three
had zero frozen raw switch recall, the same 4,000 ms miss-penalized lag, and
zero no-collar switch accuracy; step 800 won the next tie-breaker with 67.5 raw
changes/min versus 105.0 at step 970 and 153.75 at step 1600. This is not
positive switch detection evidence: every shortlisted state still missed both
local switches.

The external prefix results below report language-macro recall and macro-F1.
The sample is balanced except for four utterances excluded at 4 s.

| Prefix | n | ECAPA recall | ECAPA macro-F1 | Step 800 recall | Step 800 macro-F1 | Step 1600 recall | Step 1600 macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | 700 | 20.14% | 15.72% | 14.29% | 3.58% | 14.43% | 3.88% |
| 1 s | 700 | 37.71% | 36.70% | 14.14% | 3.60% | 14.29% | 3.91% |
| 2 s | 700 | 75.29% | 75.10% | 17.14% | 9.60% | 16.00% | 8.71% |
| 4 s | 696 | 87.40% | 87.28% | 21.92% | 16.57% | 22.32% | 18.62% |
| Full | 700 | 97.71% | 97.71% | 22.86% | 16.55% | 22.71% | 18.79% |

ECAPA's selection composite was 70.24%, with a 57.00% worst-language
composite. Its per-language composites were 88.33% English, 70.67% Hindi,
57.00% Marathi, 67.00% Bengali, 73.00% Tamil, 76.67% Telugu, and 59.00%
Gujarati. Its much weaker 0.5 s and 1 s results also show that prefix duration
is intrinsically consequential, but the student remains near a single-class
baseline well after the teacher becomes strong.

Step 800's external gain over step 1600 was only 0.38 points. The paired
bootstrap 95% interval was **[-0.48, +1.24] points**, so it includes zero and
cannot reach the predeclared +2-point material-gain gate. Both snapshots had
0% worst-language composite. The local winner, step 970, had appeared 4.76
points better than final on the repeatedly consulted 21 clips, but was only
0.33 points better externally; that local advantage did not materially
transfer.

| Promotion gate for step 800 | Passed? |
|---|:---:|
| Earlier than step 1600 | yes |
| External composite gain at least 2 pp | **no** |
| Paired bootstrap lower bound above zero | **no** |
| Worst-language recall no more than 5 pp below final | yes |
| Local committed false switches no worse than final | yes |

## Verdict: reject

Reject promotion of every earlier snapshot and retain step 1600. The frozen
selector's step-800 choice improves the external composite by just 0.38 points,
with an interval spanning zero, while every candidate has zero worst-language
recall. The earlier step-970 result that motivated this confirmation run does
not reproduce at a useful magnitude on pinned external validation.

This rejects the checkpoint replacement, not the already adopted practice of
capturing checkpoints and selecting them on genuinely external development
data. There is no result worth adding as an adoption recommendation: the next
student iteration needs materially better supervision/training behavior, not a
different snapshot from this trajectory.

## Reproducibility and limits

- Run identity:
  `8ec024940fdd6f38731cc26f2d2f5a5ec962bdb9018d6f3a395f49b30a6097ea`
- Results SHA-256:
  `90d511775dcc6ff0ae49b7ef6e380eb4d6d573d57f0affc269f812beee1bb397`
- External manifest SHA-256:
  `ef0ec36427f742074b1bc88bd42f1b7e5b93b43bb390f9382f5030f3d8d2a637`
- External pool-metadata SHA-256:
  `2c825a340a0e9dbbfa9d10f0df658afb68e14a389d3148704d1b5950649cf675`
- Synthetic held-out input SHA-256:
  `3dff6504f54815cfc4175f1506e6893aa0a84b84a897c85d966d8198cb02a13f`
- Trajectory SHA-256:
  `f87f2aa46d7de7992a53c8d01135602a41485aae5bc203ae3a6f480a32468eac`
- ECAPA artifact SHA-256:
  `f193a0548951e8fbd6ca438a492b98b5c48bf7d98c86451ca3878dd4a7c0f706`
- End-of-run guards revalidated the imported source snapshot, reference
  artifacts, trajectory, all state hashes, external manifest and audio, exact
  local parity, and stable-streaming/full equivalence for each state.
- The bootstrap conditions on the externally selected pair and lacks speaker
  clusters because the published FLEURS schema has no speaker identifier. Its
  interval is descriptive rather than a population-level or
  selection-adjusted guarantee; this does not affect the rejection because it
  failed both statistical and effect-size gates.
- Evidence remains one deterministic training seed and a 100-clip-per-language
  subset of read-speech validation. It does not establish natural Hinglish,
  telephony, or deployment performance. Locked FLEURS test data and every
  natural code-switch test remained unread.
