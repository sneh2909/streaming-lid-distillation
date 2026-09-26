# Hybrid-loss external validation

## Question

Does the frozen step-1,520 equal-clip `0.5 * T^2 KL + 0.5 * CE`
candidate from `loss-factorial` improve pinned external validation enough to
replace pure KD, without reopening its local checkpoint selection?

## Setup

- Regenerated only the already-frozen `hybrid_kd_ce` step-1,520 candidate and
  its identity-matched `frame_kd` step-1,520 control. Both used the exact
  initialization, minibatch order, 71 training clips, target timing, causal
  TCN, and six-thread CPU settings from the parent experiment.
- Loaded main-pipeline code from the immutable source snapshot
  `9bf2ae32...1202d`, not the concurrently changing live pipeline. Both
  regenerated states, all 1,520 losses, and all 1,520 pre-clip gradient norms
  exactly matched the parent result. The hybrid state hash is
  `9e2dd55e...97c36a`; the same-step control is `a16278cf...ea8b9`.
- The shared target cache had meanwhile migrated from schema 5 to schema 6 to
  add native-teacher audit fields. A read-only adapter validated the schema-6
  manifest/audio/file hashes, probability arrays, dense anchor expansion, and
  switch availability ledger. Exact state/loss/gradient reproduction then
  proved that the tensors consumed by this training run were unchanged by the
  migration. The builder-owned cache was not modified or downgraded.
- Re-evaluated the same 21 voice-disjoint synthetic held-out clips and both
  mirrored switch clips. Their complete step-1,520 rows exactly reproduced
  `loss-factorial` before any external result was inspected.
- Evaluated the fixed 700-clip `google/fleurs` **validation** cohort: 100
  SHA-ranked clips for each supported language at revision
  `70bb2e84...94dd`. The experiment independently re-derived labels and
  selection from all seven pinned `dev.tsv` files, matched every selected WAV
  byte-for-byte to the seven pinned validation archives, and rehashed the
  materialized audio after scoring. Release-lock SHA-256 is
  `a164dcde...7085a`; external manifest SHA-256 is `ef0ec364...2a637`.
  FLEURS test was not read.
- Scored true leading 0.5/1/2/4 s prefixes and natural full duration. Short
  clips were excluded rather than zero-padded; 696/700 qualify at 4 s and all
  700 qualify at 0.5/1/2 s and full duration. The frozen selection composite
  is language-macro recall averaged over 1 s, 2 s, and full.
- Compared the hybrid with both its same-step control and the current
  step-1,600 main checkpoint. Paired uncertainty resampled each audio once per
  language and reused the same weight across all three prefix views, so
  repeated views were not treated as independent observations.

Command:

```bash
MPLCONFIGDIR=/tmp/mpl-hybrid-external \
  UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf \
  TORCH_HOME=./.cache/torch HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -u \
  experiments/hybrid-external-validation/run.py --fresh
```

## Numbers

The local gain did not transfer. On the same synthetic development clips,
hybrid step 1,520 again scored 30.16% versus 19.05% for its matched control.
On FLEURS validation it instead scored **15.29%**, below both the matched
control's **18.00%** and the current final checkpoint's **17.67%**.

| Model | 0.5 s recall / F1 | 1 s recall / F1 | 2 s recall / F1 | 4 s recall / F1 | Full recall / F1 | 1/2/full composite |
|---|---:|---:|---:|---:|---:|---:|
| Frame KD, step 1,520 | 14.71 / 4.46% | 14.43 / 5.05% | 18.29 / 14.34% | 21.42 / 19.54% | 21.29 / 19.31% | **18.00%** |
| Hybrid KD+CE, step 1,520 | 14.43 / 3.90% | 14.00 / 3.84% | 15.57 / 9.03% | 15.25 / 11.35% | 16.29 / 12.81% | **15.29%** |
| Frame KD, step 1,600 | 14.43 / 3.88% | 14.29 / 3.91% | 16.00 / 8.71% | 22.32 / 18.62% | 22.71 / 18.79% | **17.67%** |

The hybrid's composite difference was **-2.71 points** versus its matched
control and **-2.38 points** versus the current final checkpoint. The paired
audio-cluster descriptive 95% intervals were entirely negative:

- versus matched step 1,520: **[-4.14, -1.29] points**;
- versus current step 1,600: **[-3.71, -1.05] points**.

Those intervals are descriptive because this validation cohort was already
used by the earlier checkpoint experiment; they are not locked-test or
population-level guarantees. The negative point estimates independently fail
the predeclared `>=2`-point material-gain gates.

The external per-language 1/2/full composite exposes a different collapse,
not a balanced recovery:

| Language | Matched control | Hybrid | Current final |
|---|---:|---:|---:|
| English | 0.33% | 4.00% | 0.00% |
| Hindi | 1.00% | 1.67% | 2.67% |
| Marathi | 15.00% | **0.00%** | 10.00% |
| Bengali | 59.00% | 68.33% | 88.00% |
| Tamil | 20.00% | 12.67% | 15.00% |
| Telugu | 28.00% | 19.67% | 5.33% |
| Gujarati | 2.67% | 0.67% | 2.67% |

Thus the hybrid also failed the all-seven-nonzero gate. Its small English and
Hindi changes did not offset zero Marathi recall and losses in four other
classes.

On the same local clips, hybrid step 1,520 retained its 61/70 all-training and
17/21 exact-Edge fits, but produced 13 committed monolingual false switches
versus eight for its same-step control. All three relevant states still detect
0/2 raw and 0/2 committed switches, with no correct source precondition, so no
switch lag is reportable.

The complete run took 310.59 s. Exact paired training took 65.63 s, external
feature extraction 12.80 s, and three-state scoring 3.25 s; the remaining time
was chiefly independent hashing and archive-to-audio validation. Current-final
external metrics and all stored predictions exactly reproduced the prior
external-validation artifact.

## Verdict: reject

Reject this exact 50:50 hybrid loss recipe and its step-1,520 weights as a
main-pipeline change. It passed the local fit gates that advanced it, but
regressed external validation against both controls, its paired intervals were
entirely below zero, and it still had a zero-recall language and missed both
switches. The result demonstrates local over-selection rather than a
generalizing repair for teacher bias or class collapse.

This does not establish that every use of labelled supervision is harmful.
It rejects this one fixed mixture, seed, tiny synthetic training corpus, and
checkpoint rule. Any later supervised objective needs a newly predeclared
recipe and development protocol; it must not tune another mixture on these
already-consulted synthetic or FLEURS validation rows.

Run identity:
`ea051d051708a1f461f689f5e9c6a4da86ba8f0b20e209aad67edefd06ac1a0a`.
Results SHA-256:
`fd540823a30fad8d1147a9a58a674b61e8f3db7fca67cb910d1bf7b35ee1eb7c`.

## Limitations

- One deterministic seed and 71 synthetic training clips remain too small for
  a general statement about supervised LID objectives.
- FLEURS validation is monolingual read speech, not natural Hinglish or
  telephone audio, and has no published speaker ID for speaker-clustered
  uncertainty.
- The validation set is reused project evidence. Locked FLEURS test and all
  natural switch test sets remained unread.
- The two synthetic switches reverse one source pair and cannot establish
  population switch behavior.
- Known-label CE changes the pure-distillation claim and cannot supervise
  unlabelled call audio without another labelling path.
