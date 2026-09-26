# Voice × text familiarity factorial

## Question

Does the frozen student's held-out collapse come primarily from an unseen
synthetic Edge voice, or is performance already too weak on exact familiar
audio to support that attribution?

## Setup

The experiment holds the submitted architecture, checkpoint, frontend,
streaming schedule, policy, and pinned ECAPA teacher fixed. For each of the
seven languages it takes three Edge-training texts (indices 5--7) and the same
three held-out texts used by the main pipeline (indices 10--12), then renders
all six texts through both the training Edge voice and held-out Edge voice.
This gives a balanced 2 × 2 `voice familiarity × text familiarity` factorial:

- 84 fresh clips: 21 in each cell;
- 42 original cached controls: 21 matching training clips and the exact 21
  main held-out clips;
- 0.5, 1, 2, 4 s and full-length scoring, or 630 teacher/student requests.

The primary endpoint was declared before scoring: full-clip student class from
the majority of delay-aligned stable frame predictions. Calling voice
familiarity the primary failure mechanism required all of:

1. at least a 10-point paired unseen-voice accuracy loss;
2. a positive seen-voice advantage in at least five of seven languages;
3. less than two points of nonnegative teacher loss and at least 90% teacher
   accuracy in both voice arms; and
4. at least 80% student accuracy on the exact seen-voice/seen-text training
   controls.

The fourth gate prevents a large voice contrast from being called the primary
cause when the model does not even establish a strong familiar-audio ceiling.
The frozen threshold/margin/EMA/dwell policy is included descriptively; its
chunks inherit the main pipeline's reconstructed output-aligned schedule and
are not an online-latency result.

The fresh Edge bytes are cached under
`data/experiments/voice-text-factorial/` and bound by recipe and audio hashes;
audio is ignored by Git. The first numerical attempt synthesized all 84 clips
but correctly refused to publish because the builder replaced the main
checkpoint/summary during scoring. The published rerun reused those exact 84
content-bound files against the now-stable main run
`lidrun-651b5a1f…f9c98`.

Command:

```bash
MPLCONFIGDIR=/tmp/mpl-voice-text-factorial \
UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python experiments/voice-text-factorial/run.py --fresh
```

## Numbers

### Primary full-clip comparison

| Metric | Seen voice | Unseen voice | Unseen-voice loss |
|---|---:|---:|---:|
| Student accuracy, all texts | 66.67% (28/42) | 16.67% (7/42) | 50.00 points |
| Student macro-F1, all texts | 59.22% | 9.88% | 49.33 points |
| Teacher accuracy, all texts | 90.48% (38/42) | 97.62% (41/42) | -7.14 points |
| Student accuracy, seen texts | 71.43% | 14.29% | 57.14 points |
| Student accuracy, unseen texts | 61.90% | 19.05% | 42.86 points |

The paired student outcomes were 26 seen-only correct, 5 unseen-only correct,
2 both correct, and 9 both wrong. Per-language unseen-voice losses were English
0.00, Hindi +100.00, Marathi -83.33, Bengali +100.00, Tamil +83.33, Telugu
+83.33, and Gujarati +66.67 points. Thus the aggregate 50-point loss and the
five-of-seven direction gate passed, but the effect was not universal: Marathi
strongly reversed and English was wrong under both voices.

### Prefix curve

| Prefix | Student seen | Student unseen | Gap | Teacher seen | Teacher unseen |
|---|---:|---:|---:|---:|---:|
| 0.5 s | 45.24% | 26.19% | 19.05 pp | 45.24% | 50.00% |
| 1 s | 54.76% | 23.81% | 30.95 pp | 57.14% | 69.05% |
| 2 s | 57.14% | 21.43% | 35.71 pp | 76.19% | 92.86% |
| 4 s | 64.29% | 19.05% | 45.24 pp | 88.10% | 97.62% |
| Full | 66.67% | 16.67% | 50.00 pp | 90.48% | 97.62% |

The full factorial cells further separate text from voice:

| Voice × text cell | Student accuracy | Student macro-F1 | Student frame accuracy | Teacher accuracy |
|---|---:|---:|---:|---:|
| Seen voice × seen text | 71.43% | 62.14% | 63.39% | 90.48% |
| Seen voice × unseen text | 61.90% | 56.12% | 51.79% | 90.48% |
| Unseen voice × seen text | 14.29% | 10.00% | 15.90% | 95.24% |
| Unseen voice × unseen text | 19.05% | 8.33% | 18.32% | 100.00% |

The voice contrast is much larger than the within-voice text contrast, so the
student is plainly voice-sensitive. However, the exact original
seen-voice/seen-text controls reached only 71.43% clip accuracy and 62.14%
macro-F1, below the predeclared 80% familiar-audio gate. English and Marathi
recall were both 0/3 even on those exact training waveforms. Fresh rerenders of
the matching seen/seen cell reproduced the same 71.43% accuracy with zero
prediction flips, ruling out provider drift as the explanation for that weak
ceiling. On the unseen/unseen matching cell, exact and fresh accuracy were both
19.05% with one prediction flip.

The exact 21 held-out controls reproduce the current main summary exactly:
student frame micro/macro accuracy 18.2265%/18.4668%, student clip accuracy
4/21, teacher clip accuracy 21/21, and identical student/teacher frame
agreement. This is the same held-out audio, not a recreated proxy.

The published pass scored 1,439.28 s of requested audio. ECAPA inference took
105.46 s (RTF 0.0733), and student frontend plus uncached streaming replay took
15.42 s (RTF 0.0107), on six Torch threads. The fresh input fingerprint is
`4744c7d7…e0d62dc`; the selected original-control fingerprint is
`78b29cc1…eb6880`; and the pinned teacher artifact is `f193a054…c0f706`.

## Verdict: reject

Reject **voice familiarity as the primary, sufficient explanation** and do not
make a voice-diversification-only retrain the next causal claim. Four of five
gates passed—including a real 50-point paired effect—but the exact familiar
training-audio ceiling was only 71.43%, with two languages at zero recall.
That redirects the main diagnosis toward optimization/target/class-collapse
alongside a material voice sensitivity, rather than toward voice identity
alone.

The useful guardrail is nuanced: crossed voices remain worth including in a
future corpus because five languages show a large effect, but METHOD-23 should
not be treated as the sole remedy or as proof that the current held-out failure
is memorisation. First inspect exact-training class collapse and checkpoint
selection; if a stronger baseline establishes a high familiar-audio ceiling,
rerun the same factorial before a crossed-voice retrain.

Run identity:
`011c73142adefb2f9c861937e2c6349a7e4fdcebdd8d1033c1e494f84e41a99a`.
Results SHA-256:
`f12b5045dfc3816affcf837f9948cdab8321d953b68d6b1d301f8cfb89daaccb`.

## Validation and limits

- All 630 restricted teacher and student posteriors are finite and normalized.
- Driver and imported-source snapshots, checkpoint, summary, manifest, 42
  original controls, and 84 fresh audio files were unchanged at publication.
- The main held-out metrics match `results/summary.json` to numerical equality.
- `py_compile`, JSON parsing, numerical/count/hash assertions, and scoped
  `git diff --check` pass.

This is one frozen checkpoint, one synthesis pass, six scripts per language,
and synthetic voices whose remote service revision is not exposed. It shows
renderer/voice-profile sensitivity, not a human-speaker causal effect. The
80% familiar-audio gate is a diagnostic threshold rather than a population
estimate, and no confidence interval from 42 paired texts would make these
synthetic clips representative of real calls.
