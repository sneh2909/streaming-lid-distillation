# Loss-factorial experiment

## Question

Can clip normalization or trusted monolingual labels repair the current
student's class collapse without changing its architecture, causal target
timing, data order, or update budget?

## Setup

Four 42,567-parameter causal TCNs started from one identical initialization
and consumed the same shuffled minibatches for 1,600 AdamW updates:

| Arm | Objective |
|---|---|
| `frame_kd` | Current frame-normalized `T=2` KD control |
| `equal_clip_kd` | Normalize KD inside each clip, then average clips |
| `hybrid_kd_ce` | Equal-clip `0.5 * T^2 KL + 0.5 * CE` on labelled monolingual clips |
| `teacher_error_gate` | Equal-clip KD, replaced by hard CE only where the monolingual teacher is wrong |

The mixed Hindi-to-English training clip retained its schema-5,
previous-anchor-hold local KD target in every arm. The run used the main 71
training clips, the same 21 voice-disjoint held-out monolingual clips at true
1 s, true 2 s, and natural full duration, and both main switch clips. No
external validation or locked test row was read. Model states were captured in
memory every ten updates and all 161 states per arm were scored.

The local sufficiency gate was fixed before training: a checkpoint had to have
nonzero recall for all seven languages across the 70 monolingual training
clips, at least 80% all-training accuracy, and at least 80% accuracy on the 21
exact Edge controls at indices 5/6/7. Passing this gate advances only a loss
recipe to external validation; it does not promote an experiment checkpoint.

Command:

```bash
UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -u experiments/loss-factorial/run.py --fresh
```

## Target audit

Balanced clip counts do not produce balanced KD supervision. The bound ECAPA
cache is selected-seven top-1 correct on 6/10 English clips and 10/10 for each
other class. The four errors are `en_train_06` through `en_train_09`, predicted
as Hindi, Hindi, Hindi, and Gujarati. English receives 6.529% of the current
loss-weighted `T=2` target mass versus 20.649% for Hindi.

The four deterministic effective English priors matched the research
proposal: 6.529% frame KD, 6.615% equal-clip KD, 10.486% hybrid KD+CE, and
11.748% teacher-error-gated. Direct arithmetic over the bound ten English
targets gives mean known-label probability 51.927% at `T=1` and 45.026% at
`T=2`. Those values correct the research memo's inconsistent 55.6085% and
48.0021% aggregates; the memo's profile rows and the four bad-clip values
otherwise match this artifact.

## Numbers

The production-loss control exactly reproduced the main model-state hash,
all 1,600 losses, and all 1,600 gradient norms. All arms completed 1,600 finite
post-update checks, retained the 127-frame receptive field, four-frame
lookahead, and 435 ms conservative latency bound, and passed stable
streaming/full-prefix equivalence.

Final-step behavior was unstable and is not sufficient for selection:

| Arm at step 1,600 | All-train | English train recall | Exact Edge | 1 s/2 s/full dev composite | Held-out false commits | Raw/policy switches |
|---|---:|---:|---:|---:|---:|---:|
| Frame KD | 55/70 (78.57%) | 0/10 | 15/21 (71.43%) | 25.40% | 14 | 0/2, 0/2 |
| Equal-clip KD | 56/70 (80.00%) | 1/10 | 15/21 (71.43%) | 19.05% | 13 | 0/2, 0/2 |
| Hybrid KD+CE | 59/70 (84.29%) | 0/10 | 18/21 (85.71%) | 23.81% | 11 | 0/2, 0/2 |
| Teacher-error gate | 55/70 (78.57%) | 0/10 | 15/21 (71.43%) | 22.22% | 10 | 0/2, 0/2 |

Only `hybrid_kd_ce` produced eligible checkpoints, at steps 1,520 and 1,580.
The frozen tie-break selected step 1,520 because its development composite was
higher:

- All-training accuracy was 61/70 (87.14%), with recalls
  `en/hi/mr/bn/ta/te/gu = 4/9/8/10/10/10/10` out of ten.
- Exact Edge accuracy was 17/21 (80.95%). This aggregate pass is not an
  English-profile repair: Edge-English remained 0/3; Marathi rose to 2/3 and
  the other five classes were 3/3.
- Held-out 1 s/2 s/full clip accuracy was 7/21, 6/21, and 6/21, for a 30.16%
  composite. The same-step frame-KD control was 6/21, 3/21, and 3/21
  (19.05%); the submitted final control composite is 25.40%.
- Held-out English and Hindi clip recall remained zero at every selected
  deadline. Full-frame language-macro accuracy was only 18.83%.
- The selected hybrid had 13 committed false switches in 78.97 s of
  monolingual audio, versus eight for the same-step control and 14 for the
  submitted final control.
- Neither switch direction established a correct raw or committed source
  state; raw and policy detection were both 0/2, so no lag is reportable.

Equal-clip KD and teacher-error-gated KD never passed all three local gates.
The former's one final English training hit shows a small duration-weighting
effect. The targeted error gate's best diagnostic checkpoint reached only
2/10 English training recall and 76.19% exact-Edge accuracy. Broad hard-label
regularization therefore helped local fit more than correcting only the four
teacher errors; the four errors are not the sole collapse mechanism.

## Verdict: adopt, narrowly

Adopt the 50:50 monolingual KD+CE objective as the sole loss candidate for one
frozen external/natural validation run, with local KD unchanged on switch
clips and periodic checkpointing retained. Do **not** adopt the step-1,520
weights or change the main checkpoint yet.

Why: it was the only arm to pass the predeclared local class-coverage and
familiar-audio gates, at two checkpoints, and its selected checkpoint improved
the same-data 1 s/2 s/full composite by 11.11 points over the matched control.
The adoption is deliberately procedural because the focal Edge-English slice
is still 0/3, held-out minimum-language recall is zero, stability worsened
against the same-step control, and both switches remain unevaluable. A main
integration would also require documentation to stop claiming that known
labels never enter optimization.

Run identity:
`044ef348f33663f014b6d5b868ea23e969758438f70e5cfd500439d4e7cdbdf5`.
Results SHA-256:
`b22125b174c76d073c7569f44cb9ebd0ef2562fba9f98fa42d767a1366ab91ea`.
The run took 266.88 s on six Torch threads. JSON parsing, numerical/identity
assertions, `py_compile`, scoped whitespace checks, and the full 57-test suite
passed.

## Limitations

- This is one seed on a tiny synthetic corpus and repeatedly consulted
  synthetic development data.
- Aggregate exact-Edge accuracy hides 0/3 English recall; it is a smoke gate,
  not evidence of profile transfer.
- The selected checkpoint was chosen locally and has not been scored on the
  pinned external validation manifest.
- The two reversed synthetic switches are neither independent nor natural,
  and every trained arm failed their source precondition.
- Hard CE is available only where trusted monolingual labels exist; unlabelled
  call audio still requires valid teacher targets or another supervision path.
