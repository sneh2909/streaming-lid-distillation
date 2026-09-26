# Clean-teacher / degraded-student cross-view KD

## Question

Can the unchanged 42,567-parameter causal TCN recover at least half of its
paired narrowband-PCM, G.711 A-law, and G.711 mu-law macro-F1 loss by training
on randomly degraded student views while retaining clean ECAPA targets?

The predeclared gate required pooled recovery of at least 50% of the clean
control's channel loss, no more than two points of clean macro-F1 loss, no
additional switch miss, and no inference-latency or architecture change.

## Setup

This is one paired, one-seed experiment. Both arms used the main pipeline's 71
training clips, clean cached ECAPA soft targets, seed 7 initialization, exact
minibatch order, 1,600 optimizer steps, batch size 7, AdamW at `1e-3`, and the
same delayed KL and 42,567-parameter TCN. The clean control always received the
native waveform. For each example presentation, the treatment independently
sampled one of four views with equal probability:

- native 16 kHz PCM;
- 8 kHz 300--3,400 Hz narrowband PCM;
- the same narrowband path plus G.711 A-law; or
- the same narrowband path plus G.711 mu-law.

The realized 11,200-view schedule was 2,830 clean, 2,813 narrowband PCM, 2,782
PCMA, and 2,775 PCMU presentations. Each transformed waveform retained its
clean teacher target. The experiment imported the main frontend, dataset,
loss, model, optimizer-state checks, timing constants, and identity code. It
also imported the exact validated transforms from `telephony-robustness`;
fixed-vector A-law/mu-law checks passed, the causal FIR delay was 8.0 ms, and
no training or evaluation sample clipped before codec encoding.

Evaluation used the same 21 speaker-disjoint held-out clips and two mirrored
switch clips as the main pipeline. Monolingual scoring transformed each whole
clip before taking 1, 2, and 4 s leading prefixes, giving 63 requests per
condition. Switch scoring corrected the reference boundary for the 8 ms FIR
delay. Evaluation-input SHA-256 was
`fe1ed0b98a5dd2e77c186594cde9d23d2a1763a37dddb9d2ca23a73e6cc55952`.

Command:

```bash
UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python experiments/cross-view-kd/run.py --fresh
```

## Numbers

Aggregate known-label results over the 1/2/4 s prefixes were:

| Training arm | Evaluation view | Macro-F1 | Accuracy | Prediction flips vs that arm's clean view |
|---|---|---:|---:|---:|
| Clean control | Clean | 9.63% | 20.63% | 0/63 |
| Clean control | Narrowband PCM | 3.96% | 14.29% | 28/63 |
| Clean control | PCMA | 3.96% | 14.29% | 29/63 |
| Clean control | PCMU | 4.02% | 14.29% | 29/63 |
| Cross-view KD | Clean | 22.08% | 30.16% | 0/63 |
| Cross-view KD | Narrowband PCM | 14.44% | 25.40% | 11/63 |
| Cross-view KD | PCMA | 15.21% | 26.98% | 11/63 |
| Cross-view KD | PCMU | 15.21% | 26.98% | 11/63 |

The clean control's pooled degraded macro-F1 was 3.98%, a 5.66-point loss from
its 9.63% clean score. Cross-view KD raised pooled degraded macro-F1 to 14.95%,
a 10.97-point gain over the degraded control, or 194% of the predeclared
recoverable loss. It did not spend clean performance to do so: clean macro-F1
rose by 12.45 points to 22.08%. On clean prefixes, seven control errors became
correct while one correct prediction regressed; on each degraded view, 9--10
errors became correct and two correct predictions regressed.

This is a level improvement, not channel invariance. Relative to its own much
higher clean score, the treatment still lost 6.87--7.64 macro-F1 points under
the three channels, versus 5.62--5.68 for the weak control. It also remained
poor in absolute terms: English, Hindi, and Telugu recall were zero even on
clean prefixes, and only Marathi, Bengali, Tamil, and Gujarati obtained any
clean recall.

Both optimization traces were finite and decreased. First-10 to last-10 mean
KD loss was 6.8697 to 1.9158 for the control and 6.8557 to 2.3014 for
cross-view KD. The clean control's final state hash was exactly the submitted
main model hash, `89bf3f70...75f8`, so the baseline was reproduced bit for bit.

Neither arm established the source route or detected either held-out switch,
under clean, narrowband PCM, PCMA, or PCMU input; raw 500 ms-stable detection
was also 0/2 throughout. Therefore the “no additional miss” gate passes only
mechanically and supplies no positive switch evidence. Cross-view outside-
collar switch accuracy was 0% in all conditions.

The architecture and timing contract were unchanged: both arms had 42,567
parameters, a 127-frame receptive field, four lookahead frames, and the same
435 ms conservative bound. Median six-thread frontend-plus-streaming-replay
RTF was 0.00590 for the control and 0.00569 for cross-view KD; the small
difference is benchmark noise, not an inference-cost change.

Run identity:
`14a7048a5da76c90da714643a86f14984a53b360875e49a2e36c4a880b9d9127`.
Results SHA-256:
`35307a1846ac5c86db2e91d4baf4b949aae2c6ce78966a3f2f5397cf189a2d11`.
All 94 targets were validated at launch and publication; source, corpus,
target, reference bundle, batch order, view schedule, and both final model
states are bound in `results.json`.

## Verdict: adopt

Adopt the **cross-view input recipe** for the next controlled student training
run. It exceeded the 50% recovery gate, improved rather than regressed clean
macro-F1, reduced clean-to-channel prediction flips from 28--29 to 11, and
changes neither parameters nor algorithmic latency. The adoption target is the
training augmentation, not these ephemeral one-seed weights and not a claim
that the resulting model is telephone-ready.

The result does not unblock a delay sweep or validate switch behavior. The
student still misses every switch, absolute prefix accuracy remains poor, the
two switch fixtures are mirrored synthetic splices, and the cached main switch
targets retain the known linear-expansion future-information defect. Integrate
the augmentation in the next identity-bound retrain after (or alongside a
clearly separated fix for) METHOD-0, and keep real 8 kHz call evaluation as a
separate release gate.
