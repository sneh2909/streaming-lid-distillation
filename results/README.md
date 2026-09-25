# Recorded run

These artifacts were produced on CPU from the commands in the repository README. Audio, cached teacher posteriors, and the student checkpoint are intentionally excluded.

- `summary.json` is the comparison-friendly result required by the assignment.
- `teacher_metrics.json`, `train_metrics.json`, and `eval_metrics.json` retain stage-level evidence; all three carry the same content-bound cache identity, including the pinned teacher artifact and target-generator source, and evaluation records that four provisional tail frames are withheld.
- `switch_plot.png` plots the offline teacher against the deployed chunk-EMA student trace.
- `tests.txt` records the final 16-test run, including causality/alignment boundaries and target-cache tamper/provenance rejection.

This is a tiny synthetic-data sanity run with disjoint provider voice IDs between training and evaluation. The poor held-out student result (18.51% frame accuracy; missed switch) is retained deliberately: it exposes failed unseen-voice transfer that the earlier same-voice split hid. It is consistent with—but does not by itself prove—voice memorisation. Linear interpolation between 250 ms teacher anchors also gives most switch targets more future evidence than the aligned student, so the switch artifact is not a latency-valid result. Stable stateful replay has a six-thread median RTF of 0.0082. Do not interpret teacher agreement, label accuracy, or a single switch outcome as a population estimate.
