# Recorded run

These artifacts were produced on CPU from the commands in the repository README. Audio, cached teacher posteriors, and the student checkpoint are intentionally excluded.

- `summary.json` is the comparison-friendly result required by the assignment.
- `teacher_metrics.json`, `train_metrics.json`, and `eval_metrics.json` retain stage-level evidence. Training and evaluation share content-derived run ID `lidrun-1c3d0003…370d7`, which binds the model/checkpoint, all 94 audio files, manifest, target index, teacher, timing/model/source configuration, settings, and exact training evidence. Training records 1,600 requested/successful/post-update-checked steps with finite model and optimizer state; evaluation validates the complete identity and all 94 target files before scoring, and records that four provisional tail frames are withheld.
- `switch_plot.png` plots the offline teacher against the deployed chunk-EMA student trace.
- `tests.txt` records the final 27-test run, including causality/alignment boundaries, target-cache tamper/provenance rejection, invalid training settings, injected post-update corruption, and run-identity mismatch rejection.

This is a tiny synthetic-data sanity run with disjoint provider voice IDs between training and evaluation. The poor held-out student result (18.51% frame accuracy; missed switch) is retained deliberately: it exposes failed unseen-voice transfer that the earlier same-voice split hid. It is consistent with—but does not by itself prove—voice memorisation. Linear interpolation between 250 ms teacher anchors also gives most switch targets more future evidence than the aligned student, so the switch artifact is not a latency-valid result. Stable stateful replay has a six-thread median RTF of 0.0084. Do not interpret teacher agreement, label accuracy, or a single switch outcome as a population estimate.
