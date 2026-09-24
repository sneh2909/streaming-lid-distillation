# Recorded run

These artifacts were produced on CPU from the commands in the repository README. Audio, cached teacher posteriors, and the student checkpoint are intentionally excluded.

- `summary.json` is the comparison-friendly result required by the assignment.
- `teacher_metrics.json`, `train_metrics.json`, and `eval_metrics.json` retain stage-level evidence.
- `switch_plot.png` plots the offline teacher against the deployed chunk-EMA student trace.
- `tests.txt` records the final causality/alignment test run.

This is a tiny synthetic-data sanity run. Do not interpret its agreement or lag as a population estimate.
