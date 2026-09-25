# Training and evaluation review

## Summary

- The normal training path is real and correctly ordered: forward, delayed-KD loss, finite-loss check, backward, finite-gradient check, clipping, and `AdamW.step()`. The current 800-step checkpoint is finite, and 7 focused alignment/causality/split tests pass.
- The held-out metric definitions are now unusually transparent for a small demo: hard teacher agreement is separated from ground-truth student/teacher accuracy, and micro, clip-macro, per-language, and clip-level results are all emitted. The current result is nevertheless a failed generalization sanity check: 14.30% teacher agreement and 3/21 correct student clips after 72.79 effective epochs.
- The switch number cannot diagnose the student. Evaluation hard-codes the known Hindi-to-English answer, ignores the available reverse-direction clip, and does not measure the local teacher's own very large response lag. On the current target cache, the teacher first changes to English 1.435 s after the true join and reaches the same chunk/EMA/dwell policy only after 2.615 s; the student misses the switch entirely.
- CPU RTF arithmetic is correct for the path actually timed, including log-mel extraction and the uncached chunked model. It is an offline-replay throughput number, not a measurement of online per-chunk latency.
- Evaluation lacks a run identity/config contract. A checkpoint, target cache, manifest, and `train_metrics.json` from different runs can be combined silently, and timing can be computed from current globals rather than the loaded model.

## Findings

### [BUG] scripts/train.py:46 — The success flags can claim a valid, NaN-free optimizer step when none survived

`--steps` and `--learning-rate` accept any integer/float. There is no post-`optimizer.step()` finiteness check, the checkpoint is saved at lines 135–147 before final validation, and `real_audio_optimizer_step` is hard-coded to `True` at line 178.

There are two direct counterexamples:

- With `--steps 0`, the loop at line 92 never runs, but line 178 still records `true`. `window` is zero, both empty means become `NaN`, and Python's JSON encoder writes the non-standard token `NaN`.
- In a probe using the same `AdamW`/clip ordering and `lr=float("inf")`, the loss was finite (`1.0`), the gradient was finite (`-2.0`), and the clipped norm was finite (`2.0`), but the parameter became `NaN` after `optimizer.step()`. A one-step invocation of this script would save that final state and compute `nan_free=True`, because lines 105–125 inspect only the pre-step loss/gradients/norm. The CLI accepts `inf` as a float.

The submitted default run is not affected: all 800 stored losses/norms and all tensors in the checkpoint inspected during this review were finite. The issue is that the recorded invariants do not actually prove what their names claim.

Suggested fix: reject `steps < 1`, non-positive/non-finite learning rates, batch sizes, and thread counts during argument parsing; after every optimizer step check model parameters (and, if claiming full optimizer health, optimizer state); only save after those checks; and derive `real_audio_optimizer_step` from an observed successful-step counter rather than a literal.

### [METHOD] scripts/train.py:46 — Model selection is many tiny-data epochs plus a noisy training-batch comparison

The only training-side learning check compares the first and last ten stochastic minibatch losses (lines 149–175). Those windows contain different shuffled examples and, because 71 is not divisible by batch size 7, can include a one-example final batch. There is no fixed validation loss, early stopping, or best-checkpoint selection; the final iterate is always saved.

The old concern of 800 × 7 / 21 = 266.7 epochs is stale for the current generator and default: there are now 71 training clips and the source default is 400 steps, approximately 2,584 / 71 = 36.39 effective epochs with the actual partial batches. The checked-in/current recorded run still used 800 steps and reports 5,168 examples, or **72.7887 effective epochs**. Its mean loss falls from **6.5341** to **4.9895**, but independent evaluation gives only **0.1430** frame agreement with the teacher, **0.1429** student clip accuracy versus **1.0** teacher clip accuracy, and no detected Hindi-to-English switch. Thus `loss_decreased=True` is evidence that optimization executed, not that the student learned transferable LID. The single switch clip is also only 1/71 of uniform clip sampling.

For this assignment, a few fixed plumbing steps are sufficient. Suggested fix: reserve a speaker-disjoint validation subset, evaluate the same fixed validation KD loss/checkpoint at intervals, select or stop without consulting the final test clips, and call the first/last stochastic loss comparison only an optimizer smoke test. Add more independently sourced switches or balance switch sampling before interpreting switch behavior.

### [BUG] scripts/eval.py:177 — Evaluation can silently mix incompatible model, timing, target, and training artifacts

The sole checkpoint compatibility check is language order (lines 177–180). Yet alignment and timestamps use current module globals (`LABEL_DELAY_FRAMES` and `MODEL_LOOKAHEAD_FRAMES` at lines 60, 64, 75, and 330), the summary uses current `TEACHER_NAME`/`ALGORITHMIC_LATENCY_MS` (lines 434 and 445), cached targets are selected only by clip ID (line 220), and training history is loaded from whichever `train_metrics.json` happens to share `--results-dir` (line 399).

A concrete silent case is a valid checkpoint whose stored `model_kwargs["lookahead_frames"]` is 2 while the current configuration says 4. The model is instantiated correctly from the checkpoint and its weights can load, but agreement drops the wrong number of tail frames, availability timestamps add the wrong lookahead, and the summary still reports the current 435 ms latency. Likewise, `train.py` stores `checkpoint["teacher"]`, but `eval.py` never compares it and always labels the result with the current teacher constant. There is no assertion that the loaded training metrics produced this checkpoint. The artifacts present at the end of this review happen to agree; that agreement is an unguarded convention rather than a checked invariant.

Suggested fix: assign a run ID and store the manifest/audio hashes, target metadata/hash, complete timing/model configuration, teacher ID, and checkpoint hash in both stage metrics and the checkpoint. Make evaluation reject mismatches, take lookahead from the loaded model, take label delay/latency from immutable checkpoint metadata, and read training metrics by the same run ID.

### [METHOD] scripts/eval.py:105 — The switch evaluator knows the answer and evaluates only one of two switch clips

The policy function is specifically `detect_hi_to_en_switch`: it can arm only on Hindi and can challenge only with English. `main()` then selects exactly `switch_hi_en_eval` with `next(...)` (lines 317–319), even though the current manifest contains both Hindi→English and English→Hindi evaluation clips and the summary reports `n_switch_eval_clips=2` at line 440. The output contains one scalar lag and no per-clip miss, premature-commit, or false-switch record.

This is adequate to draw one illustrative trace, but it is not evidence that the stated commit rule follows code-switches. Conditioning the evaluator on the true start and destination languages also prevents it from exposing an incorrect initial language, a switch to a third class, direction asymmetry, or flip-flopping.

Suggested fix: implement the same generic state machine described in `DESIGN.md`: initial winner over all classes, active state, any challenger versus the active class, EMA/dwell/cooldown, and explicit `UNKNOWN`. Run it on every annotated segment boundary in every switch record. Emit per-boundary commits plus aggregate miss rate, premature-switch rate, false switches, and lag only for correctly detected switches; keep this named clip solely as a plot example.

### [METHOD] scripts/eval.py:325 — End-to-end switch lag does not separate teacher lag from student/policy lag

Lines 325–341 load and plot teacher probabilities but calculate a detection time only for the student. Therefore `switch_lag_ms` folds together at least four effects: the teacher target's response, student approximation, chunk availability, and EMA/threshold/dwell. It cannot answer the review brief's key question of whether the student is slow or merely learned a slow teacher.

The current `switch_hi_en_eval` cache makes the confound concrete. Using target frame time `(i*160+400)/16000`, the teacher's English posterior first exceeds Hindi at **5.435 s**, already **1,435 ms** after the true 4.000 s join; it first satisfies English ≥ 0.60 and English−Hindi ≥ 0.10 at **5.475 s**. Delaying those teacher targets to their corresponding student emissions and applying this script's identical 160 ms chunk average, EMA weight 0.30, threshold/margin, and three-chunk dwell yields a teacher-policy commit at **6.615 s**, or **2,615 ms** lag. The current student never commits. Reporting only the student scalar would obscure both the teacher-dominated floor and the miss.

Suggested fix: report at least (1) ground-truth-boundary to raw teacher transition, (2) ground-truth-boundary to teacher under the identical availability/policy simulation, (3) ground-truth-boundary to student commit, and (4) student-minus-teacher excess delay, with misses kept separate from finite lag. Use the exact target/emission timing contract when constructing the teacher baseline; do not visually compare teacher target-frame time to student wall-clock time as though they were the same axis semantics.

### [DEFEND] scripts/eval.py:153 — The CPU RTF is replay throughput, not online chunk latency

The denominator is correct audio duration, `time.perf_counter()` brackets both frontend and model work, CPU needs no asynchronous-device synchronization, and the median of five current runs is correctly reported as **0.0059618**. One frontend/model warm-up is also performed outside the timed region.

However, each complete waveform is passed to `frontend(waveform)` before chunked model inference (lines 166–169). This gets vectorized full-utterance FFT behavior rather than exercising a streaming feature buffer, scheduling deadlines, or per-chunk tail latency. Conversely, `streaming_forward` deliberately recomputes left context, making it slower than a production state cache. Audio loading, EMA, routing, and scheduling are excluded. The number is therefore valid for this reference replay path but is neither a deployed streaming latency measurement nor a clear upper/lower bound on one.

Suggested fix: retain this metric but name it `offline_replay_compute_rtf`; additionally feed incremental waveform chunks through a stateful frontend/model implementation and report per-chunk p50/p95/max compute time and deadline misses on named hardware. Keep algorithmic availability (435 ms worst case here), policy delay, and compute time separate.

## Verified correct

- **Training step wiring:** lines 94–121 zero gradients, run the causal student, compute the delayed soft-target KL, reject a non-finite loss, backpropagate, reject non-finite gradients, clip, and step in the correct order. The optimizer receives the model parameters, and the current recorded run contains 800 losses and 800 gradient norms.
- **Alignment in held-out agreement:** `aligned_classes` compares teacher target `i` with student output `i+D` and retains exactly `length-D-L` targets, matching `alignment_mask`'s strict `i+D+L < length`. The focused test command `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider tests/test_alignment.py tests/test_causality.py tests/test_data_split.py` passed **7/7 in 40.62 s**.
- **Agreement is not mislabeled as accuracy:** the output separately names teacher agreement, student label accuracy, and teacher label accuracy. It supplies duration-weighted micro, equal-clip macro, per-language, and mode-based clip results. Recomputing the current 21 clips/8,090 valid frames reproduced micro agreement **0.1430160692** and macro agreement **0.1395579692** exactly; each language has three clips.
- **Chunk clock arithmetic:** a chunk is stamped when its last emitted frame's required lookahead feature ends: `(latest_feature*hop + window)/sample_rate`. This matches how `streaming_forward` slices full chunks. Excluding student frames before `D` is consistent with those outputs never receiving a teacher target during training.
- **EMA is applied once to decisions:** raw chunk averages enter `detect_hi_to_en_switch`, which performs one EMA internally. The separately smoothed array is used only for plotting, and both currently use weight 0.30. The three consecutive qualifying chunks reset correctly on failure.
- **RTF aggregation:** the current run list `[0.00618345, 0.00598465, 0.00596176, 0.00592388, 0.00589076]` has stored median **0.00596176**, and total samples divided by 16 kHz is the appropriate denominator for aggregate compute RTF.

## Interview questions

1. **What exactly is “held-out teacher agreement” here?** For each valid teacher frame `i`, it is equality of `argmax(q_i)` and `argmax(z_{i+D})`; tail frames lacking `L` real lookahead are excluded. Micro pools frames and therefore weights longer clips more; macro averages clip rates. It is fidelity to the teacher, not language-label accuracy, which is why the script reports both separately.
2. **Did expanding from 21 to 71 training clips resolve the memorisation concern?** It reduces the recorded 800-step exposure from about 267 to 72.79 effective epochs, and the new 400-step source default would be about 36.39. It does not resolve it: all audio remains tiny/synthetic, the model is selected on training loss, and the current held-out agreement is at roughly one-of-seven chance.
3. **Why can the switch lag not be called student lag?** The target teacher itself responds late because its local window contains 1.75 s of past audio. On the current clip its raw class transition is 1.435 s late and its identical-policy commit is 2.615 s late. Student lag must be reported alongside that baseline and as excess delay, not alone.
4. **Does three-chunk dwell mean the policy adds only 480 ms?** Not necessarily. After a perfectly abrupt Hindi-to-English posterior flip, EMA weight 0.30 produces English values 0.30, 0.51, 0.657, 0.760, 0.832. The threshold first qualifies on the third English chunk, and three qualifying chunks commit on the fifth; chunk scheduling/model availability add further wall-clock delay. `dwell_ms=480` describes the qualifying evidence window, not total response latency.
5. **How is RTF different from the 435 ms algorithmic latency?** RTF is CPU work divided by audio duration and can be far below 1 while a decision still waits for frame analysis, evidence delay/lookahead, chunk closure, EMA, and dwell. The present RTF also processes the frontend as a whole utterance, so it is replay throughput rather than per-chunk service latency.
6. **What would make the no-NaN claim airtight?** Require at least one step and finite hyperparameters, check the loss and gradients before the update, check parameters and optimizer state after the update, refuse to save on failure, and record the number of successfully completed updates. The present code covers only the pre-update half of that contract.
