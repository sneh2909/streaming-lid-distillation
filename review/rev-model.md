## Summary

- The core delayed-distillation math is correct: the implementation computes weighted `T^2 KL(q_T || p_T)`, aligns teacher frame `i` with student frame `i + 21`, and masks any prediction whose four feature-lookahead frames are unavailable.
- The TCN is bounded-lookahead and its finite receptive-field calculation is correct. All five assigned tests pass, including feature/waveform causality and complete-utterance chunk equivalence.
- `streaming_forward` is not safe as a growing-prefix emission API: it returns the last four logits even though they were computed with zero-padded, not observed, right context and will change when more audio arrives.
- An all-invalid alignment mask silently produces a zero loss and zero gradients, so a nominal optimizer step can succeed without training on any frame.
- The 435 ms number is a conservative loose bound rather than the exact worst case of the implemented chunk schedule, and the aligned student has substantially less past evidence than the switch teacher. Both qualifications should be explicit in a live defense.

## Findings

### [BUG] src/streaming_lid/model.py:121 — `streaming_forward` publishes provisional zero-padded tail logits

For the final chunk, `stop` is capped at `total_frames`. `_stack_right_context` then pads the missing future with zeros, but lines 125–126 still return all `n_emit` logits. On a live growing prefix, the last `lookahead_frames` outputs therefore do not have the right context promised by lines 109–110 and change when real audio replaces the padding. A deterministic check with a 24-frame prefix, a 50-frame continuation, and `lookahead_frames=4` found maximum absolute logit changes of `0.0406953`, `0.2299691`, `0.0863701`, and `0.2043671` for returned frames 20–23. Only frames 0–19 were fully contextualized (their maximum difference was numerical noise, `5.96e-08`). `test_chunked_and_whole_sequence_inference_match` cannot catch this because both paths see the complete utterance and deliberately apply the same zero padding at its end.

Make the online API buffer the last `lookahead_frames` features and emit only indices strictly below `received_frames - lookahead_frames`; track the next unemitted index/state so growing calls do not duplicate output. If end-of-stream padding is desired, expose it as an explicit flush and either train that case or mark those logits provisional. Add a test that feeds successively longer prefixes, concatenates only newly emitted logits, and compares them with the stable part of full inference.

### [BUG] src/streaming_lid/loss.py:52 — an empty valid mask silently becomes a successful zero-gradient step

The function rejects `total_frames <= delay_frames`, but a sequence can be longer than the delay and still have no frame satisfying `i + delay + lookahead < length`. Clamping the denominator to 1 then returns a differentiable zero. With 25 frames, `delay_frames=21`, and `lookahead_frames=4`, the function returned `loss=0.0`, `valid_frames=0.0`, and `weight_sum=0.0`; after `backward()`, all logit gradients were zero. This can make the required "real optimizer step" appear to succeed on a short-audio batch while performing no learning.

Raise a clear error when `weights.sum() == 0` (or have the training loop explicitly skip and count such a batch), and add a regression test at the exact `D + L` boundary. Retaining `clamp_min` for numerical protection after a non-empty check is harmless.

### [DEFEND] src/streaming_lid/model.py:129 — aligned targets have only about 1.075 s of past evidence versus the teacher's 1.75 s

The default TCN receptive field is correctly `1 + 2 * sum(1,2,4,8,16,32) = 127` context frames. However, a teacher target at `i` is matched to student output `j=i+21`. That output depends on stacked raw features from `j-126` through `j+4`, i.e. from `i-105` through `i+25`. Relative to the teacher anchor at the end of feature `i`, the uncentered frontend therefore supplies about 1.075 s of past audio and exactly 250 ms of future audio. The local switch teacher is configured for 1.75 s past and 250 ms future, leaving 675 ms of teacher history architecturally inaccessible to the student.

This is not a causality violation—the future-evidence bookkeeping is exact—and a smaller finite history may intentionally improve switch responsiveness. It is nevertheless an irreducible target mismatch that is currently unexplained and is an obvious interview question. Document the trade-off and ablate teacher-window history versus student receptive field; alternatively enlarge the receptive field (which adds state/compute but no lookahead latency) or shorten the teacher window.

### [DEFEND] src/streaming_lid/config.py:40 — 435 ms is a loose upper bound, not the exact implemented worst case

The formula adds a full 160 ms chunk to 25 ms analysis, 210 ms label delay, and 40 ms model lookahead. Under the repository's actual 16-frame output-chunk schedule, the extra wait after an aligned output ranges from 0 to 15 hops, not 16. Reproducing that schedule over 256 frames gave 250–400 ms from the teacher frame's end anchor to availability, or 275–425 ms from the frame's start. Thus 435 ms is safely conservative by 10 ms, but it is not an exact latency and `streaming_forward` itself does not enforce an online availability schedule.

Name this quantity as a conservative bound wherever it is serialized, state whether latency is measured from frame start or teacher anchor, and preferably report the implemented phase-dependent range (or the exact 425 ms worst case). A timing unit test should derive availability from `D`, `L`, frame geometry, and chunk phase so the claim cannot drift from inference behavior.

### [DEFEND] tests/test_alignment.py:17 — the tests do not establish the full claimed KD objective

The first test uses one-hot targets at `T=1` and checks only that one indexing choice is much better than another. It would still pass if `T^2` or the ramp were removed, and it does not distinguish the stated `KL(q || p)` from ordinary hard-label cross-entropy. The second test at `T=2` compares two calls whose valid targets are identical, so it also cannot verify temperature scaling, KL direction, ramp weights, or masking of a shorter padded item. A hand calculation on a soft two-class example did confirm the current code: implemented and manual weighted `T^2 KL(q || p)` differed by only `5.96e-08`; the problem is missing regression coverage, not the present formula.

Add a small analytic soft-target test that calculates `q * (log q - log_softmax(z/T))`, applies known ramp weights and `T^2`, and checks the scalar and gradients. Add a two-item padded batch with unequal `lengths`, plus the zero-valid-frame case above. These tests would support the derivation an interviewer is likely to ask for.

### [MINOR] src/streaming_lid/model.py:118 — `chunk_frames` is not validated

`chunk_frames=0` fails indirectly in `range`, while a negative value leaves `outputs` empty and fails later in `torch.cat`. Since this is a public inference argument, reject non-positive values with a direct `ValueError` and a focused test.

## Verified correct

- `CausalDepthwiseBlock` pads only on the left. Both `LayerNorm` instances normalize the channel dimension independently at each frame, so neither introduces temporal leakage.
- `_stack_right_context` gives output `t` raw features only through `t + L`. With the causal TCN, the default output depends on context indices `t-126..t`, and `receptive_field_frames=127` is correct for six kernel-3 dilated convolutions.
- The frontend causality cutoff in `test_waveform_future_cannot_leak_through_frontend` is correct: feature `t+L` consumes samples ending exclusively at `(t+L)*HOP_LENGTH + N_FFT`, and mutation starts at that exclusive boundary.
- The delay/lookahead identity is exact for the teacher's future boundary. Student output `i+21` can use feature `i+25`; that feature ends 250 ms after the end anchor of teacher frame `i`, matching `TEACHER_FUTURE_MS=250`.
- `alignment_mask` uses the correct strict bound: maximum required feature index `i+D+L` must be less than the unpadded feature length. A worked mask for lengths 4 and 3 with `D=L=1` was `[[1,1,0], [1,0,0]]`.
- `F.kl_div(log_softmax(student/T), teacher_soft_targets)` has the intended `KL(q_T || p_T)` direction. The teacher call site precomputes `q_T` using the same configured `T=2`, and multiplication by `T^2` occurs after the weighted frame mean.
- The ramp implements `min((i+1)/100, 1)` and the denominator is the sum of the same validity-times-ramp weights used in the numerator.
- Command `UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch .venv/bin/pytest -vv tests/test_causality.py tests/test_alignment.py` completed with all 5 tests passing in 26.76 s.

## Interview questions

1. **Why is the KL `KL(q_T || p_T)`, and where does `T^2` come from?** The teacher distribution is the fixed target, so minimizing its cross-entropy with the student's softened posterior gives forward KL up to constant teacher entropy. Dividing both logits by `T` shrinks logit gradients by roughly `1/T^2`; multiplying the loss by `T^2` keeps their useful scale comparable.
2. **Why is the label delay 21 frames rather than the teacher's full 25-frame future?** The student architecture already consumes four explicit lookahead frames. Matching teacher `i` to student `i+21` makes the latest consumed feature `i+21+4=i+25`, exactly the teacher's 250 ms future boundary.
3. **What audio can an aligned student decision actually use?** With the default receptive field it uses raw features `i-105..i+25`, approximately 1.075 s before and 250 ms after the teacher frame-end anchor. It therefore matches teacher future availability but not the teacher's full 1.75 s history.
4. **Does `streaming_forward` currently guarantee stable online emissions?** No. It proves chunk/full equivalence for a complete tensor, but returns a zero-padded final lookahead tail. A real online wrapper must withhold those logits until the corresponding future features arrive.
5. **What exactly does the 435 ms latency mean?** It is a conservative frame-start upper bound formed by adding a full chunk. For the implemented aligned chunk schedule, exact availability is 250–400 ms after the teacher frame-end anchor, or 275–425 ms after frame start, depending on chunk phase.
6. **Why down-weight the first second?** Monolingual targets repeat a full-utterance teacher posterior that a causal student cannot support at call start. The linear ramp reduces that impossible early supervision while retaining later frames; it mitigates rather than eliminates the asymmetry.
