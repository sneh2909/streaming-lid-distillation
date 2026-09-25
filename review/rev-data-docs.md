# Data generation, results, and documentation review

## Summary

- The source-level split is speaker-ID and utterance disjoint: the current generator makes 70 monolingual training clips from 14 synthetic voice IDs, 21 held-out clips from seven unseen voice IDs, and uses train/held-out components for the corresponding switch clips. This is a sound plumbing split, though it remains a very small synthetic-voice experiment.
- README and DESIGN now match the final rebuilt artifacts: 71 training clips, 1,600 steps, 18.51% held-out agreement, 4/21 correct clips, and a missed switch represented by null lag. The JSON summaries are also internally consistent.
- The switch set is not two independent examples: the same Hindi and English held-out utterances are reused in reverse order, and fixed four-second padding puts 158 ms or 903 ms of silence immediately before the nominal boundary. The reverse clip therefore tests silence-to-Hindi more than English-to-Hindi.
- `DESIGN.md` directly covers all five Part 2 bullets with a sensible calibrated state machine, ASR overlap re-decode, hysteresis and lag metrics, priors/fallback, and a versioned teacher/human feedback loop. Its fixed 2.5 s ring buffer nevertheless cannot guarantee the promised replay from VAD onset when a decision is delayed indefinitely.
- The poor unseen-voice result is honest evidence of failed generalization, but it does not by itself prove the docs' stronger causal conclusion that voice memorization is the principal cause; no seen-voice evaluation or training accuracy is reported. All seven focused tests pass.

## Findings

### [BUG] scripts/prepare_data.py:273-293,360-364 and README.md:24,122 — Cached WAVs are not bound to their text and regeneration is non-transactional

Adding the voice slug to the filename prevents one class of stale-cache error, but the key is still only `clip_id + voice`. If `SENTENCES["en"][0]` changes while its voice does not, a normal rerun sees the old WAV at line 276, skips synthesis, then writes the **new** text into the manifest at line 293. The waveform and declared utterance silently disagree. Provider model/version changes are also unrecorded.

`--force` writes each live WAV in place, while the manifest is replaced only after all 91 monolingual requests and three switches finish. A final retry failure can therefore leave a mixture of old/new audio under the old manifest; a subsequent non-force run accepts that state. This can invalidate the claimed speaker/text audit and any cached targets without an error.

Use a content-addressed key over language, text, provider, exact voice, synthesis options, and generator version (or validate the same fields in a sidecar). Build a complete corpus in a temporary directory, validate non-silence/rate/count/split/checksums, and atomically publish it with its manifest. Bind target caches to those audio hashes.

### [METHOD] scripts/prepare_data.py:198-230,305-322 — The two switch clips are one padded utterance pair, not two independent boundaries

Both evaluation rows use exactly `hi_heldout_10` and `en_heldout_10`; the second merely reverses them. Those same waveforms also appear in the monolingual held-out metrics. Thus there is no train/evaluation leakage, but there are not two independent switch trials either.

More importantly, `four_second_segment` pads a short source at its **end** and metadata always declares the language active through 4.0 s. In the newly generated corpus, Hindi is 3.8423125 s and English is 3.09675 s. The nominal Hindi→English boundary follows 157.7 ms of zero padding, while the English→Hindi boundary follows **903.3 ms of zero padding**. For longer inputs the function instead truncates at 4.0 s, potentially mid-word. Lag relative to that synthetic 4.0 s timestamp does not cleanly measure a spoken-language transition.

Concatenate VAD-trimmed complete utterances and set the annotation to the actual first-segment sample count. Use distinct utterance/speaker pairs for each direction and enough boundaries to report misses and lag distributions; keep calibration pairs separate from final evaluation. If silence is intentionally allowed, annotate speech end and next-speech onset separately and define which boundary the metric uses.

### [METHOD] results/teacher_metrics.json:15 — “Mean anchor label agreement” still hides the switch regime

This artifact reports `0.9477756`, which is the equal-clip mean of 94 clip accuracies even though 91 clips have one anchor and each switch has 33. Recomputing from `data/generated/targets/metadata.json` gives 190 anchors and a pooled result of **156/190 = 0.8210526**; switch-clip accuracies are only **22/33, 23/33, and 24/33**. The displayed scalar sharply down-weights the only temporal targets.

This is the same aggregation defect already detailed in `review/rev-teacher.md`, now verified against the final artifact assigned here. Rename the existing number `macro_clip_anchor_accuracy`, and add pooled integer counts plus separate stationary/switch and per-clip metrics. Do not use the 94.78% number as evidence that switch supervision is accurate.

### [METHOD] README.md:40,128 and results/README.md:10 — Poor held-out accuracy does not isolate voice memorization as the cause

The final artifacts establish failed generalization: the teacher is correct, while the student gets 18.51% of held-out frames and 4/21 clips. They do **not** establish that the student “memorises these voices” or that voice memorization is the “principal open issue.” There is no training or seen-voice held-out classification metric, and the comparison to the earlier run simultaneously changes corpus size, text set, TTS provider/voice allocation, audio, targets, number of steps, batching, and effective epochs. Target quality, class collapse, optimization, provider/gender shift, or other domain mismatch remain competing explanations.

Call the result “poor generalization to unseen synthetic voices, consistent with overfitting” unless an ablation isolates cause. Report train versus held-out agreement/known-label accuracy from the same checkpoint, plus matched-text or matched-provider seen/unseen-voice slices; a factorial provider/voice split would separate provider shift from identity shift. This still supports the honest conclusion that the current student fails its sanity check.

### [METHOD] DESIGN.md:7,44,71 — A 2.5 s rolling buffer cannot guarantee replay from VAD onset

The design promises to retain “at least 2.5 s” and, on a later commit, send audio “from the VAD onset” to the selected ASR. But the confidence policy has no decision deadline; line 71 explicitly handles the case where it never passes, and the included evaluation has `initial_commit_seconds=null`. Once an utterance has lasted more than 2.5 s, a rolling buffer has discarded its onset, so a newly selected monolingual ASR cannot perform the promised full replay. A multilingual shadow transcript does not restore the missing waveform for re-decode.

Either spool the complete pre-commit utterance (with a bounded maximum and privacy/memory policy), or impose a decision deadline no longer than buffer retention and permanently keep that utterance on the fallback after timeout. Specify behavior across long speech, silence/VAD endpoints, and ASR startup delay when sizing the buffer.

### [DEFEND] scripts/prepare_data.py:23-26,264-272 — The claimed gender coverage is not encoded for the gTTS voices

The only explicit gender metadata in code is a male Edge voice for training and a female Edge voice for held-out. The gTTS call chooses only a locale and stores `gtts:<language>:default`; it neither selects nor records a stable voice/gender. Therefore “broad gender conditions both occur in training” is an unsupported assumption about an external service, and the speaker audit proves only disjoint string identifiers, not independent human speakers.

Keep the accurate “synthetic voice identities” qualification, remove the gender-generalization claim unless voice metadata is documented, and describe this as a service-voice holdout. A defensible speaker evaluation needs multiple explicitly identified voices/speakers per language and split, with provider and gender not perfectly confounded.

## Verified correct

- The current generator has 13 unique sentences per language and enforces that count. Indices 0-9 are train and 10-12 held-out, yielding 70/21 monolingual clips with zero exact train/held-out text overlap. The train switch uses train components; both evaluation switches use held-out components.
- On the newly generated manifest, the 14 training voice IDs (gTTS plus male Edge per language) and seven held-out IDs (female Edge) have an empty intersection. Switch-evaluation IDs are the held-out Hindi/English IDs, and `require_speaker_disjoint` includes them. There is no train/held-out voice-ID leakage in the current source policy.
- Audio decode is mono 16 kHz, silence trimming preserves an 80 ms margin, peak limiting avoids clipping above 0.95, and every source rate is checked before manifest emission or concatenation. The fixed switch construction is sample-exact; the objection above is to its evaluation semantics, not an off-by-one join.
- README's final table correctly rounds the rebuilt JSON values (counts, speaker IDs, 1,600 steps/157.7465 epochs, losses, 18.51% agreement, 4/21 clips, null switch, parameters, and RTF). `DESIGN.md` correctly treats null lag as a miss rather than zero/large latency, and the exact default commands now match the recorded 1,600-step configuration.
- The final reviewed JSON snapshot is internally consistent: `train_metrics` has 1,600 finite losses and 1,600 finite gradient norms; recomputed first/last ten-step means match exactly; the stored RTF is the median of its five runs; and every overlapping field and the complete loss list in `summary.json` matches its stage artifact.
- Recomputing the held-out aggregate over 21 clips and 7,330 valid frames reproduced micro agreement `0.1851296044`, macro agreement `0.1880330373`, and 4/21 correct clip predictions. Null switch detection/lag is represented honestly in JSON, README, DESIGN, and the plot.
- `DESIGN.md` covers every Part 2 requirement: threshold/margin/dwell calibration against business cost (lines 29-40), buffered ASR routing and overlap re-decode cost (42-53), stricter challenger hysteresis and ingress-clock lag/miss/false-switch metrics (55-65), capped/decaying priors plus multilingual/reprompt/unknown fallback (67-71), and privacy-filtered teacher relabeling with human audit, versioning, replay, shadowing, canary, and rollback (73-77).
- The focused alignment, causality, waveform, and split suite passed all seven tests (the refreshed artifact records 24.81 s). Parameter count (42,567), language list, and the 435 ms stated worst-case model availability arithmetic remain consistent with the inspected implementation/results.

## Interview questions

1. **Is the evaluation genuinely speaker disjoint?** By current manifest and artifact voice IDs, yes: 14 train IDs and seven held-out IDs have no overlap, and switch evaluation uses held-out IDs. It is still only a synthetic service-voice holdout, not evidence across human speakers.
2. **Are the two switch clips independent evidence for both directions?** No. They are the same Hindi and English utterances in reverse order, and the English-first clip contains about 903 ms of padding before the annotated boundary. They are useful wiring fixtures, not two statistical trials.
3. **What does the submitted sanity check currently show?** Optimization runs and remains finite, but generalization fails: 18.51% frame agreement, 4/21 correct clips, and no Hindi→English commit. The documentation now reports that failure accurately; it does not prove which mechanism caused it.
4. **How would you make the result reproducible?** Freeze an explicit config/CLI and data recipe, content-hash audio and targets, attach Git/config/manifest/checkpoint hashes and one run ID to every stage, reject incompatible inputs, then generate README numbers from that immutable bundle.
5. **How can a router replay from VAD onset after a slow decision?** A 2.5 s ring alone cannot. Either retain the full bounded pre-commit utterance or force a fallback decision before the ring overwrites the onset; after timeout, keep that utterance on fallback.
6. **How should the Part 2 operating point be selected?** Calibrate on speaker-disjoint, codec-matched calls and optimize threshold, margin, EMA, and dwell for business cost subject to false-route and p95-latency limits, then report switch misses, premature/false switches, lag percentiles, and downstream ASR/task impact.
