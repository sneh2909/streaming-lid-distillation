# Teacher, student, policy, and route switch-lag decomposition

Research cutoff: **2026-09-26**. Web sources and Hugging Face Hub cards/API metadata were checked on that date. This note addresses backlog **METHOD-2 / SEED-2** and **POLISH-9** only.

## Bottom line

The earlier blanket statement that teacher lag had not been separated is now stale. Two isolated experiments measure the frozen ECAPA teacher trajectory without the student:

- the one-direction teacher bake-off records the first target run and its three-anchor confirmation;
- the later window and anchor-hop audits measure a 500 ms-stable target run in both directions.

That is a useful **teacher-only target-trajectory baseline**, but the main evaluator still does not publish a matched teacher -> student raw -> policy -> route decomposition. The submitted student never acquires the initial Hindi route on its scored Hindi-to-English clip, so its switch remains a miss with `lag_ms=null`; no finite student-excess or policy-overhead number exists for that event.

The key timing correction is:

- on the bake-off's Hindi-to-English trace, the stable target run starts at semantic lag **1,525 ms** and its third confirming anchor is labelled at **2,025 ms**;
- every local teacher posterior uses 250 ms of future audio, so the first target posterior is available at **1,775 ms** and the three-anchor confirmation is actionable at **2,275 ms**, not 2,025 ms.

The 2,025 ms value is a valid hindsight/semantic timestamp. It is not an online decision time. This correction does not change the bake-off's retain-ECAPA decision because all three teachers tie on that trace.

The newer two-direction audit gives stronger but still very limited evidence. At the current true-call 250 ms grid with previous-anchor hold, median 500 ms-stable semantic onset is **1,400 ms**, onset-posterior availability is **1,650 ms**, and stable confirmation availability is **2,150 ms**, all measured from the manifest's nominal 4.000 s join. The current waveforms' first threshold-active target sample is at 4.080 s, so the corresponding onset-anchored medians are **1,320 / 1,570 / 2,070 ms**. That 4.080 s point comes from the repository's amplitude heuristic, not human phonetic annotation, and is therefore not a verified speech boundary.

Call these values a *baseline on two mirrored synthetic fixtures*, not a production estimate or a mathematical latency floor. A student can lead, lag, disagree with, or miss its teacher; a different target trajectory can also behave differently.

## 1. The clocks that must not be collapsed

For each reference event, retain these clocks:

| Clock | Definition | Current example |
|---|---|---|
| `b_join` | Array concatenation point | 4.000 s |
| `b_speech` | Human/VAD-audited target-language speech onset | 4.080 s under the current amplitude heuristic; **phonetic onset unverified** |
| `t_label` | Audio frame the posterior semantically labels | ECAPA target run starts at 5.525 s on Hindi-to-English |
| `t_available` | Ingress time of the newest sample required for that posterior | 5.775 s for that 5.525 s teacher frame |
| `t_confirm_available` | Availability of the output that completes the declared stability horizon | 6.275 s for the 500 ms Hindi-to-English confirmation |
| `t_emit_wall` | Monotonic time inference emits the raw output | Not measured by the teacher-only audits |
| `t_policy_commit_available` / `t_policy_commit_wall` | Audio-ingress and monotonic clocks for the committed route decision | Main result has no commit on the scored switch |
| `t_route_wall` | Monotonic time the downstream ASR accepts the route | No router implementation/result exists |

`t_label` can be backdated relative to what the system knew. `t_available` is the earliest architectural action time, before compute. `t_emit_wall` and `t_route_wall` add compute, queue, IPC, startup, and handoff. RTF is throughput and cannot replace either wall-clock delay.

Use target-speech onset as the primary boundary:

```text
teacher_stable_lag = t_teacher_confirm_available - b_speech
student_stable_lag = t_student_confirm_available - b_speech
student_excess     = t_student_confirm_available - t_teacher_confirm_available
policy_delta       = t_policy_commit_available - t_student_confirm_available
route_handoff      = t_route_wall - t_policy_commit_wall
end_to_end_route   = t_route_wall - wall_time(b_speech)
```

The first four timestamps must refer to the same direction-aware, one-to-one matched reference event. `student_excess` and `policy_delta` are signed diagnostics, not guaranteed non-negative physical stages: a student can anticipate its teacher, and a policy can use a different dwell horizon from the diagnostic stability scorer. Always publish the absolute clocks and definitions beside any delta.

If the teacher misses, there is no teacher baseline for that event. If the student misses or fails to acquire the source language, student excess and policy delta are `null`, not zero, infinity, or the recording timeout. Route latency is separately null until an actual route handoff occurs.

## 2. Exact local evidence

### 2.1 Current 250 ms teacher trajectory

The current, identity-valid anchor audit is run `9b3e47e3c7cba82a4a8c4de8a9325dd1fb59c1ce8eff61ffece47ff8275dfd44`. It uses the pinned teacher [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9), true teacher calls, the `[t-1.75 s,t+0.25 s]` window, and previous-anchor hold.

| Direction | Stable-run semantic onset from 4.000 s join | Onset posterior available | 500 ms semantic confirmation | 500 ms confirmation available |
|---|---:|---:|---:|---:|
| Hindi -> English | 1,525 ms | 1,775 ms | 2,025 ms | 2,275 ms |
| English -> Hindi | 1,275 ms | 1,525 ms | 1,775 ms | 2,025 ms |
| Median, `n=2` | **1,400 ms** | **1,650 ms** | **1,900 ms** | **2,150 ms** |

Relative to the threshold-derived 4.080 s target-activity onset, subtract 80 ms from each entry. The median becomes **1,320 / 1,570 / 1,820 / 2,070 ms**. The direction values for actionable 500 ms confirmation become 2,195 ms and 1,945 ms.

A read-only run-boundary reconstruction found one long post-boundary target run per direction and reproduced the stored onset/confirmation timestamps. This matters because the experiment's generic matcher scans every index inside a run and can manufacture a later onset in adversarial traces. The bug does not change these two stored events, but it must be fixed before the scorer is reused.

### 2.2 Dense 10 ms teacher reference

The same audit calls ECAPA at every 10 ms feature frame. This is a dense numerical reference, not human truth:

| Direction | Stable-run semantic onset from join | Onset available | 500 ms semantic confirmation | 500 ms confirmation available |
|---|---:|---:|---:|---:|
| Hindi -> English | 1,335 ms | 1,585 ms | 1,835 ms | 2,085 ms |
| English -> Hindi | 1,185 ms | 1,435 ms | 1,685 ms | 1,935 ms |
| Median, `n=2` | **1,260 ms** | **1,510 ms** | **1,760 ms** | **2,010 ms** |

Relative to threshold-active target onset, the median is **1,180 / 1,430 / 1,680 / 1,930 ms**. Dense calls improve median semantic onset by only 140 ms and actionable confirmation by the same 140 ms versus 250 ms calls, while requiring 24.18 times as many teacher requests. They also produce 40 unmatched top-1 changes across the two clips. Density therefore does not turn this backward-heavy teacher window into a fast or stable production target.

### 2.3 What remains unmeasured

The published main result has:

- a student architecture/scheduling bound of 435 ms;
- offline replay RTF, not live per-chunk response latency;
- `initial_commit_seconds=null` and `switch_lag_ms=null` on the scored Hindi-to-English clip;
- no generic scoring of the reverse direction;
- no saved student raw/EMA/commit event table;
- no teacher trace passed through the identical student router;
- no downstream route-accept timestamp.

Therefore it is correct to say **teacher-only lag is isolated, while pipeline-wide teacher/student/policy/route decomposition is still missing**. It is incorrect to say simply that teacher lag has never been separated.

## 3. What external work supports

The literature supports separating evidence, availability, stability, and compute, but it does not provide a standard teacher/student LID decomposition.

- [Baumann, Atterer, and Schlangen](https://aclanthology.org/N09-1043/) (**NAACL, 2009-06**) distinguish a first correct incremental hypothesis from the time it becomes final and define correction time between them. That supports keeping first crossing and stable confirmation separate; the language-switch adaptation here is project-defined.
- [Zhang et al.](https://www.isca-archive.org/interspeech_2022/zhang22da_interspeech.html) (**Interspeech, 2022-09**) use a frame-synchronous streaming LID head whose second pass has 0.9 s right context and report accuracy at successive evidence checkpoints. This establishes that right context belongs in the availability budget. Their monolingual voice-search evaluation does not report code-switch event latency or teacher/student excess.
- [Červa et al.](https://doi.org/10.1016/j.csl.2020.101180) (**online 2020-12-15**) define online LID latency as input-frame-to-output-label delay and also score around language changes. That is an architectural/output delay, not automatically target-speech-onset-to-stable-route latency.
- [Aperdannier, Schacht, and Piazza](https://arxiv.org/abs/2407.04293) (**arXiv, 2024-07-05**) evaluate online diarizers on the same hardware and define latency from audio input to its label output. This supports a separate live compute/scheduler measurement; it is speaker diarization, not LID.
- [Streaming Sortformer](https://arxiv.org/abs/2507.18446) (**arXiv, 2025-07-24**) and NVIDIA's newer model card show that cache/chunk/right-context settings are part of a streaming contract. Their diarization results do not transfer numerically to language switching.

The exact additive ledger, stability horizon, and event matcher below are therefore **project proposals**, not a published standard.

## 4. Hugging Face Hub audit

This targeted audit was performed on **2026-09-26** and is non-exhaustive.

| Hub artifact | Pinned metadata | Relevant reporting lesson | Missing for this task |
|---|---|---|---|
| [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9) | revision `0253049a...`; Apache-2.0; Hub last modified 2024-11-27 | Card clearly presents utterance classification at 16 kHz. | No native streaming/frame event, future-context, or switch-lag contract. The local rolling-window schedule is this repository's construction. |
| [`tflite-hub/conformer-lang-id`](https://huggingface.co/tflite-hub/conformer-lang-id/tree/213db93aebf552836020c029a405b355569f4739) | revision `213db93a...`; Apache-2.0; Hub last modified 2024-09-19 | Public streaming LID reference covering the project languages. | Card example returns a final `top_lang`; no switch event, raw-versus-policy lag, or stability table. |
| [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization/tree/f667ed73aee57d40cc39428eb768b4fd87a0a29e) | revision `f667ed73...`; OpenMDW-1.1; released 2026-09-23, Hub last modified 2026-09-24 | Explicitly defines latency as chunk plus lookahead audio and excludes compute; reports DER and RTF separately at 1.04/0.64/0.32 s settings. | Speaker rather than language labels; no boundary-to-stable-decision or teacher/student decomposition. Hardware is GPU, so its speed numbers are not CPU evidence here. |
| [`nvidia/diar_streaming_sortformer_4spk-v2`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2/tree/84edd514b8ef68004c10086918cd62f2148cbd59) | revision `84edd514...`; CC BY 4.0; Hub last modified 2026-09-23 | Separates input-buffer latency from RTF and exposes chunk/right-context configuration. | Same speaker/LID transfer limitation and no event-stage decomposition. |

The useful Hub pattern is naming buffer latency and throughput separately. **Unverified search conclusion:** no reviewed LID card found in this audit reports a matched offline-teacher, streaming-student, policy-commit, and route-handoff event ledger.

## 5. Required machine-readable event ledger

Store one record per reference boundary and per trajectory, rather than only an aggregate `switch_lag_ms`:

```json
{
  "recording_id": "switch_hi_en_eval",
  "event_id": "boundary_0",
  "source_language": "hi",
  "target_language": "en",
  "join_sample": 64000,
  "target_speech_onset_sample": 65280,
  "boundary_annotation": "amplitude_threshold_v1_unverified",
  "teacher": {
    "raw_run_onset_label_sample": 88400,
    "raw_run_onset_latest_audio_sample": 92400,
    "stable_confirmation_label_sample": 96400,
    "stable_confirmation_latest_audio_sample": 100400,
    "policy_commit_latest_audio_sample": null
  },
  "student": {
    "raw_run_onset_label_sample": null,
    "raw_run_onset_latest_audio_sample": null,
    "stable_confirmation_latest_audio_sample": null,
    "policy_commit_latest_audio_sample": null,
    "outcome": "precondition_failure"
  },
  "route": {
    "emit_monotonic_ns": null,
    "accepted_monotonic_ns": null
  }
}
```

The numeric samples above illustrate the schema only; the builder must derive them from bound traces rather than copy this example. Record detector version, cadence, threshold, margin, smoothing, dwell, stability horizon, future context, chunk phase, and source hashes. Preserve both sample indices and derived milliseconds so rounding cannot silently move an event.

Run the same direction-aware scorer on:

1. teacher posterior/top-1 trace at a declared anchor/expansion schedule;
2. student raw posterior/top-1 trace at a common availability cadence;
3. calibrated/EMA trace;
4. committed router state;
5. route-accept events when a router exists.

Do not apply an oracle Hindi-to-English branch. Match actual run boundaries one-to-one, require the correct source state immediately before the event, reject premature target runs, and never rescan an interior point of one continuous run as a fresh onset.

## 6. Acceptance tests and decision rules

At minimum, deterministic tests should prove:

1. A teacher label at 5.525 s with 250 ms future context is available at 5.775 s; a 500 ms confirmation ending at 6.025 s is actionable at 6.275 s.
2. With `b_speech=4.080 s`, the corresponding lags are 1,695 and 2,195 ms, not the nominal-join values 1,775 and 2,275 ms.
3. A target run beginning long before a 4.0 s boundary cannot be re-matched from an interior frame at 3.75 s or later.
4. Missing source acquisition yields `precondition_failure`; missing target commit yields `lag=null` and `student_excess=null`.
5. One predicted transition can match at most one reference event, and direction must agree.
6. Semantic, availability, emission, and route timestamps are monotonic and close to the sample ledger within one sample plus a declared wall-clock tolerance.
7. Replaying identical posteriors at a finer internal cadence cannot improve a committed event merely by giving the matcher more interior indices.
8. Aggregate medians always carry detected `n/N`; with the present `N=2`, list both values and do not publish inferential p95 claims.

Do not optimize the student to make its excess lag look small while the teacher baseline remains unusably late. Primary deployment reporting remains all-boundary recall at deadlines, misses, wrong/UNKNOWN time, false switches, and end-to-end route time. The decomposition diagnoses which stage to change.

## 7. Concrete next steps

1. **Correct existing teacher-only reporting without rerunning models.** Add first-onset semantic/available and stable-confirmation semantic/available columns to the bake-off and two-direction audit summaries. Expected model gain: **none**. Acceptance: Hindi-to-English must read 1,525/1,775/2,025/2,275 ms from the nominal join, and every result must state `n`, boundary origin, future context, and whether it is actionable.
2. **Build one shared event scorer and run it over teacher, student raw, EMA, and committed traces.** Save per-direction records for both switch files, including misses and precondition failures. Expected result (**unverified**): the current student will still have at least the published Hindi-to-English precondition failure; the value is attribution, not a promised gain. Accept only if the adversarial run-interior fixtures above pass and the scorer reproduces the teacher run-boundary reconstruction exactly.
3. **After METHOD-6 supplies independent true-speech boundaries, publish the full decomposition on a frozen policy.** Report teacher stable availability, student signed excess, policy commit delta, compute/queue p50/p95/p99, route handoff, and all-boundary deadline recall. Expected direction and magnitude are **unverified**. A model change earns a latency claim only if end-to-end recall/lag improves without more misses, false switches, or wrong-route time; a smaller student-excess number alone is insufficient.

## Scope and unverified points

- The two switch files reverse one source pair and are not independent population samples.
- Their 4.080 s target onset is amplitude-threshold evidence, not human-verified phonetic timing.
- The teacher scorer's stored values are reproduced on these traces, but its run-interior matcher remains unsafe for adversarial or future data.
- The 10 ms ECAPA trajectory is a costly numerical reference with high churn, not ground truth.
- Teacher-only measurements do not prove the student can learn the trajectory or that ECAPA defines the fastest achievable switch.
- No local live router exists, so compute, queue, ASR startup, and handoff distributions remain unmeasured.
- External ASR/diarization timing conventions support the ledger design; their numerical results do not predict Hindi-English LID performance.

## Sources and dates

- Baumann, Atterer, and Schlangen, [*Assessing and Improving the Performance of Speech Recognition for Incremental Systems*](https://aclanthology.org/N09-1043/), **NAACL 2009-06**; accessed 2026-09-26.
- Červa et al., [*Identification of related languages from spoken data: Moving from off-line to on-line scenario*](https://doi.org/10.1016/j.csl.2020.101180), online **2020-12-15**; accessed 2026-09-26.
- Zhang et al., [*Streaming End-to-End Multilingual Speech Recognition with Joint Language Identification*](https://www.isca-archive.org/interspeech_2022/zhang22da_interspeech.html), **Interspeech 2022-09**; accessed 2026-09-26.
- Aperdannier, Schacht, and Piazza, [*Systematic Evaluation of Online Speaker Diarization Systems Regarding their Latency*](https://arxiv.org/abs/2407.04293), submitted **2024-07-05**; accessed 2026-09-26.
- Medennikov et al., [*Streaming Sortformer*](https://arxiv.org/abs/2507.18446), submitted **2025-07-24**; accessed 2026-09-26.
- Hugging Face model cards and API metadata linked in the table; revisions and modification dates checked **2026-09-26**.
