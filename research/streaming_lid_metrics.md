# Streaming-LID metrics: decision time, switch latency, and stability

Research cutoff: **2026-09-26**. Web sources and Hugging Face model cards/API metadata were checked on that date. Publication and artifact dates are given beside each source.

## Bottom line

There is no single, established spoken-LID benchmark that jointly standardizes **time to first correct decision, code-switch detection latency, and flip-flop rate**. This is a dated search finding, not proof that no such work exists. Reviewed papers measure incompatible pieces:

- prefix accuracy or saved ASR work for a one-language utterance;
- per-frame output delay for an online decoder;
- diarization error over the whole recording;
- or final accuracy at a fixed evidence duration.

The project should therefore publish a small, explicit metric contract. Its key rules are:

1. Time every decision on the **audio-ingress/availability clock**, not the semantic frame index.
2. Keep architectural buffering/lookahead, CPU response time, evidence/policy delay, and ASR handoff delay separate.
3. Treat an undetected event as a **miss with `lag_ms=null`**, never as zero or as an arbitrary large latency.
4. Report both conditional lag among detections **and** detection coverage at fixed deadlines. Neither is meaningful alone.
5. Count premature target predictions as false/premature events, not as negative “excellent” latency.
6. Measure instability on both raw fixed-cadence posteriors and actual committed router states. Normalize committed errors by voiced time, not by the number of frames.
7. Report time-weighted language diarization error alongside event metrics; an event score alone can hide long wrong-route intervals.

For the current repository, `algorithmic_latency_ms=435` and `switch_lag_ms=null` are correct but fundamentally different quantities. The first is a model/scheduling bound. The second says the policy never produced a valid Hindi→English event. The current evaluator also selects only `switch_hi_en_eval`, even though the manifest contains both directions, so it is a wiring check rather than a switch benchmark.

## What prior work actually measures

| Source | Date | Metric/evidence | What it establishes—and what it does not |
|---|---:|---|---|
| [Chandak et al., *Streaming Language Identification using Combination of Acoustic Representations and ASR Hypotheses*](https://arxiv.org/html/2006.00703) | arXiv 2020-06-01 | Confidence-threshold early stopping and percentage of ASR decoding saved. At threshold 0.99, examples include 43% work saved with 56% of en-US utterances stopped early and 40%/65% for es-US. The abstract reports identification about 1.5 s early for more than half of utterances. | Useful precedent for a **deadline/coverage trade-off**. It emits one utterance label; the authors disable early stopping for commonly code-switched pairs, and the mostly code-switched Indian-English/Hindi bucket still has one label. It is not a boundary-latency metric. |
| [Zhang et al., *Streaming End-to-End Multilingual Speech Recognition with Joint Language Identification*](https://www.isca-archive.org/interspeech_2022/zhang22da_interspeech.pdf) | Interspeech 2022 | Seven-language clustered accuracy is 88.1%, 94.7%, 97.6%, and 98.8% at frame checkpoints 0, 15, 30, and final; mean accuracy across time is 96.2%. The second-pass encoder has 0.9 s right context. | Shows why accuracy should be plotted against available evidence. It does not report first stable decision, wrong-first-commit rate, or switch events. |
| [Wang et al., *Attentive Temporal Pooling for Conformer-based Streaming Language Identification*](https://arxiv.org/html/2202.12163) | arXiv v4 2022-05-01 | Average accuracy on about 3.3 s voice queries and long monolingual content, plus GFLOP/s. | A genuine stateful streaming model, but cumulative prefix pooling and monolingual tests do not yield switch latency or stability metrics. |
| [Červa et al., *Identification of related languages from spoken data in an on-line mode*](https://doi.org/10.1016/j.csl.2020.101180) | online 2020-12-15; journal 2021 | Defines latency as the average time between an input frame and the decoder output of its language label; reports about 2.5 s. Also uses frame error rate and precision/recall/F-score around language changes. | Important online change-tracking precedent. Its latency includes feature/TDNN/decoder delay for a frame; it is not necessarily boundary-to-stable-router-commit time. |
| [DISPLACE 2023 challenge overview](https://arxiv.org/html/2303.00830) | arXiv v1 2023-03-01; v3 2023-06-05 | Language diarization error rate: false alarm + missed speech + wrong-language time, divided by reference duration; overlap is scored and the primary metric uses no forgiveness collar. | Supplies the complementary **time-weighted quality** metric. DER alone does not reveal when a correct switch first became usable. |
| [Patino et al., *Low-latency speaker spotting with online diarization and detection cost*](https://www.eurecom.fr/en/publication/5522/download/sec-publi-5522.pdf) | Odyssey 2018 | Event latency is threshold-crossing time minus target-start time; evaluates detection cost versus latency and at fixed deadlines. It distinguishes absolute delay from how much target speech was consumed. | Speaker spotting, not LID, but the event/deadline method transfers. This project should **not** copy the paper's miss-time substitution: misses remain null and are reported separately. |
| [Baumann, Atterer, and Schlangen, *Assessing and Improving the Performance of Speech Recognition for Incremental Systems*](https://aclanthology.org/N09-1043/) | NAACL 2009-06 | Separates “word first correct” from “word first final” and introduces correction time and edit overhead. | Strong precedent for separating a transient correct output from a stable/committed correct output. The exact LID adaptation below is a project proposal, not a published standard. |
| [Bruguier et al., *Flickering Reduction in End-to-End Contextual Biasing for Streaming Speech Recognition*](https://www.bruguier.com/pub/deflickering.pdf) | IEEE SLT 2023-01 | Reports partial-result quality, latency, and unstable-word/unstable-segment ratios together. | Supports measuring user-visible churn separately from final accuracy. It concerns ASR words, so an LID “flip-flop rate” still needs an explicit local definition. |
| [NIST LRE22 evaluation site](https://lre.nist.gov/) and [NIST's LRE17 performance analysis](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=925490) | LRE22 / 2018 analysis of LRE17 | LRE17 used nested 3, 10, and 30 s speech segments; LRE22 test speech spans roughly 3–30 s. | Supports publishing fixed evidence budgets, but these are offline trials rather than streaming commit times. |

The literature therefore does **not** justify using one number called “latency.” At least three delays and one throughput measure can otherwise be conflated.

## A clock contract for reproducible latency

For every model or router output, record four times:

| Symbol | Meaning | Example |
|---|---|---|
| `t_label` | Audio time whose state the output semantically labels | Center/end of a 10 ms feature frame |
| `t_available` | Audio-ingress time of the newest sample required before that output can exist | Includes analysis window, bounded lookahead, label delay, and chunk scheduling |
| `t_emit_wall` | Monotonic wall-clock time when inference/policy emits it | Includes CPU queueing and compute |
| `t_route_wall` | Monotonic wall-clock time when the downstream route accepts/acts on it | Includes IPC, model start, and handoff |

Then report these quantities separately:

- **Architectural/input-buffer delay:** `t_available - t_label`. For a fixed streaming graph this is a bound or distribution driven by chunk alignment.
- **Compute/scheduler response:** `t_emit_wall - wall_time_when(t_available_arrived)`, measured on the target CPU with p50/p95/p99 and deadline-miss rate.
- **Evidence + policy decision delay:** decision `t_available - speech_onset` for initial LID, or decision `t_available - switch_boundary` for a switch.
- **End-to-end route delay:** `t_route_wall - wall_time_when(reference_event_arrived)`.

Real-time factor is useful capacity evidence, but **RTF is not response latency**. A system can average RTF 0.01 offline and still queue a 2 s window before emitting anything. Conversely, fixed lookahead can dominate latency even when compute is negligible.

Offline replay may reconstruct `t_available` exactly from the causal schedule. It must not time an event from `t_label`, because that back-dates the decision to the frame being classified. Live tests should use a monotonic clock and preserve audio capture timestamps through buffering and IPC.

The [NVIDIA Nemotron-3-Diarization model card](https://huggingface.co/nvidia/Nemotron-3-Diarization) (last modified 2026-09-24) is a useful **speaker-diarization transfer example**, not LID evidence: it reports DER at several input-buffer configurations and explicitly defines algorithmic latency from chunk plus right context while excluding compute. That separation should be copied; its speaker results should not be treated as language results.

## 1. Initial-language decision metrics

Let `s` be the human- or VAD-verified onset of target speech, excluding leading file silence. Let committed router states be `(a_j, z_j)`, where `a_j` is availability time and `z_j` is a supported language or `UNKNOWN`.

### Required event times

1. **Time to first known commit (`TTFK`)**: `min(a_j - s : z_j != UNKNOWN)`.
2. **First-commit correctness**: whether that first known commit equals reference language `y`.
3. **Time to first correct committed decision (`TTFCD`)**: `min(a_j - s : z_j = y)`. This is null if the system never commits correctly.
4. **Time to stable correct route (`TTSCR`)**: first correct commit that remains correct for a predeclared stability horizon `H_stable`, or until the reference segment ends if it is shorter. Use `H_stable=1,000 ms` as a starting protocol, frozen on development data.
5. **Correction time after a wrong first commit**: time from the wrong first commit to the first subsequent stable correct route.

`TTFCD` alone is gameable: a system can guess rapidly, route to the wrong recognizer, and look good once it later touches the right label. First-commit correctness, stable time, and wrong-route duration expose that failure.

### Fixed-deadline outcome curves

At `τ = 250, 500, 1,000, 2,000, 3,000 ms` after speech onset, report a three-way partition:

- **correct first commit by τ**;
- **wrong first commit by τ**;
- **no known commit by τ**.

Also report **active route correct at τ**, because a wrong-first system may have recovered by then. Plot these values over τ if the sample is large enough. This mirrors early-exit work while preserving the operational cost of wrong routing.

Report conditional median/p90/p95 `TTFCD` and `TTSCR` only among successful detections, always beside their numerator, denominator, and miss rate. Do not substitute the recording end, zero, or a timeout for a miss. With fewer than roughly 20 detected events, list all event values; tail percentiles are not credible.

Slice at minimum by language, speaker, clip-duration bin, channel, and signal-to-noise regime. The seven-language macro result must not let English volume conceal Hindi or the five other Indian languages.

## 2. Switch-detection latency

### Reference events

Each reference boundary should contain:

```text
(recording_id, boundary_sample, source_language, target_language,
 target_span_end_sample, boundary_uncertainty, switch_type, speaker_id)
```

`boundary_sample` is the onset of target-language **speech**, not a concatenation timestamp or the end of padded silence. Mark sustained language changes separately from single borrowed words/insertions. Keep overlap and genuinely uncertain regions explicit. Synthetic boundaries can be sample exact; natural boundaries should retain annotator uncertainty rather than hiding it behind an arbitrary scoring collar.

### Predicted events and eligibility

A predicted switch is a change in the **committed known-language sequence** from `A` to `B`. An `UNKNOWN→A` acquisition with no earlier known state is an initial commit, not a switch. If the route goes `A→UNKNOWN→B`, the eventual `B` commit can represent the known-language change, but the unknown interval is also charged as route churn/abstention time.

A reference `A→B` event is eligible for normal switch-latency scoring only if the committed route was correctly `A` immediately before the boundary. Report the **prior-state-correct rate**. If the system never acquired `A`, classify the event as a `precondition_failure` and as an all-boundary miss; do not award a fast `B` guess as a successful switch. A secondary “target acquisition after boundary” diagnostic may still be recorded.

### Matching and lag

Match predicted and reference events one-to-one, chronologically, and direction-aware. For an exact reference boundary `b_k`, a candidate must occur in:

```text
[b_k, min(b_k + deadline, target_span_end, next_reference_boundary))
```

and must commit the correct target language. A predicted event may match only one boundary. A short target span that ends before the system commits is a miss; the match must not leak into the next segment.

For a matched event at availability time `d_k`:

```text
switch_lag_k = d_k - b_k
```

A target commit before `b_k` is a **premature/false switch**, not a successful event with negative latency. If that premature state persists through the real boundary, record `already_target_due_to_premature`; wrong-route time after the boundary may be zero, but event detection was not successful. Signed raw-posterior crossings may be retained for diagnostics, clearly separated from the operational metric.

For uncertain natural boundaries, retain `[b_low, b_high]` and report the lag interval or nominal adjudicated lag plus uncertainty. A 250 ms collar may be shown as an annotation-sensitivity result, but a collar must never be subtracted to claim lower detection latency.

### Required switch outputs

- Boundary recall at 500, 1,000, 2,000, and 3,000 ms, both across **all** boundaries and conditional on a correct prior state.
- Matched lag median/p90/p95 with detected `n`, total `N`, and confidence intervals.
- Counts/rates of `miss`, `precondition_failure`, `premature`, wrong-direction, and other unmatched events.
- False committed switches per voiced hour.
- Wrong-route, `UNKNOWN`, and correct-route speech seconds after each boundary.
- Results by `A→B` direction, target-span duration, speaker, channel, and switch type.

The all-boundary view answers whether the router works end to end. The eligible-boundary view isolates transition behavior after successful initial acquisition. Both are necessary.

## 3. Flip-flop and stability metrics

**Unverified/project-defined:** the review found no standard spoken-LID definition of “flip-flop rate.” The following definitions adapt incremental-ASR edit/flicker concepts and must be versioned with the evaluator.

### Raw diagnostic stability

Sample raw argmax, calibrated argmax, and EMA winner on a fixed cadence, such as every emitted 160 ms chunk. For each stream report:

- known-language transitions per voiced minute;
- posterior total variation per voiced minute, `Σ ||p_i - p_(i-1)||_1 / 2`;
- first crossing and first `H_stable`-persistent crossing for each real boundary.

Raw transition counts cannot be compared across systems with different output cadence unless both are resampled to the same availability-time grid. They diagnose the model/smoother, not user-visible routing.

### Committed router stability

Use these event rates:

- **False-switch rate:** unmatched committed known-language changes per voiced hour.
- **Ping-pong/reversal rate:** a committed `A→B` followed by `B→A` within a development-frozen horizon `H_flip` (start with 2 s), with no matching reference reversal, per voiced hour.
- **Brief false-episode rate:** unmatched known-language route episodes shorter than `H_flip`, per voiced hour.
- **UNKNOWN churn:** transitions into or out of `UNKNOWN` per voiced hour and total `UNKNOWN` speech seconds.
- **Route-change rate:** every downstream ASR handoff per voiced hour, since known-language and UNKNOWN transitions may have different product costs.

On a monolingual control, every committed known-language transition is false. Silence should not vote for a language: either exclude it from voiced denominators and report false alarm on nonspeech separately, or freeze/decay toward `UNKNOWN` using one declared rule. A system must not reduce its false-switch rate merely by abstaining indefinitely, which is why UNKNOWN time and decision coverage are mandatory companions.

Normalize by voiced duration rather than frames or update count. Report speaker/call-level distributions, not only a pooled rate; one pathological long call can otherwise dominate.

## 4. Time-weighted quality: language DER

Use no-collar language diarization error rate as the primary time-weighted score:

```text
LD-DER = (missed-reference-speech + false-alarm-speech + wrong-language-speech)
         / total-reference-speech
```

For this router, decompose it into:

- supported-language speech routed to `UNKNOWN` or no route;
- nonspeech routed to a language;
- speech routed to the wrong supported language;
- optional out-of-set speech incorrectly forced into a supported language.

Report the components, because the same total can represent very different product behavior. Keep overlaps scored or state exactly how they are handled. A supplementary collar result is acceptable for comparison with older diarization papers, but the no-collar result is the one aligned with low-latency routing.

LD-DER complements rather than replaces boundary metrics. A model could detect every boundary late and still have modest DER on long segments, or detect boundaries quickly while producing many short false episodes.

## 5. Recommended scorecard

| Question | Primary outputs |
|---|---|
| How soon can a call be routed? | Correct/wrong/no-commit proportions at 0.25/0.5/1/2/3 s; first-commit correctness; TTFCD and stable-route median/p90/p95 with misses |
| Can it follow real switches? | All-boundary and prior-correct recall at 0.5/1/2/3 s; matched lag distribution; precondition failures; misses; premature events |
| Does it chatter? | False known-language switches/hour, 2 s reversals/hour, brief false episodes/hour, UNKNOWN churn/hour, route handoffs/hour |
| How much audio is routed incorrectly? | No-collar LD-DER components; wrong-route and UNKNOWN speech seconds; downstream re-decoded/buffered seconds |
| Is latency architectural or computational? | Architectural availability delay; per-chunk compute p50/p95/p99; queue/deadline misses; end-to-end handoff delay; RTF reported separately |
| Does threshold tuning merely trade one failure for another? | Accuracy/coverage/latency/false-switch Pareto curves on frozen development data; one untouched test operating point |
| Does routing help the actual application? | Per-language ASR WER or task success, wrong-ASR seconds, ASR restarts, buffered/redecoded seconds |

Do not collapse this table to one leaderboard score prematurely. Threshold, margin, EMA, and dwell sweeps should expose the Pareto frontier. Choose the operating point on speaker-disjoint development data under explicit constraints, then run it once on frozen test data.

Confidence intervals should resample independent calls or speakers, not frames. For small switch sets, publish the event table itself. At least hundreds of boundaries across speakers, directions, segment lengths, and channels are needed before p95 claims are useful; the present two synthetic reversals are only deterministic fixtures.

## Minimal machine-readable event record

```json
{
  "recording_id": "...",
  "speaker_id": "...",
  "reference": {
    "source": "hi",
    "target": "en",
    "boundary_sample": 64000,
    "boundary_seconds": 4.0,
    "uncertainty_ms": 0,
    "target_end_seconds": 8.0,
    "switch_type": "sustained"
  },
  "pre_boundary_route_correct": false,
  "outcome": "precondition_failure",
  "matched": false,
  "decision_available_seconds": null,
  "decision_emit_wall_seconds": null,
  "lag_ms": null,
  "raw_first_crossing_ms": null,
  "raw_stable_crossing_ms": null,
  "ema_stable_crossing_ms": null,
  "wrong_route_ms_after_boundary": 4000,
  "unknown_ms_after_boundary": 4000
}
```

Aggregate artifacts should carry evaluator version, policy configuration, output cadence, stability and flip horizons, match deadlines, collar/overlap rules, clock type, dataset revision, and bootstrap unit. Without those fields, rates from two runs are not comparable.

## Hugging Face Hub audit

This was a targeted card/API audit on **2026-09-26**, not an exhaustive proof about all Hub uploads.

| Artifact | Hub metadata at audit | Metrics exposed on the card |
|---|---|---|
| [`tflite-hub/conformer-lang-id`](https://huggingface.co/tflite-hub/conformer-lang-id) ([API metadata](https://huggingface.co/api/models/tflite-hub/conformer-lang-id)) | Apache-2.0; last modified 2024-09-19 | Described as streaming, but the high-level example returns one final `top_lang`. No public-checkpoint TTFCD, switch recall/lag, flip rate, or accuracy table was found. **Unverified:** its operational switch behavior. |
| [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) ([API metadata](https://huggingface.co/api/models/speechbrain/lang-id-voxlingua107-ecapa)) | Apache-2.0; last modified 2024-11-27 | Reports 6.7% VoxLingua107 development error for utterance classification. No prefix/deadline, boundary, or stability evaluation. |
| [`facebook/mms-lid-126`](https://huggingface.co/facebook/mms-lid-126) ([API metadata](https://huggingface.co/api/models/facebook/mms-lid-126)) | CC-BY-NC-4.0; last modified 2023-06-13 | The card documents utterance classification and labels but provides no streaming metric table. |
| [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization) ([API metadata](https://huggingface.co/api/models/nvidia/Nemotron-3-Diarization)) | OpenMDW License Agreement 1.1 (`openmdw-1.1`); last modified 2026-09-24 | Speaker diarization, not LID. It usefully reports DER at 30.4/1.04/0.64/0.32 s input-buffer latencies, defines the buffer equation, and reports RTF separately. This is a reporting template only. |

The audit reinforces the literature finding: even a model advertised as streaming may expose only a final label or only average accuracy. Model cards should report the decision/coverage/stability scorecard above before claiming switch-capable streaming LID.

## Repository-specific gap analysis

Checked against the repository on 2026-09-26:

- `scripts/eval.py` is explicitly a Hindi→English evaluator. It selects `switch_hi_en_eval`, uses hard-coded `hi` and `en` indices, and ignores the held-out reverse-direction clip for switch scoring.
- It emits one `switch_lag_ms`; it does not perform generic event matching, deadline recall, wrong-first/no-commit partitions, false-switch counting, or flip/reversal analysis.
- `results/eval_metrics.json` correctly has `switch_detected_seconds=null` and `switch_lag_ms=null`. Because the initial Hindi state was never committed, the event should additionally be labelled `precondition_failure` and an all-boundary miss.
- The reported `algorithmic_latency_ms=435` is not evidence of a 435 ms decision. Teacher-only evidence is now partially separated: on two mirrored synthetic switches, the current 250 ms previous-hold ECAPA trajectory has 1,400 ms median 500 ms-stable semantic onset and 2,150 ms median confirmation availability from the nominal join (1,320/2,070 ms from the amplitude-derived target onset). The main evaluator still lacks matched student-raw, EMA/commit, compute, and route stages; its scored student event is a precondition failure/miss. Exact clocks and the one-direction 2,275 ms confirmation correction are in [`teacher_student_policy_lag_decomposition.md`](teacher_student_policy_lag_decomposition.md).
- The fixed 4 s nominal join is not necessarily a speech boundary: backlog audits found 158 ms and 903 ms of silence before nominal joins in the two directions. Scoring needs sample-accurate speech onsets.

### Deterministic evaluator fixtures

A generic scorer should be unit-tested with at least these cases:

1. No commit: miss, `lag_ms=null`; never zero or the timeout.
2. Correct initial commit and no reference switch: no predicted switch.
3. Target commit before boundary: one premature/false event, not a negative matched lag.
4. Correct `A→B` followed by unreferenced `B→A` within 2 s: one matched event plus one false reversal.
5. Multiple reference switches: chronological one-to-one matches; no prediction reused.
6. `UNKNOWN→A`: initial acquisition, not a language switch; UNKNOWN transitions counted as churn.
7. `A→UNKNOWN→B`: known-language target event plus separately charged UNKNOWN duration/churn.
8. Semantic frame time 1.0 s, availability time 1.4 s, boundary 0.8 s: lag must be 600 ms, not 200 ms.
9. Target span ends before a late commit: miss; the late commit cannot match the following boundary.
10. Missing correct pre-boundary route: `precondition_failure`, even if target language is already active at the boundary.

These fixtures are a stronger near-term acceptance test than any percentile calculated from two synthetic clips.

## Unverified points and scope limits

- **Unverified/search conclusion:** no standard complete trio of TTFCD, switch-detection lag, and flip-flop rate was found. Terminology differs across LID, diarization, incremental ASR, and speaker spotting.
- **Project proposal, not published standard:** the exact deadline grid, `H_stable=1 s`, `H_flip=2 s`, eligibility rule, and false-switch/reversal definitions. Freeze and version them after development-set validation.
- Published speaker-spotting, ASR flicker, and speaker-diarization metrics are methodological analogies. Their numerical results do not predict Hindi–English LID performance.
- The Hugging Face audit records what model cards exposed on 2026-09-26. Absence from a card is not proof that an author never measured a quantity elsewhere.
- P95 latency, per-hour false-switch rates, and speaker bootstrap intervals cannot be estimated credibly from this repository's two synthetic switch clips.

## Sources and dates

- Chandak et al., [arXiv:2006.00703](https://arxiv.org/abs/2006.00703), submitted **2020-06-01**; HTML and tables checked **2026-09-26**.
- Zhang et al., [ISCA Interspeech 2022 paper](https://www.isca-archive.org/interspeech_2022/zhang22da_interspeech.html), published **2022**; PDF checked **2026-09-26**.
- Wang et al., [arXiv:2202.12163](https://arxiv.org/abs/2202.12163), v4 **2022-05-01**; checked **2026-09-26**.
- Červa et al., [Computer Speech & Language DOI 10.1016/j.csl.2020.101180](https://doi.org/10.1016/j.csl.2020.101180), available online **2020-12-15**; checked **2026-09-26**.
- Kukkadapu et al., [DISPLACE challenge overview, arXiv:2303.00830](https://arxiv.org/abs/2303.00830), v1 **2023-03-01**, v3 **2023-06-05**; checked **2026-09-26**.
- Patino et al., [Odyssey 2018 paper](https://www.eurecom.fr/en/publication/5522), published **2018**; PDF checked **2026-09-26**.
- Baumann, Atterer, and Schlangen, [ACL Anthology N09-1043](https://aclanthology.org/N09-1043/), published **2009-06**; checked **2026-09-26**.
- Bruguier et al., [author-hosted SLT paper](https://www.bruguier.com/pub/deflickering.pdf), conference **2023-01**; checked **2026-09-26**.
- NIST, [LRE22](https://lre.nist.gov/) and [*Performance Analysis of the 2017 NIST Language Recognition Evaluation*](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=925490), evaluation/paper years **2022/2018**; checked **2026-09-26**.
- Hugging Face cards and API metadata linked in the Hub table, checked **2026-09-26**.
