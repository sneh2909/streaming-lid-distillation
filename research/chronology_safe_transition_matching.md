# Chronology-safe transition matching for streaming LID

Research cutoff: **2026-09-26**. Web sources and Hugging Face Hub cards/API
metadata were checked on that date. This note addresses builder backlog
**METHOD-41** only.

## Bottom line

The project needs one scorer that extracts each predicted state run once, at
its real start, and then matches predicted acquisitions to reference language
changes in chronological one-to-one order. The current experiment scorers do
not provide that contract:

- some scan every index inside a run, so one target run can first be rejected
  as premature and later be accepted from an invented interior onset;
- some let any old source run arm the detector forever;
- one reports a target detection even when `source_armed_before_boundary` is
  false;
- policy scorers can acquire the source only after the reference boundary and
  still report the later target as a detected switch; and
- the main evaluator has no reference boundary argument at all, so it cannot
  distinguish a valid switch from a late source acquisition.

The replacement should operate on integer sample clocks, compress a trace to
maximal runs, qualify each run without moving its onset, require the source
state to be current and already qualified at the reference boundary, and use
a monotone maximum-cardinality one-to-one match. Semantic time, latest-audio
availability, wall-clock emission, and route acceptance must remain separate.

This is a scoring correction, not a model improvement. A read-only audit of
the stored traces predicts that the current 1.75 s-past/250 ms-future teacher
baseline and anchor-hop conclusions remain unchanged. Three already-rejected
shorter-window events become precondition failures, strengthening rather than
reversing that negative result. These reclassifications are **not official
rerun results** until the shared scorer is implemented, versioned, and the
artifacts are republished.

## 1. Exact local failure modes

### 1.1 A premature run can be re-entered

`experiments/teacher-window-audit/run.py:394-488` and
`experiments/teacher-anchor-hop-audit/run.py:288-377` call `run_end` from every
array index. They do not restrict candidates to index zero or a class change.

An exact six-thread probe of the current window-audit function used:

- source class at 0.025, 0.275, and 0.525 s;
- one uninterrupted target run beginning at 0.775 s;
- a 4.000 s reference boundary;
- 500 ms stability; and
- 250 ms future context.

The function simultaneously returned:

| Field | Current result |
|---|---:|
| `premature_target_onset_seconds` | 0.775 |
| `detected` | `true` |
| manufactured semantic onset | 3.775 s |
| semantic lag | -225 ms |
| onset availability | 4.025 s |
| confirmation availability | 4.525 s |

The 3.775 s point is not a transition. It is an interior sample of the run
that began at 0.775 s. Once that real run is classified as premature, no later
index from it may be offered to the matcher.

### 1.2 Detection is not gated by the source precondition

`experiments/telephony-robustness/run.py:892-962` has two independent issues.
`stable_run` also scans interior indices, and `stable_switch_event` records
`source_armed_before_boundary` but never requires it for `detected=true`.

On a trace containing the target class at every 250 ms sample, with a 1.000 s
boundary, the current function returned:

```text
source_armed_before_boundary = false
detected                     = true
first_stable_seconds         = 1.000
confirmed_seconds            = 1.500
```

There was no source state and no source-to-target event.

The policy path at `run.py:965-1018` can also arm after the boundary. A second
exact probe used a 160 ms cadence, a 0.500 s boundary, an initially wrong
class, a long source run beginning after the boundary, and then a target run.
The current scorer committed the source at 1.280 s, correctly exposed
`source_committed_before_boundary=false`, but still returned a detection at
3.040 s with `lag_ms=2540`.

The related target-ablation functions at
`experiments/target-type-ablation/run.py:763-854` likewise have no boundary in
their source-arming step. The main `detect_hi_to_en_switch` at
`scripts/eval.py:111-160` has no boundary input and cannot express the
precondition at all.

### 1.3 An ancient source run remains armed

The teacher audit scans for any stable source run whose confirmation is before
the boundary. A correct source interval near the beginning of a clip therefore
remains armed even if the model is in a third language immediately before the
reference change. For routing, the relevant question is not "was the source
ever observed?" but "was the source the current qualified state at the
boundary?"

## 2. Read-only audit of stored results

The audit below is bound to these exact cached result bytes:

| Artifact | SHA-256 |
|---|---|
| `experiments/teacher-window-audit/results.json` | `791a6345c0ea48e99a1e240d88ce603b2b322aec617a54135ed56ca41c11a66c` |
| `experiments/teacher-anchor-hop-audit/results.json` | `89a4a24378dcde5ce5e7553ac6f839059a90a8d595c0d88ed441a53be05178a7` |
| `experiments/telephony-robustness/results.json` | `93712aeb7086bf8e712f89441af12dffa0eee08c5b6db5bcdf4db7129d0940b3` |

No model was run and no result artifact was modified.

### 2.1 Actual-run-start check

Across 63 stored detected teacher events from the anchor-hop audit, every
window-audit stability horizon, and every telephony condition, **63/63 stored
onsets happen to be true maximal-run starts**. These are repeated diagnostics
on two mirrored fixtures, not 63 independent trials. Thus the interior-run bug
is real and regression-reproducible, but it does not by itself alter those
stored events.

### 2.2 Boundary-adjacent source qualification

For each 500 ms event, I treated the prediction stream as a previous-state-hold
sequence on its **latest-audio availability clock**. The source precondition
passes only when the state active at the boundary is the reference source and
that same source run has already spanned at least 500 ms.

| Teacher window | Stored detections | Precondition-valid detections | Expected effect |
|---|---:|---:|---|
| current 1,750 ms past / 250 ms future | 2/2 | 2/2 | unchanged |
| bounded 750 ms past / 250 ms future | 2/2 | 1/2 | reverse event becomes precondition failure |
| causal 500 ms past | 2/2 | 1/2 | reverse event becomes precondition failure |
| causal 1,000 ms past | 2/2 | 1/2 | reverse event becomes precondition failure |
| causal 2,000 ms past | 2/2 | 2/2 | unchanged |
| cumulative prefix | 1/2 | 1/2 | unchanged |

All four anchor-hop arms retain 2/2 source-qualified events. All 22 online
teacher events across the 11 telephony conditions also retain a qualified
source at the boundary on the availability clock.

This last detail shows why clocks cannot be substituted. In the reverse
`noise_10db` and `noise_10db_pcmu` traces, the last *semantic* top-1 at or
before 4.000 s is not English, but that posterior needs 250 ms of future audio
and is not available at 4.000 s. The most recent *available* state is English,
so the online source precondition passes. Scoring semantic coordinates as if
they were online states would incorrectly reject those two events.

The shorter-window study already rejects every challenger on its full gate,
the anchor-hop study rejects every sparse grid on fidelity, and every student
telephony arm already misses both switches. The expected count changes above
therefore do not rescue or overturn an adopted result. They remove false
positive evidence from three rejected arms.

## 3. Canonical inputs

### 3.1 Reference events

Build reference events from sample-indexed speech intervals, not from model
outputs:

```text
ReferenceEvent {
  recording_id
  event_id
  source_language
  target_language
  boundary_sample
  target_end_sample_exclusive
  annotation_uncertainty_samples
  annotation_method
}
```

`boundary_sample` should eventually be the human/VAD-audited target-speech
onset from METHOD-6. Until then, every result must say that the current 64,000
sample / 4.000 s value is an array join rather than a verified phonetic
boundary.

### 3.2 Timed state points

Represent teacher raw, student raw, EMA, and committed route states through
one schema:

```text
StatePoint {
  sequence_index
  label
  semantic_sample
  latest_audio_sample_exclusive
  emitted_monotonic_ns | null
  route_accepted_monotonic_ns | null
}
```

Integer samples are canonical. Derived seconds are presentation fields. The
loader must reject non-monotonic sequence indices, semantic clocks, or
latest-audio clocks; missing labels; duplicate stable emissions; and a latest
audio sample inconsistent with the bound trajectory definition.

For a committed trajectory, the state point is the actual commit, after its
threshold/margin/dwell logic. Do not apply the raw 500 ms stability qualifier
again unless that is explicitly named as a separate diagnostic.

## 4. Extract events once

### 4.1 Maximal runs

Compress each state trace into maximal equal-label runs. A run starts only at
index zero or where `label[i] != label[i-1]`. Assign an immutable `run_id`.
Never call an interior index a new onset.

For a requested raw/EMA stability horizon, a run qualifies at the first point
in the same run whose semantic span from the real onset reaches the horizon.
Record both onset and confirmation clocks. If the trace ends before a
confirming observation, the run does not qualify; do not infer stability from
right padding or clip termination.

An offline teacher can semantically backdate an onset because of right
context. Match on `onset_latest_audio_sample`, not the backdated semantic
coordinate. For example, a semantic onset at 3.775 s requiring audio through
4.025 s is an online +25 ms acquisition around a 4.000 s boundary, while a run
that really began at 0.775 s and was available at 1.025 s is premature even if
it persists through the boundary.

### 4.2 Source precondition

At each reference boundary, evaluate the state held on the availability clock:

1. take the newest state whose `latest_audio_sample_exclusive <=
   boundary_sample`;
2. require its active maximal run to have the reference source label;
3. for raw/EMA trajectories, require that source run's stability confirmation
   to have been available no later than the boundary; and
4. for a committed trajectory, require the source route to have committed by
   and remain active at the boundary.

If any condition fails, emit `precondition_failure`. A target acquired later
cannot turn it into a successful source-to-target switch. This prevents both
the ancient-arm and late-arm errors.

### 4.3 Target acquisition candidates

Offer only qualified maximal runs whose label equals the reference target.
The run's actual availability onset must fall no earlier than the boundary
minus the explicit annotation uncertainty and before the target interval ends.
Its confirmation must also occur before that target interval ends. A model
lookahead is represented by the latest-audio clock and must not be hidden in an
extra forgiveness collar.

Intervening `UNKNOWN` or wrong-language time after a valid source precondition
does not erase an eventual correct target acquisition. It is charged
separately as UNKNOWN/wrong-route duration and as unmatched state changes.
Record the target run's immediate predecessor so a stricter direct
`source -> target` diagnostic remains available, but do not silently change
the primary latency definition. This choice is a **project proposal**, not a
published LID standard.

## 5. Chronological one-to-one matching

Sort references by boundary sample and candidates by availability onset plus
sequence index. Construct an edge only when:

- the reference source precondition passes;
- the candidate label equals the reference target;
- onset and confirmation fall in the reference target interval; and
- any declared annotation uncertainty/deadline rule is satisfied.

Use monotone dynamic programming, not independent per-boundary scans:

1. maximize the number of matched reference events;
2. among maximum-cardinality solutions, minimize total confirmation-availability
   lag; and
3. break remaining ties by earlier candidate sequence index.

The monotone constraint requires `i < k` and `j < l` for matches
`reference[i] -> candidate[j]` and `reference[k] -> candidate[l]`. One
candidate can match at most one reference and matches cannot cross in time.
Persist the matcher version and tie-break order in every result.

This adapts two established ideas rather than claiming a new standard. NIST's
2016 keyword-occurrence scorer uses minimum-cost one-to-one bipartite matching
inside a temporal tolerance so two hypotheses cannot claim one occurrence
([evaluation plan, 2016-02-08](https://www.nist.gov/system/files/documents/itl/iad/mig/KWS16-evalplan-v04-ComparedTo-KWS15v05.pdf)).
The official `sed_eval` event scorer offers optimal graph matching because its
legacy greedy matcher can depend on reference ordering
([documentation, accessed 2026-09-26](https://tut-arg.github.io/sed_eval/generated/sed_eval.sound_event.EventBasedMetrics.html)).
Neither defines source-conditioned, streaming language-switch latency; the
monotone direction/precondition rules here are project-specific.

## 6. Outcomes and clocks

Each reference event must have exactly one outcome:

- `detected`: source precondition valid and one target run matched;
- `precondition_failure`: source was not the qualified current state at the
  boundary;
- `miss`: source precondition valid but no candidate matched before target end;
  or
- `not_evaluable`: required reference timing/region is absent or excluded by a
  predeclared protocol.

Unmatched predicted events are separately classified as premature, wrong
direction/label, false switch, or reversal. A miss or precondition failure has
`lag=null`; it is never zero, infinity, or clip timeout.

For a match, retain at least:

```text
semantic_onset_lag
onset_availability_lag
semantic_confirmation_lag
confirmation_availability_lag
emission_wall_lag | null
route_accept_wall_lag | null
```

Aggregate every latency with detected `n/N`, all-boundary recall at fixed
deadlines, precondition-valid recall, misses, precondition failures, unmatched
events, wrong/UNKNOWN speech time, and no-collar language DER. Do not report a
detected-only median alone.

Duration and event views are complementary. DISPLACE evaluates Indian
language diarization using duration-weighted DER, with overlap and no collar
([Baghel et al., Interspeech 2023-08](https://www.isca-archive.org/interspeech_2023/baghel23_interspeech.html)).
Event-based and segment-based metrics answer different questions, as reviewed
by [Mesaros, Heittola, and Virtanen, published 2016-05-25](https://doi.org/10.3390/app6060162).
DER therefore cannot replace one-to-one acquisition latency, and event latency
cannot replace wrong-language duration.

## 7. Required deterministic fixtures

The shared scorer should fail publication unless all of these pass:

1. **Interior re-entry:** the exact 0.775 s target run above is one premature
   event; the 3.775 s interior is never a candidate.
2. **No source:** an all-target trace yields `precondition_failure`, not a
   zero-lag detection.
3. **Late source arm:** source acquired after the boundary remains
   `precondition_failure` even if target is later acquired.
4. **Wrong direction:** a stable `B -> A` prediction cannot match reference
   `A -> B`.
5. **Reused event:** one predicted run offered to two references matches at
   most one; the other reference is a miss or precondition failure.
6. **Crossing candidates:** two references and two candidates use a monotone
   assignment even when unconstrained minimum-cost edges would cross.
7. **Short target span:** confirmation after `target_end_sample_exclusive` is a
   miss, not a delayed hit.
8. **UNKNOWN bridge:** `A -> UNKNOWN -> B` can acquire B after a valid A
   boundary state, while UNKNOWN duration/churn remains charged.
9. **Availability clock:** semantic time 1.000 s, latest audio 1.400 s, and a
   0.800 s boundary yield 600 ms availability lag, not 200 ms.
10. **Right-context lead:** semantic onset 3.775 s/latest audio 4.025 s around a
    4.000 s boundary keeps both -225 ms semantic lead and +25 ms availability
    lag without collapsing them.
11. **Clip tail:** an unconfirmed final target run remains a miss.
12. **Cadence invariance:** inserting duplicate interior samples into a run
    cannot create a new event, change its onset, or improve recall.

Replay the two stored switch fixtures as regression controls. Under the
availability-clock precondition, the current 250 ms teacher grid should retain
2/2 events and its existing per-direction clocks. The three shorter-window
reverse events listed in Section 2 should become precondition failures.

## 8. Hugging Face Hub audit

This targeted audit is non-exhaustive and was performed on **2026-09-26**.

| Hub artifact | Pinned metadata | What it establishes | Missing here |
|---|---|---|---|
| [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9) | revision `0253049a...`; last modified 2024-11-27; Apache-2.0 | The current teacher is an utterance-level 16 kHz LID classifier. | No native language-transition events or switch matcher; the rolling window is local project logic. |
| [`tflite-hub/conformer-lang-id`](https://huggingface.co/tflite-hub/conformer-lang-id/tree/213db93aebf552836020c029a405b355569f4739) | revision `213db93a...`; last modified 2024-09-19; Apache-2.0 | A public model described as streaming LID. | The card helper returns one final `top_lang`; no reference-event matching, precondition outcome, or switch ledger. |
| [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization/tree/f667ed73aee57d40cc39428eb768b4fd87a0a29e) | revision `f667ed73...`; released 2026-09-23, last modified 2026-09-24; OpenMDW-1.1 | The card binds DER to exact reference annotations and separates input-buffer latency from compute/RTF. | Speaker rather than language events; no source-conditioned boundary matcher or switch lag. |
| [`nvidia/diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1/tree/cd03eee90fbec18297ac31b8c21546e596b7f71c) | revision `cd03eee9...`; last modified 2026-09-23; NVIDIA Open Model License on card | Streaming speaker states use arrival-order identity. | Speaker labels, not LID; no reusable language-switch scoring contract. |

The Nemotron card's insistence on exact reference labels is directly relevant:
changing the boundary annotation changes the metric. Its numerical DER and GPU
throughput do not transfer to this Hindi-English CPU LID task.

**Unverified search conclusion:** no reviewed Hub LID card found in this audit
publishes a chronology-safe, source-conditioned, one-to-one switch-event
matcher with semantic, availability, emission, and route clocks.

## 9. Concrete next steps

1. **Implement one pure, versioned run/event scorer and its adversarial tests.**
   Put reference normalization, maximal-run extraction, source-precondition
   evaluation, monotone matching, and aggregation behind one API used by every
   caller. Expected model gain: **none**. Acceptance is the 12 fixtures above,
   integer-sample outputs, one-use `run_id`s, and no pair-specific language
   branch.
2. **Rescore cached teacher traces before rerunning inference.** Publish a new
   analysis schema beside, not over, the historical artifacts. Expected
   read-only outcome (**unverified until official publication**): current
   window 2/2, bounded-750 1/2, causal-500 1/2, causal-1000 1/2, causal-2000
   2/2, cumulative-prefix 1/2; all anchor-hop and online telephony teacher
   counts remain unchanged. No adopted rejection should reverse.
3. **Migrate teacher, student raw, EMA, and commits together.** Save one
   per-boundary ledger with semantic/latest-audio samples and optional wall
   clocks, then derive all deadline, lag, churn, and DER summaries from it.
   Expected current main outcome: Hindi-to-English remains
   `precondition_failure` with `lag=null`; any finite latency change is
   **unverified**. Do not promote a model or policy unless the scorer version,
   reference-boundary identity, detected `n/N`, and false-event/wrong-time
   metrics are all frozen.

## Scope and unverified points

- The exact monotone objective and the decision to allow an UNKNOWN/wrong-state
  bridge before eventual target acquisition are proposed project semantics,
  not an accepted streaming-LID standard. Version them before comparing runs.
- The source qualification horizon should be explicit per trajectory. A raw
  500 ms diagnostic and a router's already-dwelled committed state are not the
  same object.
- The cached trace rescore is deterministic evidence about these files, not a
  new experiment run or a population estimate.
- The two switch fixtures reverse one source pair and use an unverified nominal
  join. They cannot establish production recall or lag.
- Wall-clock emission, queueing, and ASR route acceptance are still unmeasured.
- External sound-event, keyword-search, and speaker-diarization conventions
  support one-to-one matching and clock discipline; their scores do not predict
  Hindi-English LID performance.

## Sources and dates

- NIST, [*KWS16 Evaluation Plan v04*](https://www.nist.gov/system/files/documents/itl/iad/mig/KWS16-evalplan-v04-ComparedTo-KWS15v05.pdf), **2016-02-08**; one-to-one temporal occurrence alignment; accessed 2026-09-26.
- Mesaros, Heittola, and Virtanen, [*Metrics for Polyphonic Sound Event Detection*](https://doi.org/10.3390/app6060162), published **2016-05-25**; accessed 2026-09-26.
- `sed_eval`, [`EventBasedMetrics` documentation](https://tut-arg.github.io/sed_eval/generated/sed_eval.sound_event.EventBasedMetrics.html), optimal versus order-sensitive greedy matching; accessed **2026-09-26**.
- Baghel et al., [*The DISPLACE Challenge 2023*](https://www.isca-archive.org/interspeech_2023/baghel23_interspeech.html), **Interspeech 2023-08**; language DER with overlap and no collar; accessed 2026-09-26.
- Hugging Face cards and API metadata linked in Section 8; revisions and modification dates checked **2026-09-26**.

