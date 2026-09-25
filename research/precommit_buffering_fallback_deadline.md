# Pre-commit buffering and fallback deadlines for streaming LID -> ASR

Research cutoff: **2026-09-26**. Web sources and Hugging Face cards/API metadata were checked on that date. Publication, release, and artifact dates are stated beside the evidence.

## Bottom line

The current design's **2.5 s ring buffer does not make an unbounded `UNKNOWN` state safe**. It says that audio remains buffered and that a multilingual recognizer *may* run in shadow, while also promising that the selected recognizer can replay from VAD onset. If no recognizer consumes the audio and LID stays uncertain for more than the retained history, the onset has already been overwritten. The submitted run demonstrates that the no-commit branch is reachable: its initial Hindi route is never acquired.

The production contract should separate two deadlines:

1. **ASR-attach deadline:** by this time, at least one recognizer has accepted every speech sample, including validated pre-roll. The safest default is zero additional delay: start an auto/multilingual fallback at VAD onset and feed it the pre-roll plus the live stream.
2. **Route deadline:** by this later time, either promote a language-specific route or expose the fallback transcript instead of waiting indefinitely for LID. LID may keep running and can still trigger a timestamped overlap handoff.

This yields one simple invariant:

> `UNKNOWN` may mean “language not committed”; it must never mean “no ASR owns the audio.”

If cost rules out a shadow recognizer, the fallback must cold-start at a hard deadline derived from measured queue/start-up latency and the ring length. It must not be an arbitrary confidence timeout:

```text
P_pre_roll,p99 + D_attach + S_start,p99 + Q_p99 + safety <= B_ring

D_attach <= B_ring - (P_pre_roll,p99 + S_start,p99 + Q_p99 + safety)
```

Here `P_pre_roll` covers late VAD onset, `D_attach` is time from detected onset until fallback dispatch, `S_start` is model/session start time, `Q` is queue/IPC time until the recognizer accepts replay, and `B_ring` is retained waveform history. If the right-hand side is non-positive, cold-start-on-uncertainty is not a valid architecture; keep a recognizer warm or enlarge/pin the buffer.

For the headline Hindi-English **within-utterance** switch case, do not automatically kill the only code-switch-capable fallback after the initial LID commit. Chandak et al. explicitly disabled early termination for pairs where code-switching was common. A specialized monolingual ASR may become primary, but the fallback should remain warm for the utterance or be re-armed under a measured policy.

No reviewed paper or model card supplies a universal 1.0, 1.5, or 2.5 s deadline. The numerical route deadline must be selected on representative calls from correct/wrong/no-commit curves, first-usable-transcript latency, downstream cost, and measured ASR start/catch-up time. Any proposed value below is a test grid, not a published standard.

## Repository audit

The current pipeline design says:

- a timestamped **2.5 s ring buffer** feeds VAD, LID, and ASR;
- before commit, audio remains buffered and a multilingual recognizer **may** run in shadow;
- on commit, buffered audio is sent **from VAD onset** to the selected ASR;
- when confidence never passes, it routes to multilingual ASR and **keeps buffering**;
- an ASR switch replays about **0.8-1.5 s** of overlap.

These statements are in [`DESIGN.md`](../DESIGN.md), but the ring, ASR router, and fallback are intentionally not implemented in `src/`, `scripts/`, or `tests/`. A repository search on 2026-09-26 found only the student's finite feature-history buffer, not a waveform ring or ASR-attach deadline. Therefore this memo proposes an implementation contract; it does not claim a working router.

### Concrete failure trace

Assume a 2.5 s circular waveform buffer and no shadow ASR:

1. VAD declares onset at audio time 0.
2. The LID threshold/margin/dwell never passes.
3. At 2.6 s the oldest 0.1 s has been evicted.
4. The fallback is finally requested.
5. “Replay from VAD onset” is now impossible, even with infinite ASR speed.

The submitted evaluation already reaches step 2. A fixed ring without a deadline merely turns uncertainty into deterministic first-word loss.

### The buffer is cheap; the missing contract is the issue

For uncompressed mono audio:

| History | PCM16, 16 kHz | G.711 payload, 8 kHz |
|---:|---:|---:|
| 1.0 s | 32,000 B / 31.25 KiB | 8,000 B / 7.81 KiB |
| 1.5 s | 48,000 B / 46.88 KiB | 12,000 B / 11.72 KiB |
| 2.5 s | 80,000 B / 78.12 KiB | 20,000 B / 19.53 KiB |

This arithmetic excludes container, allocator, metadata, encryption, replication, and recognizer state. It shows only that the raw waveform ring is not the dominant memory cost relative to a 600M-parameter ASR. Privacy/retention policy and concurrent-call scale still matter.

The documented LID model bound is 435 ms. Once the first chunk passes the policy, a three-chunk dwell needs two further 160 ms intervals, so a rough strong-evidence scheduling bound is about `435 + 2*160 = 755 ms`, before VAD, compute, queues, or ASR startup. This is project arithmetic, **not a measured commit SLO**. Low confidence can extend it forever under the current policy.

## What prior work supports

| Evidence | Date | Relevant result | Implication and limit |
|---|---:|---|---|
| [Chandak et al., *Streaming Language Identification using Combination of Acoustic Representations and ASR Hypotheses*](https://arxiv.org/html/2006.00703v1) | arXiv **2020-06-01** | The runtime starts competing monolingual ASRs in parallel, sends partial hypotheses to LID every 600 ms, and stops a nonmatching recognizer only after a development-tuned confidence threshold. At end of input without a crossing, it uses the final maximum. The authors do **not** apply early stopping to pairs expected to code-switch often. | Direct support for “consume audio first, narrow later” and for a hard terminal fallback. Its labels are utterance-level, its systems/data are private, and its Hindi-English examples do not validate frame-level switch routing. |
| [Punjabi et al., *Joint ASR and Language Identification Using RNN-T*](https://www.amazon.science/publications/joint-asr-and-language-identification-using-rnn-t-an-efficent-approach-to-dynamic-language-switching) | ICASSP **2021-06** | For English-Hindi dynamic language selection, a joint streaming RNN-T reports 6.4-9.2% relative WER reduction, 53.9-56.1% relative LID error reduction, and up to 46% lower memory than parallel monolingual ASRs. | A joint multilingual ASR-LID is a credible long-term alternative to parallel decoders. The paper distinguishes interaction-to-interaction switching from within-utterance code-switching, so its numbers are not evidence for natural Hinglish boundaries. No public checkpoint was found in this audit. |
| [Zhang et al., *Streaming End-to-End Multilingual Speech Recognition with Joint Language Identification*](https://www.isca-archive.org/interspeech_2022/zhang22da_interspeech.pdf) | Interspeech **2022-09** | A zero-right-context first pass produces low-latency ASR while a 0.9 s right-context second pass and lightweight frame-synchronous LID improve the final path; the LID head adds 0.5% parameters and reaches 96.2% average seven-language accuracy. | Strong precedent for **not blocking first-pass transcription on LID**. Its private voice-search task and language inventory do not establish Hindi-English switch performance or a fallback deadline for this project. |
| [NVIDIA NeMo cache-aware streaming documentation](https://docs.nvidia.com/nemo-framework/user-guide/26.02/nemotoolkit/asr/models.html) | documentation checked **2026-09-26** | Cache-aware Conformers process new chunks while preserving bounded model state; the documentation distinguishes algorithmic latency from compute and warns that offline checkpoints used with small chunks can lose accuracy. | Supports measuring start/queue/step latency separately and avoiding repeated full-buffer inference. It does not supply this service's cold-start or catch-up distribution. |

The 2020 early-stop result is often summarized as identifying more than half of utterances about 1.5 s before their end. That is **not** “commit by 1.5 s after onset”: it is relative to each utterance's end and should not be reused as this project's timeout.

## Recommended router contract

### Required clocks and acknowledgements

Every waveform chunk should have a monotonically increasing sample interval and these events:

```text
t_ingress(sample)
t_vad_onset_detected
t_asr_dispatch
t_asr_accept(first_sample, last_sample)   # acknowledgement, not request time
t_asr_first_partial
t_lid_commit
t_route_publish
t_asr_caught_up
```

The buffer may release a sample only after every consumer that still owns it has acknowledged it, or after the state machine has explicitly abandoned that consumer. A single circular write pointer is insufficient when a slow/cold ASR is replaying history; use per-consumer cursors, a pinned immutable segment, or copy the bounded replay payload before allowing eviction.

Measure the following separately:

- VAD onset error and selected pre-roll;
- ASR dispatch-to-accept p50/p95/p99;
- first-partial and stable-partial latency;
- replay catch-up time and whether live chunks queued behind replay;
- time to language commit and time to published route;
- ring high-water mark, overwritten-needed-sample count, and deadline misses;
- ASR compute-seconds and peak concurrent recognizers per voiced minute.

If a recognizer processes audio at measured replay RTF `r < 1`, an idealized backlog `A` seconds drains in `A*r/(1-r)` seconds while live audio continues. This is a derived lower-bound model, not a service guarantee: batching, throttling, decoder state, and network flow control can dominate. If `r >= 1`, the recognizer cannot catch up without dropping or pausing live input.

### State machine

```text
PRE_SPEECH
  retain validated pre-roll
       |
       | VAD onset
       v
UNKNOWN_SHADOW
  fallback owns pre-roll + every live chunk; LID runs concurrently
       |                              |
       | stable language commit       | route deadline, still UNKNOWN
       v                              v
SPECIALIST_HANDOFF               FALLBACK_COMMITTED
  replay bounded overlap;          publish fallback partials;
  merge by timestamps             continue LID in background
       |                              |
       +-------------+----------------+
                     |
                     | low fallback confidence / no usable transcript
                     v
                  REPROMPT
```

Recommended semantics:

1. **Start fallback ingestion at VAD onset.** Include pre-roll sized from the measured VAD onset-error distribution, not merely from a guessed 30 ms pad.
2. **Select a route deadline on development calls.** Sweep, for example, `{0.8, 1.2, 1.6, 2.0}` s after detected speech onset, but freeze the value before test. This grid is a project proposal. At the deadline, `UNKNOWN` becomes a fallback route, not a forced language label.
3. **Do not discard the fallback prefix.** If LID commits after the ring no longer contains onset, the fallback supplies stable earlier words; a specialist receives only the bounded recent overlap. Documentation must stop promising specialist replay from onset in this branch.
4. **Keep irreversible bot actions behind a transcript/route stability gate.** Publishing a fallback ASR partial is not the same as committing an NLU action.
5. **For Hindi-English code-switch traffic, keep the auto/bilingual path warm** until end of utterance unless a validated re-arm policy matches its WER and latency. The first language commit should not destroy the only recognizer able to follow a later switch.
6. **Use a second task deadline for reprompt.** If the fallback has no usable partial or is also low confidence/OOS, ask a language-choice question or transfer; do not extend the waveform wait indefinitely.

### Cost-aware cold-start variant

If a shadow ASR is too expensive:

1. keep the fallback process/model warm if possible, but do not create decoder state yet;
2. dispatch no later than the bound derived from `B_ring` and measured p99 terms;
3. atomically pin/copy `[VAD onset - pre-roll, now]` before returning from dispatch;
4. feed replay faster than real time only if measured service flow control permits it;
5. join the live stream by sample timestamp, never by arrival order;
6. fall back earlier when queue depth makes the inequality fail.

The attach deadline should be adaptive to queue/startup telemetry but fail **earlier**, never later, under load. A service that waits longer when overloaded compounds both buffer overrun and catch-up delay.

## Hugging Face Hub audit: fallback ASR candidates

This was a dated, non-exhaustive audit on **2026-09-26**. An RNNT label or a “streaming” tag is not by itself proof of bounded startup, live chunk semantics, code-switch WER, or telephone robustness.

| Artifact | Verified Hub facts | Fit for this router | Unverified / blocker |
|---|---|---|---|
| [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) ([API](https://huggingface.co/api/models/nvidia/nemotron-3.5-asr-streaming-0.6b)) | Official; released **2026-06-04**, API last-modified **2026-09-10**; pinned revision `ea30d66debe3740a08b573244286791d423d6b3e`; 600M-parameter cache-aware FastConformer-RNNT; chunks 80/160/320/560/1120 ms; `auto` language mode; OpenMDW-1.1. Hindi and English are both “transcription-ready.” On monolingual FLEURS at 320 ms, Hindi WER is 7.41 with supplied LangID versus 9.88 in auto mode; English is 8.27 versus 8.84. | **Best public Hindi-English shadow candidate found**, because it can consume one stream without waiting for external LID and later shows why a correct explicit language route can still help. | FLEURS is not Hinglish, 8 kHz telephony, or live-call evidence. Auto mode emits an utterance language tag; the card does not report word-level language switches. CPU RTF, service cold start, queue p99, first-partial latency, and prompt-switching within one decoder state are unreported. Official support covers only Hindi and English among the project's seven. The pinned [`processor_config.json`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/processor_config.json) contains prompt IDs for several other Indic languages, but the card's trained/out-of-box list does not; an ID is not a quality claim. |
| [`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) ([API](https://huggingface.co/api/models/ai4bharat/indic-conformer-600m-multilingual)) | Official AI4Bharat; API revision `e9b71b369c048e2c6b634d4c131061c34e441179`, last-modified **2026-02-07**; MIT; gated by automatic contact acceptance; 600M multilingual hybrid CTC/RNNT; supports all six selected Indian languages (`hi,bn,gu,mr,ta,te`) among IN-22. | Possible specialist ASR after LID for the six Indic routes. | English is absent, and the example requires a caller-supplied language code. The card demonstrates full-waveform inference and gives no cache/chunk/first-partial contract, so it is **not verified as a live fallback** despite the RNNT decoder. Only one card-level Hindi Vaani result is exposed; cross-language, code-switch, CPU, and 8 kHz performance are unverified. |
| [`ai4bharat/indic-asr-nemotron-600m`](https://huggingface.co/ai4bharat/indic-asr-nemotron-600m) | Search metadata cached on 2026-09-26 describes a roughly 600M hybrid RNNT/CTC model for 26 Indian languages with cache-aware streaming. | Potential future all-Indic shadow. | Direct card and API access returned **401** during this audit. Cached text calls it an unreleased, case-by-case, research-only model. Exact revision, language list including English, license, metrics, and artifacts are therefore **unverified**; do not put it on a production path. |
| [`smajji/nemotron-hinglish-v4`](https://huggingface.co/smajji/nemotron-hinglish-v4) ([API](https://huggingface.co/api/models/smajji/nemotron-hinglish-v4)) | Community fine-tune created **2026-09-17**, revision `19d5c968adc7dd1eebf744909d4c5ca00bb8819f`; 600M cache-aware base; English/Hindi/Hinglish; card reports 4.0/12.0/29.2% WER on an unspecified held-out sample. | Relevant challenger for the headline pair only. | No sample counts, split hashes, baseline, per-domain results, 8 kHz data, live latency, or reproducible evaluation are given. The child card says Apache-2.0 while the declared base is OpenMDW-1.1; applicable derivative terms are **unresolved**. Treat every quality and deployment claim as unverified until the artifact, data rights, and evaluation are audited. |

The official NVIDIA checkpoint is not a recommendation to add a 600M GPU dependency to this CPU take-home. It is a concrete **reference implementation and offline routing experiment**. The main design should remain model-agnostic and require the timing acknowledgements above from any local or remote ASR.

## Testable implementation plan

### 1. Prove the buffer invariant with a fake clock

Implement a small router/ring simulator before integrating an ASR. For every needed sample, assert either:

```text
fallback_accepted(sample) OR pinned_for_named_consumer(sample)
```

before eviction. Required fixtures:

1. LID remains `UNKNOWN` for 10 s: the shadow path receives sample zero plus pre-roll, the route falls back exactly at the configured deadline, and the ring never overwrites an unacknowledged sample.
2. Cold path with `B=2.5 s`: a synthetic `pre-roll + deadline + startup + queue + safety == 2.5 s` passes; adding one sample fails closed or dispatches one sample earlier.
3. LID commits before the deadline: specialist overlap has continuous, duplicate-free sample timestamps and the fallback prefix remains available to the transcript merger.
4. LID commits after onset has left the ring: the router never claims specialist replay from onset; it keeps fallback stable words and replays only the declared overlap.
5. A silence gap freezes LID policy but does not drop or duplicate the ASR clock.
6. Hindi -> English after an initial Hindi commit: the configured code-switch-capable fallback remains alive or is re-armed early enough to consume the declared overlap.

Expected gain: **no model-accuracy claim**. The acceptance result is zero overwritten-needed samples and zero hidden onset gaps across deterministic schedules.

### 2. Measure shadow versus cold fallback on target infrastructure

Replay the same timestamped calls through:

- `shadow_from_vad`;
- `cold_at_deadline` for `{0.8,1.2,1.6,2.0}` s;
- `language_specialist_only` as an oracle-cost reference;
- and, for Hindi-English only, two parallel specialists if available.

Inject controlled ASR queue/startup distributions as well as real load. Report first-word deletion rate, WER/task success, dispatch-to-accept and first-usable-partial p50/p95/p99, catch-up p95, ring overruns, wrong-ASR seconds, ASR compute-seconds per voiced minute, and peak decoder concurrency. Expected direction (**unverified**): shadowing eliminates onset loss and lowers first-partial latency but costs more compute; a warm cold-at-deadline path may recover most of the latency benefit if the p99 inequality has margin. Select on a cost/latency/quality Pareto frontier, not LID accuracy alone.

### 3. Evaluate the official Hindi-English auto-ASR as a fallback reference

Pin NVIDIA revision `ea30d66debe3740a08b573244286791d423d6b3e` and compare `auto` shadowing with explicit `hi-IN`/`en-US` specialists at 160 and 320 ms. Use held-out FLEURS for monolingual controls, HiACC/IITB-timed data for natural Hinglish, and the pinned IndicTelephony benchmark for 8 kHz calls; do not tune on their test splits. Report code-switch WER by Hindi/English token, deletions in the first second, first partial, stable partial, GPU memory/concurrency, and LID-to-specialist handoff discontinuities. Expected result (**unverified**): auto mode protects no-commit calls but trails a correct explicit route, especially for Hindi, while a late handoff can recover later words without recovering a lost prefix. It is a reference only unless real-call and license review pass.

### 4. Test whether a Hinglish fallback must remain warm after initial commit

On natural sustained switches, compare:

- terminate fallback at initial monolingual commit;
- retain fallback until end of utterance;
- re-arm on raw challenger posterior before committed switch.

Freeze the transcript merge rule and charge all compute. Expected direction (**unverified**): retaining the fallback improves English-span recall and reduces re-decode lag, at the cost of roughly one extra active decoder during the utterance. Adopt a cheaper re-arm policy only if code-switch WER is within 1 absolute point, switch recall@1 s within 2 points, and p95 usable-transcript lag within 100 ms of always-warm fallback, with no first-word regression.

## Scope limits and explicitly unverified points

- There is no implemented ASR router in this repository; no fallback timing or WER result is claimed.
- The proposed deadline grid, buffer inequality safety term, state names, and acceptance gates are project proposals, not published standards.
- No reviewed source supplies this service's model-load, session-start, queue, replay-flow-control, catch-up, or first-partial p99. They must be measured under target concurrency.
- Chandak et al.'s “1.5 s early” result is relative to utterance end and does not choose an onset deadline.
- NVIDIA's FLEURS WER table is monolingual and cannot establish within-utterance Hinglish, 8 kHz, or noisy-call quality.
- A Hugging Face `streaming`, RNNT, or language-prompt tag is not proof that the public API preserves state across chunks or can change prompts midstream.
- Hub metadata can change. Revisions above were resolved on 2026-09-26 and should be pinned before an experiment.
- The Hub audit is dated and non-exhaustive; “best public candidate found” means among the reviewed artifacts, not a proof that no other deployment exists.

## Sources and dates

- Chandak et al., [*Streaming Language Identification using Combination of Acoustic Representations and ASR Hypotheses*](https://arxiv.org/abs/2006.00703), submitted **2020-06-01**; HTML/runtime and early-stop tables checked **2026-09-26**.
- Punjabi et al., [*Joint ASR and Language Identification Using RNN-T: An Efficient Approach to Dynamic Language Switching*](https://www.amazon.science/publications/joint-asr-and-language-identification-using-rnn-t-an-efficent-approach-to-dynamic-language-switching), ICASSP **2021-06**; checked **2026-09-26**.
- Zhang et al., [ISCA Interspeech 2022 paper](https://www.isca-archive.org/interspeech_2022/zhang22da_interspeech.html), conference **2022-09-18 to 2022-09-22**; checked **2026-09-26**.
- NVIDIA, [NeMo cache-aware streaming model documentation](https://docs.nvidia.com/nemo-framework/user-guide/26.02/nemotoolkit/asr/models.html), release 26.02 documentation; checked **2026-09-26**.
- NVIDIA, [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) and [Hub API metadata](https://huggingface.co/api/models/nvidia/nemotron-3.5-asr-streaming-0.6b), released **2026-06-04**, revision resolved/API last-modified **2026-09-10**; checked **2026-09-26**.
- AI4Bharat, [`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) and [Hub API metadata](https://huggingface.co/api/models/ai4bharat/indic-conformer-600m-multilingual), repository created **2025-03-15**, audited revision last-modified **2026-02-07**; checked **2026-09-26**.
- AI4Bharat, [`ai4bharat/indic-asr-nemotron-600m`](https://huggingface.co/ai4bharat/indic-asr-nemotron-600m), search metadata checked and direct access blocked **2026-09-26**.
- Community artifact, [`smajji/nemotron-hinglish-v4`](https://huggingface.co/smajji/nemotron-hinglish-v4) and [Hub API metadata](https://huggingface.co/api/models/smajji/nemotron-hinglish-v4), created/last-modified **2026-09-17**; checked **2026-09-26**.
