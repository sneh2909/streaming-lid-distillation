# Part 2: Streaming LID in front of ASR

All numbers come from `results/` and are produced by the scripts named next to them. The switches are synthetic concatenations of held-out clips.

## 1. Where it sits

```mermaid
flowchart LR
  A[Caller leg<br/>8 kHz mu-law, 20 ms frames] --> B[Resample 16 kHz<br/>+ ring buffer, 3 s pre-roll]
  B --> V[VAD]
  V -- speech frames --> L[Streaming LID student<br/>80 ms frames, chunk 320 ms]
  L -- posteriors p_t --> C[Commit / hysteresis FSM]
  C -- COMMIT or SWITCH at pause --> R[ASR router]
  C -- UNCERTAIN --> R
  B --> R
  R --> ASR1[ASR hi / Hinglish]
  R --> ASR2[ASR en-IN]
  R --> ASRm[Multilingual fallback]
  ASR1 & ASR2 & ASRm --> NLU[LLM / NLU] --> TTS[TTS in committed language]
  B -. audio + LID trace .-> S[(Call store)]
  S -. nightly .-> T[Offline teacher relabel] -. mined hard cases .-> TR[Retrain student]
```

The LID runs only on the caller's inbound track (never the bot's own TTS or echo), and only on VAD speech frames, so silence cannot move the decision.

**Turn-level flow (the design we'd ship):**

```
audio ─► Silero VAD ─speech─► LID student (streaming, fresh state per turn)
                │                  ├─ EARLY ROUTE: first commit (θ) → start streaming the turn to that ASR
                │                  └─ (keeps updating while the caller speaks)
                └─ end of turn (≥300 ms silence) ─► FINAL LID = student posterior at the endpoint
                                                     ├─ final == early → transcript already done
                                                     └─ final ≠ early  → re-decode the buffered turn
                                                                          with the right ASR
```

- **The bot acts on a turn only at the endpoint.** So the number that decides what it *says* is the **final LID at the endpoint**. The early route only decides which ASR produces streaming partials; a wrong early route costs one re-decode of a ≤10 s turn (~100–300 ms), not a wrong reply.
- **Mid-turn switch lag** then affects only partial transcripts. Between turns, the fresh stream per turn turns every switch into a first decision.
- **Short turns** (< ~1 s, e.g. "haan", "ok") rarely reach θ: keep the previous turn's language.
- **At the endpoint you can afford a bigger model once per turn** (Indic-Transcribe batched runs ~20 ms per clip on a GPU) as a verifier. That's optional; we have not measured it end to end.

**Measured** with `scripts/turn_eval.py` (Silero VAD turns on the 90 switch clips → 200 turns, fresh student per turn, θ 0.8; `results/final/turn_eval.json`):

| | value |
|---|---|
| **final LID at the endpoint** | **94.5%** of turns correct (94.7% on turns ≥ 1.5 s) |
| early route issued | on 85% of turns, median **1.23 s** after the turn starts |
| early route correct | 94% |
| re-decode needed (early ≠ final) | 3% of turns |

The old model (trained before the data fix) scored 72% final LID and routed early on only 52% of turns.

## 2. Commit and calibration policy

The student emits a posterior every 80 ms frame. Raw argmax is jittery, so routing reads a small state machine (`slid/commit.py`, tested in `tests/test_commit.py`):

- **Smoothing:** s_t = (1 − α) s_{t−1} + α p_t, with α = 0.3 (≈ 0.25 s time constant).
- **First commit:** when max_k s_tk ≥ θ_commit (0.8, chosen from the sweep below; the code default and the README evaluation tables use 0.7). Until then the call is *uncommitted* and routes to the prior (section 5).
- **Switch:** a different language must reach θ_switch = θ_commit + 0.1 for `dwell` consecutive frames (3 = 240 ms), and the router only acts on it at the next pause or endpoint.

**Why these three knobs:**
- A threshold alone flip-flops near 0.5.
- A margin (top-1 minus top-2) is equivalent for 2 classes, but less interpretable with 7.
- Dwell time is what actually suppresses one-frame blips.
- θ_switch > θ_commit is the hysteresis: once committed, it takes stronger evidence to leave than it took to enter.

**How to set the operating point.** This is a cost trade-off, not a number to guess:
- **Cost of a wrong commit:** the ASR decodes the turn in the wrong language, so the transcript is garbage. The bot either answers nonsense or re-prompts (several seconds plus user frustration).
- **Cost of waiting:** each extra 100 ms before commit is 100 ms of audio buffered and decoded late.

The procedure, on a labelled dev set of real calls:
1. Sweep (θ_commit, dwell).
2. Plot time-to-first-correct-commit against wrong-first-commit rate.
3. Pick the knee under a product constraint, e.g. wrong-first-commit ≤ 2%, then minimise the median commit time.

The student is trained on information-matched targets, so its early posteriors are honestly uncertain, and a probability threshold means what it says. A student distilled from future-informed targets can be *confident for the wrong reason*: our centred-target student names the language from pure silence 64% of the time (README §3). A threshold would happily pass that shortcut confidence, and on real phone lines it would not hold.

**Measured** (final student, 320 ms chunks, 480 held-out speech-only clips, `scripts/commit_sweep.py` → `results/final/commit_sweep_final.json`; dwell 3 frames, θ_switch = θ_commit + 0.1):

| θ_commit | median first *correct* commit | wrong first commit | hi↔en committed switch lag | switches missed | flips/min |
|---|---|---|---|---|---|
| 0.5 | 0.98 s | 32% | 1.84 s | 30% | 2.0 |
| 0.6 | 1.06 s | 19% | 1.90 s | 33% | 1.1 |
| 0.7 | 1.23 s | 15% | 1.99 s | 38% | 0.4 |
| **0.8** | **1.47 s** | **11%** | 2.24 s | 40% | 0.0 |
| 0.9 | 1.78 s | 7.5% | 2.48 s | 40% | 0.0 |

![commit trade-off](results/figures/commit_tradeoff.png)

**Reading the sweep:**
- **Each 0.1 of θ costs ~0.1–0.3 s and roughly halves-to-thirds the wrong first commits.**
- **Operating point: θ_commit = 0.8.** 1.5 s to a first route, 11% of first routes wrong.
- **Why 11% wrong is acceptable here:** in the turn pipeline (§1) the first route only chooses which ASR streams partials. The **final LID at the endpoint is right on 95% of turns**, and only 3% of turns need a re-decode (`results/final/turn_eval.json`). A product that acts *before* the endpoint (barge-in, very early intent) should take θ = 0.9.
- **The cost of a high threshold is switch recall, not first-commit latency.** This is why the switch threshold and the first-commit threshold are separate knobs.
- **Tune on real calls, not synthetic data.** In production this sweep runs on a labelled dev set of real calls; the numbers here come from synthetic concatenations.

## 3. ASR routing and the cost of switching

**Routing:** the committed language picks the ASR session: Hindi/Hinglish, English (en-IN), or a multilingual fallback.

**Before the first commit:**
- Audio is buffered in a 3 s ring buffer, so nothing is lost.
- It is streamed to the **prior's** ASR (section 5).

**After the commit:**
- **Prior was right (the common case):** nothing to redo.
- **Prior was wrong:** open the correct session, replay the ring buffer into it, and discard the wrong partials.
- **Cost of a late commit:** the replay (≤ 3 s of audio at >10× real time, so ~200–300 ms) plus lost partial transcripts.
- **Cost of a wrong early commit:** the same replay once the error is detected, plus whatever the bot already said based on the wrong transcript. That is why the first-commit threshold is conservative, and why the bot does not *act* on a turn until the turn's language is committed.

**Mid-call switch:**
- **Two sessions:** flip only at a pause or endpoint, never mid-word. The old session finalises its turn; the new session is warmed and gets the pre-roll since the switch evidence began (dwell + smoothing ≈ 0.5 s), so the first words in the new language aren't lost.
- **Language-prompted ASR:** with a model like NVIDIA Nemotron 3.5 ASR (a cache-aware FastConformer), the language is a one-hot prompt concatenated after the encoder. A switch is then just a new prompt for the next chunk: no reconnect, no encoder-cache reset, no replay.
  - Its card reports FLEURS WER with the language supplied (e.g. Hindi 6.81 at 1.12 s chunks) alongside an auto-detect mode. A dedicated streaming LID is what lets us supply it. (We did not measure the gap ourselves.)

**Route on the matrix language, not per word.** In Hinglish, English nouns sit inside Hindi grammar. Switching ASR on every English word would fragment the transcript. Mixed turns go to the Hindi/Hinglish ASR, which is trained on code-mixed speech. A switch to the English ASR happens only when the *sentence frame* changes, which is why the switch rule needs sustained evidence (dwell) and not one frame.

## 4. Code-switching: following without flip-flopping

- **Hysteresis:** the θ_switch > θ_commit gap plus dwell (above).
- **Forgetting old evidence:**
  - The EMA forgets old frames.
  - The student is distilled from a **causal 3 s window** teacher. Its target at time t reflects only the last 3 s, so it is trained to let go of the previous language.
  - A pure "everything so far" teacher cannot switch until the new language outweighs the whole history. Our comparison shows this: a prefix-target student misses 69/90 committed switches, against 41/90 for causal (README §3).
- **Measuring switch-detection lag:**
  - **Clips:** concatenated clips with a known switch time t_s (`data/manifests/switch.jsonl`, 9 language pairs including Hindi↔Indian-English).
  - **Lag:** time from t_s until the *committed* label becomes the new language **and stays there**, so a flicker doesn't count.
  - **Reported separately for:**
    1. the teacher's own targets
    2. the student's raw argmax
    3. the committed label

    This separates teacher lag, student lag and policy lag.
  - **Also reported:** misses (never switched) and extra label changes per minute (flip-flops).
  - **On real data:** replace synthetic boundaries with word-level language tags from MUCS 2021 Hindi-English (script-based alignment), or DISPLACE language-diarization labels.

**Measured** (Hindi→English, 10 held-out clips, 320 ms chunks, medians, no per-turn reset):

| | switch lag |
|---|---|
| teacher target | 1.90 s |
| student raw | 1.56 s |
| committed | 2.28 s |

- **Flip-flops:** 18.5/min raw vs **0/min committed**.
- **Most of the lag is the 3 s teacher window,** not the student and not the policy. We tested both levers.

**Lever 1: shorter teacher window W** (original data; same ensemble teacher, streaming-v2 student, 320 ms chunks):

| W | teacher lag | student raw lag | committed lag | missed switches | premature switches | acc end (1-language) | telephony end |
|---|---|---|---|---|---|---|---|
| 3 s | 2.34 s | 2.07 s | 2.62 s | 20/90 | 2% | **.87** | **.80** |
| 1.5 s | **1.07 s** | **1.14 s** | **1.91 s** | **12/90** | 9% | .71 | .66 |

Halving W halves the lag, but targets from 1.5 s of audio are noisy. The student learns to be jumpy: late-clip accuracy drops 16 points and premature switches rise to 9%. So W trades switch speed against stability, and **3 s is the better default**.

**Lever 2: reset at turn boundaries.** This is where real switches mostly happen: a caller changes language between sentences, not mid-word. The cleaned switch clips are sentence + 0.25–0.6 s pause + sentence. We detected the pause with a 20-line energy VAD (320 ms below −35 dB; found in 82/90 clips) and compared (final student, θ 0.8, `scripts/turn_reset.py` → `results/final/turn_reset_final.json`):

| at a detected pause | committed lag | missed | flips/min | premature |
|---|---|---|---|---|
| nothing | 2.69 s | 29/90 | 0.1 | 0% |
| reset the commit smoothing | 2.69 s | 29/90 | 0.2 | 0% |
| **reset smoothing + start a fresh student stream** (keep routing the old language until the new turn commits) | **0.95 s** | **16/90** | 0.1 | 0% |

- **A fresh stream per turn turns every switch into a first decision,** so the switch lag becomes the first-commit time: under a second.
- **The centred-target student** (original data) was even faster with the same reset (~1.0–1.25 s vs 2.08 s then), but it also names the language from pure silence 64% of the time. After the data fix the causal student gets below a second honestly.
- **This is the turn-level design of §1.** It keeps a stable long-window student *within* a turn and gets fresh decisions *between* turns.

- **Weakest pair:** Hindi→Indian-English within a clip (9/10 missed without reset), inherited from the teacher's Indian-English accuracy (.72).

## 5. Fallback and priors

- **Prior before the first commit**, in this order:
  1. The caller's last-call language (CRM, keyed by phone number).
  2. The campaign or number's configured language.
  3. The circle or region prior from the phone number's state (e.g. Maharashtra → mr/hi/en).
  4. Default Hindi.

  The prior can also enter the math as a log-prior added to the smoothed log-posterior (Bayes). Then a confident student overrides it and an uncertain one does not.
- **Low confidence for too long** (no commit after ~3 s of speech):
  - Stream to **both** likely ASRs for that turn and pick at the endpoint by LID + ASR confidence. That costs 2× ASR compute for a few seconds, which is cheap next to a wrong route.
  - Or use the multilingual fallback ASR.
- **Re-prompt:** "Hindi ya English?" is the last resort, used when both the LID and ASR confidence stay low for a full turn.
- **Pin:** after N switches in a call (e.g. 3), the caller is genuinely code-mixing. Pin to the Hinglish or multilingual ASR for the rest of the call instead of chasing every switch.

## 6. The loop: teacher as labeller again

1. **Log** every call's audio and the student's posterior trace (with consent and retention limits).
2. **Mine** the calls that need labelling:
   - late or no commit
   - wrong-first-commit caught downstream (ASR confidence collapsed, or a replay happened)
   - more than N flips
   - low top-1 margin
   - disagreement between the student's decision and the ASR's language-confidence signals
3. **Relabel offline** with the full-context teacher. The same frozen teacher and the same causal-window target recipe produce frame targets, so the new data drops straight into `scripts/build_targets.py`.
   - Optionally ensemble two teachers.
   - Send a small sample for human verification, to catch cases where the teacher itself is wrong (e.g. Indian English: see the bake-off).
4. **Retrain** the student on the original distillation data plus the mined set (up-weighted). Gate the release on a fixed held-out set so we catch regressions on monolingual clips.
5. **Track drift:** commit latency, wrong-commit rate and flip rate per week, per circle.

The teacher's blind spots bound what this loop can fix. Anything it mislabels is learned by the student, so human spot-checks on mined data are part of the loop, not optional.

## 7. What we would build next (not implemented)

- **Constant memory without the accuracy loss:**
  - The bounded-history student (v2, implemented) costs 7 points.
  - Next: a "time since stream start" input, plus a per-call cache of high-confidence frames per language (Streaming-Sortformer-style), so the model adapts to *this* caller's accented English.
- **Two-timescale student:** a short window for switch detection (W = 1.5 s halves lag) plus a long window for stability (W = 3 s keeps accuracy). This gets both sides of the measured trade-off.
- **Export:** the streaming session (KV cache, chunk size switchable at run time) exists in PyTorch. ONNX/TorchScript export with the cache as explicit inputs/outputs is the remaining step.
- **Real telephony data and real code-switch labels** (MUCS 2021, IndicVoices, DISPLACE) instead of FLEURS + Svarah + synthetic concatenation.
- **Licence review:**
  - Indic-Transcribe's licence is share-alike, requires sign-off for hosting as a service, and explicitly bans use for robocalls and auto-dialers.
  - A voice-bot company must check whether distilling it into a production student is allowed for its use case before shipping.
