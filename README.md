# Streaming language ID by distillation

Distil a frozen, full-context (offline) spoken-language-ID **teacher** into a small **causal, chunked student** that emits a language posterior every 80 ms from the audio heard so far. The target use is a Hindi↔English voice bot in India, plus Gujarati, Marathi, Bengali, Tamil and Telugu.

- **Part 1 (this file):** the implemented distillation.
- **Part 2 ([`DESIGN.md`](DESIGN.md)):** how the streaming LID plugs into ASR.

Everything here was run on one laptop (RTX 4050 6 GB for teachers and training; the required training step also runs on CPU).

## TL;DR

- **Teacher:** chosen by a measured bake-off of 5 offline LID models, then frozen. The winner is a **log-linear ensemble of Indic-Transcribe-core and Whisper-large-v3-turbo**, because their errors are complementary (section 2).
- **The asymmetry fix:**
  - The student at frame t is trained to match the teacher run on **only the last W = 3 s of audio the student has already heard** (a *causal window*).
  - That target is a function of the student's own input, so a perfect student can match it exactly. Nothing is asked of the student that needs the future.
  - It is the same as the "centred window + emit delay Δ = W/2" recipe, written down honestly (section 3).
- **Student:** a 3.1M-parameter causal Conformer with 80 ms frames and chunked attention. Trained with random chunk sizes (80/160/320/640 ms), so one model serves 4 latencies.
  - Runs at ⟨TBD⟩× real time on one CPU thread.
- **Evidence:**
  - one real optimizer step, then N steps, all finite
  - held-out KD loss falls
  - student-teacher agreement on held-out clips
  - a four-way comparison of target types on the same student
  - switch lag split into teacher, student and commit-policy lag
  - causality unit tests

## 1. Data (`scripts/prepare_data.py`)

| Set | Source | Size |
|---|---|---|
| Train, monolingual | FLEURS **dev** split, 7 languages × 200 clips | 4.2 h |
| Train, Indian English | AI4Bharat **Svarah**, 200 clips, speaker keys disjoint from eval | |
| Train, switches | 600 synthetic clips: 2–3 segments of 2.5–5 s; half hi↔en, rest random pairs | 1.4 h |
| Eval, monolingual | FLEURS **test** split, 7 × 60; Svarah 60 (other speaker keys) | 1.3 h |
| Eval, switches | 90 clips: ~4 s A + ~4 s B. Pairs: hi↔en (US), hi↔en-IN, and gu/mr/bn/ta/te→en | |
| Telephony | Every eval clip also scored after 300–3400 Hz band-pass, 8 kHz μ-law, 20 dB SNR noise | |

**Honest caveats:**
- **FLEURS:** train and eval use different sentences, but FLEURS publishes no speaker IDs, so speaker overlap is possible.
- **Svarah:** it has no speaker IDs either. We split on the (native language, state, district, gender, age group) tuple.
- **Switch clips:** they are synthetic concatenations, not real code-switching.

Real code-switched data (MUCS 2021, DISPLACE) is the obvious next step.

## 2. Teacher bake-off (`scripts/teacher_bakeoff.py`, `results/teachers/`)

Choosing the teacher was part of the task, so it is measured. Every teacher's native posterior is folded onto the routing set: Urdu merged into Hindi (they sound alike), and every other language summed into "other". It is then **restricted** (renormalised) to our 7 languages.

Accuracy on held-out clips (restricted), 480 clips:

| Teacher | 1 s | 2 s | 3 s | Full | Full, telephony | Indian English (full) | Hindi @2 s | Marathi | Telugu | ECE @3 s | ms/call |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SpeechBrain VoxLingua107 ECAPA | .31 | .64 | .76 | .87 | .82 | .13 | .70 | .93 | 1.0 | .115 | 6 |
| NVIDIA AmberNet | .35 | .65 | .79 | .87 | .84 | .03 | .77 | .95 | 1.0 | .111 | 12 |
| TalTech XLS-R-300M VoxLingua | .36 | .66 | .82 | .92 | .90 | .38 | .82 | .97 | 1.0 | .061 | 390 |
| Whisper-large-v3-turbo (language token) | .28 | .41 | .49 | .61 | .57 | **1.00** | .73 | .33 | **.00** | .357 | 131 |
| Indic-Transcribe-core (Bodhan/AI4Bharat) | **.43** | **.81** | **.89** | .90 | .90 | .20 | **.95** | 1.0 | 1.0 | .068 | 11 |
| **Ensemble (Indic-T × Whisper)** | ⟨TBD⟩ | | | | | | | | | | |

**Findings that drove the choice:**
1. **Restriction matters on short audio.** At 1 s, 70–96% of every VoxLingua teacher's probability sits on languages outside our 7.
2. **The VoxLingua teachers (ECAPA, AmberNet, XLS-R) cannot recognise Indian-accented English** (3–38%). Their "English" is YouTube English. For an Indian voice bot this is disqualifying.
3. **Indic-Transcribe labels Indian English as the speaker's native language.** Marathi speakers' English comes out Marathi, Nepali speakers' Nepali, Konkani speakers' Konkani. It identifies the accent, not the language.
4. **Whisper is perfect on Hindi, English and Indian English** but misses Marathi, Gujarati, Bengali and Telugu (0–33% on full clips).
5. **So the two best teachers fail on opposite cases.** The ensemble weight *w* is tuned on a slice of the **training** pool by the NLL of the true language: w = ⟨TBD⟩.

**Also not used as teachers:**
- **Nemotron 3.5 ASR:** among our languages it covers only hi and en, and it emits the language tag only after a finished transcript.
- **MMS-LID:** CC-BY-NC licence and ~1B parameters.

**Licence flag:** Indic-Transcribe's licence is share-alike, needs sign-off for hosting as a service, and forbids robocall/auto-dialer use. A voice-bot company must clear this before distilling it into production.

## 3. The asymmetry: what the student is asked to match, and why

**Notation:**
- e(t) is the last input sample that student frame t depends on (`slid/student.py:frame_end_sample`).
- The student's posterior at frame t is p_θ(· | x[0:e(t)]).
- The target is q_t = T(V_t(x)), the frozen teacher applied to a *view* V_t of the clip.

**Objective** (`scripts/train.py:kd_loss`):

    L(θ) = E_x  mean_t  KL( q_t ‖ p_θ(· | x[0:e(t)]) )
         = E_x  mean_t  [ −H(q_t) − Σ_k q_tk log p_θ(k | x[0:e(t)]) ]

Only the cross-entropy term depends on θ. Cross-entropy is a strictly proper scoring rule, so over all functions of the student's input the minimiser is the **conditional expectation of the target given what the student has heard**:

    p*(· | x[0:e(t)]) = E[ q_t | x[0:e(t)] ]

The loss left over at that optimum is E[ KL(q_t ‖ E[q_t | x[0:e(t)]]) ]. This is **zero if and only if q_t is a function of x[0:e(t)]**, i.e. the teacher's view is inside the student's view. That one condition sorts the options:

| View V_t (`slid/targets.py`) | Inside the student's view? | What the optimal student learns |
|---|---|---|
| **full:** x[0:N] | No | E[final clip label ∣ prefix]: a *forecast* of the whole-clip answer. Early frames get a confident target the student cannot justify, so it hedges or guesses. On a Hindi→English clip the one label is wrong for half the clip. |
| **centred:** x[e−W/2 : e+W/2] | No (W/2 of future) | A forecast of the next 1.5 s. Near a switch it is asked to anticipate a language not yet spoken, so the loss cannot reach zero and the posterior blurs. |
| **prefix:** x[0:e] | Yes | The teacher exactly. Early targets are honestly uncertain, so the student is calibrated. But after a switch the teacher still hears all the earlier audio, so the *target itself* is slow to switch. |
| **causal (ours):** x[e−W : e], W = 3 s | Yes | The teacher exactly, **and** it forgets audio older than 3 s, so it can follow a switch. |

**Connection to the "emit delay" recipe.** The causal window ending at e(t) is the centred window of the frame W/2 earlier. So training on causal targets means reporting the teacher's centred decision about time t − W/2 at time t: an **emit delay Δ = W/2 = 1.5 s** that is built into the target, not a separate knob. Shrinking W trades less lag for noisier targets, since the teacher is weaker on short audio (see the bake-off).

**Weighting and temperature:**
- **No per-frame down-weighting.** With information-matched targets, early frames already have high-entropy targets, so there's no need to hand-tune "don't trust early frames". The first target is placed after 0.4 s of audio because a 0.1 s teacher call is meaningless.
- **Temperature T = 1.** The targets are already soft and calibrated (ECE in the bake-off). Sharpening or smoothing them would break the calibration that the Part 2 commit threshold relies on.

**Measured, same student and same teacher, 4 target types** (`results/ablation/`):

⟨TBD table: acc@1/2/3 s, end, ECE, teacher agreement, hi→en lag (raw / committed), flips/min⟩

## 4. Student and latency budget (`slid/student.py`)

**Architecture:**
- 80-bin log-mel, 25 ms window, 10 ms hop, `center=False`.
- **Global** CMVN: per-utterance normalisation would leak future audio into every frame.
- 3 causal stride-2 convolutions give 80 ms frames.
- 6 Conformer blocks: d = 144, 4 heads, FFN 576, causal depthwise conv with kernel 15, LayerNorm instead of BatchNorm.
- Linear head over the 7 languages. 3.06M parameters.

**Causality:**
- Convolutions are left-padded.
- Self-attention uses a **chunk mask**: a frame sees its own chunk and all earlier ones. So the lookahead is bounded by the chunk no matter how many layers are stacked (per-frame lookahead masks would add up across layers).
- The decision is read at the last frame of each chunk.
- `tests/test_student.py` checks that perturbing audio after a chunk ends leaves every earlier output bit-identical, for chunk sizes 1/2/4/8.

**Latency** (why this architecture):
- **Algorithmic latency** = 25 ms window + one chunk of buffering: 80 / 160 / 320 / 640 ms at chunk 1 / 2 / 4 / 8. The default operating point is 320 ms.
- **Compute** = ⟨TBD⟩× real time on one CPU thread (a 10 s clip in ⟨TBD⟩ ms). Per-call cost is negligible next to ASR, so **the latency budget is set by evidence, not compute**: the commit policy (Part 2) waits for enough speech to be confident.
- **Dynamic chunk training:** chunk size is re-drawn every batch from {1,2,4,8}. One model serves every operating point, and section 5 reports accuracy per chunk: the latency/accuracy curve comes for free.
- **Why not a GRU or TCN:** a GRU is causal but has no bounded lookahead to trade. A TCN has lookahead that grows with depth. The chunked Conformer is the standard streaming-ASR encoder (U2/WeNet, NeMo cache-aware), so it could share a front-end with the production ASR.

## 5. Sanity checks and results

⟨TBD: loss curve, held-out KD + agreement, per-chunk table, telephony, switch lag decomposition⟩

## 6. Reproduce

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -e . speechbrain transformers sentencepiece safetensors pyarrow tqdm
# accept the licences for bodhan-ai/indic-transcribe-core and ai4bharat/Svarah on Hugging Face, then `hf auth login`
python scripts/prepare_data.py                               # FLEURS + Svarah, manifests, switch + mix clips
python scripts/teacher_bakeoff.py --teacher indic-transcribe   # repeat per teacher
python scripts/build_targets.py --teacher indic-transcribe --kinds causal prefix centered full
python scripts/train.py --teacher indic-transcribe --kind causal
python scripts/eval_student.py --ckpt checkpoints/indic-transcribe_causal/student.pt --teacher indic-transcribe
pytest -q
```

## 7. Implemented vs. not

**Implemented and run:**
- teacher interface for 5 teachers
- the bake-off
- the ensemble
- 4 target types
- the student
- training with a NaN guard
- streaming evaluation with a runnable commit policy
- unit tests

**Stubbed or not done:**
- **True cached streaming inference:** we simulate streaming with masks, which gives identical outputs.
- **Real code-switched evaluation data.**
- **Training to convergence** on hundreds of hours.
- **Bounded-memory / per-call language cache** in the student (see DESIGN §7).

**With more compute:**
- ~1k hours of IndicVoices + MUCS + Svarah-style accented English, relabelled by the ensemble teacher
- telephony augmentation in training
- a sweep of window W and chunk size
- calibration (temperature scaling) checked on real calls
