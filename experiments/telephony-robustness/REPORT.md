# Telephony robustness matrix

## Question

Are the pinned ECAPA teacher and submitted causal TCN student robust to a
validated 8 kHz telephone passband, G.711 A-law/mu-law companding, white noise,
and noise-before-codec interactions on the same held-out clips as the main
pipeline?

## Setup

The experiment imports the main pipeline's manifest/audio utilities, log-mel
frontend, student implementation, streaming timing helpers, teacher window
extractor, selected-language mapping, and pinned ECAPA artifact resolver. It
does not edit main-pipeline code. The models are:

- `speechbrain/lang-id-voxlingua107-ecapa` at revision
  `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`, artifact SHA-256
  `f193a054...c0f706`;
- the frozen 42,567-parameter student checkpoint
  `bb2a7e2...6c962e8`, run ID `lidrun-cc5e6cf7...ffd504`.

Inputs are the main pipeline's same 21 speaker-disjoint held-out monolingual
clips and both held-out switch clips. Their canonical manifest-record-plus-WAV
fingerprint is
`fe1ed0b98a5dd2e77c186594cde9d23d2a1763a37dddb9d2ca23a73e6cc55952`.
For monolingual scoring, each complete waveform is transformed once and then
leading 1, 2, and 4 s prefixes are taken, yielding 63 paired requests per
condition. This is a leading-prefix protocol, not the earlier teacher
bake-off's centred-crop protocol, so only paired deltas within this matrix are
directly comparable.

The 11 conditions are clean 16 kHz PCM; `resample_only` (16 -> 8 -> 16 kHz);
explicit narrowband PCM; G.711 A-law and mu-law; native 20 and 10 dB white
noise; and 20/10 dB noise before each codec. Narrowband arms use a causal
129-tap 300--3,400 Hz windowed-sinc FIR at 8 kHz. Its measured incremental
delay is 8.0 ms, exactly the theoretical delay, and switch boundaries are
shifted by that amount before lag scoring. No arm uses peak normalization and
no sample clipped before coding. The deterministic noise realizations are
shared across SNR/codec arms and achieved their requested whole-clip SNRs.
SoundFile/libsndfile A-law and mu-law bytes and decoded PCM values passed fixed
reference vectors before inference.

Teacher clip posteriors are the restricted, renormalized seven-way posterior.
Student clip posteriors average delay-aligned stable streaming frame
posteriors. The predeclared monolingual gate requires, relative to clean,
macro-F1 loss at most 5 points, every class-recall loss at most 10 points, and
median conditional `KL(clean || transformed)` at most 0.2. The switch gate
requires no extra miss and at most 100 ms worse lag. Teacher switch lag is the
online availability time confirming a 500 ms-stable target run. The unchanged
student policy is scored only when the clean student detects both directions.
All inference used six Torch CPU threads.

## Numbers

### Teacher

| Condition | Correct / 63 | Macro-F1 | F1 drop | Worst class recall drop | Median / p95 KL | Paired flips | Mono gate | Teacher switches | Worst lag delta |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|
| clean | 57 | 90.36% | 0.00 pp | 0.00 pp | 0 / 0 | 0 | pass | 2/2 | 0 ms |
| resample only | 50 | 77.87% | 12.49 pp | 55.56 pp | 0.016 / 1.634 | 11 | fail | 2/2 | 0 ms |
| narrowband PCM | 51 | 80.83% | 9.53 pp | 44.44 pp | 0.046 / 2.208 | 12 | fail | 2/2 | -8 ms |
| A-law | 50 | 78.10% | 12.25 pp | 55.56 pp | 0.077 / 1.550 | 13 | fail | 2/2 | -8 ms |
| mu-law | 51 | 79.40% | 10.95 pp | 55.56 pp | 0.074 / 1.669 | 12 | fail | 2/2 | -8 ms |
| noise 20 dB | 57 | 90.38% | -0.02 pp | 11.11 pp | 0.003 / 0.191 | 4 | fail | 2/2 | 0 ms |
| noise 10 dB | 49 | 77.90% | 12.45 pp | 33.33 pp | 0.020 / 1.641 | 9 | fail | 2/2 | 0 ms |
| noise 20 dB + A-law | 47 | 72.26% | 18.10 pp | 55.56 pp | 0.076 / 1.879 | 13 | fail | 2/2 | +492 ms |
| noise 10 dB + A-law | 44 | 67.28% | 23.07 pp | 66.67 pp | 0.185 / 4.394 | 17 | fail | 2/2 | +492 ms |
| noise 20 dB + mu-law | 45 | 69.24% | 21.12 pp | 55.56 pp | 0.071 / 1.887 | 15 | fail | 2/2 | +492 ms |
| noise 10 dB + mu-law | 43 | 65.49% | 24.87 pp | 66.67 pp | 0.177 / 4.159 | 15 | fail | 2/2 | +492 ms |

Clean teacher accuracy was 16/21, 20/21, and 21/21 at leading 1, 2, and 4 s.
Resampling alone reduced those counts to 15/21, 17/21, and 18/21. Explicit
narrowband PCM recovered the 1 s and 4 s cells to 16/21 and 20/21 but fell to
15/21 at 2 s, so its 9.53-point aggregate macro-F1 loss still failed the gate.
The largest channel-specific weakness was Marathi: recall fell 44.44--55.56
points in every codec-only arm and 55.56--66.67 points in the interaction arms.

Median KL alone hides sparse severe failures: resampling has median KL 0.016
but p95 1.634 and 11/63 prediction flips. At 20 dB, noise alone preserved
aggregate F1, but one lost Bengali and one lost Gujarati request make the
worst-class drop 11.11 points, just beyond the predeclared 10-point gate. The
noise-plus-codec interactions are clearly worse than either mild component:
they lose 18.10--24.87 macro-F1 points and flip 13--17 of 63 paired requests.

The clean teacher detected both stable switches. Its online confirmation lags
were 2,275 ms for Hindi-to-English and 2,025 ms for English-to-Hindi.
Resampling, passband, codecs, and noise alone introduced no material extra
lag after the 8 ms filter-delay correction. Every noise-plus-codec arm delayed
English-to-Hindi confirmation to 2,517 ms, a worst paired increase of 492 ms;
Hindi-to-English remained at 2,267 ms on the delay-adjusted clock. Thus all
four interaction arms also fail the switch gate despite detecting 2/2 events.

### Student

| Condition | Correct / 63 | Macro-F1 | F1 change vs clean | Median / p95 KL | Paired flips | Mono gate | Policy switches |
|---|---:|---:|---:|---:|---:|:---:|:---:|
| clean | 13 | 9.63% | 0.00 pp | 0 / 0 | 0 | pass | 0/2 |
| resample only | 12 | 10.12% | +0.49 pp | 0.905 / 3.278 | 30 | fail | 0/2 |
| narrowband PCM | 9 | 3.96% | -5.68 pp | 0.561 / 3.052 | 28 | fail | 0/2 |
| A-law | 9 | 3.96% | -5.68 pp | 0.600 / 3.096 | 29 | fail | 0/2 |
| mu-law | 9 | 4.02% | -5.62 pp | 0.639 / 3.123 | 29 | fail | 0/2 |
| noise 20 dB | 13 | 14.59% | +4.95 pp | 0.328 / 1.170 | 22 | fail | 0/2 |
| noise 10 dB | 9 | 7.44% | -2.19 pp | 1.129 / 3.290 | 31 | fail | 0/2 |
| noise 20 dB + A-law | 9 | 3.90% | -5.74 pp | 0.687 / 3.982 | 29 | fail | 0/2 |
| noise 10 dB + A-law | 12 | 9.91% | +0.28 pp | 0.528 / 4.317 | 23 | fail | 0/2 |
| noise 20 dB + mu-law | 9 | 3.90% | -5.74 pp | 0.689 / 3.987 | 29 | fail | 0/2 |
| noise 10 dB + mu-law | 12 | 9.91% | +0.28 pp | 0.544 / 4.309 | 23 | fail | 0/2 |

The student is at a floor even on clean input, so small macro-F1 increases
under resampling or noise are accidental redistribution, not robustness gains.
The paired measures make the channel sensitivity unambiguous: every non-clean
arm fails, median KL is 0.328--1.129, and 22--31 of 63 predictions flip. On
both clean switches the policy fails to establish the source-language commit
and detects 0/2 transitions. Student switch degradation is therefore
**inconclusive**, not zero: there is no valid clean event against which to
measure an extra miss or lag.

## Verdict: reject

Reject the hypothesis that either frozen model is telephone-channel robust
under the predeclared gates. `resample_only` is already enough to fail both
models and must remain explicitly named as a synthetic resampling check, not
evidence of telephony readiness. For ECAPA, the narrowband/codec arms lose
9.53--12.49 macro-F1 points and show class-concentrated tail failures; adding
noise before either G.711 codec worsens the loss to 18.10--24.87 points and
adds 492 ms to one teacher switch. For the student, absolute accuracy is too
low for a useful channel claim, while paired KL and flips still reject
invariance.

Adopt this as the gate to run the already-proposed clean-teacher/degraded-
student cross-view KD ablation next: hold targets, architecture, seed, steps,
and clean view fixed, and train only the student input over the validated
narrowband PCM/A-law/mu-law views. Do not change teachers from this experiment;
it compares channels, not teacher candidates. Keep real 8 kHz
IndicTelephony-Bench evaluation separate, because synthetic TTS, white noise,
and two mirrored splices cannot estimate real-call performance.

The run self-validated the frozen checkpoint, exact corpus bytes, historical
target metadata, pinned teacher artifacts, inference-critical source files,
driver, inputs, and unchanged launch snapshot. It does not claim the current
main release gate: `scripts/train.py` and `src/streaming_lid/run_identity.py`
already differed from the historical checkpoint source, but neither is used
for channel inference. Canonical experiment run ID:
`87aa5b0b13e590684426ac62d8b48b749c0ab0ba3c9fe0ca6adf53c392217b32`;
`results.json` SHA-256:
`93712aeb7086bf8e712f89441af12dffa0eee08c5b6db5bcdf4db7129d0940b3`.
Full paired posteriors, per-prefix confusion tables, switch traces, transform
diagnostics, timings, package versions, and identity checks are in
`results.json`.
