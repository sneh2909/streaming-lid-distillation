# 8 kHz / telephony robustness for teacher and student LID

Research date: **2026-09-26**. This addresses backlog **METHOD-10 / SEED-6** only. It distinguishes a sample-rate bottleneck, a telephone codec, and a real telephone domain; they are not interchangeable. All web and Hugging Face sources were accessed on 2026-09-26 unless another access date is stated. “Unverified” marks a hypothesis or a property not established by the cited artifact.

## Bottom line

1. The existing teacher bake-off did **not** test a telephone codec. It applied `torchaudio.functional.resample(16 kHz -> 8 kHz -> 16 kHz)`, which is band-limited interpolation. It did not explicitly impose the 300–3400 Hz narrowband passband, G.711 A-law or mu-law quantization, packet loss, level/clipping variation, or background noise.
2. Even that mild condition moved ECAPA from **58/63 to 54/63** correct restricted seven-way crop decisions, a **6.35-point absolute drop**. A paired read of the existing artifact finds **8/63 top-1 flips**, including five correct-to-wrong and one wrong-to-correct flip. The sample is only 21 synthetic clips, its three duration crops are correlated, and 15/21 four-second crops contain padding, so this is a warning rather than a production estimate.
3. A new Hub dataset materially improves the evaluation path. [`ConvoZenAI/indictelephony-bench`](https://huggingface.co/datasets/ConvoZenAI/indictelephony-bench) v1.0, released **2026-09-22**, provides 25,393 human-curated turns / 30.18 h / 320 calls recorded over live SIP lines as 8 kHz mono PCM. It covers **all seven project languages**, is CC BY 4.0, and is test-only. Pin revision `d1a7902dd956cd3eb10304d7142df602273d8bd6`.
4. IndicTelephony-Bench is not a switch-latency benchmark. It gives an utterance-level set of languages but no token or frame times. It also omits speaker IDs and per-call codec negotiation. Use its single-language turns for real-telephone LID, and its Hindi–English turns only for set/presence diagnostics—not boundary lag.
5. The defensible experiment has three separately reported strata: **paired clean-to-degraded transforms** to isolate channel effects; **IndicTelephony-Bench** for clean real 8 kHz SIP output; and **GramVaani** for noisier, spontaneous Hindi domain stress. A cross-corpus accuracy difference must never be called a codec delta.
6. A dated, non-exhaustive Hub audit found **no reviewed released LID checkpoint with documented short Hindi–English 8 kHz accuracy**. Automatic resampling to 16 kHz is an input adapter, not evidence of narrowband robustness.

## 1. What the existing artifact proves

### The current transform is bandwidth-only

`experiments/teacher-bakeoff/run.py::telephony_bandlimit` does only:

```text
16 kHz float waveform
  -> torchaudio resample to 8 kHz
  -> torchaudio resample to 16 kHz
  -> trim/pad to the original sample count
```

The pinned project uses TorchAudio 2.6.0. Its official [`resample` documentation](https://docs.pytorch.org/audio/2.6.0/generated/torchaudio.functional.resample.html) describes band-limited interpolation with a default rolloff of 0.99 of Nyquist (TorchAudio 2.6.0 docs; accessed 2026-09-26). Thus the downsample removes information near and above 4 kHz, but it is not a complete narrowband telephone path.

ITU-T defines narrowband codec input as **300–3400 Hz at 8,000 samples/s**. G.711 then maps 13- or 14-bit linear PCM to 8-bit A-law or mu-law samples at 64 kbit/s; G.711 itself does not mandate packet loss or VAD/DTX. See [ITU-T J.361, section 8.4](https://www.itu.int/rec/dologin_pub.asp?id=T-REC-J.361-200611-I%21%21PDF-E&lang=e&type=items) (**2006-11**, accessed 2026-09-26). Therefore:

- `16 -> 8 -> 16 kHz` should be named **`resample_only`**, not “telephone” or “G.711”;
- an explicit 300–3400 Hz condition is needed to isolate the conventional passband;
- A-law and mu-law must be encoded/decoded as separate conditions;
- noise, packet loss, clipping, automatic gain control, and room/channel effects need separate axes rather than being silently bundled into “8 kHz.”

### Paired re-analysis of the existing ECAPA predictions

I recomputed paired diagnostics from the already-published `experiments/teacher-bakeoff/results.json`; no inference was rerun. For each crop, the stored selected-class probabilities were renormalized over `en hi mr bn ta te gu` before posterior divergence was computed. KL and Jensen–Shannon values below are in nats.

| Crop | Pairs | Restricted top-1 flips | Correct -> wrong | Wrong -> correct | Median / mean `KL(clean || 8k)` | Median / mean JS |
|---|---:|---:|---:|---:|---:|---:|
| 1 s | 21 | 3 | 1 | 1 | 0.0234 / 0.1470 | 0.0048 / 0.0344 |
| 2 s | 21 | 2 | 1 | 0 | 0.0025 / 0.1343 | 0.0007 / 0.0294 |
| 4 s | 21 | 3 | 3 | 0 | 0.0008 / 0.1895 | 0.0002 / 0.0396 |
| **All** | **63** | **8 (12.70%)** | **5** | **1** | **0.0052 / 0.1569** | **0.0017 / 0.0345** |

Across all 63 pairs, mean retained seven-language mass changed by `-0.0360` and mean conditional maximum confidence by `-0.0321`. All eight top-1 flips occurred on English or Marathi crops, so the aggregate hides strong class concentration. The median KL looks benign while the mean is about 30 times larger, showing that a median-only gate misses a small heavy tail. Future reports should include p90/p95 and the individual flip table.

Artifact provenance for this read-only calculation:

- evaluation-input SHA-256 recorded by the experiment: `52bca81e32db34edb0a9ad4529b01bd6ddd36aad6f587733a01ace669538d6fc`;
- local `results.json` SHA-256: `79189284439acc250895a1f3bbb0ff68a154c4e1b3c53666026ccd1d3b0c30fa`;
- local experiment-driver SHA-256: `66207f615a75386ef94097dc3d481bc86d27e8b6eacb9aecc259694ad7829e53`.

This still says nothing about the **student** under 8 kHz. METHOD-10 is incomplete until teacher and student are evaluated on identical paired conditions, including the live streaming path and policy.

## 2. What counts as telephony

### Prioritize PCMA and PCMU

Altwlkany, Kuric, and Lacic analyzed PSTN, VoIP, and neural codecs over more than two million files. Their provider-side H2-2024 traffic sample contained 2,579,290,920 PCMA calls, 1,701,341,847 PCMU calls, and 2,313,558 G.729A calls: **99.946% of those three-codec calls used G.711 A-law or mu-law**. They also found codec-dependent language and gender effects in objective quality. This is one provider’s traffic—not a global codec census—but it makes PCMA/PCMU the right small core matrix. See [*On the Language and Gender Biases in PSTN, VoIP and Neural Audio Codecs*](https://www.isca-archive.org/interspeech_2025/altwlkany25_interspeech.pdf) (Interspeech **2025**, especially Tables 1–4; accessed 2026-09-26).

Indian LID evidence independently supports testing encoding, not only resampling. Dey, Sahidullah, and Saha used A-law, mu-law, ADPCM, and lossy codec augmentations; codec-encoding augmentation was the strongest individual augmentation for both of their cross-corpus evaluations, while domain-invariant extensions improved cross-corpus EER by up to 5.23%. Their system and data differ from this student, so the direction transfers but the gain does **not**. See [*Cross-Corpora Spoken Language Identification with Domain Diversification and Generalization*](https://arxiv.org/abs/2302.05110) (submitted **2023-02-10**, journal article 2023; accessed 2026-09-26).

### Minimal controlled matrix

Apply each transformation to the **whole clip once**, then take prefixes/windows. Resetting a filter or codec independently on every prefix creates boundary artifacts that a real stream would not have.

| ID | Transform | What it isolates |
|---|---|---|
| `native_pcm16` | Original lossless 16 kHz waveform | Reference |
| `resample_8k` | Current 16 -> 8 -> 16 kHz band-limited interpolation | Information above the 8 kHz Nyquist limit |
| `nb_pcm` | Fixed, versioned 300–3400 Hz band-pass -> 8 kHz PCM -> 16 kHz | Conventional narrowband passband without companding |
| `g711_pcma` | `nb_pcm` plus 8-bit A-law encode/decode at 8 kHz | PCMA quantization beyond passband loss |
| `g711_pcmu` | `nb_pcm` plus 8-bit mu-law encode/decode at 8 kHz | PCMU quantization beyond passband loss |
| `noise_only_{20,10}db` | Fixed licensed noise at controlled active-speech SNR, no codec | Acoustic corruption alone |
| `pcma/noise`, `pcmu/noise` | Add noise before encoding, then codec | Interaction, reported separately |

Do not add AMR-NB, G.729, Opus, packet loss, or packet-loss concealment merely to make the grid look realistic. Add them only if production SDP/RTP telemetry shows material traffic, and preserve codec/rate as separate labels.

The existing dependency set already has a clean implementation route. libsndfile supports `SF_FORMAT_ULAW` and `SF_FORMAT_ALAW` ([libsndfile API](https://libsndfile.github.io/libsndfile/api.html), accessed 2026-09-26), and the locally pinned SoundFile 0.13.1 reports `ULAW` and `ALAW` as valid WAV subtypes. TorchAudio also exposes generic [`mu_law_encoding`](https://docs.pytorch.org/audio/2.6.0/generated/torchaudio.functional.mu_law_encoding.html), but its documentation only promises generic companding into a chosen number of quantization channels. Do not label that path bit-exact PCMU without a reference-vector parity test.

Implementation invariants for the eventual experiment:

- no per-condition peak normalization; it would erase real level/clipping sensitivity;
- encode/decode at 8 kHz, then upsample once for 16 kHz-only models;
- preserve sample count and measure/compensate deterministic filter group delay before scoring switch latency;
- assert finite samples in `[-1, 1]`, exact mono shape, and deterministic byte/output hashes;
- validate PCMA and PCMU with fixed reference vectors and record libsndfile/TorchAudio versions;
- transform clean, lossless source audio. The repository’s gTTS/Edge files originated as MP3 before decoding to WAV, so they are acceptable smoke tests but **not** clean codec-isolation sources. Altwlkany et al. likewise avoided already-lossy source files in their codec comparison.

## 3. A newly available real 8 kHz evaluation set

### IndicTelephony-Bench v1.0

The Hugging Face repository was created on **2026-09-22** and resolved on the research date to commit `d1a7902dd956cd3eb10304d7142df602273d8bd6` ([Hub API metadata](https://huggingface.co/api/datasets/ConvoZenAI/indictelephony-bench), accessed 2026-09-26). The [card at that revision](https://huggingface.co/datasets/ConvoZenAI/indictelephony-bench/blob/d1a7902dd956cd3eb10304d7142df602273d8bd6/README.md) says:

- 25,393 single-speaker turns, 30.18 h, 320 call IDs;
- 8 kHz mono 16-bit PCM captured over live SIP telephone connections;
- Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Tamil, and Telugu;
- 70.4% code-mixed, 30.0% shorter than 2 s, median duration 3.10 s;
- test-only; CC BY 4.0, including commercial adaptation with attribution;
- consented speakers, with a non-license request not to identify speakers or clone their voices.

All seven deployment languages account for **19,525 rows / 22.96 h**. The Hub’s [dataset-server statistics](https://datasets-server.huggingface.co/statistics?dataset=ConvoZenAI%2Findictelephony-bench&config=all&split=test) expose useful exact subsets:

| Exact single-language `language_tag` | `bn` | `en` | `gu` | `hi` | `mr` | `ta` | `te` | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rows | 205 | 3,714 | 277 | 662 | 46 | 439 | 1,284 | **6,627** |

There are also **2,148 `en-hi` turns**, plus thousands of English–Bengali/Gujarati/Marathi/Tamil/Telugu turns. Use `language_tag`, not the broader `language`/configuration field, as the evaluation truth: the statistics already show 3,714 exact-English tags but only 3,672 rows whose base `language` is English.

Recommended pinned loading recipe:

```python
from datasets import load_dataset

REVISION = "d1a7902dd956cd3eb10304d7142df602273d8bd6"
rows = load_dataset(
    "ConvoZenAI/indictelephony-bench",
    "all",
    split="test",
    revision=REVISION,
    streaming=True,
)
```

For a fast smoke test, take a deterministic, call-stratified maximum of 46 exact-singleton rows per project language (322 total, limited by Marathi), with duration bins `<2`, `2–5`, and `>5` seconds. The final result should score all 6,627 singleton rows. Hash each decoded waveform and record `utterance_id`, `call_id`, language tag, source revision, original sample rate, duration, and selected/not-selected reason.

### What it can and cannot establish

It is unusually well matched to a voice-bot domain, but the card itself documents important constraints:

- wording is scripted business speech, even though pairs spoke over a live call;
- calls are clean, so the set does not test substantial background noise;
- each language uses few lines and the release has `call_id` but no speaker ID;
- speaker/channel and language are therefore confounded;
- there is no clean microphone counterpart;
- `language_tag` gives a set of languages, not token or frame timing;
- the actual negotiated SIP codec is not a released column. The claim that codec and band limit are “real” is credible provenance, but whether a row is PCMA, PCMU, transcoded, or another codec is **unverified**.

Consequently:

1. Treat the 6,627 exact-singleton rows as the primary real-telephone seven-way test. Report per-language recall and macro averages; bootstrap by `call_id`, not utterance or frame.
2. Treat `en-hi` and other mixed tags as **set-valued clip diagnostics**. Report whether the model’s accumulated top-2 languages recover the annotated set, extra-language false alarms, and time to observe both labels only as a diagnostic. Do not report switch latency, order, or frame accuracy from these labels.
3. Freeze all thresholds and smoothing before looking at this test-only set. It supplies no development split.
4. Compare teachers/students *paired on the same rows*. Do not interpret across-language ranking as intrinsic language difficulty because recording lines and speakers are confounded.

### GramVaani remains useful, but for a different question

[OpenSLR SLR118](https://www.openslr.org/118/) (**2022**, accessed 2026-09-26) contains spontaneous regional Hindi telephone speech with natural noise and crowd-sourced transcripts. It is therefore a better ecological stress test than IndicTelephony-Bench’s clean scripted calls. However:

- it is Hindi-only;
- it is academic-use by default; commercial use requires permission;
- files are MP3 with mixed sampling rates from 8 to 48 kHz; the official page says 60.87% of labeled train/development files are 8 kHz, not that every eval file is native 8 kHz.

Filter and report its native sampling rate instead of silently calling the entire 3 h eval set “8 kHz.” Keep it separate from paired FLEURS/Svarah transforms and from IndicTelephony-Bench.

## 4. Hugging Face model audit for 8 kHz readiness

The Hub API query `filter=spoken-language-identification` returned 12 tagged repositories on 2026-09-26. Tags are incomplete and cards can be wrong, so this is a dated, non-exhaustive screen rather than proof that no other model exists.

| Artifact | Card evidence relevant to telephony | Conclusion |
|---|---|---|
| [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) | Trained on 16 kHz single-channel YouTube/VoxLingua audio; loader resamples input. No 8 kHz or codec metric. Card last updated 2024-11-27. | Current baseline; robustness **unverified** beyond this repository’s small resampling test. |
| [`surogate/ambernet-langid`](https://huggingface.co/surogate/ambernet-langid) | Expects 16 kHz; card explicitly calls telephone-band 8 kHz out of domain. Updated 2026-08-13. | Must pass the same matrix; CPU speed does not imply channel robustness. |
| [`facebook/mms-lid-126`](https://huggingface.co/facebook/mms-lid-126) | Transformers example resamples to 16 kHz; MMS publishes a BABEL aggregate but no short Hindi–English 8 kHz slice. Updated 2023-06-13. | Telephone proxy only, not verified for this operating point. |
| [`desert-ant-labs/ear`](https://huggingface.co/desert-ant-labs/ear) | New on-device Whisper-tiny LID, updated 2026-09-24; SDK resamples any rate to 16 kHz and uses 30 s windows. Card validation is 22 ordinary/podcast files and says clips under 30 s are less certain; no telephone breakdown. Source-available license. | Not a short streaming/8 kHz teacher claim. |

No reviewed card supplies the needed combination: all seven labels, 0.5–5 s duration results, true 8 kHz Hindi/English calls, codec-specific results, and CPU cost. Local evaluation remains mandatory.

## 5. Evaluation contract

### Paired clean/degraded set

Use the already-proposed frozen FLEURS seven-language test plus Svarah accented English and the transform matrix above. Record each source encoding, use the original FLEURS WAV bytes, and admit Svarah rows to the codec-isolation slice only after verifying that their source encoding is lossless. Keep the current 21 synthetic clips only as a regression smoke test.

For the frozen teacher and student, at `0.5/1/2/3/5 s` and full clip, report:

- known-label recall by language, macro-F1, NLL/Brier, and confidence;
- native full-space prediction, restricted seven-way prediction, and retained in-set mass;
- paired top-1 flips split into correct->wrong, wrong->correct, and wrong->different-wrong;
- conditional seven-way KL and JS with median, p90, p95, and mean;
- student-to-teacher agreement **and** student/teacher known-label accuracy;
- raw posterior transitions, committed transitions, UNKNOWN/low-confidence time, and flips per voiced minute;
- on exact-boundary synthetic switches, raw and policy switch recall/lag under every channel condition;
- teacher and student CPU timing separately, with codec decode/resampling timing either included and named or excluded and named.

Resample speakers/clips—not frames—for uncertainty intervals. Use paired bootstrap differences for channel transforms. If speaker IDs are absent, use the highest independent grouping available and disclose it.

### Real 8 kHz set

For IndicTelephony-Bench:

- primary: all exact-singleton project-language rows, with per-language and call-clustered intervals;
- shortness slices: `<1`, `1–2`, `2–5`, and `>5 s`, plus fixed prefixes where sufficient audio exists;
- mixed diagnostic: `en-hi` separately from every other language set; never convert its set label into a fake frame path;
- keep uncommitted/UNKNOWN outcomes in the denominator;
- publish row/call counts for every slice.

There is no paired clean reference, so posterior divergence from FLEURS or synthetic audio is meaningless. Use this set to rank systems paired on real phone audio, not to estimate “the codec cost.”

### Engineering gates, not literature claims

The following are proposed repository gates and are **not** established universal thresholds:

- paired telephone transform macro-F1 drop no more than 5 points, and no class recall drop over 10 points;
- median conditional KL no more than 0.2 **plus** p95 and flip tables, because the current mean/median split is heavy-tailed;
- transformed switch median lag no more than 100 ms worse and no additional missed boundary versus the same clean clip;
- channel adaptation, if added, must recover at least half of the paired macro-F1 loss while reducing clean macro-F1 by no more than 2 points and adding no inference latency.

Do not impose a “real telephony must be within X points of FLEURS” gate: corpus, speaker, style, noise, and label distributions differ, so that number would not isolate channel robustness.

## 6. If the student fails: smallest defensible training change

First evaluate; do not train on either test set. If the causal student is channel-sensitive, the simplest next arm is **cross-view KD**:

```text
teacher target = frozen ECAPA posterior on clean training audio
student input  = same waveform after a randomly selected
                 native / nb-PCM / PCMA / PCMU condition
loss           = existing delayed KL, same alignment and masks
```

This teaches channel invariance without copying the teacher’s degraded-audio errors. Compare it with an in-domain arm where both teacher and student hear the degraded waveform. Keep target type, delay, seed, step count, and batch order fixed. Expected recovery is **unverified**; an experiment goal of recovering at least half the paired performance loss is an engineering target, not a paper-backed forecast.

If cross-view KD helps only the synthetic transform but not IndicTelephony-Bench, the remaining gap is likely domain/speaker/style/noise rather than the encoded passband alone. Do not respond by tuning on the test set; add a separately licensed telephone training/development pool.

## 7. What remains unverified

- Teacher and student accuracy on IndicTelephony-Bench; no model was run in this research iteration.
- Exact codec(s), transcoding chain, packet-loss rate, and per-speaker identity for its calls.
- Correctness/inter-annotator agreement of its utterance language sets; the card says human-curated but publishes no LID agreement statistic.
- Whether PCMA and PCMU materially differ for these seven languages or for the causal student.
- Whether the current model’s 8 kHz errors arise from lost high-frequency information, companding, source MP3 artifacts, accent/speaker mismatch, or interactions among them.
- Any gain from cross-view codec KD.
- A public short-duration Hindi–English 8 kHz result for any reviewed off-the-shelf LID model.

## Source and command record

Primary/official sources, all accessed **2026-09-26**:

- ITU-T, [Recommendation J.361, narrowband and G.711 definitions](https://www.itu.int/rec/dologin_pub.asp?id=T-REC-J.361-200611-I%21%21PDF-E&lang=e&type=items), **2006-11**.
- Altwlkany, Kuric, and Lacic, [*On the Language and Gender Biases in PSTN, VoIP and Neural Audio Codecs*](https://www.isca-archive.org/interspeech_2025/altwlkany25_interspeech.pdf), Interspeech **2025**.
- Dey, Sahidullah, and Saha, [*Cross-Corpora Spoken Language Identification with Domain Diversification and Generalization*](https://arxiv.org/abs/2302.05110), submitted **2023-02-10**, Computer Speech & Language **2023**.
- ConvoZen AI, [`ConvoZenAI/indictelephony-bench`](https://huggingface.co/datasets/ConvoZenAI/indictelephony-bench), v1.0 released **2026-09-22**; [Hub API](https://huggingface.co/api/datasets/ConvoZenAI/indictelephony-bench); [dataset-server statistics](https://datasets-server.huggingface.co/statistics?dataset=ConvoZenAI%2Findictelephony-bench&config=all&split=test).
- GramVaani/OpenSLR, [SLR118](https://www.openslr.org/118/), challenge/data release **2022**.
- SpeechBrain, [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa), card last modified **2024-11-27**.
- Surogate, [`surogate/ambernet-langid`](https://huggingface.co/surogate/ambernet-langid), card/repository updated **2026-08-13**.
- Meta, [`facebook/mms-lid-126`](https://huggingface.co/facebook/mms-lid-126), repository updated **2023-06-13**.
- Desert Ant Labs, [`desert-ant-labs/ear`](https://huggingface.co/desert-ant-labs/ear), repository updated **2026-09-24**.
- TorchAudio, [`functional.resample` 2.6.0](https://docs.pytorch.org/audio/2.6.0/generated/torchaudio.functional.resample.html) and [`mu_law_encoding` 2.6.0](https://docs.pytorch.org/audio/2.6.0/generated/torchaudio.functional.mu_law_encoding.html).
- libsndfile, [API format/subtype definitions](https://libsndfile.github.io/libsndfile/api.html).

Read-only local/Hub checks:

```text
git ls-remote https://huggingface.co/datasets/ConvoZenAI/indictelephony-bench refs/heads/main
  -> d1a7902dd956cd3eb10304d7142df602273d8bd6

GET /api/datasets/ConvoZenAI/indictelephony-bench
  -> createdAt 2026-09-22T11:59:22Z; lastModified 2026-09-22T19:02:41Z

GET /splits, /size, /statistics from datasets-server.huggingface.co
  -> 25,393 rows; 1,731,495,327 parquet bytes; 320 call IDs;
     7,518 non-code-mixed / 17,875 code-mixed rows

SoundFile 0.13.1 available_subtypes("WAV")
  -> PCM_16, ULAW, ALAW all supported locally

paired parse of experiments/teacher-bakeoff/results.json
  -> ECAPA 8/63 top-1 flips; 5 correct->wrong; 1 wrong->correct;
     conditional KL median/mean 0.0052/0.1569
```
