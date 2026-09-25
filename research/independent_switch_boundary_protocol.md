# Independent switch trials at true speech boundaries

Research cutoff: **2026-09-26**. Web sources and Hugging Face Hub cards/API metadata were checked on that date. This note addresses backlog **METHOD-6** only: replace reversed, fixed-padding switch fixtures with source-disjoint trials whose speech boundary is represented at an exact sample.

## Bottom line

The current two switch-evaluation files are **not two independent trials**. They reverse the same Hindi and English source utterances. Both annotate the switch at exactly 4.000 s, even though the new language's first threshold-active sample is at 4.080 s. Fixed four-second padding also places 157.69 ms of zeros before the nominal Hindi→English join and 903.25 ms before the English→Hindi join. Including each component's retained 80 ms endpoint margin, the last source-language activity to first target-language activity gaps are about **317.69 ms** and **1,063.25 ms**.

This causes three separable validity problems:

1. Reversing one pair measures two directions, but it does not create two independent acoustic, lexical, speaker, or channel trials.
2. A nominal file join is not necessarily a language-change event. Before target speech begins, the correct acoustic state is nonspeech, not the old or new language.
3. Fixed four-second truncation cuts both training components while speech is still threshold-active; fixed padding makes direction and gap duration perfectly correlated in evaluation.

The minimal correction is a **variable-length, sample-indexed manifest**. Preserve every complete spoken utterance, store source asset identities and hashes, represent nonspeech explicitly, and score switch latency from `to_speech_onset_sample`, the first sample at which the new language can be heard. Keep `join_sample` and `from_speech_end_sample` as separate diagnostic timestamps.

The clean diagnostic suite should cross language change with voice change. Same-language/voice-change trials reveal speaker shortcuts; same-language/same-voice trials reveal splice artifacts; language-change/same-voice trials provide the least-confounded synthetic switch test. Same-voice Hindi/English synthesis appears feasible with five Microsoft bilingual voice families, but live endpoint availability remains **unverified** in this environment.

No accuracy or latency gain is claimed. This is an evaluation-validity correction; every expected model effect below is **unverified** until measured.

## 1. Read-only audit of the current fixtures

The audit read [`scripts/prepare_data.py`](../scripts/prepare_data.py), [`scripts/eval.py`](../scripts/eval.py), `data/generated/manifest.jsonl`, and the generated WAV samples. It did not regenerate data or run a model.

### 1.1 How the files are built

- `decode_and_write` identifies activity with `abs(sample) > 0.003`, trims around it, and retains 80 ms at each end.
- `four_second_segment` truncates components longer than four seconds and right-pads shorter components with exact zeros.
- `write_switch` concatenates two such four-second arrays and writes language segments `[0,4]` and `[4,8]` seconds.
- The evaluator scores only `switch_hi_en_eval` and treats the first segment's floating-point end time, 4.0 s, as truth.
- `switch_hi_en_eval` uses `hi_heldout_10` then `en_heldout_10`; `switch_en_hi_eval` uses the exact same components in reverse.
- The manifest stores combined text, voice IDs, and floating-point segment times, but not component asset IDs, source hashes, source sample spans, the join sample, speech onset/offset samples, or boundary-annotation provenance.

### 1.2 Exact waveform observations

All files are 16 kHz. “Active” below means only the repository's amplitude heuristic `abs(x) > 0.003`; it is **not a verified phonetic or human speech annotation**.

| Fixture | First source duration before 4 s transform | Transform at 4 s | Last first-source active sample | First second-source active sample | Consequence |
|---|---:|---|---:|---:|---|
| `switch_hi_en_train` | Hindi 5.8318125 s | truncate 1.8318125 s | 63,999 | 65,280 | Source remains active at the cut; target speech begins 80 ms after the annotation |
| `switch_hi_en_eval` | Hindi 3.8423125 s | pad 0.1576875 s | 60,196 | 65,280 | Last source activity is 237.69 ms before annotation; target starts 80 ms after it; total activity gap ≈317.69 ms |
| `switch_en_hi_eval` | English 3.0967500 s | pad 0.9032500 s | 48,267 | 65,280 | Last source activity is 983.25 ms before annotation; target starts 80 ms after it; total activity gap ≈1,063.25 ms |

The second training component is also longer than four seconds (4.4504375 s), so its output is truncated by 0.4504375 s. The first training component's absolute sample discontinuity at the fixed join is about 0.07123 full scale; this proves a waveform discontinuity, not that the cut lands inside a word. Whether either cut removes a phoneme or word ending is **unverified** because no word alignment or listening annotation exists.

For the evaluation pair, fixed padding alone explains the backlog's 158/903 ms gaps. The amplitude-based gaps are another 160 ms larger end to end because every decoded component retains approximately 80 ms of margin at both sides. The current label assigns source language to the pre-join zeros and target language to the post-join leading margin.

### 1.3 Independence and shortcut audit

The exact number of source-disjoint evaluation pairs is **one**, not two. Reversal preserves both source identities, texts, synthetic voices, renderer characteristics, and channel. Frames inside a file are repeated measurements, not independent trials; the two directed files also cannot be treated as independent samples.

Every current Hindi↔English switch changes synthetic voice together with language. Thus a model can respond to speaker/rendering change instead of language. The held-out set has one Hindi voice and one English voice, so frame- or clip-level confidence intervals cannot represent a population of voices. This is a controlled smoke test only.

## 2. What prior work says about synthetic boundaries

### 2.1 Speaker change is a known language-diarization confound

[SAGE-LD](https://arxiv.org/abs/2510.00582) (**posted 2025-10-01**) states that naive concatenation of monolingual utterances couples the language boundary to speaker change and can let a language-diarization model behave like a speaker diarizer. Its simulated-data pipeline uses voice conversion to unify speaker identity, and its 25 ms output rate outperformed 105/205 ms variants on its benchmarks. The paper's 23–52% relative improvements and temporal-resolution result do **not** establish gains for this small causal student.

[UniCoM](https://aclanthology.org/2025.findings-emnlp.715/) (**Findings of EMNLP 2025**) likewise defines code-switching as multiple languages within one speaker and argues that simple monolingual concatenation loses speaker identity and linguistic nuance. It combines forced alignment, language-aware segmentation, and voice conversion. That is evidence for controlling identity, not proof that voice-converted audio is artifact-free.

Microsoft's [September 2024 Azure Speech release notes](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/releasenotes#september-2024-release) list five voice families with both `hi-IN` and `en-IN` variants: Aarav, Ananya, Kavya, Kunal, and Rehaan. They are the cheapest plausible source of language-change/same-voice-family trials. The repository's unofficial Edge endpoint could not be validated during the preceding voice audit because DNS failed, so successful rendering and acoustic identity across the two locale variants remain **unverified**.

### 2.2 Nonspeech must be explicit

[Liu et al., *End-to-End Language Diarization for Bilingual Code-Switching Speech*](https://www.isca-archive.org/interspeech_2021/liu21d_interspeech.html) (**Interspeech, 2021-08-30 to 2021-09-03**) simulated code switching by concatenating at most five chronological monolingual utterances from the same recording. Their with-silence condition treats silence as a third class, and WSTCSMC reference labels use 200 ms segments. This supports separating nonspeech from language labels; its reported 89.84% simulated accuracy does not transfer to this dataset.

The [DISPLACE 2024 evaluation plan](https://displace2024.github.io/docs/DISPLACE2024_Evaluation_Plan_v2.pdf) (**2024**) uses human reference segmentation for language diarization, explicitly specifies how pauses are treated, and scores DER without a tolerance collar. Its rule that pauses of at most 300 ms do not create a segment break is an annotation convention for that challenge, not a justified switch-latency clock for this router. The transferable lesson is to version the pause and boundary policy.

### 2.3 Duration and splice treatment change the task

[Mishra and Prasanna, *Spoken Language Change Detection Inspired by Speaker Change Detection*](https://doi.org/10.1007/s00034-024-02743-w) (**Circuits, Systems, and Signal Processing, 2024**) attribute an important synthetic-versus-practical mismatch to monolingual segment-duration distributions and find that language-change decisions need longer evidence than speaker-change decisions. Therefore one fixed four-second context is not representative; results should be stratified by actual pre/post speech duration.

[Speech Collage](https://arxiv.org/abs/2309.15674) (**posted 2023-09-27; ICASSP 2024**) synthesizes code-switched ASR data by splicing aligned word/character units and uses overlap-add to smooth joins. Overlap-add is useful training augmentation, but it makes the exact language boundary or overlap interval ambiguous. Its ASR gains are not streaming-LID evidence. Do not use crossfaded/overlapped speech as the headline exact-latency reference; if included, mark overlap explicitly and score it separately.

## 3. A sample-indexed boundary contract

There is no single “true switch sample” when the old language ends, silence follows, and the new language begins. Store all three events:

```json
{
  "sample_rate_hz": 16000,
  "join_sample": 61234,
  "from_speech_end_sample": 60000,
  "to_speech_onset_sample": 62800,
  "decision_event_sample": 62800,
  "gap_samples": 2800,
  "boundary_kind": "speech_nonspeech_speech",
  "annotation": {
    "method": "human_audited_vad_v1",
    "uncertainty_samples": 160,
    "annotator_count": 2
  }
}
```

Definitions:

- `from_speech_end_sample` is the first sample after the last reference speech of the old language.
- `to_speech_onset_sample` is the first reference speech sample of the new language.
- `join_sample` is merely where two stored component arrays meet. It may lie inside nonspeech.
- `gap_samples = to_speech_onset_sample - from_speech_end_sample`; it must be nonnegative unless an explicit overlap condition is represented.
- `decision_event_sample` is `to_speech_onset_sample` for routing latency. Before that sample, target-language acoustic evidence does not exist.
- Reference intervals between speech regions are `nonspeech`, not Hindi or English. A router may retain its last route during them, but they do not train or score a language decision.

The construction time can be exact to a sample even when the semantic onset has annotation uncertainty. Synthetic fixtures created under a deterministic, human-audited onset rule may be scored without a construction collar; natural annotations must retain their uncertainty or versioned collar and must not be pooled silently with construction-exact results.

The scorer should translate `decision_event_sample` into the established semantic/availability clock. Feature framing, lookahead, chunk closure, and CPU emission occur later; they must not be folded into or subtracted from the reference event.

## 4. Required provenance and integrity checks

Each component needs:

- canonical dataset/model ID, pinned revision, row/utterance ID, local SHA-256, original sample rate, and renderer/model version;
- source waveform sample span and output waveform sample span;
- normalized transcript and transcript hash;
- language, speaker/voice ID, provider/recording ID, and channel ID where available;
- speech onset/offset samples, annotation method, annotator count, and uncertainty;
- every transform, including resampling, gain, trim, explicit gap, fade, overlap, or codec;
- complete-utterance confirmation and the reason for any excluded content.

Automated acceptance tests should prove:

1. Every output sample is reconstructed from named source spans plus an explicit gap/transform; hashes match.
2. Reference intervals are integer sample half-open ranges, ordered, within the waveform, and cover every scored sample exactly once.
3. `decision_event_sample == to_speech_onset_sample`, `gap_samples >= 0`, and target speech is absent before the decision event under the versioned annotation rule.
4. A “complete” fixture never truncates annotated speech or transcript content. An amplitude threshold alone cannot certify this.
5. No evaluation source asset, source waveform hash, normalized transcript, or unordered component-pair key is reused in another evaluation fixture. In particular, `A→B` forbids `B→A` with the same assets.
6. Train/development/test source identities and waveform hashes are disjoint. Grouped metrics use speaker/voice/video as the sampling unit; frames are never treated as independent.
7. Same-language controls are built and scored with the identical splice pipeline, so a click or renderer boundary cannot masquerade as language evidence.
8. If a fade or overlap touches speech, the fixture is tagged `overlap_or_crossfade` and excluded from exact single-event latency unless a separate overlap reference is supplied.

The smallest repair with existing sources is to use `hi_heldout_10 → en_heldout_10` and a different pair such as `en_heldout_11 → hi_heldout_11`, preserve full utterances, and store exact samples. That yields two source-disjoint smoke fixtures, but still only one voice per language and no population-level claim.

## 5. Recommended controlled suite

Build a `language change × voice change` factorial rather than only positive switches:

| Condition | What it diagnoses | Headline use |
|---|---|---|
| Same language, same voice, different complete utterances | Join/click/text-boundary artifact | Negative control |
| Same language, different voice | Speaker/provider shortcut | Negative control |
| Different language, same bilingual voice family | Language effect with the principal speaker cue controlled | Primary synthetic switch result |
| Different language, different voice | Current confounded deployment stress | Secondary stress result |

Use both Hindi→English and English→Hindi for language-change cells, and Hindi→Hindi plus English→English for controls. A reasonable first release is eight source-disjoint boundaries per row/direction or language, **64 boundaries total**, with no component or transcript reused and at least four voice families represented. This number is a test-budget proposal, not a power calculation.

Select complete utterances into actual target-speech-duration bins `{0.5–<1, 1–<2, 2–<4, ≥4 s}` rather than cropping every component to a fixed length. Add explicit gap strata `{0–50, 100–300, 500–1,000 ms}` as a separate factor or balanced diagnostic slice. Never allow direction, voice condition, duration, or gap to be perfectly correlated.

Report for each condition and cluster:

- raw and committed recall at 0.25/0.5/1/2/3 s after target-speech onset;
- correct/wrong/no-commit counts, missed boundaries, median/p95 stable lag with `n/N`;
- same-language false committed changes and inverse flip-flops per voiced hour;
- accuracy and target recall by speech-duration and gap bin;
- voice-family-macro and source-pair-macro results, not frame-only uncertainty;
- whether the old language was correctly acquired before the event.

A voice-only boundary causing a committed language change is a false switch. A same-voice language boundary that is missed while different-voice language boundaries succeed is direct evidence of speaker dependence. The size and direction of those effects are **unverified**.

## 6. Natural-data validation and Hugging Face Hub audit

Hub findings are dated **2026-09-26** and non-exhaustive. None of the reviewed public cards provides sample-accurate Hindi↔English language boundaries ready for streaming-latency scoring.

| Hub dataset | Dated facts | Boundary suitability |
|---|---|---|
| [`byan/cs-yodas`](https://huggingface.co/datasets/byan/cs-yodas), revision `e51028041b403f63c99ac91a4af040e72d0cad0e` | 34,030 mined records, including 1,834 Hindi; 68.7 GB; card says CC-BY-NC-4.0. The Hub release exposes six language-pair archives, while the [paper](https://arxiv.org/abs/2606.11514) (**posted 2026-06-09**) describes seven matrix languages. The paper's audited Hindi precision is 82.9%; this is candidate-mining precision, not boundary accuracy. | Useful in-the-wild Hindi-English pool only after listening and manual boundary annotation. JSON metadata has utterance/context text but no token-language times or speaker IDs. Evaluation/non-commercial use only unless separately cleared. |
| [`byan/cs-fleurs`](https://huggingface.co/datasets/byan/cs-fleurs), revision `0cdbf166c5517ae4b6eb1c54248522eedec53017` | 90,742 rows / about 40.7 GB, read and generative/concatenative TTS; card tag CC-BY-NC-4.0. The [Interspeech 2025 paper](https://www.isca-archive.org/interspeech_2025/yan25c_interspeech.html) describes 113 language pairs and roughly 294 h, while the card rounds to 300 h. | Has no switch timestamps. Useful for clip/set LID or manual alignment, not direct event latency. Synthetic subsets also require speaker/splice controls. |
| [`Trelis/cs-fleurs-hineng-read-test`](https://huggingface.co/datasets/Trelis/cs-fleurs-hineng-read-test), revision `2b1db663c7b1bdc7f8e25efcc7efe460eca124c0` | 233 Hindi-English read-test rows; CC-BY-NC-4.0; schema exposes audio/transcription/file/duration only. | Small listening pool; no language intervals, switch times, or speaker identity in the exposed schema. |
| [`Trelis/hiacc-adult-test-eval`](https://huggingface.co/datasets/Trelis/hiacc-adult-test-eval), revision `45782b4752f1e353173cd0b940332a984fc0a249` | 664 adult eval rows; card says CC BY 4.0; fields include participant ID, transcript, duration, code-switch count, and direction counts. | Better natural Hinglish candidate, but no token/word language timestamps. Prefer the canonical HiACC release and manually annotate, as specified in [`indic_codeswitch_evaluation_datasets.md`](indic_codeswitch_evaluation_datasets.md). |
| [`liva-ai/hindi-english-asr`](https://huggingface.co/datasets/liva-ai/hindi-english-asr), revision `f9424dd86463507ec251bbad23892f9e9dc87a8e` | Ten rows with speaker-turn millisecond timestamps. | Turns themselves can contain both languages; speaker-turn times are not token-language boundaries. Too small and not directly scoreable. |

For the first natural check, pin the CS-YODAS revision above, sample 50 Hindi-English candidates with unique source-video/utterance identifiers where recoverable, listen to every item, reject mislabeled or borrowing-only cases into explicit categories, and manually annotate sustained switch intervals. Double-annotate at least 20%. Because the Hub data has no reliable speaker field, cluster uncertainty by source video, not by frame. The expected synthetic-to-natural gap is **unverified**.

HiACC remains the preferable licensed natural benchmark if its canonical audio can be downloaded and speaker split verified; the earlier dataset memo already proposes at least 200 manually marked sustained boundaries. Do not infer a switch timestamp from a transcript, `code_switch_count`, a speaker turn, or an LLM-mined utterance label.

## 7. Concrete experiment sequence

1. **Correctness-only v2 builder:** remove fixed four-second truncation/padding; emit full components and the sample-indexed schema; add integrity tests. Re-run the current model only to show how nominal-join versus target-onset clocks differ.
2. **Two source-disjoint smoke trials:** use different component pairs for the two directions. Treat results as deterministic fixtures, not an accuracy estimate.
3. **Factorial synthetic diagnostic:** build the 64-boundary suite, freeze model and policy, and quantify voice-only false switches before training on any new switch data.
4. **Natural boundary slice:** manually annotate pinned CS-YODAS or canonical HiACC items and keep these locked from threshold selection.
5. Only after the evaluator is valid, decide whether source-diverse switch training or voice-invariant augmentation improves the causal student.

Success for METHOD-6 is not a higher score. It is that every reported lag has an acoustically available, sample-indexed target; every direction uses distinct source material; nonspeech and overlap are explicit; and the model cannot pass solely by detecting a voice or splice change.

## Source and claim limits

- Local amplitude observations are exact for the generated WAVs but do not establish phonetic speech boundaries.
- SAGE-LD, UniCoM, Liu et al., DISPLACE, Mishra–Prasanna, and Speech Collage support design choices; none validates this repository's proposed sample counts or expected gains.
- Voice-family equivalence across `hi-IN` and `en-IN` is a provider naming/control assumption, not verified same-speaker acoustics.
- Hub licenses and metadata can change. Pin revisions and retain the license text with every manifest.
- CS-YODAS's paper/card release-count discrepancy and mined-label noise must be preserved in provenance, not silently reconciled.
