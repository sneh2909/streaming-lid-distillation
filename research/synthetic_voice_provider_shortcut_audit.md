# Synthetic-voice, text, and provider shortcut attribution

**Topic:** backlog METHOD-7: what must be measured before the 18.51% unseen-voice result is attributed to memorisation or a voice/provider shortcut  
**Evidence cutoff and access date:** 2026-09-26  
**Scope:** one attribution question, not a new model run. “Voice” below means a synthetic voice ID, not a human speaker.

## Bottom line

The current result proves one narrow fact: the frozen student transfers poorly from 14 training synthetic voice IDs to seven disjoint held-out synthetic voice IDs. It does **not** identify why.

The cheapest decisive next step is to score the same checkpoint on its exact training audio and on a provider-fixed `voice familiarity × text familiarity` factorial. That separates failure to fit from exact-audio memorisation, text generalisation, and an Edge-voice/profile shift. A gTTS-versus-Edge comparison can show renderer-family sensitivity, but it cannot by itself prove a provider shortcut because provider and voice cannot be held independently fixed in the present corpus.

For the headline Hindi–English pair, Microsoft documents five voice names that support both `en-IN` and `hi-IN`. They permit a much cleaner counterfactual: make every voice speak both labels, then hold out whole voice identities. Availability through this repository's unofficial `edge-tts` endpoint is **unverified**; the official documentation is for Azure Speech.

No accuracy gain is expected from the diagnostic evaluation itself. Whether crossed-voice retraining improves this 42,567-parameter causal student is **unverified**.

## What the repository currently identifies

The following is a read-only audit of [`scripts/prepare_data.py`](../scripts/prepare_data.py), [`results/train_metrics.json`](../results/train_metrics.json), and [`results/eval_metrics.json`](../results/eval_metrics.json).

| Factor | Current design | What can be concluded |
|---|---|---|
| Training provider | For each language, five monolingual clips use the locale's default gTTS voice and five use one named Edge voice. Thus the 70 monolingual clips have the same 5:5 provider count in every class. | A provider-only main effect does not predict the language in this monolingual table. This does **not** remove provider-by-voice or provider-by-locale shortcuts. The one extra gTTS Hindi–English switch clip is not a provider control. |
| Training voice | Each of the 14 training voice IDs occurs under exactly one language label. | Exact synthetic voice ID is a perfect label proxy in training: `H(language | voice_id) = 0`. A model is able to use voice identity, but the current results do not show that it did. |
| Held-out condition | Three new texts per language use one new Edge voice per language. There are seven held-out voice IDs and no train/evaluation overlap. | The split demonstrates failure on these seven voices. With only one held-out voice per language, language performance is inseparable from that voice/profile. |
| Simultaneous shifts | Training-to-held-out changes exact voice, text, and the Edge voice profile documented as male-to-female; held-out also contains no gTTS clips. | The 18.51% frame accuracy cannot be assigned uniquely to voice, text/sample memorisation, profile, or mixture/domain shift. No gender claim is made for gTTS. |
| Fit evidence | The run performs 1,600 updates / 157.746 effective epochs and reduces loss, but the published artifacts do not report train-set label accuracy or train teacher agreement. | Many epochs do not prove memorisation. Low train accuracy would instead point first to optimisation, capacity, or target mismatch. |
| Switch clips | Every synthetic Hindi↔English splice also changes synthetic voice ID at the boundary. | These clips are useful pipeline fixtures, but they are not same-speaker code-switch evidence. Voice change may act as either an extra cue or an extra domain shift. |

The published held-out student numbers are 18.51% frame-micro label accuracy, 18.80% frame-macro label accuracy, and 4/21 (19.05%) clip accuracy. The teacher is correct on all 21 known synthesis labels. English and Hindi clip accuracy are both 0/3. These facts rule out teacher top-1 error on the existing monolingual held-out set, but not teacher-target quality during training or under new controls.

## Why unseen-voice failure is not yet a causal diagnosis

A useful attribution test changes one axis while holding the others fixed:

| Frozen-checkpoint slice | Question answered | Still not answered |
|---|---|---|
| Exact cached training audio | Did the final checkpoint fit/memorise the samples it repeatedly saw? | Whether it learned linguistic or shortcut features. |
| Seen Edge voice, unseen text | Does a familiar voice generalise to new linguistic content? | Pure provider or voice-profile causality. |
| Unseen Edge voice, seen text | Does familiar content survive an Edge voice/profile change? | Pure speaker identity, because the two current Edge voices also differ in documented sex/profile. |
| Unseen Edge voice, unseen text | Reproduces the current joint shift. | Which component caused the failure. |
| Same text through gTTS and Edge | How sensitive is the checkpoint to the current renderer families? | A causal provider effect: both the renderer and voice change. |
| Natural human speech | Does any synthetic finding transfer to deployment-like speech? | Which particular synthetic artifact caused a gap. |

Calling the present outcome “memorisation” skips the first three rows. Calling it a “provider shortcut” is even less supported: held-out audio uses Edge, a provider already used for half of every class's monolingual training clips. The sharper working hypothesis is **voice/profile or synthetic-domain shortcut**, pending counterfactual tests.

## External evidence

### Speaker leakage can dominate audio classification

Kuparinen's [VarDial 2026 speaker-bias study](https://aclanthology.org/2026.vardial-1.3/) (published March 2026; accessed 2026-09-26) explicitly compared speaker-dependent and speaker-independent dialect-ID partitions. Audio macro-F1 fell from 88.77 to 24.29 on Finnish and from 91.69 to 50.47 on Norwegian. In an even more diagnostic speaker-dependent counterfactual, converting every dialect to its own assigned voice reduced macro-F1 to 16.07 and 12.58, near the respective chance regimes. Pitch/noise and voice-conversion remedies were small or inconsistent across the two languages.

This supports speaker/voice control as a requirement, not the claim that voice leakage caused this project's failure. The paper studies Finnish/Norwegian dialect ID with different data and models; transfer to seven-language Indic LID is **unverified**.

### Cross every voice with every label

Abdullah et al.'s [Interspeech 2025 controlled voice-conversion study](https://www.isca-archive.org/interspeech_2025/abdullah25_interspeech.html) (published 2025; accessed 2026-09-26) is the strongest reviewed intervention. Their [paper tables](https://arxiv.org/html/2505.24713) use the same 12 converted target voices across all five dialects in an “unbiased” condition, versus 12 disjoint target voices per dialect in a “biased” condition. With MMS, accuracy was:

| Training condition | In-domain | Cross-domain |
|---|---:|---:|
| Natural-speech baseline | 75.94% | 59.60% |
| Converted, same 12 voices across every dialect | 83.38% | 76.61% |
| Converted, 12 disjoint voices per dialect | 27.33% | 24.32% |

Their best natural-plus-four-voice system reached 85.32% in-domain and 80.73% mean cross-domain accuracy, a 34.07% relative cross-domain improvement over their MMS baseline. The released [`badrex/mms-300m-arabic-dialect-identifier`](https://huggingface.co/badrex/mms-300m-arabic-dialect-identifier) is a 0.3B-parameter wav2vec2/MMS model under CC BY 4.0 (Hub revision `6220d51bce99d1fbca7ed96a5ba7966ffde99d22`, checked 2026-09-26).

The transferable design principle is to distribute each synthetic voice across every label, not merely to increase the number of voices. The reported gains are Arabic-dialect results from a much larger offline model; their magnitude for this causal student is **unverified**.

### Domain shift is a separate failure mode

Abdullah et al.'s earlier [cross-domain spoken LID study](https://arxiv.org/abs/2008.00545) (submitted 2020-08-02; accessed 2026-09-26) found strong same-domain but severe read/broadcast cross-domain degradation for related Slavic languages. It supports keeping a natural-domain outer control separate from voice attribution. It does not isolate TTS provider effects and is not Indic evidence.

## Web and Hugging Face resources for a controlled audit

### A practical Hindi–English same-voice control

Microsoft's [September 2024 Azure Speech release notes](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/releasenotes#september-2024-release) (accessed 2026-09-26) state that five generally available voice families support both `en-IN` and `hi-IN`:

- `Aarav` (male): `en-IN-AaravNeural`, `hi-IN-AaravNeural`
- `Ananya` (female): `en-IN-AnanyaNeural`, `hi-IN-AnanyaNeural`
- `Kavya` (female): `en-IN-KavyaNeural`, `hi-IN-KavyaNeural`
- `Kunal` (male): `en-IN-KunalNeural`, `hi-IN-KunalNeural`
- `Rehaan` (male): `en-IN-RehaanNeural`, `hi-IN-RehaanNeural`

This is the lowest-cost reviewed way to break the headline pair's deterministic `voice → language` association. Two cautions are material:

1. Same base name plus Microsoft's cross-locale support statement is not acoustic proof that each pair preserves identical identity. Validate it.
2. The repository calls an unofficial consumer endpoint through `edge-tts`, whereas the documentation describes Azure Speech. A local `edge-tts --list-voices` check failed because `speech.platform.bing.com` could not resolve during this iteration. Endpoint availability is therefore **unverified**, not absent.

### Hub audit

| Hub resource | Relevant facts as checked 2026-09-26 | Use here |
|---|---|---|
| [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | Apache-2.0; ECAPA-TDNN speaker embeddings; trained on English VoxCeleb1+2; 16 kHz; pinned revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`. | Descriptive validation of whether same-named `en-IN`/`hi-IN` outputs are closer than different-name pairs. Its English human-speech calibration does not establish identity for cross-language synthetic voices, so use ranks/AUC and raw cosines, not a universal threshold. |
| [`sarvamai/sarvam-dub-benchmark-set`](https://huggingface.co/datasets/sarvamai/sarvam-dub-benchmark-set) | Evaluation-only card; 64 reference speakers × 11 target languages = 704 rows; includes all seven project languages; one-shot conditioning; pinned revision `ad489da29596f95ae527c7947fa123e75a4d7a0a`. The schema contains reference audio, target text, and target language—not synthesized target audio. Hub license is `other`. | A possible prompt scaffold for a separately licensed cross-lingual voice-cloning system, not a ready-to-score LID corpus. Do not download into the main benchmark or generate derivatives until rights and output terms are reviewed. |
| [`badrex/mms-300m-arabic-dialect-identifier`](https://huggingface.co/badrex/mms-300m-arabic-dialect-identifier) | Released 300M-model artifact associated with the controlled study; CC BY 4.0; Arabic dialects only. | Reproducible external evidence that the intervention exists; not a teacher or evaluator for the project's labels. |

This dated, non-exhaustive Hub review found no ready-to-score corpus that simultaneously provides the same speakers across all seven required languages, synthesized target waveforms, and an unambiguous permissive license. Absence from this search is **not proof that none exists**.

## Recommended frozen-checkpoint protocol

### Stage 0 — establish whether the model fit its training audio

Without retraining, score all 70 monolingual training clips, separated into gTTS and Edge, with exactly the held-out evaluator. Report known-label clip accuracy, frame-micro/macro accuracy, student↔teacher agreement, and teacher known-label accuracy. Also publish per-language predictions and 0.5/1/2/4 s prefix results.

Interpretation:

- Low exact-train performance: investigate optimisation, capacity, target alignment, and evaluation first. “Memorisation” is unsupported.
- High exact-train performance plus poor held-out performance: a generalisation gap is established, but its axis still needs Stages 1–2.

### Stage 1 — Edge-only `voice familiarity × text familiarity`

Use three already-seen Edge-training scripts and the three held-out scripts per language. Render every script through both that language's seen Edge voice and unseen Edge voice, giving `7 languages × 6 scripts × 2 voices = 84` fresh clips. This is a fully crossed evaluation:

| Cell | Voice | Text |
|---|---|---|
| A | Seen Edge | Seen during training |
| B | Seen Edge | Held-out |
| C | Unseen Edge | Seen during training |
| D | Unseen Edge | Held-out |

Keep the student and operating policy frozen. Generate all four cells in one versioned acquisition job, hash every waveform, record request date/tool versions, and also retain the original cached A/D clips as an exact-sample ceiling/regression check. Score the frozen teacher on every new clip before interpreting student deltas.

Primary contrasts:

- `C − A`, paired by script: voice/profile effect on seen text.
- `D − B`, paired by script: voice/profile effect on unseen text.
- `B − A` and `D − C`: text-familiarity effects within a voice.
- Difference of differences: whether voice and text shifts interact.

Report clip-macro accuracy/F1 first, plus per-language raw counts, frame metrics, prefixes, and teacher agreement. Bootstrap whole script pairs within language; never treat adjacent frames as independent samples. With seven languages and only two voice profiles each, intervals are descriptive and every per-language result must remain visible.

A proposed, **unvalidated** diagnostic gate is a ≥10 percentage-point macro clip-accuracy loss under unseen versus seen voice at fixed text, with the same direction in at least five of seven languages and a teacher loss <2 points. Passing it is evidence of a voice/profile shortcut, not human-speaker bias in deployment.

### Stage 2 — renderer-family sensitivity, with honest naming

Render the same new scripts through gTTS and Edge and score the frozen checkpoint. Split by language and prefix duration. Optionally train a balanced linear probe on pooled student hidden states to predict `gTTS` versus `Edge`, evaluating on a language held out from probe training.

- A large counterfactual prediction change shows sensitivity to the combined renderer/voice family.
- A successful probe shows that provider-family information is encoded.
- Neither establishes that provider information caused LID decisions, because the speaker/voice is not constant across providers and decodability is not causal use.

Do not close the “provider shortcut” question without either the same reference voice rendered by two providers or another intervention that holds identity/content fixed. The Sarvam scaffold above does not supply those rendered clips and has unresolved terms.

### Stage 3 — Hindi–English crossed-voice ablation

First score the frozen checkpoint on matched Hindi and English sentence banks rendered with all five Microsoft bilingual voice families. Verify same-name cross-locale identity descriptively by comparing same-name versus different-name cross-language cosine similarities from the pinned SpeechBrain speaker model; report rank/AUC and manual-listening notes, not a hard “same speaker” threshold.

Only if Stage 1 indicates a material voice/profile effect, run a matched-count retraining ablation:

1. In each of five folds, hold out one bilingual voice family entirely.
2. With the other four voices, create a **nested** arm in which two voice families render only Hindi and two render only English.
3. Create a **crossed** arm using the identical texts, voices, clip count, teacher targets, seed, steps, and other five-language data, but rotate assignments so every training voice renders both Hindi and English.
4. Evaluate both arms on both languages from the held-out fifth voice; rotate through all five held-out voices. Freeze all natural/external tests before training.

The causal comparison is crossed minus nested, not crossed versus the current run. Expected direction is better Hindi–English unseen-voice transfer in the crossed arm; numerical gain is **unverified**. A proposed success gate is ≥10 points higher held-out Hindi–English macro clip accuracy averaged over the five voice folds, teacher accuracy within 2 points, and ≤2-point macro regression on the unchanged other-language external control.

## Attribution decision table

| Observed pattern | Most supported next diagnosis |
|---|---|
| Exact train is also poor | Underfit/optimisation/target mismatch before memorisation. |
| Exact train high; seen voice + unseen text high; unseen voice + seen text low | Voice/profile shortcut is supported. |
| Exact train high; seen voice + unseen text low regardless of voice | Exact-audio or text/template memorisation is supported. |
| Edge factorial is stable; matched-script gTTS↔Edge shifts predictions | Renderer/voice-family sensitivity; provider causality remains unresolved. |
| Teacher drops with the student in a condition | Teacher/domain problem must be separated from student generalisation. |
| Synthetic controls pass but frozen natural controls fail | Broader synthetic-to-natural/domain mismatch. |
| Crossed Hindi–English training beats matched nested training on held-out voices | Direct evidence that breaking the voice-label association helps this student. |

## Limitations and unverified items

- The strongest interventions above come from dialect ID, not seven-way Indic language ID or streaming switch detection.
- Current “male” and “female” metadata is documented for named Microsoft voices. gTTS exposes no comparable gender metadata; none is inferred here.
- The Edge-only factorial changes a complete voice/profile, including documented sex, and therefore cannot isolate timbre from sex, prosody, or provider-internal model differences.
- Five bilingual voice families are enough for a controlled diagnostic but too few for a population claim. Publish fold-level results.
- TTS services can change server-side while retaining a voice name. Content hashes and acquisition dates are part of the experimental condition.
- The Microsoft bilingual voices' `edge-tts` availability and their acoustic identity preservation across locales remain **unverified**.
- Natural generalisation still requires the frozen FLEURS/Svarah/HiACC controls already proposed in [`indic_codeswitch_evaluation_datasets.md`](indic_codeswitch_evaluation_datasets.md).

