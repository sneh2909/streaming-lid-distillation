# Crossed multilingual voice-profile diversification after the factorial

**Topic:** backlog METHOD-23: diversify voice profiles in parallel with checkpoint/target diagnosis, then rerun the factorial  
**Evidence cutoff / access date:** 2026-09-26  
**Scope:** one corpus-design question. This is not a TTS quality benchmark and no student was trained.

## Bottom line

The completed factorial establishes a large **synthetic profile-bundle effect**, not a complete explanation of failure. Full-clip student accuracy was `28/42 = 66.67%` on seen profiles and `7/42 = 16.67%` on unseen profiles, a 50-point paired gap. Yet exact familiar training controls reached only `15/21 = 71.43%`; English and Marathi recall were both `0/3`. Voice diversification should therefore proceed **in parallel** with checkpoint/target/class-collapse diagnosis, not replace it.

Merely adding more locale-specific voices does not solve the design defect. In the 70 current monolingual training clips, each of 14 exact voice IDs occurs under exactly one label. A read-only reconstruction gives

```text
H(language)             = log2(7) = 2.807354922 bits
H(language | voice_id)  = 0
I(language; voice_id)   = 2.807354922 bits
```

This is the maximum possible exact voice-ID association with language. The correct intervention is a **crossed voice × language corpus** in which every retained voice family speaks every label. A fully balanced crossed pool has `I(language; voice_family)=0`; adding 56 such rows uniformly to the current 70 reduces the manifest-level association only to 1.5596 bits (44.44%), while a 50:50 sampler gives 1.4037 bits. These are design arithmetic, not acoustic-independence guarantees.

The best first route is an eight-voice, seven-language matched `label-confounded vs fully crossed` experiment using one generator and identical voice, language, intent, duration, training, and evaluation marginals. The current Microsoft Edge endpoint exposes enough `MultilingualNeural` names to preflight this design, but language quality, same-persona preservation, service revision, and output rights are **not yet verified** for the proposed corpus. Use the paid Azure TTS endpoint if distributable/commercial-use rights matter; Microsoft's current product terms grant output use rights for prebuilt neural voices specifically to paid-tier TTS customers.

Expected direction: crossed training should improve unseen-profile language macro-F1 and reduce voice-triggered flips. The numerical gain, natural-speech transfer, and switch-lag effect are all **unverified**. Treat `>=10` macro-F1 points over the matched confounded arm as an adoption threshold, not as a forecast.

## What the completed factorial now establishes

This audit is bound to:

- [`experiments/voice-text-factorial/results.json`](../experiments/voice-text-factorial/results.json), SHA-256 `f12b5045dfc3816affcf837f9948cdab8321d953b68d6b1d301f8cfb89daaccb`;
- [`experiments/voice-text-factorial/REPORT.md`](../experiments/voice-text-factorial/REPORT.md), SHA-256 `55bac1bb5261c23d7134dcb9eae935fb66ce0b8c59fca84dc92029512c685c54`;
- `data/generated/manifest.jsonl`, SHA-256 `7df6fc5d94db4c88cac9df6b2de8bc9dc5fcacf1bf36853f839bf494ad59314`.

| Observation | Supported conclusion | Unsupported conclusion |
|---|---|---|
| Seen/unseen profile accuracy is 66.67%/16.67%; macro-F1 is 59.22%/9.88%. | The frozen student is strongly sensitive to these particular Edge profile bundles. | Voice familiarity is the sole or sufficient cause of failure. |
| With held-out text fixed, seen/unseen profile accuracy is 61.90%/19.05%. | The profile contrast is much larger than the within-profile text contrast. | The effect is human speaker identity rather than voice, documented sex, prosody, or renderer-internal differences. |
| Five of seven languages favor seen profiles; Marathi reverses by 83.33 points and English is wrong under both. | The aggregate effect is real but heterogeneous. | One global correction will necessarily help every class. |
| Exact familiar controls reach 71.43%; English and Marathi are `0/3`. | Optimisation, target, snapshot selection, or class collapse remains independently material. | A voice-only retrain can be promoted if its relative gain comes from a still-collapsed baseline. |
| Fresh seen/seen rerenders have zero prediction flips and the same 71.43% accuracy. | Provider drift does not explain the weak familiar ceiling in this acquisition. | The remote renderer is revision-pinned or generally stable. |

The reviewer correctly separates two claims: the 50-point profile effect is the largest measured factor, while failure of the 80% familiar-audio gate shows that profile familiarity is insufficient. Neither negates the other.

## External evidence: crossing labels is the intervention

### Controlled speech-classification evidence

[Abdullah et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/abdullah25_interspeech.html) (published 2025; accessed 2026-09-26) used voice conversion to mitigate speaker bias in Arabic dialect identification and reported cross-domain gains up to 34.1%. Their detailed controlled table, reviewed in the earlier METHOD-7 memo, showed good performance when the same converted voices covered every dialect and collapse when target voices were dialect-disjoint. The transferable point is the **crossed assignment**, not the reported magnitude: their task, model, data volume, and natural-speech domains differ from this tiny seven-language causal student.

[Kuparinen, VarDial 2026](https://aclanthology.org/2026.vardial-1.3/) (published March 2026; accessed 2026-09-26) independently found large speaker-dependent versus speaker-independent dialect-ID effects, but pitch/noise/voice-conversion modifications did not give major consistent gains. Together these studies justify measuring a crossed intervention; they do not guarantee it will work here.

The TTS literature exposes the same confound from the generator side. [Zhang et al., Interspeech 2019](https://research.google/pubs/learning-to-speak-fluently-in-a-foreign-language-multilingual-speech-synthesis-and-cross-language-voice-cloning/) (published 2019; accessed 2026-09-26) explicitly describes speaker identity as perfectly correlated with language in its multilingual training data and uses adversarial disentanglement to enable cross-language voices. This supports tracking the assignment structure, not assuming that a multilingual generator automatically disentangles it.

### Same named or conditioned voice is not proof of same identity

[Ahtasam et al., IWSLT 2026](https://aclanthology.org/2026.iwslt-1.12/) (published July 2026; accessed 2026-09-26) systematically evaluated four zero-shot cross-lingual voice-cloning systems and found a language-dependent trade-off between content consistency and speaker identity. It is reviewed evidence that reference conditioning alone does not guarantee either property.

The 2026 [LASE preprint](https://arxiv.org/abs/2605.00777) (submitted 2026-05-01; accessed 2026-09-26) reports that ordinary ECAPA and WavLM speaker embeddings themselves shift across English, Hindi, Telugu, and Tamil scripts, including on synthetic same-voice pairs. It is **unreviewed**, and its claimed released checkpoint was not found in a non-exhaustive Hugging Face search on 2026-09-26. Its useful warning is methodological: ECAPA cosine similarity cannot certify that two cross-script samples are the same identity.

For this project, speaker-embedding scores may be descriptive diagnostics only. Report within-family cross-language retrieval against different-family controls, language-balanced rank/AUC, and both ECAPA and WavLM results. Never convert one cosine threshold into `identity_verified=true`, and retain blinded listening notes from speakers of the target languages.

## Current Microsoft feasibility audit

### Official Azure catalog

Microsoft's live [Azure Speech language and voice table](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts) (accessed 2026-09-26) lists seven base names under both `en-IN` and `hi-IN`:

| Family | English ID | Hindi ID | Catalog metadata |
|---|---|---|---|
| Aarav | `en-IN-AaravNeural` | `hi-IN-AaravNeural` | male |
| Aarti | `en-IN-AartiNeural` | `hi-IN-AartiNeural` | female |
| Arjun | `en-IN-ArjunNeural` | `hi-IN-ArjunNeural` | male |
| Ananya | `en-IN-AnanyaNeural` | `hi-IN-AnanyaNeural` | female |
| Kavya | `en-IN-KavyaNeural` | `hi-IN-KavyaNeural` | female |
| Kunal | `en-IN-KunalNeural` | `hi-IN-KunalNeural` | male |
| Rehaan | `en-IN-RehaanNeural` | `hi-IN-RehaanNeural` | male |

This expands the five-family September-2024 list cited in the earlier memo. The current catalog is evidence that the IDs exist in Azure, not that paired locale IDs are acoustically identical.

The same documentation says names containing `MultilingualNeural`, `DragonHDLatestNeural`, or `DragonHDOmniLatestNeural` support multiple languages and can auto-detect input text. Its [REST voice-list API](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech#get-a-list-of-voices) exposes `SecondaryLocaleList`, but requires an Azure key or bearer token. Query and preserve that response for the exact deployment region before synthesis; do not infer complete seven-language support from the `Multilingual` name alone.

### Endpoint actually used by this repository

The repository uses [`rany2/edge-tts`](https://github.com/rany2/edge-tts), which calls Microsoft Edge's online service rather than a pinned local model. A successful live query on 2026-09-26 at 13:27 IST with installed `edge-tts==7.2.8` found 12 `MultilingualNeural` IDs:

```text
de-DE-FlorianMultilingualNeural
de-DE-SeraphinaMultilingualNeural
en-AU-WilliamMultilingualNeural
en-US-AndrewMultilingualNeural
en-US-AvaMultilingualNeural
en-US-BrianMultilingualNeural
en-US-EmmaMultilingualNeural
fr-FR-RemyMultilingualNeural
fr-FR-VivienneMultilingualNeural
it-IT-GiuseppeMultilingualNeural
ko-KR-HyunsuMultilingualNeural
pt-BR-ThalitaMultilingualNeural
```

The same endpoint exposed only `en-IN-NeerjaNeural`, `en-IN-PrabhatNeural`, `hi-IN-SwaraNeural`, and `hi-IN-MadhurNeural`; none of the seven same-name Hindi/English families above appeared. Therefore:

- the 12 multilingual IDs are the lowest-friction all-seven preflight candidates;
- the seven Indian paired families require official Azure access or a later endpoint change;
- availability is not language-quality evidence, and the service exposes no immutable model revision.

The distinction also matters legally. Current [Microsoft Product Terms for Azure](https://www.microsoft.com/licensing/terms/en-US/productoffering/MicrosoftAzure/EAEAS) (accessed 2026-09-26) say paid-tier TTS customers may use prebuilt-neural-voice output, including commercially. That grant is explicitly scoped to the paid TTS tier. The Edge consumer endpoint's redistribution/training rights are **unverified**; an open-source client license does not license Microsoft-generated audio. Until reviewed, keep Edge-generated candidates ignored/internal and do not publish them as a redistributable corpus.

## Hugging Face Hub alternatives

Live Hub API/card checks were performed on 2026-09-26. Revisions below are full commits, not mutable `main` aliases.

| Hub ID and pinned revision | Coverage / conditioning | Size and license | Fitness for this experiment |
|---|---|---|---|
| [`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts/tree/7b527af5ee8ed1f9a28d80b19703ed9bb8ba10ca) | All seven project languages; caption-conditioned; card lists 69 named voices but language-specific recommended sets | 937,803,241 parameters, F32; Apache-2.0; gated; last modified 2025-09-24 | Good renderer-diversity challenger, but a repeated generic caption or same spelling is not a controlled cross-language identity. CPU generation cost is unmeasured and likely high. |
| [`ai4bharat/IndicF5`](https://huggingface.co/ai4bharat/IndicF5/tree/ba85abedf18dc479a447eaa0eccbd76ab78a47d5) | Reference-audio conditioning; Bengali, Gujarati, Hindi, Marathi, Tamil, Telugu; **no English** | 350,681,834 parameters; MIT; gated; custom code; last modified 2026-03-03 | Plausible six-Indic crossed auxiliary arm, not a complete headline Hindi-English generator. Cross-language identity and CPU cost are **unverified**. |
| [`BosonLab/chatterbox-desi`](https://huggingface.co/BosonLab/chatterbox-desi/tree/c3556b021a89428c945b846ea0e062d8b0153cf9) | Reference-audio zero-shot examples reuse the same two prompts across Bengali, Hindi, Marathi, Gujarati, Tamil, and Telugu; no official English support | MIT; gated; 3.22 GB repository storage; community upload requiring a source patch; last modified 2026-04-05 | Strong six-language procedural prototype, but no reviewed cross-language identity/intelligibility metrics and not a complete seven-label arm. |
| [`bharatgenai/sooktam2`](https://huggingface.co/bharatgenai/sooktam2/tree/eb270c0ceebd58e0e513820eb71c637e9f486a7b) | Reference-guided voice conditioning; exact seven-label coverage plus other Indian languages | 6.72 GB repository storage; custom code; card body says BharatGen non-commercial license while Hub license metadata is empty; last modified 2026-03-16 | Technically closest all-seven open artifact, but non-commercial terms block it from a reusable commercial corpus. Quality and identity claims are self-reported/unverified. |
| [`sarvamai/sarvam-dub-benchmark-set`](https://huggingface.co/datasets/sarvamai/sarvam-dub-benchmark-set/tree/ad489da29596f95ae527c7947fa123e75a4d7a0a) | 64 reference speakers × 11 target languages, including all seven; 704 prompt rows; no synthesized target output | Dataset license metadata `other`; evaluation-only; last modified 2026-02-06 | Useful design precedent and possible reference scaffold only after rights review. It is not a ready crossed audio corpus and must not be used for training as-is. |

Model-code licenses do not automatically grant rights to a cloned person's likeness, reference audio, training data, or generated output. Every reference-conditioned route requires an explicit reference-audio license/consent field. No Hub artifact above is recommended for integration before that gate.

[`ai4bharat/indicvoices_r`](https://huggingface.co/datasets/ai4bharat/indicvoices_r/tree/5f4495c91d500742a58d1be2ab07d77f73c0acf8) is a separate natural-data option: its card reports 10,496 speakers, 1,704 hours, speaker/demographic metadata, and CC BY 4.0. It provides valuable speaker diversity but excludes English in current Hub language metadata and does not establish that the same people span labels. Adding monolingual speakers lowers memorisation pressure; it does not make `voice -> language` non-predictive. Use it as an external/natural companion, not as the crossed intervention.

## Proposed matched corpus experiment

### Stage 0 — preflight voice-language cells before training

Start with 12 endpoint-listed multilingual voices, all seven languages, and two short semantically parallel customer-service intents per language: `12 × 7 × 2 = 168` candidate clips. This is a preflight, not the training set.

For every voice-language cell, record:

- provider, endpoint, exact voice ID, proposed `voice_family`, package/API version, request UTC time, all synthesis options, text/intent ID, raw response and decoded-WAV hashes;
- target script, sample rate, sample count, duration, peak/clipping, integrated loudness, leading/trailing silence, and decoding recipe;
- frozen ECAPA native 107-way top label/probability, seven-language retained mass, conditional top label, and teacher revision/artifact hash;
- a blinded native-listener intelligibility/language check on at least one clip per cell;
- ECAPA and WavLM within-family cross-language versus between-family retrieval scores, marked diagnostic rather than identity proof;
- provider terms and a machine-readable `training_allowed`, `redistribution_allowed`, and `commercial_allowed` decision.

Reject a whole voice-language cell rather than silently selecting easy utterances if the language is unintelligible, the teacher systematically leaves the expected label, or rights are unresolved. Predeclare the exact quality threshold after a small calibration slice; no threshold is validated yet. Preserve all failures so the selection cannot be retrospectively tuned to the LID model.

### Stage 1 — exact `confounded vs crossed` intervention

If at least eight voice families pass all seven languages, construct eight voice-family-held-out folds. In each fold, reserve one family for evaluation and use seven for training. Prepare 14 semantically parallel intent IDs per language.

For the seven training voices, build two 98-clip arms with identical marginals:

| Arm | Assignment | Count |
|---|---|---:|
| `label_confounded` | Assign one voice to each language; that voice renders all 14 intent IDs for its language. Rotate the mapping across folds. | `7 voices × 14 = 98` |
| `fully_crossed` | Every voice renders two intent IDs in every language; use a balanced rotation so each language still contains all 14 intents. | `7 voices × 7 languages × 2 = 98` |

Both arms therefore have 14 clips per language, 14 clips per voice, the same intent set, provider, synthesis batch, duration bins, teacher, initialization, update count, batch order, optimizer, targets, delay, and checkpoint selector. The intended difference is `I(language; voice_family)=log2(7)` versus zero at the manifest level. Assert those quantities from integer contingency tables before training.

Train with the completed METHOD-4 trajectory selector rather than comparing arbitrary final iterates. Publish step-zero, exact-training per-class, held-out-family, and natural external metrics. Keep the original 84-clip factorial as a regression diagnostic, but it is now repeatedly consulted development data and cannot be the final test.

Primary adoption gate, proposed and **unvalidated**:

1. `fully_crossed - label_confounded >= 10` percentage points in held-out-family seven-language macro-F1 averaged across folds;
2. nonnegative accuracy change in at least five of seven languages, with no language at zero exact-training recall;
3. teacher native-space quality within two points across arms;
4. no more than two macro-F1 points lost on frozen natural FLEURS/Svarah controls;
5. identical requested/completed updates and identity-bound corpus/target/checkpoint evidence.

Expected direction is positive because the current profile effect is large. The 10-point value is a usefulness threshold, **not a predicted effect size**. If both arms remain below the familiar-audio fit gate or retain zero-recall classes, record the corpus result as non-evaluable for integration and continue the target/model diagnosis.

### Stage 2 — make switch fixtures same-voice rather than voice-assisted

Only after a crossed monolingual arm clears its gate, use four passing multilingual families to create source-disjoint Hindi↔English boundaries in both directions. Cross:

- language change versus same-language control;
- voice held constant versus voice changed;
- two pre-boundary and two post-boundary duration bins.

`4 voices × 2 directions × 2 pre bins × 2 post bins = 32` language-change boundaries; add 32 matched same-language/voice-change controls for a 64-boundary factorial. Store source-speech end, array join, target-speech onset, gap/overlap, and voice family at sample resolution. This directly operationalizes METHOD-40.

Expected result (**unverified**): same-voice training/evaluation should reduce boundary responses driven by synthetic voice joins and may lower false switch/flop counts. It is not expected to beat the teacher's own transition floor automatically; switch-recall and lag gains are unverified and must use the chronology-safe scorer.

## Why not use pitch shifts, speaker embeddings, or “more speakers” as substitutes?

- Pitch/rate perturbations create correlated variants of one renderer, not independent voice identities. VarDial 2026 found such modifications inconsistent.
- A speaker embedding can demonstrate separability or retrieval behavior, but cross-language embedding drift means it cannot certify identity preservation.
- Adding many speakers that each occur under one language keeps exact voice identity perfectly predictive of label in the training table. It can regularize memorisation, but it does not identify the benefit of breaking the correlation.
- A new TTS provider changes renderer, codec, prosody, voice, and often licensing together. Treat it as domain diversification, not a voice-only causal intervention.
- Synthetic success does not establish natural-call transfer. Natural speaker-disjoint FLEURS/Svarah/IndicVoices and the planned real-call benchmark remain mandatory outer controls.

## Concrete artifact contract

Each generated record should add:

```text
generator_id / endpoint / artifact_revision_or_unavailable_reason
voice_endpoint_id
voice_family_claim
voice_family_evidence = provider_name | reference_audio_hash | unverified
reference_audio_license_and_consent (if applicable)
language / locale / script / parallel_intent_id
provider_terms_snapshot_date
training_allowed / redistribution_allowed / commercial_allowed
raw_response_sha256 / wav_sha256 / exact synthesis recipe
native_teacher_top1 / probability / retained_selected_mass
cross_language_identity_diagnostic_version (never a boolean proof)
```

The run identity must bind the complete voice × language contingency table and its empirical `I(language; voice_family)`. A publication check should fail if a supposedly crossed voice lacks a language cell, if split folds share a voice family/reference prompt, if rights are unresolved, or if a server-generated waveform is replaced under the same voice name.

## Limitations and unverified items

- No candidate TTS model or voice was synthesized or scored in this iteration; only the live Edge voice list was queried. Seven-language intelligibility and persona consistency remain **unverified**.
- Microsoft's official Azure catalog and the Edge consumer endpoint expose different inventories. They are not interchangeable products or licensing regimes.
- The profile effect is measured on one checkpoint and particular male-seen/female-unseen Edge bundles. It is not a human-speaker population estimate.
- The proposed mutual-information values describe manifest identifiers. Acoustic identity may drift within a name or remain similar across names.
- All Hub quality and coverage statements are card metadata unless a reviewed paper is linked. Chatterbox-Desi and Sooktam-2 have no independent LID-oriented validation identified here.
- Reference-guided voice cloning raises consent, likeness, and redistribution questions independent of code/model licenses.
- The best matched experiment is larger than the take-home's current 70 monolingual clips. Its value is causal diagnosis; no claim is made that 98 synthetic clips are sufficient for production accuracy.

## Recommendation

Proceed with a small **preflight**, not an immediate main-corpus rewrite:

1. snapshot the 12 live multilingual voice IDs and synthesize the 168-cell quality/rights matrix;
2. if eight all-language families pass, run the 98-vs-98 crossed assignment experiment with held-out voice folds and the fixed checkpoint selector;
3. integrate only after the absolute fit/natural-transfer gates pass, then build same-voice Hindi↔English switches and voice-change controls.

This directly tests whether breaking voice-label association helps while preserving the separate evidence that the current model also has optimisation/target/class-collapse problems.
