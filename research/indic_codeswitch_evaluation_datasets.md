# Hindi–English code-switch and Indic speech datasets for evaluation

Research cutoff: **2026-09-26**. All linked pages and Hugging Face cards were checked on that date. A year/date beside a source is its publication or release date, not the access date.

## Bottom line

There is no single reviewed corpus that simultaneously provides:

- natural Hindi–English switching;
- many speaker-disjoint test speakers;
- human time-aligned language regions suitable for switch-latency scoring;
- 8 kHz telephone audio; and
- a clear license permitting commercial evaluation and redistribution.

The best defensible evaluation is therefore layered:

1. **Natural, speaker-disjoint Hinglish:** use **HiACC test splits** as the primary permissively licensed code-switch set. It has adult and child speech, speaker IDs, and token-level Hindi/English labels, but not token times.
2. **True boundary timing:** request the **IIT Bombay DAPLab TextGrid corpus**. It is the only reviewed Hindi–English resource with human time-aligned language regions, but it contains only two public-figure speakers and is non-commercial research only.
3. **Broader natural Hinglish:** use **Vaani Benchmark V1.0** and **MUCS test** as complementary domains. Vaani supplies 1,103 geographically diverse speakers and marks English tokens; MUCS supplies technical/tutorial speech. Neither exposes word-level switch times.
4. **Seven-language monolingual controls:** use **FLEURS test** for `en hi mr bn ta te gu`, then **Svarah** for Indian-accented English and **Kathbath test_unknown** for unseen-speaker Indic confusions.
5. **Telephone stress:** use the **GramVaani 3 h Hindi eval** for ecological 8 kHz Hindi, and paired 8 kHz/codec transforms of the same FLEURS/Svarah utterances to isolate channel degradation.

CoSHE-Eval is a useful conversational stress set but is non-commercial. CoSTA's small IndicVoices-derived subsets are attractive, but the repository does not declare a license for the released annotations and podcast material. Common Voice remains useful monolingual data, but its dedicated code-switch collection still has no release.

## Code-switch datasets

| Corpus | Speech and coverage | Language-time ground truth | Split / speaker evidence | License and access | Recommended use |
|---|---|---|---|---|---|
| **HiACC** (2025) | 5.24 h, 16 kHz mono; read and spontaneous Hinglish from 24 adults and 20 children; the paper reports 3,318 adult and 1,858 child utterances | Manual transcript plus token-level Hindi/English labels in `code_switched_labels.json`; **no token/word timestamps documented** | Train/validation/test directories; the primary paper says the split is speaker-independent | **CC BY 4.0**; canonical [Zenodo record 15551669](https://zenodo.org/records/15551669); paper: [Singh et al., 2025-07-17](https://doi.org/10.1016/j.dib.2025.111886). Community test mirrors: [`Trelis/hiacc-adult-test-eval`](https://huggingface.co/datasets/Trelis/hiacc-adult-test-eval), [`Trelis/hiacc-child-test-eval`](https://huggingface.co/datasets/Trelis/hiacc-child-test-eval) | **Primary natural Hinglish utterance/frame-accuracy source after alignment.** Keep adult and child results separate. Audit speaker IDs before accepting the claimed split |
| **IIT Bombay DAPLab Hindi–English** (Interspeech 2018) | Seven 7–15 min YouTube recordings from two speakers: BK Shivani and Alia Bhatt; natural lecture/interview discourse | Two-tier Praat TextGrids are aligned to audio: `H`, `H(E)`, `E`, `E(H)`, and long silence `S`; tier 2 collapses these to Hindi/English. Running-text transcripts themselves are not time aligned | Only two speakers and long recordings; no standard train/dev/test | **Non-commercial research only**; email request to the address on the [official IITB page](https://www.ee.iitb.ac.in/student/~daplab/datasets/code_switch.html) (2018). The page points to the source videos and does not promise redistributable audio | **Primary human boundary/latency benchmark**, but report speaker-macro results and never present it as population accuracy |
| **Vaani Benchmark V1.0** (2026) | 5,050 image-prompted Hindi segments, about 10.9 h, 1,103 speakers, 104 districts in 16 states; natural Hindi with code-switching; variable sample rate and three independent transcripts | `{EnglishWord}` marks English tokens, but there are **no word times or frame labels** | Test-only; explicit `speakerID`, state, district, duration, and gender | **CC BY 4.0**, auto-gated contact sharing; HF [`ARTPARK-IISc/Vaani-Benchmark-V1.0`](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Benchmark-V1.0) (2026) | **Primary broad adult natural-speech check.** Select one or a bounded number of clips per speaker with genuine English spans; manually annotate a boundary subset |
| **MUCS 2021 / OpenSLR SLR104** | Hindi–English: 89.86 h train / 5.18 h test; Bengali–English: 46.11 h / 7.02 h; 16 kHz, 16-bit technical tutorials | Kaldi `segments` gives sentence timestamps, **not language-token or switch boundaries** | Official train/test and blind test; speaker-disjointness is not documented on the data page and must be audited | **CC BY-SA 4.0**; direct tarballs from [OpenSLR SLR104](https://www.openslr.org/104/) (2021). Community HF [`dianavdavidson/MUCS-Hinglish`](https://huggingface.co/datasets/dianavdavidson/MUCS-Hinglish) (2026) has 55,961 rows and speaker/word-ratio fields | **Secondary domain check** for technical English insertions. Prefer official tarballs and test split; the community mirror's `cc-by-4.0` tag conflicts with the official share-alike license, so CC BY-SA is controlling |
| **CoSHE-Eval** (Hub card checked 2026-09-26) | 1,985 natural conversational Hindi–English clips, about 30 h; 0.6–59.8 s, with a reported 53.3 s mean | Verified mixed-script transcripts; only `audio_file_name`, `transcription`, and `audio` are exposed. “Timestamp validation” describes segment/audio consistency, not word-language timing | One `eval` split; no speaker ID on the card, so speaker independence and repeated speakers are **unverified** | **CC BY-NC 4.0**; HF [`soketlabs/CoSHE-Eval`](https://huggingface.co/datasets/soketlabs/CoSHE-Eval) | Strong **non-commercial conversational stress set**. Filter truly mixed clips rather than trusting every row to contain both languages |
| **IndicVoices + CoSTA release** (2024/2025) | IndicVoices now reports 23.7K h across 22 scheduled languages, 51K speakers and 400+ districts; 76% extempore and 15% conversational. CoSTA selects high-CMI English-mixed subsets: hi 2.5 h/728 clips/85 speakers, bn 2.1 h/624/124, mr 2.2 h/575/110, te 2.3 h/587/102 | Transcripts and CMI selection only; the released CoSTA spreadsheets do **not** contain word times | IndicVoices exposes `speaker_id`; Hub has `train` and `valid`, but speaker disjointness of those splits is **not documented**. CoSTA is evaluation-only by construction | Base [`ai4bharat/IndicVoices`](https://huggingface.co/datasets/ai4bharat/IndicVoices) is gated **CC BY 4.0**. [CoSTA paper](https://aclanthology.org/2025.coling-main.618/) (COLING 2025) and [release repo](https://github.com/csalt-research/CoSTA) provide annotations/audio links, but the repo has **no declared license** as checked | Mine natural switches from IndicVoices only after checking speaker overlap. Do **not** treat CoSTA's added annotations or podcast audio as commercially reusable until the authors clarify their license |
| **Hi–En IITG-HingCoS** (2018/2019) | 25 h, 9,251 code-switched sentences, realistic mobile/landline recordings; papers report 101 speakers and 8 kHz telephone speech | Utterance transcripts; no public frame/word language timing documented | Published train/dev/test counts exist, but the distributed metadata must be inspected before claiming speaker-disjoint tests | Paid request: INR 50,000 + GST for an Indian organisation / INR 100,000 + GST abroad; research/R&D only, no commercial use or redistribution. [Official corpus page](https://www.iitg.ac.in/eee/emstlab/HingCoS_Database/HingCoS.html) and [request terms](https://www.iitg.ac.in/eee/emstlab/HingCoS_Database/db_copy.php) | Potentially excellent **native 8 kHz Hinglish** evaluation, but cost and restrictive terms make it unsuitable for the default take-home path |
| **liva-ai sample** (Hub, checked 2026-09-26) | Ten 60 s, 48 kHz conversational multi-speaker clips with overlap | Millisecond speaker-turn timecodes and mixed-script transcript; turn intervals are not token-language intervals | One ten-row split; too small for aggregate claims | **CC BY 4.0**; HF [`liva-ai/hindi-english-asr`](https://huggingface.co/datasets/liva-ai/hindi-english-asr) | Qualitative regression set for overlap, disfluency and long-context behavior only |

### HiACC cautions

[The primary HiACC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12329218/) (available online 2025-07-17) is the best open fit found for ordinary Hinglish evaluation, but it needs a manifest audit before use:

- its abstract/methods report 3,318 adult utterances, while Table 2 displays 2,668; recount the downloaded files and publish the exact record version/hash;
- the paper says splits are speaker-independent, while at least one community model card claims overlap; compute the intersection of speaker IDs across train/validation/test rather than inheriting either assertion;
- token language labels do not provide acoustic boundaries. A forced aligner may create a proposal, but manually corrected intervals are required for defensible switch-latency ground truth;
- read and spontaneous, adult and child, and clean/classroom subsets should be reported separately.

### Why the IITB TextGrid set matters

Most ASR corpora call an entire sentence “code-switched.” That can measure whether a model notices both languages somewhere, but it cannot measure whether a streaming decision changed 300 ms or 2 s after the speaker changed language. The [IITB DAPLab annotations](https://www.ee.iitb.ac.in/student/~daplab/datasets/code_switch.html) (2018) directly mark language regions in time and distinguish a matrix language containing insertions (`H(E)` or `E(H)`) from a wholly switched region. For this project, tier 2 supplies a simple Hindi/English reference and tier 1 supports a stricter analysis of single-word borrowing versus sustained switching.

Its limitations are equally important: two celebrity/public speakers, YouTube capture, long discourse, no standard split, non-commercial terms, and fragile dependence on source-video availability. It is a boundary unit test, not a generalization benchmark.

## Monolingual and accent controls

| Corpus | Relevant coverage and channel | Split / temporal labels | License and download | Fit |
|---|---|---|---|---|
| **FLEURS** (2022) | 102 languages at 16 kHz; all project configs: `en_us`, `hi_in`, `mr_in`, `bn_in`, `ta_in`, `te_in`, `gu_in`; read, parallel-domain speech | About 10 h training per language plus validation/test; train speakers differ from validation/test speakers. Utterance labels only | **CC BY 4.0**, ungated; HF [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) | Best compact, balanced **seven-way monolingual** baseline and exact-duration prefix source; not natural code-switch speech |
| **Common Voice Scripted Speech v27.0** (latest 2026-09) | 295 languages and 42,593 h overall. Official v27 metadata contains `en`, `hi`, `bn`, `mr`, `ta`, `te`, but **not `gu`**. Coverage is extremely imbalanced: Telugu has only 98 test clips / 0.94 validated h, versus Tamil's 12,241 test clips / 234.82 validated h | Read speech with client IDs and train/dev/test; locale-level labels, no switch regions. The Code Switching collection remains Alpha with no release | Audio is **CC0 for computational use**, but current platform terms prohibit mirroring/rehosting. Available only through [Mozilla Data Collective](https://community.mozilladatacollective.com/commonvoice-27/) and its API, not current HF. Release metadata: [`cv-corpus-27.0-2026-09-11.json`](https://github.com/common-voice/cv-dataset/blob/main/datasets/scripted-speech/cv-corpus-27.0-2026-09-11.json) | Useful extra speakers and Indian-accent English (`accent=indian`), but balance by speaker/class and obey MDC storage/redistribution terms |
| **Kathbath / IndicSUPERB** (2022) | 1,684 h read speech, 1,218 contributors, 12 languages; covers the six project Indic classes (`hi mr bn ta te gu`) but not English | Official clean `test_known` and `test_unknown`, plus noisy versions; speaker/gender encoded in filenames. HF mirror exposes only train/valid | **License conflict:** the [HF card](https://huggingface.co/datasets/ai4bharat/Kathbath) tags CC BY 4.0, while its body and the [official IndicSUPERB repo](https://github.com/AI4Bharat/IndicSUPERB) say the packaging is CC0 and disclaim ownership of crawled source text. Original test archives are 2 GB clean unknown and 1.4 GB noisy unknown | Excellent unseen-speaker Indic-confusion/noise set. Preserve the original license notice and seek clarification before redistribution; use official `test_unknown`, not HF `valid`, for the cleanest claim |
| **Svarah** (Interspeech 2023) | 9.6 h Indian-accented English, 6,656 test clips, 117 speakers across 65 districts/19 states; read and spontaneous conversational speech; native languages cover 19 scheduled languages | Test-only; utterance-level English label | **CC BY 4.0**, gated HF token; [`ai4bharat/Svarah`](https://huggingface.co/datasets/ai4bharat/Svarah) | Best English-side control for the known failure mode “Hindi accent mistaken for Hindi.” Keep it separate from generic FLEURS English |
| **GramVaani SLR118** (2022) | 100 h labeled train, 5 h dev, 3 h eval and 1,000 h unlabeled spontaneous regional Hindi telephone speech. 60.87% of labeled train/dev and 67.63% of unlabeled files are 8 kHz | Corpus-level Hindi only; no English or switch regions | Free for academic use; **commercial use requires permission** from GramVaani. Direct tarballs at [OpenSLR SLR118](https://www.openslr.org/118/) | Best ecological Hindi telephony stress source; pair with transformed Svarah/FLEURS English for a balanced diagnostic, but do not call the cross-corpus difference a pure codec effect |
| **Full Vaani** (2026) | About 31,278 h image-prompted spontaneous speech, 156K speakers, 165 districts, 105 languages; 2,122 h transcribed | Broad metadata; no documented language boundary timings | **CC BY 4.0**, gated; HF [`ARTPARK-IISc/Vaani`](https://huggingface.co/datasets/ARTPARK-IISc/Vaani) | Valuable future mining pool, but much too large for the take-home. Prefer the fixed benchmark for evaluation |

### Common Voice access changed

The [official v27 announcement](https://community.mozilladatacollective.com/commonvoice-27/) is dated 2026-09-21. The [metadata repository](https://github.com/common-voice/cv-dataset) lists Scripted Speech v27.0, Spontaneous Speech v5.0, and Code Switching as Alpha with no release. Mozilla says current Common Voice files are [exclusively distributed through MDC](https://community.mozilladatacollective.com/faq-can-i-get-the-common-voice-or-other-mdc-datasets-from-other-platforms-like-github-or-hugging-face/) and explains that [CC0 remains the computational-use license while non-mirroring is a platform term](https://community.mozilladatacollective.com/faq-why-cant-i-re-host-or-share-common-voice-datasets-that-i-download-from-mdc/). Old `mozilla-foundation/common_voice_*` Hub cards are not the current download route.

Use an MDC account, accept the dataset terms, then use the browser archive or MDC API credentials. Record the exact release, locale archive checksum, and consent/terms snapshot. Do not put downloaded Common Voice audio into this repository or a new Hub mirror.

## Download recipes

Official Hub datasets can be streamed so the evaluator need not download hundreds of gigabytes:

```python
from datasets import load_dataset

fleurs_hi = load_dataset("google/fleurs", "hi_in", split="test", streaming=True)
svarah_en = load_dataset("ai4bharat/Svarah", split="test", streaming=True)
vaani_hi = load_dataset(
    "ARTPARK-IISc/Vaani-Benchmark-V1.0", "Hindi", split="test", streaming=True
)
indicvoices_hi = load_dataset(
    "ai4bharat/IndicVoices", "hindi", split="valid", streaming=True
)
coshe = load_dataset("soketlabs/CoSHE-Eval", split="eval", streaming=True)
```

Svarah, Vaani and IndicVoices require accepting their Hub gates and authenticating. For a complete IndicVoices language archive, the [official card](https://huggingface.co/datasets/ai4bharat/IndicVoices/blob/main/README.md) says the full-length audio is split across non-overlapping `v1`–`v5` archives; downloading only one version is incomplete.

For non-Hub sources:

- download canonical HiACC bytes from [Zenodo DOI 10.5281/zenodo.15551669](https://doi.org/10.5281/zenodo.15551669), then hash the archive and verify speaker-ID intersections;
- download MUCS train/test tarballs directly from [SLR104](https://www.openslr.org/104/) and retain `LICENSE`/README beside derived manifests;
- download Kathbath's clean/noisy `test_unknown` archives from the [IndicSUPERB repository](https://github.com/AI4Bharat/IndicSUPERB), because the Hub mirror does not expose those official test splits;
- request IITB TextGrids by email through its official page; do not automate-download or redistribute the linked YouTube audio;
- use the Mozilla Data Collective API for Common Voice rather than an obsolete HF loader.

## Proposed evaluation manifest

### A. Seven-way monolingual generalization

Take up to 200 test clips per FLEURS class for `en hi mr bn ta te gu`, stratified by duration, and do not use those clips for distillation or threshold tuning. Add separate rows for:

- Svarah Indian-accented English;
- Kathbath clean `test_unknown` for the six Indic languages;
- Kathbath noisy `test_unknown`;
- GramVaani Hindi eval; and
- paired native/low-pass-8 kHz/μ-law/A-law versions of the *same* FLEURS and Svarah clips.

Always report macro-over-language results. Corpus-level pooled accuracy would otherwise be dominated by whichever source contributed the most clips.

### B. Natural Hindi–English switching

Use the untouched HiACC adult and child test manifests. Report at least four slices: adult/read, adult/spontaneous, child/read, child/spontaneous. Add Vaani Benchmark clips containing brace-marked English tokens, capped per speaker and district, and MUCS official test as a technical-domain slice. CoSHE can be an additional research-only slice.

Transcript script is not reliable boundary truth: named entities, acronyms, borrowings, transliterated Hindi, and English words written in Devanagari all break a simple Unicode-script heuristic. Use the supplied token labels where available, then manually review uncertain/borrowed tokens.

### C. Streaming boundary evaluation

Use IITB tier-2 TextGrids directly. In parallel, select a modest HiACC/Vaani subset with at least 200 sustained Hindi↔English boundaries across at least 50 speakers and annotate word intervals manually. Store:

```text
recording_id, speaker_id, start_s, end_s, language,
switch_type, lexical_item, annotator, confidence, source_version
```

Define `switch_type` at minimum as sustained switch, single-word insertion/borrowing, silence-adjacent, and overlap/uncertain. Double-annotate at least 20% and adjudicate disagreements. Forced alignment may seed intervals but must not be the final ground truth because the ASR/aligner can fail precisely at accented English switches.

### D. Controlled timing sanity checks

Retain exact-boundary synthetic concatenations to test alignment and decoder delay, but label them separately. A speaker or channel change at the join is an easy shortcut. Natural TextGrid/manual-boundary results must be the headline; splices are plumbing tests, not evidence of real code-switch tracking.

## Hub audit: recent community uploads not adopted as benchmarks

- [`addyo07/noisy-hinglish-asr`](https://huggingface.co/datasets/addyo07/noisy-hinglish-asr) (checked 2026-09-26) claims 36.6 h, 28,681 rows and Apache-2.0, but the card exposes a `youtube_harvest` source and does not document per-source audio rights, speaker identities, or switch intervals. An aggregate Apache tag cannot automatically relicense every upstream recording. **Provenance/license unverified; do not use for headline evaluation.**
- [`sonexis-ai/hinglish-code-switched-conversations-v1`](https://huggingface.co/datasets/sonexis-ai/hinglish-code-switched-conversations-v1) (2026 Hub listing) is gated, under CC BY-NC 4.0, and had fewer than 1,000 items but no accessible card statistics during this review. **Size, source provenance, split integrity and temporal labels unverified.**
- [`DataCatalystAI/DataCatalyst_Multilingual_TTS_Sample`](https://huggingface.co/datasets/DataCatalystAI/DataCatalyst_Multilingual_TTS_Sample) v1.2 (2026-04) has only six utterances per Hindi/English/Hinglish group and 31 rows including processed duplicates. Its custom evaluation license forbids commercial use, training and redistribution. It is a playback demo, not a benchmark.
- [`liva-ai/hindi-english-asr`](https://huggingface.co/datasets/liva-ai/hindi-english-asr) is useful precisely because it is small and difficult; its ten minutes cannot support confidence intervals or model selection.
- [`dianavdavidson/MUCS-Hinglish`](https://huggingface.co/datasets/dianavdavidson/MUCS-Hinglish) is a convenient processed mirror, not the authoritative license source. Its CC BY 4.0 metadata conflicts with SLR104's CC BY-SA 4.0.

## Unavailable, ambiguous, or non-default resources

- The Gujarati/Tamil/Telugu–English MSCS/WSTCSMC data used by published language-diarization papers has 200 ms labels, but this review did not locate a public download with a clear license. It is scientifically relevant but **not currently reproducible/usable without author access**.
- IITG-HingCoS is genuinely relevant, especially at 8 kHz, but its current paid, non-commercial, no-redistribution agreement is not an open-data license.
- The IIIT-H phonetically balanced Hindi–English read corpus is described in 2018 papers as about 11 h with speaker-independent train/test speakers, but no current canonical download/license was verified.
- CoSTA's paper says podcast permission was obtained, but permission for a paper experiment is not equivalent to a public reuse license. The GitHub release lacks a LICENSE file.
- Kathbath's HF license tag and card body conflict. Common Voice's CC0 computational license coexists with MDC platform restrictions. These are not interchangeable details; preserve both notices and seek owner clarification where redistribution matters.

## Recommended minimal stack for this repository

For a CPU take-home rather than a leaderboard campaign:

1. **FLEURS:** 50–100 test clips per project language, with 0.5/1/2/3/5 s prefix scoring.
2. **Svarah:** 100–200 Indian-accented English test clips.
3. **HiACC:** adult and child official test splits, reporting read/spontaneous separately.
4. **IITB TextGrid:** every permitted long recording, but speaker-macro and per-recording results only.
5. **GramVaani:** a small, fixed Hindi eval slice plus paired synthetic 8 kHz transforms of FLEURS/Svarah.
6. **Vaani Benchmark:** a manually reviewed, speaker-capped high-code-mix subset if gate approval and annotation time permit.

Record dataset version/commit, source split, speaker ID, original sample rate, resampling/codec transform, clip hash, duration, transcript-language evidence, and whether a boundary is human, forced-aligned, or synthetic. Never merge forced-aligned labels and human labels into one unnamed “ground truth” column.

## Primary sources

- Conneau et al., [FLEURS](https://arxiv.org/abs/2205.12446), 2022-05-25; [HF dataset card](https://huggingface.co/datasets/google/fleurs).
- Mozilla Data Collective, [Common Voice Scripted Speech v27 / Spontaneous Speech v5 announcement](https://community.mozilladatacollective.com/commonvoice-27/), 2026-09-21; [release metadata repository](https://github.com/common-voice/cv-dataset).
- Javed et al., [IndicVoices](https://aclanthology.org/2024.findings-acl.639/), Findings of ACL 2024; [HF card](https://huggingface.co/datasets/ai4bharat/IndicVoices).
- Javed et al., [IndicSUPERB / Kathbath](https://arxiv.org/abs/2208.11761), 2022-08-25; [official repository](https://github.com/AI4Bharat/IndicSUPERB).
- MUCS 2021, [official data page](https://navana-tech.github.io/MUCS2021/data.html), 2021; [OpenSLR SLR104](https://www.openslr.org/104/).
- Singh, Singh and Kadyan, [HiACC](https://doi.org/10.1016/j.dib.2025.111886), available online 2025-07-17; [Zenodo data](https://doi.org/10.5281/zenodo.15551669).
- Rao et al., [IITB Hindi–English TextGrid corpus](https://www.ee.iitb.ac.in/student/~daplab/datasets/code_switch.html), Interspeech 2018.
- Ganji, Dhawan and Sinha, [IITG-HingCoS corpus paper](https://arxiv.org/abs/1810.00662), 2018-09-24; [official access terms](https://www.iitg.ac.in/eee/emstlab/HingCoS_Database/db_copy.php).
- Pulikodan et al., [Vaani](https://arxiv.org/abs/2603.28714), 2026; [Vaani Benchmark V1.0 card](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Benchmark-V1.0).
- Javed et al., [Svarah](https://www.isca-archive.org/interspeech_2023/javed23_interspeech.html), Interspeech 2023; [HF card](https://huggingface.co/datasets/ai4bharat/Svarah).
- GramVaani, [OpenSLR SLR118](https://www.openslr.org/118/), challenge/data release 2022.
- Dongre et al., [CoSTA](https://aclanthology.org/2025.coling-main.618/), COLING 2025; [release repository](https://github.com/csalt-research/CoSTA).
- Soket Labs, [CoSHE-Eval card](https://huggingface.co/datasets/soketlabs/CoSHE-Eval), checked 2026-09-26.
