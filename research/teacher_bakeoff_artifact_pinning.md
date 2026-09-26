# Reproducible teacher-bakeoff artifact pinning (METHOD-20)

**Audit date:** 2026-09-26 (IST)

**Scope:** the three teachers in `experiments/teacher-bakeoff/run.py`; model
identity and replayability only. This note does not re-rank the teachers.

## Bottom line

The reported commit strings are accurate descriptions of the cache today, but
they are not enforced inputs to the experiment:

- ECAPA loads a mutable repository ID and then assigns a constant revision
  string without checking what was loaded.
- Whisper loads its processor and model from the default branch in separate
  calls, then records only the model configuration's `_commit_hash`.
- MMS similarly performs three independent default-branch resolutions: feature
  extractor, model, and the `model.safetensors` used by its tensor check.

This is a provenance defect, not evidence that the published numerical result
is wrong. Live Hugging Face API checks on **2026-09-26** found that every
repository's `main` still resolves to the full commit reported in
`results.json`; all 18 locally materialized files resolve under those exact
snapshot directories and pass the byte hashes below. A pinned rerun is
therefore expected to preserve the teacher ranking, but that is **unverified**
until it is run.

Hugging Face Hub 0.24.7 documents that an unqualified download uses the latest
`main`, while `revision` selects a branch, tag, or commit
([official guide, accessed 2026-09-26](https://huggingface.co/docs/huggingface_hub/v0.24.7/en/guides/download#from-specific-version)).
Transformers 4.44.2 likewise declares `revision="main"` and
`use_safetensors=None` as `from_pretrained` defaults
([official API, accessed 2026-09-26](https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)).
Observing `_commit_hash` after those calls cannot constrain what they already
downloaded, cannot bind the separately loaded processor, and cannot prevent a
branch update between calls.

Expected accuracy or latency gain from this work: **none**. It makes a future
bake-off falsifiable and repeatable.

## Current code audit

| Teacher | Current load path | What the result records | Failure mode |
|---|---|---|---|
| ECAPA | `EncoderClassifier.from_hparams(source=model_id, savedir=fixed_path)` with no revision | A hard-coded `0253049...` | The recorded value need not describe the files. The non-revision-qualified saved directory can also reuse previously linked files. |
| Whisper-small | `WhisperProcessor.from_pretrained(model_id)` and `WhisperForConditionalGeneration.from_pretrained(model_id)` | `model.config._commit_hash` only | Processor and model are two mutable resolutions. Processor files have no recorded identity. Weight-format choice is implicit. |
| MMS-LID-126 | Unpinned feature-extractor and model loads, followed by unpinned `hf_hub_download(..., "model.safetensors")` | `model.config._commit_hash`; equality of two positional-convolution tensors | The model, preprocessing config, and validation checkpoint can come from different revisions. The two-tensor check is valuable but does not bind the full artifact. |

There is an additional SpeechBrain 1.0.3 trap. `from_hparams` accepts a
`revision` and supplies it when fetching the YAML, but `Pretrainer.collect_files`
constructs its weight fetch with `"revision": None`. Therefore merely adding
`revision=...` at the existing ECAPA call is insufficient. This is visible in
the official tagged sources for
[`interfaces.py`](https://github.com/speechbrain/speechbrain/blob/v1.0.3/speechbrain/inference/interfaces.py#L401-L531)
and
[`parameter_transfer.py`](https://github.com/speechbrain/speechbrain/blob/v1.0.3/speechbrain/utils/parameter_transfer.py#L184-L287)
(inspected 2026-09-26). The main target generator already uses the safe
workaround: resolve and verify one local snapshot, load that local directory,
and override `pretrained_path` to the same directory.

The experiment's resumable output compounds the issue. Existing model entries
are retained when `--models` reruns a subset, but the resume gate checks only
the audio/input fingerprint. It can therefore combine model rows produced by
different Hub snapshots or loader environments while presenting one table.

## Immutable Hub snapshot audit

The Hub API was queried both at `main` and at each full revision on
2026-09-26. Each live `main` matched the listed full revision:

| Model | Full revision | Commit date | Exact-revision evidence | Current cached loader footprint | Combined artifact SHA-256 |
|---|---|---:|---|---:|---|
| `speechbrain/lang-id-voxlingua107-ecapa` | `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9` | 2024-11-27 | [commit](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/commit/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9), [API](https://huggingface.co/api/models/speechbrain/lang-id-voxlingua107-ecapa/revision/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9?blobs=true) | 4 files / 85,240,633 bytes | `f193a0548951e8fbd6ca438a492b98b5c48bf7d98c86451ca3878dd4a7c0f706` |
| `openai/whisper-small` | `973afd24965f72e36ca33b3055d56a652f456b4d` | 2024-02-29 | [commit](https://huggingface.co/openai/whisper-small/commit/973afd24965f72e36ca33b3055d56a652f456b4d), [API](https://huggingface.co/api/models/openai/whisper-small/revision/973afd24965f72e36ca33b3055d56a652f456b4d?blobs=true) | 11 files / 971,367,937 bytes | `7d5828fd8c8d0602a6b400b177bc673a47cfc71c1180dbb6783ae904e8aee9f3` |
| `facebook/mms-lid-126` | `53da6e311f3ce48f324fe335924216187e3109a4` | 2023-06-13 | [commit](https://huggingface.co/facebook/mms-lid-126/commit/53da6e311f3ce48f324fe335924216187e3109a4), [API](https://huggingface.co/api/models/facebook/mms-lid-126/revision/53da6e311f3ce48f324fe335924216187e3109a4?blobs=true) | 3 files / 3,864,500,287 bytes | `e1b75df414cb4f4587a397f944b5763a5a78354b153913bb03a8921fe821aa80` |

The combined digest above follows the main pipeline's existing convention:
iterate filenames in lexical order, update SHA-256 with the UTF-8 filename and
then the 32 raw bytes represented by that file's SHA-256. The per-file byte
count and hash remain authoritative; the algorithm name/order must be stored
with the combined value.

The Whisper commit is not a meaningless label: it changed `merges.txt`,
`special_tokens_map.json`, `tokenizer.json`, and `tokenizer_config.json`
([Hub diff dated 2024-02-29](https://huggingface.co/openai/whisper-small/commit/973afd24965f72e36ca33b3055d56a652f456b4d)).
The MMS revision added `model.safetensors` with LFS SHA-256
`69fa8def...e0ecd` and changed `config.json`
([Hub diff dated 2023-06-13](https://huggingface.co/facebook/mms-lid-126/commit/53da6e311f3ce48f324fe335924216187e3109a4)).
These are exactly the kinds of coupled config/weight/tokenizer changes that a
post-load model-only commit string cannot represent.

### Exact file manifests

These SHA-256 values were computed over the locally cached bytes underneath
the exact snapshot paths on 2026-09-26. Hub LFS hashes and sizes agree for all
large files; ordinary Git files expose a Git blob ID through the API, so their
content SHA-256 was independently computed locally.

#### ECAPA

| File | Bytes | SHA-256 |
|---|---:|---|
| `classifier.ckpt` | 762,555 | `a50d9024ff58d317031c9787d4c6c614d454a87a8ef32f9d36338cd3ff57adbc` |
| `embedding_model.ckpt` | 84,474,355 | `ab750d5c06d713477045fa798fab5d33e959dbc0dfe4de510a9a47844c79a19a` |
| `hyperparams.yaml` | 1,519 | `88fec9791a8416a152fb10834327e18d38e5bf7a351e9b714e08cdc4af05de6f` |
| `label_encoder.txt` | 2,204 | `9f566d83c4f19168be4a0bf86c0c7dac7d3264a95105bcbf33a7c32b83ccc17f` |

#### Whisper-small

| File | Bytes | SHA-256 |
|---|---:|---|
| `added_tokens.json` | 34,604 | `9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1` |
| `config.json` | 1,967 | `e6a2b489da1b5aed65a8eb8d1e7466fa867ad5643a8bc138ba708bd56b2875c4` |
| `generation_config.json` | 3,868 | `71565b8ef50d0bf7a1193ed4bbed195b94e70c18894d81bba2f1233dcec3ab53` |
| `merges.txt` | 493,869 | `2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6` |
| `model.safetensors` | 966,995,080 | `1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b` |
| `normalizer.json` | 52,666 | `bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd` |
| `preprocessor_config.json` | 184,990 | `9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666` |
| `special_tokens_map.json` | 2,194 | `e67ae3a0aaa99abcd9f187138e12db1f65c16a14761c50ef10eef2c174a7a691` |
| `tokenizer.json` | 2,480,466 | `27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566` |
| `tokenizer_config.json` | 282,683 | `2a4c4281cf9f51ac6ccc406fdc711a087afe6530f671fa7b80953edc498275ce` |
| `vocab.json` | 835,550 | `8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9` |

The 11 files are the current cached footprint produced by the two loader calls,
not a syscall trace proving that every tokenizer byte affects this audio-only
forward pass. Treating the complete footprint as the allowlist is intentionally
conservative. Alternate `pytorch_model.bin`, Flax, and TensorFlow weights exist
at the revision but were not part of this load and must not be silently
substituted.

#### MMS-LID-126

| File | Bytes | SHA-256 |
|---|---:|---|
| `config.json` | 4,267 | `f5f50a99828c0334299ca9180c026518ca69b6b2a41dd3bbede74eea762e4ba3` |
| `model.safetensors` | 3,864,495,808 | `69fa8def8f0242660b47d8b47c2a1b645d9fc70098cc217fbd52087b68de0ecd` |
| `preprocessor_config.json` | 212 | `a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335` |

The alternate `pytorch_model.bin` is not in the contract. Its Hub LFS SHA-256
is different, even though it is intended to encode the same model.

### Label-map checks

Artifact hashes should be accompanied by semantic checks because the reported
seven-way restriction depends on mappings, not just tensor shapes:

- ECAPA selected indices remain `(20, 35, 63, 9, 91, 92, 31)` for
  `(en, hi, mr, bn, ta, te, gu)`.
- Whisper `generation_config.lang_to_id` contains 99 entries and maps the same
  order to `(50259, 50276, 50320, 50302, 50287, 50299, 50333)`.
- MMS `config.id2label` contains 126 entries and maps
  `(eng, hin, mar, ben, tam, tel, guj)` to
  `(2, 16, 29, 12, 46, 23, 74)`.

These values are exact for the audited snapshots. They are not claims about
model accuracy.

## Runtime identity is also incomplete

The reproduction command pins only `transformers==4.44.2`; `results.json`
stores only `torch_version`. The cached ephemeral environment created just
before the historical run contains `transformers==4.44.2`,
`huggingface-hub==0.24.7`, `tokenizers==0.19.1`, and
`safetensors==0.8.0`, while the project environment pins
`speechbrain==1.0.3`, `torch==2.6.0+cpu`, and `torchaudio==2.6.0+cpu`.
That cache is useful local evidence but is not checkpoint-bound proof of the
historical process environment. At minimum, a new result must bind those seven
packages, Python, NumPy, the experiment-driver SHA-256, and the imported main
pipeline source identity.

## Input chronology caveat

The historical result records input fingerprint
`52bca81e32db34edb0a9ad4529b01bd6ddd36aad6f587733a01ace669538d6fc`.
Recomputing the same function over the current manifest and WAVs on
2026-09-26 gives
`4dff38b5b8deb615e5849826744f4c4c1df8c569f066ef155a97d3bd7dff4317`.
Thus a model-pinning rerun on today's corpus is a **new bake-off**, not a
bit-for-bit reproduction of the published one. Whether the complete old input
bundle can still be reconstructed is **unverified**; the old digest alone is
not a recovery recipe. Preserve the old report as historical evidence and
publish the pinned current run under a new identity rather than overwriting it
and implying only the loader changed.

## Recommended fail-closed contract

1. Define one immutable registry entry per teacher containing repository ID,
   full 40-character revision, exact allowlisted filenames, byte counts,
   per-file SHA-256, aggregate algorithm/version, and aggregate digest.
2. Resolve all files once with `snapshot_download(..., revision=full_sha,
   allow_patterns=...)`; verify every byte before constructing a model. Load
   the verified local directory with `local_files_only=True`. Do not let model,
   processor, or validation code resolve a repository ID independently.
3. Pass `use_safetensors=True` for both Transformers models. The MMS tensor
   check must open the verified local `model.safetensors`, not call
   `hf_hub_download` again. Retain its weight-normalization compatibility check
   and reject any other missing, unexpected, mismatched, or error keys.
4. For ECAPA, use the main pipeline's local-snapshot plus
   `overrides={"pretrained_path": snapshot}` pattern and a saved directory
   keyed by full revision plus aggregate digest. Do not rely on SpeechBrain
   1.0.3's partial `revision` propagation.
5. Record the complete model artifact identity, label mapping, loader choices,
   runtime/source identity, and input identity in each model result. A partial
   resume must reject any mismatch before retaining old rows.
6. After online prefetch, exercise the exact run with `HF_HUB_OFFLINE=1` and
   `TRANSFORMERS_OFFLINE=1`. Network independence is a test, not the identity
   mechanism; hashes remain the authority.

## Concrete acceptance tests

- **No mutable resolution:** monkeypatch Hub download entry points and assert
  that each receives a full expected revision during prefetch; after prefetch,
  assert all loaders receive only verified local paths.
- **Corruption and omission:** alter one byte of a small copied config in a
  temporary directory and separately remove a required file; both must fail
  before model construction or result reuse.
- **No split snapshots:** substitute a processor/config directory with a
  different aggregate identity while keeping weights fixed; reject it.
- **Deterministic format:** remove `model.safetensors` while leaving alternate
  weights available; reject instead of falling back to `.bin`.
- **Semantic mapping:** assert the ECAPA, Whisper, and MMS selected maps above,
  not merely their class counts.
- **Resume identity:** seed a result with one changed model digest, driver hash,
  dependency version, or input fingerprint and run a different `--models`
  subset; reject the entire mixed resume.
- **Fresh offline replay:** on the current pinned input, run all three teachers
  twice from the verified cache and compare an identity-excluding-timing
  prediction payload. Exact class decisions and finite posteriors should match;
  floating equality across different CPUs/BLAS implementations is
  **unverified**, so store a declared tolerance if cross-host replay is tested.

## What this does and does not establish

It establishes a precise recipe for proving which bytes produced a bake-off
row. It does not repair the small synthetic evaluation, the chronology-unsafe
historical switch matcher, the changed corpus, or the lack of real-call
evidence. It also does not justify changing teachers. The earlier retain-ECAPA
decision remains the available result, subject to its already documented data
and scoring limits.
