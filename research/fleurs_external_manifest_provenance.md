# Pinned FLEURS external-manifest provenance (BUG-17)

Research and Hugging Face Hub audit: **2026-09-26 (Asia/Calcutta)**.

## Decision

The current 700-row FLEURS validation release is correct when independently
rederived from the pinned Hub source. The reusable cached-manifest validator is
not sufficient to establish that fact.

`validate_cached_external_manifest()` trusts a hash stored inside the same
mutable JSON and rechecks only language counts plus the audio facts asserted by
each record. After recomputing that self-hash, it accepts swapped labels,
duplicate records, a false pool digest/count, and a forged source-artifact
table. Correct WAV bytes do not make the labels or selection correct.

Keep the current external metrics only as a historical diagnostic whose input
was independently audited. Before calling the experiment reproducible, add an
**independent release lock** to the staged experiment source, reconstruct every
semantic record from that lock, and publish a schema-bumped rerun. The simplest
runtime design is:

1. pin the exact source artifacts and the already-audited canonical release
   digest in a tracked lock;
2. validate the lock before trusting any manifest field;
3. derive config, label, selection membership, IDs, paths, and duration rather
   than accepting them from cached JSON;
4. hash every local selected WAV and compare the complete reconstructed record
   list with the locked release; and
5. retain a separate rebuild mode that regenerates the lockable release from
   the pinned TSVs and archives.

This has **zero expected model or accuracy gain**. Exact prediction/metric
parity after the source-stage and manifest-schema repair is **unverified** until
the experiment is rerun, although no input byte is intended to change.

## Why the current validator is not a provenance check

The creation path in
[`run.py`](../experiments/checkpoint-external-validation/run.py#L403) is
substantially stronger than the cached path. On first creation it resolves the
full commit, downloads seven TSV/archive pairs, parses the TSVs, applies the
frozen SHA-256 ranking, extracts the selected WAVs, and records source hashes.

The cached path in
[`validate_cached_external_manifest()`](../experiments/checkpoint-external-validation/run.py#L366)
checks:

- a high-level identity copied from constants;
- 700 rows and 100 occurrences of each project label;
- `manifest_sha256`, recomputed over the other fields in that same JSON; and
- each referenced WAV against that record's own `num_samples`, hash, byte
  count, format, and subtype.

It does not establish any of the following:

- `config -> project language` is the frozen mapping;
- a record exists in the pinned config's TSV at the declared row;
- its ID, basename, sample count, or gender came from that row;
- it is one of the deterministic lowest-ranked 100 rows;
- its clip ID, destination path, label, language name, or duration was derived
  correctly;
- selected clip IDs, paths, source rows, or archive members are unique;
- `pool_rows`, `pool_metadata_sha256`, or `source_artifacts` are true; or
- the WAV is the selected member of the pinned archive rather than a different
  valid WAV whose new hash was written into the same self-hashed JSON.

A self-contained checksum is useful for detecting an accidental byte change
when the checksum is not also updated. It is not an independent trust root for
the semantics it covers.

### Reproduced accepting attack

In memory, without changing the checked-in release, I made all of these changes
at once:

- swapped the first `en_us` record's project label with the first `hi_in`
  record's label, preserving aggregate label counts;
- replaced a second English record with a duplicate of another English record;
- set the English TSV/archive paths, sizes, and hashes to forged values;
- set `pool_metadata_sha256` to 64 zeroes and `pool_rows` to `1`; and
- recomputed the JSON's own canonical hash.

The production validator accepted the object after rehashing all local WAVs:

```text
accepted=True
swapped=en_us->hi,hi_in->en
duplicate=fleurs-en_us-validation-0006-1646
source_tsv_sha=0000000000000000000000000000000000000000000000000000000000000000
pool_hash=0000000000000000000000000000000000000000000000000000000000000000
pool_rows=1
```

The validation call took 17.313 s on this shared host because it still read the
selected audio. The elapsed time is diagnostic only; it is not a benchmark.
The same semantic weakness was independently found by the reviewer with a
minimal English/Hindi label-swap fixture in
[`changes-2026-09-26-ea58c4a.md`](../review/changes-2026-09-26-ea58c4a.md).

## Positive audit of the current release

The accepting attack does **not** imply that the current checked-in manifest is
forged. A clean-room read-only reconstruction produced the following exact
positive results:

- Hub dataset: `google/fleurs`;
- pinned revision:
  `70bb2e84b976b7e960aa89f1c648e09c59f894dd`;
- config-to-label map:
  `en_us/en`, `hi_in/hi`, `mr_in/mr`, `bn_in/bn`, `ta_in/ta`, `te_in/te`,
  `gu_in/gu`;
- source validation rows:
  `394/239/443/402/377/311/432`, totaling `2,598`;
- pool metadata SHA-256:
  `2c825a340a0e9dbbfa9d10f0df658afb68e14a389d3148704d1b5950649cf675`;
- zero rank collisions across all seven source pools;
- exactly the lowest 100 declared ranks per config;
- 700/700 manifest records exactly equal to independently derived records;
- 700 unique clip IDs, 700 unique audio paths, and 700 unique
  `(config,row_idx,id,basename)` source keys;
- 100 records for every project language;
- selected-audio total: `499,935,640` bytes, `124,973,760` samples, or
  `7,810.86` s / 2.1697 h at 16 kHz;
- exact manifest-file SHA-256:
  `5a3335cda8b679ca4a6de8f084bff23c8fa82c18e1670e8bfdc6da751b246614`;
- canonical manifest SHA-256:
  `ef0ec36427f742074b1bc88bd42f1b7e5b93b43bb390f9382f5030f3d8d2a637`;
  and
- the independently reconstructed canonical manifest has that same digest.

The reviewer independently rederived all records, checked every pinned
TSV/archive hash, and checked all 700 local audio hashes, lengths, and formats.
I rechecked the TSV derivation, release fields, local audio hashes/formats, and
Hub object metadata; I did not repeat full archive decompression.

The current report/result hashes at this audit were:

```text
results.json  90d511775dcc6ff0ae49b7ef6e380eb4d6d573d57f0affc269f812beee1bb397
REPORT.md     bbcc1cdf137337193365a8c531df5cb4266defab916ad6355938662e19e32437
```

Those values identify the historical diagnostic. They do not repair its
loader.

### Source `id` is deliberately not unique

The TSV column currently named `dataset_row_id` in the experiment is the
FLEURS `id`, not a unique row key. FLEURS is an n-way-parallel corpus and can
contain up to three recordings for one sentence. The pinned validation pools
contain only `150/126/150/150/150/140/150` distinct IDs for
`394/239/443/402/377/311/432` rows. In the selected release, the unique ID
counts are:

| Config | Selected WAVs | Unique source IDs | Maximum multiplicity |
|---|---:|---:|---:|
| `en_us` | 100 | 79 | 3 |
| `hi_in` | 100 | 78 | 3 |
| `mr_in` | 100 | 76 | 3 |
| `bn_in` | 100 | 79 | 2 |
| `ta_in` | 100 | 72 | 3 |
| `te_in` | 100 | 76 | 3 |
| `gu_in` | 100 | 75 | 3 |

The validator must therefore require uniqueness of `(config,row_idx)` and the
derived clip/path/basename keys, **not** uniqueness of `id`. Renaming the field
to `source_sentence_id` or documenting that meaning would prevent a future
"deduplication" fix from deleting legitimate recordings. The 535 selected
`(language,id)` groups and 150 global IDs are also the clustering structure
identified in the separate uncertainty memo; provenance validation must retain
them exactly.

## Dated Hub source audit

On **2026-09-26**, both `main` and the explicit revision resolved to
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`. The official Hub API reported
`lastModified=2026-05-15T09:35:34Z`, 1,004 siblings, public/ungated access, and
`license: cc-by-4.0`
([pinned API response](https://huggingface.co/api/datasets/google/fleurs/revision/70bb2e84b976b7e960aa89f1c648e09c59f894dd)).
The runtime used `huggingface-hub==0.24.7`.

The [pinned dataset card](https://huggingface.co/datasets/google/fleurs/blob/70bb2e84b976b7e960aa89f1c648e09c59f894dd/README.md?code=true)
lists the configurations, 16 kHz audio schema, `lang_id` names, and CC BY 4.0
metadata; it explicitly gives `hi_in` as the Hindi config. The card itself is
385,614 bytes, raw SHA-256
`688f79f2a5c731af3796e9f683eb02f9b3f09d040decd8c5625d0f37098e71c6`,
and Hub Git blob `dcc0872174e54ad416dee938651d777378d4ba4f`.

The exact source objects are:

| Config | Label | Rows | TSV bytes | TSV raw SHA-256 | Hub Git blob | Archive bytes | Hub/local LFS SHA-256 |
|---|---:|---:|---:|---|---|---:|---|
| `en_us` | `en` | 394 | 213,065 | `9d57ee7e91e9d4c92edb39f6bbea668ef8dc2a3ff96eb510d5580b2ad05d17ec` | `4fca22b561c05159f76a7249d653c3e7677103cf` | 171,250,900 | `2658fda72f199e12676ecac9415094667a4e14e149b146e568ea00b2a2f0954c` |
| `hi_in` | `hi` | 239 | 249,853 | `cea87c57a37a0d38ed0afce30e68a35ad7b3945414648430fee09a54ca7b72ba` | `47ef62d67bc3ccb61607649c2033d64c57f50390` | 131,741,732 | `9adbca6d6fc70e40c121910941bcd7c8906eee60b402b6d21b4bd160e20030c7` |
| `mr_in` | `mr` | 443 | 513,623 | `aa8219008c8a14584acb7b7e3cd20c1c32b55c3fb96f45d6e901e8eacbe43c9c` | `31b7196a7df68b41effc5d0e842f0cda27d9f2c5` | 292,468,044 | `a8cdc2253a0de4b1d43170e836313a2139c2451946a49344b43cef1a200fc6c1` |
| `bn_in` | `bn` | 402 | 465,963 | `239a59e4d60a76d4d388d44f72df6d28b91e4526d829186d15ef33fde7a089db` | `d28105d42aa6c409aaf17d7ebfa94435c0777331` | 278,667,778 | `b7f380b67d50bc59f328af3e1bd8096ab9e728bd0103349b9746ca95ab75ff30` |
| `ta_in` | `ta` | 377 | 507,783 | `0a5ad9f10d3284f48c268e947efba178ec41c368dce191f652a09a6697a54a8b` | `b5d163f709c6941b56e653fd8e3d6f6ed2dee6b5` | 238,893,728 | `63faa53c804949dae73c9a953927fb50d59540830261cff374588ba4c901f09e` |
| `te_in` | `te` | 311 | 350,269 | `fbedc2a5b7e394ff5a6e8c269f616017d2ac9a247ffa707c10efaa08132a0630` | `3083d65010458ccb2de3a30e3ee99b95f0143e5d` | 166,827,973 | `2ac52fa35be22b05ad041e2a2a443d10f4ef59449db07fe42cd644eb81ed7123` |
| `gu_in` | `gu` | 432 | 474,781 | `91711c64667a6cac092ccf2eac2e48b0945ad49d10712ea41852d00f96d8d724` | `dfcef5c82a7d47b45c7442fb1d7b6adfa9e31543` | 225,657,147 | `67b61fe0b78c80585ffc3743eba2528e18d756819495abd3c603e0824c0b78bc` |

The archives' Git-LFS pointer blob IDs, also returned by the pinned Hub API,
are:

```text
en_us e851349799c9689aa86a1cfaf39e8bfd573c54a6
hi_in 111efdd10487ef314ae8f85e8b56712d6c9a76e4
mr_in a8f4396a06de12bfc826ee2615ca2a5dfa9a6f36
bn_in b60735df054f9097516bbb9c83569dded2fe24c0
ta_in 61c0900ce80dd14be6f50f154b696a293e222444
te_in 229a3a8002355e6394e740b153a05fe5d505e325
gu_in ff4de7b09e4513dbbb08016f539bab9cfb25d1d4
```

The seven TSVs total 2,775,337 bytes; the seven compressed archives total
1,505,507,302 bytes (1.402 GiB). `git hash-object` over every local TSV exactly
matched the Hub API's Git blob ID. Every local archive SHA-256 exactly matched
the Hub LFS SHA-256.

This distinction follows the official Hub cache contract: ordinary Git files
use their Git SHA-1 as ETag/object key, while Git-LFS objects use SHA-256
([Hub 0.24.7 download documentation, accessed 2026-09-26](https://huggingface.co/docs/huggingface_hub/v0.24.7/en/package_reference/file_download#downloading-files)).
The same documentation states that a full commit hash can be passed as
`revision`, and snapshots are stored per commit. The project should retain the
raw SHA-256 as its uniform runtime checksum **and** the Hub object identifier as
evidence that the bytes are the objects named by the pinned repository.

The [FLEURS paper](https://arxiv.org/abs/2205.12446), submitted
**2022-05-25**, describes the 102-language, n-way-parallel corpus and its use
for Speech-LangID. That structure explains repeated source IDs; it does not
make any cached project label trustworthy without the config mapping. The
dataset is licensed CC BY 4.0; the official license requires attribution, a
license link, and notice of modifications
([Creative Commons deed, accessed 2026-09-26](https://creativecommons.org/licenses/by/4.0/)).

## Proposed provenance contract

### 1. Put the trust root outside the mutable cache

Add one small, tracked `fleurs-validation-lock.json` beside the experiment
driver, and include it in BUG-16's pre-import executed-source stage. It should
contain:

```json
{
  "schema_version": 1,
  "dataset_id": "google/fleurs",
  "repo_type": "dataset",
  "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
  "license": "cc-by-4.0",
  "readme": {
    "repo_path": "README.md",
    "bytes": 385614,
    "raw_sha256": "688f79f2...e71c6",
    "hub_blob_id": "dcc08721...ba4f"
  },
  "configs_by_language": {
    "en": "en_us", "hi": "hi_in", "mr": "mr_in", "bn": "bn_in",
    "ta": "ta_in", "te": "te_in", "gu": "gu_in"
  },
  "artifacts": {
    "en_us": {
      "tsv": {
        "repo_path": "data/en_us/dev.tsv",
        "bytes": 213065,
        "raw_sha256": "9d57ee7e...d17ec",
        "hub_blob_id": "4fca22b5...103cf"
      },
      "archive": {
        "repo_path": "data/en_us/audio/dev.tar.gz",
        "bytes": 171250900,
        "raw_sha256": "2658fda7...954c",
        "hub_blob_id": "e8513497...54a6",
        "hub_lfs_sha256": "2658fda7...954c"
      }
    }
  },
  "derivation": {
    "split": "validation",
    "sample_per_config": 100,
    "rank_version": 1,
    "rank_fields": ["revision", "config", "row_idx", "id", "basename"],
    "rank_separator_hex": "00",
    "sort_tie_key": ["config", "row_idx", "id", "basename"],
    "sample_rate": 16000
  },
  "release": {
    "pool_metadata_sha256": "2c825a34...675",
    "manifest_file_sha256": "5a3335cd...614",
    "manifest_canonical_sha256": "ef0ec364...637",
    "records": 700
  }
}
```

The source table and pointer list above supply the omitted six configs' full
values. Under the exact canonical encoding used for this proposed complete
source-lock object, its current positive-fixture SHA-256 is
`fe505973d5662867f5744b8c408eddc06bcd6c61d44fbda6163b7a41abeace19`.
That digest is a reproducibility aid, not an existing artifact; changing the
lock schema requires recomputing it explicitly.

The lock must be immutable for one run and bound into the experiment run ID,
results, teacher cache, and report. Reading a lock from the live worktree after
the staged child starts would recreate BUG-16's execute/certify race.

### 2. Make derivation a pure function

Given the lock and pinned TSV bytes, construct the complete pool without
consulting `manifest.json`:

1. require the exact TSV path, byte count, raw SHA-256, and Hub Git object ID;
2. parse exactly seven tab-separated fields per row;
3. require safe basename-only `.wav` names, positive integer sample counts,
   allowed genders, and integer source IDs;
4. derive `config` from the file being parsed and project `language` only from
   the inverse locked config map;
5. assign `row_idx` from the pinned byte order;
6. require unique `(config,row_idx)` and basename within a config, while
   allowing repeated source IDs;
7. compute the versioned rank from the exact five fields and NUL separators;
8. reject a rank collision or apply the explicit total tie key before taking
   exactly 100 per config; and
9. derive `clip_id`, relative audio path, `language_name`, integer sample
   count, and display duration from that selected source row.

The cache record is then an output to compare, never an input to the
derivation. `num_samples` is the duration source of truth; floating-point
`duration_seconds` should be regenerated as `num_samples / 16000`.

### 3. Validate the cached release offline without 1.4 GiB archive scans

For the take-home, the locked exact/canonical manifest digests are the simplest
independent runtime anchor. On every cached load:

1. verify the staged lock's own digest and schema;
2. read the manifest bytes once and require both the locked exact-file and
   canonical digests;
3. reconstruct the expected 700 semantic records from the pinned TSV snapshot,
   or compare the complete records against a separately locked derived-record
   digest/table;
4. require exact record equality, order, config counts, label counts, and all
   composite uniqueness invariants;
5. resolve each path beneath the declared data directory, reject symlinks or
   escape, and verify the current WAV's bytes/hash, format, sample count, and
   subtype; and
6. only then load waveforms, teacher predictions, a trajectory, or any locked
   test resource.

This keeps ordinary reruns offline and rehashes only the 476.8 MiB selected
release, not all 1.402 GiB of compressed source archives. If the project does
not want a locked release digest, the alternative is to retain all TSV/archive
sources and rederive the selected archive members on every load. That is more
general but materially heavier. A self-hashed cache without either independent
anchor is not an acceptable third option.

### 4. Keep a source-derived rebuild path

The explicit rebuild command should:

- query `dataset_info(..., revision=full_sha, files_metadata=True)` and require
  the exact revision, public/ungated state, license, README object, and all 14
  source object IDs/sizes;
- fetch only by the full revision, never by `main`;
- verify raw bytes before parsing or extraction;
- scan each tar for the exact `dev/<basename>` member, reject duplicate member
  names, non-regular members, missing/extra selected matches, unsafe names, and
  payload size/hash mismatch;
- create the complete 700-record release in a sibling staging directory;
- validate it against the tracked lock; and
- publish audio first and the manifest last, without silently rewriting the
  lock.

The current extractor streams member contents rather than calling unsafe bulk
`extractall`, which is good. It should match the exact archive member path
rather than only `Path(member.name).name` and should reject two members with
the same selected basename.

`main` happened to equal the pin on the audit date. Correctness must not depend
on that remaining true.

### 5. Invalidate every downstream consumer on a provenance change

The repaired external-manifest schema/version and lock digest must flow into:

- the external examples' ordered identity;
- `teacher-cache.json` identity and prediction order;
- every per-checkpoint score table;
- bootstrap/selection inputs;
- `results.json` and its run ID; and
- the human report.

A changed lock, TSV, archive, rank version, mapping, record order, WAV, or
validator source must invalidate the teacher cache and any partial result. A
source failure must occur before ECAPA/student load or scoring. Repair BUG-16
and BUG-17 in one separately named schema-bumped rerun so the data lock itself
is executed from the certified source tree.

The existing result rejected checkpoint promotion. That conservative outcome
may remain numerically identical, but parity is **unverified** until this
identity chain is exercised. Do not overwrite the historical schema-1 result
and describe it retroactively as source-derived.

This input-provenance contract does not by itself certify which code/runtime
produced reused ECAPA predictions. The later reviewer finding calls that
separate issue BUG-18: the teacher cache also needs its producer's staged
source, dependency/runtime identity, exact language-index mapping, ordered
inputs, and payload digest bound into the experiment identity. Binding the
FLEURS lock into the cache is necessary but not sufficient to close BUG-18;
that producer audit is outside this one-topic iteration.

## Acceptance tests

1. **Current positive fixture.** Reconstruct exactly 2,598 source rows with
   counts `394/239/443/402/377/311/432`, pool digest `2c825a34...675`, zero
   rank collisions, 700 selected records, canonical release
   `ef0ec364...637`, and 700 unique composite source/clip/path keys.
2. **Label swap.** Swap one English and Hindi `language`, recompute every
   mutable JSON hash, and require rejection before any waveform/model load.
3. **Duplicate replacement.** Replace one selected row with a same-language
   duplicate so counts remain 100 each; require rejection for membership and
   uniqueness.
4. **Source-table forgery.** Change a TSV/archive path, size, raw hash, Hub
   object ID, pool row count, and pool digest while recomputing the manifest's
   self-hash; every mutation must fail against the independent lock.
5. **Selection forgery.** Substitute the 101st-ranked valid source row for the
   100th, copy its real WAV, and recompute all record/self hashes; require
   rejection even though every byte and format is individually valid.
6. **Derived-field forgery.** Mutate in turn `config`, `language`, `row_idx`,
   source ID, basename, gender, `language_name`, clip ID, path, sample count,
   and duration; require exact-field rejection.
7. **Audio substitution.** Replace a selected WAV with another valid 16 kHz
   mono WAV and update its record/self hashes; require rejection against the
   locked selected payload.
8. **Repeated source IDs are legal.** The positive fixture must retain exactly
   535 `(language,id)` and 150 global ID clusters; a validator that demands 700
   unique IDs is wrong.
9. **Archive fixture.** A tiny tar fixture must reject traversal, symlink,
   duplicate basename, wrong exact member path, missing member, and payload
   mismatch while accepting one exact regular `dev/<basename>` member.
10. **Online/offline parity.** A fresh pinned download and a
    `local_files_only=True` run must produce byte-identical source-lock,
    selection, manifest, and ordered input digests. Network unavailability may
    not fall back from the full commit to `main`.
11. **Downstream invalidation.** Changing any lock/derivation/release field
    must invalidate `teacher-cache.json`, stored model scores, selector input,
    and run ID; failure must occur before teacher/student inference.
12. **Staged-lock race.** Capture lock/source A, replace the live lock with B,
    and prove the fresh staged child both executes and certifies A through final
    publication. Missing the lock from the staged closure must fail.

## Concrete proposal summary

1. Add the independent FLEURS source/release lock and pure record rederiver,
   using the exact current hashes above as positive fixtures.
2. Replace the cached self-hash check with fail-closed semantic equality and
   adversarial label/selection/source/audio tests; preserve repeated source IDs
   while enforcing the correct composite uniqueness keys.
3. After BUG-16 stages the driver and lock, publish a separate schema-bumped
   offline rerun and compare every prediction and metric with result
   `90d51177...397`; expected model gain is none and parity remains unverified.

## Sources

- [FLEURS paper](https://arxiv.org/abs/2205.12446), submitted
  **2022-05-25**.
- [Pinned `google/fleurs` tree](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd),
  revision last modified **2026-05-15**, queried **2026-09-26**.
- [Pinned FLEURS dataset card](https://huggingface.co/datasets/google/fleurs/blob/70bb2e84b976b7e960aa89f1c648e09c59f894dd/README.md?code=true),
  queried **2026-09-26**.
- [Pinned Hub API metadata](https://huggingface.co/api/datasets/google/fleurs/revision/70bb2e84b976b7e960aa89f1c648e09c59f894dd),
  queried **2026-09-26**.
- [Hugging Face Hub 0.24.7 download/cache reference](https://huggingface.co/docs/huggingface_hub/v0.24.7/en/package_reference/file_download),
  accessed **2026-09-26**.
- [Creative Commons Attribution 4.0 deed](https://creativecommons.org/licenses/by/4.0/),
  accessed **2026-09-26**.
- Local experiment driver, manifest, result, report, and reviewer finding,
  audited **2026-09-26**.
