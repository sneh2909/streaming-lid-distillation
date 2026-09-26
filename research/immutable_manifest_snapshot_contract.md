# One manifest generation for corpus and target identity

Research cutoff: **2026-09-26**. Web sources and Hugging Face Hub state were
checked on that date. This note addresses builder backlog **BUG-13** only.

## Bottom line

The race is real and reproducible. Training currently reads the manifest once
for its outer `records`, a second time inside `TeacherTargetCache`, and a third
time while capturing `corpus_identity`. Each read can be internally valid, but
the three reads do not form one transaction. A deterministic A/B/A fixture made
the current launch capture succeed with corpus A and target cache B:

```text
outer records / final live manifest: A
target-cache records:               B
corpus manifest-record digest:      d7db5696...5c5548
target manifest-record digest:      fd4be47e...b6be31
cross-manifest equality:            false
capture_run_dependency_snapshot:    succeeded
```

Atomic publication prevents a torn individual manifest; it does not make
several later path opens observe the same generation. The fix is to read the
manifest **once into an immutable value**, then pass that exact snapshot to the
speaker audit, `TeacherTargetCache`, corpus identity, dataset construction, and
evaluation. The launch/checkpoint contract must also reject unless the corpus,
target metadata, and root manifest binding all carry the same exact-byte and
canonical-record digests.

Expected model-quality gain is **zero**. This is a correctness and provenance
fix. The checked-in run already has equal corpus/target record digests, so this
audit finds no evidence that its checkpoint was mixed. A patched one-step run
and full retrain were not performed here; implementation behavior and overhead
are therefore **unverified**.

## 1. Exact local audit

This audit is bound to Git `54e376441615ea65d5535022cd0b2bb05cfa0529`
and these relevant live-file SHA-256 values at 13:50 IST:

| Path | SHA-256 |
|---|---|
| `scripts/train.py` | `efbd805d...9a8fe87` |
| `src/streaming_lid/data.py` | `0a7b5906...fa6b1c9` |
| `src/streaming_lid/run_identity.py` | `2a972199...591139` |
| `tests/test_run_identity.py` | `0305ac3b...85e4e` |

Unrelated concurrent edits under `scripts/eval.py`, `model.py`, the causality
test, and an experiment directory were present and were not modified.

### 1.1 Three manifest reads create the mixed-generation window

The current launch order is:

1. `scripts/train.py:216` calls `read_manifest()` and receives outer records A;
2. line 218 constructs `TeacherTargetCache`, whose
   `data.py:598-645` independently rereads the pathname and may receive B;
3. lines 226-232 capture dependencies; `corpus_identity()` at
   `run_identity.py:151-170` rereads the pathname and may receive A again; and
4. `capture_run_dependency_snapshot()` places corpus A and target identity B
   next to each other without comparing their manifest-record digests.

The target cache does correctly prove that its own metadata matches the
manifest generation it saw. `corpus_identity()` also correctly proves that its
single byte read parses to the outer records. The missing invariant is across
those two individually valid objects:

```text
snapshot.corpus.manifest_records_sha256
    == snapshot.target_cache.identity.manifest_records_sha256
```

The later equality gates rebuild the same mixed tuple from outer A, live A,
the already-created B cache object, and B metadata. Returning the path to A can
therefore make launch and publication checks agree. Rechecking a mutable name
does not identify an intermediate generation that has disappeared.

### 1.2 A held-out-only B change evades training-time per-record checks

`TeacherTargetCache.load()` compares every requested item with the cache's own
manifest record. A B change to a training row will normally fail when dataset
preload requests that row. A B release differing only in held-out records does
not: optimization consumes outer A's train rows, which can be identical in A
and B, while `validate_all()` certifies B's complete target release. The
checkpoint can then describe corpus A plus targets B and evaluation under A
will reject the B target identity.

This is why the issue is not merely a theoretical digest mismatch. It can
survive all optimizer steps without changing the model inputs and still make
the claimed run bundle internally contradictory.

### 1.3 Deterministic reproduction against current code

The probe used one temporary held-out row, one fixed audio file, and valid
schema-4 target metadata for release B. It performed the same calls as the
trainer, with atomic `os.replace()` between them:

```text
publish A -> read_manifest()                 # outer A
publish B -> TeacherTargetCache()            # validates B metadata
publish A -> capture_run_dependency_snapshot # corpus A, targets B
```

Exact output:

```json
{
  "capture_succeeded": true,
  "corpus_manifest_records_sha256":
    "d7db56965e7c84278233a58be6e54d113c1963dd5f7cf2b1e682cf957c5c5548",
  "cross_manifest_equal": false,
  "final_live_record_text": "release A",
  "outer_record_text": "release A",
  "target_cache_record_text": "release B",
  "target_manifest_records_sha256":
    "fd4be47e7d50f6e29068f43a68f0286c4529c31e6c43399c13a5d6fa60b6be31"
}
```

The probe set `torch.set_num_threads(6)` and did not load a model, target
tensor, or audio decoder. It demonstrates contract acceptance, not malicious
exploitation or a frequency estimate.

### 1.4 The submitted artifact is presently self-consistent

The current live manifest is 164,407 bytes with file SHA-256
`7df6fc5d...9314c`. Its canonical 94-record digest is
`d47a8b62...ecc01`. The live `TeacherTargetCache` has the same record digest.
Both `results/train_metrics.json` and `results/summary.json` bind that same
digest on the corpus and target sides for run
`lidrun-f2d2fbcd...45f2c3`.

This equality is evidence about the retained artifact, not proof that the
current code rejects the race. Conversely, the fixture proves reachability but
does not show that the race occurred during the submitted run.

## 2. Why atomic replacement is necessary but insufficient

The corpus publisher correctly writes content-addressed audio first and uses
`os.replace()` to switch `manifest.jsonl` last. Python 3.10 documents a
successful replacement as atomic
([Python 3.10.21 `os.replace`, documentation updated 2026-09-16](https://docs.python.org/3.10/library/os.html#os.replace)).
Linux likewise specifies that the destination pathname is atomically replaced
and an already-open file descriptor remains attached to its original file
([`rename(2)`, accessed 2026-09-26](https://man7.org/linux/man-pages/man2/rename.2.html),
[`open(2)`, accessed 2026-09-26](https://man7.org/linux/man-pages/man2/open.2.html)).

The consequence is useful but narrower than the present claim:

```text
one open/read  -> complete A or complete B
three opens    -> A/B/A is permitted
```

The second line is an inference from the documented per-operation semantics,
and the local fixture confirms it on this filesystem. Atomic rename is not
snapshot isolation across consumers.

MITRE's CWE-367 describes the general pattern as checking a resource and later
using a resource whose state can change between the operations; the page was
last updated **2026-04-30**
([CWE-367](https://cwe.mitre.org/data/definitions/367.html)). This repository's
finding is primarily a reproducibility/correctness defect. Whether an
untrusted actor can control publication is **unverified**, so this note does
not claim a security vulnerability.

## 3. Hugging Face Hub analogue

Current Hugging Face Hub guidance describes almost the same failure mode. The
v2.0.0 cache guide says that separate file calls resolving mutable `main` can
land on different commits if the repository changes between calls. It
recommends one `snapshot_download()` or resolving the revision once, after
which every file uses the same immutable commit
([Hub v2.0.0 cache guide, accessed 2026-09-26](https://huggingface.co/docs/huggingface_hub/v2.0.0/guides/manage-cache#pin-a-revision-advanced)).
Its cache separates mutable `refs/` from per-commit `snapshots/` and
content-addressed `blobs`.

The project's installed `huggingface-hub==0.24.7` predates the current
`ResolvedRevision` helper, but its version-matched docs already accept a commit
hash as `revision`, store one snapshot directory per commit, and describe
content-addressed blobs
([v0.24.7 file-download reference, accessed 2026-09-26](https://huggingface.co/docs/huggingface_hub/v0.24.7/en/package_reference/file_download)).
The design lesson does not depend on upgrading the package: resolve/capture
once, then hand every consumer the resolved immutable identity.

As a live relevant dataset check, the Hub API reported
[`google/fleurs`](https://huggingface.co/api/datasets/google/fleurs) `main` at
full commit `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, last modified
**2026-05-15**, with 1,004 repository siblings. The exact
[commit tree](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd)
remains addressable. Hugging Face Datasets also documents `revision` as a tag,
branch, or commit hash
([Datasets loading guide, accessed 2026-09-26](https://huggingface.co/docs/datasets/en/loading#hugging-face-hub)).
FLEURS is an analogy and a future evaluation input; no FLEURS audio was
downloaded or scored in this iteration.

SLSA v1.2 gives the same provenance principle: resolved build inputs belong in
`resolvedDependencies` with a digest, and it recommends designing the build
definition as the sole top-level input where possible
([SLSA v1.2 build provenance, accessed 2026-09-26](https://slsa.dev/spec/v1.2/build-provenance)).
This project should not claim SLSA compliance; the relevant point is to bind
the resolved manifest value, not repeated observations of its mutable path.

## 4. Proposed manifest-snapshot contract

### 4.1 Capture one immutable value

Introduce one `ManifestSnapshot` captured before `require_speaker_disjoint()`
or target-cache construction. A robust small representation is:

```text
ManifestSnapshot {
  source_path
  base_directory
  raw_bytes                         # immutable bytes read once
  manifest_file_sha256              # exact-byte identity
  canonical_records_bytes           # immutable canonical JSON array
  manifest_records_sha256           # semantic record identity
  n_records
}
```

Do not expose a shared mutable `list[dict]`. Either parse defensive copies from
`canonical_records_bytes` for consumers or recursively freeze the parsed
value. A frozen dataclass containing a mutable list is not actually immutable.

The single read must decode UTF-8, parse every nonblank JSONL row, reject
duplicate/non-string IDs, compute both digests, and retain the base directory
used to resolve relative audio paths. Under the current atomic-publisher
contract, the captured bytes are a complete generation. If arbitrary in-place
writers must later be supported, file-descriptor/stat stability or a
content-addressed release directory needs a separate design; that scenario is
**unverified** here.

### 4.2 Thread the value through every consumer

Change the data flow to:

```text
capture_manifest_snapshot(path) exactly once
  +-> require_speaker_disjoint(snapshot.records_copy())
  +-> TeacherTargetCache(snapshot, targets_dir)
  +-> corpus_identity(snapshot, audio hashes)
  +-> DistillationDataset(snapshot, target_cache, splits=("train",))
  +-> launch dependency snapshot / checkpoint
```

`TeacherTargetCache` must never reopen `manifest_path`; it should compare its
metadata against the supplied snapshot. `corpus_identity` must use the same
snapshot rather than rereading the path. Dataset construction must consume a
defensive record copy derived from that value. Evaluation needs the identical
pattern: capture its live manifest once, compare that snapshot with the
checkpoint, then build the cache and datasets from it.

Keep live-path equality checks as drift alarms, but make them compare the live
file with the launch snapshot rather than reconstructing the consumed inputs.
An A/B/A change can evade a final drift alarm, but it cannot mix consumers once
all consumers already hold A. The immutable snapshot, not the end recheck, is
the correctness primitive.

### 4.3 Make equality an executable schema invariant

Construct the corpus and target objects first, then fail before target/audio
preload unless all of these are equal:

```text
root_manifest.manifest_records_sha256
corpus.manifest_records_sha256
target_cache.identity.manifest_records_sha256
```

Also bind `root_manifest.manifest_file_sha256` in the run identity. Bump the
target-cache metadata schema to record that exact-byte digest as well as its
existing canonical-record digest, then require both at target generation,
training, and evaluation. Today the target metadata records only the canonical
record digest, so byte-distinct but semantically equal JSONL generations are
not distinguishable on the target side.

A compact checkpoint shape is:

```text
manifest_snapshot {
  schema_version
  manifest_file_sha256
  manifest_records_sha256
  n_records
}
corpus.manifest_snapshot_sha256
target_cache.identity.manifest_snapshot_sha256
manifest_bindings_equal: true
```

The duplicated child references are intentional defense in depth, but the
builder must validate them rather than merely serialize them. `build_run_identity`
and `validate_evaluation_run_contract` should both reject a forged mismatch.
The retained snapshot bytes or a content-addressed path must remain recoverable
for a reproducibility claim; a digest alone proves equality when bytes are
present but cannot recover missing inputs.

### 4.4 Keep the fix proportionate

For this take-home, one 164,407-byte in-memory manifest snapshot plus explicit
digest gates is simpler than cross-process file locking and sufficient under
the existing atomic publisher. A stronger production layout can publish:

```text
corpus/releases/<manifest_file_sha256>/manifest.jsonl
targets/releases/<manifest_file_sha256>/<target_config_sha256>/...
current -> one resolved release descriptor
```

The trainer would resolve `current` once and use the named release thereafter,
mirroring the Hub's ref-to-commit snapshot model. Directory immutability and
garbage collection for that stronger layout are **unverified** and are not
required to close BUG-13.

## 5. Required deterministic tests

1. **Exact A/B/A regression.** Build valid temporary releases in which only a
   held-out record differs. Pause after outer capture A, atomically publish B
   for cache construction, restore A before dependency capture, and prove the
   legacy sequence accepts unequal digests. The patched public entry point must
   reject before `validate_all()`, dataset preload, or optimization.
2. **No second manifest open.** Capture A, monkeypatch subsequent manifest
   pathname reads to raise, and construct the speaker audit, target cache,
   corpus identity, and dataset successfully from A. This verifies data flow,
   not timing luck.
3. **Explicit forged mismatch.** Supply a corpus identity with digest A and a
   target identity with digest B directly to launch identity construction and
   to evaluation validation. Both must fail even if their individual metadata
   hashes and end-of-run path checks are valid.
4. **Mutable-alias defense.** Mutate a list/dict returned to one consumer and
   prove the snapshot digest, target-cache records, and another consumer's
   records remain A.
5. **Exact bytes versus semantic records.** Reorder JSON object keys or change
   whitespace without changing records. The file digest must change while the
   canonical-record digest stays equal. Test and document whether exact target
   generation requires both; this proposal requires both under the new schema.
6. **Live drift.** Capture A, leave B live, and require the pre-preload drift
   gate to fail. Restore A after that failure and prove the aborted run cannot
   be resumed or published as successful.
7. **Evaluation parity.** A one-step checkpoint built from A must validate
   under A and reject B before model scoring. Corpus, target, and root manifest
   child digests must equal the checkpoint's one authoritative snapshot.
8. **Current fixture.** The existing 94-record bundle must pass with exact-byte
   digest `7df6fc5d...9314c` and canonical-record digest
   `d47a8b62...ecc01`; this preserves a known-good path while adding no model
   quality claim.

Avoid a probabilistic threaded test as the only regression. The test should
control publication at named hook/barrier points so the A/B/A ordering is
deterministic on every machine.

## 6. Rejected shortcuts

- **Read twice and require equality.** A/B/A can make the first and last reads
  equal while an intermediate consumer uses B.
- **Add only the missing digest comparison.** This catches the reproduced
  mismatch, but independent consumers still do not share one value and future
  code can reintroduce another gap. Do both: shared snapshot and explicit gate.
- **Rely on atomic `os.replace()`.** It prevents partial publication, not a
  transaction across independent opens.
- **Compare only size, mtime, or inode.** They are not content identities and
  are unnecessary when the exact bytes are already small enough to hash.
- **Lock only inside the trainer.** Advisory locks require every publisher to
  participate and complicate failure recovery. A captured immutable value
  makes later path changes irrelevant to consumers.
- **Store only the hash.** A hash detects a mismatch but cannot supply the
  bytes that downstream consumers must share or recover a missing manifest.

## 7. Concrete builder handoff

1. Add a single-read immutable `ManifestSnapshot`, pass it through every
   train/eval consumer, and remove manifest reopening from `TeacherTargetCache`
   and `corpus_identity`. Expected model gain: **none**. The A/B/A fixture must
   fail before work starts.
2. Add a root manifest binding plus explicit corpus/target equality assertions
   to capture, checkpoint build, and evaluation validation. Bump target
   metadata to bind raw-file and canonical-record digests. Expected numerical
   change: **none**; current-cache migration cost is **unverified** until the
   builder chooses compatibility versus regeneration.
3. Add deterministic no-reread, alias-mutation, byte-versus-semantic, drift,
   and evaluation tests, then run one optimizer step before a full retrain.
   Expected overhead is negligible but **unverified**; measure capture time and
   checkpoint-size change rather than asserting it.

## Sources and dates

- Local repository and artifacts: audited **2026-09-26 13:50 IST**; exact
  hashes are listed above.
- [Python 3.10.21 `os.replace`](https://docs.python.org/3.10/library/os.html#os.replace),
  documentation updated **2026-09-16**.
- Linux [`open(2)`](https://man7.org/linux/man-pages/man2/open.2.html) and
  [`rename(2)`](https://man7.org/linux/man-pages/man2/rename.2.html), accessed
  **2026-09-26**.
- [MITRE CWE-367](https://cwe.mitre.org/data/definitions/367.html), page last
  updated **2026-04-30**.
- [Hugging Face Hub v2.0.0 cache guide](https://huggingface.co/docs/huggingface_hub/v2.0.0/guides/manage-cache#pin-a-revision-advanced),
  [installed-version v0.24.7 download reference](https://huggingface.co/docs/huggingface_hub/v0.24.7/en/package_reference/file_download),
  and [Datasets loading guide](https://huggingface.co/docs/datasets/en/loading#hugging-face-hub),
  accessed **2026-09-26**.
- [`google/fleurs` Hub API](https://huggingface.co/api/datasets/google/fleurs)
  and [pinned tree](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd),
  API `lastModified` **2026-05-15**, checked **2026-09-26**.
- [SLSA v1.2 build provenance](https://slsa.dev/spec/v1.2/build-provenance),
  approved v1.2 specification, accessed **2026-09-26**.
