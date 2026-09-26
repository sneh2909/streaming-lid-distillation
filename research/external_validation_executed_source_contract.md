# External-validation executed-source and publication contract (BUG-16)

Research, web, and Hugging Face Hub cutoff: **2026-09-26
(Asia/Calcutta)**.

## Decision

The schema-1 external-validation result does not prove which local source its
interpreter executed. The driver imports NumPy, Torch, `scripts.teacher_targets`,
and the `streaming_lid` package before its first source snapshot; Python then
keeps those loaded objects independently of later pathname contents. Its
handwritten source list also omits the executed package initializer, and its
last source check occurs before more local functions run, before result
construction, and before the atomic publication.

The correct repair is to extend the existing main-pipeline source stage to this
experiment. A standard-library-only parent should capture the experiment
driver, the complete local Python/config closure, and dependency locks before
any numerical or project import; a fresh child should execute only from that
verified stage; and the same executed-source/environment identity should flow
into the teacher cache, experiment run ID, and final result. Reverify the stage
and all bound inputs at the final `os.replace()` publication boundary.

This work has **zero expected model or accuracy gain**. Exact metric parity,
runtime overhead, and the result hash of a repaired schema-2 rerun are
**unverified** until that rerun is performed. There is no evidence that the
historical run actually experienced a source swap: its declared hashes are
recoverable and its arithmetic has been independently reviewed. Preserve it as
historical diagnostic evidence rather than retroactively relabelling it.

This note addresses backlog **BUG-16 only**. The semantic FLEURS derivation
problem (BUG-17), reusable ECAPA prediction lineage (BUG-18), selector gates,
and clustered uncertainty are separate even though their repaired artifacts
should consume this source identity.

## Exact local audit

### The first source observation is after execution has begun

The current driver order is visible in
[`run.py`](../experiments/checkpoint-external-validation/run.py):

1. lines 17--38 import standard-library modules, NumPy, SoundFile, and Torch;
2. lines 41--45 add the live repository and `src` directories to `sys.path`;
3. lines 51--78 import functions, classes, and constants from
   `scripts.teacher_targets` and five `streaming_lid` submodules;
4. `main()` starts at line 1127; and
5. only line 1133 calls `source_snapshot()` for the first time.

Python 3.10 documents that import lookup checks `sys.modules` first and returns
the existing module object when present. It also warns that external references
to previously imported objects are not rebound by reloading another module
([Python 3.10 import system](https://docs.python.org/3.10/reference/import.html),
[Python 3.10 `importlib`](https://docs.python.org/3.10/library/importlib.html),
both accessed **2026-09-26**). Therefore later hashes of `module.__file__` or a
named path are not a transcript of the bytes that populated the running
namespace.

The reachable schedule is:

```text
execute driver A and import project objects A
replace the live paths with stable B
source_at_start hashes B
score/select/report with already-loaded A objects
the end snapshot hashes unchanged B and passes
publish an identity that names B
```

A fresh child is the useful boundary because it starts without the parent's
project-module cache. Python recommends `sys.executable` when launching the
current interpreter again
([Python 3.10 `subprocess`](https://docs.python.org/3.10/library/subprocess.html),
accessed **2026-09-26**).

### Deterministic A/B/C probe

A temporary-directory probe imported a one-line module A, changed its pathname
to B, and then called the production driver's exact `source_snapshot()` twice.
Both snapshots certified B while the loaded object still returned A. After the
second check, the probe changed the pathname to C and called the production
`atomic_json_write()` with the already-approved claim:

```text
executed value                         A
executed A SHA-256                     e7f788c2...82e07f
certified B SHA-256 at start and end   5db0c1b2...60c27b
start snapshot == end snapshot         true
published source_snapshot_unchanged    true
live C SHA-256 at publication          34411ed4...eb7a9
live publication bytes == certified B  false
```

This used the exact current driver utilities, a six-thread Torch setting, and
only `/tmp`; no retained artifact was edited. It proves both races are reachable
under the current contract. It does **not** prove they occurred in the retained
71.23-second run.

### The declared closure omits code Python actually executed

The retained
[`results.json`](../experiments/checkpoint-external-validation/results.json)
declares **7 files / 161,974 bytes** in `experiment_identity.source_files`.
A direct origin probe of the driver observed these local files during import:

```text
experiments/checkpoint-external-validation/run.py
scripts/teacher_targets.py
src/streaming_lid/__init__.py
src/streaming_lid/audio.py
src/streaming_lid/config.py
src/streaming_lid/data.py
src/streaming_lid/model.py
src/streaming_lid/run_identity.py
```

The handwritten list omits `src/streaming_lid/__init__.py`. Python executes
that package initializer before importing the requested package submodules; the
probe's `__spec__.origin` and `__file__` both resolve to it. At the historical
snapshot it is 150 bytes with SHA-256
`86291e2217654d75f83aece033713de7117969f3b883646ebb4830c8e404e100`,
so the minimum observed local execution set is **8 files / 162,124 bytes**.

The official importlib API exposes a module's import origin through
`module.__spec__.origin`; Python notes that it is normally the same as
`module.__file__`, while also warning that the two fields are not synchronized
if mutated at runtime
([Python `ModuleSpec`](https://docs.python.org/3.10/library/importlib.html#importlib.machinery.ModuleSpec),
accessed **2026-09-26**). An origin ledger is consequently a useful assertion
inside an already verified stage, not a replacement for pre-import capture.

Adding only `__init__.py` is not durable. A later lazy helper would recreate
the omission while `SOURCE_FILES` remained unchanged. The live main-pipeline
[`source_stage.py`](../scripts/source_stage.py) already captures all
`scripts/*.py`, every `src/streaming_lid/**/*.py`, `pyproject.toml`, and
`uv.lock`. At **2026-09-26 15:46 IST**, that was 14 files / 380,756 bytes;
including the external-validation driver made 15 files / 436,632 bytes. These
counts are a time-bound worktree diagnostic, not a permanent allowlist. The
future BUG-17 source/release lock must also enter the stage when it exists.

### Historical source is recoverable, but recovery is not execution proof

All seven source hashes declared by result
`8ec024940fdd6f38731cc26f2d2f5a5ec962bdb9018d6f3a395f49b30a6097ea`
match autosave commit
`041e727d9670b925cd73dd9ab7bd8d9b1cfd7cea` exactly. The checked result has
SHA-256
`90d511775dcc6ff0ae49b7ef6e380eb4d6d573d57f0affc269f812beee1bb397`,
and its declared driver hash is
`6707d18f36eb115923ed7ff3c27b9e9111b6ae256704401bb94fccab73c8b196`.
The same commit contains the omitted initializer plus:

| Reproducibility material | Bytes | SHA-256 | In schema-1 source identity? |
|---|---:|---|:---:|
| `src/streaming_lid/__init__.py` | 150 | `86291e22...04e100` | no |
| `pyproject.toml` | 867 | `d203d763...cfce67` | no |
| `uv.lock` | 132,888 | `b1155aa9...eb0c` | no |

This is positive recoverability evidence. It establishes a plausible source
candidate for a new rerun, not what the historical interpreter loaded. The
result does not bind that Git commit, and its snapshots were observed only
after imports.

### Runtime is descriptive, not part of the run ID

The result records Python, platform, six package versions, and Torch thread
count under the top-level `environment` object. `run_id`, however, is the hash
of `experiment_identity`, which does not include `environment`,
`pyproject.toml`, or `uv.lock`.

A read-only reconstruction exactly reproduced the stored run ID. Changing the
recorded NumPy version to `999.0.0` changed the full result payload but left the
recomputed run ID exactly
`8ec024940fdd6f38731cc26f2d2f5a5ec962bdb9018d6f3a395f49b30a6097ea`.
This does not show that the retained version is false; it shows that a different
runtime can retain the same identity.

SLSA v1.2 treats the process definition and resolved dependencies as provenance
inputs and describes provenance as the information needed to establish where,
when, and how an artifact was produced. It also cautions that unsigned local
provenance is easy to forge at its lowest level
([SLSA v1.2 Build Provenance](https://slsa.dev/spec/v1.2/build-provenance),
[Build Track basics](https://slsa.dev/spec/v1.2/build-track-basics), Approved;
accessed **2026-09-26**). This project need not claim SLSA compliance, but it
should follow the applicable distinction: runtime/lock identity belongs in the
run identity, not only in descriptive output.

### The final check is not at the publication boundary

The last `source_snapshot()` comparison is at lines 1243--1244. After it, the
driver still:

- rehashes reference files through imported `file_sha256`;
- calls the driver-defined external-manifest validator;
- constructs configuration and experiment identity;
- derives the run ID;
- executes roughly 95 lines of result assembly;
- checks JSON finiteness; and
- writes and replaces the output at line 1390.

`atomic_json_write()` correctly writes a sibling temporary file and uses
`os.replace()`. Python documents that a successful same-filesystem replace is
atomic
([Python 3.10 `os.replace`](https://docs.python.org/3.10/library/os.html#os.replace),
accessed **2026-09-26**). Atomic visibility protects readers from a partial
JSON file; it does not make a pre-publication source assertion true. The A/B/C
probe above published a complete, valid JSON object containing a stale source
claim.

Once execution comes from a retained content-addressed stage, later *live*
worktree changes should no longer abort the run; they are irrelevant to the
child's executed bytes. The publication gate must reverify the retained stage
and bound inputs. A separately named `live_workspace_changed_after_capture`
field may remain diagnostic, but must not be confused with execution identity.

## Hugging Face Hub audit

The external materials are substantially better pinned than the local
producer. A live query with installed `huggingface-hub==0.24.7` on
**2026-09-26** returned:

| Hub artifact | Exact revision | Last modified | Access/license metadata | Files |
|---|---|---|---|---:|
| [`google/fleurs`](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd) | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | 2026-05-15T09:35:34Z | public, ungated, CC BY 4.0 | 1,004 |
| [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9) | `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9` | 2024-11-27T20:12:38Z | public, ungated, Apache-2.0 | 9 |

The exact API endpoints are
[`FLEURS revision`](https://huggingface.co/api/datasets/google/fleurs/revision/70bb2e84b976b7e960aa89f1c648e09c59f894dd)
and
[`ECAPA revision`](https://huggingface.co/api/models/speechbrain/lang-id-voxlingua107-ecapa/revision/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9).
The ECAPA response still reports the consumed `classifier.ckpt` as 762,555
bytes with LFS SHA-256 `a50d9024...57adbc` and `embedding_model.ckpt` as
84,474,355 bytes with LFS SHA-256 `ab750d5c...9a19a`.

The version-matched Hub guide says downloads are version-aware, that
`snapshot_download()` accepts a revision, and that a commit pin must use the
full hash
([Hugging Face Hub 0.24.7 download guide](https://huggingface.co/docs/huggingface_hub/v0.24.7/en/guides/download#from-specific-version),
accessed **2026-09-26**). These pins establish remote input candidates. They do
not establish which local Python functions consumed those bytes, whether the
cached FLEURS labels were correctly derived, or which producer generated the
teacher cache. BUG-16 supplies only the missing local execution link.

## Proposed schema-2 execution contract

### Extend the generic stage instead of adding a third snapshot mechanism

Generalize `scripts/source_stage.py` to accept one or more additional tracked
entry points/source roots. For this experiment the parent should:

1. import only standard-library bootstrap code;
2. before NumPy, SoundFile, Torch, SpeechBrain, `scripts`, or `streaming_lid`,
   read each regular source/config file exactly once;
3. capture all `scripts/*.py`, all `src/streaming_lid/**/*.py`, the external
   driver, `pyproject.toml`, `uv.lock`, and the future BUG-17 release lock;
4. reject symlinks, non-regular files, missing files, duplicate relative paths,
   and path escapes;
5. derive a canonical root digest from relative path, mode, byte count, and
   SHA-256, then atomically retain that tree under an ignored content-addressed
   directory; and
6. launch the staged driver with a fresh `sys.executable` child.

The parent bootstrap is not the evidence-producing evaluator. The child must
reverify the stage before imports, run the experiment from the staged
entrypoint, and retain the stage for later offline inspection.

### Make the child prove every local origin

Before numerical/project imports, the child should reject preloaded local
modules, remove every live-workspace project path (including editable `.pth`
entries), put only the staged `src`, `scripts`, and required stage root ahead of
third-party paths, and disable bytecode writes. After imports, record and
validate:

- the staged driver's exact `sys.argv[0]` and `__file__`;
- `scripts.teacher_targets` origin;
- `streaming_lid.__init__` and every loaded `streaming_lid.*` origin;
- every package `__path__` entry;
- absence of undeclared local module origins; and
- unchanged stage file set, bytes, modes, and symlink status.

The current main source stage imports every package module as a conservative
closure test. Retain that behavior and add the experiment driver/helper
origins; do not fall back to a handwritten seven-file allowlist.

### Bind source, environment, and consumers once

A replacement `executed_source` object should contain at least:

```text
schema_version
snapshot_kind: committed_git_tree | captured_worktree
snapshot_root_sha256
git_commit | null
entrypoint: {relative_path, sha256}
files: {relative_path -> {sha256, bytes, mode}}
local_module_origins
site_path_policy
runtime: {python, distributions, project_lock_files}
stage_verified_before_import: true
stage_verified_before_publication: true
workspace_escape_checked: true
```

Use that exact immutable value in:

- `experiment_identity` before deriving the run ID;
- `results.json` and its identity audit;
- the BUG-18 teacher prediction producer identity;
- the accepted teacher-cache identity; and
- any normalized report-generation input.

Do not regenerate equivalent-looking dictionaries independently. Equality of
one shared value is simpler to audit than agreement among separately sampled
live paths.

### Move verification into the atomic publication transaction

The child should construct and serialize the full candidate JSON to a sibling
temporary file, re-read and validate that candidate, then immediately before
`os.replace()`:

1. reverify the retained source stage;
2. reverify trajectory/result/state inputs;
3. reverify the staged BUG-17 release and selected audio identity;
4. validate and bind the accepted BUG-18 teacher-cache payload; and
5. assert that the candidate's identities equal those exact values.

After `os.replace()`, re-read the published bytes and record their SHA-256 for
the report. A hash stored only inside the object cannot authenticate itself;
this remains an unsigned local reproducibility contract, not protection from an
active writer controlling both artifact and metadata. Stronger signing or an
independent verifier is **out of scope and unverified**.

Keep schema 1 and result `90d51177...b397` unchanged. Publish a separately
named schema-2 rerun after BUG-16/17/18, and compare all student predictions,
teacher predictions, metrics, selection, and verdict against the historical
artifact. Expected parity is **unverified**. The existing rejection does not
need provenance inflation to remain a useful negative diagnostic.

## Required regressions

1. **Import A / paths B:** import an A helper, replace live paths with B, and
   prove the legacy routine certifies B while executing A.
2. **Stage A / live B:** capture A, replace live paths with B, and require the
   fresh child to execute and certify staged A with no live origin.
3. **Driver origin:** execute a sentinel unique to the staged experiment driver
   and bind its exact path/hash.
4. **Package initializer:** mutate only `streaming_lid/__init__.py`; stage/root
   identity and run ID must change.
5. **Lazy helper:** import an otherwise unused staged local module late; accept
   its staged origin and reject a live-only helper.
6. **Editable-path escape:** inject the workspace through a `.pth`/`sys.path`
   entry and require pre-import removal plus post-import rejection.
7. **Stage tamper:** independently reject changed bytes, deletion, extra file,
   symlink substitution, mode change, and entrypoint mismatch.
8. **Runtime/lock drift:** change a bound package version, Python identity,
   `pyproject.toml`, or `uv.lock`; cache reuse and run identity must invalidate.
9. **Publication barrier:** mutate the retained stage after result assembly but
   before `os.replace()`; publication must abort and preserve the old result.
10. **Live drift after capture:** change only the live worktree after staging;
    the child should finish from A, record diagnostic drift, and still certify
    A rather than relabel execution as B.
11. **Consumer equality:** result, teacher producer/cache, and report input must
    carry the identical executed-source root; any mismatch fails before score
    publication.
12. **Fresh/offline parity:** one fresh schema-2 teacher pass and one offline
    reuse under the same stage must produce equal raw prediction and metric
    digests; exact parity with schema 1 is separately reported, never assumed.

## Concrete recommendation

Repair BUG-16 together with BUG-17 and BUG-18 before publishing another
external-validation result. The minimum acceptance condition is a schema-2
fresh-child run whose source stage is verified before imports and at
publication, whose module origins cannot escape to the live workspace, whose
runtime/locks and accepted cache enter the run ID, and whose fresh/offline
reuse agrees exactly. Expected model gain is **none**; expected evidence gain
is that `source_snapshot_unchanged=true` becomes a claim about the bytes that
actually executed rather than two reads of mutable pathnames.
