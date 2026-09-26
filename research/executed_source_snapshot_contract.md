# Bind the source Python actually executed

Research cutoff: **2026-09-26**. Web sources and Hugging Face Hub state were
checked on that date. This note addresses builder backlog **BUG-12** only.

## Bottom line

The current launch snapshot is too late to identify the code that trained the
model. `scripts/train.py` imports every project module at lines 18--44, but
`capture_run_dependency_snapshot()` does not read the source paths until line
226. Python normally satisfies later imports from the already-created module
objects in `sys.modules`. If version A was imported and the live paths are then
replaced by version B, the current helper hashes B while training continues to
call A. Stable launch/end hashes of B do not repair that mismatch.

Use a small standard-library-only parent launcher to create and publish a
content-addressed source tree **before starting the training interpreter**.
Train and evaluate in fresh child processes whose local imports resolve only
inside that tree. Store the tree manifest/root digest and an import-origin
ledger in the checkpoint, and rehash the staged tree before publication. This
is the same useful separation the Hugging Face cache makes between a mutable
ref such as `main` and a resolved snapshot directory named by a full commit.

This change has expected model-quality gain **zero**. It makes the provenance
claim truthful and makes concurrent repository edits unable to change a run
that has already started. A complete staged training/evaluation run has not
been executed in this research iteration, so runtime parity is **unverified**.

## 1. Exact local audit

### 1.1 The snapshot reads path bytes after import execution

The relevant order is:

1. the interpreter executes `scripts/train.py`;
2. lines 14--44 import NumPy, Torch, and `streaming_lid` modules;
3. importing `streaming_lid.config` first executes the parent package
   `streaming_lid/__init__.py`;
4. `main()` reads the manifest and constructs `TeacherTargetCache`; and only
5. lines 226--232 call `capture_run_dependency_snapshot()`, whose
   `pipeline_source_identity()` rereads named paths from the working tree.

Python 3.10's import reference says that an import of `foo.bar` first imports
`foo`, and that an entry already present in `sys.modules` completes the import
search using the existing module object. The loader executes module code only
when it initially loads that object. Thus `module.__file__`, `__spec__.origin`,
`inspect.getsource()`, or a later read of that path describes a location, not
proof of the bytes that populated an already-loaded namespace
([Python import system, version 3.10.21, documentation updated 2026-09-16](https://docs.python.org/3.10/reference/import.html)).

The dangerous A/B schedule is therefore concrete:

```text
import train/config/data/loss/model/run_identity from source A
atomically replace the named working-tree files with source B
capture source hashes by reading the paths -> hashes B
train using functions/classes/constants already loaded from A
rehash unchanged B at publication -> equality gate passes
```

`pipeline_configuration()` can even become a hybrid: its function and imported
configuration objects are A, while its nested `pipeline_source_identity()`
reads B. A change confined to the loss/model/training implementation can leave
the scalar configuration equal and evade evaluation's current comparison.

### 1.2 The hand-maintained source list already omits executed code

At 13:40 IST on 2026-09-26, the current `PIPELINE_SOURCE_FILES` covered eight
paths and produced source identity
`d52b65911bb28a5ef9a9d8938f12024f6ac7e718097986c3d257707c01cb6374`.
It includes `scripts/eval.py`, which is not executed by the training process,
but omits `src/streaming_lid/__init__.py`, which Python necessarily executes
before its submodules. The omitted file is 150 bytes with SHA-256
`86291e2217654d75f83aece033713de7117969f3b883646ebb4830c8e404e100`.

A fresh runtime-origin probe loaded seven local module objects, all through
`SourceFileLoader` from the live editable tree:

| Loaded module | Origin under live `src/` |
|---|---|
| `streaming_lid` | `streaming_lid/__init__.py` |
| `streaming_lid.config` | `streaming_lid/config.py` |
| `streaming_lid.audio` | `streaming_lid/audio.py` |
| `streaming_lid.data` | `streaming_lid/data.py` |
| `streaming_lid.loss` | `streaming_lid/loss.py` |
| `streaming_lid.model` | `streaming_lid/model.py` |
| `streaming_lid.run_identity` | `streaming_lid/run_identity.py` |

The complete current training/evaluation Python tree is only nine files:
those seven package files plus `scripts/train.py` and `scripts/eval.py`. A
sorted `sha256sum`-line audit over those nine paths yielded
`02dc6cd9affe0c92fa309326aae10854a6b571f1b78c476a42272cfec391f096`.
That audit digest is not the project's canonical-JSON identity and must not be
substituted for it; it only demonstrates the closure checked here.

The correct durable rule is not another manually maintained import list.
Snapshot the complete regular-file package tree plus the named entry points,
then prove after import that every local module origin is represented in the
manifest. Extra staged files are harmless; one omitted executed file defeats
the claim.

### 1.3 `-I` alone does not isolate this editable installation

The installed distribution is editable. Its 41-byte
`_editable_impl_streaming_lid_distillation.pth` contains the absolute live
workspace `src` path. In the current `.venv`, an exact
`.venv/bin/python -I -c ...` probe still produced:

```text
/mnt/d/Work/Projects/asr-navana-codex/.venv/lib/python3.10/site-packages
/mnt/d/Work/Projects/asr-navana-codex/src
```

This is consistent with the documented contracts: `-I` excludes the current
directory and *user* site, while site initialization processes `.pth` files in
site-package directories and adds their entries to `sys.path`
([Python command-line `-I`, updated 2026-09-16](https://docs.python.org/3.10/using/cmdline.html),
[Python `site` and `.pth` processing, updated 2026-09-16](https://docs.python.org/3.10/library/site.html)).

Therefore a staged child must also remove/reject the editable live-tree entry,
or run with `-S` and add the exact third-party site-package directory without
processing `.pth` files. A small `-I -S` smoke probe succeeded after explicitly
adding that directory for Torch 2.6.0+cpu, torchaudio 2.6.0+cpu, NumPy 2.2.6,
and SoundFile 0.13.1, but the full pipeline under this mode is **unverified**.

## 2. What external practice supports—and what it does not

### 2.1 Resolve a moving name to one content snapshot

The repository pins its ECAPA teacher to full Hub commit
`0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`. On 2026-09-26 the exact
[Hub tree at that revision](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9)
was still available, and the [commit page](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/commit/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9)
dates it to **2024-11-27**. The local Hub cache separates `refs/main` from
`snapshots/0253049...`, whose model files point to content blobs.

The project's installed `huggingface-hub==0.24.7` documentation says that a
full commit hash can select a specific version, `snapshot_download()` fetches
the repository at a revision, and returned cache files must not be modified
([version-matched download guide, accessed 2026-09-26](https://huggingface.co/docs/huggingface_hub/v0.24.7/en/guides/download)).
That is a useful design precedent, not proof that ordinary cache permissions
are an adversarial immutability boundary. The local source stage should copy
regular bytes, reject symlinks that escape the stage, and verify its own
manifest rather than trusting a directory name.

For a clean release, Git can provide the same primitive: a commit refers to a
tree snapshot, and `git archive <full-commit>` emits that tree
([Git archive manual, last updated 2026-04-20](https://git-scm.com/docs/git-archive)).
That route excludes uncommitted work. A developer run therefore still needs a
byte snapshot of the working tree, explicitly labeled as such.

### 2.2 Provenance must name resolved inputs

SLSA v1.2 describes build provenance as the information needed to trace an
artifact to where, when, and how it was produced. Its build-provenance schema
puts resolved dependency digests—not merely mutable input names—in
`resolvedDependencies`
([SLSA v1.2 build provenance, approved/current in 2026, accessed 2026-09-26](https://slsa.dev/spec/v1.2/build-provenance)).
This take-home need not claim SLSA compliance, but the distinction is exactly
right: the checkpoint should bind the staged source digest that was executed,
while `workspace_path` and Git branch are descriptive metadata only.

### 2.3 Hash-based `.pyc` files are not the fix

PEP 552 makes a `.pyc` a deterministic function of source bytes and can check
the source hash when a cached file is selected. It does not rebind functions
that are already alive in `sys.modules`, and it does not prove that a later
pathname read matches an earlier execution
([PEP 552, final; last modified 2025-02-01](https://peps.python.org/pep-0552/)).
Similarly, `importlib.reload()` is not a provenance repair: Python documents
that outside references imported with `from ... import ...` are not rebound,
and existing instances keep old class definitions
([Python `importlib`, updated 2026-09-16](https://docs.python.org/3.10/library/importlib.html#importlib.reload)).

## 3. Proposed execution contract

### 3.1 Parent launcher: snapshot before project import

Create a new launcher that imports only the standard library and never imports
`streaming_lid`, NumPy, or Torch. It should:

1. collect `scripts/train.py`, `scripts/eval.py`, a staged child bootstrap,
   and every regular `*.py` file recursively under `src/streaming_lid/`;
2. read each source once, record relative path, byte count, SHA-256, and mode,
   reject symlinks/non-regular files, and compute a canonical manifest digest;
3. write a sibling temporary tree with exactly those captured bytes, verify it,
   and atomically rename it to a content-addressed directory such as
   `data/generated/source_snapshots/<root_sha256>/`;
4. publish the canonical manifest inside that tree and make the files
   read-only as an accidental-mutation guard; and
5. spawn a **fresh** child interpreter against that tree.

The child, not the live worktree, is the model-producing process. If the live
tree changes while the parent is copying, the captured byte set remains the
truthful source of that run. For a release run, prefer an archive from an
already-resolved full Git commit and record both the commit and the independently
verified tree digest.

### 3.2 Child bootstrap: prevent import escape

Before importing any project module, the staged child should:

- insert `<snapshot>/src` at the front of `sys.path`;
- remove every path resolving inside the live workspace, including the path
  injected by the editable `.pth` file, or use a tested `-I -S` environment;
- reject any pre-existing `streaming_lid` entry in `sys.modules`;
- verify every staged byte against the manifest;
- execute the staged `scripts/train.py`; and
- after imports, require every `streaming_lid*` module's resolved
  `__spec__.origin` to lie inside the snapshot and appear in the manifest.

The origin ledger is a coverage check, not the primary identity. Reading an
origin after import has the same old race unless the origin is the private,
verified stage. Rehash the stage before checkpoint publication so a changed
or deleted stage aborts. Lazy project imports must also resolve under the same
regular package `__path__`.

### 3.3 Checkpoint identity and evaluation handoff

Replace the current path-reread source object with an executed-source object:

```text
executed_source {
  schema_version
  snapshot_root_sha256
  manifest: {relative_path -> {sha256, bytes, mode}}
  snapshot_kind: committed_git_tree | captured_worktree
  git_commit | null
  git_tree | null
  child_entrypoint
  local_module_origins
  workspace_escape_checked: true
  stage_verified_before_import: true
  stage_verified_before_publication: true
}
```

`scripts/eval.py` must run from the same retained source snapshot, or from a
separately staged tree whose digest exactly equals the checkpoint's required
evaluation source. Do not make evaluation reread the current worktree and call
that historical evidence. A live-tree end hash can remain as a release-policy
field such as `workspace_changed_after_snapshot`; it must never replace the
executed snapshot identity.

Also bind the Python version, `pyproject.toml`, `uv.lock`, installed
distribution versions/build tags, and the site-path policy as a separate
environment object. The current observed runtime is Python 3.10.12,
Torch/torchaudio 2.6.0+cpu, NumPy 2.2.6, SoundFile 0.13.1,
SpeechBrain 1.0.3, and huggingface-hub 0.24.7. Version recording is useful but
does not establish byte-for-byte environment identity; cross-host numerical
reproducibility remains **unverified**.

## 4. Required adversarial tests

1. **Import A / replace live path with B.** Build a tiny temporary A package
   whose loss/model function returns an A sentinel. Pause after staging, replace
   the live package atomically with B, and release the child. It must execute A,
   record A's digest, and show every local origin under the staged A tree. The
   current path-reread helper should be retained in the test only long enough
   to prove it would report B.
2. **Replace before the launch snapshot.** Import A in the old in-process
   layout, replace only `loss.py` with B while keeping configuration equal,
   then show that the old launch/end equality can pass while the callable still
   returns A. This is the exact regression BUG-12 describes.
3. **Package initializer coverage.** Mutate only staged
   `streaming_lid/__init__.py`; the snapshot root must change. Remove it from a
   forged manifest; pre-import verification or the post-import origin-coverage
   check must fail.
4. **Editable-install escape.** Leave the current live `src` path in the
   virtualenv `.pth`, make live and staged modules return different sentinels,
   and require the staged result. Assert no origin resolves to the workspace.
5. **Stage tamper and deletion.** Change one staged byte, add an unexpected
   local Python file, delete a declared file, and replace a file with a symlink;
   each case must fail before optimization or publication.
6. **Lazy import.** Import a helper for the first time after one optimizer
   step. It must come from the same snapshot and appear in the final origin
   ledger; a helper available only in the live tree must be unimportable.
7. **Training/evaluation parity.** Run a one-step CPU fixture twice from the
   same source snapshot while the live tree changes between invocations. Source
   identity and deterministic, timing-excluded outputs must match. Then change
   one staged semantic byte and require a new source/run identity.

No full model retrain is needed to close the race. A one-step real-audio run,
the adversarial fixtures above, and the existing 1,600-step result only after
the new gate passes are proportionate evidence.

## 5. Rejected shortcuts

| Shortcut | Why it does not close BUG-12 |
|---|---|
| Hash `module.__file__` or `__spec__.origin` after import | Those are paths and may now contain B while the module object contains A. |
| Call `loader.get_source()` or `inspect.getsource()` | A file loader may reread the current path; it is not an execution transcript. |
| Serialize code objects | It omits a reliable complete source/dependency closure and is sensitive to interpreter compilation; dynamic and extension code remain outside it. |
| Delete from `sys.modules` and re-import | Existing `from ... import ...` bindings and instances can retain A; reload semantics are explicitly partial. |
| Add only `-I` | The current virtualenv still adds the live editable `src` through its site `.pth`. |
| Add `__init__.py` to the existing list | It repairs one omission but leaves the post-import A/B race and future import-closure drift. |
| Use a branch name or current `HEAD` at the end | A moving name is descriptive; the executed byte snapshot/full resolved commit is the dependency. |
| Hold an advisory lock on each live file | Atomic rename can replace a pathname/inode, and unrelated writers need not honor the lock. A private staged tree is simpler. |

## 6. Expected effect and limits

- Expected accuracy, loss, parameter, latency, and CPU-RTF change: **none**.
- Expected provenance effect: an import-A/path-B checkpoint becomes impossible
  under the stated accidental-concurrency model; the child either executes and
  certifies the same staged bytes or aborts.
- Stage construction time, full-pipeline compatibility with `-I -S`, and exact
  numerical parity are **unverified** until the builder runs the fixture.
- Read-only modes are not a security boundary against a privileged/malicious
  writer. A read-only container/mount or trusted build service is needed for a
  stronger adversarial guarantee.
- This closes local project-source identity only. It does not by itself prove
  dependency-wheel bytes, native libraries, kernel/CPU behavior, dataset
  identity, or deterministic training across machines.

## Source record

Primary/official sources checked **2026-09-26**:

- [Python 3.10.21 import system](https://docs.python.org/3.10/reference/import.html),
  [command line](https://docs.python.org/3.10/using/cmdline.html),
  [`site`](https://docs.python.org/3.10/library/site.html), and
  [`importlib`](https://docs.python.org/3.10/library/importlib.html), all PSF
  documentation last updated **2026-09-16**.
- [PEP 552: deterministic pycs](https://peps.python.org/pep-0552/), final;
  page last modified **2025-02-01**.
- [Hugging Face Hub 0.24.7 download guide](https://huggingface.co/docs/huggingface_hub/v0.24.7/en/guides/download),
  version-matched to this repository and accessed **2026-09-26**; exact
  [ECAPA revision](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9),
  committed **2024-11-27**.
- [Git `archive` manual](https://git-scm.com/docs/git-archive), last updated
  **2026-04-20**.
- [SLSA v1.2 build provenance](https://slsa.dev/spec/v1.2/build-provenance),
  approved/current 2026 specification, accessed **2026-09-26**.

