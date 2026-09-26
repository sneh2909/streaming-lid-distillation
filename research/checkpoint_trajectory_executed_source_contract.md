# Bind the checkpoint-trajectory experiment to the source it executed

Research cutoff: **2026-09-26**. Web pages and Hugging Face Hub state were
checked on that date. This note addresses builder backlog **BUG-15** only. It
extends the main-pipeline BUG-12 source-snapshot work to the experiment driver;
it does not revisit checkpoint-selection quality or transition matching.

## Bottom line

`experiments/checkpoint-trajectory/run.py` certifies source too late. Python has
already executed the experiment driver through line 85 and imported
`scripts.eval`, `scripts.train`, and the complete `streaming_lid` package before
`main()` first calls `snapshot_sources()` at line 1172. If those live paths move
from version A to stable version B in between, both the start and end snapshots
describe B while the process continues to call A objects already held in
memory.

The exact final-weight/loss/gradient replay is strong numerical evidence for
the training path, but it is not source provenance. In particular, it cannot
detect an A/B substitution confined to `smooth_posteriors`, switch scoring,
checkpoint selection, result construction, or the already-running experiment
driver. Those functions can change the selected result without changing a
single optimizer update.

Do not add another post-import hash. Reuse one generic pre-import source-stage
launcher for the main pipeline and experiments: resolve or capture a complete
regular-file source tree, verify and retain it, then start a fresh interpreter
whose experiment entry point and every local import come from that tree. Bind
the staged root digest and import-origin ledger into both `results.json` and the
trajectory artifact.

Expected model-quality gain is **zero**. A repaired rerun's numerical parity,
staging overhead, and selected checkpoint are **unverified** until executed.

## 1. Exact local audit

### 1.1 The experiment has the same A/B race as training, plus a self-race

The current execution order is:

1. the interpreter reads and executes the live experiment driver;
2. lines 33--36 import NumPy and Torch;
3. lines 49--85 bind functions, classes, and constants from `scripts.eval`,
   `scripts.train`, and six `streaming_lid` submodules;
4. `main()` begins at line 1164; and only then
5. line 1172 rereads source pathnames through `snapshot_sources()`.

Python's import documentation says the import search checks `sys.modules`
first and, when a name is present, returns the existing module object. Deleting
and reimporting can even leave two different module objects alive when other
references remain. Thus a later read of `module.__file__`, `__spec__.origin`,
or the named pathname is not a transcript of bytes that populated an already
loaded namespace
([Python 3.10.21 import system, accessed 2026-09-26](https://docs.python.org/3.10/reference/import.html)).

The experiment-specific failure schedule is:

```text
execute driver A and import eval/train/package A
atomically replace the live named paths with stable B
snapshot_sources() reads B
train/evaluate/select/report with driver and imported objects A
end snapshot rereads unchanged B and passes
publish an identity that claims B
```

This is broader than the main trainer race because the running `__main__`
driver itself defines evaluation, selector, identity, and publication logic.
It cannot truthfully certify its own executed bytes by rereading its pathname
after it has begun.

### 1.2 What exact final replay covers—and what it cannot cover

The stored run reports exact reproduction of the reference final model hash,
all 1,600 losses, and all 1,600 gradient norms. That would expose most changes
to initialization, data order, forward/loss behavior, or optimizer updates.
It does **not** establish which source produced the numbers, and it is silent
about code downstream of the last update.

| Code surface | Could alter the chosen/report result? | Necessarily alters the exact final model replay? |
|---|---:|---:|
| model, loss, data, training helpers | yes | usually, but not logically always |
| `scripts.eval.smooth_posteriors` | yes | no |
| raw/policy switch scorers | yes | no |
| `evaluate_snapshot` and metric assembly | yes | no |
| `shortlist_and_select` and usefulness gates | yes | no |
| result/identity construction in the driver | yes | no |

An evaluation-only A/B regression is therefore decisive: keep every training
callable identical, make A and B produce different smoothing or selector
sentinels, and show that the legacy start/end path hashes can agree on B while
the published decision comes from A. The final model should still reproduce
exactly; that is the point of the test.

### 1.3 The historical source bytes are recoverable, but execution is unproven

The experiment's schema-1 identity stores **9 paths / 177,969 bytes**. All nine
stored SHA-256 values match autosave commit
`54e376441615ea65d5535022cd0b2bb05cfa0529` exactly. This includes experiment
driver SHA-256 `119a4e39...2697cbc`. Therefore a resolved clean source candidate
exists for a repair rerun.

That finding does not prove the old interpreter executed those bytes. The
hashes were observed after import, and the commit was an autosave record rather
than the process's launch root. It establishes recoverability, not historical
execution provenance.

The ignored 29,759,358-byte trajectory artifact is still present with its
declared SHA-256 `f87f2aa4...68eac`. The old tracked summary can be recovered
from the commit, but the original ignored reference checkpoint with declared
SHA-256 `9ed06da1...f729a8` is no longer the live checkpoint as of this audit.
A bit-for-bit rerun of the complete old bundle is therefore **unverified**
unless that checkpoint is recovered from an external/autosave artifact. A new
current-schema rerun must be published as a new run rather than silently
rewriting the historical result.

There is no evidence that the historical experiment actually suffered an A/B
swap. The review reproduced its stored arithmetic and state hashes. This memo
shows that the current identity cannot rule the swap out.

### 1.4 The handwritten closure omits executed package code

`SOURCE_FILES` names the driver, `scripts/{train,eval}.py`, and six package
modules, but not `src/streaming_lid/__init__.py`. Python executes that regular
package initializer before its submodules. At the historical commit it is 150
bytes with SHA-256 `86291e22...4e100`, so the actual local Python set is at
least **10 files / 178,119 bytes**, not the declared nine.

Adding that one path is not a durable fix. A future helper import can recreate
the omission. Stage the complete regular-file Python trees under
`experiments/checkpoint-trajectory/`, `scripts/`, and `src/streaming_lid/`, then
use a post-import origin ledger as a coverage assertion. Extra staged Python
files are acceptable; an executed local file missing from the manifest is not.

### 1.5 `-I` alone still permits the editable live source

The current virtual environment contains
`_editable_impl_streaming_lid_distillation.pth`, whose sole path entry is the
live workspace `src`. A fresh local probe on 2026-09-26 found that both normal
Python and `.venv/bin/python -I` include
`/mnt/d/Work/Projects/asr-navana-codex/src`.

This follows the documented boundary: `-I` removes the current directory and
user site and ignores `PYTHON*` environment variables, while the `site` module
processes path-configuration files in the environment's site-packages.
`-S` disables those site-dependent path manipulations
([Python command-line options, accessed 2026-09-26](https://docs.python.org/3.10/using/cmdline.html),
[Python `site` documentation, accessed 2026-09-26](https://docs.python.org/3.10/library/site.html)).

The child must therefore either use a tested `-I -S` bootstrap that adds the
exact third-party site directory without processing `.pth` files, or remove
and reject every live-workspace path before importing any project module.
Full trajectory compatibility under `-I -S` remains **unverified**.

## 2. External precedents and their limits

### 2.1 A new process prevents reuse of the parent's module cache

Python documents `subprocess.Popen` as executing a child program in a new
process, using `execvpe`-like behavior on POSIX
([Python subprocess documentation, accessed 2026-09-26](https://docs.python.org/3.10/library/subprocess.html)).
That fresh process is the useful boundary: the lightweight parent must not
import the experiment or pipeline first.

Running the staged driver through `runpy` inside the already-contaminated
parent is not a repair. Python explicitly says `runpy` is not a sandbox and
that cached-import side effects remain in the current process
([Python `runpy`, accessed 2026-09-26](https://docs.python.org/3.10/library/runpy.html)).
A dedicated fresh child may use a tiny pre-import bootstrap internally, but it
must begin with no project modules loaded.

### 2.2 Git and the Hub both distinguish resolved trees from moving names

Git's `archive` command creates an archive from a named tree or commit, and its
manual distinguishes tree IDs from commit IDs
([Git 2.55 `git-archive`, last changed 2026-04-20; accessed 2026-09-26](https://git-scm.com/docs/git-archive)).
For this clean historical source candidate, archive the full resolved commit
`54e3764...` and independently verify the selected source manifest. Do not
resolve a branch name again inside the child.

The installed `huggingface-hub==0.24.7` guide similarly documents that
`snapshot_download()` downloads a repository at a revision and that a specific
commit requires the full-length hash
([Hugging Face Hub v0.24.7 download guide, accessed 2026-09-26](https://huggingface.co/docs/huggingface_hub/v0.24.7/en/guides/download)).
A live Hub/API check on 2026-09-26 confirmed the project's teacher tree still
resolves exactly to
[`speechbrain/lang-id-voxlingua107-ecapa@0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9):
the API returned that full SHA, 9 siblings, and `lastModified` 2024-11-27; the
[commit page](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/commit/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9)
dates the commit to **2024-11-27**.

This is a design precedent only. A Hub cache directory or Git checkout is not
automatically an adversarial immutability boundary. The experiment must verify
its own staged regular bytes before imports and before publication.

### 2.3 Provenance should identify resolved dependencies

SLSA v1.2 describes provenance as verifiable information about where, when,
and how an artifact was produced and places resolved dependency digests in
`resolvedDependencies`
([SLSA v1.2 build provenance, Approved; accessed 2026-09-26](https://slsa.dev/spec/v1.2/build-provenance)).
The take-home need not claim SLSA compliance, but the distinction applies:
`workspace_path`, branch, or end-of-run `HEAD` is descriptive; the executed
source-stage digest is the dependency.

## 3. Proposed experiment execution contract

### 3.1 Reuse the BUG-12 launcher; do not fork a second snapshot design

The generic launcher should accept an entry point plus declared source roots.
For this experiment it should:

1. before any NumPy, Torch, experiment, `scripts`, or `streaming_lid` import,
   capture a resolved Git tree or a labeled working-tree byte snapshot;
2. include the complete regular-file Python trees for the experiment,
   `scripts/`, and `src/streaming_lid/`, plus the standard-library child
   bootstrap;
3. record relative path, mode, byte count, and SHA-256, reject symlinks and
   non-regular files, and derive a canonical root digest;
4. verify and atomically retain the stage under a content-addressed ignored
   directory; and
5. start a fresh interpreter against the staged experiment driver.

The stage is code identity only. The manifest, audio, targets, reference
bundle, batch order, and model states retain their existing independent
content identities. BUG-13's single manifest snapshot should be composed with
this contract rather than replaced by it.

### 3.2 Make the child prove it did not escape

Before project imports, the child should verify the stage manifest, reject any
preloaded `scripts.*` or `streaming_lid*` module, establish a site-path policy,
and place staged roots ahead of third-party paths. After imports and again
before publication, require:

- every local module origin to resolve inside the retained stage;
- every origin to name a declared regular file with matching bytes;
- the staged driver path/digest to equal the declared child entry point;
- no origin or package `__path__` entry to resolve into the live workspace;
- no stage file added, deleted, changed, or replaced by a symlink; and
- the final result and trajectory payload to carry the same root digest.

An origin ledger is a closure check, not a substitute for pre-import staging.
On live paths it has the original A/B problem; inside a private verified stage
it becomes useful evidence.

### 3.3 Replace the ambiguous schema-1 flag

Keep schema-1 `results.json` unchanged as historical evidence. A schema-2 run
should replace the bare `source_files` map and
`source_snapshot_unchanged=true` claim with an object such as:

```text
executed_source {
  schema_version
  snapshot_kind: committed_git_tree | captured_worktree
  snapshot_root_sha256
  git_commit | null
  git_tree | null
  entrypoint_relative_path
  entrypoint_sha256
  manifest: {relative_path -> {sha256, bytes, mode}}
  local_module_origins
  site_path_policy
  stage_verified_before_import: true
  stage_verified_before_publication: true
  workspace_escape_checked: true
}
```

Include that object in `experiment_identity`, `results.json`, and
`trajectory.pt`; make the canonical experiment run ID depend on it. Preserve a
separate `live_workspace_changed_after_snapshot` field if useful, but never use
later worktree bytes as evidence about past execution.

The reference checkpoint's old launch snapshot still has BUG-12 semantics.
An experiment-stage repair does not retroactively strengthen that checkpoint.
Either consume a new BUG-12-fixed reference bundle, or call the old checkpoint
an expected numerical fixture while the newly staged full replay becomes the
source-verified trajectory.

## 4. Required adversarial and release tests

1. **Evaluation-only import A / path B.** Keep training code byte-identical,
   import A where `smooth_posteriors` emits a recognizable sentinel, replace
   the live path with B before the old snapshot point, and require the legacy
   harness to show B hashes plus A evaluation behavior. The final model/loss/
   gradient replay must remain exact, proving that replay is not the missing
   gate. The staged child must execute and certify A consistently.
2. **Driver-self A / path B.** Change only `shortlist_and_select` or result
   assembly after driver A starts. The old helper must be shown capable of
   hashing B while using A; the new child must publish the staged entry-point
   digest and A result.
3. **Closure and editable escape.** Mutate only
   `streaming_lid/__init__.py`; add a lazy local helper; leave the editable
   workspace `.pth` active. The stage root must change, every used helper must
   be declared, and no origin may escape to the workspace.
4. **Stage integrity.** Change, delete, add, or symlink one staged Python file
   after the first verification. Each case must abort before evaluation or
   artifact publication.
5. **Dual-artifact binding.** Forge `results.json` or `trajectory.pt` to carry
   another stage digest. Validation must reject it even if all model-state
   hashes match.
6. **Current-schema replay.** After BUG-12/13 and current builder changes
   settle, run the complete 1,600-step experiment once from a resolved stage,
   offline for Hub/model access, and publish it as a new schema-2 run. Compare
   timing-excluded losses, state hashes, curve, shortlist, and decision with
   the historical run. Numerical parity and the selected step are
   **unverified**, not assumed.

## 5. Rejected shortcuts

| Shortcut | Why it does not close BUG-15 |
|---|---|
| Move `snapshot_sources()` to the first line of `main()` | Module top-level driver code and all imports have already executed. |
| Hash `__file__`, `__spec__.origin`, or loader source after import | These can reread B while the live object remains A. |
| Trust exact final model/loss replay | Evaluation and selector code can change without touching training. |
| Add only `streaming_lid/__init__.py` | Repairs one omission, not the post-import race or future closure drift. |
| Delete modules from `sys.modules` and reimport | Existing direct bindings, instances, and driver globals can retain A. |
| Run the driver with in-process `runpy` from the live parent | `runpy` shares the process and import side effects; it is not isolation. |
| Use `-I` alone | This environment still receives the live editable `src` through site `.pth` processing. |
| Record current `HEAD` at publication | A later moving-worktree observation is not the executed dependency. |
| Rewrite the old result in place | It destroys the distinction between historical evidence and a repaired rerun. |

## 6. Expected effect and limits

- Expected accuracy, loss, checkpoint choice, parameter count, and inference
  latency change: **none guaranteed**.
- Expected integrity effect: a driver/import-A plus pathname-B result cannot
  pass the stated stage and origin gates under accidental concurrent edits.
- The old run remains numerically well audited; historical corruption is
  **unverified** and no corruption is alleged.
- Full `-I -S` compatibility, staging time, new-run parity, and the schema-2
  selected checkpoint remain **unverified**.
- A read-only directory is an accidental-mutation guard, not a security
  boundary against a privileged writer. Strong adversarial provenance needs a
  trusted builder/container or immutable mount.
- This contract covers local Python source. Dependency wheels/native
  libraries, CPU/kernel behavior, data identity, and bitwise cross-host
  determinism require separate identities and claims.

## Source record

Primary/official sources checked **2026-09-26**:

- [Python 3.10.21 import system](https://docs.python.org/3.10/reference/import.html)
- [Python 3.10.21 subprocess management](https://docs.python.org/3.10/library/subprocess.html)
- [Python 3.10.21 `runpy`](https://docs.python.org/3.10/library/runpy.html)
- [Python 3.10.21 command-line isolation](https://docs.python.org/3.10/using/cmdline.html)
- [Python 3.10.21 `site` and `.pth` processing](https://docs.python.org/3.10/library/site.html)
- [Git `archive` manual; last changed 2026-04-20](https://git-scm.com/docs/git-archive)
- [SLSA v1.2 build provenance; Approved](https://slsa.dev/spec/v1.2/build-provenance)
- [Hugging Face Hub v0.24.7 download guide](https://huggingface.co/docs/huggingface_hub/v0.24.7/en/guides/download)
- [Pinned SpeechBrain ECAPA tree](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9)
- [Pinned SpeechBrain ECAPA commit, dated 2024-11-27](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/commit/0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9)

Local evidence checked **2026-09-26**:

- `experiments/checkpoint-trajectory/run.py`, `results.json`, `REPORT.md`, and
  review `changes-2026-09-26-f03269e.md`
- exact stored/current/Git-object SHA-256 and byte-count comparisons
- current import lists, source allowlists, dependency snapshot, and publication
  order
- normal versus `-I` virtual-environment `sys.path` plus editable `.pth`
- current ignored trajectory and reference-bundle hashes
- installed `huggingface-hub==0.24.7`, Torch `2.6.0+cpu`, and Matplotlib
  `3.10.9`

