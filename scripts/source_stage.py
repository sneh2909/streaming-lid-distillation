#!/usr/bin/env python3
"""Create and verify the immutable source tree used by train/eval children.

This module intentionally imports only the Python standard library.  The live
entry points import it before NumPy, Torch, or any ``streaming_lid`` module,
publish a content-addressed copy of the local Python sources, and then start a
fresh interpreter from that copy.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_MANIFEST = ".source_snapshot.json"
SNAPSHOT_DIRECTORY = Path("data/generated/source_snapshots")
STAGE_ROOT_ENV = "STREAMING_LID_SOURCE_STAGE_ROOT"
STAGE_DIGEST_ENV = "STREAMING_LID_SOURCE_STAGE_SHA256"
WORKSPACE_ROOT_ENV = "STREAMING_LID_LIVE_WORKSPACE_ROOT"
CHILD_ENTRYPOINT_ENV = "STREAMING_LID_SOURCE_STAGE_ENTRYPOINT"
CHILD_PROCESS_ENV = "STREAMING_LID_SOURCE_STAGE_CHILD"
PREIMPORT_VERIFIED_ENV = "STREAMING_LID_SOURCE_STAGE_PREIMPORT_VERIFIED"

PROJECT_CONFIG_FILES = ("pyproject.toml", "uv.lock")
RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "torch",
    "torchaudio",
    "soundfile",
    "speechbrain",
    "huggingface-hub",
)

_CHILD_CONTEXT: dict[str, Any] | None = None
_STAGE_VERIFICATION_COUNT = 0


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _assert_regular_unsymlinked_file(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source path escapes workspace: {path}") from error
    relative = path.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"source snapshot refuses symlink: {cursor}")
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"source snapshot requires a regular file: {path}")


def source_paths(workspace_root: str | Path) -> tuple[Path, ...]:
    """Return the complete local source/config closure captured by the launcher."""
    root = Path(workspace_root).resolve()
    scripts_root = root / "scripts"
    package_root = root / "src" / "streaming_lid"
    if not scripts_root.is_dir() or not package_root.is_dir():
        raise FileNotFoundError("workspace is missing scripts/ or src/streaming_lid/")

    paths = [*scripts_root.glob("*.py"), *package_root.rglob("*.py")]
    paths.extend(root / name for name in PROJECT_CONFIG_FILES)
    unique = sorted({path.resolve() for path in paths}, key=lambda item: item.as_posix())
    if not unique:
        raise ValueError("source snapshot would be empty")
    for path in unique:
        _assert_regular_unsymlinked_file(path, root)
    required = {
        "scripts/train.py",
        "scripts/eval.py",
        "scripts/source_stage.py",
        "src/streaming_lid/__init__.py",
        *PROJECT_CONFIG_FILES,
    }
    relative_paths = {path.relative_to(root).as_posix() for path in unique}
    missing = sorted(required - relative_paths)
    if missing:
        raise FileNotFoundError(f"source snapshot is missing required files: {missing}")
    return tuple(unique)


def _captured_files(workspace_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    files: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for path in source_paths(workspace_root):
        relative = path.relative_to(workspace_root).as_posix()
        payload = path.read_bytes()
        payloads[relative] = payload
        files[relative] = {"sha256": _sha256(payload), "bytes": len(payload)}
    return files, payloads


def _snapshot_root_sha256(files: Mapping[str, Mapping[str, Any]]) -> str:
    content = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "files": {
            name: {
                "sha256": properties["sha256"],
                "bytes": properties["bytes"],
            }
            for name, properties in sorted(files.items())
        },
    }
    return _sha256(_canonical_json_bytes(content))


def _write_snapshot_tree(
    temporary_root: Path,
    *,
    root_sha256: str,
    files: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, bytes],
) -> None:
    staged_files: dict[str, dict[str, Any]] = {}
    for relative, properties in sorted(files.items()):
        destination = temporary_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payloads[relative])
        # This is an accidental-mutation guard, not a security boundary.  Some
        # mounted filesystems preserve an execute bit while clearing writes.
        destination.chmod(0o444)
        staged_files[relative] = {
            **dict(properties),
            "mode": stat.S_IMODE(destination.stat().st_mode),
        }

    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_root_sha256": root_sha256,
        "snapshot_kind": "captured_worktree",
        "files": staged_files,
        "read_only_requested": True,
    }
    manifest_path = temporary_root / SNAPSHOT_MANIFEST
    manifest_path.write_bytes(_canonical_json_bytes(manifest) + b"\n")
    manifest_path.chmod(0o444)


def capture_source_snapshot(
    workspace_root: str | Path,
    *,
    snapshots_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Capture local source bytes once and atomically publish their stage."""
    root = Path(workspace_root).resolve()
    files, payloads = _captured_files(root)
    root_sha256 = _snapshot_root_sha256(files)
    parent = (
        Path(snapshots_directory).resolve()
        if snapshots_directory is not None
        else root / SNAPSHOT_DIRECTORY
    )
    parent.mkdir(parents=True, exist_ok=True)
    stage_root = parent / root_sha256
    if stage_root.exists():
        manifest = verify_source_snapshot(stage_root, expected_root_sha256=root_sha256)
        return {"stage_root": stage_root, "manifest": manifest}

    temporary_root = Path(tempfile.mkdtemp(prefix=".source-stage-", dir=parent))
    try:
        _write_snapshot_tree(
            temporary_root,
            root_sha256=root_sha256,
            files=files,
            payloads=payloads,
        )
        verify_source_snapshot(temporary_root, expected_root_sha256=root_sha256)
        try:
            temporary_root.rename(stage_root)
        except FileExistsError:
            # Another launcher published the same content first.
            verify_source_snapshot(stage_root, expected_root_sha256=root_sha256)
        manifest = verify_source_snapshot(stage_root, expected_root_sha256=root_sha256)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
    return {"stage_root": stage_root, "manifest": manifest}


def verify_source_snapshot(
    stage_root: str | Path, *, expected_root_sha256: str | None = None
) -> dict[str, Any]:
    """Fail closed on changed, missing, extra, or symlinked staged files."""
    global _STAGE_VERIFICATION_COUNT
    root = Path(stage_root).resolve()
    manifest_path = root / SNAPSHOT_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(f"source stage manifest is missing or unsafe: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"source stage manifest is invalid: {error}") from error
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise RuntimeError("source stage schema version is missing or unsupported")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("source stage manifest has no file ledger")
    declared_root = manifest.get("snapshot_root_sha256")
    recomputed_root = _snapshot_root_sha256(files)
    if declared_root != recomputed_root:
        raise RuntimeError("source stage root digest contradicts its manifest")
    if expected_root_sha256 is not None and declared_root != expected_root_sha256:
        raise RuntimeError("source stage root digest differs from the launcher request")

    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"source stage contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative != SNAPSHOT_MANIFEST:
                actual_files.add(relative)
    declared_files = set(files)
    if actual_files != declared_files:
        missing = sorted(declared_files - actual_files)
        extra = sorted(actual_files - declared_files)
        raise RuntimeError(
            f"source stage file set differs from manifest; missing={missing}, extra={extra}"
        )
    for relative, properties in sorted(files.items()):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"source stage file is missing or unsafe: {relative}")
        payload = path.read_bytes()
        if len(payload) != properties.get("bytes"):
            raise RuntimeError(f"source stage byte count changed: {relative}")
        if _sha256(payload) != properties.get("sha256"):
            raise RuntimeError(f"source stage content changed: {relative}")
        if stat.S_IMODE(path.stat().st_mode) != properties.get("mode"):
            raise RuntimeError(f"source stage mode changed: {relative}")
    _STAGE_VERIFICATION_COUNT += 1
    return manifest


def is_staged_child() -> bool:
    return os.environ.get(CHILD_PROCESS_ENV) == "1"


def run_snapshot_child(
    snapshot: Mapping[str, Any],
    *,
    entrypoint: str,
    argv: Sequence[str],
    workspace_root: str | Path,
    extra_environment: Mapping[str, str] | None = None,
) -> int:
    """Run one staged entry point in a fresh interpreter."""
    stage_root = Path(snapshot["stage_root"]).resolve()
    manifest = verify_source_snapshot(stage_root)
    if entrypoint not in manifest["files"]:
        raise ValueError(f"entry point is absent from source stage: {entrypoint}")
    workspace = Path(workspace_root).resolve()
    environment = os.environ.copy()
    if extra_environment:
        environment.update({str(key): str(value) for key, value in extra_environment.items()})
    environment.update(
        {
            STAGE_ROOT_ENV: str(stage_root),
            STAGE_DIGEST_ENV: manifest["snapshot_root_sha256"],
            WORKSPACE_ROOT_ENV: str(workspace),
            CHILD_ENTRYPOINT_ENV: entrypoint,
            CHILD_PROCESS_ENV: "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(stage_root / entrypoint), *argv],
        cwd=workspace,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def launch_in_source_snapshot(entrypoint: str, argv: Sequence[str]) -> int:
    """Capture the live workspace before project imports, then launch a child."""
    workspace = Path(__file__).resolve().parents[1]
    snapshot = capture_source_snapshot(workspace)
    return run_snapshot_child(
        snapshot,
        entrypoint=entrypoint,
        argv=argv,
        workspace_root=workspace,
    )


def _resolved_sys_path(raw_path: str) -> Path:
    return Path(raw_path or os.getcwd()).resolve()


def _is_live_project_import_path(path: Path, workspace: Path, stage_root: Path) -> bool:
    if _is_relative_to(path, stage_root):
        return False
    live_src = workspace / "src"
    live_scripts = workspace / "scripts"
    return path == workspace or _is_relative_to(path, live_src) or _is_relative_to(
        path, live_scripts
    )


def _runtime_environment_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    distributions: dict[str, str] = {}
    for name in RUNTIME_DISTRIBUTIONS:
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required runtime distribution is missing: {name}") from error
    project_files = {
        name: {
            "sha256": manifest["files"][name]["sha256"],
            "bytes": manifest["files"][name]["bytes"],
        }
        for name in PROJECT_CONFIG_FILES
    }
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "distributions": distributions,
        "project_files": project_files,
    }


def prepare_staged_child(entrypoint: str) -> dict[str, Any]:
    """Verify/isolate a staged child before any project or numeric import."""
    global _CHILD_CONTEXT
    if not is_staged_child():
        raise RuntimeError("source-stage child marker is missing")
    stage_root = Path(os.environ[STAGE_ROOT_ENV]).resolve()
    workspace = Path(os.environ[WORKSPACE_ROOT_ENV]).resolve()
    expected_digest = os.environ[STAGE_DIGEST_ENV]
    if os.environ.get(CHILD_ENTRYPOINT_ENV) != entrypoint:
        raise RuntimeError("source-stage entry point differs from child request")
    manifest = verify_source_snapshot(
        stage_root, expected_root_sha256=expected_digest
    )
    expected_script = (stage_root / entrypoint).resolve()
    if Path(sys.argv[0]).resolve() != expected_script:
        raise RuntimeError("child interpreter did not start from the staged entry point")
    if Path(__file__).resolve() != (stage_root / "scripts/source_stage.py").resolve():
        raise RuntimeError("child imported source_stage outside the verified stage")
    preloaded = sorted(
        name
        for name in sys.modules
        if name == "streaming_lid" or name.startswith("streaming_lid.")
    )
    if preloaded:
        raise RuntimeError(f"local project modules were imported before isolation: {preloaded}")

    retained_paths: list[str] = []
    for raw_path in sys.path:
        resolved = _resolved_sys_path(raw_path)
        if _is_live_project_import_path(resolved, workspace, stage_root):
            continue
        if str(resolved) not in retained_paths:
            retained_paths.append(str(resolved))
    stage_src = str((stage_root / "src").resolve())
    stage_scripts = str((stage_root / "scripts").resolve())
    retained_paths = [item for item in retained_paths if item not in {stage_src, stage_scripts}]
    sys.path[:] = [stage_src, stage_scripts, *retained_paths]
    importlib.invalidate_caches()
    if any(
        _is_live_project_import_path(_resolved_sys_path(item), workspace, stage_root)
        for item in sys.path
    ):
        raise RuntimeError("live workspace import path survived source-stage isolation")

    context = {
        "snapshot_root_sha256": manifest["snapshot_root_sha256"],
        "snapshot_kind": manifest["snapshot_kind"],
        "manifest": manifest,
        "stage_root": stage_root,
        "workspace_root": workspace,
        "entrypoint": entrypoint,
        "environment": _runtime_environment_identity(manifest),
        "stage_verified_before_import": True,
        "workspace_escape_checked": True,
        "site_path_policy": {
            "name": "stage_src_first_remove_live_project_paths_v1",
            "stage_src_first": True,
            "live_workspace_project_paths_removed": True,
            "preloaded_local_modules_rejected": True,
            "bytecode_writes_disabled": os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        },
    }
    _CHILD_CONTEXT = context
    os.environ[PREIMPORT_VERIFIED_ENV] = manifest["snapshot_root_sha256"]
    return context


def child_context() -> dict[str, Any]:
    if _CHILD_CONTEXT is None:
        raise RuntimeError("source-stage child was not prepared before project imports")
    return _CHILD_CONTEXT


def _package_module_name(relative_path: str) -> str:
    prefix = "src/streaming_lid/"
    if not relative_path.startswith(prefix) or not relative_path.endswith(".py"):
        raise ValueError(f"not a package Python file: {relative_path}")
    suffix = relative_path[len("src/") : -len(".py")]
    parts = suffix.split("/")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def executed_source_identity() -> dict[str, Any]:
    """Return the verified, entrypoint-neutral source identity for run binding."""
    context = child_context()
    manifest = verify_source_snapshot(
        context["stage_root"],
        expected_root_sha256=context["snapshot_root_sha256"],
    )
    package_files = sorted(
        path
        for path in manifest["files"]
        if path.startswith("src/streaming_lid/") and path.endswith(".py")
    )
    expected_modules = {_package_module_name(path): path for path in package_files}
    for module_name in sorted(expected_modules, key=lambda name: (name.count("."), name)):
        importlib.import_module(module_name)

    origins: dict[str, str] = {}
    stage_root = context["stage_root"]
    for module_name, expected_relative in sorted(expected_modules.items()):
        module = sys.modules.get(module_name)
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if not isinstance(origin, str):
            raise RuntimeError(f"local module has no file origin: {module_name}")
        resolved = Path(origin).resolve()
        if not _is_relative_to(resolved, stage_root):
            raise RuntimeError(f"local module origin escaped source stage: {module_name}")
        relative = resolved.relative_to(stage_root).as_posix()
        if relative != expected_relative or relative not in manifest["files"]:
            raise RuntimeError(f"local module origin contradicts manifest: {module_name}")
        origins[module_name] = relative

    unexpected_modules = []
    for module_name, module in sorted(sys.modules.items()):
        if module_name != "streaming_lid" and not module_name.startswith("streaming_lid."):
            continue
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if isinstance(origin, str) and module_name not in expected_modules:
            unexpected_modules.append(module_name)
    if unexpected_modules:
        raise RuntimeError(f"undeclared local modules were imported: {unexpected_modules}")

    package = sys.modules["streaming_lid"]
    for raw_path in getattr(package, "__path__", []):
        if not _is_relative_to(Path(raw_path).resolve(), stage_root):
            raise RuntimeError("streaming_lid package search path escaped source stage")

    entrypoints = {
        role: {
            "path": path,
            "sha256": manifest["files"][path]["sha256"],
        }
        for role, path in (
            ("train", "scripts/train.py"),
            ("eval", "scripts/eval.py"),
        )
    }
    files = {
        name: {
            "sha256": properties["sha256"],
            "bytes": properties["bytes"],
            "mode": properties["mode"],
        }
        for name, properties in sorted(manifest["files"].items())
    }
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": "executed_source_snapshot",
        "source_sha256": manifest["snapshot_root_sha256"],
        "snapshot_root_sha256": manifest["snapshot_root_sha256"],
        "snapshot_kind": manifest["snapshot_kind"],
        "files": files,
        "entrypoints": entrypoints,
        "local_module_origins": origins,
        "environment": context["environment"],
        "site_path_policy": context["site_path_policy"],
        "stage_verified_before_import": context["stage_verified_before_import"],
        "workspace_escape_checked": context["workspace_escape_checked"],
    }


def stage_verification_count() -> int:
    return _STAGE_VERIFICATION_COUNT
