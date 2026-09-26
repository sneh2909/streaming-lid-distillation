import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

from scripts.source_stage import (
    SNAPSHOT_MANIFEST,
    capture_source_snapshot,
    run_snapshot_child,
    verify_source_snapshot,
)


FIXTURE_TRAIN = """\
import json
import os
import sys
from pathlib import Path

from source_stage import executed_source_identity, prepare_staged_child

prepare_staged_child("scripts/train.py")
from streaming_lid.sentinel import VALUE

try:
    import streaming_lid.live_only  # noqa: F401
except ModuleNotFoundError:
    live_only_imported = False
else:
    live_only_imported = True

identity = executed_source_identity()
Path(os.environ["SOURCE_STAGE_FIXTURE_OUTPUT"]).write_text(
    json.dumps(
        {
            "value": VALUE,
            "identity": identity,
            "live_only_imported": live_only_imported,
            "sys_path": sys.path,
        }
    ),
    encoding="utf-8",
)
"""


def _fixture_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    scripts = workspace / "scripts"
    package = workspace / "src" / "streaming_lid"
    scripts.mkdir(parents=True)
    package.mkdir(parents=True)
    repository_root = Path(__file__).resolve().parents[1]
    shutil.copyfile(repository_root / "scripts/source_stage.py", scripts / "source_stage.py")
    (scripts / "train.py").write_text(FIXTURE_TRAIN, encoding="utf-8")
    (scripts / "eval.py").write_text("# staged evaluation fixture\n", encoding="utf-8")
    (package / "__init__.py").write_text("MARKER = 'initializer-a'\n", encoding="utf-8")
    (package / "sentinel.py").write_text("VALUE = 'source-a'\n", encoding="utf-8")
    (workspace / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (workspace / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return workspace


def test_staged_child_executes_captured_a_after_live_path_becomes_b(
    tmp_path: Path,
) -> None:
    workspace = _fixture_workspace(tmp_path)
    sentinel = workspace / "src/streaming_lid/sentinel.py"

    spec = importlib.util.spec_from_file_location("legacy_source_probe", sentinel)
    assert spec is not None and spec.loader is not None
    legacy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_module)
    snapshot = capture_source_snapshot(workspace)

    sentinel.write_text("VALUE = 'source-b'\n", encoding="utf-8")
    (workspace / "src/streaming_lid/live_only.py").write_text(
        "VALUE = 'live-only'\n", encoding="utf-8"
    )
    assert legacy_module.VALUE == "source-a"
    assert "source-b" in sentinel.read_text(encoding="utf-8")

    output = tmp_path / "child.json"
    return_code = run_snapshot_child(
        snapshot,
        entrypoint="scripts/train.py",
        argv=(),
        workspace_root=workspace,
        extra_environment={
            "SOURCE_STAGE_FIXTURE_OUTPUT": str(output),
            "PYTHONPATH": str(workspace / "src"),
        },
    )

    assert return_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["value"] == "source-a"
    assert not result["live_only_imported"]
    identity = result["identity"]
    assert identity["kind"] == "executed_source_snapshot"
    assert identity["snapshot_root_sha256"] == snapshot["manifest"][
        "snapshot_root_sha256"
    ]
    assert identity["local_module_origins"] == {
        "streaming_lid": "src/streaming_lid/__init__.py",
        "streaming_lid.sentinel": "src/streaming_lid/sentinel.py",
    }
    assert str((workspace / "src").resolve()) not in result["sys_path"]


def test_initializer_is_in_source_root_and_changing_it_changes_identity(
    tmp_path: Path,
) -> None:
    workspace = _fixture_workspace(tmp_path)
    first = capture_source_snapshot(workspace)
    initializer = workspace / "src/streaming_lid/__init__.py"
    initializer.write_text("MARKER = 'initializer-b'\n", encoding="utf-8")
    second = capture_source_snapshot(workspace)

    assert "src/streaming_lid/__init__.py" in first["manifest"]["files"]
    assert first["manifest"]["snapshot_root_sha256"] != second["manifest"][
        "snapshot_root_sha256"
    ]


@pytest.mark.parametrize("tamper", ["content", "delete", "extra", "symlink", "mode"])
def test_source_stage_rejects_tamper(tmp_path: Path, tamper: str) -> None:
    workspace = _fixture_workspace(tmp_path)
    snapshot = capture_source_snapshot(workspace)
    stage_root = Path(snapshot["stage_root"])
    sentinel = stage_root / "src/streaming_lid/sentinel.py"
    if tamper == "content":
        sentinel.chmod(0o644)
        sentinel.write_text("VALUE = 'tampered'\n", encoding="utf-8")
    elif tamper == "delete":
        sentinel.unlink()
    elif tamper == "extra":
        (stage_root / "src/streaming_lid/extra.py").write_text(
            "VALUE = 'extra'\n", encoding="utf-8"
        )
    elif tamper == "symlink":
        sentinel.unlink()
        sentinel.symlink_to(stage_root / "src/streaming_lid/__init__.py")
    elif tamper == "mode":
        sentinel.chmod(0o700)
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(tamper)

    with pytest.raises(RuntimeError, match="source stage"):
        verify_source_snapshot(
            stage_root,
            expected_root_sha256=snapshot["manifest"]["snapshot_root_sha256"],
        )


def test_source_stage_rejects_forged_manifest_root(tmp_path: Path) -> None:
    workspace = _fixture_workspace(tmp_path)
    snapshot = capture_source_snapshot(workspace)
    manifest_path = Path(snapshot["stage_root"]) / SNAPSHOT_MANIFEST
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop("src/streaming_lid/__init__.py")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="root digest contradicts"):
        verify_source_snapshot(Path(snapshot["stage_root"]))
