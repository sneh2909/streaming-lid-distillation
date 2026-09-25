import json
from pathlib import Path

import numpy as np
import pytest

from streaming_lid.config import LANGUAGE_CODES, TEACHER_NAME
from streaming_lid.data import (
    TARGET_CACHE_SCHEMA_VERSION,
    TeacherTargetCache,
    canonical_json_sha256,
    file_sha256,
    load_teacher_target_array,
    manifest_record_sha256,
    manifest_records_sha256,
    target_cache_configuration,
    target_configuration_sha256,
)


def _probabilities(frames: int) -> np.ndarray:
    value = np.full((frames, len(LANGUAGE_CODES)), 0.05, dtype=np.float32)
    value[:, 0] = 0.70
    return value


def _write_target_file(
    path: Path,
    item: dict,
    audio_hash: str,
    *,
    language_codes: tuple[str, ...] = LANGUAGE_CODES,
) -> None:
    probabilities = _probabilities(3)
    np.savez_compressed(
        path,
        teacher_probs=probabilities,
        teacher_soft_targets=probabilities,
        anchor_frames=np.asarray([1], dtype=np.int64),
        anchor_probs=probabilities[:1],
        in_set_mass=np.asarray([0.75], dtype=np.float32),
        language_codes=np.asarray(language_codes),
        cache_schema_version=np.asarray(TARGET_CACHE_SCHEMA_VERSION),
        clip_id=np.asarray(item["id"]),
        target_kind=np.asarray("converged_utterance"),
        num_frames=np.asarray(3),
        audio_sha256=np.asarray(audio_hash),
        manifest_record_sha256=np.asarray(manifest_record_sha256(item)),
        target_configuration_sha256=np.asarray(target_configuration_sha256()),
        teacher_name=np.asarray(TEACHER_NAME),
    )


def _write_valid_cache(tmp_path: Path) -> tuple[Path, Path, dict]:
    item = {
        "id": "clip",
        "audio_path": "clip.wav",
        "split": "train",
        "language": "en",
        "speaker_id": "speaker",
        "text": "hello",
    }
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"audio-version-one")
    targets_dir = tmp_path / "targets"
    targets_dir.mkdir()
    target_path = targets_dir / "clip.npz"
    audio_hash = file_sha256(audio_path)
    _write_target_file(target_path, item, audio_hash)
    metadata = {
        "schema_version": TARGET_CACHE_SCHEMA_VERSION,
        "target_configuration": target_cache_configuration(),
        "target_configuration_sha256": target_configuration_sha256(),
        "manifest_records_sha256": manifest_records_sha256([item]),
        "target_files_sha256": canonical_json_sha256(
            {item["id"]: file_sha256(target_path)}
        ),
        "clips": [
            {
                "id": item["id"],
                "frames": 3,
                "target_kind": "converged_utterance",
                "audio_sha256": audio_hash,
                "manifest_record_sha256": manifest_record_sha256(item),
                "target_file_sha256": file_sha256(target_path),
            }
        ],
    }
    (targets_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return manifest_path, targets_dir, item


def test_basic_loader_rejects_reordered_target_columns(tmp_path: Path) -> None:
    target_path = tmp_path / "reordered.npz"
    item = {"id": "clip"}
    _write_target_file(
        target_path,
        item,
        "unused",
        language_codes=("hi", "en", *LANGUAGE_CODES[2:]),
    )

    with pytest.raises(ValueError, match="language order"):
        load_teacher_target_array(target_path, "teacher_soft_targets")


def test_strict_cache_accepts_content_bound_target(tmp_path: Path) -> None:
    manifest_path, targets_dir, item = _write_valid_cache(tmp_path)
    cache = TeacherTargetCache(manifest_path, targets_dir)

    value = cache.load(item, "teacher_soft_targets", expected_frames=3)

    assert value.shape == (3, len(LANGUAGE_CODES))
    assert cache.audit()["validated_clips"] == 1


def test_strict_cache_rejects_changed_audio(tmp_path: Path) -> None:
    manifest_path, targets_dir, item = _write_valid_cache(tmp_path)
    (tmp_path / "clip.wav").write_bytes(b"audio-version-two")
    cache = TeacherTargetCache(manifest_path, targets_dir)

    with pytest.raises(ValueError, match="audio SHA-256 mismatch"):
        cache.load(item, "teacher_soft_targets", expected_frames=3)


def test_strict_cache_rejects_changed_manifest(tmp_path: Path) -> None:
    manifest_path, targets_dir, item = _write_valid_cache(tmp_path)
    item["text"] = "changed declaration"
    manifest_path.write_text(json.dumps(item) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="different manifest"):
        TeacherTargetCache(manifest_path, targets_dir)


def test_strict_cache_rejects_modified_target_file(tmp_path: Path) -> None:
    manifest_path, targets_dir, item = _write_valid_cache(tmp_path)
    _write_target_file(targets_dir / "clip.npz", item, file_sha256(tmp_path / "clip.wav"))
    with (targets_dir / "clip.npz").open("ab") as handle:
        handle.write(b"tamper")
    cache = TeacherTargetCache(manifest_path, targets_dir)

    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        cache.load(item, "teacher_soft_targets", expected_frames=3)
