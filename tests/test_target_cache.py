import json
from pathlib import Path

import numpy as np
import pytest
import torch

from streaming_lid.config import (
    LANGUAGE_CODES,
    TEACHER_ARTIFACT_SHA256,
    TEACHER_LABELS,
    TEACHER_LANGUAGE_INDICES,
    TEACHER_NAME,
    TEACHER_REVISION,
)
from streaming_lid.data import (
    DENSE_TARGET_VALIDATION_ATOL,
    DENSE_TARGET_VALIDATION_RTOL,
    TARGET_CACHE_SCHEMA_VERSION,
    DistillationDataset,
    TeacherTargetCache,
    build_training_target_audit,
    canonical_json_sha256,
    capture_manifest_snapshot,
    expand_local_posteriors,
    file_sha256,
    load_teacher_target_array,
    local_target_availability_ledger,
    manifest_record_sha256,
    manifest_records_sha256,
    require_speaker_disjoint,
    target_cache_configuration,
    target_configuration_sha256,
    target_generator_identity,
)
from streaming_lid.run_identity import corpus_identity


def _probabilities(frames: int) -> np.ndarray:
    value = np.full((frames, len(LANGUAGE_CODES)), 0.05, dtype=np.float32)
    value[:, 0] = 0.70
    return value


def _soften(probabilities: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    values = np.power(probabilities.astype(np.float64), 1.0 / temperature)
    return np.asarray(values / values.sum(axis=-1, keepdims=True), dtype=np.float32)


def _write_target_file(
    path: Path,
    item: dict,
    audio_hash: str,
    *,
    language_codes: tuple[str, ...] = LANGUAGE_CODES,
    teacher_revision: str = TEACHER_REVISION,
    manifest_file_sha256: str = "unused-manifest",
    num_frames: int = 3,
    anchor_probs: np.ndarray | None = None,
    in_set_mass: float = 0.75,
    full_top1_index: int | None = None,
    full_top1_probability: float | None = None,
) -> None:
    anchor_probs = _probabilities(1) if anchor_probs is None else anchor_probs
    anchor_soft_targets = _soften(anchor_probs)
    probabilities = np.repeat(anchor_probs, num_frames, axis=0)
    soft_targets = np.repeat(anchor_soft_targets, num_frames, axis=0)
    selected_column = int(np.argmax(anchor_probs[0]))
    full_index = (
        TEACHER_LANGUAGE_INDICES[selected_column]
        if full_top1_index is None
        else full_top1_index
    )
    full_probability = (
        float(anchor_probs[0, selected_column] * in_set_mass)
        if full_top1_probability is None
        else full_top1_probability
    )
    np.savez_compressed(
        path,
        teacher_probs=probabilities,
        teacher_soft_targets=soft_targets,
        anchor_frames=np.asarray([num_frames // 2], dtype=np.int64),
        anchor_probs=anchor_probs,
        anchor_soft_targets=anchor_soft_targets,
        in_set_mass=np.asarray([in_set_mass], dtype=np.float32),
        full_top1_indices=np.asarray([full_index], dtype=np.int64),
        full_top1_probabilities=np.asarray([full_probability], dtype=np.float32),
        full_top1_labels=np.asarray([TEACHER_LABELS[full_index]]),
        language_codes=np.asarray(language_codes),
        cache_schema_version=np.asarray(TARGET_CACHE_SCHEMA_VERSION),
        clip_id=np.asarray(item["id"]),
        target_kind=np.asarray("converged_utterance"),
        num_frames=np.asarray(num_frames),
        audio_sha256=np.asarray(audio_hash),
        manifest_record_sha256=np.asarray(manifest_record_sha256(item)),
        manifest_file_sha256=np.asarray(manifest_file_sha256),
        target_configuration_sha256=np.asarray(target_configuration_sha256()),
        teacher_name=np.asarray(TEACHER_NAME),
        teacher_revision=np.asarray(teacher_revision),
        teacher_artifact_sha256=np.asarray(TEACHER_ARTIFACT_SHA256),
        target_generator_source_sha256=np.asarray(
            target_generator_identity()["source_sha256"]
        ),
    )


def _write_valid_cache(tmp_path: Path) -> tuple[Path, Path, dict]:
    item = {
        "id": "clip",
        "audio_path": "clip.wav",
        "split": "heldout",
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
    manifest_snapshot = capture_manifest_snapshot(manifest_path)
    _write_target_file(
        target_path,
        item,
        audio_hash,
        manifest_file_sha256=manifest_snapshot.manifest_file_sha256,
    )
    configuration = target_cache_configuration()
    training_target_audit = build_training_target_audit([item], targets_dir)
    metadata = {
        "schema_version": TARGET_CACHE_SCHEMA_VERSION,
        "teacher_identity": configuration["teacher"],
        "target_generator": configuration["target_generator"],
        "target_configuration": configuration,
        "target_configuration_sha256": canonical_json_sha256(configuration),
        "manifest_snapshot": manifest_snapshot.identity(),
        "manifest_file_sha256": manifest_snapshot.manifest_file_sha256,
        "manifest_records_sha256": manifest_records_sha256([item]),
        "target_expansion": "previous_anchor_hold",
        "availability_sample_index_semantics": "exclusive_right_edge_unclipped",
        "availability_checked_frames": 0,
        "availability_contract_valid": True,
        "dense_target_checked_frames": 3,
        "dense_target_expansion_valid": True,
        "dense_target_validation_rtol": DENSE_TARGET_VALIDATION_RTOL,
        "dense_target_validation_atol": DENSE_TARGET_VALIDATION_ATOL,
        "training_target_audit": training_target_audit,
        "training_target_audit_sha256": training_target_audit["audit_sha256"],
        "target_files_sha256": canonical_json_sha256(
            {item["id"]: file_sha256(target_path)}
        ),
        "clips": [
            {
                "id": item["id"],
                "frames": 3,
                "target_kind": "converged_utterance",
                "target_expansion": "constant_utterance_repeat",
                "availability_checked_frames": 0,
                "availability_contract_valid": None,
                "minimum_availability_margin_samples": None,
                "dense_target_checked_frames": 3,
                "dense_target_expansion_valid": True,
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


def _write_valid_local_cache(
    tmp_path: Path,
    *,
    teacher_latest_offset: int = 0,
    dense_expansion: str = "previous_anchor_hold",
) -> tuple[Path, Path, dict]:
    item = {
        "id": "switch",
        "audio_path": "switch.wav",
        "split": "switch",
        "language": "mixed",
        "speaker_ids": ["speaker-a", "speaker-b"],
        "segments": [
            {"start_seconds": 0.0, "end_seconds": 1.0, "language": "en"},
            {"start_seconds": 1.0, "end_seconds": 2.0, "language": "hi"},
        ],
    }
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    audio_path = tmp_path / "switch.wav"
    audio_path.write_bytes(b"local-audio")
    targets_dir = tmp_path / "targets"
    targets_dir.mkdir()
    target_path = targets_dir / "switch.npz"
    anchors = np.asarray([0, 2], dtype=np.int64)
    anchor_probs = _probabilities(2)
    anchor_probs[1] = np.roll(anchor_probs[1], 1)
    anchor_soft_targets = _soften(anchor_probs)
    probabilities = expand_local_posteriors(
        anchors, anchor_probs, 3, expansion=dense_expansion
    )
    soft_targets = expand_local_posteriors(
        anchors, anchor_soft_targets, 3, expansion=dense_expansion
    )
    ledger = local_target_availability_ledger(anchors, 3)
    ledger["teacher_latest_samples"] = ledger["teacher_latest_samples"].copy()
    ledger["teacher_latest_samples"][1] += teacher_latest_offset
    audio_hash = file_sha256(audio_path)
    manifest_snapshot = capture_manifest_snapshot(manifest_path)
    np.savez_compressed(
        target_path,
        teacher_probs=probabilities,
        teacher_soft_targets=soft_targets,
        anchor_frames=anchors,
        anchor_probs=anchor_probs,
        anchor_soft_targets=anchor_soft_targets,
        in_set_mass=np.asarray([0.75, 0.75], dtype=np.float32),
        full_top1_indices=np.asarray(
            [TEACHER_LANGUAGE_INDICES[0], TEACHER_LANGUAGE_INDICES[1]],
            dtype=np.int64,
        ),
        full_top1_probabilities=np.asarray([0.525, 0.525], dtype=np.float32),
        full_top1_labels=np.asarray(
            [
                TEACHER_LABELS[TEACHER_LANGUAGE_INDICES[0]],
                TEACHER_LABELS[TEACHER_LANGUAGE_INDICES[1]],
            ]
        ),
        language_codes=np.asarray(LANGUAGE_CODES),
        cache_schema_version=np.asarray(TARGET_CACHE_SCHEMA_VERSION),
        clip_id=np.asarray(item["id"]),
        target_kind=np.asarray("local_windows"),
        num_frames=np.asarray(3),
        audio_sha256=np.asarray(audio_hash),
        manifest_record_sha256=np.asarray(manifest_record_sha256(item)),
        manifest_file_sha256=np.asarray(manifest_snapshot.manifest_file_sha256),
        target_configuration_sha256=np.asarray(target_configuration_sha256()),
        teacher_name=np.asarray(TEACHER_NAME),
        teacher_revision=np.asarray(TEACHER_REVISION),
        teacher_artifact_sha256=np.asarray(TEACHER_ARTIFACT_SHA256),
        target_generator_source_sha256=np.asarray(
            target_generator_identity()["source_sha256"]
        ),
        **ledger,
    )
    configuration = target_cache_configuration()
    training_target_audit = build_training_target_audit([item], targets_dir)
    entry = {
        "id": item["id"],
        "frames": 3,
        "target_kind": "local_windows",
        "target_expansion": "previous_anchor_hold",
        "availability_checked_frames": 3,
        "availability_contract_valid": True,
        "minimum_availability_margin_samples": 0,
        "dense_target_checked_frames": 3,
        "dense_target_expansion_valid": True,
        "audio_sha256": audio_hash,
        "manifest_record_sha256": manifest_record_sha256(item),
        "target_file_sha256": file_sha256(target_path),
    }
    metadata = {
        "schema_version": TARGET_CACHE_SCHEMA_VERSION,
        "teacher_identity": configuration["teacher"],
        "target_generator": configuration["target_generator"],
        "target_configuration": configuration,
        "target_configuration_sha256": canonical_json_sha256(configuration),
        "manifest_snapshot": manifest_snapshot.identity(),
        "manifest_file_sha256": manifest_snapshot.manifest_file_sha256,
        "manifest_records_sha256": manifest_records_sha256([item]),
        "target_expansion": "previous_anchor_hold",
        "availability_sample_index_semantics": "exclusive_right_edge_unclipped",
        "availability_checked_frames": 3,
        "availability_contract_valid": True,
        "dense_target_checked_frames": 3,
        "dense_target_expansion_valid": True,
        "dense_target_validation_rtol": DENSE_TARGET_VALIDATION_RTOL,
        "dense_target_validation_atol": DENSE_TARGET_VALIDATION_ATOL,
        "training_target_audit": training_target_audit,
        "training_target_audit_sha256": training_target_audit["audit_sha256"],
        "target_files_sha256": canonical_json_sha256(
            {item["id"]: entry["target_file_sha256"]}
        ),
        "clips": [entry],
    }
    (targets_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    return manifest_path, targets_dir, item


def _rewrite_single_target_and_rebind_metadata(
    targets_dir: Path, item: dict, payload: dict[str, np.ndarray]
) -> None:
    target_path = targets_dir / f"{item['id']}.npz"
    np.savez_compressed(target_path, **payload)
    metadata_path = targets_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target_hash = file_sha256(target_path)
    metadata["clips"][0]["target_file_sha256"] = target_hash
    metadata["target_files_sha256"] = canonical_json_sha256(
        {item["id"]: target_hash}
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


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


def test_manifest_snapshot_separates_exact_bytes_from_semantic_records(
    tmp_path: Path,
) -> None:
    item = {
        "id": "clip",
        "audio_path": "clip.wav",
        "split": "train",
        "language": "en",
        "speaker_id": "speaker",
    }
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    first = capture_manifest_snapshot(manifest_path)
    manifest_path.write_text(
        "  " + json.dumps(item, sort_keys=True, separators=(", ", ": ")) + "  \n",
        encoding="utf-8",
    )
    second = capture_manifest_snapshot(manifest_path)

    assert first.manifest_file_sha256 != second.manifest_file_sha256
    assert first.manifest_records_sha256 == second.manifest_records_sha256
    assert first.canonical_records_bytes == second.canonical_records_bytes


def test_manifest_snapshot_returns_defensive_record_copies(tmp_path: Path) -> None:
    manifest_path, _, item = _write_valid_cache(tmp_path)
    snapshot = capture_manifest_snapshot(manifest_path)
    first = snapshot.records_copy()
    first[0]["text"] = "mutated by one consumer"

    assert snapshot.records_copy() == [item]
    assert snapshot.manifest_records_sha256 == manifest_records_sha256([item])


def test_shared_manifest_snapshot_feeds_consumers_without_a_second_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, targets_dir, _ = _write_valid_cache(tmp_path)
    snapshot = capture_manifest_snapshot(manifest_path)
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == manifest_path:
            raise AssertionError("manifest pathname was reopened")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(
        "streaming_lid.data.load_audio", lambda _path: torch.zeros(720)
    )

    records = snapshot.records_copy()
    assert require_speaker_disjoint(records)["speaker_disjoint"] is True
    cache = TeacherTargetCache(snapshot, targets_dir)
    assert corpus_identity(snapshot)["manifest_snapshot"] == snapshot.identity()
    assert cache.validate_all()["validated_clips"] == 1
    dataset = DistillationDataset(
        snapshot,
        targets_dir,
        splits=("heldout",),
        target_cache=cache,
    )
    assert len(dataset) == 1


def test_strict_cache_accepts_content_bound_target(tmp_path: Path) -> None:
    manifest_path, targets_dir, item = _write_valid_cache(tmp_path)
    cache = TeacherTargetCache(manifest_path, targets_dir)

    value = cache.load(item, "teacher_soft_targets", expected_frames=3)

    assert value.shape == (3, len(LANGUAGE_CODES))
    assert cache.audit()["validated_clips"] == 1


def test_strict_cache_recomputes_local_availability_ledger(tmp_path: Path) -> None:
    manifest_path, targets_dir, item = _write_valid_local_cache(tmp_path)
    cache = TeacherTargetCache(manifest_path, targets_dir)

    value = cache.load(item, "teacher_soft_targets", expected_frames=3)

    assert value.shape == (3, len(LANGUAGE_CODES))


def test_strict_cache_rejects_tampered_local_availability_ledger(
    tmp_path: Path,
) -> None:
    manifest_path, targets_dir, item = _write_valid_local_cache(
        tmp_path, teacher_latest_offset=1
    )
    cache = TeacherTargetCache(manifest_path, targets_dir)

    with pytest.raises(ValueError, match="teacher_latest_samples.*differs"):
        cache.load(item, "teacher_soft_targets", expected_frames=3)


def test_strict_cache_rejects_dense_targets_that_disagree_with_hold_ledger(
    tmp_path: Path,
) -> None:
    manifest_path, targets_dir, item = _write_valid_local_cache(
        tmp_path, dense_expansion="linear_probability"
    )
    cache = TeacherTargetCache(manifest_path, targets_dir)

    with pytest.raises(
        ValueError,
        match="teacher_probs differs from its declared previous_anchor_hold",
    ):
        cache.load(item, "teacher_soft_targets", expected_frames=3)


def test_strict_cache_rejects_soft_anchors_not_derived_at_temperature(
    tmp_path: Path,
) -> None:
    manifest_path, targets_dir, item = _write_valid_local_cache(tmp_path)
    target_path = targets_dir / "switch.npz"
    with np.load(target_path, allow_pickle=False) as target_file:
        payload = {key: np.asarray(target_file[key]) for key in target_file.files}
    bad_soft_anchors = np.roll(payload["anchor_soft_targets"], 1, axis=1)
    payload["anchor_soft_targets"] = bad_soft_anchors
    payload["teacher_soft_targets"] = expand_local_posteriors(
        payload["anchor_frames"], bad_soft_anchors, 3
    )
    np.savez_compressed(target_path, **payload)
    metadata_path = targets_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target_hash = file_sha256(target_path)
    metadata["clips"][0]["target_file_sha256"] = target_hash
    metadata["target_files_sha256"] = canonical_json_sha256(
        {item["id"]: target_hash}
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    cache = TeacherTargetCache(manifest_path, targets_dir)

    with pytest.raises(
        ValueError,
        match="anchor_soft_targets differs from its declared T=2 anchor-temperature",
    ):
        cache.load(item, "teacher_soft_targets", expected_frames=3)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("outside_index", "pinned 107-class teacher output space"),
        ("wrong_label", "does not match pinned teacher index"),
        ("impossible_probability", "below 1/107"),
    ),
)
def test_strict_cache_rejects_impossible_native_teacher_summary(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest_path, targets_dir, item = _write_valid_cache(tmp_path)
    target_path = targets_dir / "clip.npz"
    with np.load(target_path, allow_pickle=False) as target_file:
        payload = {key: np.asarray(target_file[key]) for key in target_file.files}
    if mutation == "outside_index":
        payload["full_top1_indices"] = np.asarray([999], dtype=np.int64)
    elif mutation == "wrong_label":
        payload["full_top1_labels"] = np.asarray(["hi: Hindi"])
    elif mutation == "impossible_probability":
        payload["in_set_mass"] = np.asarray([1e-6], dtype=np.float32)
        payload["full_top1_probabilities"] = np.asarray([1e-6], dtype=np.float32)
    else:  # pragma: no cover - the parameter table is exhaustive
        raise AssertionError(mutation)
    _rewrite_single_target_and_rebind_metadata(targets_dir, item, payload)
    cache = TeacherTargetCache(manifest_path, targets_dir)

    with pytest.raises(ValueError, match=message):
        cache.load(item, "teacher_soft_targets", expected_frames=3)


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


def test_strict_cache_rejects_changed_target_configuration(tmp_path: Path) -> None:
    manifest_path, targets_dir, _ = _write_valid_cache(tmp_path)
    metadata_path = targets_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["target_configuration"]["temperature"] = 99.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="configuration differs"):
        TeacherTargetCache(manifest_path, targets_dir)


def test_strict_cache_rejects_wrong_pinned_teacher(tmp_path: Path) -> None:
    manifest_path, targets_dir, item = _write_valid_cache(tmp_path)
    target_path = targets_dir / "clip.npz"
    _write_target_file(
        target_path,
        item,
        file_sha256(tmp_path / "clip.wav"),
        teacher_revision="mutable-main",
        manifest_file_sha256=capture_manifest_snapshot(
            manifest_path
        ).manifest_file_sha256,
    )
    metadata_path = targets_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["clips"][0]["target_file_sha256"] = file_sha256(target_path)
    metadata["target_files_sha256"] = canonical_json_sha256(
        {item["id"]: file_sha256(target_path)}
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    cache = TeacherTargetCache(manifest_path, targets_dir)

    with pytest.raises(ValueError, match="teacher_revision"):
        cache.load(item, "teacher_soft_targets", expected_frames=3)


def test_strict_cache_rejects_modified_target_file(tmp_path: Path) -> None:
    manifest_path, targets_dir, item = _write_valid_cache(tmp_path)
    _write_target_file(
        targets_dir / "clip.npz",
        item,
        file_sha256(tmp_path / "clip.wav"),
        manifest_file_sha256=capture_manifest_snapshot(
            manifest_path
        ).manifest_file_sha256,
    )
    with (targets_dir / "clip.npz").open("ab") as handle:
        handle.write(b"tamper")
    cache = TeacherTargetCache(manifest_path, targets_dir)

    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        cache.load(item, "teacher_soft_targets", expected_frames=3)


def test_training_target_audit_exposes_skew_despite_equal_clip_counts(
    tmp_path: Path,
) -> None:
    targets_dir = tmp_path / "targets"
    targets_dir.mkdir()
    records = []
    for language_index, language in enumerate(LANGUAGE_CODES):
        for clip_index in range(10):
            item = {
                "id": f"{language}_{clip_index}",
                "audio_path": f"{language}_{clip_index}.wav",
                "split": "train",
                "language": language,
                "speaker_id": f"{language}-voice",
                "audio_recipe": {
                    "provider": "fixture-tts",
                    "voice": f"{language}-voice",
                },
            }
            records.append(item)
            predicted_index = language_index
            if language == "en" and clip_index >= 6:
                predicted_index = LANGUAGE_CODES.index("hi")
            probabilities = np.full(
                (1, len(LANGUAGE_CODES)), 0.05, dtype=np.float32
            )
            probabilities[0, predicted_index] = 0.70
            _write_target_file(
                targets_dir / f"{item['id']}.npz",
                item,
                "unused",
                num_frames=26,
                anchor_probs=probabilities,
                in_set_mass=(1e-6 if language == "gu" else 0.75),
                full_top1_index=(0 if language == "gu" else None),
                full_top1_probability=(0.9 if language == "gu" else None),
            )

    audit = build_training_target_audit(records, targets_dir)

    assert audit["aggregate"]["equal_monolingual_clip_counts"] is True
    assert (
        audit["aggregate"][
            "equal_clip_counts_do_not_imply_equal_full_corpus_t2_target_mass"
        ]
        is True
    )
    assert audit["per_language"]["en"]["teacher_selected_top1_correct"] == 6
    assert audit["per_language"]["en"]["teacher_native_top1_correct"] == 6
    assert audit["per_language"]["gu"]["teacher_selected_top1_correct"] == 10
    assert audit["per_language"]["gu"]["teacher_native_top1_correct"] == 0
    assert all(
        audit["per_language"][language]["monolingual_clips"] == 10
        for language in LANGUAGE_CODES
    )
    assert audit["quality_floor"][
        "failed_teacher_selected_correctness_languages"
    ] == ["en"]
    assert audit["quality_floor"][
        "failed_teacher_native_correctness_languages"
    ] == ["en", "gu"]
    assert audit["quality_floor"]["failed_retained_mass_languages"] == ["gu"]
    assert audit["quality_floor"]["passed"] is False
    assert (
        audit["aggregate"]["full_corpus_target_mass_t2"]["en"]
        < audit["aggregate"]["full_corpus_target_mass_t2"]["hi"]
    )


def test_strict_cache_recomputes_self_rehashed_training_target_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, targets_dir, _ = _write_valid_cache(tmp_path)
    metadata_path = targets_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_target_audit"]["checked_monolingual_train_clips"] = 1
    audit_payload = {
        key: value
        for key, value in metadata["training_target_audit"].items()
        if key != "audit_sha256"
    }
    forged_hash = canonical_json_sha256(audit_payload)
    metadata["training_target_audit"]["audit_sha256"] = forged_hash
    metadata["training_target_audit_sha256"] = forged_hash
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(
        "streaming_lid.data.load_audio", lambda _path: torch.zeros(720)
    )

    cache = TeacherTargetCache(manifest_path, targets_dir)
    with pytest.raises(ValueError, match="does not reproduce"):
        cache.validate_all()
