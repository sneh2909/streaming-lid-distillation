import copy
import json
from pathlib import Path

import pytest
import torch

from streaming_lid.config import LANGUAGE_CODES, TEACHER_NAME
from streaming_lid.data import capture_manifest_snapshot
from streaming_lid.run_identity import (
    CHECKPOINT_SCHEMA_VERSION,
    assert_run_dependency_snapshot_unchanged,
    build_run_identity,
    capture_run_dependency_snapshot,
    configured_model_kwargs,
    run_id_for_identity,
    training_configuration,
    validate_evaluation_run_contract,
)


def _valid_contract(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    manifest_snapshot = capture_manifest_snapshot(manifest_path)
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"audio-one")
    metadata_path = tmp_path / "metadata.json"
    target_identity = {
        "schema_version": 5,
        "target_configuration_sha256": "configuration-one",
        "manifest_snapshot": manifest_snapshot.identity(),
        "manifest_file_sha256": manifest_snapshot.manifest_file_sha256,
        "manifest_records_sha256": manifest_snapshot.manifest_records_sha256,
        "target_files_sha256": "targets-one",
        "teacher_revision": "teacher-revision-one",
        "teacher_artifact_sha256": "teacher-artifact-one",
        "target_generator_source_sha256": "generator-one",
    }
    target_metadata = {
        "schema_version": target_identity["schema_version"],
        "target_configuration_sha256": target_identity[
            "target_configuration_sha256"
        ],
        "manifest_snapshot": target_identity["manifest_snapshot"],
        "manifest_file_sha256": target_identity["manifest_file_sha256"],
        "manifest_records_sha256": target_identity["manifest_records_sha256"],
        "target_files_sha256": target_identity["target_files_sha256"],
        "teacher_identity": {
            "revision": target_identity["teacher_revision"],
            "artifact_sha256": target_identity["teacher_artifact_sha256"],
        },
        "target_generator": {
            "source_sha256": target_identity["target_generator_source_sha256"]
        },
    }
    metadata_path.write_text(
        json.dumps(target_metadata) + "\n", encoding="utf-8"
    )
    model_state = {"weight": torch.tensor([[1.0, 2.0]])}
    training = training_configuration(
        steps=1,
        batch_size=1,
        learning_rate=1e-3,
        threads=6,
        seed=7,
    )
    dependency_snapshot = capture_run_dependency_snapshot(
        manifest_snapshot=manifest_snapshot,
        target_cache_identity=target_identity,
        target_metadata_path=metadata_path,
        training=training,
    )
    run_identity = build_run_identity(
        dependency_snapshot=dependency_snapshot,
        model_state=model_state,
    )
    run_id = run_id_for_identity(run_identity)
    training_contract = {
        "requested_optimizer_steps": 1,
        "successful_optimizer_steps": 1,
        "post_update_checks": 1,
        "all_requested_steps_completed": True,
        "loss_and_gradient_histories_finite": True,
        "post_update_model_state_finite": True,
        "post_update_optimizer_state_finite": True,
        "real_audio_optimizer_step": True,
        "nan_free": True,
    }
    training_evidence = {
        **training_contract,
        "batch_size": 1,
        "losses": [1.0],
    }
    checkpoint_sha256 = "c" * 64
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_identity": run_identity,
        "launch_dependency_snapshot": dependency_snapshot,
        "model_state": model_state,
        "languages": list(LANGUAGE_CODES),
        "teacher": TEACHER_NAME,
        "target_cache": target_identity,
        "model_kwargs": configured_model_kwargs(),
        "seed": 7,
        "steps": 1,
        "training_contract": training_contract,
        "training_evidence": training_evidence,
    }
    train_metrics = {
        "run_id": run_id,
        "run_identity": run_identity,
        "checkpoint_sha256": checkpoint_sha256,
        **training_evidence,
    }
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "train_metrics": train_metrics,
        "records": [item],
        "manifest_path": manifest_path,
        "manifest_snapshot": manifest_snapshot,
        "target_cache_identity": target_identity,
        "target_metadata_path": metadata_path,
        "target_metadata": target_metadata,
        "target_identity": target_identity,
        "training": training,
        "dependency_snapshot": dependency_snapshot,
        "audio_path": audio_path,
    }


def _validate(contract: dict) -> dict:
    return validate_evaluation_run_contract(
        checkpoint=contract["checkpoint"],
        checkpoint_sha256=contract["checkpoint_sha256"],
        train_metrics=contract["train_metrics"],
        manifest_snapshot=contract["manifest_snapshot"],
        target_cache_identity=contract["target_cache_identity"],
        target_metadata_path=contract["target_metadata_path"],
    )


def test_complete_run_identity_contract_accepts_one_bound_run(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)

    validated = _validate(contract)

    assert validated == contract["checkpoint"]["run_identity"]
    assert validated["corpus"]["audio_sha256_by_clip"].keys() == {"clip"}


def test_capture_rejects_an_a_b_a_manifest_generation_mix(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)
    snapshot_a = contract["manifest_snapshot"]
    item_b = {**contract["records"][0], "text": "release B"}
    contract["manifest_path"].write_text(
        json.dumps(item_b) + "\n", encoding="utf-8"
    )
    snapshot_b = capture_manifest_snapshot(contract["manifest_path"])
    target_identity_b = {
        **contract["target_identity"],
        "manifest_snapshot": snapshot_b.identity(),
        "manifest_file_sha256": snapshot_b.manifest_file_sha256,
        "manifest_records_sha256": snapshot_b.manifest_records_sha256,
    }
    metadata_b = {
        **contract["target_metadata"],
        "manifest_snapshot": snapshot_b.identity(),
        "manifest_file_sha256": snapshot_b.manifest_file_sha256,
        "manifest_records_sha256": snapshot_b.manifest_records_sha256,
    }
    contract["target_metadata_path"].write_text(
        json.dumps(metadata_b) + "\n", encoding="utf-8"
    )
    contract["manifest_path"].write_bytes(snapshot_a.raw_bytes)

    with pytest.raises(RuntimeError, match="manifest snapshot bindings differ"):
        capture_run_dependency_snapshot(
            manifest_snapshot=snapshot_a,
            target_cache_identity=target_identity_b,
            target_metadata_path=contract["target_metadata_path"],
            training=contract["training"],
        )


def test_build_run_identity_rejects_a_forged_child_manifest_binding(
    tmp_path: Path,
) -> None:
    contract = _valid_contract(tmp_path)
    forged = copy.deepcopy(contract["dependency_snapshot"])
    forged["corpus"]["manifest_snapshot"]["manifest_file_sha256"] = "b" * 64

    with pytest.raises(ValueError, match="manifest snapshot bindings differ"):
        build_run_identity(
            dependency_snapshot=forged,
            model_state=contract["checkpoint"]["model_state"],
        )


def test_evaluation_rejects_a_different_manifest_generation_before_scoring(
    tmp_path: Path,
) -> None:
    contract = _valid_contract(tmp_path)
    changed = {**contract["records"][0], "text": "evaluation release B"}
    contract["manifest_path"].write_text(
        json.dumps(changed) + "\n", encoding="utf-8"
    )
    contract["manifest_snapshot"] = capture_manifest_snapshot(
        contract["manifest_path"]
    )

    with pytest.raises(ValueError, match="evaluation manifest snapshot differs"):
        _validate(contract)


def test_main_capture_rejects_a_live_path_source_identity(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)

    with pytest.raises(RuntimeError, match="pre-import executed-source snapshot"):
        capture_run_dependency_snapshot(
            manifest_snapshot=contract["manifest_snapshot"],
            target_cache_identity=contract["target_identity"],
            target_metadata_path=contract["target_metadata_path"],
            training=contract["training"],
            require_executed_source=True,
        )


def test_launch_snapshot_is_immutable_and_recheck_detects_changed_audio(
    tmp_path: Path,
) -> None:
    contract = _valid_contract(tmp_path)
    launch_snapshot = copy.deepcopy(contract["dependency_snapshot"])
    contract["training"]["requested_steps"] = 99
    contract["audio_path"].write_bytes(b"audio-two")

    assert contract["dependency_snapshot"] == launch_snapshot
    run_identity = build_run_identity(
        dependency_snapshot=contract["dependency_snapshot"],
        model_state=contract["checkpoint"]["model_state"],
    )
    assert run_identity["training"]["requested_steps"] == 1
    with pytest.raises(RuntimeError, match="corpus"):
        assert_run_dependency_snapshot_unchanged(
            contract["dependency_snapshot"],
            manifest_snapshot=contract["manifest_snapshot"],
            target_cache_identity=contract["target_identity"],
            target_metadata_path=contract["target_metadata_path"],
            training=launch_snapshot["training"],
            stage="checkpoint publication",
        )


def test_dependency_recheck_rejects_stale_manifest_and_target_metadata(
    tmp_path: Path,
) -> None:
    manifest_contract = _valid_contract(tmp_path / "manifest-change")
    changed_item = {**manifest_contract["records"][0], "text": "changed"}
    manifest_contract["manifest_path"].write_text(
        json.dumps(changed_item) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="manifest snapshot changed"):
        assert_run_dependency_snapshot_unchanged(
            manifest_contract["dependency_snapshot"],
            manifest_snapshot=manifest_contract["manifest_snapshot"],
            target_cache_identity=manifest_contract["target_identity"],
            target_metadata_path=manifest_contract["target_metadata_path"],
            training=manifest_contract["training"],
            stage="optimization",
        )

    target_contract = _valid_contract(tmp_path / "target-change")
    changed_metadata = copy.deepcopy(target_contract["target_metadata"])
    changed_metadata["target_generator"]["source_sha256"] = "generator-two"
    target_contract["target_metadata_path"].write_text(
        json.dumps(changed_metadata) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="target cache identity changed"):
        assert_run_dependency_snapshot_unchanged(
            target_contract["dependency_snapshot"],
            manifest_snapshot=target_contract["manifest_snapshot"],
            target_cache_identity=target_contract["target_identity"],
            target_metadata_path=target_contract["target_metadata_path"],
            training=target_contract["training"],
            stage="checkpoint publication",
        )


def test_run_identity_rejects_changed_audio_bytes(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)
    contract["audio_path"].write_bytes(b"audio-two")

    with pytest.raises(ValueError, match="manifest/audio corpus"):
        _validate(contract)


def test_run_identity_rejects_changed_timing_even_if_ids_are_recomputed(
    tmp_path: Path,
) -> None:
    contract = _valid_contract(tmp_path)
    checkpoint = copy.deepcopy(contract["checkpoint"])
    checkpoint["run_identity"]["pipeline"]["distillation"][
        "label_delay_frames"
    ] += 1
    checkpoint["launch_dependency_snapshot"]["pipeline"] = copy.deepcopy(
        checkpoint["run_identity"]["pipeline"]
    )
    checkpoint["run_id"] = run_id_for_identity(checkpoint["run_identity"])
    contract["checkpoint"] = checkpoint
    contract["train_metrics"] = {
        "run_id": checkpoint["run_id"],
        "run_identity": checkpoint["run_identity"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        **checkpoint["training_evidence"],
    }

    with pytest.raises(ValueError, match="pipeline configuration/source"):
        _validate(contract)


def test_run_identity_rejects_different_target_cache(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)
    contract["target_cache_identity"] = {
        **contract["target_cache_identity"],
        "target_files_sha256": "targets-two",
    }
    changed_metadata = {
        **contract["target_metadata"],
        "target_files_sha256": "targets-two",
    }
    contract["target_metadata_path"].write_text(
        json.dumps(changed_metadata) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="target cache differs"):
        _validate(contract)


def test_run_identity_rejects_unrelated_training_metrics(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)
    contract["train_metrics"] = {
        **contract["train_metrics"],
        "run_id": "lidrun-unrelated",
    }

    with pytest.raises(ValueError, match="training metrics do not exactly match"):
        _validate(contract)


def test_run_identity_rejects_changed_model_state(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)
    contract["checkpoint"]["model_state"]["weight"][0, 0] = 99.0

    with pytest.raises(ValueError, match="model state differs"):
        _validate(contract)
