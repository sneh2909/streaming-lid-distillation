import copy
import json
from pathlib import Path

import pytest
import torch

from streaming_lid.config import LANGUAGE_CODES, TEACHER_NAME
from streaming_lid.run_identity import (
    CHECKPOINT_SCHEMA_VERSION,
    build_run_identity,
    configured_model_kwargs,
    run_id_for_identity,
    training_configuration,
    validate_evaluation_run_contract,
)


def _valid_contract(tmp_path: Path) -> dict:
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
    audio_path.write_bytes(b"audio-one")
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"target_cache": "one"}\n', encoding="utf-8")
    target_identity = {"schema_version": 2, "target_files_sha256": "targets-one"}
    model_state = {"weight": torch.tensor([[1.0, 2.0]])}
    training = training_configuration(
        steps=1,
        batch_size=1,
        learning_rate=1e-3,
        threads=6,
        seed=7,
    )
    run_identity = build_run_identity(
        records=[item],
        manifest_path=manifest_path,
        target_cache_identity=target_identity,
        target_metadata_path=metadata_path,
        training=training,
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
        "target_cache_identity": target_identity,
        "target_metadata_path": metadata_path,
        "audio_path": audio_path,
    }


def _validate(contract: dict) -> dict:
    return validate_evaluation_run_contract(
        checkpoint=contract["checkpoint"],
        checkpoint_sha256=contract["checkpoint_sha256"],
        train_metrics=contract["train_metrics"],
        records=contract["records"],
        manifest_path=contract["manifest_path"],
        target_cache_identity=contract["target_cache_identity"],
        target_metadata_path=contract["target_metadata_path"],
    )


def test_complete_run_identity_contract_accepts_one_bound_run(tmp_path: Path) -> None:
    contract = _valid_contract(tmp_path)

    validated = _validate(contract)

    assert validated == contract["checkpoint"]["run_identity"]
    assert validated["corpus"]["audio_sha256_by_clip"].keys() == {"clip"}


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
        "schema_version": 2,
        "target_files_sha256": "targets-two",
    }

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
