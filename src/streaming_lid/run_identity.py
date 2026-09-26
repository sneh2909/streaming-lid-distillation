"""Content-bound training/evaluation identity for the main pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .config import (
    ALGORITHMIC_LATENCY_MS,
    CHUNK_FRAMES,
    CHUNK_MS,
    EARLY_RAMP_FRAMES,
    FRAME_MS,
    HOP_LENGTH,
    LABEL_DELAY_FRAMES,
    LANGUAGE_CODES,
    MODEL_LOOKAHEAD_FRAMES,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    STUDENT_DILATIONS,
    STUDENT_HIDDEN_SIZE,
    TEACHER_ARTIFACT_SHA256,
    TEACHER_FUTURE_MS,
    TEACHER_HOP_FRAMES,
    TEACHER_NAME,
    TEACHER_PAST_MS,
    TEACHER_REVISION,
    TEACHER_TEMPERATURE,
    WINDOW_MS,
    WIN_LENGTH,
)
from .data import (
    canonical_json_sha256,
    file_sha256,
    manifest_records_sha256,
    resolve_audio_path,
)


RUN_IDENTITY_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 2
PIPELINE_SOURCE_FILES = (
    "scripts/train.py",
    "scripts/eval.py",
    "src/streaming_lid/audio.py",
    "src/streaming_lid/config.py",
    "src/streaming_lid/data.py",
    "src/streaming_lid/loss.py",
    "src/streaming_lid/model.py",
    "src/streaming_lid/run_identity.py",
)


def configured_model_kwargs() -> dict[str, Any]:
    """Return every constructor argument that fixes the student architecture."""
    return {
        "num_languages": len(LANGUAGE_CODES),
        "n_mels": N_MELS,
        "hidden_size": STUDENT_HIDDEN_SIZE,
        "lookahead_frames": MODEL_LOOKAHEAD_FRAMES,
        "dilations": list(STUDENT_DILATIONS),
    }


def pipeline_source_identity() -> dict[str, Any]:
    """Fingerprint source that defines training and evaluation semantics."""
    repository_root = Path(__file__).resolve().parents[2]
    files = {}
    for relative_path in PIPELINE_SOURCE_FILES:
        payload = (repository_root / relative_path).read_bytes()
        files[relative_path] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return {
        "source_sha256": canonical_json_sha256(files),
        "files": files,
    }


def pipeline_configuration() -> dict[str, Any]:
    """Return the complete model, timing, frontend, and teacher contract."""
    model_kwargs = configured_model_kwargs()
    receptive_field_frames = 1 + 2 * sum(model_kwargs["dilations"])
    return {
        "language_codes": list(LANGUAGE_CODES),
        "teacher": {
            "model_id": TEACHER_NAME,
            "revision": TEACHER_REVISION,
            "artifact_sha256": TEACHER_ARTIFACT_SHA256,
        },
        "frontend": {
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "win_length": WIN_LENGTH,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "center": False,
            "utterance_normalization": False,
        },
        "teacher_target_timing": {
            "past_ms": TEACHER_PAST_MS,
            "future_ms": TEACHER_FUTURE_MS,
            "hop_frames": TEACHER_HOP_FRAMES,
        },
        "student": {
            "class": "streaming_lid.model.CausalLIDStudent",
            "model_kwargs": model_kwargs,
            "receptive_field_frames": receptive_field_frames,
        },
        "distillation": {
            "label_delay_frames": LABEL_DELAY_FRAMES,
            "model_lookahead_frames": MODEL_LOOKAHEAD_FRAMES,
            "temperature": TEACHER_TEMPERATURE,
            "early_ramp_frames": EARLY_RAMP_FRAMES,
        },
        "streaming": {
            "chunk_frames": CHUNK_FRAMES,
            "chunk_ms": CHUNK_MS,
            "frame_ms": FRAME_MS,
            "window_ms": WINDOW_MS,
            "algorithmic_latency_ms": ALGORITHMIC_LATENCY_MS,
            "provisional_tail_withheld": True,
        },
        "source": pipeline_source_identity(),
    }


def _manifest_file_snapshot(manifest_path: str | Path) -> tuple[list[dict], str]:
    """Parse records and hash the exact same manifest byte snapshot."""
    path = Path(manifest_path)
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse manifest snapshot {path}: {error}") from error
    return records, hashlib.sha256(payload).hexdigest()


def corpus_identity(records: list[dict], manifest_path: str | Path) -> dict[str, Any]:
    """Bind one coherent manifest byte snapshot to every referenced audio file."""
    live_records, manifest_file_sha256 = _manifest_file_snapshot(manifest_path)
    if records != live_records:
        raise RuntimeError("manifest records changed while capturing run dependencies")
    ids = [item.get("id") for item in records]
    if any(not isinstance(clip_id, str) for clip_id in ids):
        raise ValueError("every manifest record must have a string clip ID")
    if len(set(ids)) != len(ids):
        raise ValueError("manifest contains duplicate clip IDs")
    audio_sha256_by_clip = {
        item["id"]: file_sha256(resolve_audio_path(item, manifest_path))
        for item in records
    }
    return {
        "n_records": len(records),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_records_sha256": manifest_records_sha256(records),
        "audio_files_sha256": canonical_json_sha256(audio_sha256_by_clip),
        "audio_sha256_by_clip": audio_sha256_by_clip,
    }


def _target_identity_from_metadata(
    metadata: Mapping[str, Any], metadata_path: Path
) -> dict[str, Any]:
    """Read the run-level target identity from one parsed metadata snapshot."""
    try:
        teacher_identity = metadata["teacher_identity"]
        target_generator = metadata["target_generator"]
        return {
            "schema_version": metadata["schema_version"],
            "target_configuration_sha256": metadata[
                "target_configuration_sha256"
            ],
            "manifest_records_sha256": metadata["manifest_records_sha256"],
            "target_files_sha256": metadata["target_files_sha256"],
            "teacher_revision": teacher_identity["revision"],
            "teacher_artifact_sha256": teacher_identity["artifact_sha256"],
            "target_generator_source_sha256": target_generator["source_sha256"],
        }
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"target metadata {metadata_path} is missing run-identity fields"
        ) from error


def target_cache_run_identity(
    target_cache_identity: Mapping[str, Any], metadata_path: str | Path
) -> dict[str, Any]:
    """Bind a semantic target identity to the same metadata bytes that declare it."""
    path = Path(metadata_path)
    payload = path.read_bytes()
    try:
        metadata = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse target metadata snapshot {path}: {error}") from error
    declared_identity = _target_identity_from_metadata(metadata, path)
    if dict(target_cache_identity) != declared_identity:
        raise RuntimeError(
            "target cache identity changed while capturing run dependencies"
        )
    return {
        "identity": declared_identity,
        "metadata_sha256": hashlib.sha256(payload).hexdigest(),
    }


def training_configuration(
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    threads: int,
    seed: int,
) -> dict[str, Any]:
    """Record every training-loop setting that can select the final weights."""
    return {
        "requested_steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 5.0,
        "optimizer": "AdamW",
        "shuffle": True,
        "drop_last": True,
        "threads": threads,
        "seed": seed,
    }


def capture_run_dependency_snapshot(
    *,
    records: list[dict],
    manifest_path: str | Path,
    target_cache_identity: Mapping[str, Any],
    target_metadata_path: str | Path,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture every live dependency before audio/target tensors are preloaded."""
    return {
        "pipeline": pipeline_configuration(),
        "corpus": corpus_identity(records, manifest_path),
        "target_cache": target_cache_run_identity(
            target_cache_identity, target_metadata_path
        ),
        "training": copy.deepcopy(dict(training)),
    }


def assert_run_dependency_snapshot_unchanged(
    launch_snapshot: Mapping[str, Any],
    *,
    records: list[dict],
    manifest_path: str | Path,
    target_cache_identity: Mapping[str, Any],
    target_metadata_path: str | Path,
    training: Mapping[str, Any],
    stage: str,
) -> None:
    """Fail publication if any source, data, target, or setting changed in-run."""
    current = capture_run_dependency_snapshot(
        records=records,
        manifest_path=manifest_path,
        target_cache_identity=target_cache_identity,
        target_metadata_path=target_metadata_path,
        training=training,
    )
    changed = [
        name
        for name in ("pipeline", "corpus", "target_cache", "training")
        if launch_snapshot.get(name) != current[name]
    ]
    if changed:
        raise RuntimeError(
            "run dependencies changed after the launch snapshot "
            f"before {stage}: {', '.join(changed)}"
        )


def model_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes, and bytes without serialization noise."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_run_identity(
    *,
    dependency_snapshot: Mapping[str, Any],
    model_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Bind the final model to the immutable dependency snapshot from launch."""
    required = {"pipeline", "corpus", "target_cache", "training"}
    if set(dependency_snapshot) != required:
        raise ValueError(
            "dependency snapshot fields differ from the run-identity contract"
        )
    return {
        "schema_version": RUN_IDENTITY_SCHEMA_VERSION,
        **copy.deepcopy(dict(dependency_snapshot)),
        "model_state_sha256": model_state_sha256(model_state),
    }


def run_id_for_identity(run_identity: Mapping[str, Any]) -> str:
    """Return a readable content ID for one complete trained-model identity."""
    return f"lidrun-{canonical_json_sha256(run_identity)}"


def validate_evaluation_run_contract(
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_sha256: str,
    train_metrics: Mapping[str, Any],
    records: list[dict],
    manifest_path: str | Path,
    target_cache_identity: Mapping[str, Any],
    target_metadata_path: str | Path,
) -> dict[str, Any]:
    """Fail closed unless checkpoint, data, targets, config, and metrics are one run."""
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version is missing or unsupported")
    run_identity = checkpoint.get("run_identity")
    if not isinstance(run_identity, dict):
        raise ValueError("checkpoint is missing its run identity")
    if run_identity.get("schema_version") != RUN_IDENTITY_SCHEMA_VERSION:
        raise ValueError("run identity schema version is missing or unsupported")

    expected_launch_snapshot = {
        key: run_identity.get(key)
        for key in ("pipeline", "corpus", "target_cache", "training")
    }
    if checkpoint.get("launch_dependency_snapshot") != expected_launch_snapshot:
        raise ValueError(
            "checkpoint launch dependency snapshot contradicts its run identity"
        )

    expected_run_id = run_id_for_identity(run_identity)
    if checkpoint.get("run_id") != expected_run_id:
        raise ValueError("checkpoint run ID does not match its run identity")
    if run_identity.get("model_state_sha256") != model_state_sha256(
        checkpoint.get("model_state", {})
    ):
        raise ValueError("checkpoint model state differs from its run identity")

    current_pipeline = pipeline_configuration()
    if run_identity.get("pipeline") != current_pipeline:
        raise ValueError(
            "evaluation pipeline configuration/source differs from the training run"
        )
    current_corpus = corpus_identity(records, manifest_path)
    if run_identity.get("corpus") != current_corpus:
        raise ValueError("evaluation manifest/audio corpus differs from the training run")
    current_targets = target_cache_run_identity(
        target_cache_identity, target_metadata_path
    )
    if run_identity.get("target_cache") != current_targets:
        raise ValueError("evaluation target cache differs from the training run")

    model_kwargs = current_pipeline["student"]["model_kwargs"]
    duplicate_expectations = {
        "languages": current_pipeline["language_codes"],
        "teacher": current_pipeline["teacher"]["model_id"],
        "target_cache": current_targets["identity"],
        "model_kwargs": model_kwargs,
        "seed": run_identity["training"]["seed"],
    }
    for key, expected in duplicate_expectations.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"checkpoint field {key!r} contradicts its run identity")

    training_evidence = checkpoint.get("training_evidence")
    if not isinstance(training_evidence, dict):
        raise ValueError("checkpoint is missing bound training evidence")
    training_settings = run_identity.get("training", {})
    evidence_expectations = {
        "requested_optimizer_steps": training_settings.get("requested_steps"),
        "batch_size": training_settings.get("batch_size"),
    }
    for key, expected in evidence_expectations.items():
        if training_evidence.get(key) != expected:
            raise ValueError(
                f"training evidence field {key!r} contradicts the run identity"
            )
    contract_keys = (
        "requested_optimizer_steps",
        "successful_optimizer_steps",
        "post_update_checks",
        "all_requested_steps_completed",
        "loss_and_gradient_histories_finite",
        "post_update_model_state_finite",
        "post_update_optimizer_state_finite",
        "real_audio_optimizer_step",
        "nan_free",
    )
    expected_training_contract = {
        key: training_evidence.get(key) for key in contract_keys
    }
    if checkpoint.get("training_contract") != expected_training_contract:
        raise ValueError("checkpoint training contract contradicts its evidence")
    expected_train_metrics = {
        "run_id": expected_run_id,
        "run_identity": run_identity,
        "checkpoint_sha256": checkpoint_sha256,
        **training_evidence,
    }
    if dict(train_metrics) != expected_train_metrics:
        raise ValueError(
            "training metrics do not exactly match the evaluated checkpoint run"
        )
    if checkpoint.get("steps") != training_evidence.get(
        "successful_optimizer_steps"
    ):
        raise ValueError("checkpoint step count contradicts its training evidence")
    if train_metrics.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("training metrics refer to a different checkpoint file")
    return run_identity
