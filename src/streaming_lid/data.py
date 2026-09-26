"""Manifest and cached-teacher-target loading with provenance checks."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .audio import LogMelFrontend, feature_frame_count, load_audio
from .config import (
    HOP_LENGTH,
    LABEL_DELAY_FRAMES,
    LANGUAGE_CODES,
    MODEL_LOOKAHEAD_FRAMES,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    TEACHER_FUTURE_MS,
    TEACHER_HOP_FRAMES,
    TEACHER_ARTIFACT_FILES,
    TEACHER_ARTIFACT_SHA256,
    TEACHER_LANGUAGE_INDICES,
    TEACHER_NAME,
    TEACHER_PAST_MS,
    TEACHER_REVISION,
    TEACHER_TEMPERATURE,
    TEACHER_TARGET_EXPANSION,
    WIN_LENGTH,
)


TARGET_CACHE_SCHEMA_VERSION = 3
TARGET_GENERATOR_SOURCE_FILES = (
    "scripts/teacher_targets.py",
    "src/streaming_lid/audio.py",
    "src/streaming_lid/config.py",
    "src/streaming_lid/data.py",
)
TARGET_PROBABILITY_KEYS = (
    "teacher_probs",
    "teacher_soft_targets",
    "anchor_probs",
)
TARGET_AVAILABILITY_KEYS = (
    "semantic_frames",
    "left_anchor_frames",
    "right_anchor_frames",
    "source_anchor_frames",
    "teacher_latest_samples",
    "student_latest_samples",
    "availability_valid",
)


def read_manifest(path: str | Path) -> list[dict]:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_audio_path(item: dict, manifest_path: str | Path) -> Path:
    return Path(manifest_path).parent / item["audio_path"]


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading the full payload into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON-compatible data independently of dictionary insertion order."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def teacher_artifact_identity() -> dict[str, Any]:
    return {
        "model_id": TEACHER_NAME,
        "revision": TEACHER_REVISION,
        "artifact_sha256": TEACHER_ARTIFACT_SHA256,
        "files": {
            filename: dict(properties)
            for filename, properties in TEACHER_ARTIFACT_FILES.items()
        },
    }


def target_generator_identity() -> dict[str, Any]:
    """Fingerprint target-producing source plus its runtime dependencies."""
    repository_root = Path(__file__).resolve().parents[2]
    files = {
        relative_path: {
            "sha256": file_sha256(repository_root / relative_path),
            "bytes": (repository_root / relative_path).stat().st_size,
        }
        for relative_path in TARGET_GENERATOR_SOURCE_FILES
    }
    dependencies = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchaudio": importlib.metadata.version("torchaudio"),
        "soundfile": importlib.metadata.version("soundfile"),
        "speechbrain": importlib.metadata.version("speechbrain"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
    }
    source_sha256 = canonical_json_sha256(files)
    return {
        "source_sha256": source_sha256,
        "identity_sha256": canonical_json_sha256(
            {"source_sha256": source_sha256, "dependencies": dependencies}
        ),
        "files": files,
        "dependencies": dependencies,
    }


def target_cache_configuration() -> dict[str, Any]:
    """Return every code/config choice that determines cached target semantics."""
    return {
        "schema_version": TARGET_CACHE_SCHEMA_VERSION,
        "teacher": teacher_artifact_identity(),
        "language_codes": list(LANGUAGE_CODES),
        "teacher_language_indices": dict(
            zip(LANGUAGE_CODES, TEACHER_LANGUAGE_INDICES, strict=True)
        ),
        "temperature": TEACHER_TEMPERATURE,
        "probability_space": "teacher_log_posterior_restricted_and_renormalized",
        "anchor_expansion": TEACHER_TARGET_EXPANSION,
        "monolingual_target_kind": "converged_utterance",
        "switch_target_kind": "local_windows",
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "win_length": WIN_LENGTH,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "window_past_ms": TEACHER_PAST_MS,
        "window_future_ms": TEACHER_FUTURE_MS,
        "target_hop_frames": TEACHER_HOP_FRAMES,
        "availability_contract": {
            "sample_index_semantics": "exclusive_right_edge_unclipped",
            "label_delay_frames": LABEL_DELAY_FRAMES,
            "model_lookahead_frames": MODEL_LOOKAHEAD_FRAMES,
            "teacher_latest_formula": (
                "source_anchor*hop_length+win_length+window_future_samples"
            ),
            "student_latest_formula": (
                "semantic_frame*hop_length+win_length+"
                "(label_delay_frames+model_lookahead_frames)*hop_length"
            ),
            "require_teacher_not_after_student": True,
        },
        "target_generator": target_generator_identity(),
    }


def target_configuration_sha256() -> str:
    return canonical_json_sha256(target_cache_configuration())


def manifest_records_sha256(records: list[dict]) -> str:
    return canonical_json_sha256(records)


def manifest_record_sha256(item: dict) -> str:
    return canonical_json_sha256(item)


def target_kind_for_item(item: dict) -> str:
    return "local_windows" if item["language"] == "mixed" else "converged_utterance"


def _validated_anchor_frames(
    anchor_frames: np.ndarray, num_frames: int
) -> np.ndarray:
    """Return a strict integer anchor grid that covers the complete frame range."""
    anchors = np.asarray(anchor_frames)
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if anchors.ndim != 1 or len(anchors) == 0:
        raise ValueError("anchor_frames must be a non-empty one-dimensional array")
    if not np.issubdtype(anchors.dtype, np.integer):
        raise ValueError("anchor_frames must contain integers")
    anchors = anchors.astype(np.int64, copy=False)
    if anchors[0] != 0 or anchors[-1] != num_frames - 1:
        raise ValueError(
            "anchor_frames must include frame zero and the final semantic frame"
        )
    if np.any(np.diff(anchors) <= 0):
        raise ValueError("anchor_frames must be strictly increasing")
    return anchors


def local_target_availability_ledger(
    anchor_frames: np.ndarray,
    num_frames: int,
    *,
    expansion: str = TEACHER_TARGET_EXPANSION,
    label_delay_frames: int = LABEL_DELAY_FRAMES,
    model_lookahead_frames: int = MODEL_LOOKAHEAD_FRAMES,
) -> dict[str, np.ndarray]:
    """Map every local target frame to the teacher/student evidence clocks.

    Sample values are exclusive right edges on the unpadded clip clock. They
    are intentionally not clipped to the waveform length: clipping both sides
    at end of stream could make an unavailable future anchor appear valid.
    """
    anchors = _validated_anchor_frames(anchor_frames, num_frames)
    if label_delay_frames < 0 or model_lookahead_frames < 0:
        raise ValueError("delay and lookahead frames must be non-negative")
    semantic_frames = np.arange(num_frames, dtype=np.int64)
    left_indices = np.searchsorted(anchors, semantic_frames, side="right") - 1
    right_indices = np.searchsorted(anchors, semantic_frames, side="left")
    left_anchor_frames = anchors[left_indices]
    right_anchor_frames = anchors[right_indices]
    if expansion == "previous_anchor_hold":
        source_anchor_frames = left_anchor_frames
    elif expansion == "linear_probability":
        # A linearly blended non-anchor target depends on the next anchor too;
        # the right anchor is therefore its latest teacher dependency.
        source_anchor_frames = right_anchor_frames
    else:
        raise ValueError(f"unsupported target expansion {expansion!r}")

    future_samples = round(TEACHER_FUTURE_MS * SAMPLE_RATE / 1_000)
    teacher_latest_samples = (
        source_anchor_frames * HOP_LENGTH + WIN_LENGTH + future_samples
    )
    student_latest_samples = (
        semantic_frames * HOP_LENGTH
        + WIN_LENGTH
        + (label_delay_frames + model_lookahead_frames) * HOP_LENGTH
    )
    availability_valid = teacher_latest_samples <= student_latest_samples
    return {
        "semantic_frames": semantic_frames,
        "left_anchor_frames": left_anchor_frames,
        "right_anchor_frames": right_anchor_frames,
        "source_anchor_frames": source_anchor_frames,
        "teacher_latest_samples": teacher_latest_samples.astype(np.int64),
        "student_latest_samples": student_latest_samples.astype(np.int64),
        "availability_valid": availability_valid,
    }


def expand_local_posteriors(
    anchor_frames: np.ndarray,
    anchor_values: np.ndarray,
    num_frames: int,
    *,
    expansion: str = TEACHER_TARGET_EXPANSION,
) -> np.ndarray:
    """Expand sparse local teacher posteriors without hiding their dependencies."""
    anchors = _validated_anchor_frames(anchor_frames, num_frames)
    values = np.asarray(anchor_values)
    if values.ndim != 2 or len(values) != len(anchors):
        raise ValueError(
            "anchor_values must have shape [len(anchor_frames), classes]"
        )
    semantic_frames = np.arange(num_frames, dtype=np.int64)
    if expansion == "previous_anchor_hold":
        indices = np.searchsorted(anchors, semantic_frames, side="right") - 1
        expanded = values[indices]
    elif expansion == "linear_probability":
        expanded = np.stack(
            [
                np.interp(semantic_frames, anchors, values[:, column])
                for column in range(values.shape[1])
            ],
            axis=-1,
        )
    else:
        raise ValueError(f"unsupported target expansion {expansion!r}")
    expanded = np.asarray(expanded, dtype=np.float32)
    expanded /= np.clip(expanded.sum(axis=-1, keepdims=True), 1e-8, None)
    return expanded


def _npz_scalar(target_file: np.lib.npyio.NpzFile, key: str, path: Path) -> Any:
    if key not in target_file:
        raise ValueError(f"target cache {path} is missing metadata field {key!r}")
    value = np.asarray(target_file[key])
    if value.size != 1:
        raise ValueError(f"target cache {path} field {key!r} must be scalar")
    return value.reshape(()).item()


def _validate_local_target_availability(
    target_file: np.lib.npyio.NpzFile,
    *,
    path: Path,
    num_frames: int,
) -> dict[str, int | bool]:
    """Recompute and validate the complete dense-frame availability ledger."""
    if "anchor_frames" not in target_file:
        raise ValueError(f"target cache {path} is missing array 'anchor_frames'")
    anchor_frames = np.asarray(target_file["anchor_frames"])
    expected = local_target_availability_ledger(
        anchor_frames,
        num_frames,
        expansion=TEACHER_TARGET_EXPANSION,
    )
    for key in TARGET_AVAILABILITY_KEYS:
        if key not in target_file:
            raise ValueError(
                f"target cache {path} is missing availability field {key!r}"
            )
        actual = np.asarray(target_file[key])
        wanted = expected[key]
        if actual.shape != (num_frames,):
            raise ValueError(
                f"target cache {path} availability field {key!r} must have "
                f"shape ({num_frames},), got {actual.shape}"
            )
        if key == "availability_valid":
            if not np.issubdtype(actual.dtype, np.bool_):
                raise ValueError(
                    f"target cache {path} availability field {key!r} must be boolean"
                )
        elif not np.issubdtype(actual.dtype, np.integer):
            raise ValueError(
                f"target cache {path} availability field {key!r} must be integer"
            )
        if not np.array_equal(actual, wanted):
            mismatch = int(np.flatnonzero(actual != wanted)[0])
            raise ValueError(
                f"target cache {path} availability field {key!r} differs at "
                f"frame {mismatch}: found {actual[mismatch]!r}, "
                f"expected {wanted[mismatch]!r}"
            )
    if not bool(expected["availability_valid"].all()):
        frame = int(np.flatnonzero(~expected["availability_valid"])[0])
        raise ValueError(
            f"target cache {path} violates teacher/student availability at frame "
            f"{frame}: teacher latest sample "
            f"{expected['teacher_latest_samples'][frame]} exceeds student latest "
            f"sample {expected['student_latest_samples'][frame]}"
        )
    margins = expected["student_latest_samples"] - expected["teacher_latest_samples"]
    return {
        "availability_checked_frames": num_frames,
        "availability_contract_valid": True,
        "minimum_availability_margin_samples": int(margins.min()),
    }


def _validate_probability_array(
    value: np.ndarray,
    *,
    name: str,
    path: Path,
    expected_classes: int,
) -> None:
    if value.ndim != 2 or value.shape[1] != expected_classes:
        raise ValueError(
            f"target cache {path} {name} must have shape [frames, "
            f"{expected_classes}], got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"target cache {path} {name} contains non-finite values")
    if np.any(value < -1e-7):
        raise ValueError(f"target cache {path} {name} contains negative probabilities")
    if len(value) and not np.allclose(value.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"target cache {path} {name} rows are not normalized")


def load_teacher_target_array(
    target_path: str | Path,
    array_name: str,
    *,
    expected_frames: int | None = None,
    expected_languages: tuple[str, ...] = LANGUAGE_CODES,
) -> np.ndarray:
    """Load one target array after validating class order and probabilities.

    This basic validator is also usable by isolated experiments with their own
    provenance scheme. The main train/eval pipeline wraps it in
    :class:`TeacherTargetCache`, which additionally binds the file to the
    manifest, waveform bytes, target configuration, and metadata index.
    """
    path = Path(target_path)
    with np.load(path, allow_pickle=False) as target_file:
        if "language_codes" not in target_file:
            raise ValueError(f"target cache {path} is missing language_codes")
        actual_languages = tuple(
            str(code) for code in target_file["language_codes"].tolist()
        )
        if actual_languages != expected_languages:
            raise ValueError(
                f"target cache {path} language order {actual_languages} differs "
                f"from configured order {expected_languages}"
            )
        if array_name not in target_file:
            raise ValueError(f"target cache {path} is missing array {array_name!r}")
        for key in TARGET_PROBABILITY_KEYS:
            if key not in target_file:
                raise ValueError(f"target cache {path} is missing array {key!r}")
            _validate_probability_array(
                np.asarray(target_file[key]),
                name=key,
                path=path,
                expected_classes=len(expected_languages),
            )
        teacher_probs = np.asarray(target_file["teacher_probs"])
        soft_targets = np.asarray(target_file["teacher_soft_targets"])
        if teacher_probs.shape != soft_targets.shape:
            raise ValueError(
                f"target cache {path} hard/soft frame shapes differ: "
                f"{teacher_probs.shape} versus {soft_targets.shape}"
            )
        if expected_frames is not None and teacher_probs.shape[0] != expected_frames:
            raise ValueError(
                f"target cache {path} has {teacher_probs.shape[0]} frames, "
                f"expected {expected_frames}"
            )
        return np.asarray(target_file[array_name]).copy()


class TeacherTargetCache:
    """Strict main-pipeline view of a content-bound teacher target directory."""

    def __init__(self, manifest_path: str | Path, targets_dir: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.targets_dir = Path(targets_dir)
        self.records = read_manifest(self.manifest_path)
        self.records_by_id = {item["id"]: item for item in self.records}
        if len(self.records_by_id) != len(self.records):
            raise ValueError("manifest contains duplicate clip IDs")

        metadata_path = self.targets_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing teacher target metadata: {metadata_path}")
        try:
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(
                f"cannot read teacher target metadata {metadata_path}: {error}"
            ) from error

        expected_configuration = target_cache_configuration()
        expected_configuration_hash = canonical_json_sha256(expected_configuration)
        if self.metadata.get("schema_version") != TARGET_CACHE_SCHEMA_VERSION:
            raise ValueError(
                "target cache schema differs from current configuration: "
                f"found {self.metadata.get('schema_version')!r}, "
                f"expected {TARGET_CACHE_SCHEMA_VERSION}"
            )
        if self.metadata.get("target_configuration") != expected_configuration:
            raise ValueError(
                "target cache configuration differs from current configuration"
            )
        if self.metadata.get("teacher_identity") != expected_configuration["teacher"]:
            raise ValueError("target cache teacher artifact identity is inconsistent")
        if (
            self.metadata.get("target_generator")
            != expected_configuration["target_generator"]
        ):
            raise ValueError("target cache generator identity is inconsistent")
        if (
            self.metadata.get("target_configuration_sha256")
            != expected_configuration_hash
        ):
            raise ValueError(
                "target cache configuration SHA-256 is missing or inconsistent"
            )

        expected_manifest_hash = manifest_records_sha256(self.records)
        if self.metadata.get("manifest_records_sha256") != expected_manifest_hash:
            raise ValueError("target cache was generated from a different manifest")

        entries = self.metadata.get("clips")
        if not isinstance(entries, list):
            raise ValueError("target cache metadata clips must be a list")
        self.entries_by_id: dict[str, dict] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                raise ValueError("target cache metadata contains an invalid clip entry")
            clip_id = entry["id"]
            if clip_id in self.entries_by_id:
                raise ValueError(f"target cache metadata repeats clip ID {clip_id!r}")
            self.entries_by_id[clip_id] = entry
        if set(self.entries_by_id) != set(self.records_by_id):
            missing = sorted(set(self.records_by_id) - set(self.entries_by_id))
            extra = sorted(set(self.entries_by_id) - set(self.records_by_id))
            raise ValueError(
                f"target cache clip index differs from manifest; missing={missing}, extra={extra}"
            )

        expected_target_files_hash = canonical_json_sha256(
            {
                clip_id: self.entries_by_id[clip_id].get("target_file_sha256")
                for clip_id in sorted(self.entries_by_id)
            }
        )
        if self.metadata.get("target_files_sha256") != expected_target_files_hash:
            raise ValueError("target cache file-set SHA-256 is missing or inconsistent")

        local_entries = [
            entry
            for entry in self.entries_by_id.values()
            if entry.get("target_kind") == "local_windows"
        ]
        availability_checked_frames = sum(
            int(entry.get("availability_checked_frames", -1))
            for entry in local_entries
        )
        availability_contract_valid = all(
            entry.get("availability_contract_valid") is True
            for entry in local_entries
        )
        availability_expectations = {
            "target_expansion": TEACHER_TARGET_EXPANSION,
            "availability_sample_index_semantics": (
                "exclusive_right_edge_unclipped"
            ),
            "availability_checked_frames": availability_checked_frames,
            "availability_contract_valid": availability_contract_valid,
        }
        for key, expected in availability_expectations.items():
            if self.metadata.get(key) != expected:
                raise ValueError(
                    f"target cache availability metadata field {key!r} is "
                    f"{self.metadata.get(key)!r}, expected {expected!r}"
                )
        self.availability_audit = availability_expectations

        self.identity = {
            "schema_version": TARGET_CACHE_SCHEMA_VERSION,
            "target_configuration_sha256": expected_configuration_hash,
            "manifest_records_sha256": expected_manifest_hash,
            "target_files_sha256": expected_target_files_hash,
            "teacher_revision": TEACHER_REVISION,
            "teacher_artifact_sha256": TEACHER_ARTIFACT_SHA256,
            "target_generator_source_sha256": expected_configuration[
                "target_generator"
            ]["source_sha256"],
        }
        self._validated_ids: set[str] = set()

    def load(
        self,
        item: dict,
        array_name: str,
        *,
        expected_frames: int | None = None,
    ) -> np.ndarray:
        """Validate one indexed cache file and return a defensive array copy."""
        clip_id = item["id"]
        if clip_id not in self.records_by_id:
            raise ValueError(f"clip {clip_id!r} is not present in the cache manifest")
        current_record_hash = manifest_record_sha256(item)
        if current_record_hash != manifest_record_sha256(self.records_by_id[clip_id]):
            raise ValueError(f"clip {clip_id!r} differs from its manifest record")

        entry = self.entries_by_id[clip_id]
        if entry.get("manifest_record_sha256") != current_record_hash:
            raise ValueError(f"target cache manifest provenance mismatch for {clip_id}")
        expected_kind = target_kind_for_item(item)
        if entry.get("target_kind") != expected_kind:
            raise ValueError(f"target cache target kind mismatch for {clip_id}")
        expected_expansion = (
            TEACHER_TARGET_EXPANSION
            if expected_kind == "local_windows"
            else "constant_utterance_repeat"
        )
        if entry.get("target_expansion") != expected_expansion:
            raise ValueError(f"target cache target expansion mismatch for {clip_id}")

        audio_path = resolve_audio_path(item, self.manifest_path)
        current_audio_hash = file_sha256(audio_path)
        if entry.get("audio_sha256") != current_audio_hash:
            raise ValueError(f"target cache audio SHA-256 mismatch for {clip_id}")

        target_path = self.targets_dir / f"{clip_id}.npz"
        if not target_path.exists():
            raise FileNotFoundError(f"missing teacher target: {target_path}")
        if entry.get("target_file_sha256") != file_sha256(target_path):
            raise ValueError(f"target cache file SHA-256 mismatch for {clip_id}")

        with np.load(target_path, allow_pickle=False) as target_file:
            cache_schema = int(
                _npz_scalar(target_file, "cache_schema_version", target_path)
            )
            if cache_schema != TARGET_CACHE_SCHEMA_VERSION:
                raise ValueError(f"target cache schema mismatch inside {target_path}")
            string_expectations = {
                "clip_id": clip_id,
                "target_kind": expected_kind,
                "audio_sha256": current_audio_hash,
                "manifest_record_sha256": current_record_hash,
                "target_configuration_sha256": self.identity[
                    "target_configuration_sha256"
                ],
                "teacher_name": TEACHER_NAME,
                "teacher_revision": TEACHER_REVISION,
                "teacher_artifact_sha256": TEACHER_ARTIFACT_SHA256,
                "target_generator_source_sha256": self.identity[
                    "target_generator_source_sha256"
                ],
            }
            for key, expected in string_expectations.items():
                actual = str(_npz_scalar(target_file, key, target_path))
                if actual != expected:
                    raise ValueError(
                        f"target cache {target_path} field {key!r} is {actual!r}, "
                        f"expected {expected!r}"
                    )
            cached_frames = int(_npz_scalar(target_file, "num_frames", target_path))
            if cached_frames != entry.get("frames"):
                raise ValueError(f"target cache frame metadata mismatch for {clip_id}")
            if "anchor_frames" not in target_file:
                raise ValueError(
                    f"target cache {target_path} is missing array 'anchor_frames'"
                )
            anchor_frames = np.asarray(target_file["anchor_frames"])
            anchor_probs = np.asarray(target_file["anchor_probs"])
            if anchor_frames.ndim != 1 or len(anchor_frames) != len(anchor_probs):
                raise ValueError(f"target cache anchor arrays mismatch for {clip_id}")
            if expected_kind == "local_windows":
                availability_audit = _validate_local_target_availability(
                    target_file,
                    path=target_path,
                    num_frames=cached_frames,
                )
            else:
                availability_audit = {
                    "availability_checked_frames": 0,
                    "availability_contract_valid": None,
                    "minimum_availability_margin_samples": None,
                }
            for key, expected in availability_audit.items():
                if entry.get(key) != expected:
                    raise ValueError(
                        f"target cache availability index field {key!r} "
                        f"mismatch for {clip_id}"
                    )

        value = load_teacher_target_array(
            target_path,
            array_name,
            expected_frames=expected_frames,
        )
        if expected_frames is not None and entry.get("frames") != expected_frames:
            raise ValueError(f"target cache index frame count mismatch for {clip_id}")
        self._validated_ids.add(clip_id)
        return value

    def validate_all(self) -> dict[str, Any]:
        """Validate every cache entry and return a serializable audit."""
        for item in self.records:
            waveform = load_audio(resolve_audio_path(item, self.manifest_path))
            self.load(
                item,
                "teacher_soft_targets",
                expected_frames=feature_frame_count(len(waveform)),
            )
        return self.audit()

    def audit(self) -> dict[str, Any]:
        return {
            **self.identity,
            **self.availability_audit,
            "provenance_validated": True,
            "validated_clips": len(self._validated_ids),
            "indexed_clips": len(self.entries_by_id),
        }


def record_speaker_ids(item: dict) -> set[str]:
    """Return the synthesis speaker IDs represented by one manifest record."""
    if "speaker_ids" in item:
        return set(item["speaker_ids"])
    speaker_id = item.get("speaker_id")
    return {speaker_id} if speaker_id else set()


def speaker_split_audit(records: list[dict]) -> dict:
    """Summarise and enforce the train/evaluation speaker boundary.

    Switch evaluation clips inherit their component speakers, so they belong on
    the evaluation side of the audit even though their split is named `switch`.
    """
    train_speakers: set[str] = set()
    heldout_speakers: set[str] = set()
    switch_speakers: set[str] = set()
    for item in records:
        speakers = record_speaker_ids(item)
        if not speakers:
            raise ValueError(f"manifest item {item.get('id', '<unknown>')} has no speaker ID")
        if item["split"] == "train":
            train_speakers.update(speakers)
        elif item["split"] == "heldout":
            heldout_speakers.update(speakers)
        elif item["split"] == "switch":
            switch_speakers.update(speakers)
    evaluation_speakers = heldout_speakers | switch_speakers
    overlap = train_speakers & evaluation_speakers
    return {
        "train_speaker_ids": sorted(train_speakers),
        "heldout_speaker_ids": sorted(heldout_speakers),
        "switch_eval_speaker_ids": sorted(switch_speakers),
        "train_evaluation_speaker_overlap": sorted(overlap),
        "speaker_disjoint": not overlap,
    }


def require_speaker_disjoint(records: list[dict]) -> dict:
    """Raise on train/evaluation speaker leakage and return the split audit."""
    audit = speaker_split_audit(records)
    if not audit["speaker_disjoint"]:
        raise ValueError(
            "train/evaluation speaker leakage: "
            + ", ".join(audit["train_evaluation_speaker_overlap"])
        )
    return audit


class DistillationDataset(Dataset):
    """Tiny-data dataset: preload deterministic features and teacher posteriors."""

    def __init__(
        self,
        manifest_path: str | Path,
        targets_dir: str | Path,
        splits: Iterable[str],
        *,
        target_cache: TeacherTargetCache | None = None,
        records: Iterable[dict] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        wanted = set(splits)
        source_records = (
            read_manifest(manifest_path)
            if records is None
            else copy.deepcopy(list(records))
        )
        self.items = [
            item for item in source_records if item["split"] in wanted
        ]
        if not self.items:
            raise ValueError(f"no manifest items for splits {sorted(wanted)}")
        frontend = LogMelFrontend().eval()
        self.examples: list[dict] = []
        with torch.inference_mode():
            for item in self.items:
                waveform = load_audio(resolve_audio_path(item, self.manifest_path))
                features = frontend(waveform).squeeze(0)
                target_path = Path(targets_dir) / f"{item['id']}.npz"
                if not target_path.exists():
                    raise FileNotFoundError(f"missing teacher target: {target_path}")
                if target_cache is None:
                    target_array = load_teacher_target_array(
                        target_path,
                        "teacher_soft_targets",
                        expected_frames=len(features),
                    )
                else:
                    target_array = target_cache.load(
                        item,
                        "teacher_soft_targets",
                        expected_frames=len(features),
                    )
                target = torch.from_numpy(target_array).float()
                if len(features) != len(target):
                    raise ValueError(
                        f"frame mismatch for {item['id']}: features={len(features)}, targets={len(target)}"
                    )
                self.examples.append(
                    {"features": features, "targets": target, "item": item}
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def collate_distillation_batch(examples: list[dict]) -> dict:
    lengths = torch.tensor(
        [len(example["features"]) for example in examples], dtype=torch.long
    )
    return {
        "features": pad_sequence(
            [example["features"] for example in examples], batch_first=True
        ),
        "targets": pad_sequence(
            [example["targets"] for example in examples], batch_first=True
        ),
        "lengths": lengths,
        "ids": [example["item"]["id"] for example in examples],
    }
