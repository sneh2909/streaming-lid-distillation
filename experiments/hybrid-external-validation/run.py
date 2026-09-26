#!/usr/bin/env python3
"""Externally validate the frozen step-1520 hybrid KD+CE candidate.

This experiment is deliberately downstream of ``loss-factorial``.  It
regenerates only the already-selected step-1520 hybrid and its same-step
frame-KD control from the exact historical pipeline source snapshot, verifies
their states/losses/gradients against the parent result, and then evaluates
them on the already-pinned FLEURS validation cohort.  The current step-1600
main checkpoint is a second baseline.  FLEURS test is never read.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import stat
import sys
import tarfile
import time
import uuid
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT_SHA256 = "9bf2ae32a60ba6608879c0d91354ae43687df838fd88753483cebc42e041202d"
STAGE_ROOT = REPO_ROOT / "data/generated/source_snapshots" / STAGE_ROOT_SHA256
STAGE_MANIFEST = STAGE_ROOT / ".source_snapshot.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_historical_stage() -> dict[str, Any]:
    if not STAGE_MANIFEST.is_file():
        raise FileNotFoundError(STAGE_MANIFEST)
    manifest = json.loads(STAGE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("historical source stage schema differs")
    if manifest.get("snapshot_root_sha256") != STAGE_ROOT_SHA256:
        raise ValueError("historical source stage root differs")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("historical source stage has no file manifest")
    actual_files: set[str] = set()
    for path in STAGE_ROOT.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"historical source stage contains symlink {path}")
        if path.is_file() and path != STAGE_MANIFEST:
            actual_files.add(path.relative_to(STAGE_ROOT).as_posix())
    if actual_files != set(expected_files):
        raise ValueError("historical source stage file set differs")
    for relative, expected in expected_files.items():
        path = STAGE_ROOT / relative
        actual = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
        if actual != expected:
            raise ValueError(f"historical source stage differs at {relative}")
    return manifest


# Validate and import the exact pipeline generation used by the parent loss
# experiment.  This avoids consuming concurrent live schema edits while still
# reusing the pipeline implementation rather than copying it.
sys.dont_write_bytecode = True
_STAGE_AT_IMPORT = validate_historical_stage()
for _path in (STAGE_ROOT, STAGE_ROOT / "src"):
    _text = str(_path)
    if _text in sys.path:
        sys.path.remove(_text)
    sys.path.insert(0, _text)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from torch.nn.utils.rnn import pad_sequence  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scripts import eval as staged_eval  # noqa: E402
from scripts import train as staged_train  # noqa: E402
from scripts.train import (  # noqa: E402
    assert_finite_post_update_state,
    seed_everything,
    validate_full_batch_training_set,
)
from streaming_lid import data as staged_data  # noqa: E402
from streaming_lid.audio import (  # noqa: E402
    LogMelFrontend,
    feature_frame_count,
    load_audio,
)
from streaming_lid.config import (  # noqa: E402
    CHUNK_FRAMES,
    LABEL_DELAY_FRAMES,
    LANGUAGE_CODES,
    MODEL_LOOKAHEAD_FRAMES,
    SAMPLE_RATE,
)
from streaming_lid.data import (  # noqa: E402
    DistillationDataset,
    capture_manifest_snapshot,
    collate_distillation_batch,
    require_speaker_disjoint,
    resolve_audio_path,
)
from streaming_lid.model import CausalLIDStudent  # noqa: E402
from streaming_lid.run_identity import (  # noqa: E402
    configured_model_kwargs,
    model_state_sha256,
)


LOSS_DRIVER = REPO_ROOT / "experiments/loss-factorial/run.py"
_loss_spec = importlib.util.spec_from_file_location(
    "frozen_loss_factorial_helpers", LOSS_DRIVER
)
if _loss_spec is None or _loss_spec.loader is None:
    raise ImportError(f"cannot load loss helpers from {LOSS_DRIVER}")
loss_helpers = importlib.util.module_from_spec(_loss_spec)
sys.modules[_loss_spec.name] = loss_helpers
_loss_spec.loader.exec_module(loss_helpers)


EXPERIMENT_NAME = "hybrid-external-validation"
SCHEMA_VERSION = 1
ANALYSIS_VERSION = "hybrid-external-validation-v1"
SELECTED_STEP = 1_520
PARENT_TOTAL_STEPS = 1_600
BATCH_SIZE = 7
LEARNING_RATE = 1e-3
SEED = 7
THREADS = 6
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 5.0
ARMS = OrderedDict((arm, loss_helpers.ARMS[arm]) for arm in ("frame_kd", "hybrid_kd_ce"))

PREFIX_SECONDS: tuple[float | str, ...] = (0.5, 1.0, 2.0, 4.0, "full")
SELECTION_PREFIX_KEYS = ("1", "2", "full")
STUDENT_BATCH_SIZE = 24
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 7
MATERIAL_GAIN_GATE_PP = 2.0
LOCAL_ACCURACY_FLOOR = 0.80

FLEURS_DATASET_ID = "google/fleurs"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_SPLIT = "validation"
FLEURS_LICENSE = "cc-by-4.0"
FLEURS_CARD = {
    "repo_path": "README.md",
    "sha256": "688f79f2a5c731af3796e9f683eb02f9b3f09d040decd8c5625d0f37098e71c6",
    "bytes": 385_614,
}
FLEURS_CONFIG_BY_LANGUAGE = OrderedDict(
    (
        ("en", "en_us"),
        ("hi", "hi_in"),
        ("mr", "mr_in"),
        ("bn", "bn_in"),
        ("ta", "ta_in"),
        ("te", "te_in"),
        ("gu", "gu_in"),
    )
)
EXPECTED_VALIDATION_ROWS = {
    "en_us": 394,
    "hi_in": 239,
    "mr_in": 443,
    "bn_in": 402,
    "ta_in": 377,
    "te_in": 311,
    "gu_in": 432,
}
SAMPLE_PER_LANGUAGE = 100
SELECTION_ALGORITHM = "lowest_sha256(revision\\0config\\0row_idx\\0id\\0basename)"
EXPECTED_POOL_METADATA_SHA256 = "2c825a340a0e9dbbfa9d10f0df658afb68e14a389d3148704d1b5950649cf675"
EXPECTED_EXTERNAL_MANIFEST_SHA256 = "ef0ec36427f742074b1bc88bd42f1b7e5b93b43bb390f9382f5030f3d8d2a637"

FLEURS_SOURCE_FILES = {
    "en_us": {
        "tsv": {
            "repo_path": "data/en_us/dev.tsv",
            "sha256": "9d57ee7e91e9d4c92edb39f6bbea668ef8dc2a3ff96eb510d5580b2ad05d17ec",
            "bytes": 213_065,
        },
        "archive": {
            "repo_path": "data/en_us/audio/dev.tar.gz",
            "sha256": "2658fda72f199e12676ecac9415094667a4e14e149b146e568ea00b2a2f0954c",
            "bytes": 171_250_900,
        },
    },
    "hi_in": {
        "tsv": {
            "repo_path": "data/hi_in/dev.tsv",
            "sha256": "cea87c57a37a0d38ed0afce30e68a35ad7b3945414648430fee09a54ca7b72ba",
            "bytes": 249_853,
        },
        "archive": {
            "repo_path": "data/hi_in/audio/dev.tar.gz",
            "sha256": "9adbca6d6fc70e40c121910941bcd7c8906eee60b402b6d21b4bd160e20030c7",
            "bytes": 131_741_732,
        },
    },
    "mr_in": {
        "tsv": {
            "repo_path": "data/mr_in/dev.tsv",
            "sha256": "aa8219008c8a14584acb7b7e3cd20c1c32b55c3fb96f45d6e901e8eacbe43c9c",
            "bytes": 513_623,
        },
        "archive": {
            "repo_path": "data/mr_in/audio/dev.tar.gz",
            "sha256": "a8cdc2253a0de4b1d43170e836313a2139c2451946a49344b43cef1a200fc6c1",
            "bytes": 292_468_044,
        },
    },
    "bn_in": {
        "tsv": {
            "repo_path": "data/bn_in/dev.tsv",
            "sha256": "239a59e4d60a76d4d388d44f72df6d28b91e4526d829186d15ef33fde7a089db",
            "bytes": 465_963,
        },
        "archive": {
            "repo_path": "data/bn_in/audio/dev.tar.gz",
            "sha256": "b7f380b67d50bc59f328af3e1bd8096ab9e728bd0103349b9746ca95ab75ff30",
            "bytes": 278_667_778,
        },
    },
    "ta_in": {
        "tsv": {
            "repo_path": "data/ta_in/dev.tsv",
            "sha256": "0a5ad9f10d3284f48c268e947efba178ec41c368dce191f652a09a6697a54a8b",
            "bytes": 507_783,
        },
        "archive": {
            "repo_path": "data/ta_in/audio/dev.tar.gz",
            "sha256": "63faa53c804949dae73c9a953927fb50d59540830261cff374588ba4c901f09e",
            "bytes": 238_893_728,
        },
    },
    "te_in": {
        "tsv": {
            "repo_path": "data/te_in/dev.tsv",
            "sha256": "fbedc2a5b7e394ff5a6e8c269f616017d2ac9a247ffa707c10efaa08132a0630",
            "bytes": 350_269,
        },
        "archive": {
            "repo_path": "data/te_in/audio/dev.tar.gz",
            "sha256": "2ac52fa35be22b05ad041e2a2a443d10f4ef59449db07fe42cd644eb81ed7123",
            "bytes": 166_827_973,
        },
    },
    "gu_in": {
        "tsv": {
            "repo_path": "data/gu_in/dev.tsv",
            "sha256": "91711c64667a6cac092ccf2eac2e48b0945ad49d10712ea41852d00f96d8d724",
            "bytes": 474_781,
        },
        "archive": {
            "repo_path": "data/gu_in/audio/dev.tar.gz",
            "sha256": "67b61fe0b78c80585ffc3743eba2528e18d756819495abd3c603e0824c0b78bc",
            "bytes": 225_657_147,
        },
    },
}

LIVE_SOURCE_FILES = (
    "experiments/hybrid-external-validation/run.py",
    "experiments/loss-factorial/run.py",
    "experiments/checkpoint-trajectory/run.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/generated/manifest.jsonl")
    )
    parser.add_argument(
        "--targets-dir", type=Path, default=Path("data/generated/targets")
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=Path("checkpoints/student.pt"),
    )
    parser.add_argument(
        "--reference-summary", type=Path, default=Path("results/summary.json")
    )
    parser.add_argument(
        "--loss-results",
        type=Path,
        default=Path("experiments/loss-factorial/results.json"),
    )
    parser.add_argument(
        "--prior-external-results",
        type=Path,
        default=Path("experiments/checkpoint-external-validation/results.json"),
    )
    parser.add_argument(
        "--external-data-dir",
        type=Path,
        default=Path(
            "data/experiments/checkpoint-external-validation/fleurs-validation"
        ),
    )
    parser.add_argument(
        "--hf-snapshot",
        type=Path,
        default=Path(
            ".cache/hf/hub/datasets--google--fleurs/snapshots/"
            + FLEURS_REVISION
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/hybrid-external-validation/results.json"),
    )
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Atomically replace an existing result after all validations pass.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.threads != THREADS:
        raise ValueError(f"threads must be {THREADS}")
    if args.output.exists() and not args.fresh:
        raise FileExistsError(f"{args.output} exists; pass --fresh to replace it")
    for path in (
        args.manifest,
        args.targets_dir / "metadata.json",
        args.reference_checkpoint,
        args.reference_summary,
        args.loss_results,
        args.prior_external_results,
        args.external_data_dir / "manifest.json",
        args.hf_snapshot / FLEURS_CARD["repo_path"],
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def atomic_json_write(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def live_source_snapshot() -> dict[str, dict[str, Any]]:
    return {
        relative: {
            "sha256": sha256_file(REPO_ROOT / relative),
            "bytes": (REPO_ROOT / relative).stat().st_size,
        }
        for relative in LIVE_SOURCE_FILES
    }


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("numpy", "torch", "torchaudio", "soundfile"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def local_module_origins() -> dict[str, str]:
    modules = {
        "scripts.train": staged_train,
        "scripts.eval": staged_eval,
        "streaming_lid.audio": sys.modules["streaming_lid.audio"],
        "streaming_lid.config": sys.modules["streaming_lid.config"],
        "streaming_lid.data": sys.modules["streaming_lid.data"],
        "streaming_lid.loss": sys.modules["streaming_lid.loss"],
        "streaming_lid.model": sys.modules["streaming_lid.model"],
        "streaming_lid.run_identity": sys.modules["streaming_lid.run_identity"],
    }
    origins: dict[str, str] = {}
    stage = STAGE_ROOT.resolve()
    for name, module in modules.items():
        origin = Path(str(module.__file__)).resolve()
        if stage not in origin.parents:
            raise RuntimeError(f"{name} escaped the historical source stage: {origin}")
        origins[name] = origin.relative_to(stage).as_posix()
    return origins


def trace_sha256(values: Sequence[float]) -> str:
    return canonical_sha256([float(value) for value in values])


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


class MigratedTargetCache:
    """Strict read-only adapter for schema-6 files with schema-5 tensors.

    The builder added native-teacher audit fields after the parent experiment.
    Those fields change the cache identity but not the four training target
    arrays.  This adapter validates the new index/file provenance and the old
    dense-target/availability contracts, then exact state/loss/gradient parity
    proves that the training-consumed tensors are unchanged.
    """

    def __init__(self, manifest_snapshot: Any, targets_dir: Path) -> None:
        self.manifest_snapshot = manifest_snapshot
        self.manifest_path = manifest_snapshot.source_path
        self.targets_dir = Path(targets_dir)
        self.records = manifest_snapshot.records_copy()
        self.records_by_id = {item["id"]: item for item in self.records}
        self.metadata_path = self.targets_dir / "metadata.json"
        self.metadata_sha256 = sha256_file(self.metadata_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != 6:
            raise ValueError("migrated target adapter requires schema 6")
        if self.metadata.get("manifest_snapshot") != manifest_snapshot.identity():
            raise ValueError("schema-6 targets bind a different manifest snapshot")
        if self.metadata.get("languages") != list(LANGUAGE_CODES):
            raise ValueError("schema-6 target language order differs")
        entries = self.metadata.get("clips")
        if not isinstance(entries, list):
            raise ValueError("schema-6 target metadata lacks clip index")
        self.entries_by_id: dict[str, dict[str, Any]] = {}
        for entry in entries:
            clip_id = entry.get("id") if isinstance(entry, dict) else None
            if not isinstance(clip_id, str) or clip_id in self.entries_by_id:
                raise ValueError("schema-6 target metadata has invalid/duplicate clip")
            self.entries_by_id[clip_id] = entry
        if set(self.entries_by_id) != set(self.records_by_id):
            raise ValueError("schema-6 target index differs from manifest")
        expected_files_hash = canonical_sha256(
            {
                clip_id: self.entries_by_id[clip_id].get("target_file_sha256")
                for clip_id in sorted(self.entries_by_id)
            }
        )
        if self.metadata.get("target_files_sha256") != expected_files_hash:
            raise ValueError("schema-6 target file-set hash differs")
        teacher = self.metadata.get("teacher_identity", {})
        self.identity = {
            "schema_version": 6,
            "metadata_sha256": self.metadata_sha256,
            "target_configuration_sha256": self.metadata.get(
                "target_configuration_sha256"
            ),
            "manifest_snapshot": self.metadata.get("manifest_snapshot"),
            "manifest_file_sha256": self.metadata.get("manifest_file_sha256"),
            "manifest_records_sha256": self.metadata.get(
                "manifest_records_sha256"
            ),
            "target_files_sha256": expected_files_hash,
            "teacher_revision": teacher.get("revision"),
            "teacher_artifact_sha256": teacher.get("artifact_sha256"),
            "target_generator_source_sha256": self.metadata.get(
                "target_generator", {}
            ).get("source_sha256"),
            "migration_relation": (
                "schema 6 adds native-teacher audit arrays; training-consumed "
                "dense tensors are parity-checked by exact trajectory reproduction"
            ),
        }
        self._validated_ids: set[str] = set()

    @staticmethod
    def _scalar(target_file: Any, key: str, path: Path) -> Any:
        if key not in target_file:
            raise ValueError(f"target cache {path} is missing scalar {key}")
        value = np.asarray(target_file[key])
        if value.size != 1:
            raise ValueError(f"target cache {path} scalar {key} is not scalar")
        return value.reshape(()).item()

    def load(
        self,
        item: dict[str, Any],
        array_name: str,
        *,
        expected_frames: int | None = None,
    ) -> np.ndarray:
        clip_id = item["id"]
        if clip_id not in self.records_by_id:
            raise ValueError(f"clip {clip_id!r} is absent from schema-6 cache")
        record_hash = staged_data.manifest_record_sha256(item)
        if record_hash != staged_data.manifest_record_sha256(
            self.records_by_id[clip_id]
        ):
            raise ValueError(f"clip {clip_id!r} differs from captured manifest")
        entry = self.entries_by_id[clip_id]
        expected_kind = staged_data.target_kind_for_item(item)
        if entry.get("manifest_record_sha256") != record_hash:
            raise ValueError(f"schema-6 manifest lineage differs for {clip_id}")
        if entry.get("target_kind") != expected_kind:
            raise ValueError(f"schema-6 target kind differs for {clip_id}")
        expected_expansion = (
            "previous_anchor_hold"
            if expected_kind == "local_windows"
            else "constant_utterance_repeat"
        )
        if entry.get("target_expansion") != expected_expansion:
            raise ValueError(f"schema-6 target expansion differs for {clip_id}")
        audio_path = resolve_audio_path(item, self.manifest_snapshot)
        audio_hash = sha256_file(audio_path)
        if entry.get("audio_sha256") != audio_hash or item.get(
            "audio_sha256"
        ) != audio_hash:
            raise ValueError(f"schema-6 audio identity differs for {clip_id}")
        target_path = self.targets_dir / f"{clip_id}.npz"
        if entry.get("target_file_sha256") != sha256_file(target_path):
            raise ValueError(f"schema-6 target file hash differs for {clip_id}")

        with np.load(target_path, allow_pickle=False) as target_file:
            if int(self._scalar(target_file, "cache_schema_version", target_path)) != 6:
                raise ValueError(f"schema differs inside {target_path}")
            string_expectations = {
                "clip_id": clip_id,
                "target_kind": expected_kind,
                "audio_sha256": audio_hash,
                "manifest_record_sha256": record_hash,
                "manifest_file_sha256": self.identity["manifest_file_sha256"],
                "target_configuration_sha256": self.identity[
                    "target_configuration_sha256"
                ],
                "teacher_revision": self.identity["teacher_revision"],
                "teacher_artifact_sha256": self.identity[
                    "teacher_artifact_sha256"
                ],
                "target_generator_source_sha256": self.identity[
                    "target_generator_source_sha256"
                ],
            }
            for key, expected in string_expectations.items():
                actual = str(self._scalar(target_file, key, target_path))
                if actual != expected:
                    raise ValueError(
                        f"schema-6 target {clip_id} field {key} differs"
                    )
            frames = int(self._scalar(target_file, "num_frames", target_path))
            if frames != entry.get("frames"):
                raise ValueError(f"schema-6 frame index differs for {clip_id}")
            dense = staged_data._validate_dense_target_expansion(
                target_file,
                path=target_path,
                num_frames=frames,
                target_kind=expected_kind,
            )
            availability = (
                staged_data._validate_local_target_availability(
                    target_file, path=target_path, num_frames=frames
                )
                if expected_kind == "local_windows"
                else {
                    "availability_checked_frames": 0,
                    "availability_contract_valid": None,
                    "minimum_availability_margin_samples": None,
                }
            )
            for key, expected in {**dense, **availability}.items():
                if entry.get(key) != expected:
                    raise ValueError(
                        f"schema-6 target {clip_id} audit field {key} differs"
                    )
            for key in (
                "in_set_mass",
                "full_top1_indices",
                "full_top1_probabilities",
                "full_top1_labels",
            ):
                if key not in target_file:
                    raise ValueError(f"schema-6 target {clip_id} lacks {key}")
        value = staged_data.load_teacher_target_array(
            target_path, array_name, expected_frames=expected_frames
        )
        if expected_frames is not None and entry.get("frames") != expected_frames:
            raise ValueError(f"schema-6 indexed frames differ for {clip_id}")
        self._validated_ids.add(clip_id)
        return value

    def validate_all(self) -> dict[str, Any]:
        if sha256_file(self.metadata_path) != self.metadata_sha256:
            raise RuntimeError("schema-6 target metadata changed during the run")
        if sha256_file(self.manifest_path) != self.manifest_snapshot.identity()[
            "manifest_file_sha256"
        ]:
            raise RuntimeError("manifest pathname changed during the run")
        for item in self.records:
            waveform = load_audio(resolve_audio_path(item, self.manifest_snapshot))
            self.load(
                item,
                "teacher_soft_targets",
                expected_frames=feature_frame_count(len(waveform)),
            )
        return self.audit()

    def audit(self) -> dict[str, Any]:
        return {
            **self.identity,
            "provenance_validated": True,
            "validated_clips": len(self._validated_ids),
            "indexed_clips": len(self.entries_by_id),
        }


def train_frozen_pair(
    args: argparse.Namespace,
    parent: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, torch.Tensor]],
    dict[str, Any],
    Any,
    MigratedTargetCache,
    dict[str, bool | None],
    DistillationDataset,
]:
    seed_everything(SEED)
    manifest_snapshot = capture_manifest_snapshot(args.manifest)
    records = manifest_snapshot.records_copy()
    speaker_audit = require_speaker_disjoint(records)
    target_cache = MigratedTargetCache(manifest_snapshot, args.targets_dir)
    target_cache.validate_all()

    train_dataset = DistillationDataset(
        manifest_snapshot,
        args.targets_dir,
        splits=("train",),
        target_cache=target_cache,
    )
    validate_full_batch_training_set(len(train_dataset), BATCH_SIZE)
    items_by_id = {
        example["item"]["id"]: example["item"] for example in train_dataset.examples
    }
    target_audit, teacher_correct_by_id = loss_helpers.teacher_target_audit(
        train_dataset, target_cache
    )

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_distillation_batch,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
        drop_last=True,
    )
    base_model = CausalLIDStudent(**configured_model_kwargs())
    initial_hash = model_state_sha256(base_model.state_dict())
    expected_initial = parent["training"]["initial_model_state_sha256"]
    if initial_hash != expected_initial:
        raise RuntimeError(f"initial model differs: {initial_hash} != {expected_initial}")
    models = OrderedDict(
        (
            ("frame_kd", base_model),
            ("hybrid_kd_ce", copy.deepcopy(base_model)),
        )
    )
    optimizers = {
        arm: torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        for arm, model in models.items()
    }
    losses: dict[str, list[float]] = {arm: [] for arm in ARMS}
    gradients: dict[str, list[float]] = {arm: [] for arm in ARMS}
    examples_seen = {arm: 0 for arm in ARMS}
    batch_order_digest = hashlib.sha256()
    for model in models.values():
        model.train()

    completed = 0
    started = time.perf_counter()
    print(
        f"regenerating frozen {SELECTED_STEP}-step frame/hybrid pair on six threads",
        flush=True,
    )
    while completed < SELECTED_STEP:
        for raw_batch in loader:
            step = completed + 1
            batch = loss_helpers.batch_with_items(raw_batch, items_by_id)
            batch_order_digest.update(
                json.dumps(batch["ids"], separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
            values: dict[str, float] = {}
            for arm, model in models.items():
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch["features"])
                loss, _ = loss_helpers.factorial_loss(
                    arm, logits, batch, teacher_correct_by_id
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite {arm} loss at step {step}")
                loss.backward()
                if not all(
                    parameter.grad is None
                    or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError(f"non-finite {arm} gradient at step {step}")
                gradient = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRADIENT_CLIP_NORM,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                assert_finite_post_update_state(model, optimizer, step=step)
                value = float(loss.detach())
                losses[arm].append(value)
                gradients[arm].append(float(gradient))
                examples_seen[arm] += len(batch["ids"])
                values[arm] = value
            completed += 1
            if completed == 1 or completed % 200 == 0:
                print(
                    f"step={completed:04d} "
                    + " ".join(f"{arm}={values[arm]:.5f}" for arm in ARMS),
                    flush=True,
                )
            if completed >= SELECTED_STEP:
                break

    states = {arm: clone_state(model) for arm, model in models.items()}
    state_hashes = {arm: model_state_sha256(state) for arm, state in states.items()}
    exact: dict[str, Any] = {}
    for arm in ARMS:
        expected_losses = parent["training"]["arms"][arm]["losses"][:SELECTED_STEP]
        expected_gradients = parent["training"]["arms"][arm]["gradient_norms"][
            :SELECTED_STEP
        ]
        expected_hash = parent["experiment_identity"][
            "model_state_sha256_by_arm_step"
        ][arm][str(SELECTED_STEP)]
        exact[arm] = {
            "model_state_sha256": state_hashes[arm],
            "expected_model_state_sha256": expected_hash,
            "model_state_exact": state_hashes[arm] == expected_hash,
            "loss_trace_exact": losses[arm] == expected_losses,
            "gradient_trace_exact": gradients[arm] == expected_gradients,
            "loss_trace_sha256": trace_sha256(losses[arm]),
            "gradient_trace_sha256": trace_sha256(gradients[arm]),
            "first_10_mean_loss": float(np.mean(losses[arm][:10])),
            "last_10_mean_loss": float(np.mean(losses[arm][-10:])),
            "final_loss": losses[arm][-1],
            "final_preclip_gradient_norm": gradients[arm][-1],
            "examples_seen": examples_seen[arm],
        }
        if not all(
            exact[arm][key]
            for key in ("model_state_exact", "loss_trace_exact", "gradient_trace_exact")
        ):
            raise RuntimeError(f"{arm} did not exactly reproduce its frozen trajectory")

    training_audit = {
        "selected_step": SELECTED_STEP,
        "parent_total_steps": PARENT_TOTAL_STEPS,
        "initial_model_state_sha256": initial_hash,
        "batch_order_prefix_sha256": batch_order_digest.hexdigest(),
        "wall_seconds": time.perf_counter() - started,
        "arms": exact,
        "target_audit_sha256": canonical_sha256(target_audit),
        "parent_dependency_snapshot_sha256": canonical_sha256(
            parent["experiment_identity"]["dependency_snapshot"]
        ),
        "migrated_target_cache": target_cache.audit(),
        "speaker_audit": speaker_audit,
    }
    return (
        states,
        training_audit,
        manifest_snapshot,
        target_cache,
        teacher_correct_by_id,
        train_dataset,
    )


def evaluate_same_synthetic_clips(
    args: argparse.Namespace,
    parent: Mapping[str, Any],
    states: Mapping[str, Mapping[str, torch.Tensor]],
    manifest_snapshot: Any,
    target_cache: MigratedTargetCache,
    teacher_correct_by_id: Mapping[str, bool | None],
    train_dataset: DistillationDataset,
) -> dict[str, Any]:
    heldout_dataset = DistillationDataset(
        manifest_snapshot,
        args.targets_dir,
        splits=("heldout",),
        target_cache=target_cache,
    )
    switch_dataset = DistillationDataset(
        manifest_snapshot,
        args.targets_dir,
        splits=("switch",),
        target_cache=target_cache,
    )
    heldout_batch = loss_helpers.collate_examples(heldout_dataset.examples)
    switch_batch = loss_helpers.collate_examples(switch_dataset.examples)
    monolingual_train = [
        example
        for example in train_dataset.examples
        if loss_helpers.is_monolingual(example["item"])
    ]
    train_batch = loss_helpers.collate_examples(monolingual_train)
    exact_indices = [
        index
        for index, item in enumerate(train_batch["items"])
        if int(str(item["id"]).rsplit("_", 1)[1])
        in loss_helpers.EXACT_EDGE_TRAIN_INDICES
    ]
    frontend = LogMelFrontend().eval()
    prefix_batches: dict[str, Mapping[str, Any]] = {"full": heldout_batch}
    for seconds in loss_helpers.PREFIX_SECONDS:
        prefix_batches[loss_helpers.trajectory.prefix_key(seconds)] = (
            loss_helpers.fixed_prefix_features(
                heldout_batch["items"], manifest_snapshot, seconds, frontend
            )
        )
    heldout_teacher = {
        item["id"]: torch.from_numpy(
            target_cache.load(
                item,
                "teacher_probs",
                expected_frames=int(heldout_batch["lengths"][index]),
            )
        )
        for index, item in enumerate(heldout_batch["items"])
    }
    train_teacher = {
        item["id"]: torch.from_numpy(
            target_cache.load(
                item,
                "teacher_probs",
                expected_frames=int(train_batch["lengths"][index]),
            )
        )
        for index, item in enumerate(train_batch["items"])
    }
    model = CausalLIDStudent(**configured_model_kwargs())
    rows: dict[str, Any] = {}
    for arm in ARMS:
        row, pending = loss_helpers.evaluate_state(
            model,
            states[arm],
            arm=arm,
            step=SELECTED_STEP,
            state_sha256=model_state_sha256(states[arm]),
            loss_history=parent["training"]["arms"][arm]["losses"][:SELECTED_STEP],
            heldout_batch=heldout_batch,
            prefix_batches=prefix_batches,
            train_batch=train_batch,
            exact_indices=exact_indices,
            switch_batch=switch_batch,
            heldout_teacher_probabilities=heldout_teacher,
            train_teacher_probabilities=train_teacher,
            teacher_correct_by_id=teacher_correct_by_id,
            check_streaming_equivalence=True,
        )
        if pending:
            raise AssertionError(f"streaming equivalence was not checked for {arm}")
        expected = next(
            candidate
            for candidate in parent["curves"][arm]
            if candidate["step"] == SELECTED_STEP
        )
        exact = canonical_sha256(row) == canonical_sha256(expected)
        if not exact:
            raise RuntimeError(f"{arm} synthetic evaluation differs from parent result")
        rows[arm] = {
            "parent_row_exact": True,
            "parent_row_sha256": canonical_sha256(expected),
            "heldout_clip_ids": [item["id"] for item in heldout_batch["items"]],
            "switch_clip_ids": [item["id"] for item in switch_batch["items"]],
            "label_composite_1s_2s_full": row["synthetic_dev"][
                "label_composite_1s_2s_full"
            ],
            "minimum_language_recall_composite_1s_2s_full": row[
                "synthetic_dev"
            ]["minimum_language_recall_composite_1s_2s_full"],
            "per_language_recall_composite_1s_2s_full": row["synthetic_dev"][
                "per_language_recall_composite_1s_2s_full"
            ],
            "full_frame": row["synthetic_dev"]["full_frame"],
            "monolingual_stability": row["synthetic_dev"]["monolingual_stability"],
            "switch": row["switch"],
            "all_train": row["familiar_audio"][
                "all_70_monolingual_train_full"
            ]["classification"],
            "exact_edge": row["familiar_audio"]["edge_indices_5_6_7_full"][
                "classification"
            ],
        }
    return rows


def parse_fleurs_rows(config: str, path: Path) -> list[dict[str, Any]]:
    gender_to_index = {"MALE": 0, "FEMALE": 1, "OTHER": 2}
    rows: list[dict[str, Any]] = []
    for row_idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(f"unexpected {config} TSV field count at row {row_idx}")
        basename = fields[1]
        if not basename.endswith(".wav") or Path(basename).name != basename:
            raise ValueError(f"unexpected FLEURS basename {basename}")
        if fields[6] not in gender_to_index:
            raise ValueError(f"unexpected FLEURS gender {fields[6]}")
        rows.append(
            {
                "config": config,
                "row_idx": row_idx,
                "dataset_row_id": int(fields[0]),
                "num_samples": int(fields[5]),
                "source_basename": basename,
                "gender_index": gender_to_index[fields[6]],
                "language_name": config,
            }
        )
    if len(rows) != EXPECTED_VALIDATION_ROWS[config]:
        raise ValueError(f"unexpected {config} validation row count {len(rows)}")
    return rows


def selection_rank(row: Mapping[str, Any]) -> str:
    payload = "\0".join(
        (
            FLEURS_REVISION,
            str(row["config"]),
            str(row["row_idx"]),
            str(row["dataset_row_id"]),
            str(row["source_basename"]),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_locked_file(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    wanted = {"sha256": expected["sha256"], "bytes": expected["bytes"]}
    if actual != wanted:
        raise ValueError(f"pinned FLEURS source differs at {path}: {actual} != {wanted}")


def expected_record(
    row: Mapping[str, Any], language: str, data_dir: Path
) -> dict[str, Any]:
    relative = Path("audio") / str(row["config"]) / (
        f"{int(row['row_idx']):04d}-{int(row['dataset_row_id'])}.wav"
    )
    path = data_dir / relative
    info = sf.info(str(path))
    if (
        info.samplerate != SAMPLE_RATE
        or info.channels != 1
        or info.frames != int(row["num_samples"])
    ):
        raise ValueError(f"materialized FLEURS audio metadata differs for {path}")
    return {
        "clip_id": (
            f"fleurs-{row['config']}-validation-{int(row['row_idx']):04d}-"
            f"{int(row['dataset_row_id'])}"
        ),
        "language": language,
        "config": row["config"],
        "row_idx": int(row["row_idx"]),
        "dataset_row_id": int(row["dataset_row_id"]),
        "source_basename": row["source_basename"],
        "gender_index": int(row["gender_index"]),
        "language_name": row["language_name"],
        "num_samples": int(row["num_samples"]),
        "duration_seconds": int(row["num_samples"]) / SAMPLE_RATE,
        "audio_path": relative.as_posix(),
        "audio_sha256": sha256_file(path),
        "audio_bytes": path.stat().st_size,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "subtype": info.subtype,
    }


def validate_external_release(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = args.external_data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != EXPECTED_EXTERNAL_MANIFEST_SHA256:
        raise ValueError("external manifest identity differs from the frozen cohort")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_sha256(payload) != EXPECTED_EXTERNAL_MANIFEST_SHA256:
        raise ValueError("external manifest self-hash differs")

    card_path = args.hf_snapshot / FLEURS_CARD["repo_path"]
    validate_locked_file(card_path, FLEURS_CARD)
    card_text = card_path.read_text(encoding="utf-8")
    if "- cc-by-4.0" not in card_text:
        raise ValueError("pinned FLEURS card no longer states cc-by-4.0")

    source_audit: dict[str, Any] = {}
    pool: list[dict[str, Any]] = []
    derived: list[dict[str, Any]] = []
    manifest_by_key = {
        (
            record["config"],
            int(record["row_idx"]),
            int(record["dataset_row_id"]),
            record["source_basename"],
        ): record
        for record in manifest["records"]
    }
    if len(manifest_by_key) != len(manifest["records"]):
        raise ValueError("external manifest has duplicate composite source keys")

    for language, config in FLEURS_CONFIG_BY_LANGUAGE.items():
        lock = FLEURS_SOURCE_FILES[config]
        tsv_path = args.hf_snapshot / lock["tsv"]["repo_path"]
        archive_path = args.hf_snapshot / lock["archive"]["repo_path"]
        print(f"validating pinned FLEURS source {config}", flush=True)
        validate_locked_file(tsv_path, lock["tsv"])
        validate_locked_file(archive_path, lock["archive"])
        rows = parse_fleurs_rows(config, tsv_path)
        pool.extend(rows)
        selected = sorted(rows, key=selection_rank)[:SAMPLE_PER_LANGUAGE]
        selected_by_basename = {row["source_basename"]: row for row in selected}
        if len(selected_by_basename) != SAMPLE_PER_LANGUAGE:
            raise ValueError(f"selected {config} basenames are not unique")

        archive_hashes: dict[str, tuple[str, int]] = {}
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                basename = Path(member.name).name
                if basename not in selected_by_basename or not member.isfile():
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f"cannot read {config} archive member {member.name}")
                raw = stream.read()
                archive_hashes[basename] = (hashlib.sha256(raw).hexdigest(), len(raw))
                if len(archive_hashes) == SAMPLE_PER_LANGUAGE:
                    break
        if set(archive_hashes) != set(selected_by_basename):
            raise ValueError(f"pinned {config} archive lacks selected files")

        for row in selected:
            record = expected_record(row, language, args.external_data_dir)
            key = (
                config,
                int(row["row_idx"]),
                int(row["dataset_row_id"]),
                row["source_basename"],
            )
            if key not in manifest_by_key:
                raise ValueError(f"derived FLEURS row is absent from manifest: {key}")
            if record != manifest_by_key[key]:
                raise ValueError(f"manifest record does not derive from pinned source: {key}")
            archive_sha, archive_bytes = archive_hashes[row["source_basename"]]
            if (archive_sha, archive_bytes) != (
                record["audio_sha256"],
                record["audio_bytes"],
            ):
                raise ValueError(f"materialized audio differs from archive for {key}")
            derived.append(record)
        source_audit[config] = {
            "pool_rows": len(rows),
            "selected_rows": len(selected),
            "tsv_sha256": lock["tsv"]["sha256"],
            "archive_sha256": lock["archive"]["sha256"],
            "archive_selected_payloads_exact": True,
        }

    if canonical_sha256(pool) != EXPECTED_POOL_METADATA_SHA256:
        raise ValueError("FLEURS pool derivation differs from frozen identity")
    derived.sort(
        key=lambda row: (
            list(FLEURS_CONFIG_BY_LANGUAGE).index(row["language"]),
            row["row_idx"],
        )
    )
    if derived != manifest["records"]:
        raise ValueError("ordered derived FLEURS records differ from manifest")
    expected_identity = {
        "schema_version": 1,
        "dataset_id": FLEURS_DATASET_ID,
        "revision": FLEURS_REVISION,
        "split": FLEURS_SPLIT,
        "license": FLEURS_LICENSE,
        "configs_by_language": dict(FLEURS_CONFIG_BY_LANGUAGE),
        "expected_pool_rows": EXPECTED_VALIDATION_ROWS,
        "sample_per_language": SAMPLE_PER_LANGUAGE,
        "selection_algorithm": SELECTION_ALGORITHM,
        "transport": "pinned Hub data/<config>/dev.tsv plus audio/dev.tar.gz",
    }
    if manifest["identity"] != expected_identity:
        raise ValueError("external manifest frozen identity differs")
    if manifest["pool_metadata_sha256"] != EXPECTED_POOL_METADATA_SHA256:
        raise ValueError("external manifest pool identity differs")
    if manifest["pool_rows"] != sum(EXPECTED_VALIDATION_ROWS.values()):
        raise ValueError("external manifest pool count differs")

    lock_payload = {
        "schema_version": 1,
        "dataset_id": FLEURS_DATASET_ID,
        "revision": FLEURS_REVISION,
        "split": FLEURS_SPLIT,
        "license": FLEURS_LICENSE,
        "card": FLEURS_CARD,
        "source_files": FLEURS_SOURCE_FILES,
        "pool_metadata_sha256": EXPECTED_POOL_METADATA_SHA256,
        "selection_algorithm": SELECTION_ALGORITHM,
        "manifest_sha256": EXPECTED_EXTERNAL_MANIFEST_SHA256,
    }
    return manifest, {
        "schema_version": 1,
        "release_lock_sha256": canonical_sha256(lock_payload),
        "release_lock": lock_payload,
        "source_audit": source_audit,
        "pool_rows": len(pool),
        "selected_rows": len(derived),
        "composite_source_keys_unique": True,
        "record_labels_and_selection_derived_from_pinned_tsv": True,
        "selected_audio_exactly_matches_pinned_archives": True,
        "fleurs_test_read": False,
    }


def load_external_features(
    manifest: Mapping[str, Any], data_dir: Path
) -> tuple[list[dict[str, Any]], float]:
    examples: list[dict[str, Any]] = []
    frontend = LogMelFrontend().eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for index, record in enumerate(manifest["records"], start=1):
            path = data_dir / record["audio_path"]
            waveform = load_audio(path)
            if len(waveform) != int(record["num_samples"]):
                raise ValueError(f"decoded length differs for {record['clip_id']}")
            features = frontend(waveform).squeeze(0).cpu()
            expected_frames = feature_frame_count(int(record["num_samples"]))
            if len(features) != expected_frames:
                raise AssertionError(f"feature count differs for {record['clip_id']}")
            examples.append(
                {
                    "id": record["clip_id"],
                    "language": record["language"],
                    "dataset_row_id": int(record["dataset_row_id"]),
                    "num_samples": int(record["num_samples"]),
                    "features": features,
                }
            )
            if index % 100 == 0:
                print(f"computed external features {index}/{len(manifest['records'])}", flush=True)
    return examples, time.perf_counter() - started


def prefix_key(value: float | str) -> str:
    if value == "full":
        return "full"
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def eligible_indices(
    examples: Sequence[Mapping[str, Any]], prefix: float | str
) -> list[int]:
    if prefix == "full":
        return list(range(len(examples)))
    wanted = round(float(prefix) * SAMPLE_RATE)
    return [
        index
        for index, example in enumerate(examples)
        if int(example["num_samples"]) >= wanted
    ]


def classification_summary(
    expected: Sequence[str], predicted: Sequence[str], maxima: Sequence[float]
) -> dict[str, Any]:
    if not (len(expected) == len(predicted) == len(maxima)) or not expected:
        raise ValueError("classification summary needs equal non-empty inputs")
    support = Counter(expected)
    confusion = {language: Counter() for language in LANGUAGE_CODES}
    for truth, guess in zip(expected, predicted, strict=True):
        confusion[truth][guess] += 1
    supported = [language for language in LANGUAGE_CODES if support[language]]
    recalls: dict[str, float] = {}
    f1s: dict[str, float] = {}
    for language in supported:
        true_positive = confusion[language][language]
        false_negative = support[language] - true_positive
        false_positive = sum(
            confusion[other][language]
            for other in LANGUAGE_CODES
            if other != language
        )
        recalls[language] = true_positive / support[language]
        denominator = 2 * true_positive + false_positive + false_negative
        f1s[language] = 0.0 if denominator == 0 else 2 * true_positive / denominator
    correct = sum(a == b for a, b in zip(expected, predicted, strict=True))
    return {
        "n_clips": len(expected),
        "correct": int(correct),
        "accuracy": correct / len(expected),
        "language_macro_accuracy": float(np.mean(list(recalls.values()))),
        "macro_f1": float(np.mean(list(f1s.values()))),
        "minimum_language_recall": min(recalls.values()),
        "languages_with_support": supported,
        "languages_without_support": [
            language for language in LANGUAGE_CODES if not support[language]
        ],
        "per_language_recall": recalls,
        "per_language_f1": f1s,
        "support_per_language": {
            language: support[language] for language in LANGUAGE_CODES
        },
        "confusion": {
            language: dict(sorted(confusion[language].items()))
            for language in LANGUAGE_CODES
        },
        "mean_max_probability": float(np.mean(maxima)),
    }


def summarize_predictions(
    examples: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for prefix in PREFIX_SECONDS:
        key = prefix_key(prefix)
        indices = eligible_indices(examples, prefix)
        expected = [str(examples[index]["language"]) for index in indices]
        guessed = [str(predictions[key]["prediction"][index]) for index in indices]
        maxima = [float(predictions[key]["max_probability"][index]) for index in indices]
        summary = classification_summary(expected, guessed, maxima)
        summary["requested_prefix_seconds"] = prefix
        summary["short_clip_policy"] = "exclude; never zero-pad"
        summary["excluded_short_clips"] = len(examples) - len(indices)
        summaries[key] = summary
    composite = float(
        np.mean(
            [summaries[key]["language_macro_accuracy"] for key in SELECTION_PREFIX_KEYS]
        )
    )
    language_composite = {
        language: float(
            np.mean(
                [
                    summaries[key]["per_language_recall"][language]
                    for key in SELECTION_PREFIX_KEYS
                ]
            )
        )
        for language in LANGUAGE_CODES
    }
    return {
        "label_composite_1s_2s_full": composite,
        "minimum_language_recall_composite_1s_2s_full": min(
            language_composite.values()
        ),
        "per_language_recall_composite_1s_2s_full": language_composite,
        "prefix": summaries,
    }


def evaluate_external_state(
    state: Mapping[str, torch.Tensor], examples: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = CausalLIDStudent(**configured_model_kwargs())
    model.load_state_dict(state)
    model.eval()
    predictions: dict[str, dict[str, list[Any]]] = {
        prefix_key(prefix): {
            "prediction": [None] * len(examples),
            "max_probability": [None] * len(examples),
        }
        for prefix in PREFIX_SECONDS
    }
    eligible = {
        prefix_key(prefix): set(eligible_indices(examples, prefix))
        for prefix in PREFIX_SECONDS
    }
    order = sorted(range(len(examples)), key=lambda index: len(examples[index]["features"]))
    with torch.inference_mode():
        for start in range(0, len(order), STUDENT_BATCH_SIZE):
            indices = order[start : start + STUDENT_BATCH_SIZE]
            feature_batch = pad_sequence(
                [examples[index]["features"] for index in indices], batch_first=True
            )
            logits = model(feature_batch)
            for batch_row, example_index in enumerate(indices):
                example = examples[example_index]
                for prefix in PREFIX_SECONDS:
                    key = prefix_key(prefix)
                    if example_index not in eligible[key]:
                        continue
                    frames = (
                        len(example["features"])
                        if prefix == "full"
                        else feature_frame_count(round(float(prefix) * SAMPLE_RATE))
                    )
                    stable_stop = frames - MODEL_LOOKAHEAD_FRAMES
                    if stable_stop <= LABEL_DELAY_FRAMES:
                        raise ValueError(f"{example['id']} has no aligned output at {key}")
                    posterior = torch.softmax(
                        logits[batch_row, LABEL_DELAY_FRAMES:stable_stop], dim=-1
                    ).mean(dim=0)
                    winner = int(posterior.argmax())
                    predictions[key]["prediction"][example_index] = LANGUAGE_CODES[winner]
                    predictions[key]["max_probability"][example_index] = float(
                        posterior[winner]
                    )
    first = examples[0]["features"].unsqueeze(0)
    with torch.inference_mode():
        whole = model(first)[:, : first.shape[1] - MODEL_LOOKAHEAD_FRAMES]
        streamed = model.streaming_forward(first, chunk_frames=CHUNK_FRAMES)
    torch.testing.assert_close(whole, streamed, rtol=1e-5, atol=1e-5)
    return summarize_predictions(examples, predictions), predictions


def compact_predictions(
    examples: Sequence[Mapping[str, Any]], predictions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "clip_ids": [example["id"] for example in examples],
        "expected": [example["language"] for example in examples],
        "by_prefix": {
            key: {
                "prediction": value["prediction"],
                "max_probability": value["max_probability"],
            }
            for key, value in predictions.items()
        },
    }


def audio_cluster_bootstrap(
    examples: Sequence[Mapping[str, Any]],
    challenger: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    language_distributions: list[np.ndarray] = []
    point_by_language: dict[str, float] = {}
    for language in LANGUAGE_CODES:
        indices = np.asarray(
            [
                index
                for index, example in enumerate(examples)
                if example["language"] == language
            ],
            dtype=np.int64,
        )
        if len(indices) != SAMPLE_PER_LANGUAGE:
            raise ValueError(f"bootstrap expected 100 {language} clips")
        per_audio = np.zeros(len(indices), dtype=np.float64)
        truth = np.asarray([language] * len(indices))
        for key in SELECTION_PREFIX_KEYS:
            challenger_correct = np.asarray(
                [challenger[key]["prediction"][index] for index in indices]
            ) == truth
            baseline_correct = np.asarray(
                [baseline[key]["prediction"][index] for index in indices]
            ) == truth
            per_audio += (
                challenger_correct.astype(np.float64)
                - baseline_correct.astype(np.float64)
            ) / len(SELECTION_PREFIX_KEYS)
        point_by_language[language] = float(per_audio.mean())
        draws = rng.integers(
            0, len(per_audio), size=(BOOTSTRAP_REPLICATES, len(per_audio))
        )
        language_distributions.append(per_audio[draws].mean(axis=1))
    distribution = np.mean(np.stack(language_distributions, axis=1), axis=1)
    point = float(np.mean(list(point_by_language.values())))
    low, high = np.quantile(distribution, (0.025, 0.975))
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "method": (
            "paired stratified audio-cluster percentile bootstrap; one shared "
            "weight vector per language is reused across 1s/2s/full views"
        ),
        "independent_cluster_count": len(examples),
        "views_per_audio_cluster": len(SELECTION_PREFIX_KEYS),
        "gain": point,
        "gain_pp": 100 * point,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci95_low_pp": 100 * float(low),
        "ci95_high_pp": 100 * float(high),
        "per_language_gain": point_by_language,
        "interval_scope": "descriptive reused validation, not locked-test confirmation",
    }


def assert_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            assert_finite_json(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            assert_finite_json(nested, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON at {path}: {value}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    launched_at = datetime.now(timezone.utc).isoformat()
    wall_started = time.perf_counter()
    torch.set_num_threads(args.threads)
    source_at_start = live_source_snapshot()
    stage_at_start = validate_historical_stage()
    origins = local_module_origins()
    reference_hashes_at_start = {
        "checkpoint": sha256_file(args.reference_checkpoint),
        "summary": sha256_file(args.reference_summary),
        "loss_results": sha256_file(args.loss_results),
        "prior_external_results": sha256_file(args.prior_external_results),
        "external_manifest": sha256_file(args.external_data_dir / "manifest.json"),
    }
    parent = json.loads(args.loss_results.read_text(encoding="utf-8"))
    prior_external = json.loads(
        args.prior_external_results.read_text(encoding="utf-8")
    )
    if parent.get("run_id") != "044ef348f33663f014b6d5b868ea23e969758438f70e5cfd500439d4e7cdbdf5":
        raise ValueError("loss-factorial parent run differs")
    if prior_external.get("run_id") != "8ec024940fdd6f38731cc26f2d2f5a5ec962bdb9018d6f3a395f49b30a6097ea":
        raise ValueError("prior external-validation run differs")
    if parent["selection"]["hybrid_kd_ce"]["selected_step"] != SELECTED_STEP:
        raise ValueError("parent no longer selects hybrid step 1520")

    checkpoint = torch.load(
        args.reference_checkpoint, map_location="cpu", weights_only=True
    )
    summary = json.loads(args.reference_summary.read_text(encoding="utf-8"))
    final_state = checkpoint["model_state"]
    final_hash = model_state_sha256(final_state)
    if final_hash != summary["model_state_sha256"]:
        raise ValueError("reference checkpoint/summary model identity differs")
    if checkpoint["run_id"] != summary["run_id"]:
        raise ValueError("reference checkpoint/summary run identity differs")

    (
        states,
        training_audit,
        manifest_snapshot,
        target_cache,
        teacher_correct_by_id,
        train_dataset,
    ) = train_frozen_pair(args, parent)
    synthetic = evaluate_same_synthetic_clips(
        args,
        parent,
        states,
        manifest_snapshot,
        target_cache,
        teacher_correct_by_id,
        train_dataset,
    )
    del train_dataset, teacher_correct_by_id
    gc.collect()
    print("synthetic held-out/switch parity passed; validating external release", flush=True)

    external_manifest, release_audit = validate_external_release(args)
    external_examples, feature_seconds = load_external_features(
        external_manifest, args.external_data_dir
    )
    external_states = OrderedDict(
        (
            ("frame_kd_step_1520", states["frame_kd"]),
            ("hybrid_kd_ce_step_1520", states["hybrid_kd_ce"]),
            ("frame_kd_step_1600", final_state),
        )
    )
    external_metrics: dict[str, Any] = {}
    raw_predictions: dict[str, Any] = {}
    scoring_started = time.perf_counter()
    for name, state in external_states.items():
        metrics, predictions = evaluate_external_state(state, external_examples)
        external_metrics[name] = metrics
        raw_predictions[name] = predictions
        print(
            f"external {name}: composite="
            f"{100 * metrics['label_composite_1s_2s_full']:.2f}%",
            flush=True,
        )
    scoring_seconds = time.perf_counter() - scoring_started

    prior_final = prior_external["students"]["1600"]
    final_metrics_exact = (
        canonical_sha256(external_metrics["frame_kd_step_1600"])
        == canonical_sha256(prior_final["external_validation"])
    )
    final_predictions_exact = (
        canonical_sha256(
            compact_predictions(
                external_examples, raw_predictions["frame_kd_step_1600"]
            )
        )
        == canonical_sha256(prior_final["external_predictions"])
    )
    if not (final_metrics_exact and final_predictions_exact):
        raise RuntimeError("current-final external scoring did not reproduce prior run")

    hybrid = external_metrics["hybrid_kd_ce_step_1520"]
    control = external_metrics["frame_kd_step_1520"]
    final = external_metrics["frame_kd_step_1600"]
    hybrid_predictions = raw_predictions["hybrid_kd_ce_step_1520"]
    control_predictions = raw_predictions["frame_kd_step_1520"]
    final_predictions = raw_predictions["frame_kd_step_1600"]
    vs_control = audio_cluster_bootstrap(
        external_examples, hybrid_predictions, control_predictions
    )
    vs_final = audio_cluster_bootstrap(
        external_examples, hybrid_predictions, final_predictions
    )
    gain_vs_control_pp = 100 * (
        hybrid["label_composite_1s_2s_full"]
        - control["label_composite_1s_2s_full"]
    )
    gain_vs_final_pp = 100 * (
        hybrid["label_composite_1s_2s_full"]
        - final["label_composite_1s_2s_full"]
    )
    if not math.isclose(gain_vs_control_pp, vs_control["gain_pp"], abs_tol=1e-10):
        raise AssertionError("control bootstrap point estimate differs")
    if not math.isclose(gain_vs_final_pp, vs_final["gain_pp"], abs_tol=1e-10):
        raise AssertionError("final bootstrap point estimate differs")

    final_local = parent["curves"]["frame_kd"][-1]
    hybrid_local = next(
        row
        for row in parent["curves"]["hybrid_kd_ce"]
        if row["step"] == SELECTED_STEP
    )
    confirmation_gates = {
        "exact_frozen_candidate_reproduction": all(
            training_audit["arms"][arm]["model_state_exact"]
            and training_audit["arms"][arm]["loss_trace_exact"]
            and training_audit["arms"][arm]["gradient_trace_exact"]
            for arm in ARMS
        ),
        "gain_vs_matched_control_at_least_2pp": (
            gain_vs_control_pp >= MATERIAL_GAIN_GATE_PP
        ),
        "gain_vs_current_final_at_least_2pp": (
            gain_vs_final_pp >= MATERIAL_GAIN_GATE_PP
        ),
        "audio_cluster_ci95_low_above_zero_vs_matched_control": (
            vs_control["ci95_low"] > 0
        ),
        "audio_cluster_ci95_low_above_zero_vs_current_final": (
            vs_final["ci95_low"] > 0
        ),
        "all_seven_external_composite_recalls_nonzero": (
            hybrid["minimum_language_recall_composite_1s_2s_full"] > 0
        ),
        "local_all_training_accuracy_at_least_80pct": (
            synthetic["hybrid_kd_ce"]["all_train"]["accuracy"]
            >= LOCAL_ACCURACY_FLOOR
        ),
        "local_exact_edge_accuracy_at_least_80pct": (
            synthetic["hybrid_kd_ce"]["exact_edge"]["accuracy"]
            >= LOCAL_ACCURACY_FLOOR
        ),
        "local_committed_false_switches_no_worse_than_current_final": (
            hybrid_local["synthetic_dev"]["monolingual_stability"][
                "committed_false_switches"
            ]
            <= final_local["synthetic_dev"]["monolingual_stability"][
                "committed_false_switches"
            ]
        ),
        "no_added_local_raw_or_policy_switch_miss": (
            hybrid_local["switch"]["raw_detected"]
            >= final_local["switch"]["raw_detected"]
            and hybrid_local["switch"]["policy_detected"]
            >= final_local["switch"]["policy_detected"]
        ),
    }
    verdict = "adopt" if all(confirmation_gates.values()) else "reject"
    if verdict == "adopt":
        verdict_reason = (
            "the frozen hybrid loss clears both external effect-size and shared-"
            "audio uncertainty gates, has nonzero composite recall for all seven "
            "languages, and preserves the predeclared local fit/stability gates; "
            "adopt the loss recipe, not the experiment weights"
        )
    else:
        failed = [name for name, passed in confirmation_gates.items() if not passed]
        verdict_reason = (
            "the frozen hybrid loss fails one or more external/local confirmation "
            f"gates ({', '.join(failed)}); do not promote the recipe or weights"
        )

    # Revalidate all mutable inputs and imported code after the expensive work.
    target_cache.validate_all()
    if live_source_snapshot() != source_at_start:
        raise RuntimeError("experiment/helper source changed during the run")
    if validate_historical_stage() != stage_at_start:
        raise RuntimeError("historical source stage changed during the run")
    reference_hashes_at_end = {
        "checkpoint": sha256_file(args.reference_checkpoint),
        "summary": sha256_file(args.reference_summary),
        "loss_results": sha256_file(args.loss_results),
        "prior_external_results": sha256_file(args.prior_external_results),
        "external_manifest": sha256_file(args.external_data_dir / "manifest.json"),
    }
    if reference_hashes_at_end != reference_hashes_at_start:
        raise RuntimeError("reference artifact changed during the run")
    # Rehash every selected waveform after scoring.  The archive-to-audio proof
    # above plus these final hashes close the external input over the run.
    for record in external_manifest["records"]:
        if sha256_file(args.external_data_dir / record["audio_path"]) != record[
            "audio_sha256"
        ]:
            raise RuntimeError(f"external audio changed for {record['clip_id']}")

    configuration = {
        "analysis_version": ANALYSIS_VERSION,
        "threads": THREADS,
        "selected_arm": "hybrid_kd_ce",
        "selected_step": SELECTED_STEP,
        "matched_control_arm": "frame_kd",
        "current_final_step": PARENT_TOTAL_STEPS,
        "prefix_seconds": [0.5, 1, 2, 4, "full"],
        "selection_prefixes": list(SELECTION_PREFIX_KEYS),
        "short_clip_policy": "exclude; never zero-pad",
        "student_batch_size": STUDENT_BATCH_SIZE,
        "material_gain_gate_pp": MATERIAL_GAIN_GATE_PP,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "fleurs_test_read": False,
        "architecture_changed": False,
        "target_timing_changed": False,
    }
    state_hashes = {
        "frame_kd_step_1520": model_state_sha256(states["frame_kd"]),
        "hybrid_kd_ce_step_1520": model_state_sha256(states["hybrid_kd_ce"]),
        "frame_kd_step_1600": final_hash,
    }
    experiment_identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "configuration": configuration,
        "historical_pipeline_source_root_sha256": STAGE_ROOT_SHA256,
        "historical_pipeline_source_manifest_sha256": sha256_file(STAGE_MANIFEST),
        "local_module_origins": origins,
        "live_source_files": source_at_start,
        "reference_hashes": reference_hashes_at_start,
        "parent_loss_run_id": parent["run_id"],
        "prior_external_run_id": prior_external["run_id"],
        "main_run_id": summary["run_id"],
        "manifest_snapshot": manifest_snapshot.identity(),
        "target_cache_identity": target_cache.identity,
        "external_release_lock_sha256": release_audit["release_lock_sha256"],
        "external_manifest_sha256": EXPECTED_EXTERNAL_MANIFEST_SHA256,
        "model_state_sha256_by_candidate": state_hashes,
    }
    run_id = canonical_sha256(experiment_identity)
    results = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "question": (
            "Does pinned external validation confirm the frozen step-1520 "
            "equal-clip 50:50 KD+CE loss candidate over its matched control and "
            "the current step-1600 pure-KD checkpoint?"
        ),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "run_id": run_id,
        "launched_at_utc": launched_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - wall_started,
        "experiment_identity": experiment_identity,
        "configuration": configuration,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "training_reproduction": training_audit,
        "synthetic_same_heldout_and_switch_clips": synthetic,
        "external_data": release_audit,
        "external_validation": {
            "role": "reused_pinned_validation",
            "test_rows_read": False,
            "manifest_sha256": EXPECTED_EXTERNAL_MANIFEST_SHA256,
            "n_clips": len(external_examples),
            "model_state_sha256_by_candidate": state_hashes,
            "metrics": external_metrics,
            "predictions": {
                name: compact_predictions(external_examples, predictions)
                for name, predictions in raw_predictions.items()
            },
            "hybrid_gain_vs_matched_control_pp": gain_vs_control_pp,
            "hybrid_gain_vs_current_final_pp": gain_vs_final_pp,
            "hybrid_vs_matched_control_audio_cluster_bootstrap": vs_control,
            "hybrid_vs_current_final_audio_cluster_bootstrap": vs_final,
        },
        "confirmation_gates": confirmation_gates,
        "timing": {
            "training_wall_seconds": training_audit["wall_seconds"],
            "feature_extraction_wall_seconds": feature_seconds,
            "three_state_scoring_wall_seconds": scoring_seconds,
            "teacher_inference_run": False,
        },
        "identity_audit": {
            "historical_pipeline_stage_verified_before_and_after": True,
            "historical_pipeline_import_origins_confined_to_stage": True,
            "experiment_and_helper_sources_unchanged": True,
            "reference_artifacts_unchanged": True,
            "parent_manifest_and_audio_generation_retained": True,
            "schema6_training_tensors_exactly_reproduced_schema5_trajectory": True,
            "target_cache_revalidated_at_end": True,
            "external_release_rederived_from_pinned_tsv_and_archives": True,
            "external_audio_rehashed_after_scoring": True,
            "current_final_external_metrics_exactly_reproduced": final_metrics_exact,
            "current_final_external_predictions_exactly_reproduced": (
                final_predictions_exact
            ),
            "stable_streaming_full_equivalence_checked_per_state": True,
        },
        "decision": {
            "verdict": verdict,
            "reason": verdict_reason,
            "adopt_loss_recipe": verdict == "adopt",
            "promote_experiment_weights": False,
            "modify_main_checkpoint": False,
            "locked_test_read": False,
        },
        "limitations": [
            "the hybrid candidate is one deterministic seed on a 71-clip synthetic training corpus",
            "FLEURS validation was already used by a prior checkpoint experiment, so intervals are descriptive and not locked-test confirmation",
            "FLEURS contains monolingual read speech and cannot establish natural Hindi-English switch behavior",
            "the same two mirrored synthetic switches remain a local diagnostic rather than independent natural trials",
            "known-label CE changes the project's pure-distillation claim and is unavailable for unlabelled call audio",
            "the current availability-valid teacher switch trajectory is unchanged",
        ],
    }
    assert_finite_json(results)
    atomic_json_write(results, args.output)
    print(
        f"wrote {args.output}; verdict={verdict}; run_id={run_id}; "
        f"gain_vs_final={gain_vs_final_pp:.2f}pp",
        flush=True,
    )


if __name__ == "__main__":
    main()
