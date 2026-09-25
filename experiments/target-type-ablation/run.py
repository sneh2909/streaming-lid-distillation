#!/usr/bin/env python3
"""Ablate offline-teacher target type with one fixed streaming student.

The three arms deliberately isolate the supervision trajectory while holding the
student, optimiser, data order, seed, delay, lookahead, and training budget fixed:

* ``full_utterance`` repeats one whole-clip posterior at every frame;
* ``centred_2s`` uses a two-second window [t-1 s, t+1 s]; and
* ``prefix_delta250`` uses the growing prefix [0, t+250 ms].

Only the prefix arm obeys the student's 250 ms evidence budget.  The other two
are useful diagnostics for the teacher-sees-the-future asymmetry, not deployable
target contracts at the latency tested here.

Run from the repository root with::

    UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
      uv run python experiments/target-type-ablation/run.py --fresh

Large target caches and checkpoints are written below the gitignored ``data/``
tree.  The compact, reviewable measurements are written to ``results.json``
next to this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PIPELINE_SOURCES = (
    "scripts/teacher_targets.py",
    "scripts/train.py",
    "scripts/eval.py",
    "src/streaming_lid/audio.py",
    "src/streaming_lid/config.py",
    "src/streaming_lid/data.py",
    "src/streaming_lid/loss.py",
    "src/streaming_lid/model.py",
)


def snapshot_pipeline_sources_before_import() -> tuple[str, dict[str, str]]:
    """Bind the exact source snapshot that this Python process imports."""
    combined = hashlib.sha256()
    per_file = {}
    for relative_path in PIPELINE_SOURCES:
        contents = (REPO_ROOT / relative_path).read_bytes()
        per_file[relative_path] = hashlib.sha256(contents).hexdigest()
        combined.update(relative_path.encode("utf-8"))
        combined.update(contents)
    return combined.hexdigest(), per_file


(
    PIPELINE_SOURCE_SHA256_AT_IMPORT,
    PIPELINE_SOURCE_FILE_SHA256_AT_IMPORT,
) = snapshot_pipeline_sources_before_import()

# Reuse the production experiment's target interpolation, label mapping, data
# loading, model, loss, batching, and emitted-chunk timing/smoothing code.
from scripts.eval import (  # noqa: E402
    chunk_availability_times,
    chunk_posteriors,
    smooth_posteriors,
)
from scripts.teacher_targets import (  # noqa: E402
    expected_language,
    interpolate_posteriors,
    label_indices,
)
from scripts.train import seed_everything  # noqa: E402
from streaming_lid.audio import (  # noqa: E402
    LogMelFrontend,
    feature_frame_count,
    load_audio,
)
from streaming_lid.config import (  # noqa: E402
    EARLY_RAMP_FRAMES,
    EVIDENCE_LOOKAHEAD_MS,
    FRAME_MS,
    HOP_LENGTH,
    LABEL_DELAY_FRAMES,
    LANGUAGE_CODES,
    MODEL_LOOKAHEAD_FRAMES,
    SAMPLE_RATE,
    TEACHER_HOP_FRAMES,
    TEACHER_NAME,
    TEACHER_TEMPERATURE,
    WIN_LENGTH,
)
from streaming_lid.data import (  # noqa: E402
    DistillationDataset,
    collate_distillation_batch,
    read_manifest,
    require_speaker_disjoint,
    resolve_audio_path,
)
from streaming_lid.loss import delayed_distillation_loss  # noqa: E402
from streaming_lid.model import CausalLIDStudent  # noqa: E402


EXPERIMENT_NAME = "target-type-ablation"
SCHEMA_VERSION = 3
ANALYSIS_VERSION = "target-type-ablation-v3"
TEACHER_REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
TEACHER_ARTIFACT_FILES = (
    "classifier.ckpt",
    "embedding_model.ckpt",
    "hyperparams.yaml",
    "label_encoder.txt",
)
VARIANTS = ("full_utterance", "centred_2s", "prefix_delta250")
CENTRED_PAST_MS = 1_000
CENTRED_FUTURE_MS = 1_000
POLICY_THRESHOLD = 0.60
POLICY_MARGIN = 0.10
POLICY_DWELL_CHUNKS = 3
POLICY_EMA_NEW_WEIGHT = 0.30
PERSISTENCE_CHUNKS = 3
BOUNDARY_COLLAR_MS = 250


@dataclass(frozen=True)
class TargetArrays:
    raw: np.ndarray
    soft: np.ndarray
    anchor_frames: np.ndarray
    raw_anchors: np.ndarray
    in_set_mass: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/manifest.jsonl"),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("data/experiments/target-type-ablation"),
        help="Gitignored target/checkpoint cache.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".cache/models/lang-id-voxlingua107-ecapa"),
        help="Base directory for the revision-and-artifact-qualified teacher cache.",
    )
    parser.add_argument("--steps", type=int, default=1_600)
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--teacher-batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Regenerate target files and replace any partial arm results.",
    )
    parser.add_argument(
        "--restart-results",
        action="store_true",
        help="Replace arm results for the current source snapshot but reuse validated targets.",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def hash_file(digest: "hashlib._Hash", path: Path) -> None:
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    hash_file(digest, path)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_fingerprint(items: list[dict], manifest: Path) -> str:
    """Hash exact manifest records and source WAV bytes used by an experiment."""
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda record: record["id"]):
        digest.update(json.dumps(item, sort_keys=True).encode("utf-8"))
        hash_file(digest, resolve_audio_path(item, manifest))
    return digest.hexdigest()


def pipeline_source_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative_path in PIPELINE_SOURCES:
        digest.update(relative_path.encode("utf-8"))
        hash_file(digest, REPO_ROOT / relative_path)
    return digest.hexdigest()


def driver_fingerprint() -> str:
    return file_sha256(Path(__file__).resolve())


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchaudio": importlib.metadata.version("torchaudio"),
        "speechbrain": importlib.metadata.version("speechbrain"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
    }


def resolve_teacher_artifact() -> tuple[Path, dict]:
    """Resolve and hash a pinned Hub snapshot before loading the teacher."""
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=TEACHER_NAME,
            revision=TEACHER_REVISION,
            allow_patterns=list(TEACHER_ARTIFACT_FILES),
        )
    ).resolve()
    file_records = {}
    combined = hashlib.sha256()
    for filename in TEACHER_ARTIFACT_FILES:
        path = snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(f"pinned teacher artifact is missing {path}")
        sha256 = file_sha256(path)
        size = path.stat().st_size
        file_records[filename] = {"sha256": sha256, "bytes": size}
        combined.update(filename.encode("utf-8"))
        combined.update(bytes.fromhex(sha256))
    identity = {
        "model_id": TEACHER_NAME,
        "revision": TEACHER_REVISION,
        "artifact_sha256": combined.hexdigest(),
        "files": file_records,
    }
    return snapshot, identity


def state_dict_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def standard_anchor_frames(num_frames: int) -> np.ndarray:
    anchors = np.arange(0, num_frames, TEACHER_HOP_FRAMES, dtype=np.int64)
    if len(anchors) == 0 or anchors[-1] != num_frames - 1:
        anchors = np.append(anchors, num_frames - 1)
    return anchors


def causal_hold_posteriors(
    anchor_frames: np.ndarray, anchor_values: np.ndarray, num_frames: int
) -> np.ndarray:
    """Expand anchors without consulting any anchor later than the target frame."""
    frame_index = np.arange(num_frames)
    anchor_index = np.searchsorted(anchor_frames, frame_index, side="right") - 1
    anchor_index = np.clip(anchor_index, 0, len(anchor_frames) - 1)
    if np.any(anchor_frames[anchor_index] > frame_index):
        raise AssertionError("causal hold selected a future anchor")
    expanded = anchor_values[anchor_index].astype(np.float32, copy=True)
    expanded /= np.clip(expanded.sum(axis=-1, keepdims=True), 1e-8, None)
    return expanded


def copy_padded_slice(
    waveform: torch.Tensor, source_start: int, source_end: int
) -> torch.Tensor:
    """Extract a fixed interval and zero-pad portions outside the clip."""
    if source_end <= source_start:
        raise ValueError("target window must have positive length")
    output = waveform.new_zeros(source_end - source_start)
    clipped_start = max(0, source_start)
    clipped_end = min(len(waveform), source_end)
    if clipped_end > clipped_start:
        destination_start = clipped_start - source_start
        output[destination_start : destination_start + clipped_end - clipped_start] = (
            waveform[clipped_start:clipped_end]
        )
    return output


def teacher_requests(
    variant: str, waveform: torch.Tensor, num_frames: int
) -> tuple[np.ndarray, list[torch.Tensor]]:
    if variant == "full_utterance":
        return np.asarray([num_frames // 2], dtype=np.int64), [waveform]

    anchors = standard_anchor_frames(num_frames)
    requests = []
    for anchor in anchors:
        frame_end_sample = int(anchor) * HOP_LENGTH + WIN_LENGTH
        if variant == "centred_2s":
            past_samples = round(CENTRED_PAST_MS * SAMPLE_RATE / 1_000)
            future_samples = round(CENTRED_FUTURE_MS * SAMPLE_RATE / 1_000)
            requests.append(
                copy_padded_slice(
                    waveform,
                    frame_end_sample - past_samples,
                    frame_end_sample + future_samples,
                )
            )
        elif variant == "prefix_delta250":
            future_samples = round(EVIDENCE_LOOKAHEAD_MS * SAMPLE_RATE / 1_000)
            prefix_end = min(len(waveform), frame_end_sample + future_samples)
            # The first request is 275 ms under the main 25 ms analysis frame
            # plus 250 ms evidence budget, comfortably above the frontend minimum.
            requests.append(waveform[:prefix_end].contiguous())
        else:
            raise ValueError(f"unknown target variant: {variant}")
    return anchors, requests


def classify_selected_logs(
    teacher: torch.nn.Module,
    requests: list[torch.Tensor],
    selected_indices: list[int],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_logs = []
    selected_masses = []
    with torch.inference_mode():
        for start in range(0, len(requests), batch_size):
            waveforms = requests[start : start + batch_size]
            lengths = torch.tensor([len(waveform) for waveform in waveforms])
            padded = pad_sequence(waveforms, batch_first=True)
            relative_lengths = lengths / padded.shape[1]
            log_probabilities, _, _, _ = teacher.classify_batch(
                padded, relative_lengths
            )
            chosen = log_probabilities[:, selected_indices]
            selected_logs.append(chosen.cpu())
            selected_masses.append(chosen.exp().sum(dim=-1).cpu())
    return torch.cat(selected_logs), torch.cat(selected_masses)


def make_target_arrays(
    teacher: torch.nn.Module,
    selected_indices: list[int],
    variant: str,
    waveform: torch.Tensor,
    num_frames: int,
    batch_size: int,
) -> TargetArrays:
    anchors, requests = teacher_requests(variant, waveform, num_frames)
    selected_logs, in_set_mass = classify_selected_logs(
        teacher, requests, selected_indices, batch_size
    )
    raw_anchors = torch.softmax(selected_logs, dim=-1).numpy()
    soft_anchors = torch.softmax(
        selected_logs / TEACHER_TEMPERATURE, dim=-1
    ).numpy()
    if variant == "full_utterance":
        # Preserve exact parity with the main full-utterance cache. With one
        # anchor, interpolation is just constant repetition.
        raw = interpolate_posteriors(anchors, raw_anchors, num_frames)
        soft = interpolate_posteriors(anchors, soft_anchors, num_frames)
    else:
        # Linear interpolation would consult the next 250 ms anchor and leak up
        # to another 240 ms of audio. Previous-anchor hold preserves each
        # target's stated evidence horizon.
        raw = causal_hold_posteriors(anchors, raw_anchors, num_frames)
        soft = causal_hold_posteriors(anchors, soft_anchors, num_frames)
    return TargetArrays(
        raw=raw,
        soft=soft,
        anchor_frames=anchors,
        raw_anchors=raw_anchors.astype(np.float32),
        in_set_mass=in_set_mass.numpy().astype(np.float32),
    )


def expected_frame_indices(item: dict, num_frames: int) -> np.ndarray:
    expected = []
    for frame in range(num_frames):
        seconds = (frame * HOP_LENGTH + WIN_LENGTH) / SAMPLE_RATE
        expected.append(LANGUAGE_CODES.index(expected_language(item, seconds)))
    return np.asarray(expected, dtype=np.int64)


def target_definition(variant: str) -> dict:
    common = {
        "teacher": TEACHER_NAME,
        "teacher_temperature": TEACHER_TEMPERATURE,
        "anchor_hop_frames": TEACHER_HOP_FRAMES,
        "anchor_hop_ms": TEACHER_HOP_FRAMES * FRAME_MS,
    }
    if variant == "full_utterance":
        return {
            **common,
            "window": "whole clip; one posterior repeated over all frames",
            "anchor_expansion": "constant repetition",
            "future_context_ms": "unbounded until clip end",
            "context_valid_at_tested_latency": False,
        }
    if variant == "centred_2s":
        return {
            **common,
            "window": "[target frame end - 1000 ms, target frame end + 1000 ms]",
            "anchor_expansion": "causal previous-anchor hold on the 10 ms grid",
            "past_context_ms": CENTRED_PAST_MS,
            "future_context_ms": CENTRED_FUTURE_MS,
            "context_valid_at_tested_latency": False,
            "future_context_excess_ms": CENTRED_FUTURE_MS - EVIDENCE_LOOKAHEAD_MS,
        }
    if variant == "prefix_delta250":
        return {
            **common,
            "window": "[clip start, target frame end + 250 ms]",
            "anchor_expansion": "causal previous-anchor hold on the 10 ms grid",
            "past_context_ms": "growing from clip start",
            "future_context_ms": EVIDENCE_LOOKAHEAD_MS,
            "context_valid_at_tested_latency": True,
        }
    raise ValueError(variant)


def validate_cached_target(
    path: Path,
    num_frames: int,
    variant: str,
    run_identity_sha256: str,
    teacher_artifact_sha256: str,
) -> None:
    with np.load(path) as target_file:
        raw = target_file["teacher_probs"]
        soft = target_file["teacher_soft_targets"]
        codes = tuple(target_file["language_codes"].tolist())
        if raw.shape != (num_frames, len(LANGUAGE_CODES)):
            raise ValueError(f"wrong raw target shape in {path}: {raw.shape}")
        if soft.shape != raw.shape:
            raise ValueError(f"raw/soft target shape mismatch in {path}")
        if codes != LANGUAGE_CODES:
            raise ValueError(f"language order mismatch in {path}: {codes}")
        if not np.isfinite(raw).all() or not np.isfinite(soft).all():
            raise ValueError(f"non-finite cached target in {path}")
        scalar_identity = {
            "target_variant": variant,
            "analysis_version": ANALYSIS_VERSION,
            "run_identity_sha256": run_identity_sha256,
            "teacher_revision": TEACHER_REVISION,
            "teacher_artifact_sha256": teacher_artifact_sha256,
        }
        for key, expected in scalar_identity.items():
            actual = str(target_file[key].item())
            if actual != expected:
                raise ValueError(
                    f"cached target identity mismatch in {path}: "
                    f"{key}={actual!r}, expected {expected!r}"
                )


def generate_variant_targets(
    teacher: torch.nn.Module,
    selected_indices: list[int],
    variant: str,
    records: list[dict],
    manifest: Path,
    target_dir: Path,
    batch_size: int,
    input_sha256: str,
    run_identity_sha256: str,
    driver_sha256: str,
    pipeline_source_sha256: str,
    teacher_identity: dict,
    fresh: bool,
) -> dict:
    metadata_path = target_dir / "metadata.json"
    expected_metadata_identity = {
        "experiment": EXPERIMENT_NAME,
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "variant": variant,
        "input_sha256": input_sha256,
        "run_identity_sha256": run_identity_sha256,
        "driver_sha256": driver_sha256,
        "pipeline_source_sha256": pipeline_source_sha256,
        "teacher_identity": teacher_identity,
        "language_codes": list(LANGUAGE_CODES),
        "definition": target_definition(variant),
    }
    if metadata_path.exists() and not fresh:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, expected_value in expected_metadata_identity.items():
            if metadata.get(key) != expected_value:
                raise ValueError(
                    f"stale target cache identity for {variant}: key {key!r}; "
                    "rerun with --fresh"
                )
        for item in records:
            waveform = load_audio(resolve_audio_path(item, manifest))
            validate_cached_target(
                target_dir / f"{item['id']}.npz",
                feature_frame_count(len(waveform)),
                variant,
                run_identity_sha256,
                teacher_identity["artifact_sha256"],
            )
        print(f"{variant}: validated cached targets", flush=True)
        return metadata["summary"]

    target_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    clip_summaries = []
    total_correct = 0
    total_frames = 0
    total_anchors = 0
    for clip_number, item in enumerate(records, start=1):
        waveform = load_audio(resolve_audio_path(item, manifest))
        num_frames = feature_frame_count(len(waveform))
        arrays = make_target_arrays(
            teacher,
            selected_indices,
            variant,
            waveform,
            num_frames,
            batch_size,
        )
        expected = expected_frame_indices(item, num_frames)
        predictions = arrays.raw.argmax(axis=-1)
        correct = int((predictions == expected).sum())
        total_correct += correct
        total_frames += num_frames
        total_anchors += len(arrays.anchor_frames)
        output_path = target_dir / f"{item['id']}.npz"
        np.savez_compressed(
            output_path,
            teacher_probs=arrays.raw,
            teacher_soft_targets=arrays.soft,
            anchor_frames=arrays.anchor_frames,
            anchor_probs=arrays.raw_anchors,
            in_set_mass=arrays.in_set_mass,
            language_codes=np.asarray(LANGUAGE_CODES),
            target_variant=np.asarray(variant),
            analysis_version=np.asarray(ANALYSIS_VERSION),
            run_identity_sha256=np.asarray(run_identity_sha256),
            teacher_revision=np.asarray(TEACHER_REVISION),
            teacher_artifact_sha256=np.asarray(
                teacher_identity["artifact_sha256"]
            ),
        )
        validate_cached_target(
            output_path,
            num_frames,
            variant,
            run_identity_sha256,
            teacher_identity["artifact_sha256"],
        )
        clip_summaries.append(
            {
                "id": item["id"],
                "split": item["split"],
                "frames": num_frames,
                "anchors": len(arrays.anchor_frames),
                "known_label_frame_accuracy": correct / num_frames,
                "mean_selected_language_mass": float(arrays.in_set_mass.mean()),
            }
        )
        print(
            f"{variant} targets [{clip_number:02d}/{len(records)}] "
            f"{item['id']}: anchors={len(arrays.anchor_frames)} "
            f"label_acc={correct / num_frames:.3f}",
            flush=True,
        )
    summary = {
        "clips": len(records),
        "anchors": total_anchors,
        "frames": total_frames,
        "known_label_frame_accuracy_micro_all_splits": total_correct / total_frames,
        "generation_wall_seconds": time.perf_counter() - started,
        "per_clip": clip_summaries,
    }
    metadata = {
        **expected_metadata_identity,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
    }
    atomic_write_json(metadata_path, metadata)
    return summary


def train_variant(
    variant: str,
    manifest: Path,
    target_dir: Path,
    checkpoint_path: Path,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    run_identity_sha256: str,
) -> tuple[CausalLIDStudent, dict]:
    seed_everything(seed)
    dataset = DistillationDataset(manifest, target_dir, splits=("train",))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_distillation_batch,
        generator=generator,
        num_workers=0,
        drop_last=True,
    )
    model = CausalLIDStudent(num_languages=len(LANGUAGE_CODES))
    initial_sha256 = state_dict_fingerprint(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    losses: list[float] = []
    gradient_norms: list[float] = []
    batch_digest = hashlib.sha256()
    examples_seen = 0
    step = 0
    started = time.perf_counter()
    model.train()
    while step < steps:
        for batch in loader:
            batch_digest.update(json.dumps(batch["ids"]).encode("utf-8"))
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["features"])
            loss, loss_stats = delayed_distillation_loss(
                logits,
                batch["targets"],
                batch["lengths"],
                delay_frames=LABEL_DELAY_FRAMES,
                lookahead_frames=MODEL_LOOKAHEAD_FRAMES,
                temperature=TEACHER_TEMPERATURE,
                early_ramp_frames=EARLY_RAMP_FRAMES,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{variant}: non-finite loss at step {step + 1}: {loss.item()}"
                )
            loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(
                    f"{variant}: non-finite gradient at step {step + 1}"
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()
            if not all(
                torch.isfinite(parameter).all() for parameter in model.parameters()
            ):
                raise FloatingPointError(
                    f"{variant}: non-finite parameter after step {step + 1}"
                )
            step += 1
            examples_seen += len(batch["ids"])
            losses.append(float(loss.detach()))
            gradient_norms.append(float(gradient_norm))
            if step == 1 or step % 100 == 0 or step == steps:
                print(
                    f"{variant} step={step:04d} loss={losses[-1]:.6f} "
                    f"grad={gradient_norms[-1]:.4f} "
                    f"valid_frames={loss_stats['valid_frames']:.0f}",
                    flush=True,
                )
            if step >= steps:
                break

    model.eval()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "languages": list(LANGUAGE_CODES),
            "target_variant": variant,
            "analysis_version": ANALYSIS_VERSION,
            "run_identity_sha256": run_identity_sha256,
            "steps": steps,
            "seed": seed,
            "model_kwargs": {
                "num_languages": len(LANGUAGE_CODES),
                "lookahead_frames": MODEL_LOOKAHEAD_FRAMES,
            },
        },
        checkpoint_path,
    )
    window = min(10, len(losses))
    metrics = {
        "optimizer_steps": steps,
        "analysis_version": ANALYSIS_VERSION,
        "run_identity_sha256": run_identity_sha256,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_train_clips": len(dataset),
        "examples_seen": examples_seen,
        "effective_epochs": examples_seen / len(dataset),
        "initial_state_sha256": initial_sha256,
        "batch_order_sha256": batch_digest.hexdigest(),
        "student_params": model.parameter_count,
        "first_10_mean_loss": float(np.mean(losses[:window])),
        "last_10_mean_loss": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:]))
        < float(np.mean(losses[:window])),
        "losses": losses,
        "gradient_norms": gradient_norms,
        "nan_free": all(
            math.isfinite(value) for value in losses + gradient_norms
        ),
        "training_wall_seconds": time.perf_counter() - started,
    }
    return model, metrics


def expected_switch_indices(item: dict, frame_count: int) -> np.ndarray:
    return expected_frame_indices(item, frame_count)


def persistent_transition(
    classes: np.ndarray,
    times: np.ndarray,
    source_index: int,
    target_index: int,
    run_length: int,
) -> dict:
    armed = False
    initial_source_confirmed = None
    transition_start = None
    transition_confirmed = None
    for index in range(len(classes) - run_length + 1):
        run = classes[index : index + run_length]
        if not armed and np.all(run == source_index):
            armed = True
            initial_source_confirmed = float(times[index + run_length - 1])
        if armed and np.all(run == target_index):
            transition_start = float(times[index])
            transition_confirmed = float(times[index + run_length - 1])
            break
    return {
        "persistence_chunks": run_length,
        "initial_source_confirmed_seconds": initial_source_confirmed,
        "transition_start_seconds": transition_start,
        "transition_confirmed_seconds": transition_confirmed,
    }


def threshold_policy_transition(
    chunk_probabilities: np.ndarray,
    times: np.ndarray,
    source_index: int,
    target_index: int,
) -> dict:
    smoothed = smooth_posteriors(
        chunk_probabilities, new_weight=POLICY_EMA_NEW_WEIGHT
    )
    active_chunks = 0
    challenger_chunks = 0
    armed = False
    initial_commit = None
    switch_commit = None
    for chunk_index, posterior in enumerate(smoothed):
        if not armed:
            if (
                posterior[source_index] >= POLICY_THRESHOLD
                and posterior[source_index] - posterior[target_index]
                >= POLICY_MARGIN
            ):
                active_chunks += 1
            else:
                active_chunks = 0
            if active_chunks >= POLICY_DWELL_CHUNKS:
                armed = True
                initial_commit = float(times[chunk_index])
            continue

        if (
            posterior[target_index] >= POLICY_THRESHOLD
            and posterior[target_index] - posterior[source_index] >= POLICY_MARGIN
        ):
            challenger_chunks += 1
        else:
            challenger_chunks = 0
        if challenger_chunks >= POLICY_DWELL_CHUNKS:
            switch_commit = float(times[chunk_index])
            break
    return {
        "threshold": POLICY_THRESHOLD,
        "margin": POLICY_MARGIN,
        "dwell_chunks": POLICY_DWELL_CHUNKS,
        "ema_new_weight": POLICY_EMA_NEW_WEIGHT,
        "initial_commit_seconds": initial_commit,
        "switch_commit_seconds": switch_commit,
    }


def transition_matches_reference_boundary(
    transition: dict,
    boundary_seconds: float,
    *,
    early_tolerance_ms: float = EVIDENCE_LOOKAHEAD_MS,
) -> bool:
    """Whether a persistent source->target event plausibly matches the boundary."""
    start = transition["transition_start_seconds"]
    confirmed = transition["transition_confirmed_seconds"]
    return bool(
        start is not None
        and confirmed is not None
        and start >= boundary_seconds - early_tolerance_ms / 1_000
        and confirmed >= boundary_seconds
    )


def flip_metrics(
    classes: np.ndarray,
    times: np.ndarray,
    *,
    matched_reference_transition: bool,
) -> dict:
    flips = int(np.sum(classes[1:] != classes[:-1])) if len(classes) > 1 else 0
    duration_minutes = (
        max(float(times[-1] - times[0]), 1e-8) / 60 if len(times) > 1 else 0.0
    )
    # Discount the one intended change only when a persistent source->target
    # event actually matched the annotated boundary. Otherwise every observed
    # class change is unmatched churn.
    matched = int(matched_reference_transition)
    unmatched = max(0, flips - matched)
    return {
        "top1_changes": flips,
        "matched_reference_transitions": matched,
        "unmatched_top1_changes": unmatched,
        "unmatched_top1_changes_per_minute": (
            unmatched / duration_minutes if duration_minutes > 0 else 0.0
        ),
    }


def safe_lag_ms(event_seconds: float | None, boundary_seconds: float) -> float | None:
    if event_seconds is None:
        return None
    return 1_000 * (event_seconds - boundary_seconds)


def aggregate_clip_metrics(per_clip: list[dict]) -> dict:
    total_frames = sum(record["frames"] for record in per_clip)
    result = {
        "n_clips": len(per_clip),
        "frames": total_frames,
        "own_target_agreement_micro": sum(
            record["own_target_agreement_correct"] for record in per_clip
        )
        / total_frames,
        "own_target_agreement_macro": float(
            np.mean([record["own_target_agreement"] for record in per_clip])
        ),
        "main_teacher_agreement_micro": sum(
            record["main_teacher_agreement_correct"] for record in per_clip
        )
        / total_frames,
        "main_teacher_agreement_macro": float(
            np.mean([record["main_teacher_agreement"] for record in per_clip])
        ),
        "known_label_accuracy_micro": sum(
            record["known_label_correct"] for record in per_clip
        )
        / total_frames,
        "known_label_accuracy_macro": float(
            np.mean([record["known_label_accuracy"] for record in per_clip])
        ),
        "own_teacher_label_accuracy_micro": sum(
            record["own_teacher_label_correct"] for record in per_clip
        )
        / total_frames,
        "own_teacher_label_accuracy_macro": float(
            np.mean([record["own_teacher_label_accuracy"] for record in per_clip])
        ),
        "student_clip_accuracy": sum(
            record["student_clip_correct"] for record in per_clip
        )
        / len(per_clip),
    }
    return result


def evaluate_heldout(
    model: CausalLIDStudent,
    heldout: list[dict],
    manifest: Path,
    target_dir: Path,
    main_target_dir: Path,
) -> dict:
    frontend = LogMelFrontend().eval()
    per_clip = []
    with torch.inference_mode():
        for item in heldout:
            waveform = load_audio(resolve_audio_path(item, manifest))
            features = frontend(waveform)
            full_logits = model(features).squeeze(0)
            streamed_logits = model.streaming_forward(features).squeeze(0)
            stable_frames = len(full_logits) - model.lookahead_frames
            torch.testing.assert_close(
                full_logits[:stable_frames],
                streamed_logits,
                rtol=1e-5,
                atol=1e-5,
            )
            valid_targets = (
                len(features[0]) - LABEL_DELAY_FRAMES - MODEL_LOOKAHEAD_FRAMES
            )
            if valid_targets <= 0:
                raise ValueError(f"held-out clip {item['id']} has no valid frames")
            student = streamed_logits[
                LABEL_DELAY_FRAMES : LABEL_DELAY_FRAMES + valid_targets
            ].argmax(dim=-1)
            with np.load(target_dir / f"{item['id']}.npz") as target_file:
                own_teacher = torch.from_numpy(
                    target_file["teacher_probs"][:valid_targets].copy()
                ).argmax(dim=-1)
            with np.load(main_target_dir / f"{item['id']}.npz") as target_file:
                main_teacher = torch.from_numpy(
                    target_file["teacher_probs"][:valid_targets].copy()
                ).argmax(dim=-1)
            expected = LANGUAGE_CODES.index(item["language"])
            own_correct = int((student == own_teacher).sum())
            main_correct = int((student == main_teacher).sum())
            label_correct = int((student == expected).sum())
            teacher_label_correct = int((own_teacher == expected).sum())
            majority = int(
                torch.bincount(student, minlength=len(LANGUAGE_CODES)).argmax()
            )
            per_clip.append(
                {
                    "id": item["id"],
                    "language": item["language"],
                    "speaker_id": item["speaker_id"],
                    "frames": valid_targets,
                    "own_target_agreement_correct": own_correct,
                    "own_target_agreement": own_correct / valid_targets,
                    "main_teacher_agreement_correct": main_correct,
                    "main_teacher_agreement": main_correct / valid_targets,
                    "known_label_correct": label_correct,
                    "known_label_accuracy": label_correct / valid_targets,
                    "own_teacher_label_correct": teacher_label_correct,
                    "own_teacher_label_accuracy": (
                        teacher_label_correct / valid_targets
                    ),
                    "student_clip_prediction": LANGUAGE_CODES[majority],
                    "student_clip_correct": int(majority == expected),
                }
            )

    aggregate = aggregate_clip_metrics(per_clip)
    per_language = {}
    for language in LANGUAGE_CODES:
        selected = [record for record in per_clip if record["language"] == language]
        per_language[language] = aggregate_clip_metrics(selected)
    return {**aggregate, "per_language": per_language, "per_clip": per_clip}


def target_available_offset_ms(variant: str) -> float | None:
    if variant == "full_utterance":
        return None
    if variant == "centred_2s":
        return float(CENTRED_FUTURE_MS)
    if variant == "prefix_delta250":
        return float(EVIDENCE_LOOKAHEAD_MS)
    raise ValueError(variant)


def evaluate_switch_clip(
    model: CausalLIDStudent,
    item: dict,
    manifest: Path,
    target_dir: Path,
    variant: str,
) -> dict:
    if len(item["segments"]) != 2:
        raise ValueError(f"expected one boundary in {item['id']}")
    source_code = item["segments"][0]["language"]
    target_code = item["segments"][1]["language"]
    source_index = LANGUAGE_CODES.index(source_code)
    target_index = LANGUAGE_CODES.index(target_code)
    boundary_seconds = float(item["segments"][0]["end_seconds"])

    frontend = LogMelFrontend().eval()
    waveform = load_audio(resolve_audio_path(item, manifest))
    with torch.inference_mode():
        features = frontend(waveform)
        logits = model.streaming_forward(features).squeeze(0)
        probabilities = torch.softmax(logits, dim=-1).numpy()
    availability = chunk_availability_times(len(probabilities))
    chunks, chunk_times = chunk_posteriors(
        probabilities, availability, min_frame=LABEL_DELAY_FRAMES
    )
    raw_chunk_classes = chunks.argmax(axis=-1)
    ema_chunks = smooth_posteriors(chunks, new_weight=POLICY_EMA_NEW_WEIGHT)
    ema_chunk_classes = ema_chunks.argmax(axis=-1)
    raw_transition = persistent_transition(
        raw_chunk_classes,
        chunk_times,
        source_index,
        target_index,
        PERSISTENCE_CHUNKS,
    )
    ema_transition = persistent_transition(
        ema_chunk_classes,
        chunk_times,
        source_index,
        target_index,
        PERSISTENCE_CHUNKS,
    )
    policy = threshold_policy_transition(
        chunks, chunk_times, source_index, target_index
    )

    valid_targets = len(features[0]) - LABEL_DELAY_FRAMES - MODEL_LOOKAHEAD_FRAMES
    student_classes = probabilities[
        LABEL_DELAY_FRAMES : LABEL_DELAY_FRAMES + valid_targets
    ].argmax(axis=-1)
    expected = expected_switch_indices(item, valid_targets)
    target_times = (
        np.arange(valid_targets) * HOP_LENGTH + WIN_LENGTH
    ) / SAMPLE_RATE
    outside_collar = (
        np.abs(target_times - boundary_seconds) > BOUNDARY_COLLAR_MS / 1_000
    )
    with np.load(target_dir / f"{item['id']}.npz") as target_file:
        teacher_probs = target_file["teacher_probs"][:valid_targets].copy()
        anchor_frames = target_file["anchor_frames"].copy()
        anchor_probs = target_file["anchor_probs"].copy()
    teacher_classes = teacher_probs.argmax(axis=-1)

    anchor_times = (anchor_frames * HOP_LENGTH + WIN_LENGTH) / SAMPLE_RATE
    teacher_transition = persistent_transition(
        anchor_probs.argmax(axis=-1),
        anchor_times,
        source_index,
        target_index,
        PERSISTENCE_CHUNKS,
    )
    raw_matches_boundary = transition_matches_reference_boundary(
        raw_transition, boundary_seconds
    )
    ema_matches_boundary = transition_matches_reference_boundary(
        ema_transition, boundary_seconds
    )
    teacher_matches_boundary = transition_matches_reference_boundary(
        teacher_transition,
        boundary_seconds,
        early_tolerance_ms=(
            0.0 if variant == "full_utterance" else target_available_offset_ms(variant)
        ),
    )
    semantic_teacher_start = teacher_transition["transition_start_seconds"]
    semantic_teacher_confirmed = teacher_transition["transition_confirmed_seconds"]
    availability_offset_ms = target_available_offset_ms(variant)
    if variant == "full_utterance":
        first_teacher_posterior_available = (
            None if semantic_teacher_start is None else len(waveform) / SAMPLE_RATE
        )
        confirmed_teacher_transition_available = (
            None
            if semantic_teacher_confirmed is None
            else len(waveform) / SAMPLE_RATE
        )
    else:
        first_teacher_posterior_available = (
            None
            if semantic_teacher_start is None
            else semantic_teacher_start + availability_offset_ms / 1_000
        )
        confirmed_teacher_transition_available = (
            None
            if semantic_teacher_confirmed is None
            else semantic_teacher_confirmed + availability_offset_ms / 1_000
        )

    return {
        "id": item["id"],
        "source_language": source_code,
        "target_language": target_code,
        "true_switch_seconds": boundary_seconds,
        "frames": valid_targets,
        "known_label_accuracy": float(np.mean(student_classes == expected)),
        "known_label_accuracy_outside_250ms_collar": float(
            np.mean(student_classes[outside_collar] == expected[outside_collar])
        ),
        "own_target_agreement": float(
            np.mean(student_classes == teacher_classes)
        ),
        "teacher_known_label_accuracy": float(
            np.mean(teacher_classes == expected)
        ),
        "teacher_known_label_accuracy_outside_250ms_collar": float(
            np.mean(teacher_classes[outside_collar] == expected[outside_collar])
        ),
        "raw_persistent_transition": {
            **raw_transition,
            "matches_reference_boundary": raw_matches_boundary,
            "transition_start_lag_ms": safe_lag_ms(
                raw_transition["transition_start_seconds"], boundary_seconds
            ),
            "transition_confirmed_lag_ms": safe_lag_ms(
                raw_transition["transition_confirmed_seconds"], boundary_seconds
            ),
        },
        "ema_persistent_transition": {
            **ema_transition,
            "matches_reference_boundary": ema_matches_boundary,
            "transition_start_lag_ms": safe_lag_ms(
                ema_transition["transition_start_seconds"], boundary_seconds
            ),
            "transition_confirmed_lag_ms": safe_lag_ms(
                ema_transition["transition_confirmed_seconds"], boundary_seconds
            ),
        },
        "policy_transition": {
            **policy,
            "switch_commit_lag_ms": safe_lag_ms(
                policy["switch_commit_seconds"], boundary_seconds
            ),
        },
        "raw_chunk_flips": flip_metrics(
            raw_chunk_classes,
            chunk_times,
            matched_reference_transition=raw_matches_boundary,
        ),
        "ema_chunk_flips": flip_metrics(
            ema_chunk_classes,
            chunk_times,
            matched_reference_transition=ema_matches_boundary,
        ),
        "teacher_target_transition": {
            **teacher_transition,
            "matches_reference_boundary": teacher_matches_boundary,
            "semantic_transition_start_lag_ms": safe_lag_ms(
                semantic_teacher_start, boundary_seconds
            ),
            "semantic_transition_confirmed_lag_ms": safe_lag_ms(
                semantic_teacher_confirmed, boundary_seconds
            ),
            "future_context_or_full_clip_availability_ms": (
                "full clip" if availability_offset_ms is None else availability_offset_ms
            ),
            "first_target_posterior_available_seconds": (
                first_teacher_posterior_available
            ),
            "first_target_posterior_availability_lag_ms": safe_lag_ms(
                first_teacher_posterior_available, boundary_seconds
            ),
            "confirmed_transition_available_seconds": (
                confirmed_teacher_transition_available
            ),
            "confirmed_transition_availability_lag_ms": safe_lag_ms(
                confirmed_teacher_transition_available, boundary_seconds
            ),
            "anchor_flips": flip_metrics(
                anchor_probs.argmax(axis=-1),
                anchor_times,
                matched_reference_transition=teacher_matches_boundary,
            ),
        },
    }


def aggregate_switch_metrics(per_clip: list[dict]) -> dict:
    policy_lags = [
        record["policy_transition"]["switch_commit_lag_ms"]
        for record in per_clip
        if record["policy_transition"]["switch_commit_lag_ms"] is not None
    ]
    raw_lags = [
        record["raw_persistent_transition"]["transition_start_lag_ms"]
        for record in per_clip
        if record["raw_persistent_transition"]["matches_reference_boundary"]
    ]
    ema_lags = [
        record["ema_persistent_transition"]["transition_start_lag_ms"]
        for record in per_clip
        if record["ema_persistent_transition"]["matches_reference_boundary"]
    ]
    teacher_confirmed_lags = [
        record["teacher_target_transition"][
            "confirmed_transition_availability_lag_ms"
        ]
        for record in per_clip
        if record["teacher_target_transition"]["matches_reference_boundary"]
    ]
    return {
        "n_clips": len(per_clip),
        "known_label_accuracy_macro": float(
            np.mean([record["known_label_accuracy"] for record in per_clip])
        ),
        "known_label_accuracy_outside_250ms_collar_macro": float(
            np.mean(
                [
                    record["known_label_accuracy_outside_250ms_collar"]
                    for record in per_clip
                ]
            )
        ),
        "own_target_agreement_macro": float(
            np.mean([record["own_target_agreement"] for record in per_clip])
        ),
        "teacher_known_label_accuracy_macro": float(
            np.mean([record["teacher_known_label_accuracy"] for record in per_clip])
        ),
        "raw_transition_detected": len(raw_lags),
        "raw_transition_missed": len(per_clip) - len(raw_lags),
        "raw_transition_start_lag_ms_mean_detected_only": (
            None if not raw_lags else float(np.mean(raw_lags))
        ),
        "ema_transition_detected": len(ema_lags),
        "ema_transition_missed": len(per_clip) - len(ema_lags),
        "ema_transition_start_lag_ms_mean_detected_only": (
            None if not ema_lags else float(np.mean(ema_lags))
        ),
        "policy_transition_detected": len(policy_lags),
        "policy_transition_missed": len(per_clip) - len(policy_lags),
        "policy_switch_lag_ms_mean_detected_only": (
            None if not policy_lags else float(np.mean(policy_lags))
        ),
        "raw_unmatched_top1_changes_per_minute_mean": float(
            np.mean(
                [
                    record["raw_chunk_flips"][
                        "unmatched_top1_changes_per_minute"
                    ]
                    for record in per_clip
                ]
            )
        ),
        "ema_unmatched_top1_changes_per_minute_mean": float(
            np.mean(
                [
                    record["ema_chunk_flips"][
                        "unmatched_top1_changes_per_minute"
                    ]
                    for record in per_clip
                ]
            )
        ),
        "teacher_confirmed_transition_detected": len(teacher_confirmed_lags),
        "teacher_confirmed_transition_missed": (
            len(per_clip) - len(teacher_confirmed_lags)
        ),
        "teacher_confirmed_transition_availability_lag_ms_mean_detected_only": (
            None
            if not teacher_confirmed_lags
            else float(np.mean(teacher_confirmed_lags))
        ),
        "per_clip": per_clip,
    }


def evaluate_variant(
    model: CausalLIDStudent,
    variant: str,
    heldout: list[dict],
    switch_records: list[dict],
    manifest: Path,
    target_dir: Path,
    main_target_dir: Path,
) -> dict:
    model.eval()
    heldout_metrics = evaluate_heldout(
        model, heldout, manifest, target_dir, main_target_dir
    )
    switch_per_clip = [
        evaluate_switch_clip(model, item, manifest, target_dir, variant)
        for item in switch_records
    ]
    return {
        "heldout_monolingual": heldout_metrics,
        "heldout_switches": aggregate_switch_metrics(switch_per_clip),
    }


def compare_full_targets_to_main(
    heldout: list[dict], full_target_dir: Path, main_target_dir: Path
) -> dict:
    maximum_raw_difference = 0.0
    maximum_soft_difference = 0.0
    for item in heldout:
        with np.load(full_target_dir / f"{item['id']}.npz") as experiment_file:
            experiment_raw = experiment_file["teacher_probs"].copy()
            experiment_soft = experiment_file["teacher_soft_targets"].copy()
        with np.load(main_target_dir / f"{item['id']}.npz") as main_file:
            main_raw = main_file["teacher_probs"].copy()
            main_soft = main_file["teacher_soft_targets"].copy()
        maximum_raw_difference = max(
            maximum_raw_difference,
            float(np.max(np.abs(experiment_raw - main_raw))),
        )
        maximum_soft_difference = max(
            maximum_soft_difference,
            float(np.max(np.abs(experiment_soft - main_soft))),
        )
    return {
        "clips_compared": len(heldout),
        "max_abs_raw_posterior_difference": maximum_raw_difference,
        "max_abs_soft_target_difference": maximum_soft_difference,
    }


def build_comparisons(arms: dict[str, dict]) -> dict:
    heldout_macro = {
        variant: result["evaluation"]["heldout_monolingual"][
            "known_label_accuracy_macro"
        ]
        for variant, result in arms.items()
    }
    switch_accuracy = {
        variant: result["evaluation"]["heldout_switches"][
            "known_label_accuracy_outside_250ms_collar_macro"
        ]
        for variant, result in arms.items()
    }
    detections = {
        variant: result["evaluation"]["heldout_switches"][
            "policy_transition_detected"
        ]
        for variant, result in arms.items()
    }
    return {
        "heldout_known_label_macro_by_variant": heldout_macro,
        "heldout_known_label_macro_best": max(heldout_macro, key=heldout_macro.get),
        "switch_outside_collar_macro_by_variant": switch_accuracy,
        "switch_outside_collar_macro_best": max(
            switch_accuracy, key=switch_accuracy.get
        ),
        "policy_switch_detections_out_of_2_by_variant": detections,
        "context_valid_variants": [
            variant
            for variant in arms
            if target_definition(variant)["context_valid_at_tested_latency"]
        ],
    }


def main() -> None:
    args = parse_args()
    if args.threads != 6:
        raise ValueError("TEAM.md requires torch.set_num_threads(6)")
    if args.steps <= 0 or args.batch_size <= 0 or args.teacher_batch_size <= 0:
        raise ValueError("step and batch counts must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if not math.isclose(EVIDENCE_LOOKAHEAD_MS, 250.0):
        raise ValueError(
            f"experiment contract expects 250 ms evidence, got {EVIDENCE_LOOKAHEAD_MS}"
        )

    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    warnings.filterwarnings("ignore", message=".*custom_fwd.*deprecated.*")

    manifest = args.manifest.resolve()
    workspace = args.workspace.resolve()
    records = read_manifest(manifest)
    speaker_audit = require_speaker_disjoint(records)
    heldout = [item for item in records if item["split"] == "heldout"]
    switch_records = [item for item in records if item["split"] == "switch"]
    heldout_counts = Counter(item["language"] for item in heldout)
    if set(heldout_counts) != set(LANGUAGE_CODES) or len(set(heldout_counts.values())) != 1:
        raise ValueError(f"held-out set is not language-balanced: {heldout_counts}")
    if {item["id"] for item in switch_records} != {
        "switch_hi_en_eval",
        "switch_en_hi_eval",
    }:
        raise ValueError("expected the main pipeline's two held-out switch clips")

    all_input_sha256 = input_fingerprint(records, manifest)
    evaluation_input_sha256 = input_fingerprint(heldout + switch_records, manifest)
    bakeoff_compatible_sha256 = input_fingerprint(
        heldout
        + [item for item in switch_records if item["id"] == "switch_hi_en_eval"],
        manifest,
    )
    source_sha256 = PIPELINE_SOURCE_SHA256_AT_IMPORT
    driver_sha256 = driver_fingerprint()
    dependencies = dependency_versions()
    teacher_snapshot, teacher_identity = resolve_teacher_artifact()
    settings = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "teacher_batch_size": args.teacher_batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "threads": args.threads,
        "label_delay_frames": LABEL_DELAY_FRAMES,
        "model_lookahead_frames": MODEL_LOOKAHEAD_FRAMES,
        "evidence_lookahead_ms": EVIDENCE_LOOKAHEAD_MS,
        "early_ramp_frames": EARLY_RAMP_FRAMES,
    }
    run_identity_payload = {
        "experiment": EXPERIMENT_NAME,
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "all_input_sha256": all_input_sha256,
        "driver_sha256": driver_sha256,
        "pipeline_source_sha256": source_sha256,
        "settings": settings,
        "dependencies": dependencies,
        "teacher_identity": teacher_identity,
    }
    run_identity_sha256 = canonical_json_sha256(run_identity_payload)
    output_path = Path(__file__).with_name("results.json")
    reset_results = args.fresh or args.restart_results
    if output_path.exists() and not reset_results:
        output = json.loads(output_path.read_text(encoding="utf-8"))
        if output.get("run_identity_sha256") != run_identity_sha256:
            raise ValueError(
                "existing results have a different driver/pipeline/input/model/"
                "dependency identity; rerun with --fresh"
            )
    else:
        output = {"arms": {}}
    output.update(
        {
            "schema_version": SCHEMA_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "experiment": EXPERIMENT_NAME,
            "question": (
                "How do full-utterance, centred two-second, and growing-prefix "
                "teacher targets affect the same delayed causal TCN?"
            ),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest.relative_to(REPO_ROOT)),
            "all_input_sha256": all_input_sha256,
            "evaluation_input_sha256": evaluation_input_sha256,
            "teacher_bakeoff_compatible_input_sha256": bakeoff_compatible_sha256,
            "run_identity_sha256": run_identity_sha256,
            "run_identity": run_identity_payload,
            "driver_sha256": driver_sha256,
            "pipeline_source_sha256": source_sha256,
            "pipeline_source_files": list(PIPELINE_SOURCES),
            "pipeline_source_file_sha256": PIPELINE_SOURCE_FILE_SHA256_AT_IMPORT,
            "dependencies": dependencies,
            "teacher_identity": teacher_identity,
            "language_codes": list(LANGUAGE_CODES),
            "heldout_ids": [item["id"] for item in heldout],
            "switch_ids": [item["id"] for item in switch_records],
            "speaker_split": speaker_audit,
            "settings": settings,
            "target_definitions": {
                variant: target_definition(variant) for variant in VARIANTS
            },
            "policy": {
                "threshold": POLICY_THRESHOLD,
                "margin": POLICY_MARGIN,
                "margin_classes": "source versus target (matches main hi/en policy)",
                "dwell_chunks": POLICY_DWELL_CHUNKS,
                "ema_new_weight": POLICY_EMA_NEW_WEIGHT,
            },
            "boundary_collar_ms": BOUNDARY_COLLAR_MS,
            "arms": {} if reset_results else output.get("arms", {}),
        }
    )
    atomic_write_json(output_path, output)

    print(f"loading frozen teacher {TEACHER_NAME}", flush=True)
    from speechbrain.inference.classifiers import EncoderClassifier

    pinned_savedir = (
        args.model_dir.resolve()
        / f"{TEACHER_REVISION}-{teacher_identity['artifact_sha256'][:12]}"
    )
    teacher = EncoderClassifier.from_hparams(
        source=str(teacher_snapshot),
        savedir=str(pinned_savedir),
        run_opts={"device": "cpu"},
    )
    teacher.eval()
    teacher.hparams.label_encoder.ignore_len()
    selected_indices = label_indices(teacher)

    generation_summaries = {}
    for variant in VARIANTS:
        target_dir = workspace / "targets" / variant
        generation_summaries[variant] = generate_variant_targets(
            teacher,
            selected_indices,
            variant,
            records,
            manifest,
            target_dir,
            args.teacher_batch_size,
            all_input_sha256,
            run_identity_sha256,
            driver_sha256,
            source_sha256,
            teacher_identity,
            args.fresh,
        )
    del teacher

    full_parity = compare_full_targets_to_main(
        heldout,
        workspace / "targets" / "full_utterance",
        REPO_ROOT / "data/generated/targets",
    )
    output["validation"] = {
        "full_utterance_heldout_target_parity_with_main": full_parity,
    }
    atomic_write_json(output_path, output)

    for arm_number, variant in enumerate(VARIANTS, start=1):
        if variant in output["arms"] and not args.fresh:
            if (
                output["arms"][variant].get("run_identity_sha256")
                != run_identity_sha256
            ):
                raise ValueError(
                    f"completed arm {variant} has a different run identity"
                )
            print(f"[{arm_number}/3] retaining completed arm {variant}", flush=True)
            continue
        print(f"[{arm_number}/3] training {variant}", flush=True)
        target_dir = workspace / "targets" / variant
        model, training = train_variant(
            variant,
            manifest,
            target_dir,
            workspace / "checkpoints" / f"{variant}.pt",
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            run_identity_sha256=run_identity_sha256,
        )
        evaluation = evaluate_variant(
            model,
            variant,
            heldout,
            switch_records,
            manifest,
            target_dir,
            REPO_ROOT / "data/generated/targets",
        )
        output["arms"][variant] = {
            "analysis_version": ANALYSIS_VERSION,
            "run_identity_sha256": run_identity_sha256,
            "teacher_identity": teacher_identity,
            "target_generation": generation_summaries[variant],
            "training": training,
            "evaluation": evaluation,
        }
        output["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(output_path, output)
        heldout_accuracy = evaluation["heldout_monolingual"][
            "known_label_accuracy_macro"
        ]
        switches = evaluation["heldout_switches"]
        print(
            f"{variant}: heldout label macro={heldout_accuracy:.4f}; "
            f"policy switches={switches['policy_transition_detected']}/2; "
            f"switch outside-collar macro="
            f"{switches['known_label_accuracy_outside_250ms_collar_macro']:.4f}",
            flush=True,
        )
        del model

    initial_hashes = {
        result["training"]["initial_state_sha256"]
        for result in output["arms"].values()
    }
    batch_hashes = {
        result["training"]["batch_order_sha256"]
        for result in output["arms"].values()
    }
    output["validation"].update(
        {
            "identical_initial_state_across_arms": len(initial_hashes) == 1,
            "identical_batch_order_across_arms": len(batch_hashes) == 1,
            "all_training_finite": all(
                result["training"]["nan_free"]
                for result in output["arms"].values()
            ),
            "pipeline_launch_snapshot_bound": True,
            "pipeline_source_matches_launch_snapshot_at_end": (
                pipeline_source_fingerprint() == source_sha256
            ),
            "driver_unchanged_during_run": (
                driver_fingerprint() == driver_sha256
            ),
            "teacher_artifact_unchanged_during_run": (
                resolve_teacher_artifact()[1] == teacher_identity
            ),
            "inputs_unchanged_during_run": (
                input_fingerprint(records, manifest) == all_input_sha256
            ),
        }
    )
    if not all(
        output["validation"][key]
        for key in (
            "identical_initial_state_across_arms",
            "identical_batch_order_across_arms",
            "all_training_finite",
            "pipeline_launch_snapshot_bound",
            "driver_unchanged_during_run",
            "teacher_artifact_unchanged_during_run",
            "inputs_unchanged_during_run",
        )
    ):
        atomic_write_json(output_path, output)
        raise RuntimeError(f"experiment validation failed: {output['validation']}")
    output["comparisons"] = build_comparisons(output["arms"])
    output["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(output_path, output)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
