#!/usr/bin/env python3
"""Audit sparse ECAPA target hops against true 10 ms window calls.

This teacher-only experiment keeps the main pipeline's pinned ECAPA model,
current [t-1.75 s, t+0.25 s] window, seven-language restriction, held-out
clips, and timing coordinates.  It changes only how frequently the expensive
teacher is called, expanding sparse targets with causal previous-anchor hold.

Run from the repository root with::

    UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
      HF_HUB_OFFLINE=1 .venv/bin/python \
      experiments/teacher-anchor-hop-audit/run.py --fresh
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PIPELINE_SOURCES = (
    "scripts/teacher_targets.py",
    "src/streaming_lid/audio.py",
    "src/streaming_lid/config.py",
    "src/streaming_lid/data.py",
)


def hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_pipeline_sources() -> tuple[str, dict[str, str]]:
    combined = hashlib.sha256()
    per_file: dict[str, str] = {}
    for relative in PIPELINE_SOURCES:
        contents = (REPO_ROOT / relative).read_bytes()
        per_file[relative] = hashlib.sha256(contents).hexdigest()
        combined.update(relative.encode("utf-8"))
        combined.update(contents)
    return combined.hexdigest(), per_file


(
    PIPELINE_SOURCE_SHA256_AT_IMPORT,
    PIPELINE_SOURCE_FILE_SHA256_AT_IMPORT,
) = snapshot_pipeline_sources()

# Import, rather than copy, the production teacher artifact checks, exact
# window extractor, label map, audio loader, manifest parser, and constants.
from scripts.teacher_targets import (  # noqa: E402
    expected_language,
    extract_window,
    label_indices,
    resolve_teacher_artifact,
)
from streaming_lid.audio import feature_frame_count, load_audio  # noqa: E402
from streaming_lid.config import (  # noqa: E402
    FRAME_MS,
    HOP_LENGTH,
    LANGUAGE_CODES,
    SAMPLE_RATE,
    TEACHER_ARTIFACT_SHA256,
    TEACHER_FUTURE_MS,
    TEACHER_HOP_FRAMES,
    TEACHER_NAME,
    TEACHER_PAST_MS,
    TEACHER_REVISION,
    WIN_LENGTH,
)
from streaming_lid.data import (  # noqa: E402
    read_manifest,
    require_speaker_disjoint,
    resolve_audio_path,
)


EXPERIMENT_NAME = "teacher-anchor-hop-audit"
SCHEMA_VERSION = 1
ANALYSIS_VERSION = "teacher-anchor-hop-audit-v1"
HOP_FRAMES = (25, 10, 5, 1)
DENSE_HOP_FRAMES = 1
BOUNDARY_COLLAR_MS = 250
STABILITY_HORIZON_MS = 500
KL_EPSILON = 1e-8
GATE_P95_KL = 0.10
GATE_NONCOLLAR_DISAGREEMENT = 0.02
GATE_CROSSING_DELTA_MS = 50
GATE_EXTRA_FLIPS_PER_CLIP = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/manifest.jsonl"),
    )
    parser.add_argument(
        "--main-target-dir",
        type=Path,
        default=Path("data/generated/targets"),
        help="Main cache, used only for input binding and parity checks.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".cache/models/lang-id-voxlingua107-ecapa"),
    )
    parser.add_argument("--teacher-batch-size", type=int, default=24)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Replace results.json after every identity check passes.",
    )
    return parser.parse_args()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def input_fingerprint(records: list[dict], manifest: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(records, key=lambda value: value["id"]):
        digest.update(
            json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        audio_path = resolve_audio_path(item, manifest)
        digest.update(audio_path.name.encode("utf-8"))
        digest.update(bytes.fromhex(hash_path(audio_path)))
    return digest.hexdigest()


def target_files_fingerprint(records: list[dict], target_dir: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(records, key=lambda value: value["id"]):
        path = target_dir / f"{item['id']}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing main target cache file: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(hash_path(path)))
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchaudio": importlib.metadata.version("torchaudio"),
        "speechbrain": importlib.metadata.version("speechbrain"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
    }


def anchor_frames(num_frames: int, hop_frames: int) -> np.ndarray:
    anchors = np.arange(0, num_frames, hop_frames, dtype=np.int64)
    if len(anchors) == 0:
        raise ValueError("audio has no feature frames")
    if anchors[-1] != num_frames - 1:
        anchors = np.append(anchors, num_frames - 1)
    if np.any(np.diff(anchors) <= 0):
        raise AssertionError("teacher anchors must increase strictly")
    return anchors


def classify_anchors(
    teacher: torch.nn.Module,
    waveform: torch.Tensor,
    anchors: np.ndarray,
    selected_indices: list[int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_logs: list[torch.Tensor] = []
    selected_masses: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(anchors), batch_size):
            batch_anchors = anchors[start : start + batch_size]
            windows = torch.stack(
                [extract_window(waveform, int(frame)) for frame in batch_anchors]
            )
            log_probabilities, _, _, _ = teacher.classify_batch(windows)
            chosen = log_probabilities[:, selected_indices]
            selected_logs.append(chosen.cpu())
            selected_masses.append(chosen.exp().sum(dim=-1).cpu())

    logs = torch.cat(selected_logs)
    probabilities = torch.softmax(logs, dim=-1).numpy().astype(np.float32)
    masses = torch.cat(selected_masses).numpy().astype(np.float32)
    expected_shape = (len(anchors), len(LANGUAGE_CODES))
    if probabilities.shape != expected_shape:
        raise AssertionError(
            f"unexpected teacher output shape {probabilities.shape}, "
            f"expected {expected_shape}"
        )
    if not np.isfinite(probabilities).all() or not np.isfinite(masses).all():
        raise FloatingPointError("teacher produced a non-finite value")
    if np.any(masses < 0) or np.any(masses > 1 + 1e-5):
        raise ValueError("teacher selected-language mass lies outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6):
        raise AssertionError("restricted teacher posterior is not normalized")
    return probabilities, masses


def previous_anchor_hold(
    anchors: np.ndarray, values: np.ndarray, num_frames: int
) -> np.ndarray:
    frame_indices = np.arange(num_frames, dtype=np.int64)
    selected = np.searchsorted(anchors, frame_indices, side="right") - 1
    selected = np.clip(selected, 0, len(anchors) - 1)
    if np.any(anchors[selected] > frame_indices):
        raise AssertionError("previous-anchor hold consulted a future anchor")
    held = values[selected]
    if not np.array_equal(held[anchors], values):
        raise AssertionError("previous-anchor hold changed an exact anchor")
    return held


def frame_times(num_frames: int) -> np.ndarray:
    return (
        np.arange(num_frames, dtype=np.float64) * HOP_LENGTH + WIN_LENGTH
    ) / SAMPLE_RATE


def expected_indices(item: dict, times: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            LANGUAGE_CODES.index(expected_language(item, float(seconds)))
            for seconds in times
        ],
        dtype=np.int64,
    )


def percentile(values: np.ndarray | list[float], quantile: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.percentile(array, quantile))


def stable_event(
    classes: np.ndarray,
    times: np.ndarray,
    *,
    source_index: int,
    target_index: int,
    boundary_seconds: float,
) -> dict:
    """Find a direction-aware target run stable for 500 ms."""

    horizon_seconds = STABILITY_HORIZON_MS / 1_000

    def run_end(start: int, wanted: int) -> int | None:
        if classes[start] != wanted:
            return None
        end = start
        while end + 1 < len(classes) and classes[end + 1] == wanted:
            end += 1
            if times[end] - times[start] + 1e-9 >= horizon_seconds:
                break
        if times[end] - times[start] + 1e-9 < horizon_seconds:
            return None
        return end

    source_armed = False
    source_confirmation_seconds = None
    for start in range(len(classes)):
        end = run_end(start, source_index)
        if end is not None and times[end] <= boundary_seconds + 1e-9:
            source_armed = True
            source_confirmation_seconds = float(times[end])
            break

    onset = None
    confirmation = None
    premature_target_onset = None
    permitted_early_seconds = TEACHER_FUTURE_MS / 1_000
    for start in range(len(classes)):
        if classes[start] != target_index:
            continue
        end = run_end(start, target_index)
        if end is None:
            continue
        if times[start] < boundary_seconds - permitted_early_seconds - 1e-9:
            if premature_target_onset is None:
                premature_target_onset = float(times[start])
            continue
        if not source_armed:
            continue
        onset = float(times[start])
        confirmation = float(times[end])
        break

    return {
        "stability_horizon_ms": STABILITY_HORIZON_MS,
        "source_armed": source_armed,
        "source_confirmation_seconds": source_confirmation_seconds,
        "detected": onset is not None,
        "premature_target_onset_seconds": premature_target_onset,
        "semantic_onset_seconds": onset,
        "semantic_onset_lag_ms": (
            None if onset is None else 1_000 * (onset - boundary_seconds)
        ),
        "onset_available_seconds": (
            None if onset is None else onset + TEACHER_FUTURE_MS / 1_000
        ),
        "onset_availability_lag_ms": (
            None
            if onset is None
            else 1_000
            * (onset + TEACHER_FUTURE_MS / 1_000 - boundary_seconds)
        ),
        "semantic_confirmation_seconds": confirmation,
        "semantic_confirmation_lag_ms": (
            None
            if confirmation is None
            else 1_000 * (confirmation - boundary_seconds)
        ),
        "confirmation_available_seconds": (
            None
            if confirmation is None
            else confirmation + TEACHER_FUTURE_MS / 1_000
        ),
        "confirmation_availability_lag_ms": (
            None
            if confirmation is None
            else 1_000
            * (confirmation + TEACHER_FUTURE_MS / 1_000 - boundary_seconds)
        ),
    }


def evaluate_clip(
    item: dict,
    anchors: np.ndarray,
    probabilities: np.ndarray,
    masses: np.ndarray,
    num_frames: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    if len(item["segments"]) != 2:
        raise ValueError(f"expected exactly one boundary in {item['id']}")
    source_code = item["segments"][0]["language"]
    target_code = item["segments"][1]["language"]
    source_index = LANGUAGE_CODES.index(source_code)
    target_index = LANGUAGE_CODES.index(target_code)
    boundary_seconds = float(item["segments"][0]["end_seconds"])
    times = frame_times(num_frames)
    expected = expected_indices(item, times)
    outside_collar = (
        np.abs(times - boundary_seconds) > BOUNDARY_COLLAR_MS / 1_000
    )
    dense_probabilities = previous_anchor_hold(anchors, probabilities, num_frames)
    dense_masses = previous_anchor_hold(anchors, masses, num_frames)
    classes = dense_probabilities.argmax(axis=-1)
    event = stable_event(
        classes,
        times,
        source_index=source_index,
        target_index=target_index,
        boundary_seconds=boundary_seconds,
    )
    changes = int(np.sum(classes[1:] != classes[:-1]))
    unmatched_changes = max(0, changes - int(event["detected"]))
    anchor_classes = probabilities.argmax(axis=-1)
    anchor_times = (anchors * HOP_LENGTH + WIN_LENGTH) / SAMPLE_RATE
    traces = []
    for index, anchor in enumerate(anchors):
        traces.append(
            {
                "anchor_frame": int(anchor),
                "semantic_seconds": float(anchor_times[index]),
                "available_seconds": float(
                    min(
                        item["duration_seconds"],
                        anchor_times[index] + TEACHER_FUTURE_MS / 1_000,
                    )
                ),
                "expected_language": expected_language(
                    item, float(anchor_times[index])
                ),
                "predicted_language": LANGUAGE_CODES[int(anchor_classes[index])],
                "selected_confidence": float(probabilities[index].max()),
                "in_set_mass": float(masses[index]),
                "probabilities": [float(value) for value in probabilities[index]],
            }
        )

    record = {
        "id": item["id"],
        "source_language": source_code,
        "target_language": target_code,
        "boundary_seconds": boundary_seconds,
        "frames": num_frames,
        "anchors": len(anchors),
        "known_label_correct": int(np.sum(classes == expected)),
        "known_label_accuracy": float(np.mean(classes == expected)),
        "outside_collar_frames": int(outside_collar.sum()),
        "known_label_correct_outside_250ms_collar": int(
            np.sum(classes[outside_collar] == expected[outside_collar])
        ),
        "known_label_accuracy_outside_250ms_collar": float(
            np.mean(classes[outside_collar] == expected[outside_collar])
        ),
        "source_segment_recall_outside_collar": float(
            np.mean(
                classes[(times < boundary_seconds) & outside_collar]
                == source_index
            )
        ),
        "target_segment_recall_outside_collar": float(
            np.mean(
                classes[(times >= boundary_seconds) & outside_collar]
                == target_index
            )
        ),
        "mean_in_set_mass_anchor_weighted": float(masses.mean()),
        "mean_in_set_mass_frame_weighted": float(dense_masses.mean()),
        "minimum_in_set_mass": float(masses.min()),
        "p05_in_set_mass": percentile(masses, 5),
        "stable_event": event,
        "top1_changes": changes,
        "unmatched_top1_changes": unmatched_changes,
        "posterior_sha256": hashlib.sha256(
            probabilities.tobytes() + masses.tobytes()
        ).hexdigest(),
        "anchor_trace": traces,
    }
    arrays = {
        "probabilities": dense_probabilities,
        "masses": dense_masses,
        "classes": classes,
        "outside_collar": outside_collar,
    }
    return record, arrays


def aggregate_arm(per_clip: list[dict], timing: dict) -> dict:
    frames = sum(record["frames"] for record in per_clip)
    correct = sum(record["known_label_correct"] for record in per_clip)
    outside_frames = sum(record["outside_collar_frames"] for record in per_clip)
    outside_correct = sum(
        record["known_label_correct_outside_250ms_collar"]
        for record in per_clip
    )
    detected = [
        record["stable_event"]
        for record in per_clip
        if record["stable_event"]["detected"]
    ]
    semantic_lags = [event["semantic_onset_lag_ms"] for event in detected]
    availability_lags = [
        event["onset_availability_lag_ms"] for event in detected
    ]
    confirmation_lags = [
        event["confirmation_availability_lag_ms"] for event in detected
    ]
    return {
        "switch_clips": len(per_clip),
        "frames": frames,
        "known_label_accuracy_micro": correct / frames,
        "known_label_accuracy_macro": float(
            np.mean([record["known_label_accuracy"] for record in per_clip])
        ),
        "known_label_accuracy_outside_250ms_collar_micro": (
            outside_correct / outside_frames
        ),
        "known_label_accuracy_outside_250ms_collar_macro": float(
            np.mean(
                [
                    record["known_label_accuracy_outside_250ms_collar"]
                    for record in per_clip
                ]
            )
        ),
        "source_segment_recall_outside_collar_macro": float(
            np.mean(
                [
                    record["source_segment_recall_outside_collar"]
                    for record in per_clip
                ]
            )
        ),
        "target_segment_recall_outside_collar_macro": float(
            np.mean(
                [
                    record["target_segment_recall_outside_collar"]
                    for record in per_clip
                ]
            )
        ),
        "stable_event": {
            "stability_horizon_ms": STABILITY_HORIZON_MS,
            "detected": len(detected),
            "missed": len(per_clip) - len(detected),
            "semantic_onset_lag_ms_median_detected_only": percentile(
                semantic_lags, 50
            ),
            "semantic_onset_lag_ms_p95_detected_only": percentile(
                semantic_lags, 95
            ),
            "onset_availability_lag_ms_median_detected_only": percentile(
                availability_lags, 50
            ),
            "confirmation_availability_lag_ms_median_detected_only": percentile(
                confirmation_lags, 50
            ),
        },
        "top1_changes_total": sum(record["top1_changes"] for record in per_clip),
        "unmatched_top1_changes_total": sum(
            record["unmatched_top1_changes"] for record in per_clip
        ),
        "mean_in_set_mass_anchor_weighted_macro": float(
            np.mean(
                [record["mean_in_set_mass_anchor_weighted"] for record in per_clip]
            )
        ),
        "mean_in_set_mass_frame_weighted_macro": float(
            np.mean(
                [record["mean_in_set_mass_frame_weighted"] for record in per_clip]
            )
        ),
        "generation_timing": timing,
        "per_clip": per_clip,
    }


def compare_clip_to_dense(
    record: dict,
    arrays: dict[str, np.ndarray],
    dense_record: dict,
    dense_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    probabilities = np.clip(arrays["probabilities"], KL_EPSILON, 1.0)
    dense_probabilities = np.clip(
        dense_arrays["probabilities"], KL_EPSILON, 1.0
    )
    kl = np.sum(
        dense_probabilities * np.log(dense_probabilities / probabilities), axis=-1
    )
    disagreement = arrays["classes"] != dense_arrays["classes"]
    outside = arrays["outside_collar"]
    mass_absolute_difference = np.abs(arrays["masses"] - dense_arrays["masses"])
    event = record["stable_event"]
    dense_event = dense_record["stable_event"]
    if event["detected"] and dense_event["detected"]:
        crossing_delta_ms = (
            event["semantic_onset_lag_ms"]
            - dense_event["semantic_onset_lag_ms"]
        )
        crossing_absolute_delta_ms = abs(crossing_delta_ms)
    else:
        crossing_delta_ms = None
        crossing_absolute_delta_ms = None
    comparison = {
        "id": record["id"],
        "frames": len(kl),
        "kl_dense_to_expanded_mean": float(kl.mean()),
        "kl_dense_to_expanded_median": percentile(kl, 50),
        "kl_dense_to_expanded_p95": percentile(kl, 95),
        "kl_dense_to_expanded_max": float(kl.max()),
        "argmax_disagreements": int(disagreement.sum()),
        "argmax_disagreement_rate": float(disagreement.mean()),
        "noncollar_frames": int(outside.sum()),
        "noncollar_argmax_disagreements": int(disagreement[outside].sum()),
        "noncollar_argmax_disagreement_rate": float(
            disagreement[outside].mean()
        ),
        "in_set_mass_absolute_difference_mean": float(
            mass_absolute_difference.mean()
        ),
        "in_set_mass_absolute_difference_p95": percentile(
            mass_absolute_difference, 95
        ),
        "in_set_mass_absolute_difference_max": float(
            mass_absolute_difference.max()
        ),
        "dense_stable_event_detected": dense_event["detected"],
        "expanded_stable_event_detected": event["detected"],
        "stable_onset_delta_ms_vs_dense": crossing_delta_ms,
        "stable_onset_absolute_delta_ms_vs_dense": crossing_absolute_delta_ms,
        "top1_changes_delta_vs_dense": (
            record["top1_changes"] - dense_record["top1_changes"]
        ),
        "unmatched_top1_changes_delta_vs_dense": (
            record["unmatched_top1_changes"]
            - dense_record["unmatched_top1_changes"]
        ),
    }
    comparison_arrays = {
        "kl": kl,
        "disagreement": disagreement,
        "outside_collar": outside,
        "mass_absolute_difference": mass_absolute_difference,
    }
    return comparison, comparison_arrays


def aggregate_comparison(
    per_clip: list[dict], comparison_arrays: list[dict[str, np.ndarray]]
) -> dict:
    kl = np.concatenate([values["kl"] for values in comparison_arrays])
    disagreement = np.concatenate(
        [values["disagreement"] for values in comparison_arrays]
    )
    outside = np.concatenate(
        [values["outside_collar"] for values in comparison_arrays]
    )
    mass_difference = np.concatenate(
        [values["mass_absolute_difference"] for values in comparison_arrays]
    )
    return {
        "reference": "true_10ms_teacher_calls",
        "frames": len(kl),
        "kl_dense_to_expanded_mean": float(kl.mean()),
        "kl_dense_to_expanded_median": percentile(kl, 50),
        "kl_dense_to_expanded_p95": percentile(kl, 95),
        "kl_dense_to_expanded_max": float(kl.max()),
        "argmax_disagreements": int(disagreement.sum()),
        "argmax_disagreement_rate": float(disagreement.mean()),
        "noncollar_frames": int(outside.sum()),
        "noncollar_argmax_disagreements": int(disagreement[outside].sum()),
        "noncollar_argmax_disagreement_rate": float(
            disagreement[outside].mean()
        ),
        "in_set_mass_absolute_difference_mean": float(mass_difference.mean()),
        "in_set_mass_absolute_difference_p95": percentile(mass_difference, 95),
        "in_set_mass_absolute_difference_max": float(mass_difference.max()),
        "all_dense_stable_events_detected": all(
            record["dense_stable_event_detected"] for record in per_clip
        ),
        "all_expanded_stable_events_detected": all(
            record["expanded_stable_event_detected"] for record in per_clip
        ),
        "maximum_stable_onset_absolute_delta_ms_vs_dense": max(
            (
                record["stable_onset_absolute_delta_ms_vs_dense"]
                for record in per_clip
                if record["stable_onset_absolute_delta_ms_vs_dense"] is not None
            ),
            default=None,
        ),
        "maximum_extra_top1_changes_per_clip_vs_dense": max(
            record["top1_changes_delta_vs_dense"] for record in per_clip
        ),
        "per_clip": per_clip,
    }


def evaluate_monolingual_full_utterance(
    teacher: torch.nn.Module,
    selected_indices: list[int],
    heldout: list[dict],
    manifest: Path,
    main_target_dir: Path,
) -> dict:
    records = []
    fresh_correct = 0
    main_correct = 0
    maximum_difference = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for item in heldout:
            waveform = load_audio(resolve_audio_path(item, manifest))
            log_probabilities, _, _, _ = teacher.classify_batch(
                waveform.unsqueeze(0)
            )
            chosen = log_probabilities[0, selected_indices].cpu()
            posterior = torch.softmax(chosen, dim=-1).numpy()
            with np.load(
                main_target_dir / f"{item['id']}.npz", allow_pickle=False
            ) as target_file:
                codes = tuple(str(code) for code in target_file["language_codes"])
                if codes != LANGUAGE_CODES:
                    raise ValueError(
                        f"main target language order differs for {item['id']}"
                    )
                main_posterior = target_file["teacher_probs"][0].copy()
            expected_index = LANGUAGE_CODES.index(item["language"])
            prediction = int(posterior.argmax())
            main_prediction = int(main_posterior.argmax())
            fresh_correct += int(prediction == expected_index)
            main_correct += int(main_prediction == expected_index)
            difference = float(np.max(np.abs(posterior - main_posterior)))
            maximum_difference = max(maximum_difference, difference)
            records.append(
                {
                    "id": item["id"],
                    "language": item["language"],
                    "prediction": LANGUAGE_CODES[prediction],
                    "main_cache_prediction": LANGUAGE_CODES[main_prediction],
                    "correct": prediction == expected_index,
                    "main_cache_correct": main_prediction == expected_index,
                    "max_abs_posterior_difference_from_main_cache": difference,
                }
            )
    return {
        "clips": len(heldout),
        "fresh_correct": fresh_correct,
        "main_cache_correct": main_correct,
        "fresh_full_utterance_accuracy": fresh_correct / len(heldout),
        "main_cache_full_utterance_accuracy": main_correct / len(heldout),
        "max_abs_posterior_difference_from_main_cache": maximum_difference,
        "wall_seconds": time.perf_counter() - started,
        "per_clip": records,
    }


def compare_250ms_to_main(
    raw_arm: dict[str, dict[str, Any]],
    switches: list[dict],
    main_target_dir: Path,
) -> dict:
    per_clip = []
    maximum_probability_difference = 0.0
    maximum_mass_difference = 0.0
    all_frames_equal = True
    for item in switches:
        values = raw_arm[item["id"]]
        with np.load(
            main_target_dir / f"{item['id']}.npz", allow_pickle=False
        ) as target_file:
            main_frames = target_file["anchor_frames"].copy()
            main_probabilities = target_file["anchor_probs"].copy()
            main_masses = target_file["in_set_mass"].copy()
        frames_equal = np.array_equal(values["anchors"], main_frames)
        if values["probabilities"].shape != main_probabilities.shape:
            raise ValueError(f"main 250 ms anchor shape differs for {item['id']}")
        probability_difference = float(
            np.max(np.abs(values["probabilities"] - main_probabilities))
        )
        mass_difference = float(np.max(np.abs(values["masses"] - main_masses)))
        all_frames_equal = all_frames_equal and frames_equal
        maximum_probability_difference = max(
            maximum_probability_difference, probability_difference
        )
        maximum_mass_difference = max(maximum_mass_difference, mass_difference)
        per_clip.append(
            {
                "id": item["id"],
                "anchor_frames_equal": frames_equal,
                "max_abs_anchor_posterior_difference": probability_difference,
                "max_abs_in_set_mass_difference": mass_difference,
            }
        )
    return {
        "clips": len(switches),
        "anchor_frames_equal": all_frames_equal,
        "max_abs_anchor_posterior_difference": maximum_probability_difference,
        "max_abs_in_set_mass_difference": maximum_mass_difference,
        "exact_match": (
            all_frames_equal
            and maximum_probability_difference == 0.0
            and maximum_mass_difference == 0.0
        ),
        "per_clip": per_clip,
    }


def decide(arms: dict[str, dict]) -> dict:
    comparisons: dict[str, dict] = {}
    passing_sparse_hops: list[int] = []
    for hop_frames in HOP_FRAMES:
        key = f"hop_{int(hop_frames * FRAME_MS)}ms"
        comparison = arms[key]["comparison_to_10ms"]
        stable_delta = comparison[
            "maximum_stable_onset_absolute_delta_ms_vs_dense"
        ]
        checks = {
            "p95_kl_le_0_10": (
                comparison["kl_dense_to_expanded_p95"] <= GATE_P95_KL
            ),
            "noncollar_argmax_disagreement_le_2pct": (
                comparison["noncollar_argmax_disagreement_rate"]
                <= GATE_NONCOLLAR_DISAGREEMENT
            ),
            "dense_detects_both_stable_switches": (
                comparison["all_dense_stable_events_detected"]
            ),
            "expanded_detects_both_stable_switches": (
                comparison["all_expanded_stable_events_detected"]
            ),
            "every_stable_crossing_within_50ms_of_dense": (
                stable_delta is not None and stable_delta <= GATE_CROSSING_DELTA_MS
            ),
            "no_more_than_one_extra_flip_per_clip": (
                comparison["maximum_extra_top1_changes_per_clip_vs_dense"]
                <= GATE_EXTRA_FLIPS_PER_CLIP
            ),
        }
        passed = all(checks.values())
        hop_ms = int(hop_frames * FRAME_MS)
        if hop_frames != DENSE_HOP_FRAMES and passed:
            passing_sparse_hops.append(hop_ms)
        comparisons[key] = {
            "checks": checks,
            "passes_shortlist_gate": passed,
        }

    selected_hop_ms = max(passing_sparse_hops, default=None)
    if selected_hop_ms is None:
        verdict = "reject"
        reason = (
            "No sparse previous-hold grid passed every fidelity, crossing, "
            "and stability gate against true 10 ms calls."
        )
    else:
        verdict = "adopt"
        reason = (
            f"{selected_hop_ms} ms is the coarsest sparse grid that passed "
            "every predeclared gate against true 10 ms calls."
        )

    return {
        "gate_definition": {
            "reference_hop_ms": int(DENSE_HOP_FRAMES * FRAME_MS),
            "maximum_pooled_p95_kl_dense_to_expanded": GATE_P95_KL,
            "maximum_pooled_noncollar_argmax_disagreement_rate": (
                GATE_NONCOLLAR_DISAGREEMENT
            ),
            "maximum_absolute_500ms_stable_crossing_delta_ms_per_clip": (
                GATE_CROSSING_DELTA_MS
            ),
            "maximum_extra_top1_flips_per_clip": GATE_EXTRA_FLIPS_PER_CLIP,
            "selection_rule": "coarsest sparse hop passing every gate",
        },
        "comparisons": comparisons,
        "passing_sparse_hops_ms": sorted(passing_sparse_hops, reverse=True),
        "selected_hop_ms": selected_hop_ms,
        "verdict": verdict,
        "reason": reason,
        "scope": (
            "A positive result selects a target-generation hop for a later "
            "identity-valid student experiment; it does not modify the pipeline."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.threads != 6:
        raise ValueError("TEAM.md requires exactly six Torch threads")
    if args.teacher_batch_size <= 0:
        raise ValueError("teacher batch size must be positive")
    if TEACHER_HOP_FRAMES != 25 or FRAME_MS != 10:
        raise ValueError("experiment expects the main 250 ms / 10 ms timing grid")
    if (TEACHER_PAST_MS, TEACHER_FUTURE_MS) != (1_750, 250):
        raise ValueError("experiment expects the main 1,750/250 ms window")
    if TEACHER_ARTIFACT_SHA256 != (
        "f193a0548951e8fbd6ca438a492b98b5c48bf7d98c86451ca3878dd4a7c0f706"
    ):
        raise ValueError("unexpected main teacher artifact")

    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    warnings.filterwarnings("ignore", message=".*custom_fwd.*deprecated.*")

    manifest = args.manifest.resolve()
    main_target_dir = args.main_target_dir.resolve()
    output_path = Path(__file__).with_name("results.json")
    if output_path.exists() and not args.fresh:
        raise FileExistsError(f"{output_path} exists; pass --fresh to replace it")

    records = read_manifest(manifest)
    speaker_audit = require_speaker_disjoint(records)
    heldout = [item for item in records if item["split"] == "heldout"]
    switches = [item for item in records if item["split"] == "switch"]
    if Counter(item["language"] for item in heldout) != Counter(
        {code: 3 for code in LANGUAGE_CODES}
    ):
        raise ValueError("held-out monolingual clips do not match the main split")
    if {item["id"] for item in switches} != {
        "switch_hi_en_eval",
        "switch_en_hi_eval",
    }:
        raise ValueError("switch clips do not match the main evaluation split")

    evaluation_records = heldout + switches
    evaluation_input_sha256 = input_fingerprint(evaluation_records, manifest)
    main_target_sha256 = target_files_fingerprint(
        evaluation_records, main_target_dir
    )
    driver_sha256 = hash_path(Path(__file__).resolve())
    teacher_snapshot, teacher_identity = resolve_teacher_artifact()
    if teacher_identity["artifact_sha256"] != TEACHER_ARTIFACT_SHA256:
        raise ValueError("resolved teacher differs from the main pinned artifact")

    run_identity = {
        "experiment": EXPERIMENT_NAME,
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "evaluation_input_sha256": evaluation_input_sha256,
        "main_evaluation_target_files_sha256": main_target_sha256,
        "driver_sha256": driver_sha256,
        "pipeline_source_sha256": PIPELINE_SOURCE_SHA256_AT_IMPORT,
        "teacher_identity": teacher_identity,
        "dependencies": dependency_versions(),
        "settings": {
            "threads": args.threads,
            "teacher_batch_size": args.teacher_batch_size,
            "hop_frames": list(HOP_FRAMES),
            "hop_ms": [int(value * FRAME_MS) for value in HOP_FRAMES],
            "dense_reference_hop_ms": int(DENSE_HOP_FRAMES * FRAME_MS),
            "teacher_past_ms": TEACHER_PAST_MS,
            "teacher_future_ms": TEACHER_FUTURE_MS,
            "anchor_expansion": "previous_anchor_hold",
            "boundary_collar_ms": BOUNDARY_COLLAR_MS,
            "stability_horizon_ms": STABILITY_HORIZON_MS,
            "gate_p95_kl": GATE_P95_KL,
            "gate_noncollar_argmax_disagreement": (
                GATE_NONCOLLAR_DISAGREEMENT
            ),
            "gate_crossing_delta_ms": GATE_CROSSING_DELTA_MS,
            "gate_extra_flips_per_clip": GATE_EXTRA_FLIPS_PER_CLIP,
        },
    }
    run_identity_sha256 = canonical_json_sha256(run_identity)

    print(
        f"loading frozen teacher {TEACHER_NAME}@{TEACHER_REVISION[:8]}",
        flush=True,
    )
    from speechbrain.inference.classifiers import EncoderClassifier

    pinned_savedir = (
        args.model_dir.resolve()
        / f"{TEACHER_REVISION}-{TEACHER_ARTIFACT_SHA256[:12]}"
    )
    teacher = EncoderClassifier.from_hparams(
        source=str(teacher_snapshot),
        savedir=str(pinned_savedir),
        overrides={"pretrained_path": str(teacher_snapshot)},
        run_opts={"device": "cpu"},
    )
    teacher.eval()
    teacher.hparams.label_encoder.ignore_len()
    selected_indices = label_indices(teacher)

    monolingual = evaluate_monolingual_full_utterance(
        teacher,
        selected_indices,
        heldout,
        manifest,
        main_target_dir,
    )
    print(
        "full-utterance held-out sanity: "
        f"{monolingual['fresh_correct']}/{monolingual['clips']}",
        flush=True,
    )

    waveforms = {
        item["id"]: load_audio(resolve_audio_path(item, manifest))
        for item in switches
    }
    raw_arms: dict[str, dict[str, dict[str, Any]]] = {}
    arms: dict[str, dict] = {}
    for arm_number, hop_frames in enumerate(HOP_FRAMES, start=1):
        hop_ms = int(hop_frames * FRAME_MS)
        key = f"hop_{hop_ms}ms"
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        request_count = 0
        request_audio_seconds = 0.0
        raw_per_clip: dict[str, dict[str, Any]] = {}
        per_clip = []
        arrays_per_clip: dict[str, dict[str, np.ndarray]] = {}
        for item in switches:
            waveform = waveforms[item["id"]]
            num_frames = feature_frame_count(len(waveform))
            anchors = anchor_frames(num_frames, hop_frames)
            probabilities, masses = classify_anchors(
                teacher,
                waveform,
                anchors,
                selected_indices,
                args.teacher_batch_size,
            )
            request_count += len(anchors)
            request_audio_seconds += (
                len(anchors)
                * (TEACHER_PAST_MS + TEACHER_FUTURE_MS)
                / 1_000
            )
            record, arrays = evaluate_clip(
                item,
                anchors,
                probabilities,
                masses,
                num_frames,
            )
            per_clip.append(record)
            arrays_per_clip[item["id"]] = arrays
            raw_per_clip[item["id"]] = {
                "anchors": anchors,
                "probabilities": probabilities,
                "masses": masses,
                "record": record,
                "arrays": arrays,
            }
        wall_seconds = time.perf_counter() - wall_started
        cpu_seconds = time.process_time() - cpu_started
        timing = {
            "model_loading_excluded": True,
            "requests": request_count,
            "request_audio_seconds": request_audio_seconds,
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "wall_rtf_over_teacher_request_audio": (
                wall_seconds / request_audio_seconds
            ),
            "cpu_rtf_over_teacher_request_audio": (
                cpu_seconds / request_audio_seconds
            ),
        }
        arms[key] = {
            "definition": {
                "hop_frames": hop_frames,
                "hop_ms": hop_ms,
                "true_teacher_calls": True,
                "teacher_window_past_ms": TEACHER_PAST_MS,
                "teacher_window_future_ms": TEACHER_FUTURE_MS,
                "dense_expansion": "previous_anchor_hold",
                "reference_only": hop_frames == DENSE_HOP_FRAMES,
            },
            "aggregate": aggregate_arm(per_clip, timing),
        }
        raw_arms[key] = raw_per_clip
        stable = arms[key]["aggregate"]["stable_event"]
        print(
            f"[{arm_number}/{len(HOP_FRAMES)}] {hop_ms} ms: "
            f"requests={request_count}, "
            f"outside-collar="
            f"{arms[key]['aggregate']['known_label_accuracy_outside_250ms_collar_macro']:.4f}, "
            f"stable={stable['detected']}/2, wall={wall_seconds:.2f}s",
            flush=True,
        )

    dense_key = f"hop_{int(DENSE_HOP_FRAMES * FRAME_MS)}ms"
    dense_raw = raw_arms[dense_key]
    for hop_frames in HOP_FRAMES:
        key = f"hop_{int(hop_frames * FRAME_MS)}ms"
        per_clip_comparison = []
        comparison_arrays = []
        for item in switches:
            values = raw_arms[key][item["id"]]
            dense_values = dense_raw[item["id"]]
            comparison, arrays = compare_clip_to_dense(
                values["record"],
                values["arrays"],
                dense_values["record"],
                dense_values["arrays"],
            )
            per_clip_comparison.append(comparison)
            comparison_arrays.append(arrays)
        arms[key]["comparison_to_10ms"] = aggregate_comparison(
            per_clip_comparison, comparison_arrays
        )

    baseline_requests = arms["hop_250ms"]["aggregate"]["generation_timing"][
        "requests"
    ]
    baseline_wall = arms["hop_250ms"]["aggregate"]["generation_timing"][
        "wall_seconds"
    ]
    for arm in arms.values():
        timing = arm["aggregate"]["generation_timing"]
        timing["request_multiplier_vs_250ms"] = (
            timing["requests"] / baseline_requests
        )
        timing["wall_time_multiplier_vs_250ms"] = (
            timing["wall_seconds"] / baseline_wall
        )

    current_parity = compare_250ms_to_main(
        raw_arms["hop_250ms"], switches, main_target_dir
    )
    decision = decide(arms)
    validation = {
        "heldout_ids_match_main_pipeline": len(heldout) == 21,
        "switch_ids_match_main_pipeline": len(switches) == 2,
        "speaker_disjoint": bool(speaker_audit["speaker_disjoint"]),
        "teacher_artifact_matches_main": (
            teacher_identity["artifact_sha256"] == TEACHER_ARTIFACT_SHA256
        ),
        "main_monolingual_teacher_accuracy_reproduced": (
            monolingual["fresh_full_utterance_accuracy"] == 1.0
            and monolingual["main_cache_full_utterance_accuracy"] == 1.0
        ),
        "current_250ms_true_calls_match_main_cache_exactly": current_parity[
            "exact_match"
        ],
        "dense_reference_calls_every_feature_frame": all(
            len(dense_raw[item["id"]]["anchors"])
            == dense_raw[item["id"]]["record"]["frames"]
            for item in switches
        ),
        "dense_reference_detects_both_stable_switches": (
            arms[dense_key]["aggregate"]["stable_event"]["detected"] == 2
        ),
        "all_probabilities_finite_and_normalized": True,
        "all_sparse_expansions_use_previous_anchor_hold": True,
        "driver_unchanged_during_run": (
            hash_path(Path(__file__).resolve()) == driver_sha256
        ),
        "pipeline_sources_unchanged_during_run": (
            snapshot_pipeline_sources()[0] == PIPELINE_SOURCE_SHA256_AT_IMPORT
        ),
        "evaluation_inputs_unchanged_during_run": (
            input_fingerprint(evaluation_records, manifest)
            == evaluation_input_sha256
        ),
        "main_evaluation_targets_unchanged_during_run": (
            target_files_fingerprint(evaluation_records, main_target_dir)
            == main_target_sha256
        ),
    }
    if not all(validation.values()):
        raise RuntimeError(f"experiment validation failed: {validation}")

    output = {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "experiment": EXPERIMENT_NAME,
        "question": (
            "What is the coarsest true-call ECAPA anchor hop whose causal "
            "previous-hold trajectory faithfully matches a 10 ms reference?"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity_sha256": run_identity_sha256,
        "run_identity": run_identity,
        "manifest": str(manifest.relative_to(REPO_ROOT)),
        "evaluation_input_sha256": evaluation_input_sha256,
        "main_evaluation_target_files_sha256": main_target_sha256,
        "pipeline_source_file_sha256": PIPELINE_SOURCE_FILE_SHA256_AT_IMPORT,
        "language_codes": list(LANGUAGE_CODES),
        "heldout_ids": [item["id"] for item in heldout],
        "switch_ids": [item["id"] for item in switches],
        "speaker_split": speaker_audit,
        "teacher_identity": teacher_identity,
        "monolingual_full_utterance_control": monolingual,
        "current_250ms_anchor_parity_with_main": current_parity,
        "arms": arms,
        "decision": decision,
        "validation": validation,
        "limitations": [
            "The 10 ms trajectory is a dense numerical reference, not ground-truth language supervision.",
            "Only two reversed synthetic switch clips are available, so crossing and flip gates are rejection guardrails rather than population estimates.",
            "The nominal 4 s joins include source-audio silence; this audit preserves the main pipeline boundaries for direct comparability.",
            "Teacher-target fidelity does not establish that the 42,567-parameter student can learn the selected trajectory.",
            "Generation timings are one warm CPU run and are descriptive rather than a throughput benchmark with confidence intervals.",
        ],
    }
    atomic_write_json(output_path, output)
    print(
        f"wrote {output_path}; verdict={decision['verdict']}; "
        f"selected={decision['selected_hop_ms']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
