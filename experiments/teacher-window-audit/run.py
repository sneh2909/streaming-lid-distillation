#!/usr/bin/env python3
"""Audit availability-valid ECAPA target windows before another student run.

This is a teacher-only experiment.  It keeps the main pipeline's pinned ECAPA
teacher, seven-language posterior restriction, 250 ms anchor grid, held-out
monolingual clips, and two held-out Hindi/English switch clips.  Only the
teacher evidence window changes.

Run from the repository root with::

    UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
      .venv/bin/python experiments/teacher-window-audit/run.py --fresh

No audio or model weights are written by this driver.  Compact measurements
and posterior traces are written to ``results.json`` next to this file.
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


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

# Reuse the production teacher artifact checks, label map, audio loader,
# manifest parser, speaker audit, and timing constants.  This experiment does
# not modify or duplicate the main target generator.
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


EXPERIMENT_NAME = "teacher-window-audit"
SCHEMA_VERSION = 1
ANALYSIS_VERSION = "teacher-window-audit-v1"
BOUNDARY_COLLAR_MS = 250
STABILITY_HORIZONS_MS = (0, 250, 500)
DEADLINES_MS = (500, 1_000, 2_000)
MIN_PREFIX_INPUT_MS = 500
BASELINE_VARIANT = "current_1750p_250f"


@dataclass(frozen=True)
class WindowVariant:
    name: str
    mode: str
    past_ms: int | None
    future_ms: int
    description: str


VARIANTS = (
    WindowVariant(
        name=BASELINE_VARIANT,
        mode="rolling",
        past_ms=1_750,
        future_ms=250,
        description="current [t-1.75 s, t+0.25 s] local window",
    ),
    WindowVariant(
        name="bounded_750p_250f",
        mode="rolling",
        past_ms=750,
        future_ms=250,
        description="short bounded [t-0.75 s, t+0.25 s] window",
    ),
    WindowVariant(
        name="causal_500p",
        mode="rolling",
        past_ms=500,
        future_ms=0,
        description="causal [t-0.5 s, t] rolling window",
    ),
    WindowVariant(
        name="causal_1000p",
        mode="rolling",
        past_ms=1_000,
        future_ms=0,
        description="causal [t-1 s, t] rolling window",
    ),
    WindowVariant(
        name="causal_2000p",
        mode="rolling",
        past_ms=2_000,
        future_ms=0,
        description="causal [t-2 s, t] rolling window",
    ),
    WindowVariant(
        name="cumulative_prefix",
        mode="prefix",
        past_ms=None,
        future_ms=0,
        description=(
            "causal cumulative [clip start, t] prefix; prefixes shorter than "
            "500 ms receive left-edge zero padding only"
        ),
    ),
)


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
        help="Main target cache, used only for held-out full-utterance parity.",
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
        help="Replace an existing results.json after all identity checks pass.",
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
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
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


def standard_anchor_frames(num_frames: int) -> np.ndarray:
    anchors = np.arange(0, num_frames, TEACHER_HOP_FRAMES, dtype=np.int64)
    if len(anchors) == 0:
        raise ValueError("audio has no feature frames")
    if anchors[-1] != num_frames - 1:
        anchors = np.append(anchors, num_frames - 1)
    if np.any(np.diff(anchors) <= 0):
        raise AssertionError("teacher anchors must increase strictly")
    return anchors


def copy_padded_interval(
    waveform: torch.Tensor, source_start: int, source_end: int
) -> torch.Tensor:
    if source_end <= source_start:
        raise ValueError("teacher interval must have positive duration")
    output = waveform.new_zeros(source_end - source_start)
    clipped_start = max(0, source_start)
    clipped_end = min(len(waveform), source_end)
    if clipped_end > clipped_start:
        destination_start = clipped_start - source_start
        count = clipped_end - clipped_start
        output[destination_start : destination_start + count] = waveform[
            clipped_start:clipped_end
        ]
    return output


def make_request(
    waveform: torch.Tensor, anchor_frame: int, variant: WindowVariant
) -> tuple[torch.Tensor, int]:
    """Return teacher audio and the latest real source sample it can consume."""
    frame_end_sample = anchor_frame * HOP_LENGTH + WIN_LENGTH
    future_samples = round(variant.future_ms * SAMPLE_RATE / 1_000)
    latest_source_sample = min(len(waveform), frame_end_sample + future_samples)

    if variant.mode == "rolling":
        if variant.past_ms is None:
            raise AssertionError("rolling variants require finite past context")
        if (
            variant.past_ms == TEACHER_PAST_MS
            and variant.future_ms == TEACHER_FUTURE_MS
        ):
            request = extract_window(waveform, anchor_frame)
        else:
            past_samples = round(variant.past_ms * SAMPLE_RATE / 1_000)
            request = copy_padded_interval(
                waveform,
                frame_end_sample - past_samples,
                frame_end_sample + future_samples,
            )
    elif variant.mode == "prefix":
        prefix_end = min(len(waveform), frame_end_sample + future_samples)
        prefix = waveform[:prefix_end].contiguous()
        minimum_samples = round(MIN_PREFIX_INPUT_MS * SAMPLE_RATE / 1_000)
        if len(prefix) < minimum_samples:
            # The frozen ECAPA frontend is not defined on the first 25 ms alone.
            # Leading zeros preserve causality and mirror the edge treatment of
            # the fixed rolling windows.
            request = waveform.new_zeros(minimum_samples)
            request[-len(prefix) :] = prefix
        else:
            request = prefix
    else:
        raise ValueError(f"unknown request mode: {variant.mode}")

    expected_latest = min(
        len(waveform),
        anchor_frame * HOP_LENGTH
        + WIN_LENGTH
        + round(variant.future_ms * SAMPLE_RATE / 1_000),
    )
    if latest_source_sample != expected_latest:
        raise AssertionError("teacher availability ledger is inconsistent")
    return request, latest_source_sample


def classify_requests(
    teacher: torch.nn.Module,
    requests: list[torch.Tensor],
    selected_indices: list[int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_logs: list[torch.Tensor] = []
    selected_masses: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            lengths = torch.tensor([len(waveform) for waveform in batch])
            padded = pad_sequence(batch, batch_first=True)
            relative_lengths = lengths / padded.shape[1]
            log_probabilities, _, _, _ = teacher.classify_batch(
                padded, relative_lengths
            )
            chosen = log_probabilities[:, selected_indices]
            selected_logs.append(chosen.cpu())
            selected_masses.append(chosen.exp().sum(dim=-1).cpu())
    logs = torch.cat(selected_logs)
    probabilities = torch.softmax(logs, dim=-1).numpy().astype(np.float32)
    masses = torch.cat(selected_masses).numpy().astype(np.float32)
    if probabilities.shape != (len(requests), len(LANGUAGE_CODES)):
        raise AssertionError(f"unexpected teacher output shape {probabilities.shape}")
    if not np.isfinite(probabilities).all() or not np.isfinite(masses).all():
        raise FloatingPointError("teacher produced a non-finite value")
    if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6):
        raise AssertionError("restricted teacher posterior is not normalized")
    return probabilities, masses


def previous_anchor_hold(
    anchors: np.ndarray, values: np.ndarray, num_frames: int
) -> np.ndarray:
    frame_indices = np.arange(num_frames)
    selected = np.searchsorted(anchors, frame_indices, side="right") - 1
    selected = np.clip(selected, 0, len(anchors) - 1)
    if np.any(anchors[selected] > frame_indices):
        raise AssertionError("previous-anchor hold consulted a future anchor")
    return values[selected]


def expected_indices(item: dict, frame_times: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            LANGUAGE_CODES.index(expected_language(item, float(seconds)))
            for seconds in frame_times
        ],
        dtype=np.int64,
    )


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def stable_event(
    classes: np.ndarray,
    times: np.ndarray,
    *,
    source_index: int,
    target_index: int,
    boundary_seconds: float,
    stable_ms: int,
    future_ms: int,
) -> dict:
    """Find a direction-aware crossing sustained for a real-time horizon."""

    horizon_seconds = stable_ms / 1_000

    def run_end(start: int, wanted: int) -> int | None:
        if classes[start] != wanted:
            return None
        if horizon_seconds == 0:
            return start
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

    permitted_early_seconds = future_ms / 1_000
    onset = None
    confirmation = None
    premature_target_onset = None
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

    availability_offset_seconds = future_ms / 1_000
    return {
        "stability_horizon_ms": stable_ms,
        "source_armed": source_armed,
        "source_confirmation_seconds": source_confirmation_seconds,
        "detected": onset is not None,
        "premature_target_onset_seconds": premature_target_onset,
        "semantic_onset_seconds": onset,
        "semantic_onset_lag_ms": (
            None if onset is None else 1_000 * (onset - boundary_seconds)
        ),
        "onset_available_seconds": (
            None if onset is None else onset + availability_offset_seconds
        ),
        "onset_availability_lag_ms": (
            None
            if onset is None
            else 1_000
            * (onset + availability_offset_seconds - boundary_seconds)
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
            else confirmation + availability_offset_seconds
        ),
        "confirmation_availability_lag_ms": (
            None
            if confirmation is None
            else 1_000
            * (confirmation + availability_offset_seconds - boundary_seconds)
        ),
    }


def clip_metrics(
    item: dict,
    waveform: torch.Tensor,
    anchors: np.ndarray,
    probabilities: np.ndarray,
    masses: np.ndarray,
    latest_samples: list[int],
    variant: WindowVariant,
) -> dict:
    if len(item["segments"]) != 2:
        raise ValueError(f"expected exactly one boundary in {item['id']}")
    source_code = item["segments"][0]["language"]
    target_code = item["segments"][1]["language"]
    source_index = LANGUAGE_CODES.index(source_code)
    target_index = LANGUAGE_CODES.index(target_code)
    boundary_seconds = float(item["segments"][0]["end_seconds"])
    num_frames = feature_frame_count(len(waveform))
    anchor_times = (anchors * HOP_LENGTH + WIN_LENGTH) / SAMPLE_RATE
    availability_times = np.asarray(latest_samples, dtype=np.float64) / SAMPLE_RATE
    expected_availability = np.minimum(
        len(waveform) / SAMPLE_RATE,
        anchor_times + variant.future_ms / 1_000,
    )
    if not np.allclose(availability_times, expected_availability, atol=1 / SAMPLE_RATE):
        raise AssertionError("sample availability does not match target definition")

    dense = previous_anchor_hold(anchors, probabilities, num_frames)
    dense_classes = dense.argmax(axis=-1)
    frame_times = (
        np.arange(num_frames, dtype=np.float64) * HOP_LENGTH + WIN_LENGTH
    ) / SAMPLE_RATE
    expected = expected_indices(item, frame_times)
    outside_collar = (
        np.abs(frame_times - boundary_seconds) > BOUNDARY_COLLAR_MS / 1_000
    )
    anchor_classes = probabilities.argmax(axis=-1)
    events = {
        str(stable_ms): stable_event(
            anchor_classes,
            anchor_times,
            source_index=source_index,
            target_index=target_index,
            boundary_seconds=boundary_seconds,
            stable_ms=stable_ms,
            future_ms=variant.future_ms,
        )
        for stable_ms in STABILITY_HORIZONS_MS
    }
    matched_transition = events["500"]["detected"]
    changes = int(np.sum(anchor_classes[1:] != anchor_classes[:-1]))
    unmatched_changes = max(0, changes - int(matched_transition))
    duration_minutes = len(waveform) / SAMPLE_RATE / 60

    deadline_recall = {}
    for deadline_ms in DEADLINES_MS:
        deadline_end = min(
            float(item["segments"][1]["end_seconds"]),
            boundary_seconds + deadline_ms / 1_000,
        )
        selected = (frame_times >= boundary_seconds) & (frame_times < deadline_end)
        deadline_recall[str(deadline_ms)] = {
            "frames": int(selected.sum()),
            "target_language_recall": float(
                np.mean(dense_classes[selected] == target_index)
            ),
        }

    traces = []
    for index, anchor in enumerate(anchors):
        traces.append(
            {
                "anchor_frame": int(anchor),
                "semantic_seconds": float(anchor_times[index]),
                "available_seconds": float(availability_times[index]),
                "expected_language": LANGUAGE_CODES[
                    expected_language_index := LANGUAGE_CODES.index(
                        expected_language(item, float(anchor_times[index]))
                    )
                ],
                "predicted_language": LANGUAGE_CODES[int(anchor_classes[index])],
                "correct": bool(anchor_classes[index] == expected_language_index),
                "selected_confidence": float(probabilities[index].max()),
                "in_set_mass": float(masses[index]),
                "probabilities": [float(value) for value in probabilities[index]],
            }
        )

    return {
        "id": item["id"],
        "source_language": source_code,
        "target_language": target_code,
        "boundary_seconds": boundary_seconds,
        "duration_seconds": len(waveform) / SAMPLE_RATE,
        "frames": num_frames,
        "anchors": len(anchors),
        "known_label_correct": int(np.sum(dense_classes == expected)),
        "known_label_accuracy": float(np.mean(dense_classes == expected)),
        "outside_collar_frames": int(outside_collar.sum()),
        "known_label_correct_outside_250ms_collar": int(
            np.sum(dense_classes[outside_collar] == expected[outside_collar])
        ),
        "known_label_accuracy_outside_250ms_collar": float(
            np.mean(dense_classes[outside_collar] == expected[outside_collar])
        ),
        "source_segment_recall_outside_collar": float(
            np.mean(
                dense_classes[(frame_times < boundary_seconds) & outside_collar]
                == source_index
            )
        ),
        "target_segment_recall_outside_collar": float(
            np.mean(
                dense_classes[(frame_times >= boundary_seconds) & outside_collar]
                == target_index
            )
        ),
        "mean_selected_language_mass": float(masses.mean()),
        "minimum_selected_language_mass": float(masses.min()),
        "deadline_target_recall": deadline_recall,
        "events": events,
        "anchor_top1_changes": changes,
        "matched_reference_changes": int(matched_transition),
        "unmatched_anchor_top1_changes": unmatched_changes,
        "unmatched_anchor_top1_changes_per_minute": (
            unmatched_changes / duration_minutes
        ),
        "posterior_sha256": hashlib.sha256(
            probabilities.tobytes() + masses.tobytes()
        ).hexdigest(),
        "anchor_trace": traces,
    }


def aggregate_variant(per_clip: list[dict], timing: dict) -> dict:
    total_frames = sum(record["frames"] for record in per_clip)
    total_correct = sum(record["known_label_correct"] for record in per_clip)
    collar_frames = sum(record["outside_collar_frames"] for record in per_clip)
    collar_correct = sum(
        record["known_label_correct_outside_250ms_collar"]
        for record in per_clip
    )
    events: dict[str, dict] = {}
    for stable_ms in STABILITY_HORIZONS_MS:
        key = str(stable_ms)
        detected = [record["events"][key] for record in per_clip if record["events"][key]["detected"]]
        semantic_lags = [event["semantic_onset_lag_ms"] for event in detected]
        onset_available_lags = [
            event["onset_availability_lag_ms"] for event in detected
        ]
        confirmation_available_lags = [
            event["confirmation_availability_lag_ms"] for event in detected
        ]
        events[key] = {
            "detected": len(detected),
            "missed": len(per_clip) - len(detected),
            "semantic_onset_lag_ms_median_detected_only": percentile(
                semantic_lags, 50
            ),
            "semantic_onset_lag_ms_p95_detected_only": percentile(
                semantic_lags, 95
            ),
            "onset_availability_lag_ms_median_detected_only": percentile(
                onset_available_lags, 50
            ),
            "onset_availability_lag_ms_p95_detected_only": percentile(
                onset_available_lags, 95
            ),
            "confirmation_availability_lag_ms_median_detected_only": percentile(
                confirmation_available_lags, 50
            ),
            "confirmation_availability_lag_ms_p95_detected_only": percentile(
                confirmation_available_lags, 95
            ),
            "detected_only_n": len(detected),
        }

    deadline = {}
    for deadline_ms in DEADLINES_MS:
        key = str(deadline_ms)
        values = [
            record["deadline_target_recall"][key]["target_language_recall"]
            for record in per_clip
        ]
        deadline[key] = {
            "target_language_recall_macro": float(np.mean(values)),
            "per_clip": {
                record["id"]: record["deadline_target_recall"][key][
                    "target_language_recall"
                ]
                for record in per_clip
            },
        }

    return {
        "switch_clips": len(per_clip),
        "frames": total_frames,
        "known_label_accuracy_micro": total_correct / total_frames,
        "known_label_accuracy_macro": float(
            np.mean([record["known_label_accuracy"] for record in per_clip])
        ),
        "known_label_accuracy_outside_250ms_collar_micro": (
            collar_correct / collar_frames
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
        "deadline_target_recall": deadline,
        "events": events,
        "anchor_top1_changes_total": sum(
            record["anchor_top1_changes"] for record in per_clip
        ),
        "unmatched_anchor_top1_changes_total": sum(
            record["unmatched_anchor_top1_changes"] for record in per_clip
        ),
        "unmatched_anchor_top1_changes_per_clip_mean": float(
            np.mean(
                [record["unmatched_anchor_top1_changes"] for record in per_clip]
            )
        ),
        "unmatched_anchor_top1_changes_per_minute_mean": float(
            np.mean(
                [
                    record["unmatched_anchor_top1_changes_per_minute"]
                    for record in per_clip
                ]
            )
        ),
        "mean_selected_language_mass": float(
            np.mean(
                [record["mean_selected_language_mass"] for record in per_clip]
            )
        ),
        "generation_timing": timing,
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
    correct = 0
    main_correct = 0
    maximum_difference = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for item in heldout:
            waveform = load_audio(resolve_audio_path(item, manifest))
            log_probabilities, _, _, _ = teacher.classify_batch(
                waveform.unsqueeze(0)
            )
            selected_logs = log_probabilities[0, selected_indices].cpu()
            posterior = torch.softmax(selected_logs, dim=-1).numpy()
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
            correct += int(prediction == expected_index)
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
        "fresh_full_utterance_accuracy": correct / len(heldout),
        "main_cache_full_utterance_accuracy": main_correct / len(heldout),
        "fresh_correct": correct,
        "main_cache_correct": main_correct,
        "max_abs_posterior_difference_from_main_cache": maximum_difference,
        "wall_seconds": time.perf_counter() - started,
        "per_clip": records,
    }


def compare_current_anchors_to_main(
    current_arm: dict, switches: list[dict], main_target_dir: Path
) -> dict:
    """Prove that the experiment's current-window anchor calls are unchanged."""
    records_by_id = {
        record["id"]: record
        for record in current_arm["aggregate"]["per_clip"]
    }
    maximum_difference = 0.0
    frame_arrays_equal = True
    per_clip = []
    for item in switches:
        record = records_by_id[item["id"]]
        experiment_frames = np.asarray(
            [trace["anchor_frame"] for trace in record["anchor_trace"]],
            dtype=np.int64,
        )
        experiment_probabilities = np.asarray(
            [trace["probabilities"] for trace in record["anchor_trace"]],
            dtype=np.float32,
        )
        with np.load(
            main_target_dir / f"{item['id']}.npz", allow_pickle=False
        ) as target_file:
            main_frames = target_file["anchor_frames"].copy()
            main_probabilities = target_file["anchor_probs"].copy()
        same_frames = np.array_equal(experiment_frames, main_frames)
        if experiment_probabilities.shape != main_probabilities.shape:
            raise ValueError(
                f"current-window anchor shape mismatch for {item['id']}: "
                f"{experiment_probabilities.shape} versus {main_probabilities.shape}"
            )
        difference = float(
            np.max(np.abs(experiment_probabilities - main_probabilities))
        )
        frame_arrays_equal = frame_arrays_equal and same_frames
        maximum_difference = max(maximum_difference, difference)
        per_clip.append(
            {
                "id": item["id"],
                "anchor_frames_equal": same_frames,
                "max_abs_anchor_posterior_difference": difference,
            }
        )
    return {
        "clips": len(switches),
        "anchor_frames_equal": frame_arrays_equal,
        "max_abs_anchor_posterior_difference": maximum_difference,
        "exact_posterior_match": maximum_difference == 0.0,
        "per_clip": per_clip,
    }


def compare_and_decide(arms: dict[str, dict]) -> dict:
    baseline = arms[BASELINE_VARIANT]["aggregate"]
    baseline_accuracy = baseline[
        "known_label_accuracy_outside_250ms_collar_macro"
    ]
    baseline_churn = baseline["unmatched_anchor_top1_changes_per_clip_mean"]
    baseline_lag = baseline["events"]["500"][
        "semantic_onset_lag_ms_median_detected_only"
    ]

    comparisons: dict[str, dict] = {}
    passing: list[str] = []
    for variant, arm in arms.items():
        aggregate = arm["aggregate"]
        stable = aggregate["events"]["500"]
        accuracy = aggregate[
            "known_label_accuracy_outside_250ms_collar_macro"
        ]
        churn = aggregate["unmatched_anchor_top1_changes_per_clip_mean"]
        checks = {
            "detects_both_switches": stable["detected"] == 2,
            "median_semantic_stable_onset_le_750ms": (
                stable["semantic_onset_lag_ms_median_detected_only"] is not None
                and stable["semantic_onset_lag_ms_median_detected_only"] <= 750
            ),
            "p95_semantic_stable_onset_le_1250ms": (
                stable["semantic_onset_lag_ms_p95_detected_only"] is not None
                and stable["semantic_onset_lag_ms_p95_detected_only"] <= 1_250
            ),
            "outside_collar_accuracy_within_2pp_of_current": (
                accuracy >= baseline_accuracy - 0.02
            ),
            "no_more_than_one_extra_unmatched_flip_per_clip": (
                churn <= baseline_churn + 1.0
            ),
            "future_context_within_250ms_student_evidence": (
                arm["definition"]["future_ms"] <= 250
            ),
        }
        passed = all(checks.values())
        if variant != BASELINE_VARIANT and passed:
            passing.append(variant)
        comparisons[variant] = {
            "outside_collar_accuracy_delta_pp_vs_current": 100
            * (accuracy - baseline_accuracy),
            "semantic_500ms_stable_onset_delta_ms_vs_current": (
                None
                if baseline_lag is None
                or stable["semantic_onset_lag_ms_median_detected_only"] is None
                else stable["semantic_onset_lag_ms_median_detected_only"]
                - baseline_lag
            ),
            "checks": checks,
            "passes_shortlist_gate": passed,
        }

    selected = None
    if passing:
        selected = min(
            passing,
            key=lambda name: (
                arms[name]["aggregate"]["events"]["500"][
                    "semantic_onset_lag_ms_median_detected_only"
                ],
                arms[name]["aggregate"]["events"]["500"][
                    "confirmation_availability_lag_ms_median_detected_only"
                ],
                -arms[name]["aggregate"][
                    "known_label_accuracy_outside_250ms_collar_macro"
                ],
            ),
        )

    if selected is None:
        verdict = "reject"
        reason = "No alternative target window passed the predeclared shortlist gate."
    else:
        selected_lag = arms[selected]["aggregate"]["events"]["500"][
            "semantic_onset_lag_ms_median_detected_only"
        ]
        improvement = None if baseline_lag is None else baseline_lag - selected_lag
        if improvement is not None and improvement >= 250:
            verdict = "adopt"
            reason = (
                f"{selected} passed every gate and moved median 500 ms-stable "
                f"semantic onset {improvement:.0f} ms earlier than the current window."
            )
        else:
            verdict = "inconclusive"
            reason = (
                f"{selected} passed the absolute gate but did not improve the "
                "current window by the predeclared 250 ms decision margin."
            )

    return {
        "gate_definition": {
            "stable_horizon_ms": 500,
            "required_switches_detected": 2,
            "semantic_onset_median_max_ms": 750,
            "semantic_onset_p95_max_ms": 1_250,
            "maximum_nonboundary_accuracy_loss_pp": 2,
            "maximum_extra_unmatched_flips_per_clip": 1,
            "maximum_future_context_ms": 250,
            "adoption_requires_lag_improvement_ms": 250,
        },
        "comparisons": comparisons,
        "passing_alternatives": passing,
        "selected_for_next_target_delay_grid": selected,
        "verdict": verdict,
        "reason": reason,
        "scope": (
            "A positive verdict shortlists a teacher target for a controlled "
            "student target-by-delay grid; it does not replace the main pipeline."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.threads != 6:
        raise ValueError("TEAM.md requires exactly six Torch threads")
    if args.teacher_batch_size <= 0:
        raise ValueError("teacher batch size must be positive")
    if TEACHER_HOP_FRAMES * FRAME_MS != 250:
        raise ValueError("experiment contract expects a 250 ms teacher hop")
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
    heldout_counts = Counter(item["language"] for item in heldout)
    if heldout_counts != Counter({code: 3 for code in LANGUAGE_CODES}):
        raise ValueError(f"unexpected held-out balance: {heldout_counts}")
    if {item["id"] for item in switches} != {
        "switch_hi_en_eval",
        "switch_en_hi_eval",
    }:
        raise ValueError("expected the main pipeline's two held-out switch clips")

    evaluation_records = heldout + switches
    evaluation_input_sha256 = input_fingerprint(evaluation_records, manifest)
    main_target_sha256 = target_files_fingerprint(
        evaluation_records, main_target_dir
    )
    driver_sha256 = hash_path(Path(__file__).resolve())
    dependencies = dependency_versions()
    teacher_snapshot, teacher_identity = resolve_teacher_artifact()
    if teacher_identity["artifact_sha256"] != TEACHER_ARTIFACT_SHA256:
        raise ValueError("resolved teacher does not match the main pinned artifact")

    run_identity = {
        "experiment": EXPERIMENT_NAME,
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "evaluation_input_sha256": evaluation_input_sha256,
        "main_evaluation_target_files_sha256": main_target_sha256,
        "driver_sha256": driver_sha256,
        "pipeline_source_sha256": PIPELINE_SOURCE_SHA256_AT_IMPORT,
        "teacher_identity": teacher_identity,
        "dependencies": dependencies,
        "settings": {
            "threads": args.threads,
            "teacher_batch_size": args.teacher_batch_size,
            "teacher_hop_frames": TEACHER_HOP_FRAMES,
            "teacher_hop_ms": TEACHER_HOP_FRAMES * FRAME_MS,
            "boundary_collar_ms": BOUNDARY_COLLAR_MS,
            "stability_horizons_ms": list(STABILITY_HORIZONS_MS),
            "deadlines_ms": list(DEADLINES_MS),
            "anchor_expansion": "previous_anchor_hold",
            "prefix_minimum_input_ms": MIN_PREFIX_INPUT_MS,
        },
        "variants": [asdict(variant) for variant in VARIANTS],
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

    arms: dict[str, dict] = {}
    for arm_number, variant in enumerate(VARIANTS, start=1):
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        per_clip = []
        request_count = 0
        request_audio_seconds = 0.0
        for item in switches:
            waveform = load_audio(resolve_audio_path(item, manifest))
            anchors = standard_anchor_frames(feature_frame_count(len(waveform)))
            requests: list[torch.Tensor] = []
            latest_samples: list[int] = []
            for anchor in anchors:
                request, latest_sample = make_request(
                    waveform, int(anchor), variant
                )
                requests.append(request)
                latest_samples.append(latest_sample)
            probabilities, masses = classify_requests(
                teacher,
                requests,
                selected_indices,
                args.teacher_batch_size,
            )
            request_count += len(requests)
            request_audio_seconds += sum(len(request) for request in requests) / SAMPLE_RATE
            per_clip.append(
                clip_metrics(
                    item,
                    waveform,
                    anchors,
                    probabilities,
                    masses,
                    latest_samples,
                    variant,
                )
            )
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
        definition = {
            **asdict(variant),
            "teacher": TEACHER_NAME,
            "future_context_within_main_student_evidence": (
                variant.future_ms <= 250
            ),
            "anchor_hop_ms": TEACHER_HOP_FRAMES * FRAME_MS,
            "anchor_expansion": "previous_anchor_hold",
        }
        arms[variant.name] = {
            "definition": definition,
            "aggregate": aggregate_variant(per_clip, timing),
        }
        stable = arms[variant.name]["aggregate"]["events"]["500"]
        accuracy = arms[variant.name]["aggregate"][
            "known_label_accuracy_outside_250ms_collar_macro"
        ]
        print(
            f"[{arm_number}/{len(VARIANTS)}] {variant.name}: "
            f"outside-collar={accuracy:.4f}, "
            f"500ms-stable={stable['detected']}/2, "
            f"median-onset={stable['semantic_onset_lag_ms_median_detected_only']}",
            flush=True,
        )

    current_anchor_parity = compare_current_anchors_to_main(
        arms[BASELINE_VARIANT], switches, main_target_dir
    )
    decision = compare_and_decide(arms)
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
        "current_window_anchor_calls_match_main_exactly": (
            current_anchor_parity["anchor_frames_equal"]
            and current_anchor_parity["exact_posterior_match"]
        ),
        "all_probabilities_finite_and_normalized": True,
        "all_expansions_use_previous_anchor_hold": True,
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
    required_validation = [
        key
        for key in validation
        if key != "pipeline_sources_unchanged_during_run"
    ]
    if not all(validation[key] for key in required_validation):
        raise RuntimeError(f"experiment validation failed: {validation}")

    output = {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "experiment": EXPERIMENT_NAME,
        "question": (
            "Which availability-valid ECAPA local target window transitions "
            "fast enough for a causal student without losing non-boundary "
            "Hindi/English target accuracy?"
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
        "current_window_anchor_parity_with_main": current_anchor_parity,
        "arms": arms,
        "decision": decision,
        "validation": validation,
        "limitations": [
            "Only two mirrored synthetic switch clips are available, so p95 is descriptive rather than a population estimate.",
            "The manifest's nominal 4 s joins include source-audio silence; this audit preserves those main-pipeline boundaries for comparability.",
            "The 250 ms sparse anchor grid can quantize crossings; a separate dense-anchor audit is still required before training.",
            "Teacher-only target quality does not establish that the 42,567-parameter student can learn the selected trajectory.",
        ],
    }
    atomic_write_json(output_path, output)
    print(
        f"wrote {output_path}; verdict={decision['verdict']}; "
        f"selected={decision['selected_for_next_target_delay_grid']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
