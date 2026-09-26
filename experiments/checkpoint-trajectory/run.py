#!/usr/bin/env python3
"""Audit every loader-pass checkpoint in the exact submitted trajectory.

The experiment reproduces the main 1,600-update training run, captures the
initial state and every 10th update, and evaluates all 161 states on the same
21 monolingual synthetic-development clips and two switch clips used by the
main pipeline.  It also measures exact familiar-audio fit on training clips.

Run from the repository root with::

    UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
      experiments/checkpoint-trajectory/run.py --fresh
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# Import the model, frontend, data, loss, timing, streaming scheduler, and
# training-safety contracts from the main pipeline.  This driver never edits
# those sources.
from scripts.eval import smooth_posteriors  # noqa: E402
from scripts.train import (  # noqa: E402
    assert_finite_post_update_state,
    build_training_contract,
    seed_everything,
    validate_full_batch_training_set,
)
from streaming_lid.audio import LogMelFrontend, load_audio  # noqa: E402
from streaming_lid.config import (  # noqa: E402
    CHUNK_FRAMES,
    EARLY_RAMP_FRAMES,
    HOP_LENGTH,
    LABEL_DELAY_FRAMES,
    LANGUAGE_CODES,
    MODEL_LOOKAHEAD_FRAMES,
    SAMPLE_RATE,
    TEACHER_TEMPERATURE,
    WIN_LENGTH,
)
from streaming_lid.data import (  # noqa: E402
    DistillationDataset,
    TeacherTargetCache,
    collate_distillation_batch,
    file_sha256,
    read_manifest,
    require_speaker_disjoint,
    resolve_audio_path,
)
from streaming_lid.loss import delayed_distillation_loss  # noqa: E402
from streaming_lid.model import CausalLIDStudent  # noqa: E402
from streaming_lid.run_identity import (  # noqa: E402
    assert_run_dependency_snapshot_unchanged,
    capture_run_dependency_snapshot,
    configured_model_kwargs,
    model_state_sha256,
    training_configuration,
)


EXPERIMENT_NAME = "checkpoint-trajectory"
SCHEMA_VERSION = 1
ANALYSIS_VERSION = "checkpoint-trajectory-v1"
DEFAULT_STEPS = 1_600
DEFAULT_BATCH_SIZE = 7
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_SEED = 7
SNAPSHOT_INTERVAL = 10
PREFIX_SECONDS: tuple[float | str, ...] = (0.5, 1.0, 2.0, 4.0, "full")
SELECTION_PREFIXES = ("1", "2", "full")
EXACT_EDGE_TRAIN_INDICES = (5, 6, 7)
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 5.0
EMA_NEW_WEIGHT = 0.30
POLICY_THRESHOLD = 0.60
POLICY_MARGIN = 0.10
POLICY_DWELL_CHUNKS = 3
RAW_STABILITY_SECONDS = 0.50
SWITCH_COLLAR_SECONDS = 0.25
SWITCH_DEADLINES_SECONDS = (0.5, 1.0, 2.0, 3.0)
MISS_PENALTY_MS = 4_000.0
LABEL_GAIN_GATE_PP = 2.0
SWITCH_RECALL_GAIN_GATE_PP = 10.0
LABEL_ELIGIBILITY_TOLERANCE = 0.02
WORST_LANGUAGE_TOLERANCE = 0.05

SOURCE_FILES = (
    "scripts/train.py",
    "scripts/eval.py",
    "src/streaming_lid/audio.py",
    "src/streaming_lid/config.py",
    "src/streaming_lid/data.py",
    "src/streaming_lid/loss.py",
    "src/streaming_lid/model.py",
    "src/streaming_lid/run_identity.py",
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
        "--output",
        type=Path,
        default=Path("experiments/checkpoint-trajectory/results.json"),
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("data/experiments/checkpoint-trajectory/trajectory.pt"),
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Replace prior experiment artifacts only after all checks pass.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    expected = {
        "steps": DEFAULT_STEPS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "seed": DEFAULT_SEED,
        "threads": 6,
    }
    for name, wanted in expected.items():
        actual = getattr(args, name)
        if actual != wanted:
            raise ValueError(
                f"{name.replace('_', '-')} must be {wanted} for the frozen trajectory; "
                f"got {actual}"
            )
    if args.learning_rate != DEFAULT_LEARNING_RATE:
        raise ValueError(
            f"learning-rate must be {DEFAULT_LEARNING_RATE} for the frozen trajectory"
        )
    if args.steps % SNAPSHOT_INTERVAL:
        raise ValueError("steps must be divisible by the snapshot interval")
    if args.output.exists() and not args.fresh:
        raise FileExistsError(f"{args.output} exists; pass --fresh to replace it")
    if args.trajectory.exists() and not args.fresh:
        raise FileExistsError(f"{args.trajectory} exists; pass --fresh to replace it")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def snapshot_sources() -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        payload = (REPO_ROOT / relative).read_bytes()
        snapshot[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return snapshot


def package_versions() -> dict[str, str]:
    names = ("numpy", "torch", "torchaudio", "soundfile")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def prefix_key(value: float | str) -> str:
    if value == "full":
        return "full"
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def fixed_prefix(waveform: torch.Tensor, seconds: float) -> torch.Tensor:
    wanted = int(round(seconds * SAMPLE_RATE))
    if len(waveform) >= wanted:
        return waveform[:wanted].contiguous()
    return torch.nn.functional.pad(waveform, (0, wanted - len(waveform)))


def feature_batch(examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "features": pad_sequence(
            [example["features"] for example in examples], batch_first=True
        ),
        "lengths": torch.tensor(
            [len(example["features"]) for example in examples], dtype=torch.long
        ),
        "items": [example["item"] for example in examples],
    }


def macro_f1(
    expected: Sequence[int], predicted: Sequence[int]
) -> tuple[float, dict[str, float]]:
    per_language: dict[str, float] = {}
    for index, language in enumerate(LANGUAGE_CODES):
        true_positive = sum(
            truth == index and guess == index
            for truth, guess in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            truth != index and guess == index
            for truth, guess in zip(expected, predicted, strict=True)
        )
        false_negative = sum(
            truth == index and guess != index
            for truth, guess in zip(expected, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        per_language[language] = (
            0.0 if denominator == 0 else 2 * true_positive / denominator
        )
    return float(np.mean(list(per_language.values()))), per_language


def classification_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected = [LANGUAGE_CODES.index(row["expected"]) for row in rows]
    predicted = [LANGUAGE_CODES.index(row["prediction"]) for row in rows]
    if not expected:
        raise ValueError("cannot summarize an empty classification table")
    recalls: dict[str, float] = {}
    support: dict[str, int] = {}
    confusion = {language: Counter() for language in LANGUAGE_CODES}
    for truth, guess in zip(expected, predicted, strict=True):
        confusion[LANGUAGE_CODES[truth]][LANGUAGE_CODES[guess]] += 1
    for index, language in enumerate(LANGUAGE_CODES):
        guesses = [
            guess
            for truth, guess in zip(expected, predicted, strict=True)
            if truth == index
        ]
        if not guesses:
            raise AssertionError(f"classification table has no {language} examples")
        recalls[language] = sum(guess == index for guess in guesses) / len(guesses)
        support[language] = len(guesses)
    f1, per_language_f1 = macro_f1(expected, predicted)
    return {
        "n_clips": len(rows),
        "correct": int(sum(a == b for a, b in zip(expected, predicted, strict=True))),
        "accuracy": float(np.mean(np.asarray(expected) == np.asarray(predicted))),
        "language_macro_accuracy": float(np.mean(list(recalls.values()))),
        "macro_f1": f1,
        "minimum_language_recall": float(min(recalls.values())),
        "per_language_recall": recalls,
        "per_language_f1": per_language_f1,
        "support_per_language": support,
        "confusion": {
            truth: dict(sorted(counts.items()))
            for truth, counts in confusion.items()
        },
        "mean_max_probability": float(
            np.mean([row["max_probability"] for row in rows])
        ),
    }


def posterior_rows(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    items: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        length = int(lengths[index])
        stable_stop = length - MODEL_LOOKAHEAD_FRAMES
        if stable_stop <= LABEL_DELAY_FRAMES:
            raise ValueError(f"clip {item['id']} has no aligned stable outputs")
        valid_logits = logits[index, LABEL_DELAY_FRAMES:stable_stop]
        probabilities = torch.softmax(valid_logits, dim=-1).mean(dim=0)
        probabilities = probabilities / probabilities.sum()
        prediction_index = int(probabilities.argmax())
        rows.append(
            {
                "clip_id": item["id"],
                "expected": item["language"],
                "prediction": LANGUAGE_CODES[prediction_index],
                "max_probability": float(probabilities[prediction_index]),
                "valid_frames": len(valid_logits),
            }
        )
    return rows


def generic_router(
    probabilities: np.ndarray, times: np.ndarray
) -> dict[str, Any]:
    if len(probabilities) != len(times) or not len(times):
        raise ValueError("router needs a non-empty aligned probability trace")
    smoothed = smooth_posteriors(probabilities, EMA_NEW_WEIGHT)
    active: int | None = None
    candidate: int | None = None
    candidate_chunks = 0
    events: list[dict[str, Any]] = []
    unknown_chunks = 0
    for index, posterior in enumerate(smoothed):
        order = np.argsort(posterior)
        winner = int(order[-1])
        runner_up = int(order[-2])
        qualifies = (
            posterior[winner] >= POLICY_THRESHOLD
            and posterior[winner] - posterior[runner_up] >= POLICY_MARGIN
        )
        wanted = winner if qualifies else None
        if active is None:
            unknown_chunks += 1
        if wanted is None or wanted == active:
            candidate = None
            candidate_chunks = 0
            continue
        if wanted == candidate:
            candidate_chunks += 1
        else:
            candidate = wanted
            candidate_chunks = 1
        if candidate_chunks >= POLICY_DWELL_CHUNKS:
            previous = active
            active = wanted
            events.append(
                {
                    "time_seconds": float(times[index]),
                    "language": LANGUAGE_CODES[active],
                    "previous_language": (
                        None if previous is None else LANGUAGE_CODES[previous]
                    ),
                }
            )
            candidate = None
            candidate_chunks = 0
    return {
        "events": events,
        "committed_changes": max(0, len(events) - 1),
        "initial_commit": events[0] if events else None,
        "unknown_chunks_before_initial_commit": unknown_chunks,
        "configuration": {
            "threshold": POLICY_THRESHOLD,
            "margin": POLICY_MARGIN,
            "dwell_chunks": POLICY_DWELL_CHUNKS,
            "ema_new_weight": EMA_NEW_WEIGHT,
        },
    }


def contiguous_runs(classes: np.ndarray, times: np.ndarray) -> list[dict[str, Any]]:
    if len(classes) != len(times):
        raise ValueError("class and time traces differ in length")
    runs: list[dict[str, Any]] = []
    start = 0
    for stop in range(1, len(classes) + 1):
        if stop < len(classes) and classes[stop] == classes[start]:
            continue
        confirmation = int(
            np.searchsorted(
                times,
                times[start] + RAW_STABILITY_SECONDS,
                side="left",
            )
        )
        stable = confirmation < stop
        runs.append(
            {
                "class_index": int(classes[start]),
                "start_index": start,
                "stop_index_exclusive": stop,
                "start_seconds": float(times[start]),
                "end_seconds": float(times[stop - 1]),
                "stable": stable,
                "confirmation_seconds": (
                    float(times[confirmation]) if stable else None
                ),
            }
        )
        start = stop
    return runs


def raw_switch_event(
    classes: np.ndarray,
    times: np.ndarray,
    *,
    source_index: int,
    target_index: int,
    boundary_seconds: float,
) -> dict[str, Any]:
    runs = contiguous_runs(classes, times)
    source_runs = [
        run
        for run in runs
        if run["class_index"] == source_index
        and run["stable"]
        and run["confirmation_seconds"] <= boundary_seconds
    ]
    source_precondition = bool(source_runs)
    premature_target_runs = [
        run
        for run in runs
        if run["class_index"] == target_index
        and run["start_seconds"] < boundary_seconds
    ]
    target_run = next(
        (
            run
            for run in runs
            if run["class_index"] == target_index
            and run["stable"]
            and run["start_seconds"] >= boundary_seconds
        ),
        None,
    )
    detected = source_precondition and target_run is not None
    return {
        "source_stable_before_boundary": source_precondition,
        "premature_target_runs": len(premature_target_runs),
        "detected": detected,
        "first_stable_seconds": (
            float(target_run["start_seconds"]) if detected else None
        ),
        "confirmed_seconds": (
            float(target_run["confirmation_seconds"]) if detected else None
        ),
        "first_stable_lag_ms": (
            1_000 * (float(target_run["start_seconds"]) - boundary_seconds)
            if detected
            else None
        ),
        "confirmed_lag_ms": (
            1_000 * (float(target_run["confirmation_seconds"]) - boundary_seconds)
            if detected
            else None
        ),
    }


def policy_switch_event(
    router: Mapping[str, Any],
    *,
    source_language: str,
    target_language: str,
    boundary_seconds: float,
) -> dict[str, Any]:
    active_at_boundary: str | None = None
    for event in router["events"]:
        if event["time_seconds"] <= boundary_seconds:
            active_at_boundary = event["language"]
        else:
            break
    source_precondition = active_at_boundary == source_language
    target_event = next(
        (
            event
            for event in router["events"]
            if event["time_seconds"] > boundary_seconds
            and event["language"] == target_language
        ),
        None,
    )
    detected = source_precondition and target_event is not None
    return {
        "active_language_at_boundary": active_at_boundary,
        "source_committed_at_boundary": source_precondition,
        "detected": detected,
        "detected_seconds": (
            float(target_event["time_seconds"]) if detected else None
        ),
        "lag_ms": (
            1_000 * (float(target_event["time_seconds"]) - boundary_seconds)
            if detected
            else None
        ),
    }


def chunk_trace(
    logits: torch.Tensor, length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return real streaming-call groups, their clocks, and stable frame posteriors.

    The first 16-frame input call emits 12 outputs because four frames remain
    pending; later full calls emit 16.  Group the stable full-pass logits by
    those actual output ranges rather than concatenating and reslicing them in
    artificial 16-output groups.
    """
    stable_stop = length - MODEL_LOOKAHEAD_FRAMES
    if stable_stop <= LABEL_DELAY_FRAMES:
        raise ValueError("trace has no delay-aligned stable outputs")
    probabilities = torch.softmax(logits[:stable_stop], dim=-1).cpu().numpy()
    groups: list[np.ndarray] = []
    times: list[float] = []
    next_output = 0
    for input_start in range(0, length, CHUNK_FRAMES):
        input_stop = min(input_start + CHUNK_FRAMES, length)
        emitted_stop = max(0, input_stop - MODEL_LOOKAHEAD_FRAMES)
        selected_start = max(next_output, LABEL_DELAY_FRAMES)
        if emitted_stop > selected_start:
            groups.append(probabilities[selected_start:emitted_stop].mean(axis=0))
            times.append(
                ((input_stop - 1) * HOP_LENGTH + WIN_LENGTH) / SAMPLE_RATE
            )
        next_output = emitted_stop
    if next_output != stable_stop or not groups:
        raise AssertionError("streaming emission schedule did not close exactly")
    return np.stack(groups), np.asarray(times, dtype=np.float64), probabilities


def monolingual_stability(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    items: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    total_seconds = float(sum(item["duration_seconds"] for item in items))
    raw_changes = 0
    committed_changes = 0
    initial_commits = 0
    correct_initial_commits = 0
    per_clip: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        chunks, times, _ = chunk_trace(logits[index], int(lengths[index]))
        classes = chunks.argmax(axis=-1)
        clip_raw_changes = int(np.sum(classes[1:] != classes[:-1]))
        router = generic_router(chunks, times)
        initial = router["initial_commit"]
        initial_commits += int(initial is not None)
        correct_initial_commits += int(
            initial is not None and initial["language"] == item["language"]
        )
        raw_changes += clip_raw_changes
        committed_changes += int(router["committed_changes"])
        per_clip.append(
            {
                "clip_id": item["id"],
                "expected": item["language"],
                "raw_top1_changes": clip_raw_changes,
                "committed_changes": int(router["committed_changes"]),
                "initial_commit": initial,
            }
        )
    return {
        "audio_seconds": total_seconds,
        "raw_top1_changes": raw_changes,
        "raw_top1_changes_per_minute": raw_changes / (total_seconds / 60),
        "committed_false_switches": committed_changes,
        "committed_false_switches_per_hour": committed_changes
        / (total_seconds / 3_600),
        "initial_commit_clips": initial_commits,
        "correct_initial_commit_clips": correct_initial_commits,
        "no_initial_commit_clips": len(items) - initial_commits,
        "per_clip": per_clip,
    }


def switch_clip_result(
    logits: torch.Tensor,
    length: int,
    targets: torch.Tensor,
    item: dict[str, Any],
) -> dict[str, Any]:
    chunks, times, frame_probabilities = chunk_trace(logits, length)
    source_language = item["segments"][0]["language"]
    target_language = item["segments"][1]["language"]
    source_index = LANGUAGE_CODES.index(source_language)
    target_index = LANGUAGE_CODES.index(target_language)
    boundary = float(item["segments"][0]["end_seconds"])
    raw_classes = chunks.argmax(axis=-1)
    raw_event = raw_switch_event(
        raw_classes,
        times,
        source_index=source_index,
        target_index=target_index,
        boundary_seconds=boundary,
    )
    router = generic_router(chunks, times)
    policy_event = policy_switch_event(
        router,
        source_language=source_language,
        target_language=target_language,
        boundary_seconds=boundary,
    )

    aligned = frame_probabilities[LABEL_DELAY_FRAMES:]
    semantic_times = (
        np.arange(len(aligned), dtype=np.float64) * HOP_LENGTH + WIN_LENGTH
    ) / SAMPLE_RATE
    expected = np.where(
        semantic_times < boundary, source_index, target_index
    ).astype(np.int64)
    predicted = aligned.argmax(axis=-1)
    keep = np.abs(semantic_times - boundary) > SWITCH_COLLAR_SECONDS
    if not keep.any():
        raise AssertionError("switch collar removed every semantic frame")
    no_collar_correct = int((predicted[keep] == expected[keep]).sum())
    no_collar_frames = int(keep.sum())

    loss, _ = delayed_distillation_loss(
        logits[:length].unsqueeze(0),
        targets[:length].unsqueeze(0),
        torch.tensor([length], dtype=torch.long),
        delay_frames=LABEL_DELAY_FRAMES,
        lookahead_frames=MODEL_LOOKAHEAD_FRAMES,
        temperature=TEACHER_TEMPERATURE,
        early_ramp_frames=EARLY_RAMP_FRAMES,
    )
    return {
        "clip_id": item["id"],
        "direction": f"{source_language}->{target_language}",
        "boundary_seconds": boundary,
        "raw_event": raw_event,
        "policy_event": policy_event,
        "raw_top1_changes": int(np.sum(raw_classes[1:] != raw_classes[:-1])),
        "policy_commit_events": router["events"],
        "no_collar": {
            "correct": no_collar_correct,
            "frames": no_collar_frames,
            "accuracy": no_collar_correct / no_collar_frames,
        },
        "distillation_loss": float(loss),
    }


def aggregate_switch(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total != 2:
        raise AssertionError(f"expected two switch clips, found {total}")
    raw_recall = {}
    policy_recall = {}
    for deadline in SWITCH_DEADLINES_SECONDS:
        key = prefix_key(deadline)
        raw_matches = sum(
            row["raw_event"]["detected"]
            and row["raw_event"]["first_stable_lag_ms"] <= 1_000 * deadline
            for row in rows
        )
        policy_matches = sum(
            row["policy_event"]["detected"]
            and row["policy_event"]["lag_ms"] <= 1_000 * deadline
            for row in rows
        )
        raw_recall[key] = {
            "matched": int(raw_matches),
            "total": total,
            "recall": raw_matches / total,
        }
        policy_recall[key] = {
            "matched": int(policy_matches),
            "total": total,
            "recall": policy_matches / total,
        }
    raw_penalties = [
        (
            max(0.0, float(row["raw_event"]["first_stable_lag_ms"]))
            if row["raw_event"]["detected"]
            else MISS_PENALTY_MS
        )
        for row in rows
    ]
    correct = sum(row["no_collar"]["correct"] for row in rows)
    frames = sum(row["no_collar"]["frames"] for row in rows)
    total_seconds = sum(2 * row["boundary_seconds"] for row in rows)
    raw_changes = sum(row["raw_top1_changes"] for row in rows)
    return {
        "n_switches": total,
        "raw_source_preconditions": sum(
            row["raw_event"]["source_stable_before_boundary"] for row in rows
        ),
        "policy_source_preconditions": sum(
            row["policy_event"]["source_committed_at_boundary"] for row in rows
        ),
        "raw_detected": sum(row["raw_event"]["detected"] for row in rows),
        "policy_detected": sum(row["policy_event"]["detected"] for row in rows),
        "raw_recall_by_first_stable_onset_deadline_seconds": raw_recall,
        "policy_recall_by_commit_deadline_seconds": policy_recall,
        "raw_miss_penalized_mean_lag_ms": float(np.mean(raw_penalties)),
        "miss_penalty_ms": MISS_PENALTY_MS,
        "no_collar_correct": int(correct),
        "no_collar_frames": int(frames),
        "no_collar_accuracy": correct / frames,
        "raw_top1_changes": int(raw_changes),
        "raw_top1_changes_per_minute": raw_changes / (total_seconds / 60),
        "mean_switch_distillation_loss": float(
            np.mean([row["distillation_loss"] for row in rows])
        ),
        "rows": list(rows),
    }


def full_frame_metrics(
    logits: torch.Tensor,
    batch: Mapping[str, Any],
    teacher_probabilities: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    clip_rows: list[dict[str, Any]] = []
    for index, item in enumerate(batch["items"]):
        length = int(batch["lengths"][index])
        valid_targets = length - LABEL_DELAY_FRAMES - MODEL_LOOKAHEAD_FRAMES
        if valid_targets <= 0:
            raise ValueError(f"clip {item['id']} has no aligned frames")
        student = logits[
            index,
            LABEL_DELAY_FRAMES : LABEL_DELAY_FRAMES + valid_targets,
        ].argmax(dim=-1)
        teacher = teacher_probabilities[item["id"]][:valid_targets].argmax(dim=-1)
        expected = LANGUAGE_CODES.index(item["language"])
        loss, _ = delayed_distillation_loss(
            logits[index, :length].unsqueeze(0),
            batch["targets"][index, :length].unsqueeze(0),
            torch.tensor([length], dtype=torch.long),
            delay_frames=LABEL_DELAY_FRAMES,
            lookahead_frames=MODEL_LOOKAHEAD_FRAMES,
            temperature=TEACHER_TEMPERATURE,
            early_ramp_frames=EARLY_RAMP_FRAMES,
        )
        clip_rows.append(
            {
                "clip_id": item["id"],
                "language": item["language"],
                "frames": valid_targets,
                "known_label_accuracy": float((student == expected).float().mean()),
                "teacher_agreement": float((student == teacher).float().mean()),
                "teacher_known_label_accuracy": float(
                    (teacher == expected).float().mean()
                ),
                "distillation_loss": float(loss),
            }
        )

    per_language: dict[str, Any] = {}
    for language in LANGUAGE_CODES:
        selected = [row for row in clip_rows if row["language"] == language]
        if not selected:
            raise AssertionError(f"no full-frame rows for {language}")
        per_language[language] = {
            "n_clips": len(selected),
            "known_label_accuracy": float(
                np.mean([row["known_label_accuracy"] for row in selected])
            ),
            "teacher_agreement": float(
                np.mean([row["teacher_agreement"] for row in selected])
            ),
            "teacher_known_label_accuracy": float(
                np.mean(
                    [row["teacher_known_label_accuracy"] for row in selected]
                )
            ),
            "distillation_loss": float(
                np.mean([row["distillation_loss"] for row in selected])
            ),
        }
    return {
        "clip_language_macro_known_label_accuracy": float(
            np.mean(
                [value["known_label_accuracy"] for value in per_language.values()]
            )
        ),
        "clip_language_macro_teacher_agreement": float(
            np.mean([value["teacher_agreement"] for value in per_language.values()])
        ),
        "clip_language_macro_teacher_known_label_accuracy": float(
            np.mean(
                [
                    value["teacher_known_label_accuracy"]
                    for value in per_language.values()
                ]
            )
        ),
        "clip_language_macro_distillation_loss": float(
            np.mean([value["distillation_loss"] for value in per_language.values()])
        ),
        "per_language": per_language,
    }


def evaluate_snapshot(
    model: CausalLIDStudent,
    state: Mapping[str, torch.Tensor],
    *,
    step: int,
    state_sha256: str,
    losses: Sequence[float],
    heldout_batch: Mapping[str, Any],
    prefix_batches: Mapping[str, Mapping[str, Any]],
    train_batch: Mapping[str, Any],
    exact_control_indices: Sequence[int],
    switch_batch: Mapping[str, Any],
    teacher_probabilities: Mapping[str, torch.Tensor],
    check_streaming_equivalence: bool,
) -> tuple[dict[str, Any], bool]:
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        heldout_logits = model(heldout_batch["features"])
        train_logits = model(train_batch["features"])
        switch_logits = model(switch_batch["features"])
        prefix_logits = {
            key: model(batch["features"])
            for key, batch in prefix_batches.items()
            if key != "full"
        }

        if check_streaming_equivalence:
            length = int(heldout_batch["lengths"][0])
            features = heldout_batch["features"][0:1, :length]
            streamed = model.streaming_forward(
                features, chunk_frames=CHUNK_FRAMES
            ).squeeze(0)
            torch.testing.assert_close(
                heldout_logits[0, : length - MODEL_LOOKAHEAD_FRAMES],
                streamed,
                rtol=1e-5,
                atol=1e-5,
            )
            check_streaming_equivalence = False

    prefix_summaries: dict[str, Any] = {}
    for key, batch in prefix_batches.items():
        logits = heldout_logits if key == "full" else prefix_logits[key]
        prefix_summaries[key] = classification_summary(
            posterior_rows(logits, batch["lengths"], batch["items"])
        )

    label_composite = float(
        np.mean(
            [
                prefix_summaries[key]["language_macro_accuracy"]
                for key in SELECTION_PREFIXES
            ]
        )
    )
    per_language_composite = {
        language: float(
            np.mean(
                [
                    prefix_summaries[key]["per_language_recall"][language]
                    for key in SELECTION_PREFIXES
                ]
            )
        )
        for language in LANGUAGE_CODES
    }

    train_rows = posterior_rows(
        train_logits, train_batch["lengths"], train_batch["items"]
    )
    exact_rows = [train_rows[index] for index in exact_control_indices]
    switch_rows = [
        switch_clip_result(
            switch_logits[index],
            int(switch_batch["lengths"][index]),
            switch_batch["targets"][index],
            item,
        )
        for index, item in enumerate(switch_batch["items"])
    ]
    switch_summary = aggregate_switch(switch_rows)
    stability = monolingual_stability(
        heldout_logits,
        heldout_batch["lengths"],
        heldout_batch["items"],
    )
    frame_metrics = full_frame_metrics(
        heldout_logits, heldout_batch, teacher_probabilities
    )
    training_window = (
        None
        if step == 0
        else float(np.mean(losses[max(0, step - SNAPSHOT_INTERVAL) : step]))
    )
    return (
        {
            "step": step,
            "model_state_sha256": state_sha256,
            "training": {
                "loss_at_step": None if step == 0 else float(losses[step - 1]),
                "previous_10_mean_loss": training_window,
                "cumulative_mean_loss": (
                    None if step == 0 else float(np.mean(losses[:step]))
                ),
            },
            "synthetic_dev": {
                "label_selection_composite_1s_2s_full": label_composite,
                "minimum_language_recall_composite_1s_2s_full": float(
                    min(per_language_composite.values())
                ),
                "per_language_recall_composite_1s_2s_full": per_language_composite,
                "prefix": prefix_summaries,
                "full_frame": frame_metrics,
                "monolingual_stability": stability,
            },
            "familiar_audio": {
                "all_70_monolingual_train_full": classification_summary(train_rows),
                "edge_indices_5_6_7_full": classification_summary(exact_rows),
            },
            "switch": switch_summary,
        },
        check_streaming_equivalence,
    )


def metric_row(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step": snapshot["step"],
        "model_state_sha256": snapshot["model_state_sha256"],
        "label_composite": snapshot["synthetic_dev"][
            "label_selection_composite_1s_2s_full"
        ],
        "minimum_language_recall": snapshot["synthetic_dev"][
            "minimum_language_recall_composite_1s_2s_full"
        ],
        "raw_switch_recall_at_1s": snapshot["switch"][
            "raw_recall_by_first_stable_onset_deadline_seconds"
        ]["1"]["recall"],
        "raw_miss_penalized_mean_lag_ms": snapshot["switch"][
            "raw_miss_penalized_mean_lag_ms"
        ],
        "switch_no_collar_accuracy": snapshot["switch"]["no_collar_accuracy"],
        "switch_raw_changes_per_minute": snapshot["switch"][
            "raw_top1_changes_per_minute"
        ],
        "monolingual_committed_false_switches": snapshot["synthetic_dev"][
            "monolingual_stability"
        ]["committed_false_switches"],
        "monolingual_committed_false_switches_per_hour": snapshot[
            "synthetic_dev"
        ]["monolingual_stability"]["committed_false_switches_per_hour"],
        "development_distillation_loss": snapshot["synthetic_dev"][
            "full_frame"
        ]["clip_language_macro_distillation_loss"],
        "exact_edge_control_accuracy": snapshot["familiar_audio"][
            "edge_indices_5_6_7_full"
        ]["accuracy"],
        "all_train_accuracy": snapshot["familiar_audio"][
            "all_70_monolingual_train_full"
        ]["accuracy"],
    }


def best_row(
    rows: Sequence[Mapping[str, Any]],
    key,
) -> Mapping[str, Any]:
    return min(rows, key=key)


def shortlist_and_select(
    snapshots: Sequence[Mapping[str, Any]], final_step: int
) -> dict[str, Any]:
    rows = [metric_row(snapshot) for snapshot in snapshots]
    by_step = {row["step"]: row for row in rows}
    final = by_step[final_step]
    midpoint_step = final_step // 2
    categories = [
        (
            "best_monolingual_label_composite",
            best_row(
                rows,
                lambda row: (-row["label_composite"], row["step"]),
            ),
        ),
        (
            "best_raw_switch",
            best_row(
                rows,
                lambda row: (
                    -row["raw_switch_recall_at_1s"],
                    row["raw_miss_penalized_mean_lag_ms"],
                    row["step"],
                ),
            ),
        ),
        (
            "lowest_development_kd",
            best_row(
                rows,
                lambda row: (row["development_distillation_loss"], row["step"]),
            ),
        ),
        ("final_step", final),
        ("fixed_midpoint", by_step[midpoint_step]),
    ]
    shortlist: list[dict[str, Any]] = []
    seen: set[int] = set()
    for reason, row in categories:
        if row["step"] in seen:
            continue
        seen.add(row["step"])
        shortlist.append({"shortlist_reason": reason, **row})

    best_label = max(row["label_composite"] for row in shortlist)
    best_minimum_recall = max(
        row["minimum_language_recall"] for row in shortlist
    )
    for row in shortlist:
        row["eligibility"] = {
            "within_2pp_of_best_label_composite": row["label_composite"]
            >= best_label - LABEL_ELIGIBILITY_TOLERANCE,
            "within_5pp_of_best_minimum_language_recall": row[
                "minimum_language_recall"
            ]
            >= best_minimum_recall - WORST_LANGUAGE_TOLERANCE,
            "monolingual_false_switches_no_worse_than_final": row[
                "monolingual_committed_false_switches"
            ]
            <= final["monolingual_committed_false_switches"],
        }
        row["eligible"] = all(row["eligibility"].values())
    eligible = [row for row in shortlist if row["eligible"]]
    if not eligible:
        raise AssertionError("predeclared selector found no eligible checkpoint")
    selected = min(
        eligible,
        key=lambda row: (
            -row["raw_switch_recall_at_1s"],
            row["raw_miss_penalized_mean_lag_ms"],
            1.0 - row["switch_no_collar_accuracy"],
            row["switch_raw_changes_per_minute"],
            row["development_distillation_loss"],
            row["step"],
        ),
    )
    comparison = {
        "selected_step": selected["step"],
        "final_step": final_step,
        "label_composite_gain_pp": 100
        * (selected["label_composite"] - final["label_composite"]),
        "raw_switch_recall_at_1s_gain_pp": 100
        * (
            selected["raw_switch_recall_at_1s"]
            - final["raw_switch_recall_at_1s"]
        ),
        "monolingual_false_switch_delta": selected[
            "monolingual_committed_false_switches"
        ]
        - final["monolingual_committed_false_switches"],
        "exact_edge_control_accuracy_gain_pp": 100
        * (
            selected["exact_edge_control_accuracy"]
            - final["exact_edge_control_accuracy"]
        ),
        "all_train_accuracy_gain_pp": 100
        * (selected["all_train_accuracy"] - final["all_train_accuracy"]),
    }
    gates = {
        "selected_is_earlier_than_final": selected["step"] < final_step,
        "label_or_switch_gain_is_material": comparison[
            "label_composite_gain_pp"
        ]
        >= LABEL_GAIN_GATE_PP
        or comparison["raw_switch_recall_at_1s_gain_pp"]
        >= SWITCH_RECALL_GAIN_GATE_PP,
        "monolingual_committed_false_switches_no_worse": comparison[
            "monolingual_false_switch_delta"
        ]
        <= 0,
    }
    return {
        "metric_rows": rows,
        "shortlist": shortlist,
        "selection_scope": (
            "provisional synthetic_dev Tier-A selection only; external Tier-B "
            "validation was not read"
        ),
        "selector_order": [
            "higher raw stable-switch recall at 1 s",
            "lower miss-penalized raw switch lag",
            "lower no-collar switch error",
            "lower raw switch changes per minute",
            "lower macro development KD",
            "earlier optimizer step",
        ],
        "selected": copy.deepcopy(selected),
        "final": copy.deepcopy(final),
        "comparison": comparison,
        "useful_change_gates": gates,
        "useful_change": all(gates.values()),
    }


def assert_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            assert_finite_json(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            assert_finite_json(nested, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"non-finite result at {path}")


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_write(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    wall_started = time.perf_counter()
    launched_at = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)

    source_snapshot_at_start = snapshot_sources()
    reference_hashes_at_start = {
        "checkpoint": file_sha256(args.reference_checkpoint),
        "summary": file_sha256(args.reference_summary),
    }
    reference_checkpoint = torch.load(
        args.reference_checkpoint, map_location="cpu", weights_only=True
    )
    reference_summary = json.loads(
        args.reference_summary.read_text(encoding="utf-8")
    )
    reference_model_hash = model_state_sha256(reference_checkpoint["model_state"])
    if reference_model_hash != reference_summary["model_state_sha256"]:
        raise ValueError("reference checkpoint and summary model hashes differ")
    if reference_checkpoint["run_id"] != reference_summary["run_id"]:
        raise ValueError("reference checkpoint and summary run IDs differ")

    records = read_manifest(args.manifest)
    speaker_audit = require_speaker_disjoint(records)
    target_cache = TeacherTargetCache(args.manifest, args.targets_dir)
    training_settings = training_configuration(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        threads=args.threads,
        seed=args.seed,
    )
    dependency_snapshot = capture_run_dependency_snapshot(
        records=records,
        manifest_path=args.manifest,
        target_cache_identity=target_cache.identity,
        target_metadata_path=target_cache.targets_dir / "metadata.json",
        training=training_settings,
    )
    if dependency_snapshot != reference_checkpoint["launch_dependency_snapshot"]:
        raise RuntimeError(
            "current pipeline/data/targets/settings differ from the reference trajectory"
        )
    target_cache.validate_all()
    assert_run_dependency_snapshot_unchanged(
        dependency_snapshot,
        records=records,
        manifest_path=args.manifest,
        target_cache_identity=target_cache.identity,
        target_metadata_path=target_cache.targets_dir / "metadata.json",
        training=training_settings,
        stage="experiment dataset preload",
    )
    train_dataset = DistillationDataset(
        args.manifest,
        args.targets_dir,
        splits=("train",),
        target_cache=target_cache,
        records=records,
    )
    validate_full_batch_training_set(len(train_dataset), args.batch_size)
    assert_run_dependency_snapshot_unchanged(
        dependency_snapshot,
        records=records,
        manifest_path=args.manifest,
        target_cache_identity=target_cache.identity,
        target_metadata_path=target_cache.targets_dir / "metadata.json",
        training=training_settings,
        stage="experiment optimization",
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_distillation_batch,
        generator=loader_generator,
        num_workers=0,
        drop_last=True,
    )
    model = CausalLIDStudent(**configured_model_kwargs())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY
    )
    snapshots: dict[int, dict[str, torch.Tensor]] = {0: clone_state_dict(model)}
    snapshot_hashes: dict[int, str] = {
        0: model_state_sha256(snapshots[0])
    }
    losses: list[float] = []
    gradient_norms: list[float] = []
    batch_order_digest = hashlib.sha256()
    examples_seen = 0
    successful_steps = 0
    model.train()
    print(
        f"training exact {model.parameter_count:,}-parameter trajectory; "
        f"snapshots=0..{args.steps}/{SNAPSHOT_INTERVAL}",
        flush=True,
    )
    while successful_steps < args.steps:
        for batch in loader:
            step = successful_steps + 1
            batch_order_digest.update(
                json.dumps(batch["ids"], separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["features"])
            loss, _ = delayed_distillation_loss(
                logits,
                batch["targets"],
                batch["lengths"],
                delay_frames=LABEL_DELAY_FRAMES,
                lookahead_frames=MODEL_LOOKAHEAD_FRAMES,
                temperature=TEACHER_TEMPERATURE,
                early_ramp_frames=EARLY_RAMP_FRAMES,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            if not all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                raise FloatingPointError(f"non-finite gradient at step {step}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP_NORM,
                error_if_nonfinite=True,
            )
            optimizer.step()
            assert_finite_post_update_state(model, optimizer, step=step)
            successful_steps += 1
            examples_seen += len(batch["ids"])
            losses.append(float(loss.detach()))
            gradient_norms.append(float(gradient_norm))
            if successful_steps % SNAPSHOT_INTERVAL == 0:
                state = clone_state_dict(model)
                snapshots[successful_steps] = state
                snapshot_hashes[successful_steps] = model_state_sha256(state)
            if successful_steps == 1 or successful_steps % 100 == 0:
                print(
                    f"step={successful_steps:04d} loss={losses[-1]:.6f} "
                    f"snapshots={len(snapshots)}",
                    flush=True,
                )
            if successful_steps >= args.steps:
                break

    training_contract = build_training_contract(
        requested_steps=args.steps,
        successful_steps=successful_steps,
        post_update_checks=successful_steps,
        examples_seen=examples_seen,
        losses=losses,
        gradient_norms=gradient_norms,
        model=model,
        optimizer=optimizer,
    )
    if not training_contract["nan_free"]:
        raise RuntimeError(f"training contract failed: {training_contract}")
    expected_steps = list(range(0, args.steps + 1, SNAPSHOT_INTERVAL))
    if sorted(snapshots) != expected_steps:
        raise AssertionError("trajectory snapshot cadence is incomplete")
    final_hash = snapshot_hashes[args.steps]
    final_model_exact = final_hash == reference_model_hash
    loss_trace_exact = losses == reference_checkpoint["training_evidence"]["losses"]
    gradient_trace_exact = (
        gradient_norms
        == reference_checkpoint["training_evidence"]["gradient_norms"]
    )
    if not final_model_exact or not loss_trace_exact or not gradient_trace_exact:
        raise RuntimeError(
            "replayed trajectory did not exactly reproduce the reference final state/trace"
        )
    print(
        f"exact final reproduction passed: {final_hash[:12]}...; "
        "preparing fixed evaluation tensors",
        flush=True,
    )

    heldout_dataset = DistillationDataset(
        args.manifest,
        args.targets_dir,
        splits=("heldout",),
        target_cache=target_cache,
        records=records,
    )
    switch_dataset = DistillationDataset(
        args.manifest,
        args.targets_dir,
        splits=("switch",),
        target_cache=target_cache,
        records=records,
    )
    heldout_batch = {
        **collate_distillation_batch(heldout_dataset.examples),
        "items": [example["item"] for example in heldout_dataset.examples],
    }
    switch_batch = {
        **collate_distillation_batch(switch_dataset.examples),
        "items": [example["item"] for example in switch_dataset.examples],
    }
    monolingual_train_examples = [
        example
        for example in train_dataset.examples
        if example["item"]["language"] in LANGUAGE_CODES
    ]
    train_batch = {
        **collate_distillation_batch(monolingual_train_examples),
        "items": [example["item"] for example in monolingual_train_examples],
    }
    exact_control_indices = [
        index
        for index, item in enumerate(train_batch["items"])
        if int(item["id"].rsplit("_", 1)[1]) in EXACT_EDGE_TRAIN_INDICES
    ]
    if len(exact_control_indices) != 21:
        raise AssertionError("expected 21 exact Edge training controls")

    frontend = LogMelFrontend().eval()
    prefix_batches: dict[str, dict[str, Any]] = {"full": heldout_batch}
    with torch.inference_mode():
        for seconds in PREFIX_SECONDS:
            if seconds == "full":
                continue
            examples = []
            for item in heldout_batch["items"]:
                waveform = load_audio(resolve_audio_path(item, args.manifest))
                features = frontend(fixed_prefix(waveform, float(seconds))).squeeze(0)
                examples.append({"features": features, "item": item})
            prefix_batches[prefix_key(seconds)] = feature_batch(examples)
    if set(prefix_batches) != {"0.5", "1", "2", "4", "full"}:
        raise AssertionError("prefix evaluation grid is incomplete")
    teacher_probabilities = {
        item["id"]: torch.from_numpy(
            target_cache.load(
                item,
                "teacher_probs",
                expected_frames=int(heldout_batch["lengths"][index]),
            )
        )
        for index, item in enumerate(heldout_batch["items"])
    }

    curve: list[dict[str, Any]] = []
    needs_streaming_check = True
    for position, step in enumerate(expected_steps):
        result, needs_streaming_check = evaluate_snapshot(
            model,
            snapshots[step],
            step=step,
            state_sha256=snapshot_hashes[step],
            losses=losses,
            heldout_batch=heldout_batch,
            prefix_batches=prefix_batches,
            train_batch=train_batch,
            exact_control_indices=exact_control_indices,
            switch_batch=switch_batch,
            teacher_probabilities=teacher_probabilities,
            check_streaming_equivalence=needs_streaming_check,
        )
        curve.append(result)
        if position == 0 or (position + 1) % 10 == 0 or step == args.steps:
            print(
                f"evaluated step={step:04d} ({position + 1}/{len(expected_steps)}) "
                f"label={result['synthetic_dev']['label_selection_composite_1s_2s_full']:.4f} "
                f"raw@1s={result['switch']['raw_recall_by_first_stable_onset_deadline_seconds']['1']['recall']:.3f}",
                flush=True,
            )
    if needs_streaming_check:
        raise AssertionError("stable streaming/full equivalence was not checked")

    selection = shortlist_and_select(curve, args.steps)
    useful_change = selection["useful_change"]
    verdict = "adopt" if useful_change else "reject"
    if useful_change:
        verdict_reason = (
            "an earlier snapshot passed the predeclared material label/switch gain "
            "and monolingual false-switch gates; adopt trajectory checkpointing and "
            "advance the candidate to external validation, not directly to release"
        )
    else:
        failed = [
            name
            for name, passed in selection["useful_change_gates"].items()
            if not passed
        ]
        verdict_reason = (
            "checkpoint selection did not produce a useful earlier state under the "
            "predeclared local gates: " + ", ".join(failed)
        )

    target_cache.validate_all()
    assert_run_dependency_snapshot_unchanged(
        dependency_snapshot,
        records=records,
        manifest_path=args.manifest,
        target_cache_identity=target_cache.identity,
        target_metadata_path=target_cache.targets_dir / "metadata.json",
        training=training_settings,
        stage="experiment artifact publication",
    )
    if snapshot_sources() != source_snapshot_at_start:
        raise RuntimeError("experiment or imported source changed during the run")
    reference_hashes_at_end = {
        "checkpoint": file_sha256(args.reference_checkpoint),
        "summary": file_sha256(args.reference_summary),
    }
    if reference_hashes_at_end != reference_hashes_at_start:
        raise RuntimeError("reference checkpoint or summary changed during the run")

    trajectory_payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "reference_run_id": reference_checkpoint["run_id"],
        "steps": expected_steps,
        "model_state_sha256_by_step": {
            str(step): snapshot_hashes[step] for step in expected_steps
        },
        "model_states": {str(step): snapshots[step] for step in expected_steps},
    }
    atomic_torch_save(trajectory_payload, args.trajectory)
    trajectory_sha256 = file_sha256(args.trajectory)

    experiment_configuration = {
        "analysis_version": ANALYSIS_VERSION,
        "steps": args.steps,
        "snapshot_interval_steps": SNAPSHOT_INTERVAL,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "seed": args.seed,
        "threads": args.threads,
        "prefix_seconds": [0.5, 1, 2, 4, "full"],
        "selection_prefixes": list(SELECTION_PREFIXES),
        "router": {
            "threshold": POLICY_THRESHOLD,
            "margin": POLICY_MARGIN,
            "dwell_chunks": POLICY_DWELL_CHUNKS,
            "ema_new_weight": EMA_NEW_WEIGHT,
        },
        "streaming_grouping": (
            "actual input-call output ranges (12, then 16 for full calls at "
            "chunk=16/lookahead=4), filtered at label-delay frame 21"
        ),
        "raw_stability_seconds": RAW_STABILITY_SECONDS,
        "switch_deadlines_seconds": list(SWITCH_DEADLINES_SECONDS),
        "switch_collar_seconds": SWITCH_COLLAR_SECONDS,
        "miss_penalty_ms": MISS_PENALTY_MS,
        "decision_gates": {
            "label_gain_pp": LABEL_GAIN_GATE_PP,
            "raw_switch_recall_gain_pp": SWITCH_RECALL_GAIN_GATE_PP,
            "monolingual_committed_false_switch_delta_max": 0,
        },
        "external_validation_read": False,
        "locked_test_read": False,
    }
    experiment_identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "configuration": experiment_configuration,
        "source_files": source_snapshot_at_start,
        "dependency_snapshot": dependency_snapshot,
        "reference_hashes": reference_hashes_at_start,
        "reference_run_id": reference_checkpoint["run_id"],
        "batch_order_sha256": batch_order_digest.hexdigest(),
        "model_state_sha256_by_step": {
            str(step): snapshot_hashes[step] for step in expected_steps
        },
        "trajectory_sha256": trajectory_sha256,
    }
    run_id = canonical_sha256(experiment_identity)
    results = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "question": (
            "Does an earlier state in the exact 1,600-step causal-TCN trajectory "
            "improve fixed synthetic-development label or switch performance over "
            "the unselected final iterate without more monolingual false switches?"
        ),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "run_id": run_id,
        "launched_at_utc": launched_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - wall_started,
        "experiment_identity": experiment_identity,
        "configuration": experiment_configuration,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "torch_threads": torch.get_num_threads(),
        },
        "data": {
            "role": "synthetic_dev",
            "n_train_clips": len(train_dataset),
            "n_train_monolingual_clips": len(monolingual_train_examples),
            "n_heldout_monolingual_clips": len(heldout_dataset),
            "n_switch_clips": len(switch_dataset),
            "heldout_clip_ids": [item["id"] for item in heldout_batch["items"]],
            "switch_clip_ids": [item["id"] for item in switch_batch["items"]],
            "speaker_audit": speaker_audit,
            "target_cache": target_cache.audit(),
        },
        "training": {
            "trajectory_completed_steps": successful_steps,
            "examples_seen": examples_seen,
            "effective_epochs": examples_seen / len(train_dataset),
            "contract": training_contract,
            "losses": losses,
            "gradient_norms": gradient_norms,
            "first_10_mean_loss": float(np.mean(losses[:10])),
            "last_10_mean_loss": float(np.mean(losses[-10:])),
            "batch_order_sha256": batch_order_digest.hexdigest(),
            "snapshot_count": len(snapshots),
            "snapshot_steps": expected_steps,
            "model_state_sha256_by_step": {
                str(step): snapshot_hashes[step] for step in expected_steps
            },
        },
        "identity_audit": {
            "dependency_snapshot_unchanged": True,
            "source_snapshot_unchanged": True,
            "reference_bundle_unchanged": True,
            "target_cache_revalidated_at_end": True,
            "stable_streaming_full_equivalence_checked": True,
            "reference_final_model_state_exactly_reproduced": final_model_exact,
            "reference_loss_trace_exactly_reproduced": loss_trace_exact,
            "reference_gradient_trace_exactly_reproduced": gradient_trace_exact,
            "reference_model_state_sha256": reference_model_hash,
        },
        "trajectory_artifact": {
            "path": str(args.trajectory),
            "sha256": trajectory_sha256,
            "contains_optimizer_state": False,
            "gitignored_model_weights": True,
        },
        "curve": curve,
        "selection": selection,
        "decision": {
            "verdict": verdict,
            "reason": verdict_reason,
            "gates": selection["useful_change_gates"],
            "selected_step": selection["selected"]["step"],
            "final_step": args.steps,
            "selected_snapshot_is_release_checkpoint": False,
            "required_next_gate": (
                "rank the at-most-five shortlist on pinned external validation "
                "before any main checkpoint replacement"
            ),
        },
        "limitations": [
            "one deterministic seed and one 71-clip synthetic training corpus",
            "the 21 repeatedly consulted held-out clips are synthetic development, not untouched test",
            "the two switch directions reverse one Hindi/English source pair and are not independent trials",
            "no external validation or locked test data were read",
            "the current availability-valid teacher trajectory remains too slow and is unchanged",
            "the selected local snapshot is a shortlist candidate, not a release checkpoint",
        ],
    }
    assert_finite_json(results)
    atomic_json_write(results, args.output)
    print(
        f"wrote {args.output}; verdict={verdict}; "
        f"selected={selection['selected']['step']} final={args.steps}; run_id={run_id}",
        flush=True,
    )


if __name__ == "__main__":
    main()
