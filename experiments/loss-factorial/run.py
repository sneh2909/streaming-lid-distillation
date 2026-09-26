#!/usr/bin/env python3
"""Identity-matched loss factorial for teacher-bias/class-collapse diagnosis.

The four causal-TCN students share initialization, minibatch order, optimizer
settings, cached targets, and checkpoint cadence.  Only loss aggregation or
the use of known monolingual labels changes:

* ``frame_kd``: the main pipeline's frame-normalized pure KD objective;
* ``equal_clip_kd``: pure KD normalized within each clip, then averaged;
* ``hybrid_kd_ce``: equal-clip 0.5 KD + 0.5 hard CE on monolingual clips;
* ``teacher_error_gate``: equal-clip KD except hard CE where the T=1 teacher
  top-1 disagrees with the known monolingual synthesis label.

The mixed training clip retains availability-valid local KD in every arm.
The same 21 held-out monolingual clips and both switch clips as the main
pipeline are scored at every ten-step checkpoint.

Run from the repository root with::

    UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u \
      experiments/loss-factorial/run.py --fresh
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# Reuse the already exercised main-pipeline evaluation primitives from the
# trajectory experiment.  The helper itself imports the main scheduler,
# policy, frontend, model, and loss rather than duplicating them here.
TRAJECTORY_DRIVER = REPO_ROOT / "experiments/checkpoint-trajectory/run.py"
_trajectory_spec = importlib.util.spec_from_file_location(
    "checkpoint_trajectory_helpers", TRAJECTORY_DRIVER
)
if _trajectory_spec is None or _trajectory_spec.loader is None:
    raise ImportError(f"cannot load trajectory helpers from {TRAJECTORY_DRIVER}")
trajectory = importlib.util.module_from_spec(_trajectory_spec)
sys.modules[_trajectory_spec.name] = trajectory
_trajectory_spec.loader.exec_module(trajectory)


# Import training, data, timing, loss, model, and identity contracts from the
# main pipeline.  This experiment never edits those sources.
from scripts.train import (  # noqa: E402
    assert_finite_post_update_state,
    build_training_contract,
    seed_everything,
    validate_full_batch_training_set,
)
from streaming_lid.audio import LogMelFrontend, load_audio  # noqa: E402
from streaming_lid.config import (  # noqa: E402
    ALGORITHMIC_LATENCY_MS,
    CHUNK_FRAMES,
    EARLY_RAMP_FRAMES,
    LABEL_DELAY_FRAMES,
    LANGUAGE_CODES,
    MODEL_LOOKAHEAD_FRAMES,
    SAMPLE_RATE,
    TEACHER_TEMPERATURE,
)
from streaming_lid.data import (  # noqa: E402
    DistillationDataset,
    TeacherTargetCache,
    capture_manifest_snapshot,
    collate_distillation_batch,
    file_sha256,
    require_speaker_disjoint,
    resolve_audio_path,
)
from streaming_lid.loss import alignment_mask, delayed_distillation_loss  # noqa: E402
from streaming_lid.model import CausalLIDStudent  # noqa: E402
from streaming_lid.run_identity import (  # noqa: E402
    assert_run_dependency_snapshot_unchanged,
    capture_run_dependency_snapshot,
    configured_model_kwargs,
    model_state_sha256,
    training_configuration,
)


EXPERIMENT_NAME = "loss-factorial"
SCHEMA_VERSION = 1
ANALYSIS_VERSION = "loss-factorial-v1"
DEFAULT_STEPS = 1_600
DEFAULT_BATCH_SIZE = 7
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_SEED = 7
SNAPSHOT_INTERVAL = 10
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 5.0
PREFIX_SECONDS = (1.0, 2.0)
SELECTION_PREFIXES = ("1", "2", "full")
EXACT_EDGE_TRAIN_INDICES = (5, 6, 7)
LOCAL_ACCURACY_FLOOR = 0.80

ARMS = OrderedDict(
    [
        (
            "frame_kd",
            "current frame-normalized T^2 KL",
        ),
        (
            "equal_clip_kd",
            "per-clip-normalized T^2 KL",
        ),
        (
            "hybrid_kd_ce",
            "per-clip 0.5*T^2*KL + 0.5*CE on monolingual clips",
        ),
        (
            "teacher_error_gate",
            "per-clip hard CE only where monolingual teacher T=1 top-1 is wrong",
        ),
    ]
)

EXPECTED_ENGLISH_TARGET_PRIORS = {
    "frame_kd": 0.06529,
    "equal_clip_kd": 0.06615,
    "hybrid_kd_ce": 0.10486,
    "teacher_error_gate": 0.11748,
}

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
    "experiments/loss-factorial/run.py",
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
        default=Path("experiments/loss-factorial/results.json"),
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Replace results only after every end-of-run identity check passes.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "steps": DEFAULT_STEPS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "seed": DEFAULT_SEED,
        "threads": 6,
    }
    for name, wanted in frozen.items():
        actual = getattr(args, name)
        if actual != wanted:
            raise ValueError(
                f"{name.replace('_', '-')} must be {wanted} for this factorial; "
                f"got {actual}"
            )
    if args.learning_rate != DEFAULT_LEARNING_RATE:
        raise ValueError(
            f"learning-rate must be {DEFAULT_LEARNING_RATE} for this factorial"
        )
    if args.steps % SNAPSHOT_INTERVAL:
        raise ValueError("steps must be divisible by the snapshot interval")
    if args.output.exists() and not args.fresh:
        raise FileExistsError(f"{args.output} exists; pass --fresh to replace it")


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


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def is_monolingual(item: Mapping[str, Any]) -> bool:
    return (
        item.get("language") in LANGUAGE_CODES
        and len(item.get("segments", ())) == 1
    )


def batch_with_items(batch: Mapping[str, Any], items_by_id: Mapping[str, dict]) -> dict:
    result = dict(batch)
    result["items"] = [items_by_id[clip_id] for clip_id in batch["ids"]]
    return result


def collate_examples(examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_distillation_batch(list(examples))
    batch["items"] = [example["item"] for example in examples]
    return batch


def slice_batch(batch: Mapping[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    selected = torch.tensor(list(indices), dtype=torch.long)
    return {
        "features": batch["features"].index_select(0, selected),
        "targets": batch["targets"].index_select(0, selected),
        "lengths": batch["lengths"].index_select(0, selected),
        "ids": [batch["ids"][index] for index in indices],
        "items": [batch["items"][index] for index in indices],
    }


def loss_components(
    student_logits: torch.Tensor,
    teacher_soft_targets: torch.Tensor,
    lengths: torch.Tensor,
    items: Sequence[Mapping[str, Any]],
    teacher_correct_by_id: Mapping[str, bool | None],
) -> dict[str, Any]:
    """Return differentiable per-clip KD/CE terms under the main alignment."""
    if student_logits.shape[:2] != teacher_soft_targets.shape[:2]:
        raise ValueError("student and teacher batch/time dimensions differ")
    if student_logits.shape[-1] != len(LANGUAGE_CODES):
        raise ValueError("unexpected student class count")
    if len(items) != student_logits.shape[0]:
        raise ValueError("item count differs from batch size")

    aligned_frames = student_logits.shape[1] - LABEL_DELAY_FRAMES
    if aligned_frames <= 0:
        raise ValueError("batch is shorter than the label delay")
    valid = alignment_mask(
        lengths,
        aligned_frames,
        LABEL_DELAY_FRAMES,
        MODEL_LOOKAHEAD_FRAMES,
    )
    if not bool(valid.any()):
        raise ValueError("batch has no causally valid frames")
    targets = teacher_soft_targets[:, :aligned_frames]
    predictions = student_logits[
        :, LABEL_DELAY_FRAMES : LABEL_DELAY_FRAMES + aligned_frames
    ]
    log_predictions = F.log_softmax(predictions / TEACHER_TEMPERATURE, dim=-1)
    kd_per_frame = TEACHER_TEMPERATURE**2 * F.kl_div(
        log_predictions, targets, reduction="none"
    ).sum(dim=-1)
    frame_index = torch.arange(
        aligned_frames, device=student_logits.device, dtype=kd_per_frame.dtype
    )
    ramp = ((frame_index + 1.0) / float(EARLY_RAMP_FRAMES)).clamp(max=1.0)
    weights = valid.to(kd_per_frame.dtype) * ramp[None, :]

    clip_kd: list[torch.Tensor] = []
    clip_ce: list[torch.Tensor | None] = []
    valid_frames: list[int] = []
    weight_sums: list[torch.Tensor] = []
    for index, item in enumerate(items):
        denominator = weights[index].sum()
        if not bool(denominator > 0):
            raise ValueError(f"clip {item['id']} has no causally valid frames")
        weight_sums.append(denominator)
        valid_frames.append(int(valid[index].sum().item()))
        clip_kd.append((kd_per_frame[index] * weights[index]).sum() / denominator)
        if is_monolingual(item):
            label = LANGUAGE_CODES.index(str(item["language"]))
            labels = torch.full(
                (aligned_frames,),
                label,
                dtype=torch.long,
                device=student_logits.device,
            )
            ce_per_frame = F.cross_entropy(
                predictions[index], labels, reduction="none"
            )
            clip_ce.append((ce_per_frame * weights[index]).sum() / denominator)
            if teacher_correct_by_id.get(str(item["id"])) is None:
                raise ValueError(f"missing teacher gate for {item['id']}")
        else:
            clip_ce.append(None)

    return {
        "clip_kd": clip_kd,
        "clip_ce": clip_ce,
        "weights": weights,
        "weight_sums": weight_sums,
        "valid_frames": valid_frames,
    }


def factorial_loss(
    arm: str,
    student_logits: torch.Tensor,
    batch: Mapping[str, Any],
    teacher_correct_by_id: Mapping[str, bool | None],
) -> tuple[torch.Tensor, dict[str, float]]:
    if arm not in ARMS:
        raise ValueError(f"unknown factorial arm {arm!r}")
    components = loss_components(
        student_logits,
        batch["targets"],
        batch["lengths"],
        batch["items"],
        teacher_correct_by_id,
    )
    clip_kd: list[torch.Tensor] = components["clip_kd"]
    clip_ce: list[torch.Tensor | None] = components["clip_ce"]

    objective_terms: list[torch.Tensor] = []
    if arm == "frame_kd":
        # Call the production loss directly so this control can reproduce the
        # current checkpoint bit-for-bit rather than merely algebraically.
        loss, main_stats = delayed_distillation_loss(
            student_logits,
            batch["targets"],
            batch["lengths"],
            delay_frames=LABEL_DELAY_FRAMES,
            lookahead_frames=MODEL_LOOKAHEAD_FRAMES,
            temperature=TEACHER_TEMPERATURE,
            early_ramp_frames=EARLY_RAMP_FRAMES,
        )
        objective_terms = clip_kd
    else:
        for item, kd_term, ce_term in zip(
            batch["items"], clip_kd, clip_ce, strict=True
        ):
            if arm == "equal_clip_kd" or not is_monolingual(item):
                term = kd_term
            elif arm == "hybrid_kd_ce":
                if ce_term is None:
                    raise AssertionError("monolingual CE term is missing")
                term = 0.5 * kd_term + 0.5 * ce_term
            else:
                if ce_term is None:
                    raise AssertionError("monolingual CE term is missing")
                teacher_correct = teacher_correct_by_id[str(item["id"])]
                term = kd_term if teacher_correct else ce_term
            objective_terms.append(term)
        loss = torch.stack(objective_terms).mean()
        main_stats = {
            "valid_frames": float(sum(components["valid_frames"])),
            "weight_sum": float(
                sum(value.detach().item() for value in components["weight_sums"])
            ),
        }

    mono_ce = [value for value in clip_ce if value is not None]
    stats = {
        "objective": float(loss.detach()),
        "mean_clip_kd": float(torch.stack(clip_kd).mean().detach()),
        "mean_monolingual_clip_ce": float(torch.stack(mono_ce).mean().detach()),
        "mean_clip_objective_term": float(
            torch.stack(objective_terms).mean().detach()
        ),
        "valid_frames": float(main_stats["valid_frames"]),
        "weight_sum": float(main_stats["weight_sum"]),
    }
    return loss, stats


def teacher_target_audit(
    train_dataset: DistillationDataset,
    target_cache: TeacherTargetCache,
) -> tuple[dict[str, Any], dict[str, bool | None]]:
    rows: list[dict[str, Any]] = []
    clip_soft_means: list[np.ndarray] = []
    frame_numerator = np.zeros(len(LANGUAGE_CODES), dtype=np.float64)
    frame_denominator = 0.0
    gate_by_id: dict[str, bool | None] = {}

    for example in train_dataset.examples:
        item = example["item"]
        length = len(example["features"])
        valid_frames = length - LABEL_DELAY_FRAMES - MODEL_LOOKAHEAD_FRAMES
        if valid_frames <= 0:
            raise ValueError(f"clip {item['id']} has no valid target frames")
        ramp = np.minimum(
            (np.arange(valid_frames, dtype=np.float64) + 1.0)
            / EARLY_RAMP_FRAMES,
            1.0,
        )
        weight_sum = float(ramp.sum())
        soft = example["targets"][:valid_frames].numpy().astype(np.float64)
        raw = target_cache.load(
            item, "teacher_probs", expected_frames=length
        )[:valid_frames].astype(np.float64)
        clip_soft_mean = (soft * ramp[:, None]).sum(axis=0) / weight_sum
        clip_raw_mean = (raw * ramp[:, None]).sum(axis=0) / weight_sum
        clip_soft_means.append(clip_soft_mean)
        frame_numerator += (soft * ramp[:, None]).sum(axis=0)
        frame_denominator += weight_sum
        row: dict[str, Any] = {
            "clip_id": item["id"],
            "manifest_language": item.get("language"),
            "provider": item.get("source"),
            "voice": item.get("speaker_id"),
            "valid_frames": valid_frames,
            "ramp_weight_sum": weight_sum,
            "mean_teacher_t1": clip_raw_mean.tolist(),
            "mean_teacher_t2": clip_soft_mean.tolist(),
        }
        if is_monolingual(item):
            label_index = LANGUAGE_CODES.index(str(item["language"]))
            teacher_index = int(clip_raw_mean.argmax())
            teacher_correct = teacher_index == label_index
            retained_mass = target_cache.load(
                item, "in_set_mass", expected_frames=length
            )
            gate_by_id[str(item["id"])] = teacher_correct
            row.update(
                {
                    "teacher_selected_seven_top1": LANGUAGE_CODES[teacher_index],
                    "teacher_top1_correct": teacher_correct,
                    "p_known_label_t1": float(clip_raw_mean[label_index]),
                    "p_known_label_t2": float(clip_soft_mean[label_index]),
                    "retained_selected_language_mass_t1": float(
                        np.mean(retained_mass)
                    ),
                    "teacher_error_gate_uses": (
                        "kd" if teacher_correct else "hard_ce"
                    ),
                }
            )
        else:
            gate_by_id[str(item["id"])] = None
            row.update(
                {
                    "teacher_selected_seven_top1": None,
                    "teacher_top1_correct": None,
                    "p_known_label_t1": None,
                    "p_known_label_t2": None,
                    "retained_selected_language_mass_t1": None,
                    "teacher_error_gate_uses": "local_kd",
                }
            )
        rows.append(row)

    if frame_denominator <= 0:
        raise AssertionError("target audit has zero total ramp weight")
    for row, clip_soft_mean in zip(rows, clip_soft_means, strict=True):
        row["frame_kd_target_mass_contribution"] = (
            np.asarray(clip_soft_mean)
            * float(row["ramp_weight_sum"])
            / frame_denominator
        ).tolist()

    priors: dict[str, np.ndarray] = {
        "frame_kd": frame_numerator / frame_denominator,
        "equal_clip_kd": np.mean(np.stack(clip_soft_means), axis=0),
    }
    hybrid_targets: list[np.ndarray] = []
    gated_targets: list[np.ndarray] = []
    for row, soft_mean in zip(rows, clip_soft_means, strict=True):
        if row["teacher_top1_correct"] is None:
            hybrid_targets.append(soft_mean)
            gated_targets.append(soft_mean)
            continue
        label_index = LANGUAGE_CODES.index(str(row["manifest_language"]))
        one_hot = np.zeros(len(LANGUAGE_CODES), dtype=np.float64)
        one_hot[label_index] = 1.0
        hybrid_targets.append(0.5 * soft_mean + 0.5 * one_hot)
        gated_targets.append(
            soft_mean if row["teacher_top1_correct"] else one_hot
        )
    priors["hybrid_kd_ce"] = np.mean(np.stack(hybrid_targets), axis=0)
    priors["teacher_error_gate"] = np.mean(np.stack(gated_targets), axis=0)

    monolingual_rows = [row for row in rows if row["teacher_top1_correct"] is not None]
    per_language: dict[str, Any] = {}
    for language in LANGUAGE_CODES:
        selected = [
            row for row in monolingual_rows if row["manifest_language"] == language
        ]
        if len(selected) != 10:
            raise AssertionError(f"expected ten monolingual training clips for {language}")
        language_index = LANGUAGE_CODES.index(language)
        per_language[language] = {
            "clips": len(selected),
            "teacher_top1_correct": int(
                sum(bool(row["teacher_top1_correct"]) for row in selected)
            ),
            "teacher_top1_wrong": int(
                sum(not bool(row["teacher_top1_correct"]) for row in selected)
            ),
            "mean_p_known_label_t1": float(
                np.mean([row["p_known_label_t1"] for row in selected])
            ),
            "mean_p_known_label_t2": float(
                np.mean([row["p_known_label_t2"] for row in selected])
            ),
            "mean_retained_selected_language_mass_t1": float(
                np.mean(
                    [
                        row["retained_selected_language_mass_t1"]
                        for row in selected
                    ]
                )
            ),
            "ramp_weight_sum": float(
                sum(row["ramp_weight_sum"] for row in selected)
            ),
            "frame_loss_share": float(
                sum(row["ramp_weight_sum"] for row in selected)
                / frame_denominator
            ),
            "aggregate_t2_target_mass_for_class": float(
                priors["frame_kd"][language_index]
            ),
            "smoke_floor_at_least_8_of_10": int(
                sum(bool(row["teacher_top1_correct"]) for row in selected)
            )
            >= 8,
        }

    if per_language["en"]["teacher_top1_correct"] != 6:
        raise AssertionError("expected the bound ECAPA cache to be 6/10 on English")
    for language in LANGUAGE_CODES[1:]:
        if per_language[language]["teacher_top1_correct"] != 10:
            raise AssertionError(f"expected the bound teacher to be 10/10 on {language}")
    prior_parity: dict[str, Any] = {}
    english_index = LANGUAGE_CODES.index("en")
    for arm, expected in EXPECTED_ENGLISH_TARGET_PRIORS.items():
        actual = float(priors[arm][english_index])
        difference = actual - expected
        prior_parity[arm] = {
            "expected_rounded": expected,
            "actual": actual,
            "difference": difference,
            "within_0_005_percentage_points": abs(difference) <= 0.00005,
        }
        if not prior_parity[arm]["within_0_005_percentage_points"]:
            raise AssertionError(
                f"{arm} English prior {actual:.8f} differs from audited {expected:.5f}"
            )

    return (
        {
            "language_codes": list(LANGUAGE_CODES),
            "n_training_clips": len(rows),
            "n_monolingual_clips": len(monolingual_rows),
            "teacher_correct_smoke_floor_passed": all(
                value["smoke_floor_at_least_8_of_10"]
                for value in per_language.values()
            ),
            "per_language": per_language,
            "effective_target_prior_by_arm": {
                arm: {
                    language: float(prior[index])
                    for index, language in enumerate(LANGUAGE_CODES)
                }
                for arm, prior in priors.items()
            },
            "english_prior_parity": prior_parity,
            "teacher_wrong_monolingual_clip_ids": [
                row["clip_id"]
                for row in monolingual_rows
                if not row["teacher_top1_correct"]
            ],
            "clips": rows,
        },
        gate_by_id,
    )


def output_bias_gradient(
    initial_state: Mapping[str, torch.Tensor],
    arm: str,
    batch: Mapping[str, Any],
    teacher_correct_by_id: Mapping[str, bool | None],
) -> dict[str, Any]:
    model = CausalLIDStudent(**configured_model_kwargs())
    model.load_state_dict(initial_state)
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(batch["features"])
    loss, stats = factorial_loss(arm, logits, batch, teacher_correct_by_id)
    loss.backward()
    bias_gradient = model.classifier[-1].bias.grad
    if bias_gradient is None or not bool(torch.isfinite(bias_gradient).all()):
        raise FloatingPointError("output-bias gradient is missing or non-finite")
    values = bias_gradient.detach().cpu().double().numpy()
    return {
        "clip_ids": list(batch["ids"]),
        "n_clips": len(batch["ids"]),
        "loss": float(loss.detach()),
        "loss_components": stats,
        "gradient_by_output_language": {
            language: float(values[index])
            for index, language in enumerate(LANGUAGE_CODES)
        },
        "gradient_sum": float(values.sum()),
        "gradient_l2_norm": float(np.linalg.norm(values)),
    }


def gradient_audit(
    initial_state: Mapping[str, torch.Tensor],
    first_batch: Mapping[str, Any],
    full_train_batch: Mapping[str, Any],
    teacher_correct_by_id: Mapping[str, bool | None],
) -> dict[str, Any]:
    per_language_batches = {
        language: slice_batch(
            full_train_batch,
            [
                index
                for index, item in enumerate(full_train_batch["items"])
                if item.get("language") == language and is_monolingual(item)
            ],
        )
        for language in LANGUAGE_CODES
    }
    audit: dict[str, Any] = {}
    for arm in ARMS:
        audit[arm] = {
            "first_training_batch": output_bias_gradient(
                initial_state, arm, first_batch, teacher_correct_by_id
            ),
            "one_full_corpus_objective": output_bias_gradient(
                initial_state, arm, full_train_batch, teacher_correct_by_id
            ),
            "per_manifest_language": {
                language: output_bias_gradient(
                    initial_state,
                    arm,
                    per_language_batches[language],
                    teacher_correct_by_id,
                )
                for language in LANGUAGE_CODES
            },
        }
    return audit


def fixed_prefix_features(
    items: Sequence[dict[str, Any]],
    manifest_snapshot,
    seconds: float,
    frontend: LogMelFrontend,
) -> dict[str, Any]:
    wanted = int(round(seconds * SAMPLE_RATE))
    examples: list[dict[str, Any]] = []
    with torch.inference_mode():
        for item in items:
            waveform = load_audio(resolve_audio_path(item, manifest_snapshot))
            if len(waveform) < wanted:
                raise ValueError(
                    f"{item['id']} has {len(waveform)} samples, below the true "
                    f"{seconds:g}s prefix deadline"
                )
            features = frontend(waveform[:wanted].contiguous()).squeeze(0)
            examples.append({"features": features, "item": item})
    return {
        "features": pad_sequence(
            [example["features"] for example in examples], batch_first=True
        ),
        "lengths": torch.tensor(
            [len(example["features"]) for example in examples], dtype=torch.long
        ),
        "items": [example["item"] for example in examples],
    }


def component_summary(
    arm: str,
    logits: torch.Tensor,
    batch: Mapping[str, Any],
    teacher_correct_by_id: Mapping[str, bool | None],
) -> dict[str, Any]:
    _, aggregate = factorial_loss(arm, logits, batch, teacher_correct_by_id)
    per_language: dict[str, Any] = {}
    for language in LANGUAGE_CODES:
        indices = [
            index
            for index, item in enumerate(batch["items"])
            if item.get("language") == language and is_monolingual(item)
        ]
        if not indices:
            raise AssertionError(f"component summary has no {language} clips")
        language_batch = slice_batch(batch, indices)
        _, stats = factorial_loss(
            arm,
            logits.index_select(0, torch.tensor(indices, dtype=torch.long)),
            language_batch,
            teacher_correct_by_id,
        )
        per_language[language] = stats
    return {"aggregate": aggregate, "per_language": per_language}


def evaluate_state(
    model: CausalLIDStudent,
    state: Mapping[str, torch.Tensor],
    *,
    arm: str,
    step: int,
    state_sha256: str,
    loss_history: Sequence[float],
    heldout_batch: Mapping[str, Any],
    prefix_batches: Mapping[str, Mapping[str, Any]],
    train_batch: Mapping[str, Any],
    exact_indices: Sequence[int],
    switch_batch: Mapping[str, Any],
    heldout_teacher_probabilities: Mapping[str, torch.Tensor],
    train_teacher_probabilities: Mapping[str, torch.Tensor],
    teacher_correct_by_id: Mapping[str, bool | None],
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
        prefix_summaries[key] = trajectory.classification_summary(
            trajectory.posterior_rows(logits, batch["lengths"], batch["items"])
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

    train_rows = trajectory.posterior_rows(
        train_logits, train_batch["lengths"], train_batch["items"]
    )
    exact_batch = slice_batch(train_batch, exact_indices)
    exact_index_tensor = torch.tensor(list(exact_indices), dtype=torch.long)
    exact_logits = train_logits.index_select(0, exact_index_tensor)
    exact_rows = [train_rows[index] for index in exact_indices]
    switch_rows = [
        trajectory.switch_clip_result(
            switch_logits[index],
            int(switch_batch["lengths"][index]),
            switch_batch["targets"][index],
            item,
        )
        for index, item in enumerate(switch_batch["items"])
    ]
    training_window = (
        None
        if step == 0
        else float(
            np.mean(loss_history[max(0, step - SNAPSHOT_INTERVAL) : step])
        )
    )
    return (
        {
            "step": step,
            "model_state_sha256": state_sha256,
            "training": {
                "loss_at_step": (
                    None if step == 0 else float(loss_history[step - 1])
                ),
                "previous_10_mean_loss": training_window,
                "cumulative_mean_loss": (
                    None if step == 0 else float(np.mean(loss_history[:step]))
                ),
            },
            "synthetic_dev": {
                "label_composite_1s_2s_full": label_composite,
                "minimum_language_recall_composite_1s_2s_full": float(
                    min(per_language_composite.values())
                ),
                "per_language_recall_composite_1s_2s_full": per_language_composite,
                "prefix": prefix_summaries,
                "full_frame": trajectory.full_frame_metrics(
                    heldout_logits,
                    heldout_batch,
                    heldout_teacher_probabilities,
                ),
                "monolingual_stability": trajectory.monolingual_stability(
                    heldout_logits,
                    heldout_batch["lengths"],
                    heldout_batch["items"],
                ),
            },
            "familiar_audio": {
                "all_70_monolingual_train_full": {
                    "classification": trajectory.classification_summary(train_rows),
                    "full_frame": trajectory.full_frame_metrics(
                        train_logits,
                        train_batch,
                        train_teacher_probabilities,
                    ),
                    "loss_components": component_summary(
                        arm,
                        train_logits,
                        train_batch,
                        teacher_correct_by_id,
                    ),
                },
                "edge_indices_5_6_7_full": {
                    "classification": trajectory.classification_summary(exact_rows),
                    "full_frame": trajectory.full_frame_metrics(
                        exact_logits,
                        exact_batch,
                        train_teacher_probabilities,
                    ),
                    "loss_components": component_summary(
                        arm,
                        exact_logits,
                        exact_batch,
                        teacher_correct_by_id,
                    ),
                },
            },
            "switch": trajectory.aggregate_switch(switch_rows),
        },
        check_streaming_equivalence,
    )


def local_gate_row(row: Mapping[str, Any]) -> dict[str, bool]:
    train_classification = row["familiar_audio"][
        "all_70_monolingual_train_full"
    ]["classification"]
    exact_classification = row["familiar_audio"][
        "edge_indices_5_6_7_full"
    ]["classification"]
    return {
        "all_seven_training_recalls_nonzero": all(
            value > 0
            for value in train_classification["per_language_recall"].values()
        ),
        "all_training_accuracy_at_least_80pct": (
            train_classification["accuracy"] >= LOCAL_ACCURACY_FLOOR
        ),
        "exact_edge_accuracy_at_least_80pct": (
            exact_classification["accuracy"] >= LOCAL_ACCURACY_FLOOR
        ),
    }


def select_arm(curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in curve if all(local_gate_row(row).values())]

    def diagnostic_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        train = row["familiar_audio"]["all_70_monolingual_train_full"][
            "classification"
        ]
        exact = row["familiar_audio"]["edge_indices_5_6_7_full"][
            "classification"
        ]
        nonzero = sum(value > 0 for value in train["per_language_recall"].values())
        return (
            nonzero,
            train["minimum_language_recall"],
            exact["accuracy"],
            train["accuracy"],
            row["synthetic_dev"]["label_composite_1s_2s_full"],
            -row["synthetic_dev"]["monolingual_stability"][
                "committed_false_switches"
            ],
            -row["step"],
        )

    best_diagnostic = max(curve, key=diagnostic_key)
    selected = (
        max(
            eligible,
            key=lambda row: (
                row["synthetic_dev"]["label_composite_1s_2s_full"],
                -row["synthetic_dev"]["monolingual_stability"][
                    "committed_false_switches"
                ],
                row["switch"]["raw_detected"],
                -row["step"],
            ),
        )
        if eligible
        else None
    )
    return {
        "eligible_steps": [int(row["step"]) for row in eligible],
        "n_eligible_steps": len(eligible),
        "selected_step": None if selected is None else int(selected["step"]),
        "selected_local_gates": (
            None if selected is None else local_gate_row(selected)
        ),
        "best_diagnostic_step": int(best_diagnostic["step"]),
        "best_diagnostic_gates": local_gate_row(best_diagnostic),
    }


def row_at_step(curve: Sequence[Mapping[str, Any]], step: int) -> Mapping[str, Any]:
    return next(row for row in curve if row["step"] == step)


def compact_comparison_row(row: Mapping[str, Any]) -> dict[str, Any]:
    train = row["familiar_audio"]["all_70_monolingual_train_full"][
        "classification"
    ]
    exact = row["familiar_audio"]["edge_indices_5_6_7_full"][
        "classification"
    ]
    return {
        "step": row["step"],
        "all_train_accuracy": train["accuracy"],
        "all_train_minimum_language_recall": train["minimum_language_recall"],
        "all_train_per_language_recall": train["per_language_recall"],
        "exact_edge_accuracy": exact["accuracy"],
        "exact_edge_minimum_language_recall": exact["minimum_language_recall"],
        "exact_edge_per_language_recall": exact["per_language_recall"],
        "synthetic_dev_label_composite_1s_2s_full": row["synthetic_dev"][
            "label_composite_1s_2s_full"
        ],
        "synthetic_dev_minimum_language_recall": row["synthetic_dev"][
            "minimum_language_recall_composite_1s_2s_full"
        ],
        "heldout_committed_false_switches": row["synthetic_dev"][
            "monolingual_stability"
        ]["committed_false_switches"],
        "switch_raw_source_preconditions": row["switch"][
            "raw_source_preconditions"
        ],
        "switch_raw_detected": row["switch"]["raw_detected"],
        "switch_policy_source_preconditions": row["switch"][
            "policy_source_preconditions"
        ],
        "switch_policy_detected": row["switch"]["policy_detected"],
        "local_gates": local_gate_row(row),
    }


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
    launched_at = datetime.now(timezone.utc).isoformat()
    wall_started = time.perf_counter()
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

    manifest_snapshot = capture_manifest_snapshot(args.manifest)
    records = manifest_snapshot.records_copy()
    speaker_audit = require_speaker_disjoint(records)
    target_cache = TeacherTargetCache(manifest_snapshot, args.targets_dir)
    training_settings = training_configuration(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        threads=args.threads,
        seed=args.seed,
    )
    dependency_snapshot = capture_run_dependency_snapshot(
        manifest_snapshot=manifest_snapshot,
        target_cache_identity=target_cache.identity,
        target_metadata_path=target_cache.targets_dir / "metadata.json",
        training=training_settings,
    )
    target_cache.validate_all()
    assert_run_dependency_snapshot_unchanged(
        dependency_snapshot,
        manifest_snapshot=manifest_snapshot,
        target_cache_identity=target_cache.identity,
        target_metadata_path=target_cache.targets_dir / "metadata.json",
        training=training_settings,
        stage="loss-factorial dataset preload",
    )

    train_dataset = DistillationDataset(
        manifest_snapshot,
        args.targets_dir,
        splits=("train",),
        target_cache=target_cache,
    )
    validate_full_batch_training_set(len(train_dataset), args.batch_size)
    items_by_id = {
        example["item"]["id"]: example["item"] for example in train_dataset.examples
    }
    target_audit, teacher_correct_by_id = teacher_target_audit(
        train_dataset, target_cache
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

    # Match the main pipeline's model-initialization point: after construction
    # of the training dataset/loader and before the loader is iterated.
    base_model = CausalLIDStudent(**configured_model_kwargs())
    initial_state = clone_state_dict(base_model)
    initial_model_hash = model_state_sha256(initial_state)
    models = OrderedDict(
        [(arm, base_model if position == 0 else copy.deepcopy(base_model))
         for position, arm in enumerate(ARMS)]
    )
    if any(
        model_state_sha256(model.state_dict()) != initial_model_hash
        for model in models.values()
    ):
        raise AssertionError("factorial arms did not start from identical weights")

    full_train_batch = collate_examples(train_dataset.examples)
    diagnostic_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_distillation_batch,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
        drop_last=True,
    )
    first_batch = batch_with_items(next(iter(diagnostic_loader)), items_by_id)
    gradients = gradient_audit(
        initial_state,
        first_batch,
        full_train_batch,
        teacher_correct_by_id,
    )

    optimizers = {
        arm: torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=WEIGHT_DECAY,
        )
        for arm, model in models.items()
    }
    losses = {arm: [] for arm in ARMS}
    kd_diagnostics = {arm: [] for arm in ARMS}
    ce_diagnostics = {arm: [] for arm in ARMS}
    gradient_norms = {arm: [] for arm in ARMS}
    examples_seen = {arm: 0 for arm in ARMS}
    snapshots: dict[str, dict[int, dict[str, torch.Tensor]]] = {
        arm: {0: clone_state_dict(model)} for arm, model in models.items()
    }
    snapshot_hashes: dict[str, dict[int, str]] = {
        arm: {0: model_state_sha256(snapshots[arm][0])} for arm in ARMS
    }
    batch_order_digest = hashlib.sha256()
    successful_steps = 0
    for model in models.values():
        model.train()
    print(
        f"training {len(ARMS)} identity-matched {base_model.parameter_count:,}-parameter "
        f"students for {args.steps} updates",
        flush=True,
    )
    while successful_steps < args.steps:
        for raw_batch in loader:
            step = successful_steps + 1
            batch = batch_with_items(raw_batch, items_by_id)
            batch_order_digest.update(
                json.dumps(batch["ids"], separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
            step_losses: dict[str, float] = {}
            for arm, model in models.items():
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch["features"])
                loss, stats = factorial_loss(
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
                    raise FloatingPointError(
                        f"non-finite {arm} gradient at step {step}"
                    )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRADIENT_CLIP_NORM,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                assert_finite_post_update_state(model, optimizer, step=step)
                value = float(loss.detach())
                losses[arm].append(value)
                kd_diagnostics[arm].append(stats["mean_clip_kd"])
                ce_diagnostics[arm].append(stats["mean_monolingual_clip_ce"])
                gradient_norms[arm].append(float(gradient_norm))
                examples_seen[arm] += len(batch["ids"])
                step_losses[arm] = value
            successful_steps += 1
            if successful_steps % SNAPSHOT_INTERVAL == 0:
                for arm, model in models.items():
                    state = clone_state_dict(model)
                    snapshots[arm][successful_steps] = state
                    snapshot_hashes[arm][successful_steps] = model_state_sha256(state)
            if successful_steps == 1 or successful_steps % 100 == 0:
                print(
                    f"step={successful_steps:04d} "
                    + " ".join(
                        f"{arm}={step_losses[arm]:.5f}" for arm in ARMS
                    ),
                    flush=True,
                )
            if successful_steps >= args.steps:
                break

    expected_steps = list(range(0, args.steps + 1, SNAPSHOT_INTERVAL))
    for arm in ARMS:
        if sorted(snapshots[arm]) != expected_steps:
            raise AssertionError(f"{arm} snapshot cadence is incomplete")
    training_contracts = {
        arm: build_training_contract(
            requested_steps=args.steps,
            successful_steps=successful_steps,
            post_update_checks=successful_steps,
            examples_seen=examples_seen[arm],
            losses=losses[arm],
            gradient_norms=gradient_norms[arm],
            model=models[arm],
            optimizer=optimizers[arm],
        )
        for arm in ARMS
    }
    if not all(contract["nan_free"] for contract in training_contracts.values()):
        raise RuntimeError("one or more factorial training contracts failed")

    frame_control_final_hash = snapshot_hashes["frame_kd"][args.steps]
    frame_control_model_exact = frame_control_final_hash == reference_model_hash
    frame_control_loss_exact = (
        losses["frame_kd"]
        == reference_checkpoint["training_evidence"]["losses"]
    )
    frame_control_gradient_exact = (
        gradient_norms["frame_kd"]
        == reference_checkpoint["training_evidence"]["gradient_norms"]
    )
    if not (
        frame_control_model_exact
        and frame_control_loss_exact
        and frame_control_gradient_exact
    ):
        raise RuntimeError(
            "frame-KD control did not exactly reproduce the current reference run"
        )
    print(
        "training complete; frame-KD control exactly reproduced the main run; "
        "preparing fixed evaluations",
        flush=True,
    )

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
    heldout_batch = collate_examples(heldout_dataset.examples)
    switch_batch = collate_examples(switch_dataset.examples)
    monolingual_train_examples = [
        example for example in train_dataset.examples if is_monolingual(example["item"])
    ]
    if len(monolingual_train_examples) != 70:
        raise AssertionError("expected 70 monolingual training examples")
    train_batch = collate_examples(monolingual_train_examples)
    exact_indices = [
        index
        for index, item in enumerate(train_batch["items"])
        if int(str(item["id"]).rsplit("_", 1)[1]) in EXACT_EDGE_TRAIN_INDICES
    ]
    if len(exact_indices) != 21:
        raise AssertionError("expected 21 exact Edge controls at indices 5/6/7")

    frontend = LogMelFrontend().eval()
    prefix_batches: dict[str, Mapping[str, Any]] = {"full": heldout_batch}
    for seconds in PREFIX_SECONDS:
        prefix_batches[trajectory.prefix_key(seconds)] = fixed_prefix_features(
            heldout_batch["items"], manifest_snapshot, seconds, frontend
        )
    if set(prefix_batches) != set(SELECTION_PREFIXES):
        raise AssertionError("prefix evaluation table differs from 1s/2s/full")

    heldout_teacher_probabilities = {
        item["id"]: torch.from_numpy(
            target_cache.load(
                item,
                "teacher_probs",
                expected_frames=int(heldout_batch["lengths"][index]),
            )
        )
        for index, item in enumerate(heldout_batch["items"])
    }
    train_teacher_probabilities = {
        item["id"]: torch.from_numpy(
            target_cache.load(
                item,
                "teacher_probs",
                expected_frames=int(train_batch["lengths"][index]),
            )
        )
        for index, item in enumerate(train_batch["items"])
    }

    curves: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    streaming_equivalence = {arm: True for arm in ARMS}
    for arm, model in models.items():
        for position, step in enumerate(expected_steps):
            row, streaming_equivalence[arm] = evaluate_state(
                model,
                snapshots[arm][step],
                arm=arm,
                step=step,
                state_sha256=snapshot_hashes[arm][step],
                loss_history=losses[arm],
                heldout_batch=heldout_batch,
                prefix_batches=prefix_batches,
                train_batch=train_batch,
                exact_indices=exact_indices,
                switch_batch=switch_batch,
                heldout_teacher_probabilities=heldout_teacher_probabilities,
                train_teacher_probabilities=train_teacher_probabilities,
                teacher_correct_by_id=teacher_correct_by_id,
                check_streaming_equivalence=streaming_equivalence[arm],
            )
            curves[arm].append(row)
            if position == 0 or (position + 1) % 20 == 0 or step == args.steps:
                train_accuracy = row["familiar_audio"][
                    "all_70_monolingual_train_full"
                ]["classification"]["accuracy"]
                print(
                    f"evaluated {arm} step={step:04d} "
                    f"train={train_accuracy:.3f} "
                    f"dev={row['synthetic_dev']['label_composite_1s_2s_full']:.3f}",
                    flush=True,
                )
        if streaming_equivalence[arm]:
            raise AssertionError(f"{arm} streaming/full equivalence was not checked")

    selections = {arm: select_arm(curves[arm]) for arm in ARMS}
    final_comparison = {
        arm: compact_comparison_row(row_at_step(curves[arm], args.steps))
        for arm in ARMS
    }
    best_diagnostic_comparison = {
        arm: compact_comparison_row(
            row_at_step(curves[arm], selections[arm]["best_diagnostic_step"])
        )
        for arm in ARMS
    }
    eligible_treatments = [
        arm
        for arm in ARMS
        if arm != "frame_kd" and selections[arm]["selected_step"] is not None
    ]
    if eligible_treatments:
        chosen_arm = max(
            eligible_treatments,
            key=lambda arm: (
                row_at_step(curves[arm], selections[arm]["selected_step"])[
                    "synthetic_dev"
                ]["label_composite_1s_2s_full"],
                -selections[arm]["selected_step"],
            ),
        )
        verdict = "adopt"
        verdict_reason = (
            f"{chosen_arm} produced at least one checkpoint passing all local "
            "class-coverage, all-training, and exact-Edge smoke gates; adopt the "
            "loss as a candidate for frozen external/natural validation, not as "
            "a release model"
        )
    else:
        chosen_arm = None
        control_final_en = final_comparison["frame_kd"][
            "all_train_per_language_recall"
        ]["en"]
        correction_improved_english = any(
            final_comparison[arm]["all_train_per_language_recall"]["en"]
            > control_final_en
            for arm in ("hybrid_kd_ce", "teacher_error_gate")
        )
        if correction_improved_english:
            verdict = "inconclusive"
            verdict_reason = (
                "known-label correction changed English recall, but no treatment "
                "checkpoint passed all predeclared local sufficiency gates"
            )
        else:
            verdict = "reject"
            verdict_reason = (
                "no treatment passed the local sufficiency gates and the two "
                "teacher-error corrections did not improve final English recall"
            )

    architecture_contract = {
        "parameter_count_by_arm": {
            arm: model.parameter_count for arm, model in models.items()
        },
        "receptive_field_frames_by_arm": {
            arm: model.receptive_field_frames for arm, model in models.items()
        },
        "lookahead_frames_by_arm": {
            arm: model.lookahead_frames for arm, model in models.items()
        },
        "algorithmic_latency_ms_by_arm": {
            arm: ALGORITHMIC_LATENCY_MS for arm in ARMS
        },
        "all_arms_unchanged": (
            len({model.parameter_count for model in models.values()}) == 1
            and len({model.receptive_field_frames for model in models.values()}) == 1
            and len({model.lookahead_frames for model in models.values()}) == 1
        ),
    }

    # Refuse publication if another process changed any imported source,
    # manifest/audio, target, or reference artifact during the expensive run.
    target_cache.validate_all()
    assert_run_dependency_snapshot_unchanged(
        dependency_snapshot,
        manifest_snapshot=manifest_snapshot,
        target_cache_identity=target_cache.identity,
        target_metadata_path=target_cache.targets_dir / "metadata.json",
        training=training_settings,
        stage="loss-factorial results publication",
    )
    if snapshot_sources() != source_snapshot_at_start:
        raise RuntimeError("experiment or imported source changed during the run")
    reference_hashes_at_end = {
        "checkpoint": file_sha256(args.reference_checkpoint),
        "summary": file_sha256(args.reference_summary),
    }
    if reference_hashes_at_end != reference_hashes_at_start:
        raise RuntimeError("reference checkpoint or summary changed during the run")

    experiment_configuration = {
        "analysis_version": ANALYSIS_VERSION,
        "arms": dict(ARMS),
        "steps": args.steps,
        "snapshot_interval_steps": SNAPSHOT_INTERVAL,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "seed": args.seed,
        "threads": args.threads,
        "prefix_seconds": [1, 2, "full"],
        "local_accuracy_floor": LOCAL_ACCURACY_FLOOR,
        "target_timing_changed": False,
        "architecture_changed": False,
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
        "initial_model_state_sha256": initial_model_hash,
        "batch_order_sha256": batch_order_digest.hexdigest(),
        "target_gate_lineage_sha256": canonical_sha256(teacher_correct_by_id),
        "model_state_sha256_by_arm_step": {
            arm: {
                str(step): snapshot_hashes[arm][step] for step in expected_steps
            }
            for arm in ARMS
        },
    }
    run_id = canonical_sha256(experiment_identity)
    results = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "question": (
            "Can equal-clip weighting or known-label correction repair the "
            "teacher-biased English/class collapse of the same causal TCN?"
        ),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "recommended_arm": chosen_arm,
        "run_id": run_id,
        "launched_at_utc": launched_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - wall_started,
        "experiment_identity": experiment_identity,
        "configuration": experiment_configuration,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": trajectory.package_versions(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "data": {
            "role": "synthetic_development",
            "n_train_clips": len(train_dataset),
            "n_train_monolingual_clips": len(monolingual_train_examples),
            "n_heldout_monolingual_clips": len(heldout_dataset),
            "n_switch_clips": len(switch_dataset),
            "heldout_clip_ids": [item["id"] for item in heldout_batch["items"]],
            "switch_clip_ids": [item["id"] for item in switch_batch["items"]],
            "speaker_audit": speaker_audit,
            "manifest_snapshot": manifest_snapshot.identity(),
            "target_cache": target_cache.audit(),
        },
        "teacher_target_audit": target_audit,
        "initial_gradient_audit": gradients,
        "training": {
            "batch_order_sha256": batch_order_digest.hexdigest(),
            "initial_model_state_sha256": initial_model_hash,
            "successful_steps": successful_steps,
            "snapshot_steps": expected_steps,
            "arms": {
                arm: {
                    "contract": training_contracts[arm],
                    "examples_seen": examples_seen[arm],
                    "effective_epochs": examples_seen[arm] / len(train_dataset),
                    "losses": losses[arm],
                    "mean_clip_kd_diagnostics": kd_diagnostics[arm],
                    "mean_monolingual_clip_ce_diagnostics": ce_diagnostics[arm],
                    "gradient_norms": gradient_norms[arm],
                    "first_10_mean_loss": float(np.mean(losses[arm][:10])),
                    "last_10_mean_loss": float(np.mean(losses[arm][-10:])),
                    "model_state_sha256_by_step": {
                        str(step): snapshot_hashes[arm][step]
                        for step in expected_steps
                    },
                }
                for arm in ARMS
            },
        },
        "curves": curves,
        "selection": selections,
        "fixed_step_1600_comparison": final_comparison,
        "best_diagnostic_comparison": best_diagnostic_comparison,
        "architecture_and_latency": architecture_contract,
        "identity_audit": {
            "source_snapshot_unchanged": True,
            "dependency_snapshot_unchanged": True,
            "reference_bundle_unchanged": True,
            "target_cache_revalidated_at_end": True,
            "all_initial_states_identical": True,
            "streaming_full_equivalence_checked_by_arm": {
                arm: not pending for arm, pending in streaming_equivalence.items()
            },
            "frame_control_final_model_exactly_reproduced": frame_control_model_exact,
            "frame_control_loss_trace_exactly_reproduced": frame_control_loss_exact,
            "frame_control_gradient_trace_exactly_reproduced": frame_control_gradient_exact,
            "reference_model_state_sha256": reference_model_hash,
        },
        "decision": {
            "verdict": verdict,
            "reason": verdict_reason,
            "recommended_arm": chosen_arm,
            "eligible_treatment_arms": eligible_treatments,
            "scope": (
                "candidate loss for frozen external and natural validation only; "
                "no experiment checkpoint is a release checkpoint"
            ),
        },
        "limitations": [
            "one deterministic seed and one 71-clip synthetic training corpus",
            "the 21 voice-disjoint held-out clips are repeatedly consulted synthetic development data, not untouched test",
            "the two switch clips reverse one synthetic source pair and are not independent natural switch trials",
            "known-label CE is available only for labelled monolingual clips and changes the pure-KD claim",
            "the availability-valid teacher switch trajectory remains slow and is unchanged",
            "no external validation, natural speech, or locked test data were read",
        ],
    }
    trajectory.assert_finite_json(results)
    atomic_json_write(results, args.output)
    print(
        f"wrote {args.output}; verdict={verdict}; recommended={chosen_arm}; "
        f"run_id={run_id}",
        flush=True,
    )


if __name__ == "__main__":
    main()
