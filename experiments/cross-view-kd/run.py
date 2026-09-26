#!/usr/bin/env python3
"""Paired clean-control versus telephony cross-view KD experiment.

The two students share initialization, clean ECAPA targets, minibatch order,
architecture, optimizer settings, and update count.  The treatment alone sees
an independently sampled clean/narrowband-PCM/PCMA/PCMU view of each training
example.  Evaluation uses the main 21 held-out clips and both switch clips.

Run from the repository root with::

    UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
      experiments/cross-view-kd/run.py --fresh
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# Reuse the already validated telephony transformations and scoring routines.
# Loading by path is necessary because the experiment directory contains a
# hyphen and therefore is not a normal Python package name.
TELEPHONY_DRIVER = REPO_ROOT / "experiments/telephony-robustness/run.py"
_telephony_spec = importlib.util.spec_from_file_location(
    "telephony_robustness_helpers", TELEPHONY_DRIVER
)
if _telephony_spec is None or _telephony_spec.loader is None:
    raise ImportError(f"cannot load telephony helpers from {TELEPHONY_DRIVER}")
telephony = importlib.util.module_from_spec(_telephony_spec)
sys.modules[_telephony_spec.name] = telephony
_telephony_spec.loader.exec_module(telephony)


# Import the training, data, loss, frontend, model, and identity contracts from
# the main pipeline rather than copying them into the experiment.
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


EXPERIMENT_NAME = "cross-view-kd"
SCHEMA_VERSION = 1
ANALYSIS_VERSION = "cross-view-kd-v1"
TRAINING_VIEWS = ("clean", "narrowband_pcm", "pcma", "pcmu")
EVALUATION_CONDITIONS = TRAINING_VIEWS
PREFIX_SECONDS = (1, 2, 4)
DEFAULT_STEPS = 1_600
DEFAULT_BATCH_SIZE = 7
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_SEED = 7
DEFAULT_VIEW_SEED = 28_007
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 5.0
RECOVERY_FRACTION_GATE = 0.50
CLEAN_MACRO_F1_LOSS_GATE_PP = 2.0

SOURCE_FILES = (
    "scripts/train.py",
    "scripts/eval.py",
    "scripts/teacher_targets.py",
    "src/streaming_lid/audio.py",
    "src/streaming_lid/config.py",
    "src/streaming_lid/data.py",
    "src/streaming_lid/loss.py",
    "src/streaming_lid/model.py",
    "src/streaming_lid/run_identity.py",
    "experiments/telephony-robustness/run.py",
    "experiments/cross-view-kd/run.py",
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
        "--reference-summary", type=Path, default=Path("results/summary.json")
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=Path("checkpoints/student.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/cross-view-kd/results.json"),
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--view-seed", type=int, default=DEFAULT_VIEW_SEED)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Replace output only after every end-of-run identity check passes.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("steps", "batch_size", "threads", "benchmark_repeats"):
        value = getattr(args, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be an integer >= 1")
    if args.threads != 6:
        raise ValueError("team protocol requires exactly six Torch threads")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if args.output.exists() and not args.fresh:
        raise FileExistsError(f"{args.output} exists; pass --fresh to replace it")
    missing_views = set(TRAINING_VIEWS) - set(telephony.CONDITION_BY_NAME)
    if missing_views:
        raise ValueError(f"telephony helper is missing views: {sorted(missing_views)}")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def snapshot_sources() -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        contents = path.read_bytes()
        snapshot[relative] = {
            "sha256": hashlib.sha256(contents).hexdigest(),
            "bytes": len(contents),
        }
    return snapshot


def input_fingerprint(records: list[dict], manifest: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(records, key=lambda value: value["id"]):
        digest.update(
            json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        )
        with resolve_audio_path(item, manifest).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def probability_payload(probabilities: np.ndarray) -> dict[str, Any]:
    prediction_index = int(probabilities.argmax())
    return {
        "probabilities": probabilities.tolist(),
        "prediction": LANGUAGE_CODES[prediction_index],
        "max_probability": float(probabilities[prediction_index]),
    }


def train_step(
    model: CausalLIDStudent,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    *,
    step: int,
) -> tuple[float, float, dict[str, float]]:
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch["features"])
    loss, stats = delayed_distillation_loss(
        logits,
        batch["targets"],
        batch["lengths"],
        delay_frames=LABEL_DELAY_FRAMES,
        lookahead_frames=MODEL_LOOKAHEAD_FRAMES,
        temperature=TEACHER_TEMPERATURE,
        early_ramp_frames=EARLY_RAMP_FRAMES,
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
    loss.backward()
    if not all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise FloatingPointError(f"non-finite gradient at step {step}")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=GRADIENT_CLIP_NORM, error_if_nonfinite=True
    )
    optimizer.step()
    assert_finite_post_update_state(model, optimizer, step=step)
    return float(loss.detach()), float(gradient_norm), stats


def training_summary(
    model: CausalLIDStudent,
    optimizer: torch.optim.Optimizer,
    losses: list[float],
    gradient_norms: list[float],
    *,
    steps: int,
    examples_seen: int,
    num_examples: int,
) -> dict[str, Any]:
    contract = build_training_contract(
        requested_steps=steps,
        successful_steps=len(losses),
        post_update_checks=len(losses),
        examples_seen=examples_seen,
        losses=losses,
        gradient_norms=gradient_norms,
        model=model,
        optimizer=optimizer,
    )
    if not contract["nan_free"] or not contract["real_audio_optimizer_step"]:
        raise RuntimeError(f"training contract failed: {contract}")
    window = min(10, len(losses))
    return {
        "contract": contract,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "first_10_mean_loss": float(np.mean(losses[:window])),
        "last_10_mean_loss": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:]))
        < float(np.mean(losses[:window])),
        "examples_seen": examples_seen,
        "effective_epochs": examples_seen / num_examples,
        "final_model_state_sha256": model_state_sha256(model.state_dict()),
    }


def summarize_arm(
    rows: list[dict[str, Any]], arm: str
) -> dict[str, dict[str, Any]]:
    clean_by_key = {
        (row["clip_id"], row["prefix_seconds"]): row
        for row in rows
        if row["condition"] == "clean"
    }
    clean_rows = [row for row in rows if row["condition"] == "clean"]
    clean_aggregate = telephony.classification_summary(clean_rows, arm)
    result: dict[str, dict[str, Any]] = {}
    for condition in EVALUATION_CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        aggregate = telephony.classification_summary(selected, arm)
        divergences: list[float] = []
        flips = 0
        correct_to_wrong = 0
        wrong_to_correct = 0
        for row in selected:
            key = (row["clip_id"], row["prefix_seconds"])
            reference = clean_by_key[key]
            reference_probs = np.asarray(
                reference[arm]["probabilities"], dtype=np.float64
            )
            candidate_probs = np.asarray(row[arm]["probabilities"], dtype=np.float64)
            divergences.append(
                telephony.conditional_kl(reference_probs, candidate_probs)
            )
            reference_prediction = reference[arm]["prediction"]
            candidate_prediction = row[arm]["prediction"]
            flips += int(reference_prediction != candidate_prediction)
            clean_correct = reference_prediction == row["expected"]
            candidate_correct = candidate_prediction == row["expected"]
            correct_to_wrong += int(clean_correct and not candidate_correct)
            wrong_to_correct += int(not clean_correct and candidate_correct)
        result[condition] = {
            "aggregate_1_2_4s": aggregate,
            "by_prefix_seconds": {
                str(seconds): telephony.classification_summary(
                    [
                        row
                        for row in selected
                        if row["prefix_seconds"] == seconds
                    ],
                    arm,
                )
                for seconds in PREFIX_SECONDS
            },
            "macro_f1_drop_pp_vs_clean": 100
            * (clean_aggregate["macro_f1"] - aggregate["macro_f1"]),
            "conditional_kl_clean_to_condition": {
                "median": float(np.median(divergences)),
                "p95": float(np.percentile(divergences, 95)),
                "maximum": float(np.max(divergences)),
            },
            "paired_prediction_flips_vs_clean": flips,
            "paired_correct_to_wrong_vs_clean": correct_to_wrong,
            "paired_wrong_to_correct_vs_clean": wrong_to_correct,
        }
    return result


def cross_arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition in EVALUATION_CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        control = telephony.classification_summary(selected, "clean_control")
        treatment = telephony.classification_summary(selected, "cross_view")
        changed = 0
        wrong_to_right = 0
        right_to_wrong = 0
        divergences = []
        for row in selected:
            control_prediction = row["clean_control"]["prediction"]
            treatment_prediction = row["cross_view"]["prediction"]
            changed += int(control_prediction != treatment_prediction)
            control_correct = control_prediction == row["expected"]
            treatment_correct = treatment_prediction == row["expected"]
            wrong_to_right += int(not control_correct and treatment_correct)
            right_to_wrong += int(control_correct and not treatment_correct)
            divergences.append(
                telephony.conditional_kl(
                    np.asarray(
                        row["clean_control"]["probabilities"], dtype=np.float64
                    ),
                    np.asarray(row["cross_view"]["probabilities"], dtype=np.float64),
                )
            )
        clean_f1 = telephony.classification_summary(
            [row for row in rows if row["condition"] == "clean"],
            "clean_control",
        )["macro_f1"]
        recoverable_loss_pp = 100 * (clean_f1 - control["macro_f1"])
        gain_pp = 100 * (treatment["macro_f1"] - control["macro_f1"])
        recovery_fraction = (
            None
            if recoverable_loss_pp <= 0
            else gain_pp / recoverable_loss_pp
        )
        by_condition[condition] = {
            "clean_control_macro_f1": control["macro_f1"],
            "cross_view_macro_f1": treatment["macro_f1"],
            "cross_view_gain_pp": gain_pp,
            "recoverable_loss_pp_from_control_clean": recoverable_loss_pp,
            "recovery_fraction": recovery_fraction,
            "paired_prediction_changes": changed,
            "paired_wrong_to_right": wrong_to_right,
            "paired_right_to_wrong": right_to_wrong,
            "conditional_kl_control_to_cross_view": {
                "median": float(np.median(divergences)),
                "p95": float(np.percentile(divergences, 95)),
            },
        }

    clean_rows = [row for row in rows if row["condition"] == "clean"]
    degraded_rows = [row for row in rows if row["condition"] != "clean"]
    control_clean = telephony.classification_summary(clean_rows, "clean_control")
    treatment_clean = telephony.classification_summary(clean_rows, "cross_view")
    control_degraded = telephony.classification_summary(
        degraded_rows, "clean_control"
    )
    treatment_degraded = telephony.classification_summary(
        degraded_rows, "cross_view"
    )
    recoverable_loss_pp = 100 * (
        control_clean["macro_f1"] - control_degraded["macro_f1"]
    )
    recovered_pp = 100 * (
        treatment_degraded["macro_f1"] - control_degraded["macro_f1"]
    )
    recovery_fraction = (
        None if recoverable_loss_pp <= 0 else recovered_pp / recoverable_loss_pp
    )
    clean_loss_pp = 100 * (
        control_clean["macro_f1"] - treatment_clean["macro_f1"]
    )
    return {
        "by_condition": by_condition,
        "pooled_degraded": {
            "conditions": list(EVALUATION_CONDITIONS[1:]),
            "n_requests": len(degraded_rows),
            "clean_control_clean_macro_f1": control_clean["macro_f1"],
            "cross_view_clean_macro_f1": treatment_clean["macro_f1"],
            "clean_macro_f1_loss_pp": clean_loss_pp,
            "clean_control_degraded_macro_f1": control_degraded["macro_f1"],
            "cross_view_degraded_macro_f1": treatment_degraded["macro_f1"],
            "recoverable_channel_loss_pp": recoverable_loss_pp,
            "recovered_pp": recovered_pp,
            "recovery_fraction": recovery_fraction,
            "recovery_gate_evaluable": recovery_fraction is not None,
            "recovered_at_least_half": recovery_fraction is not None
            and recovery_fraction >= RECOVERY_FRACTION_GATE,
            "clean_loss_le_2pp": clean_loss_pp <= CLEAN_MACRO_F1_LOSS_GATE_PP,
        },
    }


def summarize_switches(
    switch_rows: dict[str, dict[str, list[dict[str, Any]]]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary: dict[str, Any] = {}
    additional_misses: list[dict[str, str]] = []
    recovered_detections: list[dict[str, str]] = []
    for condition in EVALUATION_CONDITIONS:
        summary[condition] = {}
        control_by_id = {
            row["clip_id"]: row
            for row in switch_rows[condition]["clean_control"]
        }
        for arm in ("clean_control", "cross_view"):
            rows = switch_rows[condition][arm]
            summary[condition][arm] = {
                "policy_detected": sum(
                    row["policy_event"]["detected"] for row in rows
                ),
                "policy_source_committed_before_boundary": sum(
                    row["policy_event"]["source_committed_before_boundary"]
                    for row in rows
                ),
                "raw_500ms_stable_detected": sum(
                    row["raw_chunk_500ms_stable_event"]["detected"]
                    for row in rows
                ),
                "rows": rows,
            }
        for treatment_row in switch_rows[condition]["cross_view"]:
            control_row = control_by_id[treatment_row["clip_id"]]
            control_detected = control_row["policy_event"]["detected"]
            treatment_detected = treatment_row["policy_event"]["detected"]
            marker = {
                "condition": condition,
                "clip_id": treatment_row["clip_id"],
            }
            if control_detected and not treatment_detected:
                additional_misses.append(marker)
            if not control_detected and treatment_detected:
                recovered_detections.append(marker)
    comparison = {
        "additional_policy_misses": additional_misses,
        "n_additional_policy_misses": len(additional_misses),
        "recovered_policy_detections": recovered_detections,
        "n_recovered_policy_detections": len(recovered_detections),
        "no_additional_switch_misses": not additional_misses,
    }
    return summary, comparison


def benchmark_models(
    models: dict[str, CausalLIDStudent],
    waveforms: list[torch.Tensor],
    *,
    repeats: int,
) -> dict[str, Any]:
    frontend = LogMelFrontend().eval()
    total_audio_seconds = sum(len(waveform) for waveform in waveforms) / SAMPLE_RATE
    timings = {arm: [] for arm in models}
    with torch.inference_mode():
        warm_features = frontend(waveforms[0])
        for model in models.values():
            model.streaming_forward(warm_features, chunk_frames=CHUNK_FRAMES)
        arm_names = list(models)
        for repeat in range(repeats):
            order = arm_names if repeat % 2 == 0 else list(reversed(arm_names))
            for arm in order:
                started = time.perf_counter()
                for waveform in waveforms:
                    features = frontend(waveform)
                    models[arm].streaming_forward(
                        features, chunk_frames=CHUNK_FRAMES
                    )
                timings[arm].append(time.perf_counter() - started)
    return {
        "includes": "main log-mel frontend plus uncached-overlap streaming replay",
        "threads": torch.get_num_threads(),
        "repeats": repeats,
        "audio_seconds_per_repeat": total_audio_seconds,
        "by_arm": {
            arm: {
                "wall_seconds": values,
                "rtfs": [value / total_audio_seconds for value in values],
                "median_rtf": float(
                    np.median([value / total_audio_seconds for value in values])
                ),
            }
            for arm, values in timings.items()
        },
    }


def package_versions() -> dict[str, str]:
    names = ("numpy", "soundfile", "torch", "torchaudio")
    return {name: importlib.metadata.version(name) for name in names}


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    seed_everything(args.seed)
    launched_at = datetime.now(timezone.utc).isoformat()
    wall_started = time.perf_counter()

    source_snapshot_at_start = snapshot_sources()
    driver_sha256 = source_snapshot_at_start[
        "experiments/cross-view-kd/run.py"
    ]["sha256"]
    telephony_driver_sha256 = source_snapshot_at_start[
        "experiments/telephony-robustness/run.py"
    ]["sha256"]
    reference_file_hashes_at_start = {
        "summary": file_sha256(args.reference_summary),
        "checkpoint": file_sha256(args.reference_checkpoint),
    }
    reference_summary = json.loads(
        args.reference_summary.read_text(encoding="utf-8")
    )
    reference_checkpoint = torch.load(
        args.reference_checkpoint, map_location="cpu", weights_only=True
    )
    reference_model_state_sha256 = model_state_sha256(
        reference_checkpoint["model_state"]
    )
    if reference_summary["model_state_sha256"] != reference_model_state_sha256:
        raise ValueError("main summary and checkpoint model-state hashes disagree")

    records = read_manifest(args.manifest)
    speaker_audit = require_speaker_disjoint(records)
    train_records = [item for item in records if item["split"] == "train"]
    heldout_records = [item for item in records if item["split"] == "heldout"]
    switch_records = [item for item in records if item["split"] == "switch"]
    if len(train_records) != 71 or len(heldout_records) != 21:
        raise AssertionError("expected the main 71 training and 21 held-out clips")
    if {item["id"] for item in switch_records} != {
        "switch_hi_en_eval",
        "switch_en_hi_eval",
    }:
        raise AssertionError("unexpected main switch evaluation set")

    full_input_fingerprint_at_start = input_fingerprint(records, args.manifest)
    eval_input_fingerprint_at_start = input_fingerprint(
        heldout_records + switch_records, args.manifest
    )
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
    dataset = DistillationDataset(
        args.manifest,
        args.targets_dir,
        splits=("train",),
        target_cache=target_cache,
        records=records,
    )
    validate_full_batch_training_set(len(dataset), args.batch_size)
    clean_examples_by_id = {
        example["item"]["id"]: example for example in dataset.examples
    }

    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_distillation_batch,
        generator=loader_generator,
        num_workers=0,
        drop_last=True,
    )
    model_kwargs = configured_model_kwargs()
    clean_control = CausalLIDStudent(**model_kwargs)
    cross_view = copy.deepcopy(clean_control)
    initial_model_sha256 = model_state_sha256(clean_control.state_dict())
    if model_state_sha256(cross_view.state_dict()) != initial_model_sha256:
        raise AssertionError("paired arms did not start from identical weights")

    transform_validation = {
        "filter": telephony.validate_filter(),
        "g711": telephony.validate_g711_fixed_vectors(),
    }
    frontend = LogMelFrontend().eval()
    training_features: dict[str, dict[str, torch.Tensor]] = {}
    training_transform_diagnostics = {
        condition: [] for condition in TRAINING_VIEWS
    }
    with torch.inference_mode():
        for item in train_records:
            clip_id = item["id"]
            waveform = load_audio(resolve_audio_path(item, args.manifest))
            training_features[clip_id] = {}
            for condition in TRAINING_VIEWS:
                transformed, diagnostics = telephony.transform_waveform(
                    waveform,
                    clip_id,
                    telephony.CONDITION_BY_NAME[condition],
                )
                features = frontend(transformed).squeeze(0)
                expected_frames = len(clean_examples_by_id[clip_id]["features"])
                if len(features) != expected_frames:
                    raise AssertionError(
                        f"training transform changed frame count for {clip_id}"
                    )
                if condition == "clean":
                    torch.testing.assert_close(
                        features,
                        clean_examples_by_id[clip_id]["features"],
                        rtol=0,
                        atol=0,
                    )
                training_features[clip_id][condition] = features
                training_transform_diagnostics[condition].append(diagnostics)
    print(
        f"precomputed {len(training_features)} clips x {len(TRAINING_VIEWS)} views",
        flush=True,
    )

    models = {"clean_control": clean_control, "cross_view": cross_view}
    optimizers = {
        arm: torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=WEIGHT_DECAY,
        )
        for arm, model in models.items()
    }
    losses = {arm: [] for arm in models}
    gradient_norms = {arm: [] for arm in models}
    examples_seen = {arm: 0 for arm in models}
    view_generator = torch.Generator().manual_seed(args.view_seed)
    view_counts: Counter[str] = Counter()
    batch_order_digest = hashlib.sha256()
    view_schedule_digest = hashlib.sha256()

    for model in models.values():
        model.train()
    successful_steps = 0
    while successful_steps < args.steps:
        for clean_batch in loader:
            step = successful_steps + 1
            ids = list(clean_batch["ids"])
            batch_order_digest.update(
                json.dumps(ids, separators=(",", ":")).encode("utf-8") + b"\n"
            )
            view_indices = torch.randint(
                0,
                len(TRAINING_VIEWS),
                (len(ids),),
                generator=view_generator,
            ).tolist()
            selected_views = [TRAINING_VIEWS[index] for index in view_indices]
            view_counts.update(selected_views)
            view_schedule_digest.update(
                json.dumps(selected_views, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
            treatment_examples = []
            for clip_id, view in zip(ids, selected_views, strict=True):
                clean_example = clean_examples_by_id[clip_id]
                treatment_examples.append(
                    {
                        "features": training_features[clip_id][view],
                        "targets": clean_example["targets"],
                        "item": clean_example["item"],
                    }
                )
            treatment_batch = collate_distillation_batch(treatment_examples)
            if treatment_batch["ids"] != ids:
                raise AssertionError("treatment minibatch order differs from control")
            if not torch.equal(treatment_batch["lengths"], clean_batch["lengths"]):
                raise AssertionError("treatment lengths differ from control")
            if not torch.equal(treatment_batch["targets"], clean_batch["targets"]):
                raise AssertionError("treatment did not retain exact clean targets")

            control_loss, control_grad, control_stats = train_step(
                clean_control,
                optimizers["clean_control"],
                clean_batch,
                step=step,
            )
            treatment_loss, treatment_grad, treatment_stats = train_step(
                cross_view,
                optimizers["cross_view"],
                treatment_batch,
                step=step,
            )
            if control_stats["valid_frames"] != treatment_stats["valid_frames"]:
                raise AssertionError("paired arms used different valid-frame counts")
            losses["clean_control"].append(control_loss)
            losses["cross_view"].append(treatment_loss)
            gradient_norms["clean_control"].append(control_grad)
            gradient_norms["cross_view"].append(treatment_grad)
            examples_seen["clean_control"] += len(ids)
            examples_seen["cross_view"] += len(ids)
            successful_steps += 1
            if (
                successful_steps == 1
                or successful_steps % 100 == 0
                or successful_steps == args.steps
            ):
                print(
                    f"step={successful_steps:04d} "
                    f"control={control_loss:.6f} cross_view={treatment_loss:.6f}",
                    flush=True,
                )
            if successful_steps >= args.steps:
                break

    train_summaries = {
        arm: training_summary(
            models[arm],
            optimizers[arm],
            losses[arm],
            gradient_norms[arm],
            steps=args.steps,
            examples_seen=examples_seen[arm],
            num_examples=len(dataset),
        )
        for arm in models
    }
    submitted_baseline_exactly_reproduced = (
        train_summaries["clean_control"]["final_model_state_sha256"]
        == reference_model_state_sha256
    )
    print(
        "finished paired training; exact submitted-control reproduction="
        f"{submitted_baseline_exactly_reproduced}",
        flush=True,
    )

    for model in models.values():
        model.eval()
    evaluation_waveforms: dict[str, dict[str, torch.Tensor]] = {}
    evaluation_transform_diagnostics = {
        condition: [] for condition in EVALUATION_CONDITIONS
    }
    with torch.inference_mode():
        for item in heldout_records + switch_records:
            waveform = load_audio(resolve_audio_path(item, args.manifest))
            evaluation_waveforms[item["id"]] = {}
            for condition in EVALUATION_CONDITIONS:
                transformed, diagnostics = telephony.transform_waveform(
                    waveform,
                    item["id"],
                    telephony.CONDITION_BY_NAME[condition],
                )
                evaluation_waveforms[item["id"]][condition] = transformed
                evaluation_transform_diagnostics[condition].append(diagnostics)

    monolingual_rows: list[dict[str, Any]] = []
    for condition in EVALUATION_CONDITIONS:
        for item in heldout_records:
            whole = evaluation_waveforms[item["id"]][condition]
            for seconds in PREFIX_SECONDS:
                prefix, padded = telephony.prefix_audio(whole, seconds)
                row: dict[str, Any] = {
                    "clip_id": item["id"],
                    "language": item["language"],
                    "speaker_id": item["speaker_id"],
                    "expected": item["language"],
                    "condition": condition,
                    "prefix_seconds": seconds,
                    "padded": padded,
                }
                for arm, model in models.items():
                    probabilities, valid_frames = telephony.student_posterior(
                        model,
                        frontend,
                        prefix,
                        chunk_frames=CHUNK_FRAMES,
                        label_delay_frames=LABEL_DELAY_FRAMES,
                    )
                    row[arm] = {
                        **probability_payload(probabilities),
                        "valid_frames": valid_frames,
                    }
                monolingual_rows.append(row)
        print(f"evaluated held-out prefixes: {condition}", flush=True)

    arm_summaries = {
        arm: summarize_arm(monolingual_rows, arm) for arm in models
    }
    comparison = cross_arm_summary(monolingual_rows)

    switch_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for condition in EVALUATION_CONDITIONS:
        condition_spec = telephony.CONDITION_BY_NAME[condition]
        filter_delay_ms = (
            1_000
            * ((telephony.FIR_TAPS - 1) / 2)
            / telephony.TELEPHONY_SAMPLE_RATE
            if condition_spec["bandpass"]
            else 0.0
        )
        switch_rows[condition] = {arm: [] for arm in models}
        for item in switch_records:
            waveform = evaluation_waveforms[item["id"]][condition]
            for arm, model in models.items():
                switch_rows[condition][arm].append(
                    telephony.student_switch_result(
                        model,
                        frontend,
                        waveform,
                        item,
                        chunk_frames=CHUNK_FRAMES,
                        label_delay_frames=LABEL_DELAY_FRAMES,
                        filter_delay_ms=filter_delay_ms,
                    )
                )
        print(f"evaluated switch clips: {condition}", flush=True)
    switch_summary, switch_comparison = summarize_switches(switch_rows)

    heldout_clean_waveforms = [
        evaluation_waveforms[item["id"]]["clean"] for item in heldout_records
    ]
    runtime_benchmark = benchmark_models(
        models, heldout_clean_waveforms, repeats=args.benchmark_repeats
    )

    pooled = comparison["pooled_degraded"]
    architecture_latency_unchanged = (
        clean_control.parameter_count == cross_view.parameter_count
        and clean_control.lookahead_frames == cross_view.lookahead_frames
        and clean_control.receptive_field_frames == cross_view.receptive_field_frames
    )
    decision_gates = {
        "recovery_gate_evaluable": pooled["recovery_gate_evaluable"],
        "recovered_at_least_half_of_pooled_channel_loss": pooled[
            "recovered_at_least_half"
        ],
        "clean_macro_f1_loss_le_2pp": pooled["clean_loss_le_2pp"],
        "no_additional_switch_misses": switch_comparison[
            "no_additional_switch_misses"
        ],
        "architecture_and_algorithmic_latency_unchanged": architecture_latency_unchanged,
        "both_training_contracts_passed": all(
            summary["contract"]["nan_free"] for summary in train_summaries.values()
        ),
    }
    if not decision_gates["recovery_gate_evaluable"]:
        verdict = "inconclusive"
        verdict_reason = "the clean control had no positive pooled channel loss to recover"
    elif all(decision_gates.values()):
        verdict = "adopt"
        verdict_reason = (
            "cross-view KD passed the predeclared pooled recovery, clean-retention, "
            "switch-miss, latency, and training-validity gates"
        )
    else:
        verdict = "reject"
        failed = [name for name, passed in decision_gates.items() if not passed]
        verdict_reason = "failed predeclared gates: " + ", ".join(failed)

    # Revalidate every mutable input after the expensive work.  Publication is
    # refused if another process changed the main pipeline, corpus, targets,
    # reference bundle, transform helper, or this driver during the run.
    target_cache.validate_all()
    assert_run_dependency_snapshot_unchanged(
        dependency_snapshot,
        records=records,
        manifest_path=args.manifest,
        target_cache_identity=target_cache.identity,
        target_metadata_path=target_cache.targets_dir / "metadata.json",
        training=training_settings,
        stage="experiment results publication",
    )
    source_snapshot_at_end = snapshot_sources()
    if source_snapshot_at_end != source_snapshot_at_start:
        raise RuntimeError("experiment or imported source changed during the run")
    if input_fingerprint(records, args.manifest) != full_input_fingerprint_at_start:
        raise RuntimeError("manifest or audio changed during the run")
    reference_file_hashes_at_end = {
        "summary": file_sha256(args.reference_summary),
        "checkpoint": file_sha256(args.reference_checkpoint),
    }
    if reference_file_hashes_at_end != reference_file_hashes_at_start:
        raise RuntimeError("main reference summary or checkpoint changed during the run")

    experiment_configuration = {
        "analysis_version": ANALYSIS_VERSION,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "seed": args.seed,
        "view_seed": args.view_seed,
        "training_views": list(TRAINING_VIEWS),
        "training_view_sampling": "independent uniform draw per example presentation",
        "evaluation_conditions": list(EVALUATION_CONDITIONS),
        "prefix_seconds": list(PREFIX_SECONDS),
        "recovery_fraction_gate": RECOVERY_FRACTION_GATE,
        "clean_macro_f1_loss_gate_pp": CLEAN_MACRO_F1_LOSS_GATE_PP,
        "threads": args.threads,
    }
    experiment_identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "configuration": experiment_configuration,
        "source_files": source_snapshot_at_start,
        "dependency_snapshot": dependency_snapshot,
        "full_input_fingerprint": full_input_fingerprint_at_start,
        "evaluation_input_fingerprint": eval_input_fingerprint_at_start,
        "reference_file_hashes": reference_file_hashes_at_start,
        "reference_model_state_sha256": reference_model_state_sha256,
        "initial_model_state_sha256": initial_model_sha256,
        "batch_order_sha256": batch_order_digest.hexdigest(),
        "view_schedule_sha256": view_schedule_digest.hexdigest(),
        "final_model_state_sha256": {
            arm: summary["final_model_state_sha256"]
            for arm, summary in train_summaries.items()
        },
    }
    run_id = canonical_sha256(experiment_identity)
    results = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "question": (
            "Can clean-teacher/degraded-student cross-view KD recover at least "
            "half of the causal TCN's paired narrowband/PCMA/PCMU macro-F1 loss "
            "without losing more than two clean points or adding switch misses?"
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
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "data": {
            "n_train_clips": len(train_records),
            "n_heldout_clips": len(heldout_records),
            "n_switch_clips": len(switch_records),
            "heldout_clip_ids": [item["id"] for item in heldout_records],
            "switch_clip_ids": [item["id"] for item in switch_records],
            "speaker_audit": speaker_audit,
            "full_input_fingerprint": full_input_fingerprint_at_start,
            "evaluation_input_fingerprint": eval_input_fingerprint_at_start,
            "target_cache": target_cache.audit(),
        },
        "identity_audit": {
            "driver_sha256": driver_sha256,
            "telephony_driver_sha256": telephony_driver_sha256,
            "source_snapshot_unchanged": True,
            "dependency_snapshot_unchanged": True,
            "manifest_and_audio_unchanged": True,
            "target_cache_revalidated_at_end": True,
            "reference_bundle_unchanged": True,
            "paired_initial_state_identical": True,
            "submitted_baseline_exactly_reproduced": submitted_baseline_exactly_reproduced,
            "reference_model_state_sha256": reference_model_state_sha256,
        },
        "transform_validation": transform_validation,
        "transform_diagnostics": {
            "training": telephony.aggregate_transform_diagnostics(
                training_transform_diagnostics
            ),
            "evaluation": telephony.aggregate_transform_diagnostics(
                evaluation_transform_diagnostics
            ),
        },
        "training": {
            "view_counts": dict(sorted(view_counts.items())),
            "batch_order_sha256": batch_order_digest.hexdigest(),
            "view_schedule_sha256": view_schedule_digest.hexdigest(),
            "initial_model_state_sha256": initial_model_sha256,
            "arms": train_summaries,
        },
        "monolingual_prefix_rows": monolingual_rows,
        "monolingual_summary": arm_summaries,
        "cross_arm_comparison": comparison,
        "switch_summary": switch_summary,
        "switch_comparison": switch_comparison,
        "runtime_benchmark": runtime_benchmark,
        "latency": {
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
                arm: ALGORITHMIC_LATENCY_MS for arm in models
            },
            "algorithmic_latency_delta_ms": 0.0,
            "architecture_unchanged": architecture_latency_unchanged,
        },
        "decision": {
            "gates": decision_gates,
            "verdict": verdict,
            "reason": verdict_reason,
        },
        "limitations": [
            "one training seed on a 71-clip synthetic corpus",
            "21 synthetic held-out clips and two mirrored synthetic switches",
            "the main switch target still uses the known future-leaking linear expansion",
            "channel transforms are simulations, not a real 8 kHz call benchmark",
            "policy switch comparison can be uninformative when the clean control already misses",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(
        f"wrote {args.output}; verdict={verdict}; run_id={run_id}",
        flush=True,
    )


if __name__ == "__main__":
    main()
