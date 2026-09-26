#!/usr/bin/env python3
"""Run real CPU distillation steps on generated audio and cached teacher targets."""

from __future__ import annotations

import sys

_SOURCE_STAGE_CONTEXT = None
if __name__ == "__main__":
    from source_stage import (
        is_staged_child,
        launch_in_source_snapshot,
        prepare_staged_child,
    )

    if not is_staged_child():
        raise SystemExit(
            launch_in_source_snapshot("scripts/train.py", sys.argv[1:])
        )
    _SOURCE_STAGE_CONTEXT = prepare_staged_child("scripts/train.py")

import argparse
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from streaming_lid.config import (
    EARLY_RAMP_FRAMES,
    LABEL_DELAY_FRAMES,
    LANGUAGE_CODES,
    MODEL_LOOKAHEAD_FRAMES,
    TEACHER_NAME,
    TEACHER_TEMPERATURE,
)
from streaming_lid.data import (
    DistillationDataset,
    TeacherTargetCache,
    capture_manifest_snapshot,
    collate_distillation_batch,
    file_sha256,
    require_speaker_disjoint,
)
from streaming_lid.loss import delayed_distillation_loss
from streaming_lid.model import CausalLIDStudent
from streaming_lid.run_identity import (
    CHECKPOINT_SCHEMA_VERSION,
    assert_run_dependency_snapshot_unchanged,
    build_run_identity,
    capture_run_dependency_snapshot,
    configured_model_kwargs,
    run_id_for_identity,
    training_configuration,
)


def positive_int_arg(value: str) -> int:
    """Parse a strictly positive integer for a training CLI option."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {value!r}")
    return parsed


def positive_finite_float_arg(value: str) -> float:
    """Parse a finite, strictly positive floating-point training option."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a finite number > 0, got {value!r}"
        )
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/generated/manifest.jsonl")
    )
    parser.add_argument(
        "--targets-dir", type=Path, default=Path("data/generated/targets")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/student.pt")
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--steps", type=positive_int_arg, default=1600)
    parser.add_argument("--batch-size", type=positive_int_arg, default=7)
    parser.add_argument(
        "--learning-rate", type=positive_finite_float_arg, default=1e-3
    )
    parser.add_argument("--threads", type=positive_int_arg, default=6)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_training_args(args: argparse.Namespace) -> None:
    """Fail closed even when ``main`` receives a hand-built Namespace."""
    for name in ("steps", "batch_size", "threads"):
        value = getattr(args, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be an integer >= 1")
    if (
        not isinstance(args.learning_rate, (int, float))
        or isinstance(args.learning_rate, bool)
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        raise ValueError("learning-rate must be a finite number > 0")


def validate_full_batch_training_set(num_examples: int, batch_size: int) -> None:
    """Prevent an endless loop when ``drop_last=True`` yields no batches."""
    if num_examples < batch_size:
        raise ValueError(
            f"training split has {num_examples} examples, fewer than batch-size "
            f"{batch_size}; drop_last=True would yield no optimizer steps"
        )


def _nonfinite_paths(value: Any, prefix: str) -> list[str]:
    """Return paths to every non-finite tensor or float in a nested state."""
    if isinstance(value, torch.Tensor):
        return [] if bool(torch.isfinite(value).all()) else [prefix]
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, nested in value.items():
            paths.extend(_nonfinite_paths(nested, f"{prefix}.{key}"))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for index, nested in enumerate(value):
            paths.extend(_nonfinite_paths(nested, f"{prefix}[{index}]"))
        return paths
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    return []


def training_state_finiteness(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[bool, bool]:
    """Check both the published model state and all optimizer state/parameters."""
    model_finite = not _nonfinite_paths(model.state_dict(), "model")
    optimizer_finite = not _nonfinite_paths(optimizer.state_dict(), "optimizer")
    return model_finite, optimizer_finite


def assert_finite_post_update_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
) -> None:
    """Reject a corrupt optimizer update before it can count or be saved."""
    bad_model = _nonfinite_paths(model.state_dict(), "model")
    bad_optimizer = _nonfinite_paths(optimizer.state_dict(), "optimizer")
    if bad_model or bad_optimizer:
        details = ", ".join((bad_model + bad_optimizer)[:5])
        raise FloatingPointError(
            f"non-finite post-update training state at step {step}: {details}"
        )


def build_training_contract(
    *,
    requested_steps: int,
    successful_steps: int,
    post_update_checks: int,
    examples_seen: int,
    losses: Sequence[float],
    gradient_norms: Sequence[float],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, bool | int]:
    """Derive publication flags from observed work instead of literals."""
    model_finite, optimizer_finite = training_state_finiteness(model, optimizer)
    histories_complete = (
        len(losses) == successful_steps == len(gradient_norms)
        and post_update_checks == successful_steps
    )
    histories_finite = all(
        math.isfinite(value) for value in [*losses, *gradient_norms]
    )
    all_requested_steps_completed = (
        requested_steps >= 1 and successful_steps == requested_steps
    )
    real_audio_optimizer_step = successful_steps > 0 and examples_seen > 0
    nan_free = (
        all_requested_steps_completed
        and histories_complete
        and histories_finite
        and model_finite
        and optimizer_finite
    )
    return {
        "requested_optimizer_steps": requested_steps,
        "successful_optimizer_steps": successful_steps,
        "post_update_checks": post_update_checks,
        "all_requested_steps_completed": all_requested_steps_completed,
        "loss_and_gradient_histories_finite": histories_finite,
        "post_update_model_state_finite": model_finite,
        "post_update_optimizer_state_finite": optimizer_finite,
        "real_audio_optimizer_step": real_audio_optimizer_step,
        "nan_free": nan_free,
    }


def main() -> None:
    if _SOURCE_STAGE_CONTEXT is None:
        raise RuntimeError(
            "training must be launched through scripts/train.py so project imports "
            "come from a verified source stage"
        )
    args = parse_args()
    validate_training_args(args)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
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
    launch_dependency_snapshot = capture_run_dependency_snapshot(
        manifest_snapshot=manifest_snapshot,
        target_cache_identity=target_cache.identity,
        target_metadata_path=target_cache.targets_dir / "metadata.json",
        training=training_settings,
        require_executed_source=True,
    )
    # The run identity names the complete target release, so validate every
    # target/audio file at launch even though optimization consumes only train.
    target_cache.validate_all()
    assert_run_dependency_snapshot_unchanged(
        launch_dependency_snapshot,
        manifest_snapshot=manifest_snapshot,
        target_cache_identity=target_cache.identity,
        target_metadata_path=target_cache.targets_dir / "metadata.json",
        training=training_settings,
        stage="dataset preload",
        require_executed_source=True,
    )
    dataset = DistillationDataset(
        manifest_snapshot,
        args.targets_dir,
        splits=("train",),
        target_cache=target_cache,
    )
    assert_run_dependency_snapshot_unchanged(
        launch_dependency_snapshot,
        manifest_snapshot=manifest_snapshot,
        target_cache_identity=target_cache.identity,
        target_metadata_path=target_cache.targets_dir / "metadata.json",
        training=training_settings,
        stage="optimization",
        require_executed_source=True,
    )
    validate_full_batch_training_set(len(dataset), args.batch_size)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_distillation_batch,
        generator=generator,
        num_workers=0,
        drop_last=True,
    )
    model_kwargs = configured_model_kwargs()
    model = CausalLIDStudent(**model_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    losses: list[float] = []
    gradient_norms: list[float] = []
    successful_steps = 0
    post_update_checks = 0
    examples_seen = 0

    print(
        f"training {model.parameter_count:,}-parameter student on {len(dataset)} real-audio clips "
        f"for {args.steps} optimizer steps",
        flush=True,
    )
    model.train()
    while successful_steps < args.steps:
        for batch in loader:
            attempted_step = successful_steps + 1
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
                    f"non-finite loss at step {attempted_step}: {loss.item()}"
                )
            loss.backward()
            gradients_finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            if not gradients_finite:
                raise FloatingPointError(
                    f"non-finite gradient at step {attempted_step}"
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0, error_if_nonfinite=True
            )
            optimizer.step()
            assert_finite_post_update_state(
                model, optimizer, step=attempted_step
            )
            post_update_checks += 1
            successful_steps += 1
            examples_seen += len(batch["ids"])
            losses.append(float(loss.detach()))
            gradient_norms.append(float(gradient_norm))
            if (
                successful_steps == 1
                or successful_steps % 10 == 0
                or successful_steps == args.steps
            ):
                print(
                    f"step={successful_steps:03d} loss={losses[-1]:.6f} "
                    f"grad_norm={gradient_norms[-1]:.4f} "
                    f"valid_frames={loss_stats['valid_frames']:.0f}",
                    flush=True,
                )
            if successful_steps >= args.steps:
                break

    training_contract = build_training_contract(
        requested_steps=args.steps,
        successful_steps=successful_steps,
        post_update_checks=post_update_checks,
        examples_seen=examples_seen,
        losses=losses,
        gradient_norms=gradient_norms,
        model=model,
        optimizer=optimizer,
    )
    if not training_contract["nan_free"] or not training_contract[
        "real_audio_optimizer_step"
    ]:
        raise RuntimeError(f"training contract failed: {training_contract}")

    window = min(10, len(losses))
    first_mean = float(np.mean(losses[:window]))
    last_mean = float(np.mean(losses[-window:]))
    monolingual_train = [
        item for item in dataset.items if item["language"] in LANGUAGE_CODES
    ]
    clips_per_language = {
        language: sum(item["language"] == language for item in monolingual_train)
        for language in LANGUAGE_CODES
    }
    training_evidence = {
        "optimizer_steps": successful_steps,
        "requested_optimizer_steps": training_contract[
            "requested_optimizer_steps"
        ],
        "successful_optimizer_steps": training_contract[
            "successful_optimizer_steps"
        ],
        "post_update_checks": training_contract["post_update_checks"],
        "all_requested_steps_completed": training_contract[
            "all_requested_steps_completed"
        ],
        "batch_size": args.batch_size,
        "n_train_clips": len(dataset),
        "n_train_monolingual_clips": len(monolingual_train),
        "n_train_switch_clips": len(dataset) - len(monolingual_train),
        "train_clips_per_language": clips_per_language,
        "examples_seen": examples_seen,
        "effective_epochs": examples_seen / len(dataset),
        "n_train_speakers": len(speaker_audit["train_speaker_ids"]),
        "speaker_split": speaker_audit,
        "target_cache": target_cache.audit(),
        "student_params": model.parameter_count,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "first_10_mean_loss": first_mean,
        "last_10_mean_loss": last_mean,
        "loss_decreased": last_mean < first_mean,
        "loss_and_gradient_histories_finite": training_contract[
            "loss_and_gradient_histories_finite"
        ],
        "post_update_model_state_finite": training_contract[
            "post_update_model_state_finite"
        ],
        "post_update_optimizer_state_finite": training_contract[
            "post_update_optimizer_state_finite"
        ],
        "nan_free": training_contract["nan_free"],
        "real_audio_optimizer_step": training_contract[
            "real_audio_optimizer_step"
        ],
        "launch_dependency_snapshot_captured_before_preload": True,
        "manifest_snapshot_captured_once_before_consumers": True,
        "manifest_bindings_equal": launch_dependency_snapshot[
            "manifest_bindings_equal"
        ],
        "dependency_snapshot_validation_checks": 3,
        "dependency_snapshot_unchanged_at_publication": True,
    }
    # Reopen every indexed target and rehash all live source/data metadata at
    # the publication boundary. The final identity is built from the immutable
    # launch snapshot, never from whatever happens to be on disk now.
    target_cache.validate_all()
    assert_run_dependency_snapshot_unchanged(
        launch_dependency_snapshot,
        manifest_snapshot=manifest_snapshot,
        target_cache_identity=target_cache.identity,
        target_metadata_path=target_cache.targets_dir / "metadata.json",
        training=training_settings,
        stage="checkpoint publication",
        require_executed_source=True,
    )
    from source_stage import stage_verification_count

    training_evidence.update(
        {
            "executed_source_snapshot_validated": True,
            "source_stage_verified_before_import": True,
            "source_stage_verified_before_publication": True,
            "source_stage_verification_checks": stage_verification_count(),
            "training_child_entrypoint": "scripts/train.py",
        }
    )
    run_identity = build_run_identity(
        dependency_snapshot=launch_dependency_snapshot,
        model_state=model.state_dict(),
    )
    run_id = run_id_for_identity(run_identity)
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_identity": run_identity,
        "launch_dependency_snapshot": launch_dependency_snapshot,
        "model_state": model.state_dict(),
        "languages": list(LANGUAGE_CODES),
        "teacher": TEACHER_NAME,
        "target_cache": target_cache.identity,
        "steps": successful_steps,
        "training_contract": training_contract,
        "training_evidence": training_evidence,
        "seed": args.seed,
        "model_kwargs": model_kwargs,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    metrics = {
        "run_id": run_id,
        "run_identity": run_identity,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        **training_evidence,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"saved {args.checkpoint}; first-{window} mean={first_mean:.6f}, "
        f"last-{window} mean={last_mean:.6f}, nan_free={metrics['nan_free']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
