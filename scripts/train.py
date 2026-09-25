#!/usr/bin/env python3
"""Run real CPU distillation steps on generated audio and cached teacher targets."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

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
    collate_distillation_batch,
    read_manifest,
    require_speaker_disjoint,
)
from streaming_lid.loss import delayed_distillation_loss
from streaming_lid.model import CausalLIDStudent


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    records = read_manifest(args.manifest)
    speaker_audit = require_speaker_disjoint(records)
    dataset = DistillationDataset(args.manifest, args.targets_dir, splits=("train",))
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
    model = CausalLIDStudent(num_languages=len(LANGUAGE_CODES))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    losses: list[float] = []
    gradient_norms: list[float] = []
    nan_free = True
    step = 0
    examples_seen = 0

    print(
        f"training {model.parameter_count:,}-parameter student on {len(dataset)} real-audio clips "
        f"for {args.steps} optimizer steps",
        flush=True,
    )
    model.train()
    while step < args.steps:
        for batch in loader:
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
                nan_free = False
                raise FloatingPointError(
                    f"non-finite loss at step {step + 1}: {loss.item()}"
                )
            loss.backward()
            gradients_finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            if not gradients_finite:
                nan_free = False
                raise FloatingPointError(f"non-finite gradient at step {step + 1}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()
            step += 1
            examples_seen += len(batch["ids"])
            losses.append(float(loss.detach()))
            gradient_norms.append(float(gradient_norm))
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(
                    f"step={step:03d} loss={losses[-1]:.6f} grad_norm={gradient_norms[-1]:.4f} "
                    f"valid_frames={loss_stats['valid_frames']:.0f}",
                    flush=True,
                )
            if step >= args.steps:
                break

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "languages": list(LANGUAGE_CODES),
        "teacher": TEACHER_NAME,
        "steps": args.steps,
        "seed": args.seed,
        "model_kwargs": {
            "num_languages": len(LANGUAGE_CODES),
            "lookahead_frames": MODEL_LOOKAHEAD_FRAMES,
        },
    }
    torch.save(checkpoint, args.checkpoint)

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
    metrics = {
        "optimizer_steps": args.steps,
        "batch_size": args.batch_size,
        "n_train_clips": len(dataset),
        "n_train_monolingual_clips": len(monolingual_train),
        "n_train_switch_clips": len(dataset) - len(monolingual_train),
        "train_clips_per_language": clips_per_language,
        "examples_seen": examples_seen,
        "effective_epochs": examples_seen / len(dataset),
        "n_train_speakers": len(speaker_audit["train_speaker_ids"]),
        "speaker_split": speaker_audit,
        "student_params": model.parameter_count,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "first_10_mean_loss": first_mean,
        "last_10_mean_loss": last_mean,
        "loss_decreased": last_mean < first_mean,
        "nan_free": nan_free
        and all(math.isfinite(value) for value in losses + gradient_norms),
        "real_audio_optimizer_step": True,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(
        f"saved {args.checkpoint}; first-{window} mean={first_mean:.6f}, "
        f"last-{window} mean={last_mean:.6f}, nan_free={metrics['nan_free']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
