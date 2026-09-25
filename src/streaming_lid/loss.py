"""Temporal alignment for offline-teacher to delayed-student distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def alignment_mask(
    lengths: torch.Tensor,
    max_target_frames: int,
    delay_frames: int,
    lookahead_frames: int,
) -> torch.Tensor:
    """Mask target i when prediction i+delay has all its right context."""
    target_index = torch.arange(max_target_frames, device=lengths.device)
    latest_needed = target_index[None, :] + delay_frames + lookahead_frames
    return latest_needed < lengths[:, None]


def delayed_distillation_loss(
    student_logits: torch.Tensor,
    teacher_soft_targets: torch.Tensor,
    lengths: torch.Tensor,
    *,
    delay_frames: int,
    lookahead_frames: int,
    temperature: float,
    early_ramp_frames: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute T^2 KL(q_i || p_{i+D}) with padding and early-frame weights.

    Raises ``ValueError`` instead of returning a zero-gradient loss when no
    item has a target frame with all required delay and lookahead context.
    """
    if student_logits.shape[:2] != teacher_soft_targets.shape[:2]:
        raise ValueError("student and teacher must have equal batch/time dimensions")
    if student_logits.shape[-1] != teacher_soft_targets.shape[-1]:
        raise ValueError("student and teacher class counts differ")
    total_frames = student_logits.shape[1]
    aligned_frames = total_frames - delay_frames
    if aligned_frames <= 0:
        raise ValueError("sequence is shorter than the requested delay")

    valid = alignment_mask(lengths, aligned_frames, delay_frames, lookahead_frames)
    valid_frames = int(valid.sum().item())
    if valid_frames == 0:
        minimum_length = delay_frames + lookahead_frames + 1
        longest_item = int(lengths.max().item()) if lengths.numel() else 0
        raise ValueError(
            "distillation batch has no valid aligned frames: target frame 0 "
            f"requires length >= {minimum_length}, but the longest item has "
            f"length {longest_item}"
        )

    target = teacher_soft_targets[:, :aligned_frames]
    prediction = student_logits[:, delay_frames : delay_frames + aligned_frames]
    log_prediction = F.log_softmax(prediction / temperature, dim=-1)
    per_frame = F.kl_div(log_prediction, target, reduction="none").sum(dim=-1)

    frame_index = torch.arange(
        aligned_frames, device=student_logits.device, dtype=per_frame.dtype
    )
    ramp = ((frame_index + 1.0) / float(early_ramp_frames)).clamp(max=1.0)
    weights = valid.to(per_frame.dtype) * ramp[None, :]
    denominator = weights.sum()
    loss = temperature**2 * (per_frame * weights).sum() / denominator
    stats = {
        "valid_frames": float(valid_frames),
        "weight_sum": float(denominator.item()),
        "mean_teacher_entropy": float(
            (-(target.clamp_min(1e-8).log() * target).sum(dim=-1) * weights)
            .sum()
            .item()
            / denominator.item()
        ),
    }
    return loss, stats
