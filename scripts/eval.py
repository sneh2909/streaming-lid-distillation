#!/usr/bin/env python3
"""Evaluate agreement, CPU RTF, and Hindi-to-English streaming switch lag."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from streaming_lid.audio import LogMelFrontend, load_audio
from streaming_lid.config import (
    ALGORITHMIC_LATENCY_MS,
    CHUNK_FRAMES,
    CHUNK_MS,
    HOP_LENGTH,
    LABEL_DELAY_FRAMES,
    LANGUAGE_CODES,
    MODEL_LOOKAHEAD_FRAMES,
    SAMPLE_RATE,
    TEACHER_NAME,
    WIN_LENGTH,
)
from streaming_lid.data import read_manifest, resolve_audio_path
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
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def aligned_agreement(
    student_logits: torch.Tensor, teacher_probs: torch.Tensor, length: int
) -> tuple[int, int]:
    valid_targets = length - LABEL_DELAY_FRAMES - MODEL_LOOKAHEAD_FRAMES
    if valid_targets <= 0:
        return 0, 0
    student_class = student_logits[
        LABEL_DELAY_FRAMES : LABEL_DELAY_FRAMES + valid_targets
    ].argmax(-1)
    teacher_class = teacher_probs[:valid_targets].argmax(-1)
    return int((student_class == teacher_class).sum()), valid_targets


def chunk_availability_times(num_frames: int) -> np.ndarray:
    """Wall-clock audio time at which each chunk's logits become available."""
    times = np.empty(num_frames, dtype=np.float32)
    for start in range(0, num_frames, CHUNK_FRAMES):
        n_emit = min(CHUNK_FRAMES, num_frames - start)
        latest_feature = start + n_emit - 1 + MODEL_LOOKAHEAD_FRAMES
        available_seconds = (latest_feature * HOP_LENGTH + WIN_LENGTH) / SAMPLE_RATE
        times[start : start + n_emit] = available_seconds
    return times


def chunk_posteriors(
    probabilities: np.ndarray, availability_seconds: np.ndarray, min_frame: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    chunks = []
    times = []
    frame_index = np.arange(len(probabilities))
    for available_time in np.unique(availability_seconds):
        selected = (availability_seconds == available_time) & (frame_index >= min_frame)
        if selected.any():
            chunks.append(probabilities[selected].mean(axis=0))
            times.append(available_time)
    return np.stack(chunks), np.asarray(times)


def smooth_posteriors(probabilities: np.ndarray, new_weight: float) -> np.ndarray:
    smoothed = np.empty_like(probabilities)
    smoothed[0] = probabilities[0]
    for index in range(1, len(probabilities)):
        smoothed[index] = (1.0 - new_weight) * smoothed[
            index - 1
        ] + new_weight * probabilities[index]
    return smoothed


def detect_hi_to_en_switch(
    probabilities: np.ndarray, availability_seconds: np.ndarray
) -> tuple[float | None, dict]:
    """Chunk-rate EMA + margin + dwell, after a stable initial Hindi commit."""
    hi = LANGUAGE_CODES.index("hi")
    en = LANGUAGE_CODES.index("en")
    challenger_chunks = 0
    threshold = 0.60
    margin = 0.10
    dwell_chunks = 3
    ema_new_weight = 0.30
    smoothed = smooth_posteriors(probabilities, ema_new_weight)
    active_chunks = 0
    armed = False
    initial_commit_seconds = None
    for chunk_index, ema in enumerate(smoothed):
        if not armed:
            if ema[hi] >= threshold and ema[hi] - ema[en] >= margin:
                active_chunks += 1
            else:
                active_chunks = 0
            armed = active_chunks >= dwell_chunks
            if armed:
                initial_commit_seconds = float(availability_seconds[chunk_index])
            continue
        if ema[en] >= threshold and ema[en] - ema[hi] >= margin:
            challenger_chunks += 1
        else:
            challenger_chunks = 0
        if challenger_chunks >= dwell_chunks:
            return float(availability_seconds[chunk_index]), {
                "threshold": threshold,
                "margin": margin,
                "dwell_chunks": dwell_chunks,
                "dwell_ms": dwell_chunks * CHUNK_MS,
                "ema_new_weight": ema_new_weight,
                "initial_commit_seconds": initial_commit_seconds,
            }
    return None, {
        "threshold": threshold,
        "margin": margin,
        "dwell_chunks": dwell_chunks,
        "dwell_ms": dwell_chunks * CHUNK_MS,
        "ema_new_weight": ema_new_weight,
        "initial_commit_seconds": initial_commit_seconds,
    }


def benchmark_rtf(
    model: CausalLIDStudent,
    frontend: LogMelFrontend,
    waveforms: list[torch.Tensor],
    repeats: int,
) -> tuple[float, list[float]]:
    total_audio_seconds = sum(len(waveform) for waveform in waveforms) / SAMPLE_RATE
    with torch.inference_mode():
        warm_features = frontend(waveforms[0])
        model.streaming_forward(warm_features)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            for waveform in waveforms:
                features = frontend(waveform)
                model.streaming_forward(features)
            timings.append(time.perf_counter() - start)
    rtfs = [elapsed / total_audio_seconds for elapsed in timings]
    return float(np.median(rtfs)), rtfs


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if tuple(checkpoint["languages"]) != LANGUAGE_CODES:
        raise ValueError("checkpoint language order differs from current configuration")
    model = CausalLIDStudent(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    frontend = LogMelFrontend().eval()
    records = read_manifest(args.manifest)
    heldout = [record for record in records if record["split"] == "heldout"]
    train_records = [record for record in records if record["split"] == "train"]
    switch_records = [record for record in records if record["split"] == "switch"]

    per_clip = []
    correct = 0
    total = 0
    heldout_waveforms = []
    with torch.inference_mode():
        for item in heldout:
            waveform = load_audio(resolve_audio_path(item, args.manifest))
            heldout_waveforms.append(waveform)
            features = frontend(waveform)
            full_logits = model(features).squeeze(0)
            streamed_logits = model.streaming_forward(features).squeeze(0)
            torch.testing.assert_close(
                full_logits, streamed_logits, rtol=1e-5, atol=1e-5
            )
            with np.load(args.targets_dir / f"{item['id']}.npz") as target_file:
                teacher_probs = torch.from_numpy(target_file["teacher_probs"].copy())
            clip_correct, clip_total = aligned_agreement(
                streamed_logits, teacher_probs, len(features[0])
            )
            correct += clip_correct
            total += clip_total
            per_clip.append(
                {
                    "id": item["id"],
                    "agreement": clip_correct / clip_total,
                    "frames": clip_total,
                }
            )
    heldout_agreement = correct / total

    switch_item = next(
        item for item in switch_records if item["id"] == "switch_hi_en_eval"
    )
    switch_waveform = load_audio(resolve_audio_path(switch_item, args.manifest))
    with torch.inference_mode():
        switch_features = frontend(switch_waveform)
        switch_logits = model.streaming_forward(switch_features).squeeze(0)
        switch_probabilities = torch.softmax(switch_logits, dim=-1).numpy()
    with np.load(args.targets_dir / f"{switch_item['id']}.npz") as target_file:
        teacher_switch = target_file["teacher_probs"].copy()
    availability = chunk_availability_times(len(switch_probabilities))
    # Frames before D have no corresponding teacher target and are never trained.
    student_chunks, student_chunk_times = chunk_posteriors(
        switch_probabilities, availability, min_frame=LABEL_DELAY_FRAMES
    )
    student_smoothed = smooth_posteriors(student_chunks, new_weight=0.30)
    detected_seconds, detector_config = detect_hi_to_en_switch(
        student_chunks, student_chunk_times
    )
    true_switch_seconds = float(switch_item["segments"][0]["end_seconds"])
    switch_lag_ms = (
        None
        if detected_seconds is None
        else 1_000 * (detected_seconds - true_switch_seconds)
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    target_time = (
        np.arange(len(teacher_switch)) * HOP_LENGTH + WIN_LENGTH
    ) / SAMPLE_RATE
    hi = LANGUAGE_CODES.index("hi")
    en = LANGUAGE_CODES.index("en")
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(
        target_time,
        teacher_switch[:, hi],
        "--",
        color="#d95f02",
        label="teacher Hindi (offline window)",
    )
    axis.plot(
        target_time,
        teacher_switch[:, en],
        "--",
        color="#1b9e77",
        label="teacher English (offline window)",
    )
    axis.plot(
        student_chunk_times,
        student_smoothed[:, hi],
        color="#d95f02",
        label="student Hindi (chunk EMA)",
    )
    axis.plot(
        student_chunk_times,
        student_smoothed[:, en],
        color="#1b9e77",
        label="student English (chunk EMA)",
    )
    axis.axvline(true_switch_seconds, color="black", linewidth=1.4, label="true switch")
    if detected_seconds is not None:
        axis.axvline(
            detected_seconds,
            color="#7570b3",
            linestyle=":",
            linewidth=2,
            label="student commit",
        )
    axis.set(xlabel="audio wall-clock time (s)", ylabel="posterior", ylim=(-0.02, 1.02))
    lag_label = (
        "not detected" if switch_lag_ms is None else f"lag = {switch_lag_ms:.0f} ms"
    )
    axis.set_title(f"Hindi → English streaming LID ({lag_label})")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper center", ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(args.results_dir / "switch_plot.png", dpi=160)
    plt.close(figure)

    cpu_rtf, benchmark_rtfs = benchmark_rtf(
        model, frontend, heldout_waveforms, repeats=args.benchmark_repeats
    )
    train_metrics = json.loads((args.results_dir / "train_metrics.json").read_text())
    finite_eval = (
        math.isfinite(heldout_agreement)
        and math.isfinite(cpu_rtf)
        and np.isfinite(switch_probabilities).all()
        and np.isfinite(teacher_switch).all()
    )
    eval_metrics = {
        "heldout_agreement": heldout_agreement,
        "heldout_per_clip": per_clip,
        "switch_detected_seconds": detected_seconds,
        "true_switch_seconds": true_switch_seconds,
        "switch_lag_ms": switch_lag_ms,
        "detector": detector_config,
        "cpu_rtf": cpu_rtf,
        "cpu_rtf_runs": benchmark_rtfs,
        "cpu_threads": args.threads,
        "chunk_equivalence_checked": True,
        "nan_free": bool(finite_eval),
    }
    (args.results_dir / "eval_metrics.json").write_text(
        json.dumps(eval_metrics, indent=2) + "\n"
    )
    summary = {
        "teacher_name": TEACHER_NAME,
        "languages": list(LANGUAGE_CODES),
        "n_train_clips": len(train_records),
        "n_heldout_clips": len(heldout),
        "n_switch_eval_clips": len(switch_records),
        "student_params": model.parameter_count,
        "algorithmic_latency_ms": ALGORITHMIC_LATENCY_MS,
        "cpu_rtf": cpu_rtf,
        "losses": train_metrics["losses"],
        "optimizer_steps": train_metrics["optimizer_steps"],
        "first_10_mean_loss": train_metrics["first_10_mean_loss"],
        "last_10_mean_loss": train_metrics["last_10_mean_loss"],
        "loss_decreased": train_metrics["loss_decreased"],
        "heldout_agreement": heldout_agreement,
        "switch_lag_ms": switch_lag_ms,
        "nan_free": bool(train_metrics["nan_free"] and finite_eval),
    }
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"heldout agreement={heldout_agreement:.3f}; CPU RTF={cpu_rtf:.4f}; "
        f"switch lag={switch_lag_ms if switch_lag_ms is not None else 'not detected'} ms; "
        f"nan_free={summary['nan_free']}",
        flush=True,
    )
    print(
        f"wrote {args.results_dir / 'summary.json'} and {args.results_dir / 'switch_plot.png'}"
    )


if __name__ == "__main__":
    main()
