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
from streaming_lid.data import (
    TeacherTargetCache,
    file_sha256,
    read_manifest,
    require_speaker_disjoint,
    resolve_audio_path,
)
from streaming_lid.model import CausalLIDStudent
from streaming_lid.run_identity import validate_evaluation_run_contract


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
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


def aligned_classes(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    length: int,
    *,
    delay_frames: int,
    lookahead_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_targets = length - delay_frames - lookahead_frames
    if valid_targets <= 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    student_class = student_logits[delay_frames : delay_frames + valid_targets].argmax(
        -1
    )
    teacher_class = teacher_probs[:valid_targets].argmax(-1)
    return student_class, teacher_class


def chunk_availability_times(
    num_frames: int,
    *,
    chunk_frames: int,
    lookahead_frames: int,
    hop_length: int,
    win_length: int,
    sample_rate: int,
) -> np.ndarray:
    """Wall-clock audio time at which each chunk's logits become available."""
    times = np.empty(num_frames, dtype=np.float32)
    for start in range(0, num_frames, chunk_frames):
        n_emit = min(chunk_frames, num_frames - start)
        latest_feature = start + n_emit - 1 + lookahead_frames
        available_seconds = (
            latest_feature * hop_length + win_length
        ) / sample_rate
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
    probabilities: np.ndarray,
    availability_seconds: np.ndarray,
    *,
    language_codes: tuple[str, ...],
    chunk_ms: float,
) -> tuple[float | None, dict]:
    """Chunk-rate EMA + margin + dwell, after a stable initial Hindi commit."""
    hi = language_codes.index("hi")
    en = language_codes.index("en")
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
                "dwell_ms": dwell_chunks * chunk_ms,
                "ema_new_weight": ema_new_weight,
                "initial_commit_seconds": initial_commit_seconds,
            }
    return None, {
        "threshold": threshold,
        "margin": margin,
        "dwell_chunks": dwell_chunks,
        "dwell_ms": dwell_chunks * chunk_ms,
        "ema_new_weight": ema_new_weight,
        "initial_commit_seconds": initial_commit_seconds,
    }


def benchmark_rtf(
    model: CausalLIDStudent,
    frontend: LogMelFrontend,
    waveforms: list[torch.Tensor],
    repeats: int,
    *,
    sample_rate: int,
    chunk_frames: int,
) -> tuple[float, list[float]]:
    total_audio_seconds = sum(len(waveform) for waveform in waveforms) / sample_rate
    with torch.inference_mode():
        warm_features = frontend(waveforms[0])
        model.streaming_forward(warm_features, chunk_frames=chunk_frames)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            for waveform in waveforms:
                features = frontend(waveform)
                model.streaming_forward(features, chunk_frames=chunk_frames)
            timings.append(time.perf_counter() - start)
    rtfs = [elapsed / total_audio_seconds for elapsed in timings]
    return float(np.median(rtfs)), rtfs


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    checkpoint_sha256 = file_sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    records = read_manifest(args.manifest)
    speaker_audit = require_speaker_disjoint(records)
    target_cache = TeacherTargetCache(args.manifest, args.targets_dir)
    # Evaluation is also the release gate: validate every indexed target/audio,
    # not only the clips that happen to contribute to the headline metrics.
    target_cache.validate_all()
    train_metrics_path = args.results_dir / "train_metrics.json"
    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
    run_identity = validate_evaluation_run_contract(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        train_metrics=train_metrics,
        records=records,
        manifest_path=args.manifest,
        target_cache_identity=target_cache.identity,
        target_metadata_path=target_cache.targets_dir / "metadata.json",
    )
    pipeline = run_identity["pipeline"]
    language_codes = tuple(pipeline["language_codes"])
    frontend_config = pipeline["frontend"]
    distillation_config = pipeline["distillation"]
    streaming_config = pipeline["streaming"]
    label_delay_frames = int(distillation_config["label_delay_frames"])
    lookahead_frames = int(distillation_config["model_lookahead_frames"])
    chunk_frames = int(streaming_config["chunk_frames"])

    model = CausalLIDStudent(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    frontend = LogMelFrontend().eval()
    heldout = [record for record in records if record["split"] == "heldout"]
    train_records = [record for record in records if record["split"] == "train"]
    switch_records = [record for record in records if record["split"] == "switch"]

    per_clip = []
    agreement_correct = 0
    student_label_correct = 0
    teacher_label_correct = 0
    total = 0
    student_clip_correct = 0
    teacher_clip_correct = 0
    heldout_waveforms = []
    language_counts = {
        language: {
            "n_clips": 0,
            "frames": 0,
            "agreement_correct": 0,
            "student_label_correct": 0,
            "teacher_label_correct": 0,
            "student_clip_correct": 0,
            "teacher_clip_correct": 0,
        }
        for language in language_codes
    }
    with torch.inference_mode():
        for item in heldout:
            waveform = load_audio(resolve_audio_path(item, args.manifest))
            heldout_waveforms.append(waveform)
            features = frontend(waveform)
            full_logits = model(features).squeeze(0)
            streamed_logits = model.streaming_forward(
                features, chunk_frames=chunk_frames
            ).squeeze(0)
            stable_frames = len(full_logits) - model.lookahead_frames
            torch.testing.assert_close(
                full_logits[:stable_frames],
                streamed_logits,
                rtol=1e-5,
                atol=1e-5,
            )
            teacher_probs = torch.from_numpy(
                target_cache.load(
                    item,
                    "teacher_probs",
                    expected_frames=len(features[0]),
                )
            )
            student_class, teacher_class = aligned_classes(
                streamed_logits,
                teacher_probs,
                len(features[0]),
                delay_frames=label_delay_frames,
                lookahead_frames=lookahead_frames,
            )
            clip_total = len(student_class)
            if clip_total == 0:
                raise ValueError(f"held-out clip {item['id']} has no aligned frames")
            expected_index = language_codes.index(item["language"])
            clip_agreement_correct = int((student_class == teacher_class).sum())
            clip_student_label_correct = int((student_class == expected_index).sum())
            clip_teacher_label_correct = int((teacher_class == expected_index).sum())
            student_prediction_index = int(
                torch.bincount(student_class, minlength=len(language_codes)).argmax()
            )
            teacher_prediction_index = int(
                torch.bincount(teacher_class, minlength=len(language_codes)).argmax()
            )
            student_clip_is_correct = student_prediction_index == expected_index
            teacher_clip_is_correct = teacher_prediction_index == expected_index
            agreement_correct += clip_agreement_correct
            student_label_correct += clip_student_label_correct
            teacher_label_correct += clip_teacher_label_correct
            total += clip_total
            student_clip_correct += int(student_clip_is_correct)
            teacher_clip_correct += int(teacher_clip_is_correct)
            language_count = language_counts[item["language"]]
            language_count["n_clips"] += 1
            language_count["frames"] += clip_total
            language_count["agreement_correct"] += clip_agreement_correct
            language_count["student_label_correct"] += clip_student_label_correct
            language_count["teacher_label_correct"] += clip_teacher_label_correct
            language_count["student_clip_correct"] += int(student_clip_is_correct)
            language_count["teacher_clip_correct"] += int(teacher_clip_is_correct)
            per_clip.append(
                {
                    "id": item["id"],
                    "language": item["language"],
                    "speaker_id": item["speaker_id"],
                    "teacher_agreement": clip_agreement_correct / clip_total,
                    "student_label_accuracy": (
                        clip_student_label_correct / clip_total
                    ),
                    "teacher_label_accuracy": (
                        clip_teacher_label_correct / clip_total
                    ),
                    "student_clip_prediction": language_codes[
                        student_prediction_index
                    ],
                    "teacher_clip_prediction": language_codes[
                        teacher_prediction_index
                    ],
                    "frames": clip_total,
                }
            )
    heldout_teacher_agreement = agreement_correct / total
    heldout_student_label_accuracy = student_label_correct / total
    heldout_teacher_label_accuracy = teacher_label_correct / total
    heldout_teacher_agreement_macro = float(
        np.mean([item["teacher_agreement"] for item in per_clip])
    )
    heldout_student_label_accuracy_macro = float(
        np.mean([item["student_label_accuracy"] for item in per_clip])
    )
    heldout_teacher_label_accuracy_macro = float(
        np.mean([item["teacher_label_accuracy"] for item in per_clip])
    )
    heldout_per_language = {}
    for language, counts in language_counts.items():
        if counts["n_clips"] == 0 or counts["frames"] == 0:
            continue
        heldout_per_language[language] = {
            "n_clips": counts["n_clips"],
            "speaker_ids": sorted(
                {
                    item["speaker_id"]
                    for item in heldout
                    if item["language"] == language
                }
            ),
            "teacher_agreement": (
                counts["agreement_correct"] / counts["frames"]
            ),
            "student_label_accuracy": (
                counts["student_label_correct"] / counts["frames"]
            ),
            "teacher_label_accuracy": (
                counts["teacher_label_correct"] / counts["frames"]
            ),
            "student_clip_accuracy": (
                counts["student_clip_correct"] / counts["n_clips"]
            ),
            "teacher_clip_accuracy": (
                counts["teacher_clip_correct"] / counts["n_clips"]
            ),
        }

    switch_item = next(
        item for item in switch_records if item["id"] == "switch_hi_en_eval"
    )
    switch_waveform = load_audio(resolve_audio_path(switch_item, args.manifest))
    with torch.inference_mode():
        switch_features = frontend(switch_waveform)
        switch_logits = model.streaming_forward(
            switch_features, chunk_frames=chunk_frames
        ).squeeze(0)
        switch_probabilities = torch.softmax(switch_logits, dim=-1).numpy()
    teacher_switch = target_cache.load(
        switch_item,
        "teacher_probs",
        expected_frames=len(switch_features[0]),
    )
    availability = chunk_availability_times(
        len(switch_probabilities),
        chunk_frames=chunk_frames,
        lookahead_frames=lookahead_frames,
        hop_length=int(frontend_config["hop_length"]),
        win_length=int(frontend_config["win_length"]),
        sample_rate=int(frontend_config["sample_rate"]),
    )
    # Frames before D have no corresponding teacher target and are never trained.
    student_chunks, student_chunk_times = chunk_posteriors(
        switch_probabilities, availability, min_frame=label_delay_frames
    )
    student_smoothed = smooth_posteriors(student_chunks, new_weight=0.30)
    detected_seconds, detector_config = detect_hi_to_en_switch(
        student_chunks,
        student_chunk_times,
        language_codes=language_codes,
        chunk_ms=float(streaming_config["chunk_ms"]),
    )
    true_switch_seconds = float(switch_item["segments"][0]["end_seconds"])
    switch_lag_ms = (
        None
        if detected_seconds is None
        else 1_000 * (detected_seconds - true_switch_seconds)
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    target_time = (
        np.arange(len(teacher_switch)) * int(frontend_config["hop_length"])
        + int(frontend_config["win_length"])
    ) / int(frontend_config["sample_rate"])
    hi = language_codes.index("hi")
    en = language_codes.index("en")
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
        model,
        frontend,
        heldout_waveforms,
        repeats=args.benchmark_repeats,
        sample_rate=int(frontend_config["sample_rate"]),
        chunk_frames=chunk_frames,
    )
    finite_eval = (
        math.isfinite(heldout_teacher_agreement)
        and math.isfinite(heldout_student_label_accuracy)
        and math.isfinite(heldout_teacher_label_accuracy)
        and math.isfinite(cpu_rtf)
        and np.isfinite(switch_probabilities).all()
        and np.isfinite(teacher_switch).all()
    )
    eval_metrics = {
        "run_id": checkpoint["run_id"],
        "run_identity": run_identity,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_run_identity_validated": True,
        "launch_dependency_snapshot_captured_before_preload": train_metrics[
            "launch_dependency_snapshot_captured_before_preload"
        ],
        "dependency_snapshot_validation_checks": train_metrics[
            "dependency_snapshot_validation_checks"
        ],
        "dependency_snapshot_unchanged_at_publication": train_metrics[
            "dependency_snapshot_unchanged_at_publication"
        ],
        "heldout_teacher_agreement_micro": heldout_teacher_agreement,
        "heldout_teacher_agreement_macro": heldout_teacher_agreement_macro,
        "heldout_student_label_accuracy_micro": heldout_student_label_accuracy,
        "heldout_student_label_accuracy_macro": heldout_student_label_accuracy_macro,
        "heldout_teacher_label_accuracy_micro": heldout_teacher_label_accuracy,
        "heldout_teacher_label_accuracy_macro": heldout_teacher_label_accuracy_macro,
        "heldout_student_clip_accuracy": student_clip_correct / len(heldout),
        "heldout_teacher_clip_accuracy": teacher_clip_correct / len(heldout),
        "heldout_per_language": heldout_per_language,
        "heldout_per_clip": per_clip,
        "speaker_split": speaker_audit,
        "target_cache": target_cache.audit(),
        "switch_detected_seconds": detected_seconds,
        "true_switch_seconds": true_switch_seconds,
        "switch_lag_ms": switch_lag_ms,
        "detector": detector_config,
        "cpu_rtf": cpu_rtf,
        "cpu_rtf_runs": benchmark_rtfs,
        "cpu_threads": args.threads,
        "chunk_equivalence_checked": True,
        "provisional_tail_withheld": True,
        "withheld_tail_frames": model.lookahead_frames,
        "nan_free": bool(finite_eval),
    }
    (args.results_dir / "eval_metrics.json").write_text(
        json.dumps(eval_metrics, indent=2, allow_nan=False) + "\n"
    )
    summary = {
        "run_id": checkpoint["run_id"],
        "run_identity": run_identity,
        "run_identity_schema_version": run_identity["schema_version"],
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": run_identity["model_state_sha256"],
        "pipeline_source_sha256": run_identity["pipeline"]["source"][
            "source_sha256"
        ],
        "audio_files_sha256": run_identity["corpus"]["audio_files_sha256"],
        "target_metadata_sha256": run_identity["target_cache"][
            "metadata_sha256"
        ],
        "evaluation_run_identity_validated": True,
        "launch_dependency_snapshot_captured_before_preload": train_metrics[
            "launch_dependency_snapshot_captured_before_preload"
        ],
        "dependency_snapshot_validation_checks": train_metrics[
            "dependency_snapshot_validation_checks"
        ],
        "dependency_snapshot_unchanged_at_publication": train_metrics[
            "dependency_snapshot_unchanged_at_publication"
        ],
        "teacher_name": pipeline["teacher"]["model_id"],
        "languages": list(language_codes),
        "n_train_clips": len(train_records),
        "n_train_monolingual_clips": train_metrics["n_train_monolingual_clips"],
        "n_train_switch_clips": train_metrics["n_train_switch_clips"],
        "n_heldout_clips": len(heldout),
        "n_switch_eval_clips": len(switch_records),
        "n_train_speakers": len(speaker_audit["train_speaker_ids"]),
        "n_heldout_speakers": len(speaker_audit["heldout_speaker_ids"]),
        "speaker_disjoint": speaker_audit["speaker_disjoint"],
        "target_cache_provenance_validated": target_cache.audit()[
            "provenance_validated"
        ],
        "target_cache_schema_version": target_cache.identity["schema_version"],
        "target_configuration_sha256": target_cache.identity[
            "target_configuration_sha256"
        ],
        "manifest_records_sha256": target_cache.identity[
            "manifest_records_sha256"
        ],
        "target_files_sha256": target_cache.identity["target_files_sha256"],
        "teacher_revision": target_cache.identity["teacher_revision"],
        "teacher_artifact_sha256": target_cache.identity[
            "teacher_artifact_sha256"
        ],
        "target_generator_source_sha256": target_cache.identity[
            "target_generator_source_sha256"
        ],
        "student_params": model.parameter_count,
        "algorithmic_latency_ms": streaming_config["algorithmic_latency_ms"],
        "provisional_tail_withheld": True,
        "withheld_tail_frames": model.lookahead_frames,
        "cpu_rtf": cpu_rtf,
        "losses": train_metrics["losses"],
        "optimizer_steps": train_metrics["optimizer_steps"],
        "requested_optimizer_steps": train_metrics[
            "requested_optimizer_steps"
        ],
        "successful_optimizer_steps": train_metrics[
            "successful_optimizer_steps"
        ],
        "post_update_checks": train_metrics["post_update_checks"],
        "all_requested_steps_completed": train_metrics[
            "all_requested_steps_completed"
        ],
        "real_audio_optimizer_step": train_metrics[
            "real_audio_optimizer_step"
        ],
        "post_update_model_state_finite": train_metrics[
            "post_update_model_state_finite"
        ],
        "post_update_optimizer_state_finite": train_metrics[
            "post_update_optimizer_state_finite"
        ],
        "examples_seen": train_metrics["examples_seen"],
        "effective_epochs": train_metrics["effective_epochs"],
        "first_10_mean_loss": train_metrics["first_10_mean_loss"],
        "last_10_mean_loss": train_metrics["last_10_mean_loss"],
        "loss_decreased": train_metrics["loss_decreased"],
        "heldout_teacher_agreement_micro": heldout_teacher_agreement,
        "heldout_teacher_agreement_macro": heldout_teacher_agreement_macro,
        "heldout_student_label_accuracy_micro": heldout_student_label_accuracy,
        "heldout_student_label_accuracy_macro": heldout_student_label_accuracy_macro,
        "heldout_teacher_label_accuracy_micro": heldout_teacher_label_accuracy,
        "heldout_teacher_label_accuracy_macro": heldout_teacher_label_accuracy_macro,
        "heldout_student_clip_accuracy": student_clip_correct / len(heldout),
        "heldout_teacher_clip_accuracy": teacher_clip_correct / len(heldout),
        "heldout_per_language": heldout_per_language,
        "switch_lag_ms": switch_lag_ms,
        "nan_free": bool(train_metrics["nan_free"] and finite_eval),
    }
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"heldout teacher agreement={heldout_teacher_agreement:.3f}; "
        f"student label accuracy={heldout_student_label_accuracy:.3f}; "
        f"teacher label accuracy={heldout_teacher_label_accuracy:.3f}; "
        f"CPU RTF={cpu_rtf:.4f}; "
        f"switch lag={switch_lag_ms if switch_lag_ms is not None else 'not detected'} ms; "
        f"nan_free={summary['nan_free']}",
        flush=True,
    )
    print(
        f"wrote {args.results_dir / 'summary.json'} and {args.results_dir / 'switch_plot.png'}"
    )


if __name__ == "__main__":
    main()
