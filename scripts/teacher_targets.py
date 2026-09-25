#!/usr/bin/env python3
"""Run the frozen offline ECAPA teacher on local full-context windows."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.classifiers import EncoderClassifier

from streaming_lid.audio import feature_frame_count, load_audio
from streaming_lid.config import (
    HOP_LENGTH,
    LANGUAGE_CODES,
    SAMPLE_RATE,
    TEACHER_FUTURE_MS,
    TEACHER_HOP_FRAMES,
    TEACHER_NAME,
    TEACHER_PAST_MS,
    TEACHER_TEMPERATURE,
    WIN_LENGTH,
)
from streaming_lid.data import read_manifest, require_speaker_disjoint, resolve_audio_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/generated/manifest.jsonl")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/generated/targets")
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".cache/models/lang-id-voxlingua107-ecapa"),
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


def extract_window(waveform: torch.Tensor, anchor_frame: int) -> torch.Tensor:
    """Return [anchor-1.75 s, anchor+0.25 s], padding clip edges."""
    past_samples = round(TEACHER_PAST_MS * SAMPLE_RATE / 1_000)
    future_samples = round(TEACHER_FUTURE_MS * SAMPLE_RATE / 1_000)
    window_samples = past_samples + future_samples
    anchor_sample = anchor_frame * HOP_LENGTH + WIN_LENGTH
    source_start = anchor_sample - past_samples
    source_end = anchor_sample + future_samples
    output = waveform.new_zeros(window_samples)
    clipped_start = max(0, source_start)
    clipped_end = min(len(waveform), source_end)
    if clipped_end > clipped_start:
        destination_start = clipped_start - source_start
        output[destination_start : destination_start + clipped_end - clipped_start] = (
            waveform[clipped_start:clipped_end]
        )
    return output


def label_indices(teacher: EncoderClassifier) -> list[int]:
    index_to_label = teacher.hparams.label_encoder.ind2lab
    result = []
    for code in LANGUAGE_CODES:
        matches = [
            index
            for index, label in index_to_label.items()
            if label.startswith(f"{code}:")
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one teacher label for {code}, found {matches}")
        result.append(matches[0])
    return result


def interpolate_posteriors(
    anchor_frames: np.ndarray, anchor_values: np.ndarray, num_frames: int
) -> np.ndarray:
    frame_index = np.arange(num_frames)
    interpolated = np.stack(
        [
            np.interp(frame_index, anchor_frames, anchor_values[:, column])
            for column in range(anchor_values.shape[1])
        ],
        axis=-1,
    ).astype(np.float32)
    interpolated /= np.clip(interpolated.sum(axis=-1, keepdims=True), 1e-8, None)
    return interpolated


def expected_language(item: dict, time_seconds: float) -> str:
    for segment in item["segments"]:
        if segment["start_seconds"] <= time_seconds < segment["end_seconds"]:
            return segment["language"]
    return item["segments"][-1]["language"]


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", message=".*custom_fwd.*deprecated.*")
    print(f"loading frozen teacher {TEACHER_NAME}", flush=True)
    teacher = EncoderClassifier.from_hparams(
        source=TEACHER_NAME,
        savedir=str(args.model_dir),
        run_opts={"device": "cpu"},
    )
    teacher.eval()
    teacher.hparams.label_encoder.ignore_len()
    selected_indices = label_indices(teacher)
    records = read_manifest(args.manifest)
    speaker_audit = require_speaker_disjoint(records)
    summary_records = []

    for clip_number, item in enumerate(records, start=1):
        waveform = load_audio(resolve_audio_path(item, args.manifest))
        num_frames = feature_frame_count(len(waveform))
        if item["language"] == "mixed":
            target_kind = "local_windows"
            anchor_frames = np.arange(0, num_frames, TEACHER_HOP_FRAMES, dtype=np.int64)
            if anchor_frames[-1] != num_frames - 1:
                anchor_frames = np.append(anchor_frames, num_frames - 1)
            windows = torch.stack(
                [extract_window(waveform, int(frame)) for frame in anchor_frames]
            )
        else:
            # A converged utterance posterior is a lower-variance target when the
            # language is stationary. It is deliberately never used for a switch.
            target_kind = "converged_utterance"
            anchor_frames = np.asarray([num_frames // 2], dtype=np.int64)
            windows = waveform.unsqueeze(0)
        selected_log_probabilities = []
        in_set_masses = []
        with torch.inference_mode():
            for start in range(0, len(windows), args.batch_size):
                log_probabilities, _, _, _ = teacher.classify_batch(
                    windows[start : start + args.batch_size]
                )
                chosen = log_probabilities[:, selected_indices]
                selected_log_probabilities.append(chosen.cpu())
                in_set_masses.append(chosen.exp().sum(dim=-1).cpu())
        selected_logs = torch.cat(selected_log_probabilities)
        raw_anchors = torch.softmax(selected_logs, dim=-1).numpy()
        soft_anchors = torch.softmax(
            selected_logs / TEACHER_TEMPERATURE, dim=-1
        ).numpy()
        raw = interpolate_posteriors(anchor_frames, raw_anchors, num_frames)
        soft = interpolate_posteriors(anchor_frames, soft_anchors, num_frames)
        in_set_mass = torch.cat(in_set_masses).numpy()
        output_path = args.output_dir / f"{item['id']}.npz"
        np.savez_compressed(
            output_path,
            teacher_probs=raw,
            teacher_soft_targets=soft,
            anchor_frames=anchor_frames,
            anchor_probs=raw_anchors.astype(np.float32),
            in_set_mass=in_set_mass.astype(np.float32),
            language_codes=np.asarray(LANGUAGE_CODES),
        )

        predicted = raw_anchors.argmax(axis=-1)
        expected = [
            LANGUAGE_CODES.index(
                expected_language(
                    item, (int(frame) * HOP_LENGTH + WIN_LENGTH) / SAMPLE_RATE
                )
            )
            for frame in anchor_frames
        ]
        anchor_accuracy = float(np.mean(predicted == np.asarray(expected)))
        summary_records.append(
            {
                "id": item["id"],
                "frames": num_frames,
                "anchors": len(anchor_frames),
                "target_kind": target_kind,
                "anchor_label_accuracy": anchor_accuracy,
                "mean_selected_language_mass": float(np.mean(in_set_mass)),
            }
        )
        print(
            f"[{clip_number:02d}/{len(records)}] {item['id']}: "
            f"{len(anchor_frames)} {target_kind}, label agreement={anchor_accuracy:.3f}",
            flush=True,
        )

    metadata = {
        "teacher": TEACHER_NAME,
        "languages": list(LANGUAGE_CODES),
        "temperature": TEACHER_TEMPERATURE,
        "window_past_ms": TEACHER_PAST_MS,
        "window_future_ms": TEACHER_FUTURE_MS,
        "target_hop_frames": TEACHER_HOP_FRAMES,
        "clips": summary_records,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    mean_accuracy = np.mean(
        [record["anchor_label_accuracy"] for record in summary_records]
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    teacher_metrics = {
        "teacher_name": TEACHER_NAME,
        "languages": list(LANGUAGE_CODES),
        "n_clips": len(records),
        "n_converged_utterance_targets": sum(
            record["target_kind"] == "converged_utterance" for record in summary_records
        ),
        "n_local_window_target_clips": sum(
            record["target_kind"] == "local_windows" for record in summary_records
        ),
        "mean_anchor_label_agreement": float(mean_accuracy),
        "temperature": TEACHER_TEMPERATURE,
        "window_past_ms": TEACHER_PAST_MS,
        "window_future_ms": TEACHER_FUTURE_MS,
        "speaker_split": speaker_audit,
    }
    (args.results_dir / "teacher_metrics.json").write_text(
        json.dumps(teacher_metrics, indent=2) + "\n"
    )
    print(
        f"wrote {len(records)} target files; mean anchor label agreement={mean_accuracy:.3f}"
    )


if __name__ == "__main__":
    main()
