#!/usr/bin/env python3
"""Run the frozen offline ECAPA teacher on local full-context windows."""

from __future__ import annotations

import argparse
import hashlib
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
    TEACHER_ARTIFACT_FILES,
    TEACHER_ARTIFACT_SHA256,
    TEACHER_LANGUAGE_INDICES,
    TEACHER_NAME,
    TEACHER_PAST_MS,
    TEACHER_REVISION,
    TEACHER_TEMPERATURE,
    TEACHER_TARGET_EXPANSION,
    WIN_LENGTH,
)
from streaming_lid.data import (
    DENSE_TARGET_VALIDATION_ATOL,
    DENSE_TARGET_VALIDATION_RTOL,
    TARGET_CACHE_SCHEMA_VERSION,
    TeacherTargetCache,
    canonical_json_sha256,
    capture_manifest_snapshot,
    expand_local_posteriors,
    file_sha256,
    local_target_availability_ledger,
    manifest_record_sha256,
    require_speaker_disjoint,
    resolve_audio_path,
    target_cache_configuration,
    target_kind_for_item,
    teacher_artifact_identity,
)


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
    if tuple(result) != TEACHER_LANGUAGE_INDICES:
        raise ValueError(
            "pinned teacher label map differs from configured language indices: "
            f"found {tuple(result)}, expected {TEACHER_LANGUAGE_INDICES}"
        )
    return result


def resolve_teacher_artifact() -> tuple[Path, dict]:
    """Download/resolve the pinned snapshot and verify every consumed artifact."""
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=TEACHER_NAME,
            revision=TEACHER_REVISION,
            allow_patterns=list(TEACHER_ARTIFACT_FILES),
        )
    ).resolve()
    file_records = {}
    combined = hashlib.sha256()
    for filename in TEACHER_ARTIFACT_FILES:
        path = snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(f"pinned teacher artifact is missing {path}")
        sha256 = file_sha256(path)
        size = path.stat().st_size
        file_records[filename] = {"sha256": sha256, "bytes": size}
        combined.update(filename.encode("utf-8"))
        combined.update(bytes.fromhex(sha256))
    actual = {
        "model_id": TEACHER_NAME,
        "revision": TEACHER_REVISION,
        "artifact_sha256": combined.hexdigest(),
        "files": file_records,
    }
    expected = teacher_artifact_identity()
    if actual != expected or actual["artifact_sha256"] != TEACHER_ARTIFACT_SHA256:
        raise ValueError(
            "resolved teacher artifacts differ from the pinned identity; "
            f"found {actual}, expected {expected}"
        )
    return snapshot, actual


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
    teacher_snapshot, resolved_teacher_identity = resolve_teacher_artifact()
    pinned_savedir = (
        args.model_dir.resolve()
        / f"{TEACHER_REVISION}-{TEACHER_ARTIFACT_SHA256[:12]}"
    )
    print(
        f"loading frozen teacher {TEACHER_NAME}@{TEACHER_REVISION[:8]}", flush=True
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
    manifest_snapshot = capture_manifest_snapshot(args.manifest)
    records = manifest_snapshot.records_copy()
    speaker_audit = require_speaker_disjoint(records)
    target_configuration = target_cache_configuration()
    target_configuration_hash = canonical_json_sha256(target_configuration)
    generator_identity = target_configuration["target_generator"]
    manifest_identity = manifest_snapshot.identity()
    summary_records = []

    for clip_number, item in enumerate(records, start=1):
        waveform = load_audio(resolve_audio_path(item, manifest_snapshot))
        num_frames = feature_frame_count(len(waveform))
        target_kind = target_kind_for_item(item)
        if target_kind == "local_windows":
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
        availability_ledger = None
        if target_kind == "local_windows":
            raw = expand_local_posteriors(
                anchor_frames,
                raw_anchors,
                num_frames,
                expansion=TEACHER_TARGET_EXPANSION,
            )
            soft = expand_local_posteriors(
                anchor_frames,
                soft_anchors,
                num_frames,
                expansion=TEACHER_TARGET_EXPANSION,
            )
            availability_ledger = local_target_availability_ledger(
                anchor_frames,
                num_frames,
                expansion=TEACHER_TARGET_EXPANSION,
            )
            invalid = np.flatnonzero(~availability_ledger["availability_valid"])
            if len(invalid):
                frame = int(invalid[0])
                raise ValueError(
                    f"target availability violation for {item['id']} frame {frame}: "
                    f"teacher latest sample "
                    f"{availability_ledger['teacher_latest_samples'][frame]} exceeds "
                    f"student latest sample "
                    f"{availability_ledger['student_latest_samples'][frame]}"
                )
        else:
            raw = np.repeat(raw_anchors.astype(np.float32), num_frames, axis=0)
            soft = np.repeat(soft_anchors.astype(np.float32), num_frames, axis=0)
        in_set_mass = torch.cat(in_set_masses).numpy()
        output_path = args.output_dir / f"{item['id']}.npz"
        audio_path = resolve_audio_path(item, manifest_snapshot)
        audio_hash = file_sha256(audio_path)
        record_hash = manifest_record_sha256(item)
        target_payload = {
            "teacher_probs": raw,
            "teacher_soft_targets": soft,
            "anchor_frames": anchor_frames,
            "anchor_probs": raw_anchors.astype(np.float32),
            "anchor_soft_targets": soft_anchors.astype(np.float32),
            "in_set_mass": in_set_mass.astype(np.float32),
            "language_codes": np.asarray(LANGUAGE_CODES),
            "cache_schema_version": np.asarray(TARGET_CACHE_SCHEMA_VERSION),
            "clip_id": np.asarray(item["id"]),
            "target_kind": np.asarray(target_kind),
            "num_frames": np.asarray(num_frames),
            "audio_sha256": np.asarray(audio_hash),
            "manifest_record_sha256": np.asarray(record_hash),
            "manifest_file_sha256": np.asarray(
                manifest_identity["manifest_file_sha256"]
            ),
            "target_configuration_sha256": np.asarray(target_configuration_hash),
            "teacher_name": np.asarray(TEACHER_NAME),
            "teacher_revision": np.asarray(TEACHER_REVISION),
            "teacher_artifact_sha256": np.asarray(TEACHER_ARTIFACT_SHA256),
            "target_generator_source_sha256": np.asarray(
                generator_identity["source_sha256"]
            ),
        }
        if availability_ledger is not None:
            target_payload.update(availability_ledger)
        np.savez_compressed(
            output_path,
            **target_payload,
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
                "target_expansion": (
                    TEACHER_TARGET_EXPANSION
                    if availability_ledger is not None
                    else "constant_utterance_repeat"
                ),
                "availability_checked_frames": (
                    num_frames if availability_ledger is not None else 0
                ),
                "availability_contract_valid": (
                    bool(availability_ledger["availability_valid"].all())
                    if availability_ledger is not None
                    else None
                ),
                "dense_target_checked_frames": num_frames,
                "dense_target_expansion_valid": True,
                "minimum_availability_margin_samples": (
                    int(
                        np.min(
                            availability_ledger["student_latest_samples"]
                            - availability_ledger["teacher_latest_samples"]
                        )
                    )
                    if availability_ledger is not None
                    else None
                ),
                "anchor_label_accuracy": anchor_accuracy,
                "mean_selected_language_mass": float(np.mean(in_set_mass)),
                "audio_sha256": audio_hash,
                "manifest_record_sha256": record_hash,
                "target_file_sha256": file_sha256(output_path),
            }
        )
        print(
            f"[{clip_number:02d}/{len(records)}] {item['id']}: "
            f"{len(anchor_frames)} {target_kind}, label agreement={anchor_accuracy:.3f}",
            flush=True,
        )

    metadata = {
        "schema_version": TARGET_CACHE_SCHEMA_VERSION,
        "teacher": TEACHER_NAME,
        "teacher_identity": resolved_teacher_identity,
        "target_generator": generator_identity,
        "languages": list(LANGUAGE_CODES),
        "temperature": TEACHER_TEMPERATURE,
        "window_past_ms": TEACHER_PAST_MS,
        "window_future_ms": TEACHER_FUTURE_MS,
        "target_hop_frames": TEACHER_HOP_FRAMES,
        "target_expansion": TEACHER_TARGET_EXPANSION,
        "availability_sample_index_semantics": "exclusive_right_edge_unclipped",
        "availability_checked_frames": sum(
            record["availability_checked_frames"] for record in summary_records
        ),
        "availability_contract_valid": all(
            record["availability_contract_valid"] is not False
            for record in summary_records
        ),
        "dense_target_checked_frames": sum(
            record["dense_target_checked_frames"] for record in summary_records
        ),
        "dense_target_expansion_valid": all(
            record["dense_target_expansion_valid"] for record in summary_records
        ),
        "dense_target_validation_rtol": DENSE_TARGET_VALIDATION_RTOL,
        "dense_target_validation_atol": DENSE_TARGET_VALIDATION_ATOL,
        "target_configuration": target_configuration,
        "target_configuration_sha256": target_configuration_hash,
        "manifest_snapshot": manifest_identity,
        "manifest_file_sha256": manifest_identity["manifest_file_sha256"],
        "manifest_records_sha256": manifest_identity[
            "manifest_records_sha256"
        ],
        "target_files_sha256": canonical_json_sha256(
            {
                record["id"]: record["target_file_sha256"]
                for record in sorted(summary_records, key=lambda value: value["id"])
            }
        ),
        "clips": summary_records,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    cache_audit = TeacherTargetCache(
        manifest_snapshot, args.output_dir
    ).validate_all()
    mean_accuracy = np.mean(
        [record["anchor_label_accuracy"] for record in summary_records]
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    teacher_metrics = {
        "teacher_name": TEACHER_NAME,
        "teacher_identity": resolved_teacher_identity,
        "target_generator": generator_identity,
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
        "target_expansion": TEACHER_TARGET_EXPANSION,
        "availability_sample_index_semantics": "exclusive_right_edge_unclipped",
        "availability_checked_frames": metadata["availability_checked_frames"],
        "availability_contract_valid": metadata["availability_contract_valid"],
        "dense_target_checked_frames": metadata["dense_target_checked_frames"],
        "dense_target_expansion_valid": metadata[
            "dense_target_expansion_valid"
        ],
        "dense_target_validation_rtol": DENSE_TARGET_VALIDATION_RTOL,
        "dense_target_validation_atol": DENSE_TARGET_VALIDATION_ATOL,
        "speaker_split": speaker_audit,
        "target_cache": cache_audit,
    }
    (args.results_dir / "teacher_metrics.json").write_text(
        json.dumps(teacher_metrics, indent=2) + "\n"
    )
    print(
        f"wrote {len(records)} target files; mean anchor label agreement={mean_accuracy:.3f}"
    )


if __name__ == "__main__":
    main()
