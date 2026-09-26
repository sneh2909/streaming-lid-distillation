#!/usr/bin/env python3
"""Rank the frozen checkpoint shortlist on pinned external validation.

This experiment consumes the trajectory and shortlist produced by
``experiments/checkpoint-trajectory/run.py``.  It never trains or edits the
main pipeline.  It scores only the already-frozen shortlist on:

* a deterministic, language-balanced sample of FLEURS *validation*; and
* the exact 21 synthetic held-out clips used by the main pipeline.

FLEURS test rows are never requested.  Prefix metrics exclude clips shorter
than the requested duration instead of padding them with silence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tarfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# Reuse the main pipeline's audio, frontend, architecture, configuration,
# artifact validation, and manifest resolution.  This experiment does not
# modify those modules.
from scripts.teacher_targets import label_indices, resolve_teacher_artifact  # noqa: E402
from streaming_lid.audio import (  # noqa: E402
    LogMelFrontend,
    feature_frame_count,
    load_audio,
)
from streaming_lid.config import (  # noqa: E402
    CHUNK_FRAMES,
    LABEL_DELAY_FRAMES,
    LANGUAGE_CODES,
    MODEL_LOOKAHEAD_FRAMES,
    SAMPLE_RATE,
    TEACHER_ARTIFACT_SHA256,
    TEACHER_NAME,
    TEACHER_REVISION,
)
from streaming_lid.data import (  # noqa: E402
    file_sha256,
    manifest_records_sha256,
    read_manifest,
    require_speaker_disjoint,
    resolve_audio_path,
)
from streaming_lid.model import CausalLIDStudent  # noqa: E402
from streaming_lid.run_identity import (  # noqa: E402
    configured_model_kwargs,
    model_state_sha256,
)


EXPERIMENT_NAME = "checkpoint-external-validation"
SCHEMA_VERSION = 1
ANALYSIS_VERSION = "checkpoint-external-validation-v1"

FLEURS_DATASET_ID = "google/fleurs"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_LICENSE = "cc-by-4.0"
FLEURS_SPLIT = "validation"
FLEURS_CONFIG_BY_LANGUAGE = {
    "en": "en_us",
    "hi": "hi_in",
    "mr": "mr_in",
    "bn": "bn_in",
    "ta": "ta_in",
    "te": "te_in",
    "gu": "gu_in",
}
EXPECTED_VALIDATION_ROWS = {
    "en_us": 394,
    "hi_in": 239,
    "mr_in": 443,
    "bn_in": 402,
    "ta_in": 377,
    "te_in": 311,
    "gu_in": 432,
}
SAMPLE_PER_LANGUAGE = 100
SELECTION_ALGORITHM = "lowest_sha256(revision\\0config\\0row_idx\\0id\\0basename)"

PREFIX_SECONDS: tuple[float | str, ...] = (0.5, 1.0, 2.0, 4.0, "full")
SELECTION_PREFIX_KEYS = ("1", "2", "full")
STUDENT_BATCH_SIZE = 24
TEACHER_BATCH_SIZE = 16
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 7
EXTERNAL_LABEL_GAIN_GATE_PP = 2.0
LABEL_ELIGIBILITY_TOLERANCE = 0.02
WORST_LANGUAGE_TOLERANCE = 0.05

SOURCE_FILES = (
    "src/streaming_lid/audio.py",
    "src/streaming_lid/config.py",
    "src/streaming_lid/data.py",
    "src/streaming_lid/model.py",
    "src/streaming_lid/run_identity.py",
    "scripts/teacher_targets.py",
    "experiments/checkpoint-external-validation/run.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("data/experiments/checkpoint-trajectory/trajectory.pt"),
    )
    parser.add_argument(
        "--trajectory-results",
        type=Path,
        default=Path("experiments/checkpoint-trajectory/results.json"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/generated/manifest.jsonl")
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "data/experiments/checkpoint-external-validation/fleurs-validation"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/checkpoint-external-validation/results.json"),
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Atomically replace an existing result after every validation passes.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.threads != 6:
        raise ValueError("threads must be 6 under the shared CPU contract")
    if args.output.exists() and not args.fresh:
        raise FileExistsError(f"{args.output} exists; pass --fresh to replace it")
    for path in (args.trajectory, args.trajectory_results, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_json_write(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_snapshot() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        result[relative] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return result


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "numpy",
        "torch",
        "torchaudio",
        "soundfile",
        "speechbrain",
        "huggingface-hub",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def parse_config_rows(config: str, tsv_path: Path) -> list[dict[str, Any]]:
    expected = EXPECTED_VALIDATION_ROWS[config]
    rows: list[dict[str, Any]] = []
    gender_to_index = {"MALE": 0, "FEMALE": 1, "OTHER": 2}
    for row_idx, line in enumerate(tsv_path.read_text(encoding="utf-8").splitlines()):
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(
                f"unexpected {config} dev.tsv field count at row {row_idx}: {len(fields)}"
            )
        basename = fields[1]
        if not basename.endswith(".wav") or Path(basename).name != basename:
            raise ValueError(f"unexpected FLEURS basename {basename}")
        gender = fields[6]
        if gender not in gender_to_index:
            raise ValueError(f"unexpected FLEURS gender {gender}")
        rows.append(
            {
                "config": config,
                "row_idx": row_idx,
                "dataset_row_id": int(fields[0]),
                "num_samples": int(fields[5]),
                "source_basename": basename,
                "gender_index": gender_to_index[gender],
                "language_name": config,
            }
        )
    if len(rows) != expected:
        raise AssertionError(f"fetched {len(rows)} rows for {config}, expected {expected}")
    if sorted(row["row_idx"] for row in rows) != list(range(expected)):
        raise AssertionError(f"{config} row indices are not exactly 0..{expected - 1}")
    return rows


def selection_rank(row: Mapping[str, Any]) -> str:
    payload = "\0".join(
        (
            FLEURS_REVISION,
            str(row["config"]),
            str(row["row_idx"]),
            str(row["dataset_row_id"]),
            str(row["source_basename"]),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_audio_file(path: Path, expected_samples: int) -> dict[str, Any]:
    info = sf.info(str(path))
    if info.samplerate != SAMPLE_RATE or info.channels != 1:
        raise ValueError(
            f"{path} is {info.samplerate} Hz/{info.channels} channels, expected 16 kHz mono"
        )
    if info.frames != expected_samples:
        raise ValueError(
            f"{path} has {info.frames} samples, expected {expected_samples}"
        )
    return {
        "audio_sha256": file_sha256(path),
        "audio_bytes": path.stat().st_size,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "subtype": info.subtype,
    }


def store_audio_payload(
    row: Mapping[str, Any], payload: bytes, data_dir: Path
) -> dict[str, Any]:
    language = next(
        code
        for code, config in FLEURS_CONFIG_BY_LANGUAGE.items()
        if config == row["config"]
    )
    relative = Path("audio") / str(row["config"]) / (
        f"{int(row['row_idx']):04d}-{int(row['dataset_row_id'])}.wav"
    )
    destination = data_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError(f"non-WAV archive member for {row['config']}/{row['row_idx']}")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    audit = validate_audio_file(destination, int(row["num_samples"]))
    return {
        "clip_id": (
            f"fleurs-{row['config']}-validation-{int(row['row_idx']):04d}-"
            f"{int(row['dataset_row_id'])}"
        ),
        "language": language,
        "config": row["config"],
        "row_idx": int(row["row_idx"]),
        "dataset_row_id": int(row["dataset_row_id"]),
        "source_basename": row["source_basename"],
        "gender_index": int(row["gender_index"]),
        "language_name": row["language_name"],
        "num_samples": int(row["num_samples"]),
        "duration_seconds": int(row["num_samples"]) / SAMPLE_RATE,
        "audio_path": relative.as_posix(),
        **audit,
    }


def extract_selected_audio(
    config: str,
    rows: Sequence[Mapping[str, Any]],
    archive_path: Path,
    data_dir: Path,
) -> list[dict[str, Any]]:
    remaining = {str(row["source_basename"]): row for row in rows}
    if len(remaining) != len(rows):
        raise ValueError(f"selected {config} basenames are not unique")
    records: list[dict[str, Any]] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            basename = Path(member.name).name
            if basename not in remaining or not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"could not read archive member {member.name}")
            records.append(store_audio_payload(remaining.pop(basename), stream.read(), data_dir))
            if not remaining:
                break
    if remaining:
        missing = sorted(remaining)[:5]
        raise ValueError(f"{config} archive lacks {len(remaining)} selected files: {missing}")
    return records


def validate_cached_external_manifest(
    manifest: Mapping[str, Any], data_dir: Path
) -> dict[str, Any]:
    identity = manifest["identity"]
    expected_identity = {
        "schema_version": 1,
        "dataset_id": FLEURS_DATASET_ID,
        "revision": FLEURS_REVISION,
        "split": FLEURS_SPLIT,
        "license": FLEURS_LICENSE,
        "configs_by_language": FLEURS_CONFIG_BY_LANGUAGE,
        "expected_pool_rows": EXPECTED_VALIDATION_ROWS,
        "sample_per_language": SAMPLE_PER_LANGUAGE,
        "selection_algorithm": SELECTION_ALGORITHM,
        "transport": "pinned Hub data/<config>/dev.tsv plus audio/dev.tar.gz",
    }
    if identity != expected_identity:
        raise ValueError("cached external manifest identity differs from frozen config")
    records = manifest["records"]
    if len(records) != SAMPLE_PER_LANGUAGE * len(LANGUAGE_CODES):
        raise ValueError("cached external manifest has the wrong row count")
    counts = Counter(record["language"] for record in records)
    if counts != Counter({language: SAMPLE_PER_LANGUAGE for language in LANGUAGE_CODES}):
        raise ValueError(f"cached external language counts differ: {counts}")
    expected_hash = manifest["manifest_sha256"]
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_sha256(payload) != expected_hash:
        raise ValueError("cached external manifest canonical hash differs")
    for record in records:
        path = data_dir / record["audio_path"]
        audit = validate_audio_file(path, int(record["num_samples"]))
        for key in ("audio_sha256", "audio_bytes", "sample_rate", "channels", "frames", "subtype"):
            if audit[key] != record[key]:
                raise ValueError(f"cached audio audit differs for {record['clip_id']}: {key}")
    return dict(manifest)


def prepare_external_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    if manifest_path.is_file():
        print(f"validating cached external manifest {manifest_path}", flush=True)
        return validate_cached_external_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")), data_dir
        )

    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(
        FLEURS_DATASET_ID, revision=FLEURS_REVISION, files_metadata=False
    )
    if info.sha != FLEURS_REVISION:
        raise ValueError(f"resolved FLEURS revision {info.sha} != {FLEURS_REVISION}")
    licenses = info.card_data.get("license") if info.card_data else None
    if isinstance(licenses, str):
        licenses = [licenses]
    if FLEURS_LICENSE not in (licenses or []):
        raise ValueError(f"FLEURS license differs from {FLEURS_LICENSE}: {licenses}")

    stable_pool: list[dict[str, Any]] = []
    downloaded: list[dict[str, Any]] = []
    source_artifacts: dict[str, Any] = {}
    for language in LANGUAGE_CODES:
        config = FLEURS_CONFIG_BY_LANGUAGE[language]
        print(f"resolving pinned FLEURS dev.tsv for {config}", flush=True)
        tsv_path = Path(
            hf_hub_download(
                FLEURS_DATASET_ID,
                f"data/{config}/dev.tsv",
                repo_type="dataset",
                revision=FLEURS_REVISION,
            )
        )
        pool = parse_config_rows(config, tsv_path)
        stable_pool.extend(pool)
        chosen = sorted(pool, key=selection_rank)[:SAMPLE_PER_LANGUAGE]
        print(f"resolving pinned FLEURS dev audio archive for {config}", flush=True)
        archive_path = Path(
            hf_hub_download(
                FLEURS_DATASET_ID,
                f"data/{config}/audio/dev.tar.gz",
                repo_type="dataset",
                revision=FLEURS_REVISION,
            )
        )
        source_artifacts[config] = {
            "tsv_repo_path": f"data/{config}/dev.tsv",
            "tsv_sha256": file_sha256(tsv_path),
            "tsv_bytes": tsv_path.stat().st_size,
            "audio_archive_repo_path": f"data/{config}/audio/dev.tar.gz",
            "audio_archive_sha256": file_sha256(archive_path),
            "audio_archive_bytes": archive_path.stat().st_size,
        }
        downloaded.extend(
            extract_selected_audio(config, chosen, archive_path, data_dir)
        )
        print(f"materialized {len(chosen)} selected {config} clips", flush=True)

    downloaded.sort(key=lambda row: (LANGUAGE_CODES.index(row["language"]), row["row_idx"]))
    identity = {
        "schema_version": 1,
        "dataset_id": FLEURS_DATASET_ID,
        "revision": FLEURS_REVISION,
        "split": FLEURS_SPLIT,
        "license": FLEURS_LICENSE,
        "configs_by_language": FLEURS_CONFIG_BY_LANGUAGE,
        "expected_pool_rows": EXPECTED_VALIDATION_ROWS,
        "sample_per_language": SAMPLE_PER_LANGUAGE,
        "selection_algorithm": SELECTION_ALGORITHM,
        "transport": "pinned Hub data/<config>/dev.tsv plus audio/dev.tar.gz",
    }
    payload = {
        "identity": identity,
        "pool_metadata_sha256": canonical_sha256(stable_pool),
        "pool_rows": sum(EXPECTED_VALIDATION_ROWS.values()),
        "source_artifacts": source_artifacts,
        "records": downloaded,
    }
    manifest = {**payload, "manifest_sha256": canonical_sha256(payload)}
    atomic_json_write(manifest, manifest_path)
    return validate_cached_external_manifest(manifest, data_dir)


def prefix_key(value: float | str) -> str:
    if value == "full":
        return "full"
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def load_external_examples(
    manifest: Mapping[str, Any], data_dir: Path
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in manifest["records"]:
        waveform = load_audio(data_dir / record["audio_path"])
        if len(waveform) != int(record["num_samples"]):
            raise ValueError(f"decoded length differs for {record['clip_id']}")
        examples.append(
            {
                "id": record["clip_id"],
                "language": record["language"],
                "num_samples": len(waveform),
                "waveform": waveform,
            }
        )
    return examples


def load_synthetic_examples(manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = read_manifest(manifest_path)
    speaker_audit = require_speaker_disjoint(records)
    heldout = [record for record in records if record["split"] == "heldout"]
    if len(heldout) != 21:
        raise ValueError(f"expected the main pipeline's 21 held-out clips, got {len(heldout)}")
    examples: list[dict[str, Any]] = []
    audio_hashes: dict[str, str] = {}
    for record in heldout:
        path = resolve_audio_path(record, manifest_path)
        if file_sha256(path) != record["audio_sha256"]:
            raise ValueError(f"manifest audio hash differs for {record['id']}")
        waveform = load_audio(path)
        examples.append(
            {
                "id": record["id"],
                "language": record["language"],
                "num_samples": len(waveform),
                "waveform": waveform,
            }
        )
        audio_hashes[record["id"]] = record["audio_sha256"]
    return examples, {
        "complete_manifest_records_sha256": manifest_records_sha256(records),
        "heldout_records_sha256": canonical_sha256(heldout),
        "heldout_audio_sha256_by_clip": audio_hashes,
        "heldout_input_sha256": canonical_sha256(
            {"records": heldout, "audio_sha256_by_clip": audio_hashes}
        ),
        "speaker_audit": speaker_audit,
    }


def eligible_indices(
    examples: Sequence[Mapping[str, Any]], prefix: float | str
) -> list[int]:
    if prefix == "full":
        return list(range(len(examples)))
    samples = round(float(prefix) * SAMPLE_RATE)
    return [
        index
        for index, example in enumerate(examples)
        if int(example["num_samples"]) >= samples
    ]


def classification_summary(
    expected: Sequence[str], predicted: Sequence[str], max_probabilities: Sequence[float]
) -> dict[str, Any]:
    if not (len(expected) == len(predicted) == len(max_probabilities)) or not expected:
        raise ValueError("classification summary needs equal non-empty inputs")
    support = Counter(expected)
    recalls: dict[str, float] = {}
    f1s: dict[str, float] = {}
    confusion: dict[str, Counter[str]] = {
        language: Counter() for language in LANGUAGE_CODES
    }
    for truth, guess in zip(expected, predicted, strict=True):
        confusion[truth][guess] += 1
    for language in LANGUAGE_CODES:
        if not support[language]:
            raise ValueError(f"classification slice has no {language} examples")
        true_positive = confusion[language][language]
        false_negative = support[language] - true_positive
        false_positive = sum(
            confusion[other][language]
            for other in LANGUAGE_CODES
            if other != language
        )
        recalls[language] = true_positive / support[language]
        denominator = 2 * true_positive + false_positive + false_negative
        f1s[language] = 0.0 if denominator == 0 else 2 * true_positive / denominator
    correct = sum(truth == guess for truth, guess in zip(expected, predicted, strict=True))
    return {
        "n_clips": len(expected),
        "correct": int(correct),
        "accuracy": correct / len(expected),
        "language_macro_accuracy": float(np.mean(list(recalls.values()))),
        "macro_f1": float(np.mean(list(f1s.values()))),
        "minimum_language_recall": min(recalls.values()),
        "per_language_recall": recalls,
        "per_language_f1": f1s,
        "support_per_language": {language: support[language] for language in LANGUAGE_CODES},
        "confusion": {
            language: dict(sorted(confusion[language].items()))
            for language in LANGUAGE_CODES
        },
        "mean_max_probability": float(np.mean(max_probabilities)),
    }


def summarize_predictions(
    examples: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for prefix in PREFIX_SECONDS:
        key = prefix_key(prefix)
        indices = eligible_indices(examples, prefix)
        expected = [str(examples[index]["language"]) for index in indices]
        guessed = [str(predictions[key]["prediction"][index]) for index in indices]
        maximums = [float(predictions[key]["max_probability"][index]) for index in indices]
        summary = classification_summary(expected, guessed, maximums)
        summary["requested_prefix_seconds"] = prefix
        summary["short_clip_policy"] = "exclude; never zero-pad"
        summary["excluded_short_clips"] = len(examples) - len(indices)
        summaries[key] = summary
    composite = float(
        np.mean(
            [summaries[key]["language_macro_accuracy"] for key in SELECTION_PREFIX_KEYS]
        )
    )
    per_language_composite = {
        language: float(
            np.mean(
                [
                    summaries[key]["per_language_recall"][language]
                    for key in SELECTION_PREFIX_KEYS
                ]
            )
        )
        for language in LANGUAGE_CODES
    }
    return {
        "label_composite_1s_2s_full": composite,
        "minimum_language_recall_composite_1s_2s_full": min(
            per_language_composite.values()
        ),
        "per_language_recall_composite_1s_2s_full": per_language_composite,
        "prefix": summaries,
    }


def evaluate_teacher(
    examples: Sequence[Mapping[str, Any]], model_dir: Path
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], dict[str, Any]]:
    from speechbrain.inference.classifiers import EncoderClassifier

    snapshot, artifact_identity = resolve_teacher_artifact()
    if artifact_identity["artifact_sha256"] != TEACHER_ARTIFACT_SHA256:
        raise ValueError("teacher artifact differs from the main pinned artifact")
    savedir = model_dir / f"{TEACHER_REVISION}-{TEACHER_ARTIFACT_SHA256[:12]}"
    teacher = EncoderClassifier.from_hparams(
        source=str(snapshot),
        savedir=str(savedir),
        overrides={"pretrained_path": str(snapshot)},
        run_opts={"device": "cpu"},
    )
    teacher.eval()
    teacher.hparams.label_encoder.ignore_len()
    indices = label_indices(teacher)
    predictions: dict[str, Mapping[str, Any]] = {}
    retained_mass_by_prefix: dict[str, list[float]] = {}
    started = time.perf_counter()
    for prefix in PREFIX_SECONDS:
        key = prefix_key(prefix)
        eligible = eligible_indices(examples, prefix)
        order = sorted(eligible, key=lambda index: int(examples[index]["num_samples"]))
        guessed: list[str | None] = [None] * len(examples)
        maximums: list[float | None] = [None] * len(examples)
        retained: list[float] = []
        for start in range(0, len(order), TEACHER_BATCH_SIZE):
            batch_indices = order[start : start + TEACHER_BATCH_SIZE]
            waveforms = []
            lengths = []
            for index in batch_indices:
                waveform = examples[index]["waveform"]
                if prefix != "full":
                    waveform = waveform[: round(float(prefix) * SAMPLE_RATE)]
                waveforms.append(waveform)
                lengths.append(len(waveform))
            padded = pad_sequence(waveforms, batch_first=True)
            relative = torch.tensor(lengths, dtype=torch.float32) / padded.shape[1]
            with torch.inference_mode():
                log_probabilities, _, _, _ = teacher.classify_batch(padded, relative)
                selected_logs = log_probabilities[:, indices]
                conditional = torch.softmax(selected_logs, dim=-1)
                masses = selected_logs.exp().sum(dim=-1)
            for batch_index, posterior, mass in zip(
                batch_indices, conditional, masses, strict=True
            ):
                winner = int(posterior.argmax())
                guessed[batch_index] = LANGUAGE_CODES[winner]
                maximums[batch_index] = float(posterior[winner])
                retained.append(float(mass))
        predictions[key] = {
            "prediction": guessed,
            "max_probability": maximums,
        }
        retained_mass_by_prefix[key] = retained
    wall_seconds = time.perf_counter() - started
    summary = summarize_predictions(examples, predictions)
    for key, masses in retained_mass_by_prefix.items():
        summary["prefix"][key]["retained_selected_mass"] = {
            "mean": float(np.mean(masses)),
            "median": float(np.median(masses)),
            "below_0_5": int(np.sum(np.asarray(masses) < 0.5)),
            "below_0_1": int(np.sum(np.asarray(masses) < 0.1)),
        }
    return summary, predictions, {
        "model_id": TEACHER_NAME,
        "revision": TEACHER_REVISION,
        "artifact_sha256": TEACHER_ARTIFACT_SHA256,
        "wall_seconds": wall_seconds,
        "conditional_prediction_space": "selected_7way_conditional_T1",
    }


def compute_features(examples: Sequence[dict[str, Any]]) -> float:
    frontend = LogMelFrontend().eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for completed, example in enumerate(examples, start=1):
            features = frontend(example["waveform"]).squeeze(0).cpu()
            expected_frames = feature_frame_count(int(example["num_samples"]))
            if len(features) != expected_frames:
                raise AssertionError(
                    f"feature count differs for {example['id']}: {len(features)} != {expected_frames}"
                )
            example["features"] = features
            if completed % 100 == 0 or completed == len(examples):
                print(f"computed features {completed}/{len(examples)}", flush=True)
    return time.perf_counter() - started


def evaluate_student_state(
    model: CausalLIDStudent,
    state: Mapping[str, torch.Tensor],
    examples: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    model.load_state_dict(state)
    model.eval()
    raw_predictions: dict[str, dict[str, list[Any]]] = {
        prefix_key(prefix): {
            "prediction": [None] * len(examples),
            "max_probability": [None] * len(examples),
        }
        for prefix in PREFIX_SECONDS
    }
    order = sorted(range(len(examples)), key=lambda index: len(examples[index]["features"]))
    with torch.inference_mode():
        for start in range(0, len(order), STUDENT_BATCH_SIZE):
            batch_indices = order[start : start + STUDENT_BATCH_SIZE]
            features = pad_sequence(
                [examples[index]["features"] for index in batch_indices],
                batch_first=True,
            )
            logits = model(features)
            for batch_row, example_index in enumerate(batch_indices):
                example = examples[example_index]
                for prefix in PREFIX_SECONDS:
                    key = prefix_key(prefix)
                    if example_index not in eligible_indices(examples, prefix):
                        continue
                    if prefix == "full":
                        frames = len(example["features"])
                    else:
                        frames = feature_frame_count(round(float(prefix) * SAMPLE_RATE))
                    stable_stop = frames - MODEL_LOOKAHEAD_FRAMES
                    if stable_stop <= LABEL_DELAY_FRAMES:
                        raise ValueError(f"{example['id']} has no aligned output at {key}")
                    posterior = torch.softmax(
                        logits[batch_row, LABEL_DELAY_FRAMES:stable_stop], dim=-1
                    ).mean(dim=0)
                    winner = int(posterior.argmax())
                    raw_predictions[key]["prediction"][example_index] = LANGUAGE_CODES[winner]
                    raw_predictions[key]["max_probability"][example_index] = float(
                        posterior[winner]
                    )
    first = examples[0]["features"].unsqueeze(0)
    with torch.inference_mode():
        whole = model(first)[:, : first.shape[1] - MODEL_LOOKAHEAD_FRAMES]
        streamed = model.streaming_forward(first, chunk_frames=CHUNK_FRAMES)
    torch.testing.assert_close(whole, streamed, rtol=1e-5, atol=1e-5)
    return summarize_predictions(examples, raw_predictions), raw_predictions


def assert_local_parity(
    step: int,
    current: Mapping[str, Any],
    previous_snapshot: Mapping[str, Any],
) -> None:
    for key in SELECTION_PREFIX_KEYS:
        actual = current["prefix"][key]["language_macro_accuracy"]
        expected = previous_snapshot["synthetic_dev"]["prefix"][key][
            "language_macro_accuracy"
        ]
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8):
            raise AssertionError(
                f"step {step} local {key} metric differs: {actual} != {expected}"
            )
    actual = current["label_composite_1s_2s_full"]
    expected = previous_snapshot["synthetic_dev"][
        "label_selection_composite_1s_2s_full"
    ]
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8):
        raise AssertionError(
            f"step {step} local composite differs: {actual} != {expected}"
        )


def paired_bootstrap_gain(
    examples: Sequence[Mapping[str, Any]],
    challenger: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks: list[np.ndarray] = []
    point_blocks: list[float] = []
    for language in LANGUAGE_CODES:
        for key in SELECTION_PREFIX_KEYS:
            prefix: float | str = "full" if key == "full" else float(key)
            eligible = set(eligible_indices(examples, prefix))
            indices = np.asarray(
                [
                    index
                    for index, example in enumerate(examples)
                    if example["language"] == language and index in eligible
                ],
                dtype=np.int64,
            )
            if not len(indices):
                raise ValueError(f"bootstrap block {language}/{key} is empty")
            truth = np.asarray([examples[index]["language"] for index in indices])
            challenger_correct = np.asarray(
                [challenger[key]["prediction"][index] for index in indices]
            ) == truth
            baseline_correct = np.asarray(
                [baseline[key]["prediction"][index] for index in indices]
            ) == truth
            paired = challenger_correct.astype(np.float64) - baseline_correct.astype(
                np.float64
            )
            point_blocks.append(float(paired.mean()))
            draws = rng.integers(
                0, len(paired), size=(BOOTSTRAP_REPLICATES, len(paired))
            )
            blocks.append(paired[draws].mean(axis=1))
    distribution = np.mean(np.stack(blocks, axis=1), axis=1)
    point = float(np.mean(point_blocks))
    low, high = np.quantile(distribution, (0.025, 0.975))
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "method": (
            "paired percentile bootstrap, resampled within each language/prefix "
            "block and macro-averaged over 7 languages x 3 prefixes"
        ),
        "gain": point,
        "gain_pp": 100 * point,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "ci95_low_pp": 100 * float(low),
        "ci95_high_pp": 100 * float(high),
    }


def external_select(
    steps: Sequence[int],
    external: Mapping[int, Mapping[str, Any]],
    raw_predictions: Mapping[int, Mapping[str, Mapping[str, Any]]],
    examples: Sequence[Mapping[str, Any]],
    previous_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    final_step = 1600
    if final_step not in steps:
        raise ValueError("frozen shortlist does not contain final step 1600")
    best_label = max(external[step]["label_composite_1s_2s_full"] for step in steps)
    best_minimum = max(
        external[step]["minimum_language_recall_composite_1s_2s_full"]
        for step in steps
    )
    final_false_switches = previous_rows[final_step][
        "monolingual_committed_false_switches"
    ]
    candidates: list[dict[str, Any]] = []
    for step in steps:
        row = previous_rows[step]
        eligibility = {
            "within_2pp_of_best_external_label_composite": external[step][
                "label_composite_1s_2s_full"
            ]
            >= best_label - LABEL_ELIGIBILITY_TOLERANCE,
            "within_5pp_of_best_external_minimum_language_recall": external[step][
                "minimum_language_recall_composite_1s_2s_full"
            ]
            >= best_minimum - WORST_LANGUAGE_TOLERANCE,
            "local_committed_false_switches_no_worse_than_final": row[
                "monolingual_committed_false_switches"
            ]
            <= final_false_switches,
        }
        candidates.append(
            {
                "step": step,
                "model_state_sha256": row["model_state_sha256"],
                "shortlist_reason": row["shortlist_reason"],
                "external_label_composite": external[step][
                    "label_composite_1s_2s_full"
                ],
                "external_minimum_language_recall": external[step][
                    "minimum_language_recall_composite_1s_2s_full"
                ],
                "local_raw_switch_recall_at_1s": row["raw_switch_recall_at_1s"],
                "local_raw_miss_penalized_mean_lag_ms": row[
                    "raw_miss_penalized_mean_lag_ms"
                ],
                "local_switch_no_collar_accuracy": row[
                    "switch_no_collar_accuracy"
                ],
                "local_switch_raw_changes_per_minute": row[
                    "switch_raw_changes_per_minute"
                ],
                "local_development_distillation_loss": row[
                    "development_distillation_loss"
                ],
                "local_monolingual_committed_false_switches": row[
                    "monolingual_committed_false_switches"
                ],
                "eligibility": eligibility,
                "eligible": all(eligibility.values()),
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                -row["local_raw_switch_recall_at_1s"],
                row["local_raw_miss_penalized_mean_lag_ms"],
                1.0 - row["local_switch_no_collar_accuracy"],
                row["local_switch_raw_changes_per_minute"],
                row["local_development_distillation_loss"],
                row["step"],
            ),
        )
        selection_status = "selected_from_external_eligible_shortlist"
    else:
        selected = next(candidate for candidate in candidates if candidate["step"] == final_step)
        selection_status = "no_candidate_passed_external_eligibility; retain_final"

    selected_step = int(selected["step"])
    bootstrap = paired_bootstrap_gain(
        examples,
        raw_predictions[selected_step],
        raw_predictions[final_step],
    )
    gain_pp = 100 * (
        external[selected_step]["label_composite_1s_2s_full"]
        - external[final_step]["label_composite_1s_2s_full"]
    )
    confirmation_gates = {
        "selected_is_earlier_than_final": selected_step < final_step,
        "external_composite_gain_at_least_2pp": gain_pp
        >= EXTERNAL_LABEL_GAIN_GATE_PP,
        "paired_bootstrap_ci95_excludes_zero": bootstrap["ci95_low"] > 0.0,
        "minimum_language_recall_no_more_than_5pp_below_final": external[
            selected_step
        ]["minimum_language_recall_composite_1s_2s_full"]
        >= external[final_step]["minimum_language_recall_composite_1s_2s_full"]
        - WORST_LANGUAGE_TOLERANCE,
        "local_committed_false_switches_no_worse_than_final": selected[
            "local_monolingual_committed_false_switches"
        ]
        <= final_false_switches,
    }
    promote_earlier_snapshot = all(confirmation_gates.values())
    return {
        "candidates": candidates,
        "selection_status": selection_status,
        "selected_step": selected_step,
        "final_step": final_step,
        "external_label_composite_gain_pp": gain_pp,
        "selected_vs_final_paired_bootstrap": bootstrap,
        "confirmation_gates": confirmation_gates,
        "promote_earlier_snapshot": promote_earlier_snapshot,
        "selector_order_after_external_eligibility": [
            "higher frozen local raw stable-switch recall at 1 s",
            "lower frozen local miss-penalized raw switch lag",
            "lower frozen local no-collar switch error",
            "lower frozen local raw switch changes per minute",
            "lower frozen local development KD",
            "earlier optimizer step",
        ],
    }


def assert_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            assert_finite_json(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            assert_finite_json(nested, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}: {value}")


def compact_predictions(
    examples: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "clip_ids": [example["id"] for example in examples],
        "expected": [example["language"] for example in examples],
        "by_prefix": {
            key: {
                "prediction": values["prediction"],
                "max_probability": values["max_probability"],
            }
            for key, values in predictions.items()
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.threads)
    launched_at = datetime.now(timezone.utc).isoformat()
    wall_started = time.perf_counter()
    source_at_start = source_snapshot()
    reference_hashes_at_start = {
        "trajectory": file_sha256(args.trajectory),
        "trajectory_results": file_sha256(args.trajectory_results),
        "main_manifest": file_sha256(args.manifest),
    }

    previous = json.loads(args.trajectory_results.read_text(encoding="utf-8"))
    if previous["experiment"] != "checkpoint-trajectory":
        raise ValueError("trajectory results are from the wrong experiment")
    expected_trajectory_hash = previous["trajectory_artifact"]["sha256"]
    if reference_hashes_at_start["trajectory"] != expected_trajectory_hash:
        raise ValueError("trajectory bytes differ from the prior bound artifact")
    trajectory = torch.load(args.trajectory, map_location="cpu", weights_only=True)
    if trajectory["analysis_version"] != "checkpoint-trajectory-v1":
        raise ValueError("trajectory analysis version differs")

    shortlist = previous["selection"]["shortlist"]
    if not 1 <= len(shortlist) <= 5:
        raise ValueError(f"frozen shortlist has {len(shortlist)} states")
    steps = [int(row["step"]) for row in shortlist]
    if len(set(steps)) != len(steps):
        raise ValueError("frozen shortlist contains duplicate steps")
    previous_rows = {int(row["step"]): row for row in shortlist}
    previous_curve = {int(row["step"]): row for row in previous["curve"]}
    states: dict[int, Mapping[str, torch.Tensor]] = {}
    for step in steps:
        state = trajectory["model_states"][str(step)]
        actual_hash = model_state_sha256(state)
        expected_hash = previous_rows[step]["model_state_sha256"]
        if actual_hash != expected_hash:
            raise ValueError(f"trajectory state hash differs at step {step}")
        states[step] = state
    print(f"frozen shortlist steps: {steps}", flush=True)

    external_manifest = prepare_external_manifest(args.data_dir)
    external_examples = load_external_examples(external_manifest, args.data_dir)
    synthetic_examples, synthetic_identity = load_synthetic_examples(args.manifest)
    if Counter(example["language"] for example in external_examples) != Counter(
        {language: SAMPLE_PER_LANGUAGE for language in LANGUAGE_CODES}
    ):
        raise AssertionError("external sample is not language-balanced")
    print(
        f"loaded {len(external_examples)} external validation and "
        f"{len(synthetic_examples)} exact held-out clips",
        flush=True,
    )

    teacher_summary, teacher_predictions, teacher_identity = evaluate_teacher(
        external_examples,
        REPO_ROOT / ".cache/models/lang-id-voxlingua107-ecapa-external-validation",
    )
    print(
        "teacher external composite "
        f"{100 * teacher_summary['label_composite_1s_2s_full']:.2f}%",
        flush=True,
    )

    all_examples = external_examples + synthetic_examples
    feature_wall_seconds = compute_features(all_examples)
    for example in all_examples:
        del example["waveform"]
    external_count = len(external_examples)

    model = CausalLIDStudent(**configured_model_kwargs())
    external_metrics: dict[int, dict[str, Any]] = {}
    synthetic_metrics: dict[int, dict[str, Any]] = {}
    external_predictions: dict[int, Mapping[str, Mapping[str, Any]]] = {}
    scoring_started = time.perf_counter()
    for step in steps:
        print(f"scoring frozen step {step}", flush=True)
        external_summary, external_raw = evaluate_student_state(
            model, states[step], all_examples[:external_count]
        )
        synthetic_summary, _ = evaluate_student_state(
            model, states[step], all_examples[external_count:]
        )
        assert_local_parity(step, synthetic_summary, previous_curve[step])
        external_metrics[step] = external_summary
        synthetic_metrics[step] = synthetic_summary
        external_predictions[step] = external_raw
    student_scoring_wall_seconds = time.perf_counter() - scoring_started

    selection = external_select(
        steps,
        external_metrics,
        external_predictions,
        external_examples,
        previous_rows,
    )
    if selection["promote_earlier_snapshot"]:
        verdict = "adopt"
        verdict_reason = (
            "the externally eligible earlier snapshot clears the predeclared "
            "material-gain, paired-bootstrap, worst-language, and frozen-stability gates"
        )
    else:
        verdict = "reject"
        verdict_reason = (
            "the pinned external validation does not confirm an earlier snapshot "
            "under every predeclared promotion gate; retain step 1600"
        )

    if source_snapshot() != source_at_start:
        raise RuntimeError("experiment or imported source changed during the run")
    reference_hashes_at_end = {
        "trajectory": file_sha256(args.trajectory),
        "trajectory_results": file_sha256(args.trajectory_results),
        "main_manifest": file_sha256(args.manifest),
    }
    if reference_hashes_at_end != reference_hashes_at_start:
        raise RuntimeError("trajectory, prior results, or main manifest changed during run")
    validate_cached_external_manifest(external_manifest, args.data_dir)

    configuration = {
        "analysis_version": ANALYSIS_VERSION,
        "threads": args.threads,
        "shortlist_source": "frozen checkpoint-trajectory shortlist; no candidates added",
        "shortlist_steps": steps,
        "prefix_seconds": [0.5, 1, 2, 4, "full"],
        "selection_prefixes": list(SELECTION_PREFIX_KEYS),
        "short_clip_policy": "exclude from that prefix; never zero-pad",
        "student_batch_size": STUDENT_BATCH_SIZE,
        "teacher_batch_size": TEACHER_BATCH_SIZE,
        "external_validation": external_manifest["identity"],
        "external_label_gain_gate_pp": EXTERNAL_LABEL_GAIN_GATE_PP,
        "label_eligibility_tolerance_pp": 100 * LABEL_ELIGIBILITY_TOLERANCE,
        "worst_language_tolerance_pp": 100 * WORST_LANGUAGE_TOLERANCE,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "locked_test_read": False,
    }
    experiment_identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "configuration": configuration,
        "source_files": source_at_start,
        "reference_hashes": reference_hashes_at_start,
        "prior_experiment_run_id": previous["run_id"],
        "external_manifest_sha256": external_manifest["manifest_sha256"],
        "external_pool_metadata_sha256": external_manifest["pool_metadata_sha256"],
        "synthetic_heldout_input_sha256": synthetic_identity["heldout_input_sha256"],
        "teacher_artifact_sha256": TEACHER_ARTIFACT_SHA256,
        "model_state_sha256_by_step": {
            str(step): previous_rows[step]["model_state_sha256"] for step in steps
        },
    }
    run_id = canonical_sha256(experiment_identity)
    results = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "question": (
            "Does pinned FLEURS validation confirm promotion of an earlier state "
            "from the already-frozen five-checkpoint shortlist over step 1600?"
        ),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "run_id": run_id,
        "launched_at_utc": launched_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - wall_started,
        "experiment_identity": experiment_identity,
        "configuration": configuration,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "torch_threads": torch.get_num_threads(),
        },
        "data": {
            "external_role": "model_validation",
            "external_test_rows_read": 0,
            "external_manifest_sha256": external_manifest["manifest_sha256"],
            "external_pool_metadata_sha256": external_manifest[
                "pool_metadata_sha256"
            ],
            "external_pool_rows": external_manifest["pool_rows"],
            "external_selected_rows": len(external_manifest["records"]),
            "external_records": external_manifest["records"],
            "synthetic_role": "repeatedly_consulted_synthetic_dev",
            "synthetic_heldout_clips": len(synthetic_examples),
            "synthetic_heldout_clip_ids": [
                example["id"] for example in synthetic_examples
            ],
            "synthetic_identity": synthetic_identity,
        },
        "teacher": {
            "identity": teacher_identity,
            "external_validation": teacher_summary,
            "predictions": compact_predictions(
                external_examples, teacher_predictions
            ),
        },
        "students": {
            str(step): {
                "step": step,
                "model_state_sha256": previous_rows[step]["model_state_sha256"],
                "shortlist_reason": previous_rows[step]["shortlist_reason"],
                "external_validation": external_metrics[step],
                "synthetic_dev_same_21_clips": synthetic_metrics[step],
                "external_predictions": compact_predictions(
                    external_examples, external_predictions[step]
                ),
            }
            for step in steps
        },
        "selection": selection,
        "timing": {
            "teacher_inference_wall_seconds": teacher_identity["wall_seconds"],
            "feature_extraction_wall_seconds": feature_wall_seconds,
            "student_shortlist_scoring_wall_seconds": student_scoring_wall_seconds,
        },
        "identity_audit": {
            "source_snapshot_unchanged": True,
            "reference_files_unchanged": True,
            "trajectory_hash_matched_prior_experiment": True,
            "all_shortlist_state_hashes_matched": True,
            "external_manifest_and_audio_revalidated_at_end": True,
            "synthetic_metrics_matched_prior_trajectory_experiment": True,
            "stable_streaming_full_equivalence_checked_per_state": True,
        },
        "decision": {
            "verdict": verdict,
            "reason": verdict_reason,
            "selected_step": selection["selected_step"],
            "final_step": selection["final_step"],
            "promote_earlier_snapshot": selection["promote_earlier_snapshot"],
            "main_checkpoint_modified": False,
            "locked_test_read": False,
        },
        "limitations": [
            "FLEURS validation has no speaker identifier in the published schema",
            "the external sample is 100 deterministic clips per language, not all 2,598 validation rows",
            "FLEURS is read speech and does not validate natural Hinglish or telephony",
            "switch and stability tie-breakers remain frozen synthetic-development evidence",
            "the five states share one seed and one tiny synthetic training trajectory",
            "locked FLEURS test and every natural code-switch test remained unread",
        ],
    }
    assert_finite_json(results)
    atomic_json_write(results, args.output)
    print(
        f"wrote {args.output}; verdict={verdict}; selected={selection['selected_step']}; "
        f"run_id={run_id}",
        flush=True,
    )


if __name__ == "__main__":
    main()
