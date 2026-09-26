#!/usr/bin/env python3
"""Frozen-checkpoint voice-familiarity x text-familiarity factorial.

For every project language, render three Edge-training texts and the three
held-out texts with both the exact training Edge voice and the exact held-out
Edge voice.  Score the pinned ECAPA teacher and submitted causal student at
0.5/1/2/4 seconds and full length.  The original cached matching clips are
also scored as exact-sample controls, including the same 21 held-out clips as
the main pipeline.

Run from the repository root with::

    UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
      experiments/voice-text-factorial/run.py --fresh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# Import the corpus recipe, teacher, frontend, model, timing, and scoring
# primitives from the main pipeline.  This experiment does not edit them.
from scripts.eval import (  # noqa: E402
    chunk_availability_times,
    chunk_posteriors,
    smooth_posteriors,
)
from scripts.prepare_data import (  # noqa: E402
    EDGE_VOICES,
    SENTENCES,
    corpus_generator_identity,
    inspect_wav,
    synthesize,
    tts_audio_recipe,
)
from scripts.teacher_targets import (  # noqa: E402
    label_indices,
    resolve_teacher_artifact,
)
from streaming_lid.audio import LogMelFrontend, load_audio  # noqa: E402
from streaming_lid.config import (  # noqa: E402
    CHUNK_FRAMES,
    HOP_LENGTH,
    LABEL_DELAY_FRAMES,
    LANGUAGE_CODES,
    MODEL_LOOKAHEAD_FRAMES,
    SAMPLE_RATE,
    TEACHER_ARTIFACT_SHA256,
    TEACHER_REVISION,
    WIN_LENGTH,
)
from streaming_lid.data import (  # noqa: E402
    file_sha256,
    read_manifest,
    resolve_audio_path,
)
from streaming_lid.model import CausalLIDStudent  # noqa: E402
from streaming_lid.run_identity import model_state_sha256  # noqa: E402


EXPERIMENT_NAME = "voice-text-factorial"
SCHEMA_VERSION = 1
ANALYSIS_VERSION = "voice-text-factorial-v1"
SEEN_TEXT_INDICES = (5, 6, 7)
UNSEEN_TEXT_INDICES = (10, 11, 12)
PREFIX_SPECS: tuple[float | str, ...] = (0.5, 1.0, 2.0, 4.0, "full")
VOICE_GAP_GATE_PP = 10.0
VOICE_DIRECTION_GATE_LANGUAGES = 5
TEACHER_LOSS_GATE_PP = 2.0
TEACHER_ACCURACY_GATE = 0.90
EXACT_TRAIN_CEILING_GATE = 0.80
POLICY_THRESHOLD = 0.60
POLICY_MARGIN = 0.10
POLICY_DWELL_CHUNKS = 3
POLICY_EMA_NEW_WEIGHT = 0.30

SOURCE_FILES = (
    "scripts/prepare_data.py",
    "scripts/teacher_targets.py",
    "scripts/eval.py",
    "src/streaming_lid/audio.py",
    "src/streaming_lid/config.py",
    "src/streaming_lid/data.py",
    "src/streaming_lid/model.py",
    "src/streaming_lid/run_identity.py",
    "experiments/voice-text-factorial/run.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/generated/manifest.jsonl")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/student.pt")
    )
    parser.add_argument(
        "--reference-summary", type=Path, default=Path("results/summary.json")
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/experiments/voice-text-factorial"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".cache/models/lang-id-voxlingua107-ecapa"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/voice-text-factorial/results.json"),
    )
    parser.add_argument("--teacher-batch-size", type=int, default=24)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--refresh-audio",
        action="store_true",
        help="Fetch all 84 Edge renders again instead of reusing bound cache bytes.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Atomically replace an existing results file after all checks pass.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.threads != 6:
        raise ValueError("team protocol requires exactly six Torch threads")
    if args.teacher_batch_size < 1:
        raise ValueError("teacher-batch-size must be a positive integer")
    if args.output.exists() and not args.fresh:
        raise FileExistsError(f"{args.output} exists; pass --fresh to replace it")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def snapshot_sources() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        payload = path.read_bytes()
        result[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return result


def selected_main_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        f"{language}_{'train' if index < 10 else 'heldout'}_{index:02d}"
        for language in LANGUAGE_CODES
        for index in SEEN_TEXT_INDICES + UNSEEN_TEXT_INDICES
    }
    selected = [item for item in records if item["id"] in wanted]
    if len(selected) != 42 or {item["id"] for item in selected} != wanted:
        raise AssertionError("main manifest does not contain the expected 42 controls")
    return sorted(selected, key=lambda item: item["id"])


def input_fingerprint(records: list[dict[str, Any]], manifest: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(records, key=lambda value: value["id"]):
        digest.update(
            json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        )
        audio_path = resolve_audio_path(item, manifest)
        with audio_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def desired_fresh_records(
    main_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {item["id"]: item for item in main_records}
    generator_identity = corpus_generator_identity()
    desired: list[dict[str, Any]] = []
    for language in LANGUAGE_CODES:
        for text_familiarity, indices in (
            ("seen_text", SEEN_TEXT_INDICES),
            ("unseen_text", UNSEEN_TEXT_INDICES),
        ):
            for text_index in indices:
                split = "train" if text_index < 10 else "heldout"
                source_id = f"{language}_{split}_{text_index:02d}"
                source = by_id[source_id]
                expected_text = SENTENCES[language][text_index]
                if source["text"] != expected_text:
                    raise AssertionError(f"text mismatch for {source_id}")
                for voice_familiarity, voice_key in (
                    ("seen_voice", "male"),
                    ("unseen_voice", "female"),
                ):
                    voice = EDGE_VOICES[language][voice_key]
                    tts_recipe = tts_audio_recipe(
                        text=expected_text,
                        language=language,
                        voice=voice,
                        generator_identity=generator_identity,
                    )
                    experiment_recipe = {
                        "experiment": EXPERIMENT_NAME,
                        "analysis_version": ANALYSIS_VERSION,
                        "cache_schema_version": SCHEMA_VERSION,
                        "source_manifest_id": source_id,
                        "text_index": text_index,
                        "text_familiarity": text_familiarity,
                        "voice_familiarity": voice_familiarity,
                        "tts_recipe": tts_recipe,
                    }
                    clip_id = (
                        f"{language}_{text_familiarity}_{text_index:02d}_"
                        f"{voice_familiarity}"
                    )
                    desired.append(
                        {
                            "cache_schema_version": SCHEMA_VERSION,
                            "id": clip_id,
                            "language": language,
                            "text": expected_text,
                            "text_index": text_index,
                            "text_familiarity": text_familiarity,
                            "voice": voice,
                            "voice_familiarity": voice_familiarity,
                            "source_manifest_id": source_id,
                            "recipe": experiment_recipe,
                            "recipe_sha256": canonical_sha256(experiment_recipe),
                        }
                    )
    if len(desired) != 84:
        raise AssertionError("factorial must contain exactly 84 fresh renders")
    return desired


def read_cache_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        return []
    records = payload.get("records")
    return records if isinstance(records, list) else []


def publish_cache_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "analysis_version": ANALYSIS_VERSION,
        "records": sorted(records, key=lambda item: item["id"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.pending")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def reusable_cache_record(
    cached: dict[str, Any] | None,
    desired: dict[str, Any],
    cache_dir: Path,
) -> dict[str, Any] | None:
    if cached is None:
        return None
    for key in (
        "id",
        "language",
        "text",
        "text_index",
        "text_familiarity",
        "voice",
        "voice_familiarity",
        "source_manifest_id",
        "recipe",
        "recipe_sha256",
    ):
        if cached.get(key) != desired[key]:
            return None
    audio_path_value = cached.get("audio_path")
    audio_sha256 = cached.get("audio_sha256")
    if not isinstance(audio_path_value, str) or not isinstance(audio_sha256, str):
        return None
    try:
        root = cache_dir.resolve()
        audio_path = (cache_dir / audio_path_value).resolve(strict=True)
        audio_path.relative_to(root)
        _, duration, actual_sha256 = inspect_wav(audio_path)
    except (OSError, ValueError):
        return None
    if actual_sha256 != audio_sha256:
        return None
    expected_name = f"{desired['id']}--sha256-{actual_sha256}.wav"
    if audio_path.name != expected_name:
        return None
    result = dict(cached)
    result["duration_seconds"] = round(duration, 4)
    return result


def prepare_fresh_audio(
    desired: list[dict[str, Any]], cache_dir: Path, *, refresh: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audio_dir = cache_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    existing = {item["id"]: item for item in read_cache_manifest(manifest_path)}
    published: dict[str, dict[str, Any]] = {}
    reused = 0
    synthesized = 0
    started = time.perf_counter()
    for number, item in enumerate(desired, start=1):
        record = None
        if not refresh:
            record = reusable_cache_record(existing.get(item["id"]), item, cache_dir)
        if record is None:
            pending = audio_dir / f"{item['id']}--pending.wav"
            print(
                f"synthesizing {number:02d}/{len(desired)} {item['id']} "
                f"with {item['voice']}",
                flush=True,
            )
            synthesize(item["text"], item["language"], item["voice"], pending)
            _, duration, audio_sha256 = inspect_wav(pending)
            final_path = audio_dir / f"{item['id']}--sha256-{audio_sha256}.wav"
            os.replace(pending, final_path)
            record = {
                **item,
                "audio_path": f"audio/{final_path.name}",
                "audio_sha256": audio_sha256,
                "duration_seconds": round(duration, 4),
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "provider_revision_disclosed": False,
            }
            synthesized += 1
        else:
            reused += 1
        published[item["id"]] = record
        # Each successful response becomes resumable without exposing a partial
        # record or unverified pending path.
        interim = dict(existing)
        interim.update(published)
        publish_cache_manifest(manifest_path, list(interim.values()))
    records = [published[item["id"]] for item in desired]
    publish_cache_manifest(manifest_path, records)
    return records, {
        "manifest_path": str(manifest_path),
        "synthesized": synthesized,
        "reused": reused,
        "wall_seconds": time.perf_counter() - started,
        "provider_revision_disclosed": False,
    }


def validate_fresh_cache(
    desired: list[dict[str, Any]], cache_dir: Path
) -> tuple[list[dict[str, Any]], str]:
    manifest_path = cache_dir / "manifest.json"
    cached = {item["id"]: item for item in read_cache_manifest(manifest_path)}
    validated = []
    digest = hashlib.sha256()
    for item in desired:
        record = reusable_cache_record(cached.get(item["id"]), item, cache_dir)
        if record is None:
            raise ValueError(f"invalid or missing fresh cache record {item['id']}")
        validated.append(record)
        digest.update(
            json.dumps(
                record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        )
        audio_path = cache_dir / record["audio_path"]
        with audio_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return validated, digest.hexdigest()


def exact_control_records(
    main_records: list[dict[str, Any]], manifest: Path
) -> list[dict[str, Any]]:
    controls = []
    for item in main_records:
        text_index = int(item["id"].rsplit("_", 1)[1])
        text_familiarity = "seen_text" if text_index in SEEN_TEXT_INDICES else "unseen_text"
        voice_familiarity = "seen_voice" if text_index in SEEN_TEXT_INDICES else "unseen_voice"
        expected_voice = EDGE_VOICES[item["language"]][
            "male" if voice_familiarity == "seen_voice" else "female"
        ]
        if item["speaker_id"] != expected_voice:
            raise AssertionError(f"unexpected Edge voice for {item['id']}")
        controls.append(
            {
                "id": f"exact_{item['id']}",
                "source_manifest_id": item["id"],
                "render_kind": "exact_original",
                "language": item["language"],
                "text": item["text"],
                "text_index": text_index,
                "text_familiarity": text_familiarity,
                "voice": item["speaker_id"],
                "voice_familiarity": voice_familiarity,
                "audio_path": str(resolve_audio_path(item, manifest)),
                "audio_sha256": item["audio_sha256"],
                "duration_seconds": item["duration_seconds"],
            }
        )
    return controls


def fresh_base_records(
    records: list[dict[str, Any]], cache_dir: Path
) -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "source_manifest_id": item["source_manifest_id"],
            "render_kind": "fresh_factorial",
            "language": item["language"],
            "text": item["text"],
            "text_index": item["text_index"],
            "text_familiarity": item["text_familiarity"],
            "voice": item["voice"],
            "voice_familiarity": item["voice_familiarity"],
            "audio_path": str((cache_dir / item["audio_path"]).resolve()),
            "audio_sha256": item["audio_sha256"],
            "duration_seconds": item["duration_seconds"],
        }
        for item in records
    ]


def prefix_waveform(
    waveform: torch.Tensor, spec: float | str
) -> tuple[torch.Tensor, bool, str]:
    if spec == "full":
        return waveform, False, "full"
    seconds = float(spec)
    wanted = round(seconds * SAMPLE_RATE)
    padded = len(waveform) < wanted
    result = waveform[:wanted]
    if len(result) < wanted:
        result = F.pad(result, (0, wanted - len(result)))
    label = str(int(seconds)) if seconds.is_integer() else str(seconds)
    return result.contiguous(), padded, label


def load_teacher(model_dir: Path):
    from speechbrain.inference.classifiers import EncoderClassifier

    snapshot, artifact = resolve_teacher_artifact()
    savedir = model_dir.resolve() / f"{TEACHER_REVISION}-{TEACHER_ARTIFACT_SHA256[:12]}"
    teacher = EncoderClassifier.from_hparams(
        source=str(snapshot),
        savedir=str(savedir),
        overrides={"pretrained_path": str(snapshot)},
        run_opts={"device": "cpu"},
    )
    teacher.eval()
    teacher.hparams.label_encoder.ignore_len()
    return teacher, label_indices(teacher), artifact


def teacher_posteriors(
    teacher,
    selected_indices: list[int],
    waveforms: list[torch.Tensor],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    probabilities = []
    masses = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(waveforms), batch_size):
            batch = waveforms[start : start + batch_size]
            lengths = torch.tensor([len(value) for value in batch], dtype=torch.float32)
            padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
            relative_lengths = lengths / padded.shape[1]
            log_probabilities, _, _, _ = teacher.classify_batch(
                padded, relative_lengths
            )
            selected = log_probabilities[:, selected_indices]
            probabilities.append(torch.softmax(selected, dim=-1).cpu())
            masses.append(selected.exp().sum(dim=-1).cpu())
    wall_seconds = time.perf_counter() - started
    posterior = torch.cat(probabilities).numpy().astype(np.float32)
    retained_mass = torch.cat(masses).numpy().astype(np.float32)
    if posterior.shape != (len(waveforms), len(LANGUAGE_CODES)):
        raise AssertionError("teacher posterior shape mismatch")
    if not np.isfinite(posterior).all() or not np.isfinite(retained_mass).all():
        raise FloatingPointError("teacher returned non-finite output")
    if not np.allclose(posterior.sum(axis=-1), 1.0, atol=1e-6):
        raise AssertionError("restricted teacher probabilities are not normalized")
    return posterior, retained_mass, wall_seconds


def probability_payload(probabilities: np.ndarray) -> dict[str, Any]:
    prediction = int(probabilities.argmax())
    return {
        "probabilities": probabilities.tolist(),
        "prediction": LANGUAGE_CODES[prediction],
        "max_probability": float(probabilities[prediction]),
    }


def generic_policy(
    probabilities: np.ndarray, availability_seconds: np.ndarray
) -> dict[str, Any]:
    if len(probabilities) == 0:
        return {
            "initial_commit": None,
            "initial_commit_seconds": None,
            "transitions": [],
        }
    ema = smooth_posteriors(probabilities, POLICY_EMA_NEW_WEIGHT)
    active: int | None = None
    candidate: int | None = None
    count = 0
    initial_language = None
    initial_time = None
    transitions = []
    for index, posterior in enumerate(ema):
        order = np.argsort(posterior)
        winner = int(order[-1])
        qualifies = (
            posterior[winner] >= POLICY_THRESHOLD
            and posterior[winner] - posterior[int(order[-2])] >= POLICY_MARGIN
        )
        wanted = winner if qualifies and winner != active else None
        if wanted is None:
            candidate = None
            count = 0
            continue
        if wanted == candidate:
            count += 1
        else:
            candidate = wanted
            count = 1
        if count < POLICY_DWELL_CHUNKS:
            continue
        event = {
            "language": LANGUAGE_CODES[wanted],
            "availability_seconds": float(availability_seconds[index]),
        }
        if active is None:
            initial_language = event["language"]
            initial_time = event["availability_seconds"]
        else:
            transitions.append(
                {
                    "from": LANGUAGE_CODES[active],
                    "to": LANGUAGE_CODES[wanted],
                    "availability_seconds": event["availability_seconds"],
                }
            )
        active = wanted
        candidate = None
        count = 0
    return {
        "initial_commit": initial_language,
        "initial_commit_seconds": initial_time,
        "final_active": None if active is None else LANGUAGE_CODES[active],
        "transitions": transitions,
    }


def student_result(
    model: CausalLIDStudent,
    frontend: LogMelFrontend,
    waveform: torch.Tensor,
    *,
    expected_index: int,
    teacher_index: int,
    delay_frames: int,
    lookahead_frames: int,
    chunk_frames: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        features = frontend(waveform)
        logits = model.streaming_forward(features, chunk_frames=chunk_frames).squeeze(0)
        valid_logits = logits[delay_frames:]
        if len(valid_logits) == 0:
            raise ValueError("request has no delay-aligned stable student frames")
        frame_probabilities = torch.softmax(valid_logits, dim=-1)
        frame_classes = frame_probabilities.argmax(dim=-1)
        mean_posterior = frame_probabilities.mean(dim=0).cpu().numpy().astype(np.float32)
        majority_index = int(
            torch.bincount(frame_classes, minlength=len(LANGUAGE_CODES)).argmax()
        )
        label_correct = int((frame_classes == expected_index).sum())
        teacher_agreement = int((frame_classes == teacher_index).sum())
        all_probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    availability = chunk_availability_times(
        len(logits),
        chunk_frames=chunk_frames,
        lookahead_frames=lookahead_frames,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        sample_rate=SAMPLE_RATE,
    )
    chunks, chunk_times = chunk_posteriors(
        all_probabilities, availability, min_frame=delay_frames
    )
    policy = generic_policy(chunks, chunk_times)
    return {
        "mean_posterior": mean_posterior.tolist(),
        "mean_posterior_prediction": LANGUAGE_CODES[int(mean_posterior.argmax())],
        "majority_prediction": LANGUAGE_CODES[majority_index],
        "mean_max_probability": float(mean_posterior.max()),
        "valid_frames": len(valid_logits),
        "label_correct_frames": label_correct,
        "teacher_agreement_frames": teacher_agreement,
        "frame_label_accuracy": label_correct / len(valid_logits),
        "frame_teacher_agreement": teacher_agreement / len(valid_logits),
        "policy": policy,
    }


def macro_f1(
    expected: list[int], predicted: list[int]
) -> tuple[float, dict[str, float]]:
    values: dict[str, float] = {}
    for index, language in enumerate(LANGUAGE_CODES):
        true_positive = sum(
            truth == index and guess == index
            for truth, guess in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            truth != index and guess == index
            for truth, guess in zip(expected, predicted, strict=True)
        )
        false_negative = sum(
            truth == index and guess != index
            for truth, guess in zip(expected, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        values[language] = 0.0 if denominator == 0 else 2 * true_positive / denominator
    return float(np.mean(list(values.values()))), values


def classification_summary(
    rows: list[dict[str, Any]], model_name: str
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty request set")
    expected = [LANGUAGE_CODES.index(row["expected"]) for row in rows]
    key = "prediction" if model_name == "teacher" else "majority_prediction"
    predicted = [LANGUAGE_CODES.index(row[model_name][key]) for row in rows]
    f1, per_language_f1 = macro_f1(expected, predicted)
    confusion = {language: Counter() for language in LANGUAGE_CODES}
    recalls: dict[str, float] = {}
    support: dict[str, int] = {}
    for truth, guess in zip(expected, predicted, strict=True):
        confusion[LANGUAGE_CODES[truth]][LANGUAGE_CODES[guess]] += 1
    for index, language in enumerate(LANGUAGE_CODES):
        selected = [guess for truth, guess in zip(expected, predicted, strict=True) if truth == index]
        if not selected:
            raise AssertionError(f"summary has no examples for {language}")
        recalls[language] = sum(value == index for value in selected) / len(selected)
        support[language] = len(selected)
    return {
        "n_requests": len(rows),
        "accuracy": float(np.mean(np.asarray(expected) == np.asarray(predicted))),
        "macro_f1": f1,
        "per_language_recall": recalls,
        "per_language_f1": per_language_f1,
        "support_per_language": support,
        "confusion": {
            truth: {guess: int(count) for guess, count in sorted(values.items())}
            for truth, values in confusion.items()
        },
    }


def policy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    wrong = 0
    no_commit = 0
    false_transitions = 0
    total_audio_seconds = 0.0
    for row in rows:
        commit = row["student"]["policy"]["initial_commit"]
        if commit is None:
            no_commit += 1
        elif commit == row["expected"]:
            correct += 1
        else:
            wrong += 1
        false_transitions += len(row["student"]["policy"]["transitions"])
        total_audio_seconds += row["evaluated_audio_seconds"]
    return {
        "correct_initial_commits": correct,
        "wrong_initial_commits": wrong,
        "no_initial_commit": no_commit,
        "correct_initial_commit_rate": correct / len(rows),
        "false_transitions": false_transitions,
        "false_transitions_per_minute": (
            0.0 if total_audio_seconds == 0 else 60 * false_transitions / total_audio_seconds
        ),
        "policy": {
            "threshold": POLICY_THRESHOLD,
            "margin": POLICY_MARGIN,
            "dwell_chunks": POLICY_DWELL_CHUNKS,
            "ema_new_weight": POLICY_EMA_NEW_WEIGHT,
        },
    }


def combined_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    teacher = classification_summary(rows, "teacher")
    student = classification_summary(rows, "student")
    total_frames = sum(row["student"]["valid_frames"] for row in rows)
    label_correct = sum(row["student"]["label_correct_frames"] for row in rows)
    agreement_correct = sum(
        row["student"]["teacher_agreement_frames"] for row in rows
    )
    return {
        "teacher": teacher,
        "student": student,
        "student_teacher_clip_agreement": float(
            np.mean(
                [
                    row["student"]["majority_prediction"]
                    == row["teacher"]["prediction"]
                    for row in rows
                ]
            )
        ),
        "student_frame_label_accuracy_micro": label_correct / total_frames,
        "student_frame_label_accuracy_clip_macro": float(
            np.mean([row["student"]["frame_label_accuracy"] for row in rows])
        ),
        "student_frame_teacher_agreement_micro": agreement_correct / total_frames,
        "student_frame_teacher_agreement_clip_macro": float(
            np.mean([row["student"]["frame_teacher_agreement"] for row in rows])
        ),
        "mean_teacher_retained_mass": float(
            np.mean([row["teacher"]["retained_mass"] for row in rows])
        ),
        "policy": policy_summary(rows),
    }


def summaries_by_prefix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        label: combined_summary([row for row in rows if row["prefix"] == label])
        for label in ("0.5", "1", "2", "4", "full")
    }


def paired_voice_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seen = [row for row in rows if row["voice_familiarity"] == "seen_voice"]
    unseen = [row for row in rows if row["voice_familiarity"] == "unseen_voice"]
    seen_student = classification_summary(seen, "student")
    unseen_student = classification_summary(unseen, "student")
    seen_teacher = classification_summary(seen, "teacher")
    unseen_teacher = classification_summary(unseen, "teacher")
    seen_by_key = {
        (row["expected"], row["text_index"]): row for row in seen
    }
    unseen_by_key = {
        (row["expected"], row["text_index"]): row for row in unseen
    }
    if seen_by_key.keys() != unseen_by_key.keys():
        raise AssertionError("voice arms do not contain identical text pairs")
    paired_outcomes = Counter()
    for key in sorted(seen_by_key):
        left = seen_by_key[key]
        right = unseen_by_key[key]
        left_correct = left["student"]["majority_prediction"] == left["expected"]
        right_correct = right["student"]["majority_prediction"] == right["expected"]
        if left_correct and not right_correct:
            paired_outcomes["seen_only_correct"] += 1
        elif right_correct and not left_correct:
            paired_outcomes["unseen_only_correct"] += 1
        elif left_correct:
            paired_outcomes["both_correct"] += 1
        else:
            paired_outcomes["both_wrong"] += 1
    per_language_gap = {
        language: 100
        * (
            seen_student["per_language_recall"][language]
            - unseen_student["per_language_recall"][language]
        )
        for language in LANGUAGE_CODES
    }
    return {
        "n_paired_texts": len(seen_by_key),
        "seen_voice": {"teacher": seen_teacher, "student": seen_student},
        "unseen_voice": {"teacher": unseen_teacher, "student": unseen_student},
        "student_unseen_voice_loss_pp": 100
        * (seen_student["accuracy"] - unseen_student["accuracy"]),
        "teacher_unseen_voice_loss_pp": 100
        * (seen_teacher["accuracy"] - unseen_teacher["accuracy"]),
        "teacher_nonnegative_unseen_voice_loss_pp": max(
            0.0, 100 * (seen_teacher["accuracy"] - unseen_teacher["accuracy"])
        ),
        "student_per_language_unseen_voice_loss_pp": per_language_gap,
        "student_languages_with_positive_seen_voice_advantage": sum(
            value > 0 for value in per_language_gap.values()
        ),
        "paired_student_outcomes": dict(sorted(paired_outcomes.items())),
    }


def exact_vs_fresh_matching(
    exact_rows: list[dict[str, Any]], fresh_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    result = {}
    for text_familiarity, voice_familiarity, label in (
        ("seen_text", "seen_voice", "seen_seen"),
        ("unseen_text", "unseen_voice", "unseen_unseen"),
    ):
        exact = [
            row
            for row in exact_rows
            if row["text_familiarity"] == text_familiarity
            and row["voice_familiarity"] == voice_familiarity
        ]
        fresh = [
            row
            for row in fresh_rows
            if row["text_familiarity"] == text_familiarity
            and row["voice_familiarity"] == voice_familiarity
        ]
        exact_by_key = {(row["expected"], row["text_index"]): row for row in exact}
        fresh_by_key = {(row["expected"], row["text_index"]): row for row in fresh}
        if exact_by_key.keys() != fresh_by_key.keys():
            raise AssertionError("exact/fresh matching cells differ")
        actor_payload = {}
        for actor, prediction_key in (
            ("teacher", "prediction"),
            ("student", "majority_prediction"),
        ):
            exact_summary = classification_summary(exact, actor)
            fresh_summary = classification_summary(fresh, actor)
            flips = sum(
                exact_by_key[key][actor][prediction_key]
                != fresh_by_key[key][actor][prediction_key]
                for key in exact_by_key
            )
            actor_payload[actor] = {
                "exact": exact_summary,
                "fresh": fresh_summary,
                "fresh_minus_exact_accuracy_pp": 100
                * (fresh_summary["accuracy"] - exact_summary["accuracy"]),
                "prediction_flips": flips,
            }
        result[label] = actor_payload
    return result


def main_protocol_reproduction(
    rows: list[dict[str, Any]], reference_summary: dict[str, Any]
) -> dict[str, Any]:
    total_frames = sum(row["student"]["valid_frames"] for row in rows)
    student_correct = sum(row["student"]["label_correct_frames"] for row in rows)
    teacher_correct = sum(
        row["student"]["valid_frames"]
        for row in rows
        if row["teacher"]["prediction"] == row["expected"]
    )
    metrics = {
        "heldout_student_label_accuracy_micro": student_correct / total_frames,
        "heldout_student_label_accuracy_macro": float(
            np.mean([row["student"]["frame_label_accuracy"] for row in rows])
        ),
        "heldout_teacher_label_accuracy_micro": teacher_correct / total_frames,
        "heldout_teacher_label_accuracy_macro": float(
            np.mean(
                [row["teacher"]["prediction"] == row["expected"] for row in rows]
            )
        ),
        "heldout_teacher_agreement_micro": sum(
            row["student"]["teacher_agreement_frames"] for row in rows
        )
        / total_frames,
        "heldout_teacher_agreement_macro": float(
            np.mean([row["student"]["frame_teacher_agreement"] for row in rows])
        ),
        "heldout_student_clip_accuracy": float(
            np.mean(
                [
                    row["student"]["majority_prediction"] == row["expected"]
                    for row in rows
                ]
            )
        ),
        "heldout_teacher_clip_accuracy": float(
            np.mean(
                [row["teacher"]["prediction"] == row["expected"] for row in rows]
            )
        ),
    }
    differences = {
        key: metrics[key] - float(reference_summary[key]) for key in metrics
    }
    if max(abs(value) for value in differences.values()) > 1e-12:
        raise AssertionError(
            f"exact held-out controls do not reproduce main metrics: {differences}"
        )
    return {
        "n_exact_heldout_clips": len(rows),
        "metrics": metrics,
        "reference_metrics": {key: reference_summary[key] for key in metrics},
        "differences": differences,
        "exact_match": True,
    }


def evaluate(
    base_records: list[dict[str, Any]],
    teacher,
    selected_indices: list[int],
    model: CausalLIDStudent,
    frontend: LogMelFrontend,
    *,
    teacher_batch_size: int,
    delay_frames: int,
    lookahead_frames: int,
    chunk_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    requests = []
    waveforms = []
    for base in base_records:
        waveform = load_audio(base["audio_path"])
        if file_sha256(Path(base["audio_path"])) != base["audio_sha256"]:
            raise ValueError(f"audio hash changed for {base['id']}")
        for spec in PREFIX_SPECS:
            prefix, padded, label = prefix_waveform(waveform, spec)
            requests.append(
                {
                    **{key: value for key, value in base.items() if key != "audio_path"},
                    "expected": base["language"],
                    "prefix": label,
                    "prefix_padded": padded,
                    "source_duration_seconds": len(waveform) / SAMPLE_RATE,
                    "evaluated_audio_seconds": len(prefix) / SAMPLE_RATE,
                }
            )
            waveforms.append(prefix)
    teacher_probs, retained_mass, teacher_wall = teacher_posteriors(
        teacher, selected_indices, waveforms, teacher_batch_size
    )
    student_started = time.perf_counter()
    for index, (request, waveform) in enumerate(
        zip(requests, waveforms, strict=True)
    ):
        teacher_payload = probability_payload(teacher_probs[index])
        teacher_index = LANGUAGE_CODES.index(teacher_payload["prediction"])
        expected_index = LANGUAGE_CODES.index(request["expected"])
        request["teacher"] = {
            **teacher_payload,
            "retained_mass": float(retained_mass[index]),
        }
        request["student"] = student_result(
            model,
            frontend,
            waveform,
            expected_index=expected_index,
            teacher_index=teacher_index,
            delay_frames=delay_frames,
            lookahead_frames=lookahead_frames,
            chunk_frames=chunk_frames,
        )
        if (index + 1) % 100 == 0 or index + 1 == len(requests):
            print(f"scored {index + 1}/{len(requests)} requests", flush=True)
    return requests, {
        "teacher_inference_wall_seconds": teacher_wall,
        "student_frontend_and_replay_wall_seconds": time.perf_counter()
        - student_started,
        "n_requests": len(requests),
        "audio_seconds": float(sum(len(value) for value in waveforms) / SAMPLE_RATE),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.threads)
    run_started = time.perf_counter()
    source_snapshot_at_start = snapshot_sources()
    reference_hashes_at_start = {
        "checkpoint": file_sha256(args.checkpoint),
        "summary": file_sha256(args.reference_summary),
        "manifest": file_sha256(args.manifest),
    }
    reference_summary = json.loads(
        args.reference_summary.read_text(encoding="utf-8")
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint_model_sha256 = model_state_sha256(checkpoint["model_state"])
    if checkpoint_model_sha256 != reference_summary["model_state_sha256"]:
        raise ValueError("reference summary and checkpoint model state disagree")

    pipeline = checkpoint["run_identity"]["pipeline"]
    if tuple(pipeline["language_codes"]) != tuple(LANGUAGE_CODES):
        raise ValueError("checkpoint language order differs from current main pipeline")
    frontend_config = pipeline["frontend"]
    distillation = pipeline["distillation"]
    streaming = pipeline["streaming"]
    expected_runtime = {
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "win_length": WIN_LENGTH,
        "label_delay_frames": LABEL_DELAY_FRAMES,
        "lookahead_frames": MODEL_LOOKAHEAD_FRAMES,
        "chunk_frames": CHUNK_FRAMES,
    }
    actual_runtime = {
        "sample_rate": int(frontend_config["sample_rate"]),
        "hop_length": int(frontend_config["hop_length"]),
        "win_length": int(frontend_config["win_length"]),
        "label_delay_frames": int(distillation["label_delay_frames"]),
        "lookahead_frames": int(distillation["model_lookahead_frames"]),
        "chunk_frames": int(streaming["chunk_frames"]),
    }
    if actual_runtime != expected_runtime:
        raise ValueError(
            f"checkpoint/runtime timing mismatch: {actual_runtime} != {expected_runtime}"
        )

    records = read_manifest(args.manifest)
    main_controls = selected_main_records(records)
    main_fingerprint_at_start = input_fingerprint(main_controls, args.manifest)
    desired = desired_fresh_records(main_controls)
    fresh_records, synthesis = prepare_fresh_audio(
        desired, args.cache_dir, refresh=args.refresh_audio
    )
    fresh_records, fresh_fingerprint_at_start = validate_fresh_cache(
        desired, args.cache_dir
    )

    teacher, selected_indices, teacher_artifact = load_teacher(args.model_dir)
    if teacher_artifact["artifact_sha256"] != TEACHER_ARTIFACT_SHA256:
        raise ValueError("resolved teacher artifact differs from pinned configuration")
    model = CausalLIDStudent(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    frontend = LogMelFrontend().eval()

    base_records = exact_control_records(main_controls, args.manifest)
    base_records.extend(fresh_base_records(fresh_records, args.cache_dir))
    requests, runtime = evaluate(
        base_records,
        teacher,
        selected_indices,
        model,
        frontend,
        teacher_batch_size=args.teacher_batch_size,
        delay_frames=actual_runtime["label_delay_frames"],
        lookahead_frames=actual_runtime["lookahead_frames"],
        chunk_frames=actual_runtime["chunk_frames"],
    )
    if len(requests) != 630:
        raise AssertionError("expected 126 clips x five evaluation lengths")
    if not all(
        math.isfinite(value)
        for row in requests
        for value in (
            row["teacher"]["max_probability"],
            row["teacher"]["retained_mass"],
            row["student"]["mean_max_probability"],
            row["student"]["frame_label_accuracy"],
            row["student"]["frame_teacher_agreement"],
        )
    ):
        raise FloatingPointError("evaluation produced a non-finite metric")

    fresh = [row for row in requests if row["render_kind"] == "fresh_factorial"]
    exact = [row for row in requests if row["render_kind"] == "exact_original"]
    fresh_full = [row for row in fresh if row["prefix"] == "full"]
    exact_full = [row for row in exact if row["prefix"] == "full"]
    exact_train_full = [
        row for row in exact_full if row["text_familiarity"] == "seen_text"
    ]
    exact_heldout_full = [
        row for row in exact_full if row["text_familiarity"] == "unseen_text"
    ]

    fresh_cells = {}
    for voice_familiarity in ("seen_voice", "unseen_voice"):
        for text_familiarity in ("seen_text", "unseen_text"):
            name = f"{voice_familiarity}__{text_familiarity}"
            selected = [
                row
                for row in fresh
                if row["voice_familiarity"] == voice_familiarity
                and row["text_familiarity"] == text_familiarity
            ]
            fresh_cells[name] = summaries_by_prefix(selected)

    voice_comparisons = {}
    for prefix in ("0.5", "1", "2", "4", "full"):
        prefix_rows = [row for row in fresh if row["prefix"] == prefix]
        voice_comparisons[prefix] = {
            "pooled_texts": paired_voice_comparison(prefix_rows),
            "seen_text": paired_voice_comparison(
                [row for row in prefix_rows if row["text_familiarity"] == "seen_text"]
            ),
            "unseen_text": paired_voice_comparison(
                [row for row in prefix_rows if row["text_familiarity"] == "unseen_text"]
            ),
        }

    main_reproduction = main_protocol_reproduction(
        exact_heldout_full, reference_summary
    )
    exact_controls = {
        "training_edge_subset": summaries_by_prefix(
            [row for row in exact if row["text_familiarity"] == "seen_text"]
        ),
        "main_heldout": summaries_by_prefix(
            [row for row in exact if row["text_familiarity"] == "unseen_text"]
        ),
        "main_protocol_reproduction": main_reproduction,
        "fresh_service_drift": exact_vs_fresh_matching(exact_full, fresh_full),
    }

    primary = voice_comparisons["full"]["pooled_texts"]
    exact_train_student = classification_summary(exact_train_full, "student")
    decision_gates = {
        "student_unseen_voice_loss_ge_10pp": primary[
            "student_unseen_voice_loss_pp"
        ]
        >= VOICE_GAP_GATE_PP,
        "positive_seen_voice_advantage_in_at_least_5_of_7_languages": primary[
            "student_languages_with_positive_seen_voice_advantage"
        ]
        >= VOICE_DIRECTION_GATE_LANGUAGES,
        "teacher_unseen_voice_loss_lt_2pp": primary[
            "teacher_nonnegative_unseen_voice_loss_pp"
        ]
        < TEACHER_LOSS_GATE_PP,
        "teacher_accuracy_ge_90pct_in_both_voice_arms": min(
            primary["seen_voice"]["teacher"]["accuracy"],
            primary["unseen_voice"]["teacher"]["accuracy"],
        )
        >= TEACHER_ACCURACY_GATE,
        "exact_seen_voice_seen_text_student_accuracy_ge_80pct": exact_train_student[
            "accuracy"
        ]
        >= EXACT_TRAIN_CEILING_GATE,
    }
    teacher_valid = (
        decision_gates["teacher_unseen_voice_loss_lt_2pp"]
        and decision_gates["teacher_accuracy_ge_90pct_in_both_voice_arms"]
    )
    if not teacher_valid:
        verdict = "inconclusive"
        verdict_reason = (
            "teacher voice sensitivity or low label accuracy prevents isolating the "
            "student's voice-familiarity effect"
        )
    elif all(decision_gates.values()):
        verdict = "adopt"
        verdict_reason = (
            "the paired full-clip voice gap, cross-language direction, stable teacher, "
            "and exact familiar-audio ceiling all pass the predeclared shortcut gate"
        )
    else:
        verdict = "reject"
        failed = [name for name, passed in decision_gates.items() if not passed]
        verdict_reason = (
            "the evidence does not support voice familiarity as the primary failure "
            "mechanism; failed gates: " + ", ".join(failed)
        )

    # Revalidate every mutable dependency before publishing the result.
    source_snapshot_at_end = snapshot_sources()
    if source_snapshot_at_end != source_snapshot_at_start:
        raise RuntimeError("experiment or imported main source changed during the run")
    reference_hashes_at_end = {
        "checkpoint": file_sha256(args.checkpoint),
        "summary": file_sha256(args.reference_summary),
        "manifest": file_sha256(args.manifest),
    }
    if reference_hashes_at_end != reference_hashes_at_start:
        raise RuntimeError("reference checkpoint, summary, or manifest changed")
    current_records = selected_main_records(read_manifest(args.manifest))
    if input_fingerprint(current_records, args.manifest) != main_fingerprint_at_start:
        raise RuntimeError("selected main records or audio changed during the run")
    validated_fresh, fresh_fingerprint_at_end = validate_fresh_cache(
        desired, args.cache_dir
    )
    if fresh_fingerprint_at_end != fresh_fingerprint_at_start:
        raise RuntimeError("fresh factorial cache changed during the run")
    if [item["id"] for item in validated_fresh] != [item["id"] for item in fresh_records]:
        raise RuntimeError("fresh factorial cache order changed")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "source_snapshot": source_snapshot_at_start,
        "reference_file_sha256": reference_hashes_at_start,
        "reference_run_id": reference_summary["run_id"],
        "reference_model_state_sha256": checkpoint_model_sha256,
        "main_control_input_sha256": main_fingerprint_at_start,
        "fresh_factorial_input_sha256": fresh_fingerprint_at_start,
        "teacher": teacher_artifact,
        "runtime_contract": actual_runtime,
        "language_codes": list(LANGUAGE_CODES),
    }
    result = {
        "experiment": EXPERIMENT_NAME,
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "Does the frozen student's held-out collapse come primarily from Edge "
            "voice familiarity rather than text familiarity or broader optimization?"
        ),
        "identity": identity,
        "experiment_run_id": canonical_sha256(identity),
        "setup": {
            "fresh_factorial_clips": len(fresh_records),
            "exact_control_clips": len(exact_control_records(main_controls, args.manifest)),
            "requests": len(requests),
            "prefixes_seconds": [0.5, 1, 2, 4, "full"],
            "seen_text_indices": list(SEEN_TEXT_INDICES),
            "unseen_text_indices": list(UNSEEN_TEXT_INDICES),
            "voices": EDGE_VOICES,
            "synthesis": synthesis,
            "cache_manifest": str(args.cache_dir / "manifest.json"),
            "teacher_batch_size": args.teacher_batch_size,
            "torch_threads": args.threads,
            "checkpoint_frozen": True,
            "teacher_frozen": True,
            "policy_frozen": True,
            "provider_service_revision_exposed": False,
        },
        "runtime": {
            **runtime,
            "total_run_wall_seconds": time.perf_counter() - run_started,
            "teacher_rtf": runtime["teacher_inference_wall_seconds"]
            / runtime["audio_seconds"],
            "student_frontend_replay_rtf": runtime[
                "student_frontend_and_replay_wall_seconds"
            ]
            / runtime["audio_seconds"],
        },
        "fresh_factorial": {
            "overall_by_prefix": summaries_by_prefix(fresh),
            "cells": fresh_cells,
            "voice_comparisons": voice_comparisons,
        },
        "exact_controls": exact_controls,
        "decision": {
            "primary_endpoint": "fresh full-clip majority-frame prediction",
            "primary_voice_comparison": primary,
            "exact_training_student": exact_train_student,
            "gates": decision_gates,
            "verdict": verdict,
            "reason": verdict_reason,
            "interpretation_if_rejected": (
                "prioritize optimization/target diagnosis over a voice-diversification "
                "retrain when the exact familiar-audio ceiling or paired voice gates fail"
            ),
        },
        "requests": requests,
        "validation": {
            "same_21_main_heldout_clips_reproduced": True,
            "main_protocol_exact_match": main_reproduction["exact_match"],
            "source_snapshot_unchanged": True,
            "reference_files_unchanged": True,
            "selected_main_inputs_unchanged": True,
            "fresh_cache_unchanged": True,
            "all_probabilities_finite": True,
            "teacher_probabilities_normalized": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(".json.pending")
    temporary_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_output, args.output)
    print(
        f"wrote {args.output}; verdict={verdict}; "
        f"student voice loss={primary['student_unseen_voice_loss_pp']:.2f} pp",
        flush=True,
    )


if __name__ == "__main__":
    main()
