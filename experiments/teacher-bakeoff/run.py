#!/usr/bin/env python3
"""Compare offline LID teachers on the main pipeline's held-out audio.

Run from the repository root with::

    UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
      uv run --with transformers==4.44.2 \
      python experiments/teacher-bakeoff/run.py

The script writes ``results.json`` next to itself after each model so an
interrupted multi-model run can be resumed with ``--models``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import resource
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.teacher_targets import extract_window  # noqa: E402
from streaming_lid.audio import feature_frame_count, load_audio  # noqa: E402
from streaming_lid.config import (  # noqa: E402
    HOP_LENGTH,
    LANGUAGE_CODES,
    SAMPLE_RATE,
    TEACHER_FUTURE_MS,
    TEACHER_HOP_FRAMES,
    TEACHER_NAME,
    WIN_LENGTH,
)
from streaming_lid.data import read_manifest, resolve_audio_path  # noqa: E402


MODEL_IDS = {
    "ecapa": TEACHER_NAME,
    "whisper-small": "openai/whisper-small",
    "mms-lid-126": "facebook/mms-lid-126",
}
WINDOW_SECONDS = (1, 2, 4)
MMS_TO_REPO_CODE = {
    "eng": "en",
    "hin": "hi",
    "mar": "mr",
    "ben": "bn",
    "tam": "ta",
    "tel": "te",
    "guj": "gu",
}


@dataclass(frozen=True)
class Prediction:
    code: str
    hi_probability: float
    en_probability: float


class Predictor(Protocol):
    model_id: str
    revision: str | None
    parameter_count: int

    def predict(self, waveform: torch.Tensor) -> Prediction: ...


class ECAPAPredictor:
    def __init__(self) -> None:
        from speechbrain.inference.classifiers import EncoderClassifier

        self.model_id = MODEL_IDS["ecapa"]
        savedir = REPO_ROOT / ".cache/models/lang-id-voxlingua107-ecapa"
        warnings.filterwarnings("ignore", message=".*custom_fwd.*deprecated.*")
        self.model = EncoderClassifier.from_hparams(
            source=self.model_id,
            savedir=str(savedir),
            run_opts={"device": "cpu"},
        )
        self.model.eval()
        self.model.hparams.label_encoder.ignore_len()
        self.index_to_label = self.model.hparams.label_encoder.ind2lab
        self.code_to_index = {
            label.split(":", 1)[0]: index
            for index, label in self.index_to_label.items()
        }
        self.parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        # SpeechBrain's cached snapshot revision is fixed by the main pipeline.
        self.revision = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"

    def predict(self, waveform: torch.Tensor) -> Prediction:
        with torch.inference_mode():
            log_probabilities, _, _, _ = self.model.classify_batch(
                waveform.unsqueeze(0)
            )
        log_probabilities = log_probabilities.squeeze(0)
        probabilities = log_probabilities.exp()
        predicted_index = int(log_probabilities.argmax())
        code = self.index_to_label[predicted_index].split(":", 1)[0]
        return Prediction(
            code=code,
            hi_probability=float(probabilities[self.code_to_index["hi"]]),
            en_probability=float(probabilities[self.code_to_index["en"]]),
        )


class WhisperPredictor:
    def __init__(self) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.model_id = MODEL_IDS["whisper-small"]
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)
        self.model.eval()
        language_to_id = self.model.generation_config.lang_to_id
        self.language_ids = list(language_to_id.values())
        self.id_to_code = {
            token_id: token.removeprefix("<|").removesuffix("|>")
            for token, token_id in language_to_id.items()
        }
        self.language_position = {
            token_id: position
            for position, token_id in enumerate(self.language_ids)
        }
        self.code_to_id = {code: token_id for token_id, code in self.id_to_code.items()}
        missing = set(LANGUAGE_CODES) - set(self.code_to_id)
        if missing:
            raise ValueError(f"Whisper is missing requested languages: {sorted(missing)}")
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def predict(self, waveform: torch.Tensor) -> Prediction:
        inputs = self.processor(
            waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        decoder_input_ids = torch.tensor(
            [[self.model.config.decoder_start_token_id]], dtype=torch.long
        )
        with torch.inference_mode():
            token_logits = self.model(
                input_features=inputs.input_features,
                decoder_input_ids=decoder_input_ids,
            ).logits[0, 0]
        language_probabilities = torch.softmax(token_logits[self.language_ids], dim=-1)
        winning_position = int(language_probabilities.argmax())
        winning_id = self.language_ids[winning_position]
        return Prediction(
            code=self.id_to_code[winning_id],
            hi_probability=float(
                language_probabilities[
                    self.language_position[self.code_to_id["hi"]]
                ]
            ),
            en_probability=float(
                language_probabilities[
                    self.language_position[self.code_to_id["en"]]
                ]
            ),
        )


class MMSPredictor:
    def __init__(self) -> None:
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        self.model_id = MODEL_IDS["mms-lid-126"]
        self.processor = AutoFeatureExtractor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_id)
        self.model.eval()
        self.index_to_mms_code = {
            int(index): code for index, code in self.model.config.id2label.items()
        }
        self.mms_code_to_index = {
            code: index for index, code in self.index_to_mms_code.items()
        }
        missing = set(MMS_TO_REPO_CODE) - set(self.mms_code_to_index)
        if missing:
            raise ValueError(f"MMS is missing requested languages: {sorted(missing)}")
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def predict(self, waveform: torch.Tensor) -> Prediction:
        inputs = self.processor(
            waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        predicted_mms_code = self.index_to_mms_code[int(logits.argmax())]
        code = MMS_TO_REPO_CODE.get(predicted_mms_code, predicted_mms_code)
        return Prediction(
            code=code,
            hi_probability=float(probabilities[self.mms_code_to_index["hin"]]),
            en_probability=float(probabilities[self.mms_code_to_index["eng"]]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/manifest.jsonl"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_IDS),
        default=list(MODEL_IDS),
        help="Models to run; unselected existing results are retained.",
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard any existing model results before this run.",
    )
    return parser.parse_args()


def centered_crop(waveform: torch.Tensor, seconds: int) -> torch.Tensor:
    samples = seconds * SAMPLE_RATE
    if len(waveform) < samples:
        deficit = samples - len(waveform)
        return F.pad(waveform, (deficit // 2, deficit - deficit // 2))
    start = (len(waveform) - samples) // 2
    return waveform[start : start + samples].contiguous()


def telephony_bandlimit(waveform: torch.Tensor) -> torch.Tensor:
    """Apply an 8 kHz sampling bottleneck, then return model-rate audio."""
    narrowband = torchaudio.functional.resample(waveform, SAMPLE_RATE, 8_000)
    restored = torchaudio.functional.resample(narrowband, 8_000, SAMPLE_RATE)
    if len(restored) != len(waveform):
        restored = restored[: len(waveform)]
        if len(restored) < len(waveform):
            restored = F.pad(restored, (0, len(waveform) - len(restored)))
    return restored.contiguous()


def process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def build_eval_cases(heldout: list[dict], manifest: Path) -> list[dict]:
    cases = []
    for item in heldout:
        waveform = load_audio(resolve_audio_path(item, manifest))
        for seconds in WINDOW_SECONDS:
            clean = centered_crop(waveform, seconds)
            for bandwidth, audio in (
                ("clean_16khz", clean),
                ("roundtrip_8khz", telephony_bandlimit(clean)),
            ):
                cases.append(
                    {
                        "id": item["id"],
                        "expected": item["language"],
                        "window_seconds": seconds,
                        "bandwidth": bandwidth,
                        "waveform": audio,
                    }
                )
    return cases


def build_switch_cases(item: dict, manifest: Path) -> list[dict]:
    waveform = load_audio(resolve_audio_path(item, manifest))
    num_frames = feature_frame_count(len(waveform))
    anchor_frames = np.arange(0, num_frames, TEACHER_HOP_FRAMES, dtype=np.int64)
    if anchor_frames[-1] != num_frames - 1:
        anchor_frames = np.append(anchor_frames, num_frames - 1)
    cases = []
    for frame in anchor_frames:
        anchor_seconds = (int(frame) * HOP_LENGTH + WIN_LENGTH) / SAMPLE_RATE
        cases.append(
            {
                "anchor_frame": int(frame),
                "anchor_seconds": anchor_seconds,
                "waveform": extract_window(waveform, int(frame)),
            }
        )
    return cases


def summarize_accuracy(predictions: list[dict]) -> dict:
    by_condition: dict[str, dict[str, dict]] = {}
    for bandwidth in ("clean_16khz", "roundtrip_8khz"):
        by_condition[bandwidth] = {}
        for seconds in WINDOW_SECONDS:
            selected = [
                record
                for record in predictions
                if record["bandwidth"] == bandwidth
                and record["window_seconds"] == seconds
            ]
            correct = sum(r["predicted"] == r["expected"] for r in selected)
            by_condition[bandwidth][str(seconds)] = {
                "correct": correct,
                "total": len(selected),
                "accuracy": correct / len(selected),
            }
    return by_condition


def switch_summary(predictions: list[dict], true_switch_seconds: float) -> dict:
    source_code = "hi"
    target_code = "en"
    run_length = 3
    armed = False
    transition_start = None
    transition_confirmed = None
    for index in range(len(predictions) - run_length + 1):
        run = [record["predicted"] for record in predictions[index : index + run_length]]
        if run == [source_code] * run_length:
            armed = True
        if armed and run == [target_code] * run_length:
            transition_start = predictions[index]["anchor_seconds"]
            transition_confirmed = predictions[index + run_length - 1]["anchor_seconds"]
            break

    def lag_ms(value: float | None) -> float | None:
        return None if value is None else 1_000 * (value - true_switch_seconds)

    expected_codes = [
        source_code if record["anchor_seconds"] < true_switch_seconds else target_code
        for record in predictions
    ]
    correct = sum(
        record["predicted"] == expected
        for record, expected in zip(predictions, expected_codes, strict=True)
    )
    availability_offset_seconds = TEACHER_FUTURE_MS / 1_000
    available_transition = (
        None
        if transition_start is None
        else transition_start + availability_offset_seconds
    )
    return {
        "true_switch_seconds": true_switch_seconds,
        "persistence_anchors": run_length,
        "anchor_hop_ms": TEACHER_HOP_FRAMES * HOP_LENGTH * 1_000 / SAMPLE_RATE,
        "first_persistent_target_seconds": transition_start,
        "first_persistent_target_lag_ms": lag_ms(transition_start),
        "confirmed_target_seconds": transition_confirmed,
        "confirmed_target_lag_ms": lag_ms(transition_confirmed),
        "right_context_ms": TEACHER_FUTURE_MS,
        "first_persistent_target_available_seconds": available_transition,
        "availability_adjusted_lag_ms": lag_ms(available_transition),
        "anchor_label_accuracy": correct / len(predictions),
        "anchor_correct": correct,
        "anchor_total": len(predictions),
    }


def make_predictor(name: str) -> Predictor:
    if name == "ecapa":
        return ECAPAPredictor()
    if name == "whisper-small":
        return WhisperPredictor()
    if name == "mms-lid-126":
        return MMSPredictor()
    raise ValueError(name)


def evaluate_model(
    name: str,
    eval_cases: list[dict],
    switch_cases: list[dict],
    true_switch_seconds: float,
) -> dict:
    load_start = time.perf_counter()
    predictor = make_predictor(name)
    load_wall_seconds = time.perf_counter() - load_start

    # Warm-up is excluded from timings and prevents first-kernel setup from
    # dominating this tiny evaluation.
    predictor.predict(eval_cases[0]["waveform"])
    wall_start = time.perf_counter()
    cpu_start = process_cpu_seconds()
    predictions = []
    for case in eval_cases:
        result = predictor.predict(case["waveform"])
        predictions.append(
            {
                "id": case["id"],
                "expected": case["expected"],
                "predicted": result.code,
                "window_seconds": case["window_seconds"],
                "bandwidth": case["bandwidth"],
                "hi_probability": result.hi_probability,
                "en_probability": result.en_probability,
            }
        )
    eval_wall_seconds = time.perf_counter() - wall_start
    eval_cpu_seconds = process_cpu_seconds() - cpu_start

    switch_wall_start = time.perf_counter()
    switch_cpu_start = process_cpu_seconds()
    switch_predictions = []
    for case in switch_cases:
        result = predictor.predict(case["waveform"])
        switch_predictions.append(
            {
                "anchor_frame": case["anchor_frame"],
                "anchor_seconds": case["anchor_seconds"],
                "predicted": result.code,
                "hi_probability": result.hi_probability,
                "en_probability": result.en_probability,
            }
        )
    switch_wall_seconds = time.perf_counter() - switch_wall_start
    switch_cpu_seconds = process_cpu_seconds() - switch_cpu_start

    total_audio_seconds = sum(
        len(case["waveform"]) / SAMPLE_RATE for case in eval_cases
    ) + sum(len(case["waveform"]) / SAMPLE_RATE for case in switch_cases)
    total_wall_seconds = eval_wall_seconds + switch_wall_seconds
    total_cpu_seconds = eval_cpu_seconds + switch_cpu_seconds
    output = {
        "model_id": predictor.model_id,
        "revision": predictor.revision,
        "parameter_count": predictor.parameter_count,
        "accuracy": summarize_accuracy(predictions),
        "switch": switch_summary(switch_predictions, true_switch_seconds),
        "timing": {
            "load_wall_seconds": load_wall_seconds,
            "timed_requests": len(eval_cases) + len(switch_cases),
            "timed_audio_seconds": total_audio_seconds,
            "wall_seconds": total_wall_seconds,
            "process_cpu_seconds": total_cpu_seconds,
            "wall_rtf": total_wall_seconds / total_audio_seconds,
            "mean_wall_ms_per_request": 1_000
            * total_wall_seconds
            / (len(eval_cases) + len(switch_cases)),
            "includes_preprocessing": True,
            "includes_model_load": False,
        },
        "heldout_predictions": predictions,
        "switch_predictions": switch_predictions,
    }
    del predictor
    gc.collect()
    return output


def main() -> None:
    args = parse_args()
    if args.threads != 6:
        raise ValueError("TEAM.md requires torch.set_num_threads(6)")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)

    manifest = args.manifest.resolve()
    records = read_manifest(manifest)
    heldout = [record for record in records if record["split"] == "heldout"]
    switch_item = next(
        record for record in records if record["id"] == "switch_hi_en_eval"
    )
    heldout_counts = Counter(record["language"] for record in heldout)
    if set(heldout_counts) != set(LANGUAGE_CODES) or len(set(heldout_counts.values())) != 1:
        raise ValueError(
            "held-out manifest must contain an equal non-zero count for every "
            f"configured language, got {dict(heldout_counts)}"
        )
    eval_cases = build_eval_cases(heldout, manifest)
    switch_cases = build_switch_cases(switch_item, manifest)
    true_switch_seconds = float(switch_item["segments"][0]["end_seconds"])

    output_path = Path(__file__).with_name("results.json")
    if output_path.exists() and not args.fresh:
        output = json.loads(output_path.read_text())
    else:
        output = {"models": {}}
    output.update(
        {
            "experiment": "teacher-bakeoff",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest.relative_to(REPO_ROOT)),
            "heldout_ids": [record["id"] for record in heldout],
            "switch_id": switch_item["id"],
            "language_codes": list(LANGUAGE_CODES),
            "window_seconds": list(WINDOW_SECONDS),
            "crop": "single centred crop per held-out clip",
            "telephony_condition": "16 kHz -> 8 kHz -> 16 kHz resampling; no codec",
            "switch_window": {
                "seconds": 2.0,
                "past_ms": 1_750,
                "future_ms": TEACHER_FUTURE_MS,
                "hop_ms": TEACHER_HOP_FRAMES
                * HOP_LENGTH
                * 1_000
                / SAMPLE_RATE,
                "extractor": "scripts.teacher_targets.extract_window",
            },
            "cpu_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
        }
    )

    for model_number, name in enumerate(args.models, start=1):
        print(
            f"[{model_number}/{len(args.models)}] loading and evaluating {MODEL_IDS[name]}",
            flush=True,
        )
        output["models"][name] = evaluate_model(
            name, eval_cases, switch_cases, true_switch_seconds
        )
        output["updated_utc"] = datetime.now(timezone.utc).isoformat()
        if not all(
            math.isfinite(value)
            for value in (
                output["models"][name]["timing"]["wall_seconds"],
                output["models"][name]["timing"]["process_cpu_seconds"],
            )
        ):
            raise ValueError(f"non-finite timing for {name}")
        output_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        model_result = output["models"][name]
        clean = model_result["accuracy"]["clean_16khz"]
        print(
            f"{name}: clean accuracy 1/2/4 s="
            f"{clean['1']['accuracy']:.3f}/"
            f"{clean['2']['accuracy']:.3f}/"
            f"{clean['4']['accuracy']:.3f}; "
            f"switch lag={model_result['switch']['first_persistent_target_lag_ms']} ms; "
            f"RTF={model_result['timing']['wall_rtf']:.3f}",
            flush=True,
        )

    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
