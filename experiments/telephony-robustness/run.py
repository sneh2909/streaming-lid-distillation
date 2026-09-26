#!/usr/bin/env python3
"""Paired telephony-channel audit for the pinned teacher and submitted student.

The experiment transforms each complete held-out waveform once, then takes
1/2/4 second prefixes.  It keeps the production ECAPA artifact, frontend,
student checkpoint, streaming implementation, manifest, and switch windows.

Run from the repository root with::

    UV_CACHE_DIR=./.cache/uv HF_HOME=./.cache/hf TORCH_HOME=./.cache/torch \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
      experiments/telephony-robustness/run.py --fresh
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import struct
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


PIPELINE_SOURCES = (
    "scripts/teacher_targets.py",
    "scripts/eval.py",
    "src/streaming_lid/audio.py",
    "src/streaming_lid/config.py",
    "src/streaming_lid/data.py",
    "src/streaming_lid/model.py",
    "src/streaming_lid/run_identity.py",
)


def hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def snapshot_pipeline_sources() -> tuple[str, dict[str, dict[str, Any]]]:
    combined = hashlib.sha256()
    files: dict[str, dict[str, Any]] = {}
    for relative in PIPELINE_SOURCES:
        path = REPO_ROOT / relative
        contents = path.read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        files[relative] = {"sha256": digest, "bytes": len(contents)}
        combined.update(relative.encode("utf-8"))
        combined.update(contents)
    return combined.hexdigest(), files


(
    PIPELINE_SOURCE_SHA256_AT_IMPORT,
    PIPELINE_SOURCE_FILES_AT_IMPORT,
) = snapshot_pipeline_sources()

# Import the main pipeline rather than copying its model, frontend, artifact
# resolver, manifest parser, window extractor, or streaming timing helpers.
from scripts.eval import (  # noqa: E402
    chunk_availability_times,
    chunk_posteriors,
    detect_hi_to_en_switch,
    smooth_posteriors,
)
from scripts.teacher_targets import (  # noqa: E402
    extract_window,
    label_indices,
    resolve_teacher_artifact,
)
from streaming_lid.audio import LogMelFrontend, feature_frame_count, load_audio  # noqa: E402
from streaming_lid.config import (  # noqa: E402
    HOP_LENGTH,
    LANGUAGE_CODES,
    SAMPLE_RATE,
    TEACHER_ARTIFACT_SHA256,
    TEACHER_FUTURE_MS,
    TEACHER_HOP_FRAMES,
    TEACHER_NAME,
    TEACHER_REVISION,
    WIN_LENGTH,
)
from streaming_lid.data import (  # noqa: E402
    read_manifest,
    require_speaker_disjoint,
    resolve_audio_path,
)
from streaming_lid.model import CausalLIDStudent  # noqa: E402
from streaming_lid.run_identity import (  # noqa: E402
    corpus_identity,
    model_state_sha256,
    pipeline_configuration,
    run_id_for_identity,
)


EXPERIMENT_NAME = "telephony-robustness"
SCHEMA_VERSION = 1
ANALYSIS_VERSION = "telephony-robustness-v1"
PREFIX_SECONDS = (1, 2, 4)
TELEPHONY_SAMPLE_RATE = 8_000
FIR_TAPS = 129
PASSBAND_LOW_HZ = 300.0
PASSBAND_HIGH_HZ = 3_400.0
FILTER_REFERENCE_HZ = 1_000.0
SWITCH_COLLAR_MS = 250.0
SWITCH_STABILITY_MS = 500.0
KL_EPSILON = 1e-8
GATE_MACRO_F1_DROP_PP = 5.0
GATE_CLASS_RECALL_DROP_PP = 10.0
GATE_MEDIAN_KL = 0.2
GATE_SWITCH_LAG_DELTA_MS = 100.0

CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "clean",
        "description": "native 16 kHz PCM",
        "resample": False,
        "bandpass": False,
        "codec": None,
        "noise_snr_db": None,
    },
    {
        "name": "resample_only",
        "description": "16 -> 8 -> 16 kHz float resampling; no explicit passband or codec",
        "resample": True,
        "bandpass": False,
        "codec": None,
        "noise_snr_db": None,
    },
    {
        "name": "narrowband_pcm",
        "description": "8 kHz, causal 300-3400 Hz FIR, linear PCM16 round trip",
        "resample": True,
        "bandpass": True,
        "codec": "PCM_16",
        "noise_snr_db": None,
    },
    {
        "name": "pcma",
        "description": "8 kHz, causal 300-3400 Hz FIR, G.711 A-law",
        "resample": True,
        "bandpass": True,
        "codec": "ALAW",
        "noise_snr_db": None,
    },
    {
        "name": "pcmu",
        "description": "8 kHz, causal 300-3400 Hz FIR, G.711 mu-law",
        "resample": True,
        "bandpass": True,
        "codec": "ULAW",
        "noise_snr_db": None,
    },
    {
        "name": "noise_20db",
        "description": "native 16 kHz plus deterministic white noise at 20 dB SNR",
        "resample": False,
        "bandpass": False,
        "codec": None,
        "noise_snr_db": 20.0,
    },
    {
        "name": "noise_10db",
        "description": "native 16 kHz plus deterministic white noise at 10 dB SNR",
        "resample": False,
        "bandpass": False,
        "codec": None,
        "noise_snr_db": 10.0,
    },
    {
        "name": "noise_20db_pcma",
        "description": "20 dB white noise before 8 kHz 300-3400 Hz G.711 A-law",
        "resample": True,
        "bandpass": True,
        "codec": "ALAW",
        "noise_snr_db": 20.0,
    },
    {
        "name": "noise_10db_pcma",
        "description": "10 dB white noise before 8 kHz 300-3400 Hz G.711 A-law",
        "resample": True,
        "bandpass": True,
        "codec": "ALAW",
        "noise_snr_db": 10.0,
    },
    {
        "name": "noise_20db_pcmu",
        "description": "20 dB white noise before 8 kHz 300-3400 Hz G.711 mu-law",
        "resample": True,
        "bandpass": True,
        "codec": "ULAW",
        "noise_snr_db": 20.0,
    },
    {
        "name": "noise_10db_pcmu",
        "description": "10 dB white noise before 8 kHz 300-3400 Hz G.711 mu-law",
        "resample": True,
        "bandpass": True,
        "codec": "ULAW",
        "noise_snr_db": 10.0,
    },
)
CONDITION_BY_NAME = {condition["name"]: condition for condition in CONDITIONS}


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
    parser.add_argument(
        "--train-metrics", type=Path, default=Path("results/train_metrics.json")
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".cache/models/lang-id-voxlingua107-ecapa"),
    )
    parser.add_argument("--teacher-batch-size", type=int, default=24)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Replace results.json only after every end-of-run identity check passes.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.threads != 6:
        raise ValueError("team protocol requires exactly six Torch threads")
    if args.teacher_batch_size <= 0:
        raise ValueError("teacher batch size must be positive")
    output = Path(__file__).with_name("results.json")
    if output.exists() and not args.fresh:
        raise FileExistsError(f"{output} exists; pass --fresh to replace it")


def input_fingerprint(records: list[dict], manifest: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(records, key=lambda value: value["id"]):
        digest.update(
            json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        )
        with resolve_audio_path(item, manifest).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def design_bandpass_kernel() -> torch.Tensor:
    if FIR_TAPS % 2 != 1:
        raise AssertionError("linear-phase FIR needs an odd tap count")
    offsets = torch.arange(FIR_TAPS, dtype=torch.float64) - (FIR_TAPS - 1) / 2

    def lowpass(cutoff_hz: float) -> torch.Tensor:
        normalized = cutoff_hz / TELEPHONY_SAMPLE_RATE
        return 2 * normalized * torch.sinc(2 * normalized * offsets)

    window = torch.hann_window(FIR_TAPS, periodic=False, dtype=torch.float64)
    kernel = (lowpass(PASSBAND_HIGH_HZ) - lowpass(PASSBAND_LOW_HZ)) * window
    sample_indices = torch.arange(FIR_TAPS, dtype=torch.float64)
    phase = -2j * math.pi * FILTER_REFERENCE_HZ * sample_indices / TELEPHONY_SAMPLE_RATE
    reference_gain = torch.abs(torch.sum(kernel.to(torch.complex128) * torch.exp(phase)))
    kernel = kernel / reference_gain
    if not torch.equal(kernel, kernel.flip(0)):
        torch.testing.assert_close(kernel, kernel.flip(0), rtol=0, atol=1e-14)
    return kernel.to(torch.float32).contiguous()


BANDPASS_KERNEL = design_bandpass_kernel()


def causal_bandpass(waveform_8khz: torch.Tensor) -> torch.Tensor:
    padded = F.pad(waveform_8khz.reshape(1, 1, -1), (FIR_TAPS - 1, 0))
    kernel = BANDPASS_KERNEL.flip(0).reshape(1, 1, -1)
    return F.conv1d(padded, kernel).reshape(-1).contiguous()


def restore_length(waveform: torch.Tensor, wanted: int) -> torch.Tensor:
    if len(waveform) > wanted:
        waveform = waveform[:wanted]
    elif len(waveform) < wanted:
        waveform = F.pad(waveform, (0, wanted - len(waveform)))
    return waveform.contiguous()


def resample_down(waveform: torch.Tensor) -> torch.Tensor:
    return torchaudio.functional.resample(
        waveform, SAMPLE_RATE, TELEPHONY_SAMPLE_RATE
    ).contiguous()


def resample_up(waveform: torch.Tensor, wanted: int) -> torch.Tensor:
    restored = torchaudio.functional.resample(
        waveform, TELEPHONY_SAMPLE_RATE, SAMPLE_RATE
    )
    return restore_length(restored, wanted)


def float_to_pcm16(waveform: torch.Tensor) -> tuple[np.ndarray, int]:
    nonfinite = int((~torch.isfinite(waveform)).sum())
    if nonfinite:
        raise FloatingPointError("channel transform produced non-finite samples")
    clipped = int(((waveform < -1.0) | (waveform >= 1.0)).sum())
    limited = waveform.clamp(-1.0, 1.0 - 1.0 / 32768.0)
    pcm = torch.round(limited * 32768.0).to(torch.int16).cpu().numpy()
    return pcm, clipped


def pcm16_roundtrip(waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
    pcm, clipped = float_to_pcm16(waveform)
    decoded = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
    return decoded.contiguous(), clipped


def wav_data_bytes(payload: bytes) -> bytes:
    if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError("SoundFile did not emit RIFF/WAVE")
    offset = 12
    while offset + 8 <= len(payload):
        chunk_id = payload[offset : offset + 4]
        size = struct.unpack("<I", payload[offset + 4 : offset + 8])[0]
        start = offset + 8
        stop = start + size
        if stop > len(payload):
            raise ValueError("malformed RIFF chunk")
        if chunk_id == b"data":
            return payload[start:stop]
        offset = stop + (size % 2)
    raise ValueError("WAV payload has no data chunk")


def encode_decode_g711(
    waveform: torch.Tensor, subtype: str
) -> tuple[torch.Tensor, int]:
    if subtype not in {"ALAW", "ULAW"}:
        raise ValueError(f"unsupported G.711 subtype {subtype}")
    pcm, clipped = float_to_pcm16(waveform)
    buffer = io.BytesIO()
    sf.write(
        buffer,
        pcm,
        TELEPHONY_SAMPLE_RATE,
        format="WAV",
        subtype=subtype,
    )
    buffer.seek(0)
    decoded, sample_rate = sf.read(
        buffer, dtype="float32", always_2d=True
    )
    if sample_rate != TELEPHONY_SAMPLE_RATE or decoded.shape != (len(pcm), 1):
        raise AssertionError("G.711 round trip changed sample rate or shape")
    return torch.from_numpy(decoded[:, 0].copy()).contiguous(), clipped


def validate_g711_fixed_vectors() -> dict[str, Any]:
    values = np.asarray(
        [
            -32768,
            -30000,
            -16384,
            -8192,
            -4096,
            -1000,
            -1,
            0,
            1,
            1000,
            4096,
            8192,
            16384,
            30000,
            32767,
        ],
        dtype=np.int16,
    )
    expected = {
        "ULAW": {
            "encoded_hex": "00020f1f2f4e7fffffceaf9f8f8280",
            "decoded_pcm16": [
                -32124,
                -30076,
                -16764,
                -8316,
                -4092,
                -988,
                0,
                0,
                0,
                988,
                4092,
                8316,
                16764,
                30076,
                32124,
            ],
        },
        "ALAW": {
            "encoded_hex": "2a282535057a55d5d5fa85b5a5a8aa",
            "decoded_pcm16": [
                -32256,
                -30208,
                -16896,
                -8448,
                -4224,
                -1008,
                -8,
                8,
                8,
                1008,
                4224,
                8448,
                16896,
                30208,
                32256,
            ],
        },
    }
    results: dict[str, Any] = {}
    for subtype, reference in expected.items():
        buffer = io.BytesIO()
        sf.write(
            buffer,
            values,
            TELEPHONY_SAMPLE_RATE,
            format="WAV",
            subtype=subtype,
        )
        payload = buffer.getvalue()
        encoded_hex = wav_data_bytes(payload).hex()
        buffer.seek(0)
        decoded, rate = sf.read(buffer, dtype="int16", always_2d=False)
        decoded_values = decoded.tolist()
        encoded_matches = encoded_hex == reference["encoded_hex"]
        decoded_matches = decoded_values == reference["decoded_pcm16"]
        if rate != TELEPHONY_SAMPLE_RATE or not encoded_matches or not decoded_matches:
            raise AssertionError(
                f"libsndfile {subtype} failed the fixed-vector validation"
            )
        results[subtype] = {
            "input_pcm16": values.tolist(),
            "expected_encoded_hex": reference["encoded_hex"],
            "actual_encoded_hex": encoded_hex,
            "expected_decoded_pcm16": reference["decoded_pcm16"],
            "actual_decoded_pcm16": decoded_values,
            "encoded_matches": encoded_matches,
            "decoded_matches": decoded_matches,
        }
    return {
        "soundfile_version": sf.__version__,
        "libsndfile_version": sf.__libsndfile_version__,
        "sample_rate": TELEPHONY_SAMPLE_RATE,
        "vectors": results,
        "all_passed": True,
    }


def filter_response_db(kernel: torch.Tensor, frequency_hz: float) -> float:
    indices = torch.arange(len(kernel), dtype=torch.float64)
    phase = -2j * math.pi * frequency_hz * indices / TELEPHONY_SAMPLE_RATE
    response = torch.sum(kernel.to(torch.complex128) * torch.exp(phase))
    return float(20 * torch.log10(torch.abs(response).clamp_min(1e-12)))


def validate_filter() -> dict[str, Any]:
    frequency_response = {
        str(frequency): filter_response_db(BANDPASS_KERNEL, float(frequency))
        for frequency in (0, 100, 300, 1_000, 3_400, 3_800, 4_000)
    }
    if abs(frequency_response["1000"]) > 1e-4:
        raise AssertionError("FIR reference-frequency gain is not unity")
    if frequency_response["100"] > -35 or frequency_response["3800"] > -35:
        raise AssertionError("FIR stopband attenuation sanity check failed")

    impulse = torch.zeros(32_000, dtype=torch.float32)
    impulse_index = 8_000
    impulse[impulse_index] = 0.8
    down = resample_down(impulse)
    resampled = resample_up(down, len(impulse))
    bandpassed = resample_up(causal_bandpass(down), len(impulse))
    resample_peak = int(resampled.abs().argmax())
    bandpass_peak = int(bandpassed.abs().argmax())
    resample_delay_samples = resample_peak - impulse_index
    incremental_delay_samples = bandpass_peak - resample_peak
    measured_delay_ms = 1_000 * incremental_delay_samples / SAMPLE_RATE
    theoretical_delay_ms = (
        1_000 * ((FIR_TAPS - 1) / 2) / TELEPHONY_SAMPLE_RATE
    )
    if abs(measured_delay_ms - theoretical_delay_ms) > 1_000 / SAMPLE_RATE:
        raise AssertionError(
            "measured passband-filter delay differs by more than one model-rate sample"
        )
    return {
        "design": "causal odd-length windowed-sinc linear-phase FIR",
        "sample_rate": TELEPHONY_SAMPLE_RATE,
        "taps": FIR_TAPS,
        "passband_hz": [PASSBAND_LOW_HZ, PASSBAND_HIGH_HZ],
        "reference_gain_hz": FILTER_REFERENCE_HZ,
        "frequency_response_db": frequency_response,
        "resample_roundtrip_impulse_delay_model_samples": resample_delay_samples,
        "bandpass_incremental_impulse_delay_model_samples": incremental_delay_samples,
        "measured_bandpass_delay_ms": measured_delay_ms,
        "theoretical_bandpass_delay_ms": theoretical_delay_ms,
        "delay_validation_passed": True,
    }


def deterministic_noise(waveform: torch.Tensor, clip_id: str) -> torch.Tensor:
    seed_bytes = hashlib.sha256(
        f"{ANALYSIS_VERSION}:white-noise:{clip_id}".encode("utf-8")
    ).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big") % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(len(waveform), generator=generator, dtype=waveform.dtype)
    return noise / noise.square().mean().sqrt().clamp_min(1e-12)


def add_noise(
    waveform: torch.Tensor, clip_id: str, snr_db: float
) -> tuple[torch.Tensor, float]:
    signal_rms = waveform.square().mean().sqrt()
    if signal_rms <= 0:
        raise ValueError(f"cannot set SNR for silent clip {clip_id}")
    unit_noise = deterministic_noise(waveform, clip_id)
    target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = unit_noise * target_noise_rms
    mixed = waveform + noise
    achieved = float(
        20
        * torch.log10(
            signal_rms / (mixed.sub(waveform).square().mean().sqrt())
        )
    )
    return mixed.contiguous(), achieved


def transform_waveform(
    waveform: torch.Tensor, clip_id: str, condition: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    started = time.perf_counter()
    transformed = waveform.clone()
    achieved_snr = None
    if condition["noise_snr_db"] is not None:
        transformed, achieved_snr = add_noise(
            transformed, clip_id, float(condition["noise_snr_db"])
        )

    clipped_pre_codec = 0
    if condition["resample"]:
        narrow = resample_down(transformed)
        if condition["bandpass"]:
            narrow = causal_bandpass(narrow)
        codec = condition["codec"]
        if codec == "PCM_16":
            narrow, clipped_pre_codec = pcm16_roundtrip(narrow)
        elif codec in {"ALAW", "ULAW"}:
            narrow, clipped_pre_codec = encode_decode_g711(narrow, codec)
        elif codec is not None:
            raise ValueError(f"unsupported codec {codec}")
        transformed = resample_up(narrow, len(waveform))

    if len(transformed) != len(waveform):
        raise AssertionError("whole-clip transform changed model-rate sample count")
    if not torch.isfinite(transformed).all():
        raise FloatingPointError("whole-clip transform produced non-finite samples")
    delay_ms = (
        1_000 * ((FIR_TAPS - 1) / 2) / TELEPHONY_SAMPLE_RATE
        if condition["bandpass"]
        else 0.0
    )
    diagnostics = {
        "input_samples": len(waveform),
        "output_samples": len(transformed),
        "input_peak": float(waveform.abs().max()),
        "output_peak": float(transformed.abs().max()),
        "clipped_pre_codec_samples": clipped_pre_codec,
        "requested_snr_db": condition["noise_snr_db"],
        "achieved_snr_db": achieved_snr,
        "filter_delay_ms": delay_ms,
        "peak_normalization_applied": False,
        "wall_seconds": time.perf_counter() - started,
    }
    return transformed.contiguous(), diagnostics


def prefix_audio(waveform: torch.Tensor, seconds: int) -> tuple[torch.Tensor, bool]:
    wanted = seconds * SAMPLE_RATE
    padded = len(waveform) < wanted
    return restore_length(waveform[:wanted], wanted), padded


def load_teacher(model_dir: Path):
    from speechbrain.inference.classifiers import EncoderClassifier

    snapshot, artifact = resolve_teacher_artifact()
    pinned_savedir = (
        model_dir.resolve() / f"{TEACHER_REVISION}-{TEACHER_ARTIFACT_SHA256[:12]}"
    )
    warnings.filterwarnings("ignore", message=".*custom_fwd.*deprecated.*")
    teacher = EncoderClassifier.from_hparams(
        source=str(snapshot),
        savedir=str(pinned_savedir),
        overrides={"pretrained_path": str(snapshot)},
        run_opts={"device": "cpu"},
    )
    teacher.eval()
    teacher.hparams.label_encoder.ignore_len()
    indices = label_indices(teacher)
    return teacher, indices, artifact


def teacher_posteriors(
    teacher,
    selected_indices: list[int],
    waveforms: list[torch.Tensor],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    selected_masses = []
    with torch.inference_mode():
        for start in range(0, len(waveforms), batch_size):
            batch_waveforms = waveforms[start : start + batch_size]
            lengths = torch.tensor([len(value) for value in batch_waveforms])
            padded = torch.nn.utils.rnn.pad_sequence(
                batch_waveforms, batch_first=True
            )
            relative_lengths = lengths / padded.shape[1]
            log_probabilities, _, _, _ = teacher.classify_batch(
                padded, relative_lengths
            )
            chosen = log_probabilities[:, selected_indices]
            probabilities.append(torch.softmax(chosen, dim=-1).cpu())
            selected_masses.append(chosen.exp().sum(dim=-1).cpu())
    posterior = torch.cat(probabilities).numpy().astype(np.float32)
    masses = torch.cat(selected_masses).numpy().astype(np.float32)
    if posterior.shape != (len(waveforms), len(LANGUAGE_CODES)):
        raise AssertionError("teacher returned an unexpected posterior shape")
    if not np.isfinite(posterior).all() or not np.isfinite(masses).all():
        raise FloatingPointError("teacher returned non-finite outputs")
    if not np.allclose(posterior.sum(axis=-1), 1.0, atol=1e-6):
        raise AssertionError("restricted teacher posterior is not normalized")
    return posterior, masses


def student_posterior(
    model: CausalLIDStudent,
    frontend: LogMelFrontend,
    waveform: torch.Tensor,
    *,
    chunk_frames: int,
    label_delay_frames: int,
) -> tuple[np.ndarray, int]:
    with torch.inference_mode():
        features = frontend(waveform)
        logits = model.streaming_forward(features, chunk_frames=chunk_frames).squeeze(0)
        valid = logits[label_delay_frames:]
        if len(valid) == 0:
            raise ValueError("prefix supplies no delay-aligned stable student frames")
        frame_probabilities = torch.softmax(valid, dim=-1)
        posterior = frame_probabilities.mean(dim=0).cpu().numpy().astype(np.float32)
    posterior /= posterior.sum()
    return posterior, len(valid)


def macro_f1(
    expected: list[int], predicted: list[int], num_classes: int
) -> tuple[float, dict[str, float]]:
    per_class: dict[str, float] = {}
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
        per_class[language] = 0.0 if denominator == 0 else 2 * true_positive / denominator
    if len(per_class) != num_classes:
        raise AssertionError("macro-F1 class count mismatch")
    return float(np.mean(list(per_class.values()))), per_class


def classification_summary(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    expected = [LANGUAGE_CODES.index(row["expected"]) for row in rows]
    predicted = [LANGUAGE_CODES.index(row[model_name]["prediction"]) for row in rows]
    accuracy = float(np.mean(np.asarray(expected) == np.asarray(predicted)))
    f1, per_class_f1 = macro_f1(expected, predicted, len(LANGUAGE_CODES))
    recalls = {}
    counts = {}
    confusion = {language: Counter() for language in LANGUAGE_CODES}
    for truth, guess in zip(expected, predicted, strict=True):
        truth_code = LANGUAGE_CODES[truth]
        guess_code = LANGUAGE_CODES[guess]
        confusion[truth_code][guess_code] += 1
    for index, language in enumerate(LANGUAGE_CODES):
        selected = [guess for truth, guess in zip(expected, predicted, strict=True) if truth == index]
        if not selected:
            raise AssertionError(f"no examples for class {language}")
        recalls[language] = sum(guess == index for guess in selected) / len(selected)
        counts[language] = len(selected)
    return {
        "n_requests": len(rows),
        "accuracy": accuracy,
        "macro_f1": f1,
        "per_language_recall": recalls,
        "per_language_f1": per_class_f1,
        "support_per_language": counts,
        "confusion": {
            truth: {guess: int(count) for guess, count in sorted(values.items())}
            for truth, values in confusion.items()
        },
        "mean_max_probability": float(
            np.mean([row[model_name]["max_probability"] for row in rows])
        ),
    }


def conditional_kl(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.clip(reference.astype(np.float64), KL_EPSILON, 1.0)
    right = np.clip(candidate.astype(np.float64), KL_EPSILON, 1.0)
    return float(np.sum(left * (np.log(left) - np.log(right))))


def summarize_conditions(
    rows: list[dict[str, Any]], model_name: str
) -> dict[str, Any]:
    clean = {
        (row["clip_id"], row["prefix_seconds"]): row
        for row in rows
        if row["condition"] == "clean"
    }
    summaries: dict[str, Any] = {}
    clean_aggregate = classification_summary(
        [row for row in rows if row["condition"] == "clean"], model_name
    )
    for condition in CONDITIONS:
        name = condition["name"]
        selected = [row for row in rows if row["condition"] == name]
        aggregate = classification_summary(selected, model_name)
        by_prefix = {
            str(seconds): classification_summary(
                [row for row in selected if row["prefix_seconds"] == seconds],
                model_name,
            )
            for seconds in PREFIX_SECONDS
        }
        divergences = []
        flips = 0
        correct_to_wrong = 0
        wrong_to_correct = 0
        confidence_deltas = []
        for row in selected:
            key = (row["clip_id"], row["prefix_seconds"])
            reference = clean[key]
            reference_probs = np.asarray(
                reference[model_name]["probabilities"], dtype=np.float64
            )
            candidate_probs = np.asarray(
                row[model_name]["probabilities"], dtype=np.float64
            )
            divergences.append(conditional_kl(reference_probs, candidate_probs))
            reference_prediction = reference[model_name]["prediction"]
            candidate_prediction = row[model_name]["prediction"]
            flips += int(reference_prediction != candidate_prediction)
            clean_correct = reference_prediction == row["expected"]
            candidate_correct = candidate_prediction == row["expected"]
            correct_to_wrong += int(clean_correct and not candidate_correct)
            wrong_to_correct += int(not clean_correct and candidate_correct)
            confidence_deltas.append(
                row[model_name]["max_probability"]
                - reference[model_name]["max_probability"]
            )
        recall_drop = {
            language: 100
            * (
                clean_aggregate["per_language_recall"][language]
                - aggregate["per_language_recall"][language]
            )
            for language in LANGUAGE_CODES
        }
        macro_f1_drop_pp = 100 * (
            clean_aggregate["macro_f1"] - aggregate["macro_f1"]
        )
        median_kl = float(np.median(divergences))
        p95_kl = float(np.percentile(divergences, 95))
        monolingual_gate = {
            "macro_f1_drop_le_5pp": macro_f1_drop_pp <= GATE_MACRO_F1_DROP_PP,
            "worst_class_recall_drop_le_10pp": max(recall_drop.values())
            <= GATE_CLASS_RECALL_DROP_PP,
            "median_conditional_kl_le_0_2": median_kl <= GATE_MEDIAN_KL,
        }
        monolingual_gate["passed"] = all(monolingual_gate.values())
        summaries[name] = {
            "condition": condition,
            "aggregate_1_2_4s": aggregate,
            "by_prefix_seconds": by_prefix,
            "macro_f1_drop_pp_vs_clean": macro_f1_drop_pp,
            "per_language_recall_drop_pp_vs_clean": recall_drop,
            "worst_language_recall_drop_pp_vs_clean": max(recall_drop.values()),
            "conditional_kl_clean_to_condition": {
                "median": median_kl,
                "p95": p95_kl,
                "maximum": float(np.max(divergences)),
            },
            "paired_prediction_flips_vs_clean": flips,
            "paired_correct_to_wrong_vs_clean": correct_to_wrong,
            "paired_wrong_to_correct_vs_clean": wrong_to_correct,
            "mean_max_probability_delta_vs_clean": float(
                np.mean(confidence_deltas)
            ),
            "monolingual_gate": monolingual_gate,
        }
    return summaries


def expected_switch_class(
    item: dict, times: np.ndarray, boundary_seconds: float
) -> np.ndarray:
    source = LANGUAGE_CODES.index(item["segments"][0]["language"])
    target = LANGUAGE_CODES.index(item["segments"][1]["language"])
    return np.where(times < boundary_seconds, source, target).astype(np.int64)


def no_collar_accuracy(
    classes: np.ndarray,
    times: np.ndarray,
    item: dict,
    boundary_seconds: float,
) -> dict[str, Any]:
    keep = np.abs(times - boundary_seconds) > SWITCH_COLLAR_MS / 1_000
    expected = expected_switch_class(item, times, boundary_seconds)
    if not keep.any():
        raise AssertionError("switch collar removed every frame")
    return {
        "correct": int((classes[keep] == expected[keep]).sum()),
        "frames": int(keep.sum()),
        "accuracy": float(np.mean(classes[keep] == expected[keep])),
    }


def stable_run(
    classes: np.ndarray,
    times: np.ndarray,
    wanted: int,
    *,
    not_before: float,
) -> tuple[int, int] | None:
    horizon = SWITCH_STABILITY_MS / 1_000
    for start in range(len(classes)):
        if times[start] < not_before or classes[start] != wanted:
            continue
        confirmation = int(np.searchsorted(times, times[start] + horizon, side="left"))
        if confirmation >= len(classes):
            continue
        if np.all(classes[start : confirmation + 1] == wanted):
            return start, confirmation
    return None


def stable_switch_event(
    classes: np.ndarray,
    times: np.ndarray,
    *,
    source_index: int,
    target_index: int,
    boundary_seconds: float,
) -> dict[str, Any]:
    source_armed = False
    for start in range(len(classes)):
        if classes[start] != source_index:
            continue
        confirmation = int(
            np.searchsorted(
                times,
                times[start] + SWITCH_STABILITY_MS / 1_000,
                side="left",
            )
        )
        if confirmation >= len(classes):
            continue
        if times[confirmation] <= boundary_seconds and np.all(
            classes[start : confirmation + 1] == source_index
        ):
            source_armed = True
            break
    target_run = stable_run(
        classes,
        times,
        target_index,
        not_before=boundary_seconds,
    )
    if target_run is None:
        return {
            "source_armed_before_boundary": source_armed,
            "detected": False,
            "first_stable_seconds": None,
            "confirmed_seconds": None,
            "first_stable_lag_ms": None,
            "confirmed_lag_ms": None,
        }
    start, confirmation = target_run
    return {
        "source_armed_before_boundary": source_armed,
        "detected": True,
        "first_stable_seconds": float(times[start]),
        "confirmed_seconds": float(times[confirmation]),
        "first_stable_lag_ms": float(1_000 * (times[start] - boundary_seconds)),
        "confirmed_lag_ms": float(
            1_000 * (times[confirmation] - boundary_seconds)
        ),
    }


def detect_pair_policy(
    probabilities: np.ndarray,
    availability_seconds: np.ndarray,
    *,
    source_index: int,
    target_index: int,
    boundary_seconds: float,
) -> dict[str, Any]:
    threshold = 0.60
    margin = 0.10
    dwell_chunks = 3
    ema_new_weight = 0.30
    smoothed = smooth_posteriors(probabilities, ema_new_weight)
    source_chunks = 0
    target_chunks = 0
    armed = False
    initial_commit = None
    detected = None
    for chunk_index, posterior in enumerate(smoothed):
        if not armed:
            if (
                posterior[source_index] >= threshold
                and posterior[source_index] - posterior[target_index] >= margin
            ):
                source_chunks += 1
            else:
                source_chunks = 0
            if source_chunks >= dwell_chunks:
                armed = True
                initial_commit = float(availability_seconds[chunk_index])
            continue
        if (
            posterior[target_index] >= threshold
            and posterior[target_index] - posterior[source_index] >= margin
        ):
            target_chunks += 1
        else:
            target_chunks = 0
        if target_chunks >= dwell_chunks:
            detected = float(availability_seconds[chunk_index])
            break
    return {
        "threshold": threshold,
        "margin": margin,
        "dwell_chunks": dwell_chunks,
        "ema_new_weight": ema_new_weight,
        "initial_commit_seconds": initial_commit,
        "source_committed_before_boundary": initial_commit is not None
        and initial_commit <= boundary_seconds,
        "detected": detected is not None,
        "detected_seconds": detected,
        "lag_ms": None
        if detected is None
        else float(1_000 * (detected - boundary_seconds)),
    }


def teacher_switch_result(
    teacher,
    selected_indices: list[int],
    waveform: torch.Tensor,
    item: dict,
    *,
    batch_size: int,
    filter_delay_ms: float,
) -> dict[str, Any]:
    num_frames = feature_frame_count(len(waveform))
    anchors = np.arange(0, num_frames, TEACHER_HOP_FRAMES, dtype=np.int64)
    if anchors[-1] != num_frames - 1:
        anchors = np.append(anchors, num_frames - 1)
    windows = [extract_window(waveform, int(frame)) for frame in anchors]
    posterior, masses = teacher_posteriors(
        teacher, selected_indices, windows, batch_size
    )
    semantic_times = (anchors * HOP_LENGTH + WIN_LENGTH) / SAMPLE_RATE
    availability_times = semantic_times + TEACHER_FUTURE_MS / 1_000
    boundary = float(item["segments"][0]["end_seconds"]) + filter_delay_ms / 1_000
    source_index = LANGUAGE_CODES.index(item["segments"][0]["language"])
    target_index = LANGUAGE_CODES.index(item["segments"][1]["language"])
    classes = posterior.argmax(axis=-1)
    semantic_event = stable_switch_event(
        classes,
        semantic_times,
        source_index=source_index,
        target_index=target_index,
        boundary_seconds=boundary,
    )
    availability_event = stable_switch_event(
        classes,
        availability_times,
        source_index=source_index,
        target_index=target_index,
        boundary_seconds=boundary,
    )
    return {
        "clip_id": item["id"],
        "direction": f"{item['segments'][0]['language']}->{item['segments'][1]['language']}",
        "nominal_boundary_seconds": float(item["segments"][0]["end_seconds"]),
        "filter_delay_ms": filter_delay_ms,
        "delay_adjusted_boundary_seconds": boundary,
        "anchor_frames": anchors.tolist(),
        "semantic_times_seconds": semantic_times.tolist(),
        "availability_times_seconds": availability_times.tolist(),
        "probabilities": posterior.tolist(),
        "selected_language_mass": masses.tolist(),
        "classes": [LANGUAGE_CODES[int(index)] for index in classes],
        "no_collar_accuracy": no_collar_accuracy(
            classes, semantic_times, item, boundary
        ),
        "semantic_500ms_stable_event": semantic_event,
        "online_availability_500ms_stable_event": availability_event,
        "top1_changes": int(np.sum(classes[1:] != classes[:-1])),
    }


def student_switch_result(
    model: CausalLIDStudent,
    frontend: LogMelFrontend,
    waveform: torch.Tensor,
    item: dict,
    *,
    chunk_frames: int,
    label_delay_frames: int,
    filter_delay_ms: float,
) -> dict[str, Any]:
    with torch.inference_mode():
        features = frontend(waveform)
        logits = model.streaming_forward(features, chunk_frames=chunk_frames).squeeze(0)
        frame_probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    availability = chunk_availability_times(
        len(frame_probabilities),
        chunk_frames=chunk_frames,
        lookahead_frames=model.lookahead_frames,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        sample_rate=SAMPLE_RATE,
    )
    chunks, chunk_times = chunk_posteriors(
        frame_probabilities, availability, min_frame=label_delay_frames
    )
    boundary = float(item["segments"][0]["end_seconds"]) + filter_delay_ms / 1_000
    source_index = LANGUAGE_CODES.index(item["segments"][0]["language"])
    target_index = LANGUAGE_CODES.index(item["segments"][1]["language"])
    raw_classes = chunks.argmax(axis=-1)
    raw_event = stable_switch_event(
        raw_classes,
        chunk_times,
        source_index=source_index,
        target_index=target_index,
        boundary_seconds=boundary,
    )
    policy = detect_pair_policy(
        chunks,
        chunk_times,
        source_index=source_index,
        target_index=target_index,
        boundary_seconds=boundary,
    )
    if item["id"] == "switch_hi_en_eval" and filter_delay_ms == 0:
        main_detected, main_config = detect_hi_to_en_switch(
            chunks,
            chunk_times,
            language_codes=tuple(LANGUAGE_CODES),
            chunk_ms=chunk_frames * HOP_LENGTH * 1_000 / SAMPLE_RATE,
        )
        if main_detected != policy["detected_seconds"]:
            raise AssertionError("generic pair policy differs from main Hindi->English policy")
        if main_config["initial_commit_seconds"] != policy["initial_commit_seconds"]:
            raise AssertionError("generic initial commit differs from main policy")

    stable_probabilities = frame_probabilities[label_delay_frames:]
    semantic_indices = np.arange(len(stable_probabilities), dtype=np.int64)
    semantic_times = (semantic_indices * HOP_LENGTH + WIN_LENGTH) / SAMPLE_RATE
    semantic_classes = stable_probabilities.argmax(axis=-1)
    return {
        "clip_id": item["id"],
        "direction": f"{item['segments'][0]['language']}->{item['segments'][1]['language']}",
        "nominal_boundary_seconds": float(item["segments"][0]["end_seconds"]),
        "filter_delay_ms": filter_delay_ms,
        "delay_adjusted_boundary_seconds": boundary,
        "chunk_times_seconds": chunk_times.tolist(),
        "chunk_probabilities": chunks.tolist(),
        "chunk_classes": [LANGUAGE_CODES[int(index)] for index in raw_classes],
        "no_collar_accuracy": no_collar_accuracy(
            semantic_classes, semantic_times, item, boundary
        ),
        "raw_chunk_500ms_stable_event": raw_event,
        "policy_event": policy,
        "raw_chunk_top1_changes": int(
            np.sum(raw_classes[1:] != raw_classes[:-1])
        ),
    }


def summarize_switches(
    switch_results: dict[str, dict[str, list[dict[str, Any]]]], model_name: str
) -> dict[str, Any]:
    event_key = (
        "online_availability_500ms_stable_event"
        if model_name == "teacher"
        else "policy_event"
    )
    lag_key = "confirmed_lag_ms" if model_name == "teacher" else "lag_ms"
    clean_rows = switch_results["clean"][model_name]
    clean_by_id = {row["clip_id"]: row for row in clean_rows}
    clean_detected = all(clean_by_id[row["clip_id"]][event_key]["detected"] for row in clean_rows)
    summaries = {}
    for condition in CONDITIONS:
        name = condition["name"]
        rows = switch_results[name][model_name]
        detected_count = sum(row[event_key]["detected"] for row in rows)
        lag_deltas = []
        extra_misses = 0
        for row in rows:
            reference = clean_by_id[row["clip_id"]]
            reference_event = reference[event_key]
            event = row[event_key]
            if reference_event["detected"] and not event["detected"]:
                extra_misses += 1
            if reference_event["detected"] and event["detected"]:
                lag_deltas.append(event[lag_key] - reference_event[lag_key])
        switch_gate_evaluable = clean_detected
        worst_lag_delta = max(lag_deltas) if lag_deltas else None
        switch_gate_passed = (
            switch_gate_evaluable
            and extra_misses == 0
            and worst_lag_delta is not None
            and worst_lag_delta <= GATE_SWITCH_LAG_DELTA_MS
        )
        summaries[name] = {
            "detected_switches": detected_count,
            "total_switches": len(rows),
            "extra_misses_vs_clean": extra_misses,
            "lag_delta_ms_vs_clean_by_clip": {
                row["clip_id"]: (
                    None
                    if not row[event_key]["detected"]
                    or not clean_by_id[row["clip_id"]][event_key]["detected"]
                    else row[event_key][lag_key]
                    - clean_by_id[row["clip_id"]][event_key][lag_key]
                )
                for row in rows
            },
            "worst_lag_delta_ms_vs_clean": worst_lag_delta,
            "switch_gate_evaluable": switch_gate_evaluable,
            "switch_gate_passed": switch_gate_passed,
            "rows": rows,
        }
    return summaries


def aggregate_transform_diagnostics(
    diagnostics: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    result = {}
    for name, rows in diagnostics.items():
        achieved = [row["achieved_snr_db"] for row in rows if row["achieved_snr_db"] is not None]
        result[name] = {
            "n_clips": len(rows),
            "input_samples": sum(row["input_samples"] for row in rows),
            "output_samples": sum(row["output_samples"] for row in rows),
            "maximum_input_peak": max(row["input_peak"] for row in rows),
            "maximum_output_peak": max(row["output_peak"] for row in rows),
            "clipped_pre_codec_samples": sum(
                row["clipped_pre_codec_samples"] for row in rows
            ),
            "mean_achieved_snr_db": None if not achieved else float(np.mean(achieved)),
            "minimum_achieved_snr_db": None if not achieved else float(np.min(achieved)),
            "maximum_achieved_snr_db": None if not achieved else float(np.max(achieved)),
            "filter_delay_ms": rows[0]["filter_delay_ms"],
            "peak_normalization_applied": False,
            "wall_seconds": sum(row["wall_seconds"] for row in rows),
        }
    return result


def package_versions() -> dict[str, str]:
    names = ("numpy", "soundfile", "speechbrain", "torch", "torchaudio")
    return {name: importlib.metadata.version(name) for name in names}


def validate_frozen_checkpoint_bundle(
    *,
    checkpoint: dict[str, Any],
    checkpoint_sha256: str,
    train_metrics: dict[str, Any],
    records: list[dict],
    manifest: Path,
    target_metadata: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the immutable submitted bundle without consulting current schema code.

    The builder may advance target-cache validation while an isolated experiment
    is running.  This audit verifies the checkpoint's own recorded identity,
    exact current corpus bytes, unchanged historical target index, model state,
    duplicate fields, and training evidence.  It separately reports current
    source-file parity instead of pretending the main release gate passed.
    """
    if checkpoint.get("checkpoint_schema_version") != 1:
        raise ValueError("checkpoint schema version is missing or unsupported")
    run_identity = checkpoint.get("run_identity")
    if not isinstance(run_identity, dict) or run_identity.get("schema_version") != 1:
        raise ValueError("checkpoint is missing its schema-1 run identity")
    expected_run_id = run_id_for_identity(run_identity)
    if checkpoint.get("run_id") != expected_run_id:
        raise ValueError("checkpoint run ID does not match its recorded identity")
    if run_identity.get("model_state_sha256") != model_state_sha256(
        checkpoint.get("model_state", {})
    ):
        raise ValueError("checkpoint model state differs from its recorded identity")

    current_corpus = corpus_identity(records, manifest)
    if run_identity.get("corpus") != current_corpus:
        raise ValueError("current manifest/audio bytes differ from the checkpoint corpus")
    target_identity = run_identity.get("target_cache")
    if not isinstance(target_identity, dict):
        raise ValueError("checkpoint is missing its historical target identity")
    if target_identity.get("metadata_sha256") != hash_path(target_metadata):
        raise ValueError("historical target metadata bytes changed since training")

    stored_pipeline = run_identity.get("pipeline")
    if not isinstance(stored_pipeline, dict):
        raise ValueError("checkpoint is missing its recorded pipeline")
    duplicate_expectations = {
        "languages": stored_pipeline["language_codes"],
        "teacher": stored_pipeline["teacher"]["model_id"],
        "target_cache": target_identity["identity"],
        "model_kwargs": stored_pipeline["student"]["model_kwargs"],
        "seed": run_identity["training"]["seed"],
    }
    for key, expected in duplicate_expectations.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"checkpoint field {key!r} contradicts its run identity")

    training_evidence = checkpoint.get("training_evidence")
    if not isinstance(training_evidence, dict):
        raise ValueError("checkpoint is missing bound training evidence")
    expected_train_metrics = {
        "run_id": expected_run_id,
        "run_identity": run_identity,
        "checkpoint_sha256": checkpoint_sha256,
        **training_evidence,
    }
    if train_metrics != expected_train_metrics:
        raise ValueError("training metrics do not exactly match the checkpoint bundle")
    contract_keys = (
        "requested_optimizer_steps",
        "successful_optimizer_steps",
        "post_update_checks",
        "all_requested_steps_completed",
        "loss_and_gradient_histories_finite",
        "post_update_model_state_finite",
        "post_update_optimizer_state_finite",
        "real_audio_optimizer_step",
        "nan_free",
    )
    expected_contract = {
        key: training_evidence.get(key) for key in contract_keys
    }
    if checkpoint.get("training_contract") != expected_contract:
        raise ValueError("checkpoint training contract contradicts its evidence")

    stored_source = stored_pipeline.get("source", {})
    stored_files = stored_source.get("files", {})
    current_file_matches = {}
    for relative, identity in stored_files.items():
        path = REPO_ROOT / relative
        current_file_matches[relative] = (
            path.is_file()
            and hash_path(path) == identity.get("sha256")
            and path.stat().st_size == identity.get("bytes")
        )
    byte_exact_inference_critical = (
        "scripts/eval.py",
        "src/streaming_lid/audio.py",
        "src/streaming_lid/model.py",
    )
    if not all(
        current_file_matches.get(path, False)
        for path in byte_exact_inference_critical
    ):
        raise ValueError("an inference-critical main source differs from the checkpoint")
    current_pipeline = pipeline_configuration()
    current_pipeline.pop("source")
    stored_pipeline_without_source = dict(stored_pipeline)
    stored_pipeline_without_source.pop("source")
    semantic_pipeline_matches = current_pipeline == stored_pipeline_without_source
    if not semantic_pipeline_matches:
        raise ValueError("current runtime pipeline semantics differ from the checkpoint")
    audit = {
        "checkpoint_bundle_self_validated": True,
        "checkpoint_corpus_bytes_validated": True,
        "historical_target_metadata_bytes_validated": True,
        "training_metrics_exactly_validated": True,
        "byte_exact_inference_critical_sources_match_checkpoint": True,
        "runtime_pipeline_semantics_match_checkpoint": semantic_pipeline_matches,
        "current_source_file_matches_checkpoint": current_file_matches,
        "all_current_pipeline_sources_match_checkpoint": all(
            current_file_matches.values()
        ),
        "current_main_release_gate_validated": all(current_file_matches.values()),
        "current_main_release_gate_note": (
            "The current main source differs only in files shown false above; "
            "the isolated experiment consumes neither target arrays nor training code."
        ),
    }
    return run_identity, audit


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(0)
    np.random.seed(0)

    launched_at = datetime.now(timezone.utc).isoformat()
    driver_path = Path(__file__).resolve()
    driver_sha256_at_start = hash_path(driver_path)
    g711_validation = validate_g711_fixed_vectors()
    filter_validation = validate_filter()

    records = read_manifest(args.manifest)
    speaker_audit = require_speaker_disjoint(records)
    heldout = [item for item in records if item["split"] == "heldout"]
    switches = [item for item in records if item["split"] == "switch"]
    if len(heldout) != 21 or len(switches) != 2:
        raise AssertionError("expected the main 21 held-out and two switch clips")
    if Counter(item["language"] for item in heldout) != Counter(
        {language: 3 for language in LANGUAGE_CODES}
    ):
        raise AssertionError("held-out set is not the main balanced seven-language set")
    if {item["id"] for item in switches} != {
        "switch_hi_en_eval",
        "switch_en_hi_eval",
    }:
        raise AssertionError("unexpected switch evaluation set")
    selected_records = heldout + switches
    input_sha256_at_start = input_fingerprint(selected_records, args.manifest)

    checkpoint_sha256_at_start = hash_path(args.checkpoint)
    train_metrics_sha256_at_start = hash_path(args.train_metrics)
    target_metadata_path = args.targets_dir / "metadata.json"
    target_metadata_sha256_at_start = hash_path(target_metadata_path)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    train_metrics = json.loads(args.train_metrics.read_text(encoding="utf-8"))
    run_identity, checkpoint_bundle_audit = validate_frozen_checkpoint_bundle(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256_at_start,
        train_metrics=train_metrics,
        records=records,
        manifest=args.manifest,
        target_metadata=target_metadata_path,
    )
    pipeline = run_identity["pipeline"]
    if tuple(pipeline["language_codes"]) != tuple(LANGUAGE_CODES):
        raise AssertionError("checkpoint language order differs from experiment")
    label_delay_frames = int(pipeline["distillation"]["label_delay_frames"])
    chunk_frames = int(pipeline["streaming"]["chunk_frames"])

    model = CausalLIDStudent(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    frontend = LogMelFrontend().eval()
    teacher, selected_indices, teacher_artifact = load_teacher(args.model_dir)

    print(
        f"validated {len(heldout)} held-out and {len(switches)} switch clips; "
        f"building {len(CONDITIONS)} whole-clip channel views",
        flush=True,
    )
    transformed: dict[str, dict[str, torch.Tensor]] = {}
    transform_diagnostics: dict[str, list[dict[str, Any]]] = {
        condition["name"]: [] for condition in CONDITIONS
    }
    for item in selected_records:
        waveform = load_audio(resolve_audio_path(item, args.manifest))
        transformed[item["id"]] = {}
        for condition in CONDITIONS:
            value, diagnostics = transform_waveform(
                waveform, item["id"], condition
            )
            transformed[item["id"]][condition["name"]] = value
            transform_diagnostics[condition["name"]].append(diagnostics)

    monolingual_rows: list[dict[str, Any]] = []
    inference_seconds = {
        condition["name"]: {"teacher": 0.0, "student": 0.0}
        for condition in CONDITIONS
    }
    for condition in CONDITIONS:
        condition_name = condition["name"]
        for seconds in PREFIX_SECONDS:
            prefixes = []
            padded_by_id = {}
            for item in heldout:
                prefix, padded = prefix_audio(
                    transformed[item["id"]][condition_name], seconds
                )
                prefixes.append(prefix)
                padded_by_id[item["id"]] = padded

            started = time.perf_counter()
            teacher_probabilities, teacher_masses = teacher_posteriors(
                teacher,
                selected_indices,
                prefixes,
                args.teacher_batch_size,
            )
            inference_seconds[condition_name]["teacher"] += (
                time.perf_counter() - started
            )

            started = time.perf_counter()
            student_outputs = [
                student_posterior(
                    model,
                    frontend,
                    prefix,
                    chunk_frames=chunk_frames,
                    label_delay_frames=label_delay_frames,
                )
                for prefix in prefixes
            ]
            inference_seconds[condition_name]["student"] += (
                time.perf_counter() - started
            )

            for index, item in enumerate(heldout):
                teacher_probability = teacher_probabilities[index]
                student_probability, student_frames = student_outputs[index]
                teacher_prediction = LANGUAGE_CODES[int(teacher_probability.argmax())]
                student_prediction = LANGUAGE_CODES[int(student_probability.argmax())]
                monolingual_rows.append(
                    {
                        "condition": condition_name,
                        "clip_id": item["id"],
                        "expected": item["language"],
                        "speaker_id": item["speaker_id"],
                        "prefix_seconds": seconds,
                        "right_zero_padded": padded_by_id[item["id"]],
                        "teacher": {
                            "prediction": teacher_prediction,
                            "probabilities": teacher_probability.tolist(),
                            "max_probability": float(teacher_probability.max()),
                            "selected_language_mass": float(teacher_masses[index]),
                            "correct": teacher_prediction == item["language"],
                        },
                        "student": {
                            "prediction": student_prediction,
                            "probabilities": student_probability.tolist(),
                            "max_probability": float(student_probability.max()),
                            "averaged_delay_aligned_stable_frames": student_frames,
                            "correct": student_prediction == item["language"],
                        },
                    }
                )
        print(f"completed monolingual prefixes: {condition_name}", flush=True)

    teacher_summary = summarize_conditions(monolingual_rows, "teacher")
    student_summary = summarize_conditions(monolingual_rows, "student")

    switch_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for condition in CONDITIONS:
        condition_name = condition["name"]
        filter_delay_ms = (
            filter_validation["measured_bandpass_delay_ms"]
            if condition["bandpass"]
            else 0.0
        )
        switch_results[condition_name] = {"teacher": [], "student": []}
        for item in switches:
            waveform = transformed[item["id"]][condition_name]
            started = time.perf_counter()
            teacher_result = teacher_switch_result(
                teacher,
                selected_indices,
                waveform,
                item,
                batch_size=args.teacher_batch_size,
                filter_delay_ms=filter_delay_ms,
            )
            inference_seconds[condition_name]["teacher"] += (
                time.perf_counter() - started
            )
            started = time.perf_counter()
            student_result = student_switch_result(
                model,
                frontend,
                waveform,
                item,
                chunk_frames=chunk_frames,
                label_delay_frames=label_delay_frames,
                filter_delay_ms=filter_delay_ms,
            )
            inference_seconds[condition_name]["student"] += (
                time.perf_counter() - started
            )
            switch_results[condition_name]["teacher"].append(teacher_result)
            switch_results[condition_name]["student"].append(student_result)
        print(f"completed two switch clips: {condition_name}", flush=True)

    teacher_switch_summary = summarize_switches(switch_results, "teacher")
    student_switch_summary = summarize_switches(switch_results, "student")

    condition_decisions = {}
    for condition in CONDITIONS:
        name = condition["name"]
        teacher_mono = teacher_summary[name]["monolingual_gate"]["passed"]
        student_mono = student_summary[name]["monolingual_gate"]["passed"]
        teacher_switch = teacher_switch_summary[name]["switch_gate_passed"]
        student_switch_evaluable = student_switch_summary[name][
            "switch_gate_evaluable"
        ]
        student_switch = student_switch_summary[name]["switch_gate_passed"]
        condition_decisions[name] = {
            "teacher_monolingual_gate_passed": teacher_mono,
            "student_monolingual_gate_passed": student_mono,
            "teacher_switch_gate_passed": teacher_switch,
            "student_switch_gate_evaluable": student_switch_evaluable,
            "student_switch_gate_passed": student_switch,
            "all_evaluable_gates_passed": teacher_mono
            and student_mono
            and teacher_switch
            and (not student_switch_evaluable or student_switch),
            "fully_certified_including_student_switch": teacher_mono
            and student_mono
            and teacher_switch
            and student_switch_evaluable
            and student_switch,
        }

    source_sha256_at_end, source_files_at_end = snapshot_pipeline_sources()
    driver_sha256_at_end = hash_path(driver_path)
    input_sha256_at_end = input_fingerprint(selected_records, args.manifest)
    checkpoint_sha256_at_end = hash_path(args.checkpoint)
    train_metrics_sha256_at_end = hash_path(args.train_metrics)
    target_metadata_sha256_at_end = hash_path(target_metadata_path)
    identities_stable = {
        "driver_unchanged": driver_sha256_at_end == driver_sha256_at_start,
        "pipeline_sources_unchanged": source_sha256_at_end
        == PIPELINE_SOURCE_SHA256_AT_IMPORT,
        "inputs_unchanged": input_sha256_at_end == input_sha256_at_start,
        "checkpoint_unchanged": checkpoint_sha256_at_end
        == checkpoint_sha256_at_start,
        "train_metrics_unchanged": train_metrics_sha256_at_end
        == train_metrics_sha256_at_start,
        "historical_target_metadata_unchanged": target_metadata_sha256_at_end
        == target_metadata_sha256_at_start,
    }
    if not all(identities_stable.values()):
        raise RuntimeError(
            f"experiment inputs changed during execution: {identities_stable}"
        )
    if source_files_at_end != PIPELINE_SOURCE_FILES_AT_IMPORT:
        raise AssertionError("combined source hash passed but per-file snapshot changed")

    run_configuration = {
        "analysis_version": ANALYSIS_VERSION,
        "prefix_seconds": list(PREFIX_SECONDS),
        "conditions": list(CONDITIONS),
        "telephony_sample_rate": TELEPHONY_SAMPLE_RATE,
        "filter_taps": FIR_TAPS,
        "passband_hz": [PASSBAND_LOW_HZ, PASSBAND_HIGH_HZ],
        "switch_collar_ms": SWITCH_COLLAR_MS,
        "switch_stability_ms": SWITCH_STABILITY_MS,
        "gates": {
            "macro_f1_drop_pp": GATE_MACRO_F1_DROP_PP,
            "class_recall_drop_pp": GATE_CLASS_RECALL_DROP_PP,
            "median_conditional_kl": GATE_MEDIAN_KL,
            "switch_lag_delta_ms": GATE_SWITCH_LAG_DELTA_MS,
        },
        "teacher_batch_size": args.teacher_batch_size,
        "torch_threads": args.threads,
    }
    experiment_identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "driver_sha256": driver_sha256_at_start,
        "pipeline_source_sha256": PIPELINE_SOURCE_SHA256_AT_IMPORT,
        "evaluation_input_sha256": input_sha256_at_start,
        "checkpoint_sha256": checkpoint_sha256_at_start,
        "train_metrics_sha256": train_metrics_sha256_at_start,
        "target_metadata_sha256": target_metadata_sha256_at_start,
        "checkpoint_run_id": checkpoint["run_id"],
        "teacher_artifact_sha256": teacher_artifact["artifact_sha256"],
        "historical_target_cache_identity": run_identity["target_cache"],
        "configuration": run_configuration,
    }
    experiment_run_id = canonical_sha256(experiment_identity)
    results = {
        "schema_version": SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "experiment": EXPERIMENT_NAME,
        "experiment_run_id": experiment_run_id,
        "started_at_utc": launched_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "How do resampling, an explicit telephone passband, G.711 companding, "
            "white noise, and noise-before-codec interactions affect the pinned "
            "ECAPA teacher and submitted causal student?"
        ),
        "experiment_identity": experiment_identity,
        "identity_checks": {
            **identities_stable,
            **checkpoint_bundle_audit,
            "speaker_disjoint": speaker_audit["speaker_disjoint"],
            "pipeline_source_files_at_launch_and_end": source_files_at_end,
            "input_sha256_at_start_and_end": input_sha256_at_end,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "soundfile_libsndfile_version": sf.__libsndfile_version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "pid": os.getpid(),
        },
        "data": {
            "manifest": str(args.manifest),
            "evaluation_input_sha256": input_sha256_at_start,
            "n_heldout_monolingual_clips": len(heldout),
            "n_switch_clips": len(switches),
            "heldout_clip_ids": [item["id"] for item in heldout],
            "switch_clip_ids": [item["id"] for item in switches],
            "languages": list(LANGUAGE_CODES),
            "speaker_split": speaker_audit,
            "prefix_protocol": (
                "transform complete waveform once, then take a leading prefix; "
                "right-zero-pad only when the original clip is shorter"
            ),
        },
        "models": {
            "teacher": {
                "model_id": TEACHER_NAME,
                "revision": TEACHER_REVISION,
                "artifact": teacher_artifact,
                "parameter_count": sum(
                    parameter.numel() for parameter in teacher.mods.parameters()
                ),
                "posterior": "107-way posterior restricted and renormalized to seven classes",
            },
            "student": {
                "run_id": checkpoint["run_id"],
                "checkpoint_sha256": checkpoint_sha256_at_start,
                "parameter_count": model.parameter_count,
                "model_kwargs": checkpoint["model_kwargs"],
                "label_delay_frames": label_delay_frames,
                "chunk_frames": chunk_frames,
                "clip_posterior": "mean of delay-aligned stable frame posteriors",
            },
        },
        "channel_validation": {
            "g711_fixed_vectors": g711_validation,
            "filter": filter_validation,
            "whole_clip_transform": True,
            "peak_normalization_applied": False,
            "noise": (
                "deterministic zero-mean unit-RMS Gaussian realization per clip; "
                "scaled against whole-clip RMS and shared across SNR/codec arms"
            ),
        },
        "transform_diagnostics": aggregate_transform_diagnostics(
            transform_diagnostics
        ),
        "inference_wall_seconds_by_condition": inference_seconds,
        "monolingual_summary": {
            "teacher": teacher_summary,
            "student": student_summary,
        },
        "switch_summary": {
            "teacher": teacher_switch_summary,
            "student": student_switch_summary,
        },
        "condition_decisions": condition_decisions,
        "monolingual_requests": monolingual_rows,
        "finite": True,
    }
    serialized = json.dumps(results, indent=2, allow_nan=False) + "\n"
    output = driver_path.with_name("results.json")
    temporary = driver_path.with_name("results.json.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    parsed = json.loads(temporary.read_text(encoding="utf-8"))
    if parsed["experiment_run_id"] != experiment_run_id:
        raise AssertionError("temporary result failed round-trip validation")
    temporary.replace(output)
    print(
        f"wrote {output}; run={experiment_run_id}; "
        f"teacher clean F1={teacher_summary['clean']['aggregate_1_2_4s']['macro_f1']:.3f}; "
        f"student clean F1={student_summary['clean']['aggregate_1_2_4s']['macro_f1']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
