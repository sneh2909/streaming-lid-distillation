#!/usr/bin/env python3
"""Targeted teacher audit for the current English-supervision failure.

This experiment compares four frozen offline LID teachers on:

* all 70 monolingual training clips, including the four ECAPA-wrong Edge
  English clips;
* the exact 21 monolingual held-out clips and both held-out switch clips used
  by the main pipeline; and
* the already-frozen 100-clip FLEURS-English validation slice.

Fixed 1/2/4-second rows are true leading prefixes and are included only when
the source really reaches the deadline.  Short clips are never zero padded.
Model loading is excluded from timing; preprocessing and inference are timed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import resource
import statistics
import sys
import tarfile
import time
import urllib.request
import warnings
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.teacher_targets import (  # noqa: E402
    extract_window,
    label_indices,
    resolve_teacher_artifact,
)
from streaming_lid.audio import feature_frame_count, load_audio  # noqa: E402
from streaming_lid.config import (  # noqa: E402
    HOP_LENGTH,
    LANGUAGE_CODES,
    SAMPLE_RATE,
    TEACHER_ARTIFACT_FILES,
    TEACHER_ARTIFACT_SHA256,
    TEACHER_FUTURE_MS,
    TEACHER_HOP_FRAMES,
    TEACHER_NAME,
    TEACHER_REVISION,
    WIN_LENGTH,
)
from streaming_lid.data import (  # noqa: E402
    capture_manifest_snapshot,
    file_sha256,
    resolve_audio_path,
)


ANALYSIS_SCHEMA_VERSION = 1
EXPERIMENT_NAME = "english-teacher-bakeoff"
EXPECTED_MAIN_MANIFEST_SHA256 = (
    "7df6fc5d94db4c88cac9df6b2de8bc9dc5fcacf1bf36853f839bf494ad59314c"
)
EXPECTED_FLEURS_FILE_SHA256 = (
    "5a3335cda8b679ca4a6de8f084bff23c8fa82c18e1670e8bfdc6da751b246614"
)
EXPECTED_FLEURS_MANIFEST_SHA256 = (
    "ef0ec36427f742074b1bc88bd42f1b7e5b93b43bb390f9382f5030f3d8d2a637"
)
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
VIEWS = OrderedDict((("1s", 1), ("2s", 2), ("4s", 4), ("full", None)))
PRIMARY_VIEWS_BY_COHORT = {
    "train_mono": ("1s", "2s", "full"),
    "heldout_mono": ("1s", "2s", "full"),
    "fleurs_en_validation": ("1s", "2s", "4s"),
}
TARGETED_ENGLISH_IDS = (
    "en_train_06",
    "en_train_07",
    "en_train_08",
    "en_train_09",
)
EXPECTED_ECAPA_TARGETED_PREDICTIONS = {
    "en_train_06": "hi",
    "en_train_07": "hi",
    "en_train_08": "hi",
    "en_train_09": "gu",
}

MAIN_SOURCE_FILES = (
    "scripts/teacher_targets.py",
    "src/streaming_lid/audio.py",
    "src/streaming_lid/config.py",
    "src/streaming_lid/data.py",
)


WHISPER_FILES = OrderedDict(
    [
        ("added_tokens.json", (34_604, "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1")),
        ("config.json", (1_967, "e6a2b489da1b5aed65a8eb8d1e7466fa867ad5643a8bc138ba708bd56b2875c4")),
        ("generation_config.json", (3_868, "71565b8ef50d0bf7a1193ed4bbed195b94e70c18894d81bba2f1233dcec3ab53")),
        ("merges.txt", (493_869, "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6")),
        ("model.safetensors", (966_995_080, "1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b")),
        ("normalizer.json", (52_666, "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd")),
        ("preprocessor_config.json", (184_990, "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666")),
        ("special_tokens_map.json", (2_194, "e67ae3a0aaa99abcd9f187138e12db1f65c16a14761c50ef10eef2c174a7a691")),
        ("tokenizer.json", (2_480_466, "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566")),
        ("tokenizer_config.json", (282_683, "2a4c4281cf9f51ac6ccc406fdc711a087afe6530f671fa7b80953edc498275ce")),
        ("vocab.json", (835_550, "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9")),
    ]
)
MMS_FILES = OrderedDict(
    [
        ("config.json", (4_267, "f5f50a99828c0334299ca9180c026518ca69b6b2a41dd3bbede74eea762e4ba3")),
        ("model.safetensors", (3_864_495_808, "69fa8def8f0242660b47d8b47c2a1b645d9fc70098cc217fbd52087b68de0ecd")),
        ("preprocessor_config.json", (212, "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335")),
    ]
)
AMBERNET_FILES = OrderedDict(
    [
        ("config.json", (2_534, "2dc3ddbb0eb454c2390f358cb1dc983a0f34e109177cf1b1c5a229826c88788b")),
        ("model.safetensors", (115_989_980, "62bdabd1fb2984a75a4c45918eef1942e7e80dd9075763a37f0ba77d2d28c9ee")),
        ("modeling_ambernet.py", (10_165, "096e00c545b68787e8ffc057d48ba5cf61296f5d72aa45ea9dd7dacad87575ca")),
    ]
)


MODEL_REGISTRY = OrderedDict(
    [
        (
            "ecapa",
            {
                "model_id": TEACHER_NAME,
                "revision": TEACHER_REVISION,
                "artifact_sha256": TEACHER_ARTIFACT_SHA256,
                "files": OrderedDict(
                    (name, (entry["bytes"], entry["sha256"]))
                    for name, entry in TEACHER_ARTIFACT_FILES.items()
                ),
                "license": "Apache-2.0",
                "license_gate": "permissive",
            },
        ),
        (
            "whisper-small",
            {
                "model_id": "openai/whisper-small",
                "revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
                "artifact_sha256": "7d5828fd8c8d0602a6b400b177bc673a47cfc71c1180dbb6783ae904e8aee9f3",
                "files": WHISPER_FILES,
                "license": "Apache-2.0",
                "license_gate": "permissive",
            },
        ),
        (
            "mms-lid-126",
            {
                "model_id": "facebook/mms-lid-126",
                "revision": "53da6e311f3ce48f324fe335924216187e3109a4",
                "artifact_sha256": "e1b75df414cb4f4587a397f944b5763a5a78354b153913bb03a8921fe821aa80",
                "files": MMS_FILES,
                "license": "CC-BY-NC-4.0",
                "license_gate": "noncommercial_only",
            },
        ),
        (
            "ambernet",
            {
                "model_id": "surogate/ambernet-langid",
                "revision": "3b7b3fbfcc51753745f1d9abd107c9530c391d5b",
                "artifact_sha256": "7146b2397bb7e38f3ae871deaf80c5431551af8f8d9829a6c92ad9f63fb54bc5",
                "files": AMBERNET_FILES,
                "license": "NVIDIA NGC Terms of Use",
                "license_gate": "review_required",
            },
        ),
    ]
)

MMS_TO_REPO_CODE = {
    "eng": "en",
    "hin": "hi",
    "mar": "mr",
    "ben": "bn",
    "tam": "ta",
    "tel": "te",
    "guj": "gu",
}
REPO_TO_MMS_CODE = {repo: mms for mms, repo in MMS_TO_REPO_CODE.items()}

AMBERNET_NGC = {
    "url": "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/langid_ambernet/versions/1.12.0/files/ambernet.nemo",
    "version": "1.12.0",
    "bytes": 116_049_920,
    "sha256": "2f92d645b9ea5824d7663584fecb9ecc52557d0d700e24266747f38a61ba1681",
    "model_config_sha256": "6ae621a3def04644ad465dd5ee3870b42a83003358050ba88b7201d5b2dde3f9",
    "model_weights_sha256": "c57077f3a453549c60fa50287d5f3ceea7665f4f1798e2b56e5d0ef52af6b6e2",
}


@dataclass(frozen=True)
class Case:
    key: str
    cohort: str
    clip_id: str
    view: str
    expected: str
    waveform: torch.Tensor
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class Prediction:
    native_code: str
    restricted_code: str
    selected_absolute: tuple[float, ...]
    selected_conditional: tuple[float, ...]
    selected_mass: float


class Predictor(Protocol):
    name: str
    model_id: str
    revision: str
    parameter_count: int
    batch_size: int
    prediction_space: str
    load_validation: Mapping[str, Any]

    def predict_batch(self, waveforms: Sequence[torch.Tensor]) -> list[Prediction]: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/generated/manifest.jsonl")
    )
    parser.add_argument(
        "--fleurs-dir",
        type=Path,
        default=Path(
            "data/experiments/checkpoint-external-validation/fleurs-validation"
        ),
    )
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_REGISTRY), default=list(MODEL_REGISTRY)
    )
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require all pinned Hub snapshots and the NGC artifact to be cached.",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Discard compatible cached model rows."
    )
    return parser.parse_args()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def package_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name in (
        "numpy",
        "torch",
        "torchaudio",
        "speechbrain",
        "transformers",
        "huggingface-hub",
        "tokenizers",
        "safetensors",
        "soundfile",
        "PyYAML",
    ):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def source_identity() -> dict[str, Any]:
    files = [Path(__file__).resolve()] + [REPO_ROOT / name for name in MAIN_SOURCE_FILES]
    records = OrderedDict()
    for path in files:
        relative = path.relative_to(REPO_ROOT).as_posix()
        records[relative] = file_record(path)
    return {
        "files": records,
        "sha256": canonical_sha256(records),
    }


def artifact_aggregate(files: Mapping[str, tuple[int, str]]) -> str:
    digest = hashlib.sha256()
    for filename in sorted(files):
        digest.update(filename.encode("utf-8"))
        digest.update(bytes.fromhex(files[filename][1]))
    return digest.hexdigest()


def verify_snapshot(
    name: str, *, local_files_only: bool
) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    registry = MODEL_REGISTRY[name]
    if name == "ecapa":
        snapshot, main_identity = resolve_teacher_artifact()
        if main_identity["artifact_sha256"] != registry["artifact_sha256"]:
            raise ValueError("main ECAPA resolver returned an unexpected artifact")
    else:
        snapshot = Path(
            snapshot_download(
                repo_id=registry["model_id"],
                revision=registry["revision"],
                allow_patterns=list(registry["files"]),
                local_files_only=local_files_only,
            )
        ).resolve()

    actual_files: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for filename, (expected_bytes, expected_sha256) in registry["files"].items():
        path = snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned {name} artifact {path}")
        actual = file_record(path)
        expected = {"bytes": expected_bytes, "sha256": expected_sha256}
        if actual != expected:
            raise ValueError(f"{name} artifact differs for {filename}: {actual} != {expected}")
        actual_files[filename] = actual
    aggregate = artifact_aggregate(registry["files"])
    if aggregate != registry["artifact_sha256"]:
        raise AssertionError(f"static {name} aggregate is internally inconsistent")
    return snapshot, {
        "model_id": registry["model_id"],
        "revision": registry["revision"],
        "artifact_sha256": aggregate,
        "aggregate_algorithm": "sha256(filename_utf8 || raw_file_sha256), lexical filenames",
        "files": actual_files,
        "license": registry["license"],
        "license_gate": registry["license_gate"],
    }


def download_exact(url: str, destination: Path, expected: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            digest.update(block)
            total += len(block)
    if total != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError("downloaded official AmberNet artifact differs from the pin")
    os.replace(temporary, destination)


def resolve_official_ambernet(*, offline: bool) -> tuple[Path, Path, dict[str, Any]]:
    cache = REPO_ROOT / ".cache/models/ambernet-official"
    nemo_path = cache / "ambernet.nemo"
    if not nemo_path.is_file() or file_record(nemo_path) != {
        "bytes": AMBERNET_NGC["bytes"],
        "sha256": AMBERNET_NGC["sha256"],
    }:
        if offline:
            raise FileNotFoundError("verified official AmberNet .nemo is not cached")
        download_exact(AMBERNET_NGC["url"], nemo_path, AMBERNET_NGC)

    extracted = cache / "extracted"
    config_path = extracted / "model_config.yaml"
    weights_path = extracted / "model_weights.ckpt"
    expected_inner = {
        config_path: AMBERNET_NGC["model_config_sha256"],
        weights_path: AMBERNET_NGC["model_weights_sha256"],
    }
    needs_extract = any(
        not path.is_file() or file_sha256(path) != expected_sha
        for path, expected_sha in expected_inner.items()
    )
    if needs_extract:
        extracted.mkdir(parents=True, exist_ok=True)
        with tarfile.open(nemo_path, mode="r:") as archive:
            members = {member.name: member for member in archive.getmembers() if member.isfile()}
            if set(members) != {"./model_config.yaml", "./model_weights.ckpt"}:
                raise ValueError(f"unexpected official AmberNet archive members: {set(members)}")
            for member_name, output_path in (
                ("./model_config.yaml", config_path),
                ("./model_weights.ckpt", weights_path),
            ):
                source = archive.extractfile(members[member_name])
                if source is None:
                    raise ValueError(f"could not read {member_name}")
                temporary = output_path.with_suffix(output_path.suffix + ".tmp")
                with temporary.open("wb") as output:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                os.replace(temporary, output_path)
    for path, expected_sha in expected_inner.items():
        if file_sha256(path) != expected_sha:
            raise ValueError(f"official AmberNet inner artifact differs: {path}")

    return config_path, weights_path, {
        "source": "NVIDIA NGC",
        "url": AMBERNET_NGC["url"],
        "version": AMBERNET_NGC["version"],
        "nemo": file_record(nemo_path),
        "model_config": file_record(config_path),
        "model_weights": file_record(weights_path),
    }


def validate_ambernet_parity(
    snapshot: Path, *, offline: bool
) -> dict[str, Any]:
    import yaml
    from safetensors.torch import load_file

    config_path, weights_path, official = resolve_official_ambernet(offline=offline)
    hf_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    official_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if hf_config["labels"] != official_config["train_ds"]["labels"]:
        raise ValueError("AmberNet label order differs from the official artifact")
    if hf_config["preprocessor"]["sample_rate"] != official_config["preprocessor"]["sample_rate"]:
        raise ValueError("AmberNet sample rate differs from the official artifact")
    expected_frontend = {
        "n_fft": official_config["preprocessor"]["n_fft"],
        "win_length": round(
            official_config["preprocessor"]["window_size"]
            * official_config["preprocessor"]["sample_rate"]
        ),
        "hop_length": round(
            official_config["preprocessor"]["window_stride"]
            * official_config["preprocessor"]["sample_rate"]
        ),
        "n_mels": official_config["preprocessor"]["features"],
    }
    for key, value in expected_frontend.items():
        if hf_config["preprocessor"][key] != value:
            raise ValueError(f"AmberNet frontend field {key} differs from official")
    for converted, official_block in zip(
        hf_config["encoder"]["jasper"], official_config["encoder"]["jasper"], strict=True
    ):
        for key in ("filters", "repeat", "kernel", "dropout", "residual", "separable", "se"):
            if converted[key] != official_block[key]:
                raise ValueError(f"AmberNet encoder field {key} differs from official")
    official_decoder = official_config["decoder"]
    if hf_config["decoder"] != {
        "feat_in": official_decoder["feat_in"],
        "num_classes": official_decoder["num_classes"],
        "emb_size": official_decoder["emb_sizes"],
    }:
        raise ValueError("AmberNet decoder configuration differs from official")

    official_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    converted_state = load_file(snapshot / "model.safetensors", device="cpu")
    official_inference_keys = set(official_state) - {"loss.weight"}
    if set(converted_state) != official_inference_keys:
        raise ValueError(
            "AmberNet converted checkpoint key set differs from official inference keys"
        )
    unequal = [
        key
        for key in sorted(official_inference_keys)
        if not torch.equal(official_state[key], converted_state[key])
    ]
    if unequal:
        raise ValueError(f"AmberNet converted tensors differ from official: {unequal[:3]}")
    return {
        "official_artifact": official,
        "official_tensor_count": len(official_state),
        "converted_inference_tensor_count": len(converted_state),
        "exactly_equal_inference_tensors": len(converted_state),
        "official_only_training_tensors": ["loss.weight"],
        "label_order_exact": True,
        "explicit_architecture_fields_exact": True,
        "inference_weight_parity_passed": True,
        "caveat": (
            "standalone frontend/model code is a reimplementation; exact NeMo output "
            "parity was not run because NeMo is not a project dependency"
        ),
    }


def make_prediction(native_code: str, selected: Sequence[float]) -> Prediction:
    values = np.asarray(selected, dtype=np.float64)
    if values.shape != (len(LANGUAGE_CODES),):
        raise ValueError("selected posterior width differs from deployment languages")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("selected posterior is invalid")
    selected_mass = float(values.sum())
    if not 0 < selected_mass <= 1.00001:
        raise ValueError(f"selected posterior mass is invalid: {selected_mass}")
    conditional = values / selected_mass
    winner = LANGUAGE_CODES[int(conditional.argmax())]
    return Prediction(
        native_code=native_code,
        restricted_code=winner,
        selected_absolute=tuple(float(value) for value in values),
        selected_conditional=tuple(float(value) for value in conditional),
        selected_mass=selected_mass,
    )


class ECAPAPredictor:
    name = "ecapa"
    prediction_space = "native_107way_softmax; selected_7way conditional view"
    batch_size = 24

    def __init__(self, snapshot: Path, artifact: Mapping[str, Any]) -> None:
        from speechbrain.inference.classifiers import EncoderClassifier

        self.model_id = artifact["model_id"]
        self.revision = artifact["revision"]
        savedir = REPO_ROOT / ".cache/models/english-teacher-bakeoff-ecapa"
        warnings.filterwarnings("ignore", message=".*custom_fwd.*deprecated.*")
        self.model = EncoderClassifier.from_hparams(
            source=str(snapshot),
            savedir=str(savedir),
            overrides={"pretrained_path": str(snapshot)},
            run_opts={"device": "cpu"},
        )
        self.model.eval()
        self.model.hparams.label_encoder.ignore_len()
        selected_indices = label_indices(self.model)
        self.index_to_label = self.model.hparams.label_encoder.ind2lab
        self.code_to_index = {
            label.split(":", 1)[0]: index for index, label in self.index_to_label.items()
        }
        if selected_indices != [self.code_to_index[code] for code in LANGUAGE_CODES]:
            raise AssertionError("ECAPA selected-language map is inconsistent")
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.load_validation = {
            "pinned_local_snapshot": True,
            "main_label_map_validated": True,
        }

    def predict_batch(self, waveforms: Sequence[torch.Tensor]) -> list[Prediction]:
        lengths = torch.tensor([len(waveform) for waveform in waveforms], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        relative_lengths = lengths / padded.shape[1]
        with torch.inference_mode():
            log_probabilities, _, _, _ = self.model.classify_batch(
                padded, relative_lengths
            )
        probabilities = log_probabilities.exp()
        output = []
        for logs, probs in zip(log_probabilities, probabilities, strict=True):
            native = self.index_to_label[int(logs.argmax())].split(":", 1)[0]
            output.append(
                make_prediction(
                    native,
                    [float(probs[self.code_to_index[code]]) for code in LANGUAGE_CODES],
                )
            )
        return output


class WhisperPredictor:
    name = "whisper-small"
    prediction_space = "softmax over 99 Whisper language-token logits"
    batch_size = 2

    def __init__(self, snapshot: Path, artifact: Mapping[str, Any]) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.model_id = artifact["model_id"]
        self.revision = artifact["revision"]
        self.processor = WhisperProcessor.from_pretrained(
            str(snapshot), local_files_only=True
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(snapshot), local_files_only=True, use_safetensors=True
        )
        self.model.eval()
        language_to_id = self.model.generation_config.lang_to_id
        self.language_ids = list(language_to_id.values())
        self.id_to_code = {
            token_id: token.removeprefix("<|").removesuffix("|>")
            for token, token_id in language_to_id.items()
        }
        self.code_to_id = {code: token_id for token_id, code in self.id_to_code.items()}
        self.language_position = {
            token_id: index for index, token_id in enumerate(self.language_ids)
        }
        if set(LANGUAGE_CODES) - set(self.code_to_id):
            raise ValueError("Whisper snapshot lacks a selected language")
        expected_indices = (50259, 50276, 50320, 50302, 50287, 50299, 50333)
        actual_indices = tuple(self.code_to_id[code] for code in LANGUAGE_CODES)
        if actual_indices != expected_indices:
            raise ValueError(f"Whisper language map changed: {actual_indices}")
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.load_validation = {
            "pinned_local_snapshot": True,
            "use_safetensors": True,
            "language_token_count": len(self.language_ids),
            "selected_token_ids": list(actual_indices),
        }

    def predict_batch(self, waveforms: Sequence[torch.Tensor]) -> list[Prediction]:
        inputs = self.processor(
            [waveform.numpy() for waveform in waveforms],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        decoder_input_ids = torch.full(
            (len(waveforms), 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
        )
        with torch.inference_mode():
            token_logits = self.model(
                input_features=inputs.input_features,
                decoder_input_ids=decoder_input_ids,
            ).logits[:, 0]
        language_probs = torch.softmax(token_logits[:, self.language_ids], dim=-1)
        output = []
        for row in language_probs:
            winning_id = self.language_ids[int(row.argmax())]
            output.append(
                make_prediction(
                    self.id_to_code[winning_id],
                    [
                        float(row[self.language_position[self.code_to_id[code]]])
                        for code in LANGUAGE_CODES
                    ],
                )
            )
        return output


class MMSPredictor:
    name = "mms-lid-126"
    prediction_space = "native_126way_softmax; selected_7way conditional view"
    batch_size = 4

    def __init__(self, snapshot: Path, artifact: Mapping[str, Any]) -> None:
        from safetensors import safe_open
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

        self.model_id = artifact["model_id"]
        self.revision = artifact["revision"]
        self.processor = AutoFeatureExtractor.from_pretrained(
            str(snapshot), local_files_only=True
        )
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            str(snapshot), local_files_only=True, use_safetensors=True
        )
        self.model.eval()
        self.index_to_code = {
            int(index): code for index, code in self.model.config.id2label.items()
        }
        self.code_to_index = {code: index for index, code in self.index_to_code.items()}
        expected_indices = (2, 16, 29, 12, 46, 23, 74)
        actual_indices = tuple(
            self.code_to_index[REPO_TO_MMS_CODE[code]] for code in LANGUAGE_CODES
        )
        if actual_indices != expected_indices:
            raise ValueError(f"MMS language map changed: {actual_indices}")

        convolution = self.model.wav2vec2.encoder.pos_conv_embed.conv
        with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as checkpoint:
            saved_g = checkpoint.get_tensor(
                "wav2vec2.encoder.pos_conv_embed.conv.weight_g"
            )
            saved_v = checkpoint.get_tensor(
                "wav2vec2.encoder.pos_conv_embed.conv.weight_v"
            )
        exact_g = torch.equal(convolution.parametrizations.weight.original0.detach(), saved_g)
        exact_v = torch.equal(convolution.parametrizations.weight.original1.detach(), saved_v)
        if not exact_g or not exact_v:
            raise ValueError("MMS positional convolution did not load exactly")
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.load_validation = {
            "pinned_local_snapshot": True,
            "use_safetensors": True,
            "selected_class_indices": list(actual_indices),
            "legacy_weight_g_exact": exact_g,
            "legacy_weight_v_exact": exact_v,
        }

    def predict_batch(self, waveforms: Sequence[torch.Tensor]) -> list[Prediction]:
        inputs = self.processor(
            [waveform.numpy() for waveform in waveforms],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
        output = []
        for logit_row, prob_row in zip(logits, probabilities, strict=True):
            native_mms = self.index_to_code[int(logit_row.argmax())]
            native = MMS_TO_REPO_CODE.get(native_mms, native_mms)
            output.append(
                make_prediction(
                    native,
                    [
                        float(prob_row[self.code_to_index[REPO_TO_MMS_CODE[code]]])
                        for code in LANGUAGE_CODES
                    ],
                )
            )
        return output


class AmberNetPredictor:
    name = "ambernet"
    prediction_space = "native_107way_softmax; selected_7way conditional view"
    batch_size = 8

    def __init__(
        self,
        snapshot: Path,
        artifact: Mapping[str, Any],
        parity: Mapping[str, Any],
    ) -> None:
        self.model_id = artifact["model_id"]
        self.revision = artifact["revision"]
        module_path = snapshot / "modeling_ambernet.py"
        module_name = "english_teacher_bakeoff_ambernet_modeling"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError("could not load pinned AmberNet implementation")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.AmberNet.from_pretrained(str(snapshot))
        self.model.eval()
        self.labels = list(self.model.labels)
        self.code_to_index = {code: index for index, code in enumerate(self.labels)}
        if set(LANGUAGE_CODES) - set(self.code_to_index):
            raise ValueError("AmberNet snapshot lacks a selected language")
        expected_indices = (20, 35, 63, 9, 91, 92, 31)
        actual_indices = tuple(self.code_to_index[code] for code in LANGUAGE_CODES)
        if actual_indices != expected_indices:
            raise ValueError(f"AmberNet language map changed: {actual_indices}")
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.load_validation = {
            "pinned_local_snapshot": True,
            "selected_class_indices": list(actual_indices),
            "official_ngc_parity": parity,
        }

    def predict_batch(self, waveforms: Sequence[torch.Tensor]) -> list[Prediction]:
        lengths = torch.tensor([len(waveform) for waveform in waveforms], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        with torch.inference_mode():
            logits, _ = self.model(padded, lengths)
        probabilities = torch.softmax(logits, dim=-1)
        output = []
        for logit_row, prob_row in zip(logits, probabilities, strict=True):
            native = self.labels[int(logit_row.argmax())]
            output.append(
                make_prediction(
                    native,
                    [float(prob_row[self.code_to_index[code]]) for code in LANGUAGE_CODES],
                )
            )
        return output


def make_predictor(
    name: str,
    snapshot: Path,
    artifact: Mapping[str, Any],
    ambernet_parity: Mapping[str, Any] | None,
) -> Predictor:
    if name == "ecapa":
        return ECAPAPredictor(snapshot, artifact)
    if name == "whisper-small":
        return WhisperPredictor(snapshot, artifact)
    if name == "mms-lid-126":
        return MMSPredictor(snapshot, artifact)
    if name == "ambernet":
        if ambernet_parity is None:
            raise ValueError("AmberNet parity evidence is required")
        return AmberNetPredictor(snapshot, artifact, ambernet_parity)
    raise ValueError(name)


def process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def validate_fleurs_manifest(fleurs_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = fleurs_dir / "manifest.json"
    if file_sha256(manifest_path) != EXPECTED_FLEURS_FILE_SHA256:
        raise ValueError("FLEURS manifest file differs from the frozen cohort")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_hash = manifest.get("manifest_sha256")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_sha256(payload) != stored_hash or stored_hash != EXPECTED_FLEURS_MANIFEST_SHA256:
        raise ValueError("FLEURS manifest canonical identity differs")
    identity = manifest["identity"]
    if (
        identity["dataset_id"] != "google/fleurs"
        or identity["revision"] != FLEURS_REVISION
        or identity["split"] != "validation"
        or identity["sample_per_language"] != 100
    ):
        raise ValueError("FLEURS manifest semantics differ")
    records = [record for record in manifest["records"] if record["language"] == "en"]
    if len(records) != 100 or len({record["clip_id"] for record in records}) != 100:
        raise ValueError("expected exactly 100 unique frozen FLEURS-English clips")
    for record in records:
        path = fleurs_dir / record["audio_path"]
        actual = file_record(path)
        expected = {"bytes": record["audio_bytes"], "sha256": record["audio_sha256"]}
        if actual != expected:
            raise ValueError(f"FLEURS audio differs for {record['clip_id']}")
        if record["sample_rate"] != SAMPLE_RATE or record["channels"] != 1:
            raise ValueError("unexpected FLEURS audio format")
    return manifest, records


def add_views(
    cases: list[Case],
    *,
    cohort: str,
    clip_id: str,
    expected: str,
    waveform: torch.Tensor,
    metadata: Mapping[str, Any],
    views: Sequence[str] = tuple(VIEWS),
) -> None:
    for view in views:
        seconds = VIEWS[view]
        if seconds is not None:
            required = seconds * SAMPLE_RATE
            if len(waveform) < required:
                continue
            audio = waveform[:required]
        else:
            audio = waveform
        cases.append(
            Case(
                key=f"{cohort}:{clip_id}:{view}",
                cohort=cohort,
                clip_id=clip_id,
                view=view,
                expected=expected,
                waveform=audio,
                metadata=dict(metadata),
            )
        )


def expected_switch_language(item: Mapping[str, Any], seconds: float) -> str:
    for segment in item["segments"]:
        if segment["start_seconds"] <= seconds < segment["end_seconds"]:
            return segment["language"]
    return item["segments"][-1]["language"]


def build_cases(
    manifest_path: Path, fleurs_dir: Path
) -> tuple[list[Case], dict[str, Any]]:
    manifest_snapshot = capture_manifest_snapshot(manifest_path)
    if manifest_snapshot.identity()["manifest_file_sha256"] != EXPECTED_MAIN_MANIFEST_SHA256:
        raise ValueError("main manifest differs from the selected pipeline generation")
    records = manifest_snapshot.records_copy()
    train = [
        record
        for record in records
        if record["split"] == "train" and record["language"] in LANGUAGE_CODES
    ]
    heldout = [record for record in records if record["split"] == "heldout"]
    switches = [record for record in records if record["split"] == "switch"]
    if len(train) != 70 or Counter(record["language"] for record in train) != Counter(
        {code: 10 for code in LANGUAGE_CODES}
    ):
        raise ValueError("expected the main balanced 70-clip monolingual training set")
    if len(heldout) != 21 or Counter(record["language"] for record in heldout) != Counter(
        {code: 3 for code in LANGUAGE_CODES}
    ):
        raise ValueError("expected the main 21-clip held-out set")
    if {record["id"] for record in switches} != {
        "switch_hi_en_eval",
        "switch_en_hi_eval",
    }:
        raise ValueError("expected both main held-out switch clips")

    cases: list[Case] = []
    main_audio_identity: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for cohort, selected in (("train_mono", train), ("heldout_mono", heldout)):
        for item in selected:
            path = resolve_audio_path(item, manifest_snapshot)
            actual = file_record(path)
            if actual["sha256"] != item["audio_sha256"]:
                raise ValueError(f"main audio differs for {item['id']}")
            main_audio_identity[item["id"]] = actual
            waveform = load_audio(path)
            add_views(
                cases,
                cohort=cohort,
                clip_id=item["id"],
                expected=item["language"],
                waveform=waveform,
                metadata={
                    "speaker_id": item["speaker_id"],
                    "source": item["source"],
                    "num_samples": len(waveform),
                },
            )

    switch_metadata: dict[str, Any] = {}
    for item in switches:
        path = resolve_audio_path(item, manifest_snapshot)
        actual = file_record(path)
        if actual["sha256"] != item["audio_sha256"]:
            raise ValueError(f"main switch audio differs for {item['id']}")
        main_audio_identity[item["id"]] = actual
        waveform = load_audio(path)
        frames = feature_frame_count(len(waveform))
        anchors = np.arange(0, frames, TEACHER_HOP_FRAMES, dtype=np.int64)
        if int(anchors[-1]) != frames - 1:
            anchors = np.append(anchors, frames - 1)
        source = item["segments"][0]["language"]
        target = item["segments"][1]["language"]
        boundary = float(item["segments"][0]["end_seconds"])
        switch_metadata[item["id"]] = {
            "source": source,
            "target": target,
            "boundary_seconds": boundary,
            "anchors": len(anchors),
        }
        for frame in anchors:
            anchor_frame = int(frame)
            anchor_seconds = (anchor_frame * HOP_LENGTH + WIN_LENGTH) / SAMPLE_RATE
            cases.append(
                Case(
                    key=f"switch:{item['id']}:{anchor_frame}",
                    cohort="switch",
                    clip_id=item["id"],
                    view="anchor",
                    expected=expected_switch_language(item, anchor_seconds),
                    waveform=extract_window(waveform, anchor_frame),
                    metadata={
                        "anchor_frame": anchor_frame,
                        "anchor_seconds": anchor_seconds,
                        "boundary_seconds": boundary,
                        "source": source,
                        "target": target,
                    },
                )
            )

    fleurs_manifest, fleurs_records = validate_fleurs_manifest(fleurs_dir)
    external_audio_identity: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in fleurs_records:
        path = fleurs_dir / record["audio_path"]
        external_audio_identity[record["clip_id"]] = file_record(path)
        waveform = load_audio(path)
        if len(waveform) != record["num_samples"]:
            raise ValueError(f"FLEURS sample count differs for {record['clip_id']}")
        add_views(
            cases,
            cohort="fleurs_en_validation",
            clip_id=record["clip_id"],
            expected="en",
            waveform=waveform,
            metadata={
                "config": record["config"],
                "dataset_row_id": record["dataset_row_id"],
                "num_samples": len(waveform),
            },
            views=("1s", "2s", "4s"),
        )

    if len({case.key for case in cases}) != len(cases):
        raise ValueError("case keys are not unique")
    input_identity = {
        "main_manifest": manifest_snapshot.identity(),
        "main_audio": main_audio_identity,
        "heldout_ids": [record["id"] for record in heldout],
        "switches": switch_metadata,
        "fleurs": {
            "dataset_id": "google/fleurs",
            "revision": FLEURS_REVISION,
            "split": "validation",
            "manifest_file": file_record(fleurs_dir / "manifest.json"),
            "manifest_sha256": fleurs_manifest["manifest_sha256"],
            "english_audio": external_audio_identity,
        },
        "case_count": len(cases),
        "case_keys_sha256": canonical_sha256([case.key for case in cases]),
        "timed_audio_seconds": sum(len(case.waveform) for case in cases) / SAMPLE_RATE,
    }
    input_identity["sha256"] = canonical_sha256(input_identity)
    return cases, input_identity


def assert_input_identity_unchanged(
    expected: Mapping[str, Any], manifest_path: Path, fleurs_dir: Path
) -> None:
    if file_sha256(manifest_path) != expected["main_manifest"]["manifest_file_sha256"]:
        raise RuntimeError("main manifest changed during the run")
    snapshot = capture_manifest_snapshot(manifest_path)
    by_id = {record["id"]: record for record in snapshot.records_copy()}
    for clip_id, record in expected["main_audio"].items():
        if file_record(resolve_audio_path(by_id[clip_id], snapshot)) != record:
            raise RuntimeError(f"main audio changed during the run: {clip_id}")
    if file_record(fleurs_dir / "manifest.json") != expected["fleurs"]["manifest_file"]:
        raise RuntimeError("FLEURS manifest changed during the run")
    manifest = json.loads((fleurs_dir / "manifest.json").read_text(encoding="utf-8"))
    by_external_id = {record["clip_id"]: record for record in manifest["records"]}
    for clip_id, record in expected["fleurs"]["english_audio"].items():
        if file_record(fleurs_dir / by_external_id[clip_id]["audio_path"]) != record:
            raise RuntimeError(f"FLEURS audio changed during the run: {clip_id}")


def prediction_record(case: Case, prediction: Prediction) -> dict[str, Any]:
    output = {
        "key": case.key,
        "cohort": case.cohort,
        "clip_id": case.clip_id,
        "view": case.view,
        "expected": case.expected,
        "native_predicted": prediction.native_code,
        "restricted_predicted": prediction.restricted_code,
        "selected_language_mass": prediction.selected_mass,
        "selected_absolute": dict(zip(LANGUAGE_CODES, prediction.selected_absolute, strict=True)),
        "selected_conditional": dict(
            zip(LANGUAGE_CODES, prediction.selected_conditional, strict=True)
        ),
        "expected_absolute_probability": prediction.selected_absolute[
            LANGUAGE_CODES.index(case.expected)
        ],
        "expected_conditional_probability": prediction.selected_conditional[
            LANGUAGE_CODES.index(case.expected)
        ],
        "source_samples": len(case.waveform),
    }
    output.update(case.metadata)
    return output


def run_prediction_batches(
    predictor: Predictor, cases: Sequence[Case]
) -> tuple[list[dict[str, Any]], int]:
    ordered_indices = sorted(range(len(cases)), key=lambda index: len(cases[index].waveform))
    predictions: list[Prediction | None] = [None] * len(cases)
    batches = 0
    for start in range(0, len(ordered_indices), predictor.batch_size):
        indices = ordered_indices[start : start + predictor.batch_size]
        values = predictor.predict_batch([cases[index].waveform for index in indices])
        if len(values) != len(indices):
            raise ValueError("predictor returned the wrong number of rows")
        for index, value in zip(indices, values, strict=True):
            predictions[index] = value
        batches += 1
        if batches % 50 == 0:
            print(f"  completed {batches} batches", flush=True)
    if any(value is None for value in predictions):
        raise AssertionError("missing prediction")
    return [
        prediction_record(case, value)
        for case, value in zip(cases, predictions, strict=True)
        if value is not None
    ], batches


def quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty vector")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def classification_cell(
    rows: Sequence[Mapping[str, Any]], prediction_field: str, expected_languages: Sequence[str]
) -> dict[str, Any]:
    per_language = OrderedDict()
    for code in expected_languages:
        selected = [row for row in rows if row["expected"] == code]
        correct = sum(row[prediction_field] == code for row in selected)
        per_language[code] = {
            "correct": correct,
            "total": len(selected),
            "recall": None if not selected else correct / len(selected),
        }
    total = len(rows)
    pooled_correct = sum(row[prediction_field] == row["expected"] for row in rows)
    eligible_recalls = [
        value["recall"] for value in per_language.values() if value["recall"] is not None
    ]
    return {
        "correct": pooled_correct,
        "total": total,
        "accuracy": pooled_correct / total,
        "language_macro_accuracy": float(statistics.mean(eligible_recalls)),
        "all_expected_languages_evaluable": all(
            value["total"] > 0 for value in per_language.values()
        ),
        "per_language": per_language,
    }


def classification_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: OrderedDict[str, Any] = OrderedDict()
    cohorts = ("train_mono", "heldout_mono", "fleurs_en_validation")
    for cohort in cohorts:
        expected_languages: Sequence[str] = (
            ("en",) if cohort == "fleurs_en_validation" else LANGUAGE_CODES
        )
        output[cohort] = OrderedDict()
        for view in VIEWS:
            selected = [
                row for row in rows if row["cohort"] == cohort and row["view"] == view
            ]
            if not selected:
                continue
            output[cohort][view] = {
                "restricted_7way": classification_cell(
                    selected, "restricted_predicted", expected_languages
                ),
                "native_full_space": classification_cell(
                    selected, "native_predicted", expected_languages
                ),
                "selected_language_mass": quantiles(
                    [float(row["selected_language_mass"]) for row in selected]
                ),
                "expected_absolute_probability": quantiles(
                    [float(row["expected_absolute_probability"]) for row in selected]
                ),
                "expected_conditional_probability": quantiles(
                    [float(row["expected_conditional_probability"]) for row in selected]
                ),
            }
        primary_views = PRIMARY_VIEWS_BY_COHORT[cohort]
        output[cohort]["primary_composite"] = {
            "views": list(primary_views),
            "restricted_language_macro_accuracy": float(
                statistics.mean(
                    output[cohort][view]["restricted_7way"]["language_macro_accuracy"]
                    for view in primary_views
                )
            ),
            "native_language_macro_accuracy": float(
                statistics.mean(
                    output[cohort][view]["native_full_space"]["language_macro_accuracy"]
                    for view in primary_views
                )
            ),
        }
    return output


def maximal_runs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    runs = []
    start = 0
    for index in range(1, len(rows) + 1):
        if index == len(rows) or rows[index]["restricted_predicted"] != rows[start]["restricted_predicted"]:
            runs.append(
                {
                    "code": rows[start]["restricted_predicted"],
                    "start_index": start,
                    "stop_index_exclusive": index,
                    "anchors": index - start,
                    "start_seconds": rows[start]["anchor_seconds"],
                    "confirmation_seconds": (
                        rows[start + 2]["anchor_seconds"] if index - start >= 3 else None
                    ),
                }
            )
            start = index
    return runs


def switch_clip_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["anchor_frame"])
    if not ordered:
        raise ValueError("empty switch trace")
    source = ordered[0]["source"]
    target = ordered[0]["target"]
    boundary = float(ordered[0]["boundary_seconds"])
    runs = maximal_runs(ordered)
    stable = [run for run in runs if run["anchors"] >= 3]
    qualified_before = [run for run in stable if run["start_seconds"] < boundary]
    state_at_boundary = None if not qualified_before else qualified_before[-1]["code"]
    stable_after = [run for run in stable if run["start_seconds"] >= boundary]
    first_after = None if not stable_after else stable_after[0]
    source_precondition = state_at_boundary == source
    outcome = "miss"
    detected = False
    if not source_precondition:
        outcome = "precondition_failure"
    elif first_after is not None and first_after["code"] == target:
        outcome = "detected"
        detected = True
    elif first_after is not None:
        outcome = "wrong_stable_event"

    def lag(seconds: float | None, add_future: bool = False) -> float | None:
        if seconds is None:
            return None
        available = seconds + (TEACHER_FUTURE_MS / 1_000 if add_future else 0)
        return 1_000 * (available - boundary)

    target_run = first_after if detected else None
    expected_correct = sum(
        row["restricted_predicted"] == row["expected"] for row in ordered
    )
    raw_changes = sum(
        ordered[index]["restricted_predicted"]
        != ordered[index - 1]["restricted_predicted"]
        for index in range(1, len(ordered))
    )
    return {
        "clip_id": ordered[0]["clip_id"],
        "source": source,
        "target": target,
        "boundary_seconds": boundary,
        "source_precondition": source_precondition,
        "qualified_state_at_boundary": state_at_boundary,
        "outcome": outcome,
        "semantic_onset_lag_ms": lag(
            None if target_run is None else target_run["start_seconds"]
        ),
        "onset_availability_lag_ms": lag(
            None if target_run is None else target_run["start_seconds"], True
        ),
        "semantic_confirmation_lag_ms": lag(
            None if target_run is None else target_run["confirmation_seconds"]
        ),
        "confirmation_availability_lag_ms": lag(
            None if target_run is None else target_run["confirmation_seconds"], True
        ),
        "anchor_accuracy": expected_correct / len(ordered),
        "anchor_correct": expected_correct,
        "anchor_total": len(ordered),
        "raw_changes": raw_changes,
        "stable_runs": stable,
    }


def switch_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    clip_ids = sorted({row["clip_id"] for row in rows if row["cohort"] == "switch"})
    per_clip = OrderedDict(
        (
            clip_id,
            switch_clip_summary(
                [row for row in rows if row["cohort"] == "switch" and row["clip_id"] == clip_id]
            ),
        )
        for clip_id in clip_ids
    )
    detected = [value for value in per_clip.values() if value["outcome"] == "detected"]
    metric_names = (
        "semantic_onset_lag_ms",
        "onset_availability_lag_ms",
        "semantic_confirmation_lag_ms",
        "confirmation_availability_lag_ms",
    )
    medians = {
        name: (
            None
            if not detected
            else float(statistics.median(value[name] for value in detected))
        )
        for name in metric_names
    }
    return {
        "scorer": "maximal restricted-top1 runs; 3 anchors (500 ms) qualify a state",
        "anchor_hop_ms": TEACHER_HOP_FRAMES * HOP_LENGTH * 1_000 / SAMPLE_RATE,
        "right_context_ms": TEACHER_FUTURE_MS,
        "detected": len(detected),
        "total": len(per_clip),
        "medians_over_detected": medians,
        "per_clip": per_clip,
    }


def english_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    full_train_en = [
        row
        for row in rows
        if row["cohort"] == "train_mono" and row["view"] == "full" and row["expected"] == "en"
    ]
    by_id = {row["clip_id"]: row for row in full_train_en}
    targeted = [by_id[clip_id] for clip_id in TARGETED_ENGLISH_IDS]
    edge = [row for row in full_train_en if row.get("speaker_id") == "en-IN-PrabhatNeural"]
    gtts = [row for row in full_train_en if row.get("speaker_id") == "gtts:en:default"]

    def subset(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        correct = sum(row["restricted_predicted"] == "en" for row in selected)
        return {
            "correct": correct,
            "total": len(selected),
            "accuracy": correct / len(selected),
            "mean_absolute_p_en": float(
                statistics.mean(row["selected_absolute"]["en"] for row in selected)
            ),
            "mean_conditional_p_en": float(
                statistics.mean(row["selected_conditional"]["en"] for row in selected)
            ),
            "mean_selected_mass": float(
                statistics.mean(row["selected_language_mass"] for row in selected)
            ),
        }

    return {
        "all_10_english": subset(full_train_en),
        "edge_prabhat_5": subset(edge),
        "gtts_5": subset(gtts),
        "targeted_ecapa_wrong_4": {
            **subset(targeted),
            "rows": [
                {
                    key: row[key]
                    for key in (
                        "clip_id",
                        "restricted_predicted",
                        "native_predicted",
                        "selected_language_mass",
                        "selected_absolute",
                        "selected_conditional",
                    )
                }
                for row in targeted
            ],
        },
    }


def evaluate_model(
    name: str,
    cases: Sequence[Case],
    snapshot: Path,
    artifact: Mapping[str, Any],
    ambernet_parity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    load_start = time.perf_counter()
    predictor = make_predictor(name, snapshot, artifact, ambernet_parity)
    load_seconds = time.perf_counter() - load_start
    predictor.predict_batch([cases[0].waveform])

    wall_start = time.perf_counter()
    cpu_start = process_cpu_seconds()
    rows, batches = run_prediction_batches(predictor, cases)
    cpu_seconds = process_cpu_seconds() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    audio_seconds = sum(len(case.waveform) for case in cases) / SAMPLE_RATE

    result = {
        "model_id": predictor.model_id,
        "revision": predictor.revision,
        "artifact": artifact,
        "parameter_count": predictor.parameter_count,
        "prediction_space": predictor.prediction_space,
        "load_validation": predictor.load_validation,
        "classification": classification_summary(rows),
        "english_diagnostics": english_diagnostics(rows),
        "switch": switch_summary(rows),
        "timing": {
            "model_load_wall_seconds": load_seconds,
            "model_load_included": False,
            "warmup_included": False,
            "preprocessing_included": True,
            "batch_size": predictor.batch_size,
            "timed_batches": batches,
            "timed_requests": len(cases),
            "timed_audio_seconds": audio_seconds,
            "wall_seconds": wall_seconds,
            "process_cpu_seconds": cpu_seconds,
            "wall_rtf": wall_seconds / audio_seconds,
            "process_cpu_rtf": cpu_seconds / audio_seconds,
        },
        "predictions": rows,
    }
    del predictor
    gc.collect()
    return result


def metric(model: Mapping[str, Any], cohort: str, view: str, name: str) -> float:
    return float(model["classification"][cohort][view]["restricted_7way"][name])


def primary_composite(model: Mapping[str, Any], cohort: str) -> float:
    return float(
        model["classification"][cohort]["primary_composite"][
            "restricted_language_macro_accuracy"
        ]
    )


def build_verdict(models: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    baseline = models["ecapa"]
    baseline_switch = baseline["switch"]["medians_over_detected"][
        "confirmation_availability_lag_ms"
    ]
    comparisons: OrderedDict[str, Any] = OrderedDict()
    for name, candidate in models.items():
        if name == "ecapa":
            continue
        checks = OrderedDict(
            [
                (
                    "fix_at_least_3_of_4_ecapa_wrong_edge_english",
                    candidate["english_diagnostics"]["targeted_ecapa_wrong_4"]["correct"] >= 3,
                ),
                (
                    "english_training_quality_floor_at_least_8_of_10",
                    candidate["english_diagnostics"]["all_10_english"]["correct"] >= 8,
                ),
                (
                    "all_train_full_macro_within_2_points_of_ecapa",
                    metric(candidate, "train_mono", "full", "language_macro_accuracy")
                    >= metric(baseline, "train_mono", "full", "language_macro_accuracy") - 0.02,
                ),
                (
                    "main_heldout_primary_composite_within_2_points_of_ecapa",
                    primary_composite(candidate, "heldout_mono")
                    >= primary_composite(baseline, "heldout_mono") - 0.02,
                ),
                (
                    "fleurs_english_primary_composite_within_2_points_of_ecapa",
                    primary_composite(candidate, "fleurs_en_validation")
                    >= primary_composite(baseline, "fleurs_en_validation") - 0.02,
                ),
                (
                    "both_switch_directions_detected",
                    candidate["switch"]["detected"] == candidate["switch"]["total"] == 2,
                ),
                (
                    "switch_confirmation_availability_no_more_than_100ms_slower",
                    baseline_switch is not None
                    and candidate["switch"]["medians_over_detected"][
                        "confirmation_availability_lag_ms"
                    ]
                    is not None
                    and candidate["switch"]["medians_over_detected"][
                        "confirmation_availability_lag_ms"
                    ]
                    <= baseline_switch + 100,
                ),
                ("wall_rtf_at_most_0_10", candidate["timing"]["wall_rtf"] <= 0.10),
            ]
        )
        numerical_pass = all(checks.values())
        license_gate = candidate["artifact"]["license_gate"] == "permissive"
        comparisons[name] = {
            "checks": checks,
            "numerical_pass": numerical_pass,
            "license_gate_pass": license_gate,
            "immediate_teacher_switch_pass": numerical_pass and license_gate,
        }

    immediate = [
        name for name, value in comparisons.items() if value["immediate_teacher_switch_pass"]
    ]
    numerical_only = [
        name
        for name, value in comparisons.items()
        if value["numerical_pass"] and not value["license_gate_pass"]
    ]
    if immediate:
        verdict = "adopt"
        reason = f"{immediate} pass every predeclared numerical, timing, switch, and license gate"
    elif numerical_only:
        verdict = "inconclusive"
        reason = (
            f"{numerical_only} pass numerical gates but cannot be recommended for an immediate "
            "switch under the recorded license status"
        )
    else:
        verdict = "reject"
        reason = "no challenger passes the predeclared numerical and runtime gate set"
    return {
        "verdict": verdict,
        "reason": reason,
        "comparisons_to_ecapa": comparisons,
        "recommendation": (
            "retain ECAPA" if verdict != "adopt" else f"adopt {immediate[0]} as the teacher"
        ),
    }


def assert_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_json(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}")


def validate_ecapa_control(model: Mapping[str, Any]) -> None:
    rows = model["english_diagnostics"]["targeted_ecapa_wrong_4"]["rows"]
    observed = {row["clip_id"]: row["restricted_predicted"] for row in rows}
    if observed != EXPECTED_ECAPA_TARGETED_PREDICTIONS:
        raise ValueError(f"fresh ECAPA targeted predictions differ: {observed}")
    if model["english_diagnostics"]["all_10_english"]["correct"] != 6:
        raise ValueError("fresh ECAPA English training accuracy is not 6/10")
    heldout_full = model["classification"]["heldout_mono"]["full"]["restricted_7way"]
    if heldout_full["correct"] != heldout_full["total"] or heldout_full["total"] != 21:
        raise ValueError("fresh ECAPA no longer reaches 21/21 on natural-full heldout clips")


def main() -> None:
    args = parse_args()
    if args.threads != 6:
        raise ValueError("TEAM.md requires exactly six Torch threads")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)

    manifest_path = args.manifest.resolve()
    fleurs_dir = args.fleurs_dir.resolve()
    launch_source = source_identity()
    cases, input_identity = build_cases(manifest_path, fleurs_dir)
    counts = Counter((case.cohort, case.view) for case in cases)
    print(
        f"loaded {len(cases)} requests / {input_identity['timed_audio_seconds']:.2f} s: "
        f"{dict(counts)}",
        flush=True,
    )

    snapshots: dict[str, Path] = {}
    artifacts: OrderedDict[str, Any] = OrderedDict()
    for name in MODEL_REGISTRY:
        print(f"verifying pinned artifact {name}", flush=True)
        snapshots[name], artifacts[name] = verify_snapshot(
            name, local_files_only=args.offline
        )
    ambernet_parity = validate_ambernet_parity(
        snapshots["ambernet"], offline=args.offline
    )

    experiment_identity = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "source": launch_source,
        "inputs": input_identity,
        "artifacts": artifacts,
        "ambernet_parity": ambernet_parity,
        "language_codes": list(LANGUAGE_CODES),
        "views": {key: value for key, value in VIEWS.items()},
        "prefix_policy": "true leading prefix; ineligible if source is one sample short; no padding",
        "primary_views_by_cohort": {
            key: list(value) for key, value in PRIMARY_VIEWS_BY_COHORT.items()
        },
        "targeted_english_ids": list(TARGETED_ENGLISH_IDS),
        "switch_scorer": {
            "space": "restricted_7way",
            "run_qualification_anchors": 3,
            "hop_frames": TEACHER_HOP_FRAMES,
            "future_ms": TEACHER_FUTURE_MS,
        },
        "threads": 6,
        "seed": 0,
        "runtime": {
            "python": sys.version,
            "packages": package_versions(),
        },
    }
    run_id = canonical_sha256(experiment_identity)
    progress_path = REPO_ROOT / "data/experiments/english-teacher-bakeoff/progress.json"
    progress: dict[str, Any] = {"run_id": run_id, "models": {}}
    if progress_path.is_file() and not args.fresh:
        candidate = json.loads(progress_path.read_text(encoding="utf-8"))
        if candidate.get("run_id") != run_id:
            raise ValueError("cached model rows use a different experiment identity; pass --fresh")
        progress = candidate

    for index, name in enumerate(args.models, start=1):
        if name in progress["models"] and not args.fresh:
            print(f"[{index}/{len(args.models)}] reusing identity-matched {name}", flush=True)
            continue
        print(f"[{index}/{len(args.models)}] evaluating {name}", flush=True)
        progress["models"][name] = evaluate_model(
            name,
            cases,
            snapshots[name],
            artifacts[name],
            ambernet_parity if name == "ambernet" else None,
        )
        progress["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(progress_path, progress)
        current = progress["models"][name]
        print(
            f"  {name}: Edge-English {current['english_diagnostics']['edge_prabhat_5']['correct']}/5; "
            f"heldout composite={100*primary_composite(current, 'heldout_mono'):.2f}%; "
            f"FLEURS-en composite={100*primary_composite(current, 'fleurs_en_validation'):.2f}%; "
            f"RTF={current['timing']['wall_rtf']:.4f}",
            flush=True,
        )

    missing = set(MODEL_REGISTRY) - set(progress["models"])
    if missing:
        raise ValueError(f"final publication requires all four model rows, missing {sorted(missing)}")
    validate_ecapa_control(progress["models"]["ecapa"])
    verdict = build_verdict(progress["models"])
    end_source = source_identity()
    if end_source != launch_source:
        raise RuntimeError("experiment or imported main source changed during the run")
    assert_input_identity_unchanged(input_identity, manifest_path, fleurs_dir)

    prediction_payload = {
        name: {
            "classification": model["classification"],
            "english_diagnostics": model["english_diagnostics"],
            "switch": model["switch"],
            "predictions": model["predictions"],
        }
        for name, model in progress["models"].items()
    }
    output = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "experiment_identity": experiment_identity,
        "source_unchanged_at_publication": True,
        "inputs_unchanged_at_publication": True,
        "prediction_payload_sha256": canonical_sha256(prediction_payload),
        "request_counts": {
            f"{cohort}:{view}": count for (cohort, view), count in sorted(counts.items())
        },
        "models": progress["models"],
        "verdict": verdict,
        "limitations": [
            "FLEURS validation was already used by prior checkpoint/loss experiments and is descriptive, not untouched confirmation",
            "FLEURS-English is monolingual read speech; no pinned local Svarah cohort was available",
            "the main corpus is tiny synthetic TTS and the two switch clips are mirrored fixtures, not independent population trials",
            "AmberNet inference weights and explicit config fields match official NGC exactly, but standalone-code versus NeMo output parity was not executed",
            "selected-language mass is model-space-specific: Whisper normalizes across language tokens while ECAPA/AmberNet/MMS use native class softmaxes",
        ],
    }
    assert_finite_json(output)
    output_path = Path(__file__).with_name("results.json")
    atomic_write_json(output_path, output)
    print(f"verdict={verdict['verdict']}: {verdict['reason']}", flush=True)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
