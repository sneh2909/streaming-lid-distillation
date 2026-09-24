"""Manifest and cached-teacher-target loading."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .audio import LogMelFrontend, load_audio


def read_manifest(path: str | Path) -> list[dict]:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_audio_path(item: dict, manifest_path: str | Path) -> Path:
    return Path(manifest_path).parent / item["audio_path"]


class DistillationDataset(Dataset):
    """Tiny-data dataset: preload deterministic features and teacher posteriors."""

    def __init__(
        self,
        manifest_path: str | Path,
        targets_dir: str | Path,
        splits: Iterable[str],
    ) -> None:
        self.manifest_path = Path(manifest_path)
        wanted = set(splits)
        self.items = [
            item for item in read_manifest(manifest_path) if item["split"] in wanted
        ]
        if not self.items:
            raise ValueError(f"no manifest items for splits {sorted(wanted)}")
        frontend = LogMelFrontend().eval()
        self.examples: list[dict] = []
        with torch.inference_mode():
            for item in self.items:
                waveform = load_audio(resolve_audio_path(item, self.manifest_path))
                features = frontend(waveform).squeeze(0)
                target_path = Path(targets_dir) / f"{item['id']}.npz"
                if not target_path.exists():
                    raise FileNotFoundError(f"missing teacher target: {target_path}")
                with np.load(target_path) as target_file:
                    target = torch.from_numpy(
                        target_file["teacher_soft_targets"].copy()
                    ).float()
                if len(features) != len(target):
                    raise ValueError(
                        f"frame mismatch for {item['id']}: features={len(features)}, targets={len(target)}"
                    )
                self.examples.append(
                    {"features": features, "targets": target, "item": item}
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def collate_distillation_batch(examples: list[dict]) -> dict:
    lengths = torch.tensor(
        [len(example["features"]) for example in examples], dtype=torch.long
    )
    return {
        "features": pad_sequence(
            [example["features"] for example in examples], batch_first=True
        ),
        "targets": pad_sequence(
            [example["targets"] for example in examples], batch_first=True
        ),
        "lengths": lengths,
        "ids": [example["item"]["id"] for example in examples],
    }
