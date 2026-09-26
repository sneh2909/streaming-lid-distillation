"""Run the slow Whisper-turbo teacher on Modal L4 GPUs (fan-out), same code as local.

    modal run scripts/modal_whisper.py

The audio volume is mounted at the repo's absolute data path, so manifest paths resolve unchanged.
Outputs land locally in data/targets/whisper-turbo/causal.pt and results/teachers/whisper-turbo*.
"""
import json
import os
import subprocess
from pathlib import Path

import modal

REPO = "/mnt/d/Work/Projects/asr-navana"
DATA = f"{REPO}/data"
MODEL = "openai/whisper-large-v3-turbo"

vol = modal.Volume.from_name("slid-audio-v2", create_if_missing=True)


def _download():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "torchaudio", "transformers", "numpy", "soundfile",
                 "huggingface_hub", "tqdm", "sentencepiece", "safetensors")
    .run_function(_download)
    .add_local_python_source("slid")
    .add_local_dir(f"{REPO}/scripts", f"{REPO}/scripts")
)
app = modal.App("slid-whisper", image=image)


@app.function(volumes={DATA: vol}, timeout=3600)
def extract(manifests: dict[str, str]) -> int:
    import tarfile
    marker = Path(DATA) / ".extracted"
    if not marker.exists():
        with tarfile.open(f"{DATA}/slid_audio.tar") as tar:
            for m in tar.getmembers():
                m.name = m.name.removeprefix("data/")
                tar.extract(m, DATA, filter="data")
        marker.write_text("ok")
    (Path(DATA) / "manifests").mkdir(exist_ok=True)
    for name, text in manifests.items():
        (Path(DATA) / "manifests" / name).write_text(text)
    vol.commit()
    return sum(1 for _ in Path(DATA).rglob("*.wav"))


@app.function(gpu="L4", volumes={DATA: vol}, timeout=3600, max_containers=10)
def targets_shard(paths: list[str], window_s: float = 3.0, kind: str = "causal") -> dict:
    import numpy as np
    from slid.audio import load_wav
    from slid.targets import build_targets
    from slid.teachers import Whisper
    t = Whisper()
    t.batch_size = 32
    out = {}
    for p in paths:
        frames, q = build_targets(t, load_wav(p), kind, window_s=window_s)
        out[p] = (frames.astype(np.int64), q)
    return out


@app.function(gpu="L4", volumes={DATA: vol}, timeout=3600)
def bakeoff(extra_args: list[str]) -> dict[str, bytes]:
    subprocess.run(["python", f"{REPO}/scripts/teacher_bakeoff.py", "--teacher", "whisper-turbo", *extra_args],
                   check=True, cwd=REPO, env={**os.environ, "PYTHONPATH": "/root"})
    return {p.name: p.read_bytes() for p in Path(f"{REPO}/results/teachers").glob("whisper-turbo*")}


@app.function(gpu="L4", volumes={DATA: vol}, timeout=3600, secrets=[modal.Secret.from_name("hf-token")])
def bakeoff_indic(extra_args: list[str]) -> dict[str, bytes]:
    subprocess.run(["python", f"{REPO}/scripts/teacher_bakeoff.py", "--teacher", "indic-transcribe", *extra_args],
                   check=True, cwd=REPO, env={**os.environ, "PYTHONPATH": "/root"})
    return {p.name: p.read_bytes() for p in Path(f"{REPO}/results/teachers").glob("indic-transcribe*")}


@app.local_entrypoint()
def indic():
    root = Path(REPO)
    futs = [bakeoff_indic.spawn([]),
            bakeoff_indic.spawn(["--manifest", "train", "--max-per-lang", "25", "--no-switch"])]
    for fut in futs:
        for name, data in fut.get().items():
            (root / "results/teachers" / name).write_bytes(data)
            print("wrote results/teachers/" + name)


@app.local_entrypoint()
def main(shard_size: int = 100, window_s: float = 3.0, suffix: str = "", targets_only: bool = False,
         kind: str = "causal"):
    import torch
    root = Path(REPO)
    manifests = {f"{m}.jsonl": (root / f"data/manifests/{m}.jsonl").read_text()
                 for m in ("train", "train_mix", "eval", "switch")}
    print("wav files on volume:", extract.remote(manifests))

    futs = [] if targets_only else [bakeoff.spawn([]),
                                    bakeoff.spawn(["--manifest", "train", "--max-per-lang", "25", "--no-switch"])]

    paths = [json.loads(l)["path"] for text in manifests.values() for l in text.splitlines()]
    shards = [paths[i: i + shard_size] for i in range(0, len(paths), shard_size)]
    targets = {}
    for i, part in enumerate(targets_shard.map(shards, kwargs={"window_s": window_s, "kind": kind})):
        targets.update({p: (torch.from_numpy(f), torch.from_numpy(q)) for p, (f, q) in part.items()})
        print(f"shard {i + 1}/{len(shards)} done, {len(targets)} clips")
    out = root / "data/targets/whisper-turbo"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(targets, out / f"{kind}{suffix}.pt")
    print("saved", out / f"{kind}{suffix}.pt", len(targets))

    for fut in futs:
        for name, data in fut.get().items():
            (root / "results/teachers" / name).write_bytes(data)
            print("wrote results/teachers/" + name)
