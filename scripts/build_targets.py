"""Precompute teacher targets for every clip, for each target kind.
Output: data/targets/<teacher>/<kind>.pt = {path: (frames int[K], q float[K, n_langs])}
"""
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from slid.audio import load_wav
from slid.targets import KINDS, build_targets
from slid.teachers import load_teacher

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ["train", "train_mix", "eval", "switch"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--window-s", type=float, default=3.0)
    ap.add_argument("--suffix", default="", help="e.g. _w1.5 -> saves causal_w1.5.pt")
    args = ap.parse_args()

    teacher = load_teacher(args.teacher)
    paths = [json.loads(l)["path"] for m in MANIFESTS for l in open(ROOT / f"data/manifests/{m}.jsonl")]
    out_dir = ROOT / "data/targets" / args.teacher
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in args.kinds:
        out_path = out_dir / f"{kind}{args.suffix}.pt"
        if out_path.exists():
            print("exists:", out_path)
            continue
        targets = {}
        for p in tqdm(paths, desc=kind):
            frames, q = build_targets(teacher, load_wav(p), kind, window_s=args.window_s)
            targets[p] = (torch.from_numpy(frames), torch.from_numpy(q))
        torch.save(targets, out_path)
        print("saved", out_path, len(targets))


if __name__ == "__main__":
    main()
