"""Loudness invariance: does the student's answer change when the same audio is turned up/down?

v1 data had a shortcut (quiet FLEURS English): Hindi turned down 30 dB was called English 82% of
the time. After level normalisation + random-gain augmentation this should be ~flat.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from slid.audio import load_wav
from slid.config import LANGS
from slid.student import load_student

ROOT = Path(__file__).resolve().parents[1]
GAINS_DB = (-30, -20, -10, 0, 10)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/final/student.pt")
    ap.add_argument("--out", default="results/final/level_test.json")
    args = ap.parse_args()
    m = load_student(ROOT / args.ckpt)
    items = [json.loads(l) for l in open(ROOT / "data/manifests/eval.jsonl", encoding="utf-8")]
    en = LANGS.index("en")
    res = {"ckpt": args.ckpt, "gains_db": GAINS_DB, "acc": {}, "called_english_non_en": {}}
    wavs = [(load_wav(it["path"]), LANGS.index(it["lang"])) for it in items]
    for g in GAINS_DB:
        preds = [int(m(torch.from_numpy(x * 10 ** (g / 20))[None], 4)[0, -1].argmax()) for x, _ in wavs]
        ys = [y for _, y in wavs]
        res["acc"][g] = float(np.mean([p == y for p, y in zip(preds, ys)]))
        res["called_english_non_en"][g] = float(np.mean([p == en for p, y in zip(preds, ys) if y != en]))
        print(f"gain {g:+3d} dB: accuracy {res['acc'][g]:.3f}   non-English clips called English "
              f"{res['called_english_non_en'][g]:.3f}")
    (ROOT / args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
