"""Shortcut probe: can a student name the language from the LEADING SILENCE of a clip?

A student trained on future-informed targets (full / centred) is rewarded for predicting a label
that the audio heard so far does not support, so it is pushed to exploit anything correlated with
that label - including per-language recording conditions in FLEURS. A student trained on
information-matched targets (causal / prefix) gets the teacher's own uncertain answer on silence,
so it has nothing to exploit. Chance = 1/7.
"""
import json
from pathlib import Path

import numpy as np
import torch

from slid.audio import SR, load_wav
from slid.config import LANGS
from slid.student import load_student

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["final", "ensemble_centered", "indic-transcribe_causal", "indic-transcribe_prefix",
          "indic-transcribe_centered", "indic-transcribe_full", "indic-transcribe_hybrid"]


def leading_silence_samples(x: np.ndarray, frame: int = 400, rel_db: float = -30.0) -> int:
    n = len(x) // frame
    e = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    speech = np.flatnonzero(20 * np.log10(e / e.max()) > rel_db)
    return int(speech[0] * frame) if len(speech) else 0


@torch.no_grad()
def main() -> None:
    items = [json.loads(l) for l in open(ROOT / "data/manifests/eval.jsonl", encoding="utf-8")]
    segs = []
    for it in items:
        x = load_wav(it["path"])
        s = leading_silence_samples(x)
        if s >= int(0.45 * SR):
            segs.append((x[:s], LANGS.index(it["lang"])))
    res = {"n_clips": len(segs), "chance": 1 / len(LANGS), "models": {}}
    for name in MODELS:
        m = load_student(ROOT / f"checkpoints/{name}/student.pt")
        ps = [m(torch.from_numpy(x)[None], 4)[0].softmax(-1).numpy()[-1] for x, _ in segs]
        acc = float(np.mean([p.argmax() == y for p, (_, y) in zip(ps, segs)]))
        res["models"][name] = {"acc_on_silence": acc, "mean_top_prob": float(np.mean([p.max() for p in ps]))}
        print(f"{name:26s} accuracy on leading silence {acc:.2f}  (chance {1 / len(LANGS):.2f})")
    (ROOT / "results/final/silence_test.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
