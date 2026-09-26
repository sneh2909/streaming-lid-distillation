"""Stage 2 of data prep: make the audio carry language, not recording conditions.

Found in v1 data: ~75% of FLEURS US-English clips sit ~30 dB below every other language, and
the student learned "quiet => English" (Hindi turned down 30 dB was called English 82% of the time).
Switch clips were hard 4 s cuts between mismatched levels. This stage, for every selected clip:
  1. Silero VAD -> keep speech only (0.1 s padding), drop clips with < 1 s of speech
  2. normalise speech RMS to -23 dBFS (offline data cleaning, not a model input step)
and rebuilds switch/mix clips like a conversation: whole sentences, 0.25-0.6 s pauses
(30% of training switches have no pause), 10 ms fades, equal levels.
Old manifests are kept in data/manifests_v1/.
"""
import json
import random
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from slid.audio import SR, load_wav
from slid.config import LANGS

ROOT = Path(__file__).resolve().parents[1]
MAN, OLD, CLEAN = ROOT / "data/manifests", ROOT / "data/manifests_v1", ROOT / "data/clean"
TARGET_DBFS = -23.0
VAD = load_silero_vad()


def speech_spans(x: np.ndarray) -> list[tuple[int, int]]:
    probe = x / (np.abs(x).max() + 1e-9) * 0.5                  # VAD on a level-normalised copy
    ts = get_speech_timestamps(torch.from_numpy(probe), VAD, sampling_rate=SR)
    return [(t["start"], t["end"]) for t in ts]


def clean(x: np.ndarray) -> np.ndarray | None:
    spans = speech_spans(x)
    if not spans:
        return None
    pad = int(0.1 * SR)
    y = x[max(0, spans[0][0] - pad): spans[-1][1] + pad]
    speech = np.concatenate([x[a:b] for a, b in spans])
    if len(speech) < SR:
        return None
    gain = 10 ** ((TARGET_DBFS - 20 * np.log10(np.sqrt(np.mean(speech ** 2)) + 1e-9)) / 20)
    y = y * gain
    return (y / max(1.0, np.abs(y).max() / 0.99)).astype(np.float32)


def fade(x: np.ndarray, ms: int = 10) -> np.ndarray:
    n = int(ms * SR / 1000)
    w = np.linspace(0, 1, n, dtype=np.float32)
    x = x.copy()
    x[:n] *= w
    x[-n:] *= w[::-1]
    return x


def pause(seconds: float, rng) -> np.ndarray:
    return (rng.standard_normal(int(seconds * SR)) * 10 ** (-60 / 20)).astype(np.float32)


def join(segs: list[tuple[str, np.ndarray]], pauses: list[float], rng):
    parts, rows, t = [], [], 0.0
    for i, (lang, x) in enumerate(segs):
        if i:
            p = pause(pauses[i - 1], rng)
            parts.append(p)
            t += len(p) / SR
        x = fade(x)
        parts.append(x)
        rows.append([lang, t, t + len(x) / SR])
        t += len(x) / SR
    return np.concatenate(parts), rows


def main() -> None:
    rng, nrng = random.Random(0), np.random.default_rng(0)
    if not OLD.exists():
        shutil.copytree(MAN, OLD)
    pools = {}
    for split in ("train", "eval"):
        out = []
        for it in (json.loads(l) for l in open(OLD / f"{split}.jsonl", encoding="utf-8")):
            y = clean(load_wav(it["path"]))
            if y is None:
                continue
            g = "en-in" if it.get("source") == "svarah" else it["lang"]
            path = CLEAN / split / g / Path(it["path"]).name
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, y, SR)
            out.append({**it, "path": str(path), "dur": len(y) / SR, "raw_path": it["path"]})
        with open(MAN / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
        pools[split] = out
        print(f"{split}: kept {len(out)} clips")

    def pool(split, spec, lo, hi):
        items = [i for i in pools[split] if lo <= i["dur"] <= hi]
        if spec == "en-in":
            return [i for i in items if i.get("source") == "svarah"]
        return [i for i in items if i["lang"] == spec and i.get("source", "fleurs") == "fleurs"]

    # eval switches: sentence A, pause, sentence B
    pairs = [("hi", "en"), ("en", "hi"), ("hi", "en-in"), ("en-in", "hi")] + \
            [(l, "en") for l in LANGS if l not in ("hi", "en")]
    sw_dir = CLEAN / "switch"
    sw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for a, b in pairs:
        for k in range(10):
            ia, ib = rng.choice(pool("eval", a, 2.5, 7)), rng.choice(pool("eval", b, 2.5, 7))
            la, lb = ("en" if s == "en-in" else s for s in (a, b))
            x, segs = join([(la, load_wav(ia["path"])), (lb, load_wav(ib["path"]))], [rng.uniform(0.25, 0.6)], nrng)
            path = sw_dir / f"{a}2{b}_{k:02d}.wav"
            sf.write(path, x, SR)
            rows.append({"path": str(path), "pair": f"{a}->{b}", "segments": segs, "switch_s": segs[1][1],
                         "sources": [ia["path"], ib["path"]]})
    with open(MAN / "switch.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"switch: {len(rows)} clips")

    # training mixes: 2-3 whole sentences, half hi<->en; 30% of joins have no pause
    mix_dir = CLEAN / "train_mix"
    mix_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(600):
        langs = [rng.choice(["hi", "en"])]
        for _ in range(rng.choice([1, 1, 2])):
            if rng.random() < 0.5 and langs[-1] in ("hi", "en"):
                langs.append("en" if langs[-1] == "hi" else "hi")
            else:
                langs.append(rng.choice([l for l in LANGS if l != langs[-1]]))
        segs = []
        for l in langs:
            cands = [i for i in pools["train"] if i["lang"] == l and 1.5 <= i["dur"] <= 6]
            segs.append((l, load_wav(rng.choice(cands)["path"])))
        pauses = [0.0 if rng.random() < 0.3 else rng.uniform(0.2, 0.6) for _ in segs[1:]]
        x, seg_rows = join(segs, pauses, nrng)
        path = mix_dir / f"mix_{k:04d}.wav"
        sf.write(path, x, SR)
        rows.append({"path": str(path), "segments": seg_rows, "dur": len(x) / SR})
    with open(MAN / "train_mix.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"train_mix: {len(rows)} clips, {sum(r['dur'] for r in rows) / 3600:.2f} h")


if __name__ == "__main__":
    main()
