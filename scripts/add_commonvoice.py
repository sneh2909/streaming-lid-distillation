"""Stage 3 of data prep: add diverse, crowd-recorded speech from Common Voice 17 (CC0).

Why: on real recordings the ensemble teacher was 30/30 but the student 14/30, calling Hindi
"English". In our training data every Indian-accented voice was Svarah English and every Hindi
clip was FLEURS, so "Indian voice on an ordinary mic" predicted English. Common Voice has
thousands of volunteers on their own mics (and English with an accent field), which breaks
that correlation from both sides.

Adds (speaker-diverse: one clip per client_id first):
  train  hi 500, mr/bn/ta 150 each, en with an Indian accent 300, + 200 hi<->en(IN) switch mixes
  eval   data/manifests/eval_cv.jsonl: hi from CV's validation split (disjoint speakers) and
         en(IN) from held-out speakers, 80 each
Same cleaning as clean_data.py (Silero VAD + -23 dBFS). Labels still come from the teacher.
"""
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio.functional as AF
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_data import clean, join                                   # noqa: E402
from slid.audio import SR                                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAN, OUT = ROOT / "data/manifests", ROOT / "data/clean"
REPO = "fixie-ai/common_voice_17_0"
QUOTA = {"hi": 500, "mr": 150, "bn": 150, "ta": 150, "en": 300}


def files(prefix: str, n: int | None = None) -> list[str]:
    fs = sorted(s.rfilename for s in HfApi().dataset_info(REPO).siblings if s.rfilename.startswith(prefix))
    return fs[:n] if n else fs


def rows(paths: list[str], accent_india: bool = False):
    for p in paths:
        t = pq.read_table(hf_hub_download(REPO, p, repo_type="dataset"),
                          columns=["client_id", "audio", "accent", "down_votes"])
        for r in t.to_pylist():
            if r["down_votes"] and r["down_votes"] > 0:
                continue
            if accent_india and "india" not in (r["accent"] or "").lower():
                continue
            yield r


def decode(r) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return AF.resample(torch.from_numpy(x), sr, SR).numpy() if sr != SR else x


def pick(rs, n: int, rng, exclude_speakers=frozenset()):
    """Round-robin over speakers so the first n clips cover as many voices as possible."""
    by = {}
    for r in rs:
        if r["client_id"] not in exclude_speakers:
            by.setdefault(r["client_id"], []).append(r)
    spk = list(by)
    rng.shuffle(spk)
    out, k = [], 0
    while len(out) < n and any(by.values()):
        s = spk[k % len(spk)]
        if by[s]:
            out.append(by[s].pop(rng.randrange(len(by[s]))))
        k += 1
    return out


def write(items, lang, split, tag, rng):
    rows_out = []
    d = OUT / split / f"cv_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(items):
        y = clean(decode(r))
        if y is None or len(y) < 1.5 * SR:
            continue
        path = d / f"{tag}_{i:04d}.wav"
        sf.write(path, y, SR)
        rows_out.append({"path": str(path), "lang": lang, "source": "cv", "dur": len(y) / SR,
                         "speaker_key": r["client_id"][:16]})
    return rows_out


def main() -> None:
    rng = random.Random(3)
    train, evals = [], []
    for lang in ("hi", "mr", "bn", "ta"):
        paths = files(f"{lang}/train", None if lang in ("hi", "mr") else 1)
        train += write(pick(rows(paths), QUOTA[lang], rng), lang, "train", lang, rng)
        print(lang, "train", sum(r["lang"] == lang for r in train))
    evals += write(pick(rows(files("hi/validation")), 80, rng), "hi", "eval", "hi", rng)
    en_rows = list(rows(files("en/train", 2), accent_india=True))
    speakers = sorted({r["client_id"] for r in en_rows})
    rng.shuffle(speakers)
    held = set(speakers[: max(1, len(speakers) // 5)])
    train += write(pick(en_rows, QUOTA["en"], rng, exclude_speakers=held), "en", "train", "en_in", rng)
    evals += write(pick([r for r in en_rows if r["client_id"] in held], 80, rng), "en", "eval", "en_in", rng)
    print("en(IN) speakers:", len(speakers), "held out:", len(held))

    with open(MAN / "train.jsonl", "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in train)
    with open(MAN / "eval_cv.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in evals)
    print(f"train +{len(train)} clips, eval_cv {len(evals)} clips")

    # switch mixes between CV Hindi and CV Indian-English voices
    hi = [r for r in train if r["lang"] == "hi" and r["dur"] <= 7]
    en = [r for r in train if r["lang"] == "en" and r["dur"] <= 7]
    nrng = np.random.default_rng(3)
    d = OUT / "train_mix_cv"
    d.mkdir(parents=True, exist_ok=True)
    mixes = []
    from slid.audio import load_wav
    for k in range(200):
        order = ["hi", "en"] if rng.random() < 0.5 else ["en", "hi"]
        if rng.random() < 0.4:
            order.append(order[0])
        segs = [(l, load_wav(rng.choice(hi if l == "hi" else en)["path"])) for l in order]
        pauses = [0.0 if rng.random() < 0.3 else rng.uniform(0.2, 0.6) for _ in segs[1:]]
        x, seg_rows = join(segs, pauses, nrng)
        path = d / f"cvmix_{k:04d}.wav"
        sf.write(path, x, SR)
        mixes.append({"path": str(path), "segments": seg_rows, "dur": len(x) / SR, "source": "cv"})
    with open(MAN / "train_mix.jsonl", "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in mixes)
    print(f"train_mix +{len(mixes)} clips")


if __name__ == "__main__":
    main()
