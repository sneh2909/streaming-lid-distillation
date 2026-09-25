"""Download a small FLEURS subset and build train/eval manifests plus code-switch eval clips.

train pool  <- FLEURS `dev` split   (sentence-disjoint from eval)
eval pool   <- FLEURS `test` split
switch eval <- concatenations of two eval clips in different languages, switch time known exactly
"""
import argparse
import csv
import json
import random
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

from slid.audio import SR, load_wav, trim_silence
from slid.config import FLEURS_CODES, LANGS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fleurs"
MANIFESTS = ROOT / "data" / "manifests"
SWITCH_DIR = ROOT / "data" / "generated" / "switch"
FLEURS_SPLIT = {"train": "dev", "eval": "test"}


def download_split(lang: str, fleurs_split: str) -> Path:
    code = FLEURS_CODES[lang]
    out = RAW / lang / fleurs_split
    if out.exists() and any(out.rglob("*.wav")):
        return out
    out.mkdir(parents=True, exist_ok=True)
    tsv = hf_hub_download("google/fleurs", f"data/{code}/{fleurs_split}.tsv", repo_type="dataset")
    tgz = hf_hub_download("google/fleurs", f"data/{code}/audio/{fleurs_split}.tar.gz", repo_type="dataset")
    with tarfile.open(tgz) as tar:
        tar.extractall(out, filter="data")
    (out / "meta.tsv").write_bytes(Path(tsv).read_bytes())
    return out


def read_meta(split_dir: Path) -> list[dict]:
    wavs = {p.name: p for p in split_dir.rglob("*.wav")}
    rows = []
    with open(split_dir / "meta.tsv", encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r[1] in wavs:
                rows.append({"sentence_id": r[0], "path": str(wavs[r[1]]), "gender": r[-1],
                             "n_samples": int(r[-2])})
    return rows


def build_manifests(n_per_lang: dict[str, int], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    out = {}
    for split, fleurs_split in FLEURS_SPLIT.items():
        items = []
        for lang in LANGS:
            rows = read_meta(download_split(lang, fleurs_split))
            rows = [r for r in rows if 2 * SR <= r["n_samples"] <= 20 * SR]
            rng.shuffle(rows)
            for r in rows[: n_per_lang[split]]:
                items.append({**r, "lang": lang, "source": "fleurs", "dur": r["n_samples"] / SR})
        out[split] = items
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        with open(MANIFESTS / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"{split}: {len(items)} clips, {sum(i['dur'] for i in items) / 3600:.2f} h")
    return out


def add_svarah(items: dict[str, list[dict]], n_per_split: dict[str, int], seed: int) -> None:
    """Indian-accented English (AI4Bharat Svarah, gated test set) -> lang "en", source "svarah".
    Svarah has no speaker id, so the speaker key is the demographic tuple; keys are split
    70/30 between train and eval so no key appears on both sides."""
    import io
    import pyarrow.parquet as pq

    out_root = ROOT / "data" / "raw" / "svarah"
    rows = []
    for i in range(3):
        p = hf_hub_download("ai4bharat/Svarah", f"data/test-0000{i}-of-00003.parquet", repo_type="dataset")
        rows += pq.read_table(p).to_pylist()
    rows = [r for r in rows if 2.0 <= r["duration"] <= 20.0]
    key = lambda r: (r["primary_language"], r["native_place_state"], r["native_place_district"],
                     r["gender"], r["age-group"])
    keys = sorted({key(r) for r in rows})
    rng = random.Random(seed + 2)
    rng.shuffle(keys)
    train_keys = set(keys[: int(0.7 * len(keys))])
    for split in ("train", "eval"):
        pool = [r for r in rows if (key(r) in train_keys) == (split == "train")]
        rng.shuffle(pool)
        (out_root / split).mkdir(parents=True, exist_ok=True)
        for j, r in enumerate(pool[: n_per_split[split]]):
            x, sr = sf.read(io.BytesIO(r["audio_filepath"]["bytes"]), dtype="float32", always_2d=True)
            path = out_root / split / f"svarah_{j:04d}.wav"
            x = x.mean(axis=1)
            if sr != SR:
                import torch, torchaudio.functional as AF
                x = AF.resample(torch.from_numpy(x), sr, SR).numpy()
            sf.write(path, x, SR)
            items[split].append({"path": str(path), "lang": "en", "source": "svarah", "dur": len(x) / SR,
                                 "l1": r["primary_language"], "speaker_key": "|".join(key(r))})
        with open(MANIFESTS / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for it in items[split]:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"{split}: +{min(len(pool), n_per_split[split])} Svarah clips "
              f"({len({i['speaker_key'] for i in items[split] if i.get('source') == 'svarah'})} speaker keys)")


def pool_of(spec: str, items: list[dict]) -> list[dict]:
    """'en-in' = Svarah Indian English, 'en' = FLEURS (US) English, else the language code."""
    if spec == "en-in":
        return [i for i in items if i.get("source") == "svarah"]
    return [i for i in items if i["lang"] == spec and i.get("source", "fleurs") == "fleurs"]


def build_switch_clips(eval_items: list[dict], pairs: list[tuple[str, str]], n_per_pair: int,
                       seg_s: float, seed: int) -> None:
    """Concatenate ~seg_s of language A then ~seg_s of language B (silence-trimmed, no gap)."""
    rng = random.Random(seed)
    long_enough = [it for it in eval_items if it["dur"] >= seg_s + 0.5]
    by_lang = {spec: pool_of(spec, long_enough) for pair in pairs for spec in pair}
    SWITCH_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for a, b in pairs:
        for k in range(n_per_pair):
            ia, ib = rng.choice(by_lang[a]), rng.choice(by_lang[b])
            xa = trim_silence(load_wav(ia["path"]))[: int(seg_s * SR)]
            xb = trim_silence(load_wav(ib["path"]))[: int(seg_s * SR)]
            path = SWITCH_DIR / f"{a}2{b}_{k:02d}.wav"
            sf.write(path, np.concatenate([xa, xb]), SR)
            la, lb = ("en" if s == "en-in" else s for s in (a, b))
            rows.append({"path": str(path), "pair": f"{a}->{b}",
                         "segments": [[la, 0.0, len(xa) / SR], [lb, len(xa) / SR, (len(xa) + len(xb)) / SR]],
                         "switch_s": len(xa) / SR, "sources": [ia["path"], ib["path"]]})
    with open(MANIFESTS / "switch.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"switch: {len(rows)} clips ({', '.join(f'{a}->{b}' for a, b in pairs)})")


def build_train_mix(train_items: list[dict], n: int, seed: int) -> None:
    """Multi-segment training clips (2-3 segments of 2.5-5 s) so the student sees switches.
    Half the switches are hi<->en; the rest are random pairs of LANGS."""
    rng = random.Random(seed + 1)
    by_lang: dict[str, list[dict]] = {}
    for it in train_items:
        by_lang.setdefault(it["lang"], []).append(it)
    out_dir = ROOT / "data" / "generated" / "train_mix"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(n):
        n_seg = rng.choice([2, 2, 3])
        langs = [rng.choice(["hi", "en"])]
        for _ in range(n_seg - 1):
            if rng.random() < 0.5 and langs[-1] in ("hi", "en"):
                langs.append("en" if langs[-1] == "hi" else "hi")
            else:
                langs.append(rng.choice([l for l in LANGS if l != langs[-1]]))
        parts, segs, t = [], [], 0.0
        for lang in langs:
            x = trim_silence(load_wav(rng.choice(by_lang[lang])["path"]))[: int(rng.uniform(2.5, 5.0) * SR)]
            parts.append(x)
            segs.append([lang, t, t + len(x) / SR])
            t += len(x) / SR
        path = out_dir / f"mix_{k:04d}.wav"
        sf.write(path, np.concatenate(parts), SR)
        rows.append({"path": str(path), "segments": segs, "dur": t})
    with open(MANIFESTS / "train_mix.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"train_mix: {len(rows)} clips, {sum(r['dur'] for r in rows) / 3600:.2f} h")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=200, help="clips per language from FLEURS dev")
    ap.add_argument("--n-eval", type=int, default=60, help="clips per language from FLEURS test")
    ap.add_argument("--n-switch", type=int, default=10, help="switch clips per language pair")
    ap.add_argument("--n-mix", type=int, default=600, help="multi-language training clips")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    items = build_manifests({"train": args.n_train, "eval": args.n_eval}, args.seed)
    add_svarah(items, {"train": args.n_train, "eval": args.n_eval}, args.seed)
    pairs = [("hi", "en"), ("en", "hi"), ("hi", "en-in"), ("en-in", "hi")] + \
            [(lang, "en") for lang in LANGS if lang not in ("hi", "en")]
    build_switch_clips(items["eval"], pairs, args.n_switch, seg_s=4.0, seed=args.seed)
    build_train_mix(items["train"], args.n_mix, args.seed)


if __name__ == "__main__":
    main()
