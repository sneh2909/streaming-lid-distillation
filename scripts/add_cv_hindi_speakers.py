"""Replace Common Voice Hindi with a speaker-diverse set.

CV 17's Hindi `train` split has only 3 speakers (4,689 clips), so the student memorised them
(59/60 on their clips, 17/60 on new speakers). The `other`, `invalidated` and `validation` splits
have ~390 speakers; their transcripts may be unvalidated, but we never use transcripts: the
teacher labels the audio. The held-out set moves to the `test` split (294 speakers), and every
test speaker is excluded from training.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_commonvoice import REPO, decode, pick                        # noqa: E402
from clean_data import clean                                          # noqa: E402
from slid.audio import SR                                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAN, OUT = ROOT / "data/manifests", ROOT / "data/clean"


def split_rows(split):
    for p in sorted(s.rfilename for s in HfApi().dataset_info(REPO).siblings if s.rfilename.startswith(f"hi/{split}/")):
        for r in pq.read_table(hf_hub_download(REPO, p, repo_type="dataset"),
                               columns=["client_id", "audio", "down_votes"]).to_pylist():
            yield r


def write(items, split, tag):
    d = OUT / split / f"cv_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i, r in enumerate(items):
        y = clean(decode(r))
        if y is None or len(y) < 1.5 * SR:
            continue
        path = d / f"{tag}_{i:04d}.wav"
        sf.write(path, y, SR)
        out.append({"path": str(path), "lang": "hi", "source": "cv", "dur": len(y) / SR, "speaker_key": r["client_id"][:16]})
    return out


def main() -> None:
    rng = random.Random(7)
    test = list(split_rows("test"))
    test_spk = {r["client_id"] for r in test}
    ev = write(pick(test, 80, rng), "eval", "hi_test")                   # round-robin -> 80 distinct speakers
    pool = [r for sp in ("other", "invalidated", "validation") for r in split_rows(sp)
            if r["client_id"] not in test_spk]
    by = {}
    for r in pool:
        by.setdefault(r["client_id"], []).append(r)
    capped = [r for rs in by.values() for r in rng.sample(rs, min(3, len(rs)))]   # <= 3 clips per speaker
    tr = write(pick(capped, 800, rng), "train", "hi_multi")
    print(f"hi train pool speakers {len(by)} -> {len(tr)} clips from {len({r['speaker_key'] for r in tr})} speakers; "
          f"eval {len(ev)} clips from {len({r['speaker_key'] for r in ev})} test speakers")

    train = [json.loads(l) for l in open(MAN / "train.jsonl", encoding="utf-8")]
    train = [r for r in train if not (r.get("source") == "cv" and r["lang"] == "hi")] + tr
    with open(MAN / "train.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in train)
    evals = [json.loads(l) for l in open(MAN / "eval_cv.jsonl", encoding="utf-8")]
    evals = [r for r in evals if r["lang"] != "hi"] + ev
    with open(MAN / "eval_cv.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in evals)
    print("train", len(train), "eval_cv", len(evals))


if __name__ == "__main__":
    main()
