"""Evidence for the speaker-memorisation finding (README §1).

Records (1) how many speakers each Common Voice 17 Hindi split has, and (2) end-of-clip Hindi
accuracy of the model trained with the 3-speaker CV `train` split vs the shipped model, on
  seen   the 3-speaker CV Hindi clips the first model trained on
  new    CV Hindi from speakers neither model trained on (CV `validation` split)
-> results/final/speaker_memorisation.json
Needs the local `final_cv3spk` checkpoint and data/manifests/*.cv3spk.jsonl (not shipped).
"""
import collections
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download

from slid.audio import load_wav
from slid.config import LANGS
from slid.student import load_student

ROOT = Path(__file__).resolve().parents[1]
REPO = "fixie-ai/common_voice_17_0"


def speakers_per_split() -> dict:
    files = [s.rfilename for s in HfApi().dataset_info(REPO).siblings if s.rfilename.startswith("hi/")]
    out = {}
    for split in ("train", "validation", "test", "other", "invalidated"):
        spk = collections.Counter()
        for f in (f for f in files if f.split("/")[1] == split):
            spk.update(pq.read_table(hf_hub_download(REPO, f, repo_type="dataset"), columns=["client_id"])
                       .column("client_id").to_pylist())
        out[split] = {"clips": sum(spk.values()), "speakers": len(spk)}
    return out


@torch.no_grad()
def hindi_acc(model, rows) -> dict:
    hi = LANGS.index("hi")
    preds = [int(model(torch.from_numpy(load_wav(r["path"]))[None], 4)[0, -1].argmax()) for r in rows]
    return {"n": len(rows), "acc": sum(p == hi for p in preds) / len(rows),
            "called": dict(collections.Counter(LANGS[p] for p in preds).most_common())}


def main() -> None:
    read = lambda f: [json.loads(l) for l in open(ROOT / "data/manifests" / f, encoding="utf-8")]
    seen = [r for r in read("train.cv3spk.jsonl") if r.get("source") == "cv" and r["lang"] == "hi"][:60]
    new = [r for r in read("eval_cv.cv3spk.jsonl") if r["lang"] == "hi"][:60]
    res = {"cv_hindi_speakers_per_split": speakers_per_split(),
           "seen_speakers_in_first_cv_run": len({r["speaker_key"] for r in seen})}
    for name in ("final_cv3spk", "final"):
        m = load_student(ROOT / f"checkpoints/{name}/student.pt")
        res[name] = {"seen_speakers": hindi_acc(m, seen), "new_speakers": hindi_acc(m, new)}
        print(name, {k: (round(v["acc"], 2), v["called"]) for k, v in res[name].items()})
    print(res["cv_hindi_speakers_per_split"])
    (ROOT / "results/final/speaker_memorisation.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
