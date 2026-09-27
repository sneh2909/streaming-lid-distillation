"""Real-world check: speech that is NOT FLEURS/Svarah.

  eval_cv     Common Voice hi (validation speakers) + en with an Indian accent (held-out speakers)
  recordings  the author's own same-speaker hi/en/Hinglish recordings, per VAD turn, labelled by
              content (data/user_recordings/turn_labels.json); never used for training

Reports student accuracy (fresh stream per clip/turn) and, with --teachers, the teachers on the
same audio, so a gap between teacher and student is visible.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from slid.audio import SR, load_wav
from slid.config import LANGS
from slid.metrics import frame_time
from slid.student import load_student

ROOT = Path(__file__).resolve().parents[1]


def recording_turns():
    lab_file = ROOT / "data/user_recordings/turn_labels.json"
    if not lab_file.exists():
        return []
    labels, vad, out = json.loads(lab_file.read_text()), load_silero_vad(), []
    for f in sorted((ROOT / "data/user_recordings").glob("*.wav")):
        x = load_wav(str(f))
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 10 ** (-23 / 20)
        ts = get_speech_timestamps(torch.from_numpy(x / (np.abs(x).max() + 1e-9) * 0.5), vad, sampling_rate=SR,
                                   min_silence_duration_ms=300)
        lab = labels.get(f.stem)
        if lab and len(lab) == len(ts):
            out += [(x[t["start"]:t["end"]], "hi" if l == "mix" else l, l) for t, l in zip(ts, lab)]
    return out


def summarise(pred_end, pred_at, ys, kinds):
    ys, kinds, pred_end = np.array(ys), np.array(kinds), np.array(pred_end)
    r = {"all": float((pred_end == ys).mean()), "n": int(len(ys))}
    for k in sorted(set(kinds)):
        r[k] = float((pred_end[kinds == k] == ys[kinds == k]).mean())
    for a, v in pred_at.items():
        v = np.array(v, dtype=float)
        r[f"acc@{a}s"] = float(np.nanmean(v))
    return r


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["final"])
    ap.add_argument("--teachers", action="store_true")
    ap.add_argument("--out", default="results/final/realworld.json")
    args = ap.parse_args()
    cv = [json.loads(l) for l in open(ROOT / "data/manifests/eval_cv.jsonl", encoding="utf-8")]
    cv_wav = [(load_wav(it["path"]), it["lang"], "en-in" if it["lang"] == "en" else "hi") for it in cv]
    rec = recording_turns()
    sets = {"eval_cv": cv_wav, "recordings": rec}
    out = json.loads((ROOT / args.out).read_text()) if (ROOT / args.out).exists() else {}
    for name in args.ckpts:
        ckpt = ROOT / f"checkpoints/{name}/student.pt"
        if not ckpt.exists():
            print(f"{name}: checkpoint not present, skipped")
            continue
        m = load_student(ckpt)
        out[name] = {}
        for sname, items in sets.items():
            pe, pa, ys, ks = [], {1.0: [], 2.0: []}, [], []
            for x, lang, kind in items:
                p = m(torch.from_numpy(x)[None], 4)[0].softmax(-1).numpy()
                y = LANGS.index(lang)
                pe.append(int(p[-1].argmax())); ys.append(y); ks.append(kind)
                t = frame_time(np.arange(len(p)))
                for a in pa:
                    idx = np.flatnonzero(t <= a)
                    pa[a].append(float(p[idx[-1]].argmax() == y) if len(idx) and a <= len(x) / SR else np.nan)
            out[name][sname] = summarise(pe, pa, ys, ks)
            print(name, sname, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[name][sname].items()})
    if args.teachers:
        from slid.teachers import load_teacher, restrict
        w = json.loads((ROOT / "results/teachers/ensemble_tuning.json").read_text())["best"]["w"]
        probs = {}
        for tname in ("indic-transcribe", "whisper-turbo"):
            t = load_teacher(tname)
            probs[tname] = {s: restrict(t.probs([x for x, _, _ in items])) for s, items in sets.items()}
            del t
            torch.cuda.empty_cache()
        lq = {s: w * np.log(np.clip(probs["indic-transcribe"][s], 1e-6, 1)) +
                 (1 - w) * np.log(np.clip(probs["whisper-turbo"][s], 1e-6, 1)) for s in sets}
        probs["ensemble"] = {s: np.exp(v - v.max(1, keepdims=True)) for s, v in lq.items()}
        for tname, ps in probs.items():
            out[f"teacher:{tname}"] = {}
            for s, items in sets.items():
                ys = [LANGS.index(l) for _, l, _ in items]
                out[f"teacher:{tname}"][s] = summarise(list(ps[s].argmax(1)), {}, ys, [k for _, _, k in items])
                print("teacher", tname, s, {k: round(v, 3) if isinstance(v, float) else v
                                            for k, v in out[f"teacher:{tname}"][s].items()})
    (ROOT / args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
