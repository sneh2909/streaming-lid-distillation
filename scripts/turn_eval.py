"""Turn-level evaluation of the VAD-gated pipeline (DESIGN section 3).

For each switch clip: Silero VAD (>= 300 ms silence ends a turn) splits it into turns. Every turn
gets a fresh student stream. Per turn we record
  * early route: first commit (theta) and its time from turn start - the ASR we start streaming to
  * final LID: the student's posterior at the turn's end (the VAD endpoint) - what the bot acts on
  * re-decode: early route != final LID -> the buffered turn is re-decoded with the right ASR
Turn labels come from the known segment boundaries (the segment with most overlap).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from slid.audio import SR, load_wav
from slid.commit import commit_stream
from slid.config import LANGS
from slid.metrics import frame_time
from slid.student import load_student

ROOT = Path(__file__).resolve().parents[1]


def turns(x, vad, min_silence_ms=300):
    ts = get_speech_timestamps(torch.from_numpy(x / (np.abs(x).max() + 1e-9) * 0.5), vad, sampling_rate=SR,
                               min_silence_duration_ms=min_silence_ms)
    return [(t["start"], t["end"]) for t in ts]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/final/student.pt")
    ap.add_argument("--theta", type=float, default=0.8)
    ap.add_argument("--out", default="results/final/turn_eval.json")
    args = ap.parse_args()
    m, vad = load_student(ROOT / args.ckpt), load_silero_vad()
    rows = []
    for line in open(ROOT / "data/manifests/switch.jsonl"):
        sw = json.loads(line)
        x = load_wav(sw["path"])
        for a, b in turns(x, vad):
            ov = [max(0.0, min(b / SR, e) - max(a / SR, s)) for _, s, e in sw["segments"]]
            y = LANGS.index(sw["segments"][int(np.argmax(ov))][0])
            p = m(torch.from_numpy(x[a:b])[None], 4)[0].softmax(-1).numpy()
            c = commit_stream(p, theta_commit=args.theta, theta_switch=min(args.theta + 0.1, 0.95))
            idx = np.flatnonzero(c >= 0)
            early = int(c[idx[0]]) if len(idx) else -1
            rows.append({"pair": sw["pair"], "dur": (b - a) / SR, "y": y, "final": int(p[-1].argmax()),
                         "early": early, "early_t": float(frame_time([idx[0]])[0]) if len(idx) else np.nan})
    fin = np.array([r["final"] == r["y"] for r in rows])
    has_early = np.array([r["early"] >= 0 for r in rows])
    early_ok = np.array([r["early"] == r["y"] for r in rows])
    redecode = np.array([r["early"] >= 0 and r["early"] != r["final"] for r in rows])
    short = np.array([r["dur"] < 1.5 for r in rows])
    res = {"ckpt": args.ckpt, "theta": args.theta, "n_turns": len(rows),
           "final_lid_acc": float(fin.mean()), "final_lid_acc_turns_ge_1.5s": float(fin[~short].mean()),
           "early_route_rate": float(has_early.mean()), "early_route_acc": float(early_ok[has_early].mean()),
           "early_route_time_median_s": float(np.nanmedian([r["early_t"] for r in rows])),
           "redecode_rate": float(redecode.mean()), "turns": rows}
    print(f"{len(rows)} turns | final LID at endpoint acc {res['final_lid_acc']:.3f} "
          f"(turns >= 1.5 s: {res['final_lid_acc_turns_ge_1.5s']:.3f}) | early route in "
          f"{res['early_route_time_median_s']:.2f} s (median) on {res['early_route_rate']:.2f} of turns, "
          f"acc {res['early_route_acc']:.3f} | re-decode needed {res['redecode_rate']:.3f}")
    (ROOT / args.out).write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
