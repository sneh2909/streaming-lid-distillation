"""Does resetting at turn boundaries cut switch lag? (Part 2, code-switch section)

Uses the cleaned switch clips (two whole sentences with a natural 0.25-0.6 s pause, like a
sentence/turn boundary), detects the pause with a simple energy VAD, and compares:
  none    commit policy as is
  policy  at the end of a pause, reset the EMA and let the new turn commit at theta_commit
  state   additionally start a fresh student stream for the new turn (the previous language
          stays routed until the new turn commits)
"""
import json
from pathlib import Path

import numpy as np
import torch

from slid.audio import SR, load_wav, trim_silence
from slid.commit import commit_stream
from slid.config import LANGS
from slid.metrics import flips_per_min, frame_time, switch_lag
from slid.student import HOP, SUBSAMPLE, load_student, n_frames

ROOT = Path(__file__).resolve().parents[1]
GAP_S, SEG_S = 0.5, 4.0
FRAME = HOP * SUBSAMPLE


def paused_clip(sw, rng):
    xa = trim_silence(load_wav(sw["sources"][0]))[: int(SEG_S * SR)]
    xb = trim_silence(load_wav(sw["sources"][1]))[: int(SEG_S * SR)]
    gap = (rng.standard_normal(int(GAP_S * SR)) * 1e-4).astype(np.float32)
    return np.concatenate([xa, gap, xb]), (len(xa) + len(gap)) / SR


def vad_turn_starts(x, T, min_pause_frames=4, rel_db=-35.0):
    """Frames where speech resumes after >= min_pause_frames (320 ms) of low energy."""
    e = np.array([np.sqrt(np.mean(x[i * FRAME:(i + 1) * FRAME] ** 2) + 1e-12) for i in range(T)])
    db = 20 * np.log10(e / e.max())
    quiet = db < rel_db
    starts, run = [], 0
    for t in range(T):
        if quiet[t]:
            run += 1
        else:
            if run >= min_pause_frames:
                starts.append(t)
            run = 0
    return starts


def commit_with_reset(p, starts, **kw):
    out = commit_stream(p, **kw)
    for s in starts:
        seg = commit_stream(p[s:], **kw)                 # new turn: fresh smoothing, first-commit rule
        prev = out[s - 1] if s > 0 else -1
        out[s:] = np.where(seg >= 0, seg, prev)          # keep previous language until the turn commits
    return out


@torch.no_grad()
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/final/student.pt")
    ap.add_argument("--out", default="results/final/turn_reset_causal.json")
    args = ap.parse_args()
    model = load_student(ROOT / args.ckpt)
    chunk = 4
    post = lambda x: model(torch.from_numpy(x)[None], chunk)[0].softmax(-1).numpy()
    rng = np.random.default_rng(0)
    kw = dict(theta_commit=0.8, theta_switch=0.9, dwell=3)
    res = {m: {"lag": [], "flips": [], "premature": []} for m in ("none", "policy", "state")}
    n_detected = 0
    for line in open(ROOT / "data/manifests/switch.jsonl"):
        sw = json.loads(line)
        x, switch_s = load_wav(sw["path"]), sw["switch_s"]          # clean switch clips already have a pause
        new, old = LANGS.index(sw["segments"][1][0]), LANGS.index(sw["segments"][0][0])
        p = post(x)
        T = len(p)
        starts = [s for s in vad_turn_starts(x, T) if s > 5]
        n_detected += bool(starts)
        tt = frame_time(np.arange(T))
        p_state = p.copy()
        for s in starts:                                   # fresh student stream from the turn start
            q = post(x[s * FRAME:])
            p_state[s: s + len(q)] = q[: T - s]
        labels = {"none": commit_stream(p, **kw), "policy": commit_with_reset(p, starts, **kw),
                  "state": commit_with_reset(p_state, starts, **kw)}
        for m, c in labels.items():
            res[m]["lag"].append(switch_lag(np.arange(T), c, new, switch_s))
            res[m]["flips"].append(flips_per_min(c[c >= 0], 1))
            res[m]["premature"].append(float((c[tt < switch_s] == new).any()))
    out = {"gap_s": GAP_S, "policy": kw, "chunk": chunk, "pause_detected": n_detected, "n": 90}
    for m, d in res.items():
        lag = np.array(d["lag"])
        out[m] = {"switch_lag_median_s": float(np.nanmedian(lag)), "missed": int(np.isnan(lag).sum()),
                  "flips_per_min": float(np.nanmean(d["flips"])), "premature_rate": float(np.mean(d["premature"]))}
        print(f"{m:7s} lag {out[m]['switch_lag_median_s']:.2f}s  missed {out[m]['missed']}/90  "
              f"flips/min {out[m]['flips_per_min']:.1f}  premature {out[m]['premature_rate']:.2f}")
    print("pause detected in", n_detected, "/ 90 clips")
    (ROOT / "results/final").mkdir(parents=True, exist_ok=True)
    out["ckpt"] = args.ckpt
    (ROOT / args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
