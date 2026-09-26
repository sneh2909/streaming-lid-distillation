"""Latency/accuracy curve of the commit policy (Part 2), for one trained student.

Posteriors are computed once per clip; the policy (slid/commit.py) is swept over
theta_commit x dwell, with theta_switch = min(theta_commit + 0.1, 0.95). This is a
descriptive curve on held-out data: in production the operating point is picked on a
labelled dev set of real calls, then frozen.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from slid.audio import load_wav
from slid.commit import commit_stream
from slid.config import LANGS
from slid.metrics import flips_per_min, frame_time, switch_lag
from slid.student import StreamingLID

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(name):
    return [json.loads(l) for l in open(ROOT / f"data/manifests/{name}.jsonl", encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ensemble_causal/student.pt")
    ap.add_argument("--chunk", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ROOT / args.ckpt, map_location="cpu")
    model = StreamingLID(ck["n_langs"]).to(dev).eval()
    model.load_state_dict(ck["state"])

    @torch.no_grad()
    def post(path):
        x = torch.from_numpy(load_wav(path))[None].to(dev)
        return model(x, args.chunk)[0].softmax(-1).cpu().numpy()

    mono = [(post(it["path"]), LANGS.index(it["lang"])) for it in read_jsonl("eval")]
    sw = [(post(s["path"]), LANGS.index(s["segments"][1][0]), s["switch_s"], s["pair"]) for s in read_jsonl("switch")]

    rows = []
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        for dwell in (1, 3, 6):
            ts = min(th + 0.1, 0.95)
            t_ok, wrong = [], []
            for p, y in mono:
                c = commit_stream(p, theta_commit=th, theta_switch=ts, dwell=dwell)
                idx = np.flatnonzero(c >= 0)
                wrong.append(bool(len(idx)) and c[idx[0]] != y)
                ok = np.flatnonzero(c == y)
                t_ok.append(frame_time([ok[0]])[0] if len(ok) else np.nan)
            lags, flips = [], []
            for p, new, s, pair in sw:
                c = commit_stream(p, theta_commit=th, theta_switch=ts, dwell=dwell)
                if pair in ("hi->en", "en->hi", "hi->en-in", "en-in->hi"):
                    lags.append(switch_lag(np.arange(len(p)), c, new, s))
                    flips.append(flips_per_min(c[c >= 0], 1))
            rows.append({"theta_commit": th, "theta_switch": ts, "dwell_frames": dwell,
                         "first_correct_commit_s_median": float(np.nanmedian(t_ok)),
                         "wrong_first_commit_rate": float(np.mean(wrong)),
                         "never_correct_rate": float(np.isnan(t_ok).mean()),
                         "hi_en_switch_lag_s_median": float(np.nanmedian(lags)),
                         "hi_en_switch_missed_rate": float(np.isnan(lags).mean()),
                         "flips_per_min": float(np.nanmean(flips))})
            r = rows[-1]
            print(f"theta {th:.1f} dwell {dwell}: commit {r['first_correct_commit_s_median']:.2f}s "
                  f"wrong {r['wrong_first_commit_rate']:.3f} | switch lag {r['hi_en_switch_lag_s_median']:.2f}s "
                  f"missed {r['hi_en_switch_missed_rate']:.2f} flips/min {r['flips_per_min']:.1f}")
    out = ROOT / "results/commit_sweep.json"
    out.write_text(json.dumps({"ckpt": args.ckpt, "chunk": args.chunk, "rows": rows}, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
