"""Log-linear teacher ensemble: log q = w log q_A + (1 - w) log q_B (+ const), over the routing set.

  tune    pick w on a slice of the TRAIN pool (never eval) by mean NLL of the true language over
          prefixes 1/2/3 s + full, clean and telephony
  score   report the ensemble on the eval set in the same format as teacher_bakeoff.py
  targets combine the two teachers' precomputed targets of one kind into data/targets/ensemble/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from slid.config import LANGS
from slid.metrics import ece
from slid.teachers import restrict

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/teachers"
KEYS = ["0.5s", "1s", "2s", "3s", "full"]
TUNE_KEYS = ["1s", "2s", "3s", "full"]


def combine(qa: np.ndarray, qb: np.ndarray, w: float) -> np.ndarray:
    logq = w * np.log(np.clip(qa, 1e-6, 1)) + (1 - w) * np.log(np.clip(qb, 1e-6, 1))
    logq -= logq.max(-1, keepdims=True)
    q = np.exp(logq)
    return q / q.sum(-1, keepdims=True)


def load(tag):
    return dict(np.load(RES / f"{tag}.npz", allow_pickle=True))


def tune(a: str, b: str) -> float:
    A, B = load(f"{a}.train"), load(f"{b}.train")
    y = A["y"]
    assert (y == B["y"]).all()
    rows = []
    for w in np.round(np.arange(0, 1.0001, 0.05), 2):
        nll, acc = [], []
        for cond in ("clean", "telephony"):
            for k in TUNE_KEYS:
                q = combine(restrict(A[f"{cond}/{k}"]), restrict(B[f"{cond}/{k}"]), w)
                nll.append(-np.log(np.clip(q[np.arange(len(y)), y], 1e-9, 1)).mean())
                acc.append((q.argmax(1) == y).mean())
        rows.append({"w": float(w), "nll": float(np.mean(nll)), "acc": float(np.mean(acc))})
    best = min(rows, key=lambda r: r["nll"])
    (RES / f"ensemble_tuning.json").write_text(json.dumps({"a": a, "b": b, "grid": rows, "best": best}, indent=2))
    for r in rows:
        print(f"w={r['w']:.2f}  nll {r['nll']:.3f}  acc {r['acc']:.3f}" + ("  <-- best" if r is best else ""))
    return best["w"]


def score(a: str, b: str, w: float) -> None:
    A, B = load(a), load(b)
    y, groups = A["y"], A["groups"]
    res = {"teacher": f"ensemble({a}^{w:.2f} * {b}^{1 - w:.2f})", "w": w, "by_condition": {}}
    for cond in ("clean", "telephony"):
        rows, seq = {}, []
        for k in KEYS:
            q = combine(restrict(A[f"{cond}/{k}"]), restrict(B[f"{cond}/{k}"]), w)
            rows[k] = {"acc_restricted": float((q.argmax(1) == y).mean()), "ece_restricted": ece(q, y),
                       "acc_by_lang": {g: float((q[groups == g].argmax(1) == y[groups == g]).mean())
                                       for g in sorted(set(groups))}}
            seq.append(q.argmax(1))
        rows["prefix_flip_gt1"] = float(((np.diff(np.stack(seq, 1), axis=1) != 0).sum(1) > 1).mean())
        res["by_condition"][cond] = rows
    (RES / "ensemble.json").write_text(json.dumps(res, indent=2))
    c, t = res["by_condition"]["clean"], res["by_condition"]["telephony"]
    print("ensemble eval: " + " ".join(f"{k} {c[k]['acc_restricted']:.3f}" for k in KEYS)
          + f" | tel full {t['full']['acc_restricted']:.3f} | ECE@3s {c['3s']['ece_restricted']:.3f}")
    print("  full by group:", {g: round(v, 2) for g, v in c["full"]["acc_by_lang"].items()})
    print("  2s by group:  ", {g: round(v, 2) for g, v in c["2s"]["acc_by_lang"].items()})


def targets(a: str, b: str, w: float, kind: str) -> None:
    TA = torch.load(ROOT / f"data/targets/{a}/{kind}.pt")
    TB = torch.load(ROOT / f"data/targets/{b}/{kind}.pt")
    out = {}
    for p, (fa, qa) in TA.items():
        fb, qb = TB[p]
        assert torch.equal(fa, fb), p
        out[p] = (fa, torch.from_numpy(combine(qa.numpy(), qb.numpy(), w).astype(np.float32)))
    d = ROOT / "data/targets/ensemble"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(out, d / f"{kind}.pt")
    print("saved", d / f"{kind}.pt", len(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "score", "targets"])
    ap.add_argument("--a", default="indic-transcribe")
    ap.add_argument("--b", default="whisper-turbo")
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--kind", default="causal")
    args = ap.parse_args()
    w = args.w
    if w is None and args.cmd != "tune":
        w = json.loads((RES / "ensemble_tuning.json").read_text())["best"]["w"]
    if args.cmd == "tune":
        tune(args.a, args.b)
    elif args.cmd == "score":
        score(args.a, args.b, w)
    else:
        targets(args.a, args.b, w, args.kind)


if __name__ == "__main__":
    main()
