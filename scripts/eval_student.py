"""Evaluate a trained student as a stream, per chunk size (lookahead).

Monolingual held-out clips (FLEURS test + Svarah eval), clean and telephony:
  * accuracy of the raw argmax at 0.5/1/2/3 s after the clip starts
  * agreement with the teacher's causal-window targets, and ECE
  * time to first correct commit (commit policy in slid/commit.py) and wrong-first-commit rate
Switch clips:
  * switch lag of raw argmax, of the committed label, and of the TEACHER's own causal target
  * extra label changes per minute (flip-flops), raw vs committed
Plus CPU real-time factor.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from slid.audio import SR, load_wav, telephony
from slid.commit import commit_stream
from slid.config import LANGS
from slid.metrics import ece, flips_per_min, frame_time, switch_lag
from slid.student import StreamingLID

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = (1, 2, 4, 8)
AT_S = (0.5, 1.0, 2.0, 3.0)


def read_jsonl(name):
    return [json.loads(l) for l in open(ROOT / f"data/manifests/{name}.jsonl", encoding="utf-8")]


@torch.no_grad()
def posteriors(model, x, chunk, device):
    return model(torch.from_numpy(x)[None].to(device), chunk)[0].softmax(-1).cpu().numpy()


def frame_at(t_s: float, T: int) -> int:
    """Last frame emitted by wall-clock time t_s."""
    times = frame_time(np.arange(T))
    idx = np.flatnonzero(times <= t_s)
    return int(idx[-1]) if len(idx) else -1


def first_correct_commit(committed, y):
    idx = np.flatnonzero(committed >= 0)
    if not len(idx):
        return float("nan"), False
    first_ok = committed[idx[0]] == y
    ok = np.flatnonzero(committed == y)
    return (float(frame_time([ok[0]])[0]) if len(ok) else float("nan")), bool(not first_ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    model = StreamingLID(ck["n_langs"]).to(args.device).eval()
    model.load_state_dict(ck["state"])
    causal_t = torch.load(ROOT / f"data/targets/{args.teacher}/causal.pt")

    mono = read_jsonl("eval")
    group = lambda it: "en-in" if it.get("source") == "svarah" else it["lang"]
    switch = read_jsonl("switch")
    wav = {it["path"]: load_wav(it["path"]) for it in mono + switch}
    res = {"ckpt": args.ckpt, "teacher": args.teacher, "by_chunk": {}}

    for chunk in CHUNKS:
        r = {"lookahead_ms": chunk * 80}
        for cond in ("clean", "telephony"):
            acc_at = {a: [] for a in AT_S}
            by_group = {}
            agree, qs, ys, fc_t, fc_wrong = [], [], [], [], []
            for i, it in enumerate(mono):
                x = wav[it["path"]] if cond == "clean" else telephony(wav[it["path"]], seed=i)
                p = posteriors(model, x, chunk, args.device)
                y = LANGS.index(it["lang"])
                for a in AT_S:
                    f = frame_at(a, len(p))
                    if f >= 0 and a <= it["dur"]:
                        acc_at[a].append(p[f].argmax() == y)
                        if a == 2.0:
                            by_group.setdefault(group(it), []).append(p[f].argmax() == y)
                qs.append(p[-1]); ys.append(y)
                if cond == "clean":
                    frames, q = causal_t[it["path"]]
                    frames = frames.numpy()[frames.numpy() < len(p)]
                    agree.append((p[frames].argmax(1) == q.numpy()[: len(frames)].argmax(1)).mean())
                t_ok, wrong = first_correct_commit(commit_stream(p), y)
                fc_t.append(t_ok); fc_wrong.append(wrong)
            r[cond] = {
                "acc_at": {f"{a:g}s": float(np.mean(v)) for a, v in acc_at.items()},
                "acc_at_2s_by_group": {g: float(np.mean(v)) for g, v in sorted(by_group.items())},
                "acc_end": float((np.array(qs).argmax(1) == np.array(ys)).mean()),
                "ece_end": ece(np.array(qs), np.array(ys)),
                "first_correct_commit_s_median": float(np.nanmedian(fc_t)),
                "wrong_first_commit_rate": float(np.mean(fc_wrong)),
            }
            if cond == "clean":
                r[cond]["teacher_agreement_causal"] = float(np.mean(agree))

        lag = {}
        for sw in switch:
            x = wav[sw["path"]]
            p = posteriors(model, x, chunk, args.device)
            new = LANGS.index(sw["segments"][1][0])
            raw = p.argmax(1)
            com = commit_stream(p)
            fr = np.arange(len(p))
            tf, tq = causal_t[sw["path"]]
            d = lag.setdefault(sw["pair"], {"raw": [], "committed": [], "teacher": [], "flips_raw": [],
                                            "flips_committed": []})
            d["raw"].append(switch_lag(fr, raw, new, sw["switch_s"]))
            d["committed"].append(switch_lag(fr, com, new, sw["switch_s"]))
            d["teacher"].append(switch_lag(tf.numpy(), tq.numpy().argmax(1), new, sw["switch_s"]))
            d["flips_raw"].append(flips_per_min(raw, 1))
            d["flips_committed"].append(flips_per_min(com[com >= 0], 1))
        r["switch"] = {pair: {k: (float(np.nanmedian(v)) if not k.startswith("flips") else float(np.nanmean(v)))
                              for k, v in d.items()} | {"missed_committed": int(np.isnan(d["committed"]).sum()),
                                                         "n": len(d["raw"])}
                       for pair, d in lag.items()}
        res["by_chunk"][str(chunk)] = r
        c = r["clean"]
        hs = r["switch"].get("hi->en", {})
        print(f"chunk {chunk} ({chunk*80} ms): acc@1s {c['acc_at']['1s']:.3f} @2s {c['acc_at']['2s']:.3f} "
              f"end {c['acc_end']:.3f} agree {c['teacher_agreement_causal']:.3f} | tel end "
              f"{r['telephony']['acc_end']:.3f} | hi->en lag raw {hs.get('raw')} com {hs.get('committed')} "
              f"teacher {hs.get('teacher')}")

    cpu = StreamingLID(ck["n_langs"]).eval()
    cpu.load_state_dict(ck["state"])
    torch.set_num_threads(1)
    x = torch.randn(1, 10 * SR)
    with torch.no_grad():
        cpu(x, 4)
        t0 = time.time()
        for _ in range(3):
            cpu(x, 4)
    res["cpu_rtf_1thread"] = (time.time() - t0) / 3 / 10.0
    res["n_params"] = sum(p.numel() for p in cpu.parameters())
    out = Path(args.out or Path(args.ckpt).parent / "eval.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"params {res['n_params']/1e6:.2f}M  cpu RTF (1 thread) {res['cpu_rtf_1thread']:.4f}  -> {out}")


if __name__ == "__main__":
    main()
