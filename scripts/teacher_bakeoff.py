"""Score candidate teachers on the same held-out clips.

Per teacher:
  * accuracy on prefixes of 0.5/1/2/3 s and the full clip, clean and simulated telephony,
    unrestricted (argmax over LANGS + "other") and restricted to the routing set
  * ECE of the restricted posterior (we distil soft posteriors, so calibration matters)
  * prefix stability: share of clips whose argmax over growing prefixes changes > once
  * switch lag of the teacher itself on concatenated switch clips (causal 3 s window, 0.32 s grid)
  * wall-clock per call
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from slid.audio import SR, load_wav, telephony
from slid.config import LANGS
from slid.metrics import ece, switch_lag
from slid.targets import build_targets
from slid.teachers import ROUTE, load_teacher, restrict

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = [0.5, 1.0, 2.0, 3.0, None]


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--max-per-lang", type=int, default=60)
    ap.add_argument("--manifest", default="eval", help="eval, or train for the ensemble-weight dev slice")
    ap.add_argument("--no-switch", action="store_true")
    args = ap.parse_args()
    tag = args.teacher if args.manifest == "eval" else f"{args.teacher}.{args.manifest}"

    items = read_jsonl(ROOT / f"data/manifests/{args.manifest}.jsonl")
    group = lambda it: "en-in" if it.get("source") == "svarah" else it["lang"]
    counts: dict[str, int] = {}
    kept = []
    for it in items:
        if counts.get(group(it), 0) < args.max_per_lang:
            counts[group(it)] = counts.get(group(it), 0) + 1
            kept.append(it)
    items = kept
    groups = np.array([group(it) for it in items])
    y = np.array([LANGS.index(it["lang"]) for it in items])
    clean = [load_wav(it["path"]) for it in items]
    tel = [telephony(x, seed=i) for i, x in enumerate(clean)]

    teacher = load_teacher(args.teacher)
    teacher.probs([clean[0][:SR]])                               # warm-up
    res = {"teacher": args.teacher, "n_clips": len(items), "by_condition": {}}
    n_calls, t_total = 0, 0.0
    saved = {"y": y, "groups": groups}
    for cond, wavs in (("clean", clean), ("telephony", tel)):
        rows = {}
        argmax_seq = []
        for pre in PREFIXES:
            segs = [x if pre is None else x[: int(pre * SR)] for x in wavs]
            t0 = time.time()
            p = teacher.probs(segs)
            t_total += time.time() - t0
            n_calls += len(segs)
            q = restrict(p)
            key = "full" if pre is None else f"{pre:g}s"
            saved[f"{cond}/{key}"] = p
            per_lang_acc = {g: float((q[groups == g].argmax(1) == y[groups == g]).mean())
                            for g in sorted(set(groups))}
            rows[key] = {
                "acc_restricted": float((q.argmax(1) == y).mean()),
                "acc_unrestricted": float((p.argmax(1) == y).mean()),
                "other_mass": float(p[:, -1].mean()),
                "ece_restricted": ece(q, y),
                "acc_by_lang": per_lang_acc,
            }
            argmax_seq.append(q.argmax(1))
        seq = np.stack(argmax_seq, 1)
        rows["prefix_flip_gt1"] = float(((np.diff(seq, axis=1) != 0).sum(1) > 1).mean())
        res["by_condition"][cond] = rows

    out = ROOT / "results/teachers"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{tag}.npz", **saved)
    lags = {}
    for sw in ([] if args.no_switch else read_jsonl(ROOT / "data/manifests/switch.jsonl")):
        x = load_wav(sw["path"])
        frames, q = build_targets(teacher, x, "causal", every=4, window_s=3.0)
        b = sw["segments"][1][0]
        lag = switch_lag(frames, q.argmax(1), LANGS.index(b), sw["switch_s"])
        lags.setdefault(sw["pair"], []).append(lag)
    res["switch_lag_s_causal3s"] = {k: {"median": float(np.nanmedian(v)), "missed": int(np.isnan(v).sum()),
                                        "n": len(v)} for k, v in lags.items()}
    res["ms_per_call"] = 1000 * t_total / n_calls

    (out / f"{tag}.json").write_text(json.dumps(res, indent=2))
    c, t = res["by_condition"]["clean"], res["by_condition"]["telephony"]
    print(f"{args.teacher}: acc@1s {c['1s']['acc_restricted']:.3f} acc@3s {c['3s']['acc_restricted']:.3f} "
          f"full {c['full']['acc_restricted']:.3f} | tel full {t['full']['acc_restricted']:.3f} | "
          f"hi->en lag {res['switch_lag_s_causal3s'].get('hi->en', {}).get('median')} s | "
          f"{res['ms_per_call']:.1f} ms/call")


if __name__ == "__main__":
    main()
