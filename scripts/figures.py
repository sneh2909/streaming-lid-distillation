"""Figures for the README: switch trace, held-out agreement curves, commit-policy trade-off."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from slid.audio import load_wav
from slid.commit import commit_stream
from slid.config import LANGS
from slid.metrics import frame_time
from slid.student import load_student

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e6e5df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2})


def switch_trace():
    model = load_student(ROOT / "checkpoints/final/student.pt")
    targets = torch.load(ROOT / "data/targets/ensemble/causal.pt")
    clips = [json.loads(l) for l in open(ROOT / "data/manifests/switch.jsonl") if '"hi->en"' in l]
    sw = clips[0]
    x = torch.from_numpy(load_wav(sw["path"]))[None]
    with torch.no_grad():
        p = model(x, 4)[0].softmax(-1).numpy()
    t = frame_time(np.arange(len(p)))
    tf, tq = targets[sw["path"]]
    tt = frame_time(tf.numpy())
    hi, en = LANGS.index("hi"), LANGS.index("en")
    com = commit_stream(p, theta_commit=0.8, theta_switch=0.9, dwell=3)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(tt, tq[:, hi], color=BLUE, ls="--", lw=1.5)
    ax.plot(tt, tq[:, en], color=ORANGE, ls="--", lw=1.5)
    ax.plot(t, p[:, hi], color=BLUE)
    ax.plot(t, p[:, en], color=ORANGE)
    ax.axvline(sw["switch_s"], color=INK, lw=1)
    ax.text(sw["switch_s"] + 0.05, 1.02, "true switch", color=INK, fontsize=9)
    for k, color, name in ((hi, BLUE, "committed: hi"), (en, ORANGE, "committed: en")):
        on = com == k
        if on.any():
            ax.fill_between(t, -0.12, -0.04, where=on, color=color, step="post", lw=0)
            ax.text(t[np.flatnonzero(on)[0]], -0.2, name, color=INK, fontsize=8)
    ax.text(t[-1] + 0.05, p[-1, en], "student en", color=INK, fontsize=9, va="center")
    ax.text(t[-1] + 0.05, p[-1, hi], "student hi", color=INK, fontsize=9, va="center")
    ax.plot([], [], color=MUTED, ls="--", lw=1.5, label="teacher (causal 3 s window)")
    ax.plot([], [], color=MUTED, label="student (320 ms chunks)")
    ax.legend(loc="center right", frameon=False, fontsize=9)
    ax.set_ylim(-0.25, 1.08)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("posterior")
    ax.set_title("Hindi → English, held-out switch clip", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "switch_trace.png", dpi=150)


def agreement_curves():
    runs = [("final_v1data", "ensemble teacher, causal", BLUE),
            ("indic-transcribe_causal", "Indic-T, causal", AQUA),
            ("indic-transcribe_prefix", "Indic-T, prefix", YELLOW),
            ("indic-transcribe_centered", "Indic-T, centred", ORANGE),
            ("indic-transcribe_full", "Indic-T, full clip", MUTED)]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for name, label, color in runs:
        f = ROOT / f"checkpoints/{name}/train_log.json"
        if not f.exists():
            continue
        ev = json.loads(f.read_text())["eval"]
        s = [e["step"] for e in ev]
        a = [e["heldout_kd"] for e in ev]
        ax.plot(s, a, color=color, marker="o", ms=4)
        nudge = {"indic-transcribe_centered": 0.014, "indic-transcribe_prefix": 0.004, "indic-transcribe_causal": -0.012, "final_v1data": -0.012}.get(name, 0)
        ax.text(s[-1] + 80, a[-1] + nudge, label, color=INK, fontsize=8, va="center")
    ax.set_xlim(0, 6400)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("held-out KD loss (own targets)")
    ax.set_title("Held-out KD loss by target type (original data)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "heldout_kd.png", dpi=150)


def commit_tradeoff():
    rows = json.loads((ROOT / "results/final/commit_sweep_final.json").read_text())["rows"]
    rows = [r for r in rows if r["dwell_frames"] == 3]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    xs = [r["first_correct_commit_s_median"] for r in rows]
    ys = [100 * r["wrong_first_commit_rate"] for r in rows]
    ax.plot(xs, ys, color=BLUE, marker="o", ms=6)
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(f"θ={r['theta_commit']:.1f}", (x, y), textcoords="offset points", xytext=(6, 4),
                    color=INK, fontsize=8)
    ax.set_xlabel("median time to first correct commit (s)")
    ax.set_ylabel("wrong first commit (%)")
    ax.set_title("Commit threshold: waiting longer buys fewer wrong routes", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "commit_tradeoff.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    switch_trace()
    commit_tradeoff()
    agreement_curves()
    print("figures ->", OUT)
