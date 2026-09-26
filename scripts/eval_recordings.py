"""Run the VAD-gated turn pipeline on real recordings in data/user_recordings/ (wav/flac/ogg).

For every recording: Silero VAD turns -> fresh student stream per turn -> early route (first
commit) and final LID at the endpoint, printed as a timeline, plus a posterior plot saved to
results/recordings/<name>.png. Recordings stay local (gitignored).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from slid.audio import SR, load_wav
from slid.commit import commit_stream
from slid.config import LANGS
from slid.metrics import frame_time
from slid.student import load_student

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"hi": "#2a78d6", "en": "#eb6834"}


@torch.no_grad()
def main(theta: float = 0.8) -> None:
    m, vad = load_student(ROOT / "checkpoints/final/student.pt"), load_silero_vad()
    files = sorted(p for p in (ROOT / "data/user_recordings").iterdir() if p.suffix.lower() in (".wav", ".flac", ".ogg"))
    out_dir = ROOT / "results/recordings"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        x = load_wav(str(f))
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 10 ** (-23 / 20)
        ts = get_speech_timestamps(torch.from_numpy(x / (np.abs(x).max() + 1e-9) * 0.5), vad, sampling_rate=SR,
                                   min_silence_duration_ms=300)
        print(f"\n== {f.name} ({len(x) / SR:.1f} s, {len(ts)} turns)")
        fig, ax = plt.subplots(figsize=(10, 3))
        for t in ts:
            a, b = t["start"], t["end"]
            p = m(torch.from_numpy(x[a:b])[None], 4)[0].softmax(-1).numpy()
            c = commit_stream(p, theta_commit=theta, theta_switch=min(theta + 0.1, 0.95))
            idx = np.flatnonzero(c >= 0)
            tt = a / SR + frame_time(np.arange(len(p)))
            early = f"{LANGS[c[idx[0]]]} at +{frame_time([idx[0]])[0]:.2f} s" if len(idx) else "none (keep previous)"
            top = np.argsort(-p[-1])[:2]
            print(f"  {a / SR:6.2f}-{b / SR:6.2f} s  early route: {early:22s} final: {LANGS[top[0]]} "
                  f"({p[-1, top[0]]:.2f}; next {LANGS[top[1]]} {p[-1, top[1]]:.2f})")
            for lang, col in COLORS.items():
                ax.plot(tt, p[:, LANGS.index(lang)], color=col, lw=1.5, label=lang if t is ts[0] else None)
            other = 1 - p[:, [LANGS.index("hi"), LANGS.index("en")]].sum(1)
            ax.plot(tt, other, color="#8a8a85", lw=1, label="other 5" if t is ts[0] else None)
            ax.axvspan(a / SR, b / SR, color="#e6e5df", alpha=0.4, lw=0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("posterior")
        ax.set_title(f"{f.name}: grey bands = VAD turns (fresh stream each)", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"{f.stem}.png", dpi=130)
        plt.close(fig)
    if not files:
        print("no recordings in data/user_recordings/")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.8)
