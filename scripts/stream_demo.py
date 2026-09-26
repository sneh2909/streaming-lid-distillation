"""Stream a wav through the shipped student like a live call: 20 ms packets in, decisions out.

    python scripts/stream_demo.py path/to.wav [--chunk 4] [--theta 0.8]

Prints the committed routing language whenever it changes, with the wall-clock audio time.
"""
import argparse
from pathlib import Path

import numpy as np

from slid.audio import SR, load_wav
from slid.config import LANGS
from slid.streaming import StreamingSession
from slid.student import FRAME_S, load_student

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--chunk", type=int, default=4, help="frames per chunk: 1/2/4/8 = 80/160/320/640 ms")
    ap.add_argument("--theta", type=float, default=0.8)
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/final/student.pt"))
    args = ap.parse_args()

    session = StreamingSession(load_student(args.ckpt), args.chunk)
    x, packet = load_wav(args.wav), SR // 50                           # 20 ms telephony packets
    alpha, theta_switch, dwell = 0.3, min(args.theta + 0.1, 0.95), 3
    s, committed, run, cand, t = None, -1, 0, -1, 0
    for i in range(0, len(x), packet):
        for p in session.push(x[i: i + packet]):
            t += 1
            s = p if s is None else (1 - alpha) * s + alpha * p
            k = int(s.argmax())
            before = committed
            if committed < 0 and s[k] >= args.theta:
                committed = k
            elif committed >= 0 and k != committed and s[k] >= theta_switch:
                run, cand = (run + 1, k) if k == cand else (1, k)
                if run >= dwell:
                    committed, run = k, 0
            elif committed >= 0:
                run, cand = 0, -1
            if committed != before:
                print(f"{t * FRAME_S:6.2f} s  route -> {LANGS[committed]}  (p={s[committed]:.2f})")
    print(f"{t * FRAME_S:6.2f} s  end of audio; final route {LANGS[committed] if committed >= 0 else 'uncommitted'}")


if __name__ == "__main__":
    main()
