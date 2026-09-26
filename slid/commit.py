"""Commit policy on top of the per-frame student posterior (the Part 2 design, made runnable).

    s_t = (1 - a) s_{t-1} + a p_t                        exponential smoothing of posteriors
    first commit:  max_k s_tk >= theta_commit          -> committed = argmax
    switch:        some k != committed has s_tk >= theta_switch
                   for `dwell` consecutive frames      -> committed = k

theta_switch > theta_commit plus the dwell requirement is the hysteresis that stops
flip-flopping; -1 means "not committed yet" (route to the prior / default ASR).
"""
import numpy as np


def commit_stream(p: np.ndarray, alpha: float = 0.3, theta_commit: float = 0.7,
                  theta_switch: float = 0.8, dwell: int = 3) -> np.ndarray:
    committed, run, cand = -1, 0, -1
    s = p[0].copy()
    out = np.empty(len(p), dtype=int)
    for t in range(len(p)):
        s = (1 - alpha) * s + alpha * p[t] if t else p[0].copy()
        k = int(s.argmax())
        if committed < 0:
            if s[k] >= theta_commit:
                committed = k
        elif k != committed and s[k] >= theta_switch:
            run = run + 1 if k == cand else 1
            cand = k
            if run >= dwell:
                committed, run = k, 0
        else:
            run, cand = 0, -1
        out[t] = committed
    return out
