"""Distillation targets: which audio the teacher is allowed to hear for student frame t.

Let e(t) be the last input sample student frame t depends on (frame_end_sample).
    full      q_t = T(x[0 : N])                 whole clip, broadcast   (naive: sees the future)
    centered  q_t = T(x[e - W/2 : e + W/2])     local, sees W/2 of future (the window a
                                                bidirectional frame-level teacher would use)
    prefix    q_t = T(x[0 : e])                 everything heard so far (information-matched)
    causal    q_t = T(x[e - W : e])             last W seconds heard so far (information-matched,
                                                and able to forget the previous language)
Only `prefix` and `causal` use nothing the student has not heard, so for them the optimal
student can reach zero KL; `full` and `centered` ask it to predict audio it never sees.
"""
import numpy as np

from slid.audio import SR
from slid.student import frame_end_sample, n_frames
from slid.teachers import Teacher, restrict

KINDS = ("full", "centered", "prefix", "causal")


def grid_frames(n_samples: int, every: int, min_s: float) -> np.ndarray:
    T = n_frames(n_samples)
    return np.array([t for t in range(every - 1, T, every) if frame_end_sample(t) >= min_s * SR], dtype=int)


def teacher_inputs(x: np.ndarray, frames: np.ndarray, kind: str, window_s: float) -> list[np.ndarray]:
    W = int(window_s * SR)
    out = []
    for t in frames:
        e = min(frame_end_sample(int(t)), len(x))
        if kind == "prefix":
            seg = x[:e]
        elif kind == "causal":
            seg = x[max(0, e - W): e]
        elif kind == "centered":
            seg = x[max(0, e - W // 2): e + W // 2]
        else:
            raise ValueError(kind)
        out.append(seg)
    return out


def build_targets(teacher: Teacher, x: np.ndarray, kind: str, every: int = 4, min_s: float = 0.4,
                  window_s: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """-> (frames [K], q [K, len(LANGS)]) restricted to the routing set."""
    frames = grid_frames(len(x), every, min_s)
    if kind == "full":
        q = np.repeat(restrict(teacher.probs([x])), len(frames), axis=0)
    else:
        q = restrict(teacher.probs(teacher_inputs(x, frames, kind, window_s)))
    return frames, q.astype(np.float32)
