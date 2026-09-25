import numpy as np

from slid.student import FRAME_S, frame_end_sample
from slid.audio import SR


def ece(q: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the top-1 probability."""
    conf, pred = q.max(1), q.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(err)


def frame_time(t) -> np.ndarray:
    """Wall-clock time (s) at which encoder frame t can first be emitted."""
    return np.array([frame_end_sample(int(i)) for i in np.atleast_1d(t)]) / SR


def switch_lag(frames: np.ndarray, labels: np.ndarray, new: int, switch_s: float) -> float:
    """Seconds from the true switch until the label becomes `new` and stays `new` to the end.
    NaN if it never settles on `new`."""
    times = frame_time(frames)
    after = times >= switch_s
    ok = labels == new
    for i in np.flatnonzero(after):
        if ok[i:].all():
            return float(times[i] - switch_s)
    return float("nan")


def flips_per_min(labels: np.ndarray, n_ref_changes: int, frame_s: float = FRAME_S) -> float:
    """Label changes beyond the reference number of changes, per minute of audio."""
    extra = max(0, int((np.diff(labels) != 0).sum()) - n_ref_changes)
    return 60.0 * extra / (len(labels) * frame_s)
