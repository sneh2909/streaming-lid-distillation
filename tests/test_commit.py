import numpy as np

from slid.commit import commit_stream
from slid.metrics import switch_lag


def _stream(seq):
    return np.array([[0.9, 0.1] if s == 0 else [0.1, 0.9] if s == 1 else [0.5, 0.5] for s in seq])


def test_uncertain_start_is_uncommitted():
    out = commit_stream(_stream([2, 2, 2]))
    assert (out == -1).all()


def test_single_frame_blip_does_not_flip():
    seq = [0] * 20 + [1] + [0] * 20
    out = commit_stream(_stream(seq), alpha=1.0, dwell=3)
    assert set(out[out >= 0]) == {0}


def test_sustained_switch_is_followed_after_dwell():
    seq = [0] * 20 + [1] * 20
    out = commit_stream(_stream(seq), alpha=1.0, dwell=3)
    assert out[19] == 0 and out[-1] == 1
    assert np.flatnonzero(out == 1)[0] == 20 + 2          # third consecutive frame of evidence


def test_switch_lag_requires_label_to_stay():
    frames = np.arange(40)
    labels = np.array([0] * 20 + [1] * 3 + [0] * 2 + [1] * 15)
    lag = switch_lag(frames, labels, new=1, switch_s=0.0)
    assert np.isclose(lag, (25 * 8 * 160 + 400) / 16000)
