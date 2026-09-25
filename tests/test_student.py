import numpy as np
import torch

from slid.student import FRAME_S, StreamingLID, chunk_mask, frame_end_sample, n_frames
from slid.targets import grid_frames, teacher_inputs


def _model(left_chunks=-1):
    torch.manual_seed(0)
    return StreamingLID(n_langs=3, d=32, layers=2, heads=2, ff=64, kernel=5, dropout=0.0,
                        left_chunks=left_chunks).eval()


def test_frame_count_matches_forward():
    m = _model()
    for n in (16000, 23456, 80000):
        with torch.no_grad():
            assert m(torch.randn(1, n), chunk=4).shape[1] == n_frames(n)


def test_chunk_mask_lookahead_is_bounded_by_chunk_end():
    mask = chunk_mask(8, chunk=4, left_chunks=-1)
    assert not mask[0, 3] and mask[0, 4]        # frame 0 sees its chunk (0..3), not the next
    assert not mask[7, 0] and mask[3, 7]


def test_no_leak_beyond_chunk_end():
    """Changing audio after frame t's chunk ends must not change output at frame t."""
    m = _model()
    for chunk in (1, 2, 4, 8):
        x = torch.randn(1, 48000)
        T = n_frames(x.shape[1])
        t = chunk * (T // (2 * chunk)) - 1                 # last frame of a chunk mid-clip
        cut = frame_end_sample(t)
        y = x.clone()
        y[:, cut:] = torch.randn_like(y[:, cut:])
        with torch.no_grad():
            a, b = m(x, chunk), m(y, chunk)
        assert torch.allclose(a[:, : t + 1], b[:, : t + 1], atol=1e-5), chunk
        assert not torch.allclose(a[:, t + 1:], b[:, t + 1:], atol=1e-5)


def test_frame_rate_is_80ms():
    assert abs(FRAME_S - 0.08) < 1e-9


def test_causal_and_prefix_targets_never_see_past_frame_end():
    x = np.arange(48000, dtype=np.float32)
    frames = grid_frames(len(x), every=4, min_s=0.4)
    for kind in ("prefix", "causal"):
        for t, seg in zip(frames, teacher_inputs(x, frames, kind, window_s=3.0)):
            assert seg[-1] < frame_end_sample(int(t))
    for t, seg in zip(frames, teacher_inputs(x, frames, "centered", window_s=3.0)):
        assert seg[-1] >= min(frame_end_sample(int(t)), len(x)) - 1   # centered does peek ahead
