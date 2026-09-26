import numpy as np
import pytest
import torch

from slid.streaming import StreamingSession
from slid.student import StreamingLID


def _model(left=16):
    torch.manual_seed(0)
    m = StreamingLID(n_langs=3, d=32, layers=2, heads=2, ff=64, kernel=5, dropout=0.0,
                     left_frames=left, rel_pos=True).eval()
    torch.nn.init.normal_(m.rel_bias.weight, std=0.5)        # non-zero bias so positions matter
    return m


def _model_v1():
    torch.manual_seed(0)
    return StreamingLID(n_langs=3, d=32, layers=2, heads=2, ff=64, kernel=5, dropout=0.0).eval()


@pytest.mark.parametrize("make", [_model, _model_v1], ids=["v2-bounded-relpos", "v1-growing-abspos"])
def test_stream_matches_whole_clip_forward(make):
    m = make()
    rng = np.random.default_rng(0)
    wav = rng.standard_normal(16000 * 7).astype(np.float32)
    for chunk in (1, 2, 4, 8):
        with torch.no_grad():
            ref = m(torch.from_numpy(wav)[None], chunk)[0].softmax(-1).numpy()
        s = StreamingSession(m, chunk)
        outs, i = [], 0
        while i < len(wav):                                   # uneven pieces, like network packets
            n = int(rng.integers(100, 5000))
            outs.append(s.push(wav[i: i + n]))
            i += n
        got = np.concatenate(outs)
        assert len(got) > 0 and len(got) <= len(ref)
        np.testing.assert_allclose(got, ref[: len(got)], atol=1e-4, err_msg=f"chunk {chunk}")


def test_memory_is_constant_on_long_audio():
    m = _model(left=16)
    s = StreamingSession(m, 4)
    wav = np.random.default_rng(1).standard_normal(16000 * 120).astype(np.float32)   # 2 minutes
    for i in range(0, len(wav), 16000):
        s.push(wav[i: i + 16000])
        assert s.cache_frames() <= 16
        assert len(s.audio) < 16000 + 5000
    assert s.t > 1400                                          # ~1500 frames of 80 ms emitted
