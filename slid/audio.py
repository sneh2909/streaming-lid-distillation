import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

SR = 16000


def load_wav(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = AF.resample(torch.from_numpy(x), sr, SR).numpy()
    return x


def trim_silence(x: np.ndarray, top_db: float = 35.0, frame: int = 400) -> np.ndarray:
    """Drop leading/trailing frames more than top_db below the loudest frame."""
    n = len(x) // frame
    if n == 0:
        return x
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms / rms.max())
    keep = np.flatnonzero(db > -top_db)
    return x[keep[0] * frame: (keep[-1] + 1) * frame]


def telephony(x: np.ndarray, snr_db: float | None = 20.0, seed: int = 0) -> np.ndarray:
    """Simulate a narrowband call leg: 300-3400 Hz band, 8 kHz mu-law, back to 16 kHz."""
    t = torch.from_numpy(x).float()
    t = AF.highpass_biquad(t, SR, 300.0)
    t = AF.lowpass_biquad(t, SR, 3400.0)
    t = AF.resample(t, SR, 8000)
    t = t / (t.abs().max() + 1e-8) * 0.9
    t = AF.mu_law_decoding(AF.mu_law_encoding(t, 256), 256)
    if snr_db is not None:
        g = torch.Generator().manual_seed(seed)
        noise = torch.randn(t.shape, generator=g)
        p_sig, p_noise = t.pow(2).mean(), noise.pow(2).mean()
        t = t + noise * torch.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    return AF.resample(t, 8000, SR).numpy()
