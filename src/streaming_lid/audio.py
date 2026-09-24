"""Audio I/O and a streaming-safe log-mel frontend."""

from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE, WIN_LENGTH


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Load mono float audio and resample without requiring ffmpeg."""
    samples, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples).mean(dim=1)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform.contiguous()


def feature_frame_count(num_samples: int) -> int:
    """Number of center=False STFT frames, including short-audio padding."""
    padded = max(num_samples, WIN_LENGTH)
    return 1 + (padded - N_FFT) // HOP_LENGTH


class LogMelFrontend(torch.nn.Module):
    """25 ms / 10 ms log-mels with no centering or utterance-level CMVN.

    `center=False` matters: centered STFT and whole-utterance normalization both
    leak future samples into otherwise causal model outputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            center=False,
            power=2.0,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 2:
            raise ValueError(f"expected [batch, samples], got {tuple(waveform.shape)}")
        if waveform.shape[-1] < WIN_LENGTH:
            waveform = F.pad(waveform, (0, WIN_LENGTH - waveform.shape[-1]))
        mel = self.mel(waveform)
        return torch.log(mel.clamp_min(1e-5)).transpose(1, 2)
