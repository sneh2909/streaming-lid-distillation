"""Streaming LID student: causal log-mel -> 8x causal conv subsampling (80 ms frames)
-> chunked causal Conformer -> per-frame language logits.

Causality contract: the output at frame t depends only on audio up to the END of the chunk
containing t. Convolutions are left-padded (never see the future); self-attention uses a
chunk mask (a frame sees its own chunk plus `left_chunks` previous chunks). So lookahead is
bounded by the chunk size no matter how many layers are stacked. We read the decision at
the last frame of each chunk, where the lookahead is exactly zero.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from slid.audio import SR

HOP = 160            # 10 ms feature hop
WIN = 400            # 25 ms window
SUBSAMPLE = 8        # -> 80 ms per encoder frame
FRAME_S = HOP * SUBSAMPLE / SR


class LogMel(nn.Module):
    """center=False so feature frame i only uses samples [i*HOP, i*HOP + WIN)."""

    def __init__(self, n_mels: int = 80):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=WIN, win_length=WIN, hop_length=HOP,
                                                        n_mels=n_mels, center=False)
        # Global CMVN (fixed stats from the train set) instead of per-utterance normalisation,
        # which would leak future audio into every frame.
        self.register_buffer("mean", torch.zeros(n_mels))
        self.register_buffer("std", torch.ones(n_mels))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:          # [B, N] -> [B, T, n_mels]
        x = (self.mel(wav) + 1e-6).log().transpose(1, 2)
        return (x - self.mean) / self.std


class CausalConv1d(nn.Conv1d):
    def __init__(self, cin, cout, k, stride=1, groups=1):
        super().__init__(cin, cout, k, stride=stride, groups=groups)
        self.left = k - 1

    def forward(self, x):
        return super().forward(F.pad(x, (self.left, 0)))


class Subsample(nn.Module):
    """Three stride-2 causal convs over time: 10 ms -> 80 ms frames."""

    def __init__(self, n_mels: int, d: int):
        super().__init__()
        self.convs = nn.ModuleList([CausalConv1d(n_mels if i == 0 else d, d, 3, stride=2) for i in range(3)])

    def forward(self, x):                                        # [B, T, F] -> [B, T/8, d]
        x = x.transpose(1, 2)
        for c in self.convs:
            x = F.silu(c(x))
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, kernel: int, dropout: float):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.ln_att = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_conv = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = CausalConv1d(d, d, kernel, groups=d)
        self.norm_conv = nn.LayerNorm(d)                         # LayerNorm, not BatchNorm: no batch/time stats
        self.pw2 = nn.Linear(d, d)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.ln_out = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + 0.5 * self.ff1(x)
        h = self.ln_att(x)
        x = x + self.drop(self.att(h, h, h, attn_mask=mask, need_weights=False)[0])
        h = F.glu(self.pw1(self.ln_conv(x)), dim=-1).transpose(1, 2)
        h = F.silu(self.norm_conv(self.dw(h).transpose(1, 2)))
        x = x + self.drop(self.pw2(h))
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


def chunk_mask(T: int, chunk: int, left_chunks: int, device=None) -> torch.Tensor:
    """Boolean [T, T]; True = NOT allowed. Frame i sees frames of its own chunk and the
    `left_chunks` chunks before it (left_chunks < 0 means unlimited history)."""
    c = torch.arange(T, device=device) // chunk
    allowed = c[None, :] <= c[:, None]
    if left_chunks >= 0:
        allowed &= c[None, :] >= c[:, None] - left_chunks
    return ~allowed


def sinusoid(T: int, d: int, device=None) -> torch.Tensor:
    pos = torch.arange(T, device=device).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class StreamingLID(nn.Module):
    def __init__(self, n_langs: int, d: int = 144, layers: int = 6, heads: int = 4, ff: int = 576,
                 kernel: int = 15, dropout: float = 0.1, left_chunks: int = -1):
        super().__init__()
        self.feat = LogMel()
        self.sub = Subsample(80, d)
        self.blocks = nn.ModuleList([ConformerBlock(d, heads, ff, kernel, dropout) for _ in range(layers)])
        self.head = nn.Linear(d, n_langs)
        self.left_chunks = left_chunks

    def forward(self, wav: torch.Tensor, chunk: int) -> torch.Tensor:
        """[B, N] waveform -> [B, T, n_langs] logits, T = number of 80 ms frames."""
        x = self.sub(self.feat(wav))
        T = x.shape[1]
        x = x + sinusoid(T, x.shape[2], x.device)
        mask = chunk_mask(T, chunk, self.left_chunks, x.device)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(x)


def n_frames(n_samples: int) -> int:
    """Encoder frames produced for a waveform of n_samples (matches forward())."""
    t = (n_samples - WIN) // HOP + 1
    for _ in range(3):
        t = (t - 1) // 2 + 1
    return t


def frame_end_sample(t: int) -> int:
    """Last input sample (exclusive) that encoder frame t depends on, given causal convs."""
    f = t
    for _ in range(3):
        f = 2 * f                                                # causal stride-2 conv output f sees inputs <= 2f
    return f * HOP + WIN
