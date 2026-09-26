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


def chunk_mask(T: int, chunk: int, left_frames: int = -1, device=None) -> torch.Tensor:
    """Boolean [T, T]; True = NOT allowed. Frame i sees every frame of its own chunk and at most
    `left_frames` frames before its chunk starts (-1 = unlimited history)."""
    idx = torch.arange(T, device=device)
    c = idx // chunk
    allowed = c[None, :] <= c[:, None]
    if left_frames >= 0:
        allowed &= idx[None, :] >= (c[:, None] * chunk - left_frames)
    return ~allowed


MAX_CHUNK = 8


def rel_index(q_pos: torch.Tensor, k_pos: torch.Tensor, left_frames: int) -> torch.Tensor:
    """Bucket of the relative distance q - k, clipped to [-MAX_CHUNK, left_frames + MAX_CHUNK]."""
    d = (q_pos[:, None] - k_pos[None, :]).clamp(-MAX_CHUNK, left_frames + MAX_CHUNK)
    return d + MAX_CHUNK


def sinusoid(T: int, d: int, device=None) -> torch.Tensor:
    pos = torch.arange(T, device=device).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class StreamingLID(nn.Module):
    """left_frames > 0 bounds attention history (constant cost per chunk, any call length);
    rel_pos replaces absolute sinusoidal positions with a learned per-head bias on q - k distance.
    Both default off so the first-generation checkpoints load unchanged."""

    def __init__(self, n_langs: int, d: int = 144, layers: int = 6, heads: int = 4, ff: int = 576,
                 kernel: int = 15, dropout: float = 0.1, left_frames: int = -1, rel_pos: bool = False):
        super().__init__()
        self.feat = LogMel()
        self.sub = Subsample(80, d)
        self.blocks = nn.ModuleList([ConformerBlock(d, heads, ff, kernel, dropout) for _ in range(layers)])
        self.head = nn.Linear(d, n_langs)
        self.left_frames, self.rel_pos, self.heads = left_frames, rel_pos, heads
        if rel_pos:
            assert left_frames > 0, "relative positions need bounded history"
            self.rel_bias = nn.Embedding(left_frames + 2 * MAX_CHUNK + 1, heads)
            nn.init.zeros_(self.rel_bias.weight)

    def attn_bias(self, q_pos, k_pos):
        """[heads, Tq, Tk] additive attention bias."""
        return self.rel_bias(rel_index(q_pos, k_pos, self.left_frames)).permute(2, 0, 1)

    def forward(self, wav: torch.Tensor, chunk: int) -> torch.Tensor:
        """[B, N] waveform -> [B, T, n_langs] logits, T = number of 80 ms frames."""
        x = self.sub(self.feat(wav))
        B, T, _ = x.shape
        mask = chunk_mask(T, chunk, self.left_frames, x.device)
        if self.rel_pos:
            pos = torch.arange(T, device=x.device)
            bias = self.attn_bias(pos, pos).masked_fill(mask, float("-inf"))
            mask = bias.repeat(B, 1, 1)                           # [B*heads, T, T], batch-major like MHA
        else:
            x = x + sinusoid(T, x.shape[2], x.device)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(x)


def load_student(path, device="cpu") -> StreamingLID:
    ck = torch.load(path, map_location="cpu")
    m = StreamingLID(ck["n_langs"], **ck.get("arch", {}))
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


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
