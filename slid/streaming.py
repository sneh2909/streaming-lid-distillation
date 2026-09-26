"""True incremental inference: push audio in any piece size, get one posterior per 80 ms frame.

Per call the session keeps only
  * raw audio from 2 encoder frames before the next chunk (the conv front end's receptive field),
  * per layer: the attention keys/values of the last `left_frames` frames (KV cache),
  * per layer: the last kernel-1 inputs of the causal depthwise conv.
So memory and compute per chunk are constant no matter how long the call runs.
Outputs are identical to StreamingLID.forward(whole_wav, chunk) (tests/test_streaming.py).
"""
import numpy as np
import torch
import torch.nn.functional as F

from slid.student import HOP, SUBSAMPLE, StreamingLID, frame_end_sample, sinusoid

CTX_FRAMES = 2       # front end: output frame t needs mel frames [8t - 14, 8t] -> start 2 frames early


class StreamingSession:
    def __init__(self, model: StreamingLID, chunk: int):
        # v2 (rel_pos, bounded history): constant memory.  v1 (absolute positions, unbounded history):
        # the KV cache grows with the call (~7 kB per 80 ms frame for all layers) - fine for phone calls.
        self.m, self.C = model.eval(), chunk
        dev = next(model.parameters()).device
        self.dev = dev
        self.audio = np.zeros(0, dtype=np.float32)
        self.offset = 0                       # absolute sample index of self.audio[0]
        self.t = 0                            # next encoder frame to emit
        d = model.head.in_features
        k = model.blocks[0].dw.kernel_size[0]
        self.k_cache = [None] * len(model.blocks)
        self.v_cache = [None] * len(model.blocks)
        self.conv = [torch.zeros(1, d, k - 1, device=dev) for _ in model.blocks]

    def cache_frames(self) -> int:
        return 0 if self.k_cache[0] is None else self.k_cache[0].shape[2]

    @torch.no_grad()
    def push(self, samples: np.ndarray) -> np.ndarray:
        """Append audio; return posteriors [n_new_frames, n_langs] for every newly completed chunk."""
        self.audio = np.concatenate([self.audio, samples.astype(np.float32)])
        out = []
        while self.offset + len(self.audio) >= frame_end_sample(self.t + self.C - 1):
            x = self._frontend(self.t, self.t + self.C)
            for i, b in enumerate(self.m.blocks):
                x = self._block(i, b, x)
            out.append(self.m.head(x)[0].softmax(-1).cpu().numpy())
            self.t += self.C
            keep_from = max(0, SUBSAMPLE * (self.t - CTX_FRAMES)) * HOP
            cut = keep_from - self.offset
            if cut > 0:
                self.audio, self.offset = self.audio[cut:], keep_from
        return np.concatenate(out) if out else np.zeros((0, self.m.head.out_features), dtype=np.float32)

    def _frontend(self, t0: int, t1: int) -> torch.Tensor:
        m0 = max(0, SUBSAMPLE * (t0 - CTX_FRAMES))                    # first mel frame, multiple of 8
        m_last = SUBSAMPLE * (t1 - 1)
        s0, s1 = m0 * HOP - self.offset, frame_end_sample(t1 - 1) - self.offset
        wav = torch.from_numpy(self.audio[s0:s1]).to(self.dev)[None]
        feats = self.m.feat(wav)
        assert feats.shape[1] == m_last - m0 + 1
        y = self.m.sub(feats)                                          # output k <-> frame m0/8 + k
        first = t0 - m0 // SUBSAMPLE
        y = y[:, first: first + (t1 - t0)]
        if not self.m.rel_pos:
            y = y + sinusoid(t1, y.shape[2], y.device)[t0:t1]
        return y

    def _block(self, i: int, b, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * b.ff1(x)
        h = b.ln_att(x)
        H, d = b.att.num_heads, x.shape[-1]
        q, k, v = F.linear(h, b.att.in_proj_weight, b.att.in_proj_bias).chunk(3, dim=-1)
        split = lambda z: z.view(1, -1, H, d // H).transpose(1, 2)    # [1, H, c, dh]
        q, k, v = split(q), split(k), split(v)
        if self.k_cache[i] is not None:
            k = torch.cat([self.k_cache[i], k], dim=2)
            v = torch.cat([self.v_cache[i], v], dim=2)
        c, n_k = q.shape[2], k.shape[2]
        q_pos = torch.arange(self.t, self.t + c, device=x.device)
        k_pos = torch.arange(self.t + c - n_k, self.t + c, device=x.device)
        scores = q @ k.transpose(-1, -2) / (d // H) ** 0.5
        if self.m.rel_pos:
            scores = scores + self.m.attn_bias(q_pos, k_pos)[None]
        a = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, c, d)
        x = x + b.att.out_proj(a)
        L = self.m.left_frames
        self.k_cache[i], self.v_cache[i] = (k[:, :, -L:], v[:, :, -L:]) if L > 0 else (k, v)
        h = F.glu(b.pw1(b.ln_conv(x)), dim=-1).transpose(1, 2)        # [1, d, c]
        hin = torch.cat([self.conv[i], h], dim=2)
        y = F.conv1d(hin, b.dw.weight, b.dw.bias, groups=b.dw.groups)
        self.conv[i] = hin[:, :, -(b.dw.kernel_size[0] - 1):]
        h = F.silu(b.norm_conv(y.transpose(1, 2)))
        x = x + b.pw2(h)
        x = x + 0.5 * b.ff2(x)
        return b.ln_out(x)
