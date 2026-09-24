"""Bounded-lookahead causal TCN student and chunked inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import (
    CHUNK_FRAMES,
    MODEL_LOOKAHEAD_FRAMES,
    N_MELS,
    STUDENT_DILATIONS,
    STUDENT_HIDDEN_SIZE,
)


class CausalDepthwiseBlock(torch.nn.Module):
    """Residual temporal block with padding on the past side only."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.depthwise = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = torch.nn.Conv1d(channels, channels, kernel_size=1)
        self.normalization = torch.nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        temporal = inputs.transpose(1, 2)
        temporal = F.pad(temporal, (self.left_padding, 0))
        temporal = self.pointwise(self.depthwise(temporal)).transpose(1, 2)
        return inputs + F.silu(self.normalization(temporal))


class CausalLIDStudent(torch.nn.Module):
    """Framewise LID with explicit right context and finite causal history."""

    def __init__(
        self,
        num_languages: int,
        n_mels: int = N_MELS,
        hidden_size: int = STUDENT_HIDDEN_SIZE,
        lookahead_frames: int = MODEL_LOOKAHEAD_FRAMES,
        dilations: tuple[int, ...] = STUDENT_DILATIONS,
    ) -> None:
        super().__init__()
        self.num_languages = num_languages
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.lookahead_frames = lookahead_frames
        self.dilations = tuple(dilations)
        input_size = n_mels * (lookahead_frames + 1)
        self.input_projection = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.SiLU(),
        )
        self.temporal_blocks = torch.nn.ModuleList(
            CausalDepthwiseBlock(hidden_size, dilation) for dilation in self.dilations
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size // 2, num_languages),
        )

    def _stack_right_context(
        self, features: torch.Tensor, output_frames: int | None = None
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"expected [batch, frames, mel], got {tuple(features.shape)}"
            )
        if features.shape[-1] != self.n_mels:
            raise ValueError(f"expected {self.n_mels} mel bins")
        if output_frames is None:
            output_frames = features.shape[1]
        required = output_frames + self.lookahead_frames
        if features.shape[1] < required:
            features = F.pad(features, (0, 0, 0, required - features.shape[1]))
        return torch.cat(
            [
                features[:, offset : offset + output_frames]
                for offset in range(self.lookahead_frames + 1)
            ],
            dim=-1,
        )

    def _forward_contexts(self, contexts: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(contexts)
        for block in self.temporal_blocks:
            encoded = block(encoded)
        return self.classifier(encoded)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        contexts = self._stack_right_context(features)
        return self._forward_contexts(contexts)

    def streaming_forward(
        self, features: torch.Tensor, chunk_frames: int = CHUNK_FRAMES
    ) -> torch.Tensor:
        """Chunked inference exactly equivalent to `forward`.

        Each chunk carries finite left history and exactly `lookahead_frames` of
        right context. Recomputing the small left overlap keeps the implementation
        obvious; production would cache each convolution's state.
        """
        if features.ndim != 3:
            raise ValueError("features must be [batch, frames, mel]")
        outputs: list[torch.Tensor] = []
        total_frames = features.shape[1]
        past_frames = self.receptive_field_frames - 1
        for start in range(0, total_frames, chunk_frames):
            n_emit = min(chunk_frames, total_frames - start)
            history_start = max(0, start - past_frames)
            stop = min(total_frames, start + n_emit + self.lookahead_frames)
            raw_chunk = features[:, history_start:stop]
            local_logits = self.forward(raw_chunk)
            local_start = start - history_start
            outputs.append(local_logits[:, local_start : local_start + n_emit])
        return torch.cat(outputs, dim=1)

    @property
    def receptive_field_frames(self) -> int:
        return 1 + 2 * sum(self.dilations)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
