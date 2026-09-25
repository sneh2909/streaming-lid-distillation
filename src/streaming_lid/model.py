"""Bounded-lookahead causal TCN student and chunked inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import (
    CHUNK_FRAMES,
    MODEL_LOOKAHEAD_FRAMES,
    N_MELS,
    STUDENT_DILATIONS,
    STUDENT_HIDDEN_SIZE,
)


@dataclass
class StreamingLIDState:
    """Finite feature state for stable growing-prefix inference."""

    feature_buffer: torch.Tensor | None = None
    buffer_start_frame: int = 0
    next_output_frame: int = 0


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

    def new_streaming_state(self) -> StreamingLIDState:
        """Return empty state for one growing feature stream."""
        return StreamingLIDState()

    def streaming_step(
        self, features: torch.Tensor, state: StreamingLIDState
    ) -> torch.Tensor:
        """Consume new features and return each newly stable logit exactly once.

        The final ``lookahead_frames`` positions remain buffered because their
        required future features have not arrived. Once they become stable on a
        later call, they are emitted without repeating earlier outputs. Only the
        finite receptive-field history is retained.
        """
        if features.ndim != 3:
            raise ValueError("features must be [batch, frames, mel]")
        if features.shape[-1] != self.n_mels:
            raise ValueError(f"expected {self.n_mels} mel bins")
        if state.feature_buffer is None:
            if state.buffer_start_frame != 0 or state.next_output_frame != 0:
                raise ValueError("empty streaming state has non-zero frame offsets")
            buffered = features
        else:
            previous = state.feature_buffer
            if previous.shape[0] != features.shape[0]:
                raise ValueError("streaming batch size changed")
            if previous.shape[-1] != features.shape[-1]:
                raise ValueError("streaming feature size changed")
            if previous.device != features.device or previous.dtype != features.dtype:
                raise ValueError("streaming feature device or dtype changed")
            buffered = torch.cat((previous, features), dim=1)

        received_stop = state.buffer_start_frame + buffered.shape[1]
        stable_stop = max(0, received_stop - self.lookahead_frames)
        if stable_stop < state.next_output_frame:
            raise ValueError("streaming state moved backwards")

        local_start = state.next_output_frame - state.buffer_start_frame
        local_stop = stable_stop - state.buffer_start_frame
        if local_stop > local_start:
            buffered_logits = self.forward(buffered)
            emitted = buffered_logits[:, local_start:local_stop]
        else:
            emitted = features.new_empty(
                (features.shape[0], 0, self.num_languages)
            )

        state.next_output_frame = stable_stop
        retain_start = max(
            0, state.next_output_frame - (self.receptive_field_frames - 1)
        )
        prune_frames = retain_start - state.buffer_start_frame
        state.feature_buffer = buffered[:, prune_frames:]
        state.buffer_start_frame = retain_start
        return emitted

    def streaming_forward(
        self, features: torch.Tensor, chunk_frames: int = CHUNK_FRAMES
    ) -> torch.Tensor:
        """Replay a complete feature prefix and return only stable logits.

        The result equals ``forward(features)`` except that its provisional final
        ``lookahead_frames`` outputs are withheld. ``streaming_step`` is the live
        API: this helper feeds it successive chunks for evaluation and benchmarking.
        Recomputing the finite left overlap keeps the implementation obvious;
        production kernels would cache each convolution's hidden state.
        """
        if features.ndim != 3:
            raise ValueError("features must be [batch, frames, mel]")
        if features.shape[-1] != self.n_mels:
            raise ValueError(f"expected {self.n_mels} mel bins")
        state = self.new_streaming_state()
        outputs: list[torch.Tensor] = []
        total_frames = features.shape[1]
        if total_frames == 0:
            return features.new_empty((features.shape[0], 0, self.num_languages))
        for start in range(0, total_frames, chunk_frames):
            outputs.append(
                self.streaming_step(features[:, start : start + chunk_frames], state)
            )
        return torch.cat(outputs, dim=1)

    @property
    def receptive_field_frames(self) -> int:
        return 1 + 2 * sum(self.dilations)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
