"""Bounded-lookahead causal TCN student and chunked inference."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
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


@dataclass(frozen=True)
class StreamingLIDEmission:
    """One actual streaming call's stable outputs and scheduler coordinates."""

    logits: torch.Tensor
    output_start_frame: int
    output_stop_frame: int
    received_stop_frame: int
    emitted_monotonic_seconds: float | None = None

    @property
    def emitted_frames(self) -> int:
        return self.output_stop_frame - self.output_start_frame

    @property
    def latest_received_feature_frame(self) -> int | None:
        if self.received_stop_frame == 0:
            return None
        return self.received_stop_frame - 1


@dataclass(frozen=True)
class StreamingPolicyTrace:
    """Stable logits and policy groups from the calls that emitted them."""

    stable_logits: torch.Tensor
    stable_probabilities: np.ndarray
    chunk_posteriors: np.ndarray
    chunk_audio_available_seconds: np.ndarray
    emission_records: tuple[dict[str, object], ...]


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
        self,
        features: torch.Tensor,
        state: StreamingLIDState,
        *,
        clock: Callable[[], float] | None = None,
    ) -> StreamingLIDEmission:
        """Consume new features and record each newly stable logit exactly once.

        The final ``lookahead_frames`` positions remain buffered because their
        required future features have not arrived. Once they become stable on a
        later call, they are emitted without repeating earlier outputs. Only the
        finite receptive-field history is retained. Frame ranges are absolute
        within the stream. When supplied, ``clock`` is sampled after inference
        so callers can bind the group to its real monotonic emission time.
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

        output_start_frame = state.next_output_frame
        local_start = output_start_frame - state.buffer_start_frame
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
        emitted_monotonic_seconds = None if clock is None else float(clock())
        return StreamingLIDEmission(
            logits=emitted,
            output_start_frame=output_start_frame,
            output_stop_frame=stable_stop,
            received_stop_frame=received_stop,
            emitted_monotonic_seconds=emitted_monotonic_seconds,
        )

    def streaming_emissions(
        self,
        features: torch.Tensor,
        chunk_frames: int = CHUNK_FRAMES,
        *,
        clock: Callable[[], float] | None = None,
    ) -> list[StreamingLIDEmission]:
        """Replay a prefix and retain the exact output group from every call."""
        if features.ndim != 3:
            raise ValueError("features must be [batch, frames, mel]")
        if features.shape[-1] != self.n_mels:
            raise ValueError(f"expected {self.n_mels} mel bins")
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        state = self.new_streaming_state()
        emissions: list[StreamingLIDEmission] = []
        for start in range(0, features.shape[1], chunk_frames):
            emissions.append(
                self.streaming_step(
                    features[:, start : start + chunk_frames],
                    state,
                    clock=clock,
                )
            )
        return emissions

    def streaming_policy_trace(
        self,
        features: torch.Tensor,
        *,
        chunk_frames: int,
        min_output_frame: int,
        hop_length: int,
        win_length: int,
        sample_rate: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> StreamingPolicyTrace:
        """Run the real scheduler and retain each call's group and clocks.

        ``audio_available_seconds`` is derived from the exclusive end sample of
        the latest feature received by that call. The monotonic timestamp is
        sampled after its logits are computed. This remains an offline feature
        replay; it does not claim to time an incremental frontend.
        """
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError("streaming policy trace expects [1, frames, mel]")
        if min_output_frame < 0:
            raise ValueError("min_output_frame must be non-negative")
        if hop_length <= 0 or win_length <= 0 or sample_rate <= 0:
            raise ValueError("frontend timing values must be positive")

        emissions = self.streaming_emissions(
            features,
            chunk_frames=chunk_frames,
            clock=clock,
        )
        if not emissions:
            raise ValueError("streaming policy trace requires at least one feature")

        expected_output_start = 0
        previous_received_stop = 0
        previous_monotonic_seconds: float | None = None
        monotonic_origin_seconds: float | None = None
        stable_logits: list[torch.Tensor] = []
        chunk_posteriors: list[np.ndarray] = []
        chunk_audio_available_seconds: list[float] = []
        records: list[dict[str, object]] = []
        policy_call_index = 0

        for call_index, emission in enumerate(emissions):
            if emission.output_start_frame != expected_output_start:
                raise AssertionError(
                    "streaming emissions are duplicated or discontinuous"
                )
            if emission.emitted_frames != emission.logits.shape[1]:
                raise AssertionError(
                    "streaming emission range disagrees with its logits"
                )
            if emission.received_stop_frame <= previous_received_stop:
                raise AssertionError("streaming input clock did not advance")
            if emission.latest_received_feature_frame is None:
                raise AssertionError("non-empty streaming call has no received feature")
            emitted_monotonic_seconds = emission.emitted_monotonic_seconds
            if emitted_monotonic_seconds is None or not math.isfinite(
                emitted_monotonic_seconds
            ):
                raise ValueError("streaming emission clock must be finite")
            if (
                previous_monotonic_seconds is not None
                and emitted_monotonic_seconds < previous_monotonic_seconds
            ):
                raise ValueError("streaming emission clock moved backwards")
            if monotonic_origin_seconds is None:
                monotonic_origin_seconds = emitted_monotonic_seconds

            latest_audio_sample_exclusive = (
                emission.latest_received_feature_frame * hop_length + win_length
            )
            audio_available_seconds = latest_audio_sample_exclusive / sample_rate
            aligned_start = max(emission.output_start_frame, min_output_frame)
            aligned_frames = max(0, emission.output_stop_frame - aligned_start)
            selected_policy_call_index: int | None = None
            if aligned_frames:
                local_start = aligned_start - emission.output_start_frame
                selected_logits = emission.logits[0, local_start:]
                chunk_posteriors.append(
                    torch.softmax(selected_logits, dim=-1)
                    .mean(dim=0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                chunk_audio_available_seconds.append(audio_available_seconds)
                selected_policy_call_index = policy_call_index
                policy_call_index += 1

            stable_logits.append(emission.logits)
            records.append(
                {
                    "call_index": call_index,
                    "output_start_frame": emission.output_start_frame,
                    "output_stop_frame_exclusive": emission.output_stop_frame,
                    "emitted_frames": emission.emitted_frames,
                    "aligned_output_start_frame": (
                        aligned_start if aligned_frames else None
                    ),
                    "aligned_output_stop_frame_exclusive": (
                        emission.output_stop_frame if aligned_frames else None
                    ),
                    "aligned_frames": aligned_frames,
                    "policy_call_index": selected_policy_call_index,
                    "latest_received_feature_frame": (
                        emission.latest_received_feature_frame
                    ),
                    "latest_audio_sample_exclusive": latest_audio_sample_exclusive,
                    "audio_available_seconds": audio_available_seconds,
                    "emitted_monotonic_offset_seconds": (
                        emitted_monotonic_seconds - monotonic_origin_seconds
                    ),
                }
            )
            expected_output_start = emission.output_stop_frame
            previous_received_stop = emission.received_stop_frame
            previous_monotonic_seconds = emitted_monotonic_seconds

        expected_stable_frames = max(0, features.shape[1] - self.lookahead_frames)
        if expected_output_start != expected_stable_frames:
            raise AssertionError("streaming emission schedule did not close exactly")
        if not chunk_posteriors:
            raise ValueError("streaming trace has no delay-aligned policy group")
        concatenated_logits = torch.cat(stable_logits, dim=1).squeeze(0)
        probabilities = (
            torch.softmax(concatenated_logits, dim=-1).detach().cpu().numpy()
        )
        return StreamingPolicyTrace(
            stable_logits=concatenated_logits,
            stable_probabilities=probabilities,
            chunk_posteriors=np.stack(chunk_posteriors),
            chunk_audio_available_seconds=np.asarray(
                chunk_audio_available_seconds, dtype=np.float64
            ),
            emission_records=tuple(records),
        )

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
        emissions = self.streaming_emissions(features, chunk_frames=chunk_frames)
        if not emissions:
            return features.new_empty((features.shape[0], 0, self.num_languages))
        return torch.cat([emission.logits for emission in emissions], dim=1)

    @property
    def receptive_field_frames(self) -> int:
        return 1 + 2 * sum(self.dilations)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
