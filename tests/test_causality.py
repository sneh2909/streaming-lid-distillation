import torch

from streaming_lid.audio import LogMelFrontend
from streaming_lid.config import HOP_LENGTH, N_FFT
from streaming_lid.model import CausalLIDStudent


def test_output_does_not_change_beyond_bounded_lookahead() -> None:
    torch.manual_seed(11)
    lookahead = 4
    model = CausalLIDStudent(
        num_languages=3,
        n_mels=8,
        hidden_size=16,
        lookahead_frames=lookahead,
        dilations=(1, 2, 4),
    ).eval()
    original = torch.randn(1, 30, 8)
    changed_future = original.clone()
    output_frame = 12
    changed_future[:, output_frame + lookahead + 1 :] = torch.randn_like(
        changed_future[:, output_frame + lookahead + 1 :]
    )

    with torch.inference_mode():
        original_output = model(original)
        changed_output = model(changed_future)
    torch.testing.assert_close(
        original_output[:, : output_frame + 1],
        changed_output[:, : output_frame + 1],
        rtol=0,
        atol=0,
    )


def test_chunked_and_whole_sequence_inference_match() -> None:
    torch.manual_seed(13)
    model = CausalLIDStudent(
        num_languages=4,
        n_mels=8,
        hidden_size=16,
        lookahead_frames=3,
        dilations=(1, 2, 4),
    ).eval()
    features = torch.randn(2, 37, 8)
    with torch.inference_mode():
        whole = model(features)
        chunked = model.streaming_forward(features, chunk_frames=7)
    assert chunked.shape[1] == features.shape[1] - model.lookahead_frames
    torch.testing.assert_close(
        whole[:, : -model.lookahead_frames], chunked, rtol=1e-5, atol=1e-6
    )


def test_growing_prefix_withholds_tail_and_emits_each_stable_logit_once() -> None:
    torch.manual_seed(19)
    lookahead = 4
    model = CausalLIDStudent(
        num_languages=3,
        n_mels=8,
        hidden_size=16,
        lookahead_frames=lookahead,
        dilations=(1, 2, 4),
    ).eval()
    features = torch.randn(1, 50, 8)
    state = model.new_streaming_state()

    with torch.inference_mode():
        first = model.streaming_step(features[:, :24], state)
        second = model.streaming_step(features[:, 24:], state)
        stable_whole = model(features)[:, :-lookahead]

    # Frames 20..23 are provisional after the first prefix and appear only
    # after real right context arrives with the continuation.
    assert first.shape[1] == 24 - lookahead
    assert second.shape[1] == 50 - lookahead - first.shape[1]
    assert state.next_output_frame == 50 - lookahead
    assert state.feature_buffer is not None
    assert state.feature_buffer.shape[1] == model.receptive_field_frames - 1 + lookahead
    torch.testing.assert_close(
        torch.cat((first, second), dim=1), stable_whole, rtol=1e-5, atol=1e-6
    )


def test_waveform_future_cannot_leak_through_frontend() -> None:
    torch.manual_seed(17)
    lookahead = 4
    output_frame = 10
    model = CausalLIDStudent(
        num_languages=3,
        lookahead_frames=lookahead,
        dilations=(1, 2),
    ).eval()
    frontend = LogMelFrontend().eval()
    original = torch.randn(1, 6_000)
    changed_future = original.clone()
    # Feature output_frame+lookahead consumes samples strictly before cutoff.
    cutoff = (output_frame + lookahead) * HOP_LENGTH + N_FFT
    changed_future[:, cutoff:] = torch.randn_like(changed_future[:, cutoff:])

    with torch.inference_mode():
        original_output = model(frontend(original))
        changed_output = model(frontend(changed_future))
    torch.testing.assert_close(
        original_output[:, : output_frame + 1],
        changed_output[:, : output_frame + 1],
        rtol=0,
        atol=0,
    )
