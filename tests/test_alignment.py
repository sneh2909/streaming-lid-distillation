import pytest
import torch

from streaming_lid.loss import delayed_distillation_loss


def make_logits(
    target_classes: list[int], total_frames: int, delay: int
) -> torch.Tensor:
    logits = torch.full((1, total_frames, 3), -8.0)
    for target_frame, class_index in enumerate(target_classes):
        prediction_frame = target_frame + delay
        if prediction_frame < total_frames:
            logits[0, prediction_frame, class_index] = 8.0
    return logits


def test_delay_aligns_teacher_i_with_student_i_plus_delay() -> None:
    total_frames = 9
    delay = 2
    lookahead = 1
    classes = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    teacher = torch.nn.functional.one_hot(
        torch.tensor([classes]), num_classes=3
    ).float()
    correctly_delayed = make_logits(classes, total_frames, delay)
    incorrectly_undelayed = make_logits(classes, total_frames, 0)
    lengths = torch.tensor([total_frames])

    correct_loss, stats = delayed_distillation_loss(
        correctly_delayed,
        teacher,
        lengths,
        delay_frames=delay,
        lookahead_frames=lookahead,
        temperature=1.0,
        early_ramp_frames=1,
    )
    wrong_loss, _ = delayed_distillation_loss(
        incorrectly_undelayed,
        teacher,
        lengths,
        delay_frames=delay,
        lookahead_frames=lookahead,
        temperature=1.0,
        early_ramp_frames=1,
    )

    assert stats["valid_frames"] == total_frames - delay - lookahead
    assert correct_loss < 1e-5
    assert wrong_loss > 1.0


def test_unavailable_tail_targets_do_not_affect_loss() -> None:
    torch.manual_seed(2)
    logits = torch.randn(1, 8, 2)
    teacher = torch.softmax(torch.randn(1, 8, 2), dim=-1)
    changed_tail = teacher.clone()
    # With delay=2/lookahead=2/length=8, only teacher frames 0..3 are valid.
    changed_tail[:, 4:] = changed_tail[:, 4:].flip(-1)
    kwargs = {
        "lengths": torch.tensor([8]),
        "delay_frames": 2,
        "lookahead_frames": 2,
        "temperature": 2.0,
        "early_ramp_frames": 3,
    }
    original_loss, _ = delayed_distillation_loss(logits, teacher, **kwargs)
    changed_loss, _ = delayed_distillation_loss(logits, changed_tail, **kwargs)
    torch.testing.assert_close(original_loss, changed_loss)


def test_all_invalid_batch_is_rejected_at_delay_plus_lookahead_boundary() -> None:
    delay = 21
    lookahead = 4
    boundary_frames = delay + lookahead
    logits = torch.zeros(1, boundary_frames, 2, requires_grad=True)
    teacher = torch.tensor([0.8, 0.2]).expand(1, boundary_frames, 2).clone()

    with pytest.raises(ValueError, match="no valid aligned frames"):
        delayed_distillation_loss(
            logits,
            teacher,
            torch.tensor([boundary_frames]),
            delay_frames=delay,
            lookahead_frames=lookahead,
            temperature=2.0,
            early_ramp_frames=100,
        )


def test_first_valid_frame_after_boundary_contributes_a_gradient() -> None:
    delay = 21
    lookahead = 4
    total_frames = delay + lookahead + 1
    logits = torch.zeros(1, total_frames, 2, requires_grad=True)
    teacher = torch.tensor([0.8, 0.2]).expand(1, total_frames, 2).clone()

    loss, stats = delayed_distillation_loss(
        logits,
        teacher,
        torch.tensor([total_frames]),
        delay_frames=delay,
        lookahead_frames=lookahead,
        temperature=2.0,
        early_ramp_frames=100,
    )
    loss.backward()

    assert stats["valid_frames"] == 1.0
    assert stats["weight_sum"] == pytest.approx(0.01)
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() > 0
