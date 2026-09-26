import numpy as np

from streaming_lid.config import HOP_LENGTH, WIN_LENGTH
from streaming_lid.data import (
    expand_local_posteriors,
    local_target_availability_ledger,
)


def test_previous_hold_is_valid_while_linear_uses_unavailable_next_anchors() -> None:
    anchors = np.asarray([0, 25, 50], dtype=np.int64)

    held = local_target_availability_ledger(
        anchors, 51, expansion="previous_anchor_hold"
    )
    linear = local_target_availability_ledger(
        anchors, 51, expansion="linear_probability"
    )
    is_anchor = np.isin(linear["semantic_frames"], anchors)

    assert held["availability_valid"].all()
    assert linear["availability_valid"][is_anchor].all()
    assert not linear["availability_valid"][~is_anchor].any()
    np.testing.assert_array_equal(
        held["source_anchor_frames"][:26],
        np.asarray([0] * 25 + [25]),
    )


def test_linear_expansion_needs_450_ms_label_delay_with_four_frame_lookahead() -> None:
    anchors = np.asarray([0, 25, 50], dtype=np.int64)

    one_frame_short = local_target_availability_ledger(
        anchors,
        51,
        expansion="linear_probability",
        label_delay_frames=44,
        model_lookahead_frames=4,
    )
    paid_latency = local_target_availability_ledger(
        anchors,
        51,
        expansion="linear_probability",
        label_delay_frames=45,
        model_lookahead_frames=4,
    )

    assert not one_frame_short["availability_valid"].all()
    assert paid_latency["availability_valid"].all()


def test_irregular_final_anchor_cannot_be_hidden_by_end_padding() -> None:
    anchors = np.asarray([0, 25, 47], dtype=np.int64)
    ledger = local_target_availability_ledger(
        anchors, 48, expansion="linear_probability"
    )
    minimum_waveform_samples = (48 - 1) * HOP_LENGTH + WIN_LENGTH
    frame = 46

    assert ledger["right_anchor_frames"][frame] == 47
    assert ledger["teacher_latest_samples"][frame] > minimum_waveform_samples
    assert ledger["student_latest_samples"][frame] > minimum_waveform_samples
    assert (
        min(ledger["teacher_latest_samples"][frame], minimum_waveform_samples)
        == min(ledger["student_latest_samples"][frame], minimum_waveform_samples)
    )
    assert not ledger["availability_valid"][frame]


def test_previous_hold_selects_the_latest_past_anchor_exactly() -> None:
    anchors = np.asarray([0, 3, 7], dtype=np.int64)
    values = np.asarray(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )

    expanded = expand_local_posteriors(
        anchors, values, 8, expansion="previous_anchor_hold"
    )

    np.testing.assert_allclose(expanded[:3], np.repeat(values[0:1], 3, axis=0))
    np.testing.assert_allclose(expanded[3:7], np.repeat(values[1:2], 4, axis=0))
    np.testing.assert_allclose(expanded[7], values[2])
