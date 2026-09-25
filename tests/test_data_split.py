import pytest

from streaming_lid.data import require_speaker_disjoint, speaker_split_audit


def test_speaker_audit_includes_switch_evaluation_speakers() -> None:
    records = [
        {"id": "train", "split": "train", "speaker_id": "train-voice"},
        {"id": "heldout", "split": "heldout", "speaker_id": "eval-voice"},
        {
            "id": "switch",
            "split": "switch",
            "speaker_ids": ["eval-voice", "other-eval-voice"],
        },
    ]

    audit = require_speaker_disjoint(records)

    assert audit["speaker_disjoint"] is True
    assert audit["train_speaker_ids"] == ["train-voice"]
    assert audit["switch_eval_speaker_ids"] == [
        "eval-voice",
        "other-eval-voice",
    ]


def test_speaker_audit_rejects_train_evaluation_overlap() -> None:
    records = [
        {"id": "train", "split": "train", "speaker_id": "leaked-voice"},
        {"id": "heldout", "split": "heldout", "speaker_id": "leaked-voice"},
    ]

    audit = speaker_split_audit(records)
    assert audit["speaker_disjoint"] is False
    with pytest.raises(ValueError, match="leaked-voice"):
        require_speaker_disjoint(records)
