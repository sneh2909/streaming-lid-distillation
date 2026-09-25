import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import scripts.prepare_data as prepare_data
from streaming_lid.data import canonical_json_sha256, file_sha256


def _write_tone(path: Path, frequency: float = 220.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(1600, dtype=np.float32) / prepare_data.SAMPLE_RATE
    samples = 0.2 * np.sin(2 * np.pi * frequency * time)
    sf.write(
        path,
        samples,
        prepare_data.SAMPLE_RATE,
        subtype=prepare_data.WAV_SUBTYPE,
    )


def _stage_record(root: Path, clip_id: str = "clip") -> dict:
    pending = root / "audio" / f"{clip_id}--pending.wav"
    _write_tone(pending)
    final_path, duration_seconds, audio_sha256 = prepare_data.finalize_staged_wav(
        pending,
        clip_id=clip_id,
    )
    recipe = {"kind": "test", "version": 1, "text": "hello"}
    return {
        "corpus_schema_version": prepare_data.CORPUS_SCHEMA_VERSION,
        "id": clip_id,
        "audio_path": f"audio/{final_path.name}",
        "audio_sha256": audio_sha256,
        "audio_recipe_sha256": canonical_json_sha256(recipe),
        "audio_recipe": recipe,
        "split": "train",
        "language": "en",
        "duration_seconds": round(duration_seconds, 4),
        "speaker_id": "voice",
    }


def test_tts_recipe_binds_text_voice_options_and_generator() -> None:
    generator = {"recipe_version": 1, "source": {"sha256": "source-a"}}
    baseline = prepare_data.tts_audio_recipe(
        text="hello",
        language="en",
        voice="en-IN-PrabhatNeural",
        generator_identity=generator,
    )
    changed_text = prepare_data.tts_audio_recipe(
        text="different",
        language="en",
        voice="en-IN-PrabhatNeural",
        generator_identity=generator,
    )
    changed_voice = prepare_data.tts_audio_recipe(
        text="hello",
        language="en",
        voice="en-IN-NeerjaNeural",
        generator_identity=generator,
    )

    assert canonical_json_sha256(baseline) != canonical_json_sha256(changed_text)
    assert canonical_json_sha256(baseline) != canonical_json_sha256(changed_voice)
    assert baseline["provider_options"] == prepare_data.EDGE_SYNTHESIS_OPTIONS
    assert baseline["audio_processing"]["sample_rate"] == 16_000


def test_cache_reuse_requires_exact_recipe_and_audio_bytes(tmp_path: Path) -> None:
    output_dir = tmp_path / "published"
    record = _stage_record(output_dir)
    recipe = record["audio_recipe"]

    assert prepare_data.reusable_audio_path(
        record,
        expected_recipe=recipe,
        output_dir=output_dir,
    ) == (output_dir / record["audio_path"]).resolve()

    changed_recipe = {**recipe, "text": "different"}
    assert (
        prepare_data.reusable_audio_path(
            record,
            expected_recipe=changed_recipe,
            output_dir=output_dir,
        )
        is None
    )

    with (output_dir / record["audio_path"]).open("ab") as handle:
        handle.write(b"tamper")
    assert (
        prepare_data.reusable_audio_path(
            record,
            expected_recipe=recipe,
            output_dir=output_dir,
        )
        is None
    )


def test_staged_validation_rejects_declared_checksum_mismatch(
    tmp_path: Path,
) -> None:
    record = _stage_record(tmp_path)
    record["audio_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="audio checksum"):
        prepare_data.validate_staged_corpus(
            [record],
            tmp_path,
            expected_split_counts={"train": 1},
        )


def test_failed_manifest_swap_keeps_previous_release_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "published"
    output_audio = output_dir / "audio" / "old.wav"
    _write_tone(output_audio, frequency=110.0)
    old_audio_sha256 = file_sha256(output_audio)
    old_manifest = output_dir / "manifest.jsonl"
    old_manifest.write_text('{"id":"old","audio_path":"audio/old.wav"}\n')

    staging_dir = tmp_path / "staging"
    record = _stage_record(staging_dir, clip_id="new")
    real_replace = os.replace

    def fail_manifest_swap(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == old_manifest:
            raise OSError("injected manifest publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(prepare_data.os, "replace", fail_manifest_swap)
    with pytest.raises(OSError, match="injected manifest"):
        prepare_data.publish_staged_corpus(
            staging_dir,
            output_dir,
            [record],
            expected_split_counts={"train": 1},
        )

    assert old_manifest.read_text() == '{"id":"old","audio_path":"audio/old.wav"}\n'
    assert file_sha256(output_audio) == old_audio_sha256
    assert file_sha256(output_dir / record["audio_path"]) == record["audio_sha256"]
