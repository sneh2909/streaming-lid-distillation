# Generated data

No audio is committed. From the repository root, run:

```bash
uv run python scripts/prepare_data.py
```

This creates `data/generated/manifest.jsonl` and 16 kHz PCM WAV files using gTTS, Microsoft Edge TTS, and the Python `miniaudio` decoder (no ffmpeg). It includes 10 monolingual training clips and 3 held-out clips for each of seven languages, a Hindi→English training concatenation, and held-out switch clips in both directions.

Training uses two synthetic voice IDs per language (gTTS plus a named male Edge voice). Held-out text and both evaluation switches use a named female Edge voice that never occurs in training. The preparation script records voice IDs and refuses to write a manifest with train/evaluation overlap. Voice-qualified filenames prevent stale audio reuse after a split change, transient downloads are retried, and `--force` refreshes the speech.

These are disjoint synthetic voice identities, not human speaker IDs. The split contains only one held-out voice per language and is a plumbing/generalisation check, not a population estimate.

The source texts, split construction, exact 4 s + 4 s concatenation, and known boundary metadata live in `scripts/prepare_data.py`. The entire generated directory is ignored by Git.
