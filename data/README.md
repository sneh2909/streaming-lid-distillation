# Generated data

No audio is committed. From the repository root, run:

```bash
uv run python scripts/prepare_data.py
```

This creates `data/generated/manifest.jsonl` and 16 kHz PCM WAV files using gTTS plus the Python `miniaudio` decoder (no ffmpeg). It includes seven languages, disjoint text for train/held-out clips, a Hindi→English training concatenation, and held-out switch clips in both directions. `--force` downloads the speech again.

The source texts, split construction, exact 4 s + 4 s concatenation, and known boundary metadata live in `scripts/prepare_data.py`. The entire generated directory is ignored by Git.
