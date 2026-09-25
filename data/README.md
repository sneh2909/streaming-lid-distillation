# Generated data

No audio is committed. From the repository root, run:

```bash
uv run python scripts/prepare_data.py
```

This creates `data/generated/manifest.jsonl` and 16 kHz PCM WAV files using gTTS, Microsoft Edge TTS, and the Python `miniaudio` decoder (no ffmpeg). It includes 10 monolingual training clips and 3 held-out clips for each of seven languages, a Hindi→English training concatenation, and held-out switch clips in both directions.

Training uses two synthetic voice IDs per language (gTTS plus a named male Edge voice). Held-out text and both evaluation switches use a named female Edge voice that never occurs in training. The preparation script records voice IDs and refuses to publish a manifest with train/evaluation overlap.

Each row also stores a canonical audio recipe and its hash: text, language, provider, exact voice, synthesis options, client/runtime versions, decode/trim/peak settings, and the preparation-source hash. WAV filenames contain the full SHA-256 of their bytes. A normal run reuses audio only when the recipe, declared hash, actual bytes, format, sample rate, and non-silence check all match; `--force` fetches fresh provider output. Because gTTS and Edge do not expose a service-model revision, the manifest says so rather than claiming one.

Preparation occurs in a sibling temporary directory. All 94 records are checked for count, split, unique IDs/paths, mono 16 kHz PCM, non-silence, duration, recipe checksum, WAV checksum, and voice disjointness before publication. Immutable content-addressed WAVs are moved first and `manifest.jsonl` is atomically replaced last. A failed synthesis or final manifest swap therefore leaves the previous manifest and every referenced WAV readable; at worst it leaves an unreferenced cache file.

These are disjoint synthetic voice identities, not human speaker IDs. The split contains only one held-out voice per language and is a plumbing/generalisation check, not a population estimate.

The source texts, split construction, exact 4 s + 4 s concatenation, and known boundary metadata live in `scripts/prepare_data.py`. The entire generated directory is ignored by Git.
