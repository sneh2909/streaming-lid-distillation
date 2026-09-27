# Data

No audio is committed. Everything under `data/` except this file is produced by the scripts below and is gitignored.

## Sources

| Source | Used for | Licence / access |
|---|---|---|
| [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google) | hi, en (US), gu, mr, bn, ta, te read speech. `dev` split → train, `test` split → eval | CC-BY-4.0, open |
| [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (AI4Bharat) | Indian-accented English, split by a demographic speaker key (no speaker IDs are published) | gated on Hugging Face (accept terms) |
| [Common Voice 17](https://huggingface.co/datasets/fixie-ai/common_voice_17_0) (Mozilla, HF mirror) | Speaker-diverse hi, mr, bn, ta, and English with an Indian accent. Eval uses CV Hindi `test` speakers and held-out English speakers | CC0, open |
| Own recordings (`tools/recorder/`) | Real same-speaker hi/en/Hinglish turns, **eval only** | private, never committed (`data/user_recordings/` is gitignored) |

No split uses transcripts for training: every label the student learns from is a teacher posterior (`scripts/build_targets.py`).

## Pipeline (run in this order)

```bash
python scripts/prepare_data.py          # FLEURS + Svarah selection -> data/raw/, first manifests
python scripts/clean_data.py            # Silero VAD speech-only, speech level -> -23 dBFS,
                                        # rebuild switch/mix clips as sentence + pause + sentence
python scripts/add_commonvoice.py       # CV mr/bn/ta + Indian-accented English + CV switch mixes
python scripts/add_cv_hindi_speakers.py # CV Hindi from ~110 speakers (replaces the 3-speaker CV train split)
```

- `prepare_data.py` also builds the first-version switch and mix clips (`data/generated/`). `clean_data.py` replaces them; the originals are kept for the original-data experiments in `results/v1_data/`.
- The first manifests are backed up to `data/manifests_v1/`.

## What ends up where

| Path | Contents |
|---|---|
| `data/manifests/train.jsonl` | monolingual training clips (FLEURS dev, Svarah, Common Voice) |
| `data/manifests/train_mix.jsonl` | multi-language training clips with known segment times |
| `data/manifests/eval.jsonl` | FLEURS test + Svarah eval (480 clips) |
| `data/manifests/switch.jsonl` | 90 eval switch clips (9 language pairs), switch time known |
| `data/manifests/eval_cv.jsonl` | real-world eval: CV Hindi test speakers + held-out Indian-English speakers |
| `data/clean/` | the cleaned audio the manifests point to |
| `data/targets/<teacher>/<kind>.pt` | teacher targets per clip: `{path: (frames, q)}` |

## Why the cleaning step exists

- **Loudness.** In the original data ~75% of FLEURS US-English clips were ~30 dB quieter than every other language, and a student learned "quiet ⇒ English". The fix was level normalisation plus random-gain augmentation.
- **Speaker diversity.** Common Voice's Hindi `train` split has only 3 speakers, and a student memorised them. The fix was drawing Hindi from ~110 speakers in the other splits, excluding every test speaker.

Details and before/after numbers are in the main `README.md` (§1, §5).
