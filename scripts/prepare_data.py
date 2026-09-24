#!/usr/bin/env python3
"""Synthesize a tiny multilingual corpus and deterministic switch clips."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import miniaudio
import numpy as np
import soundfile as sf
from gtts import gTTS

from streaming_lid.config import LANGUAGES, SAMPLE_RATE

SENTENCES = {
    "en": [
        "Hello, I am calling to check the delivery status of my order today.",
        "Please tell me when the nearest service center will open tomorrow.",
        "My account number is ready, and I need help with a payment problem.",
        "I would like to change the address connected to my monthly bill.",
        "Can you explain why the balance shown in the mobile application is different?",
        "The verification code has arrived, so we can continue with the request.",
    ],
    "hi": [
        "नमस्ते, मैं आज अपने ऑर्डर की डिलीवरी की जानकारी लेने के लिए फ़ोन कर रहा हूँ।",
        "कृपया बताइए कि नज़दीकी सेवा केंद्र कल कितने बजे खुलेगा।",
        "मेरा खाता नंबर तैयार है और मुझे भुगतान की समस्या में सहायता चाहिए।",
        "मैं अपने मासिक बिल से जुड़ा पता बदलना चाहता हूँ।",
        "क्या आप बता सकते हैं कि मोबाइल ऐप में दिखाई गई शेष राशि अलग क्यों है?",
        "सत्यापन कोड आ गया है, इसलिए हम अनुरोध पर आगे बढ़ सकते हैं।",
    ],
    "mr": [
        "नमस्कार, माझ्या ऑर्डरची आजची स्थिती जाणून घेण्यासाठी मी फोन केला आहे.",
        "कृपया जवळचे सेवा केंद्र उद्या किती वाजता उघडेल ते सांगा.",
        "माझा खाते क्रमांक तयार आहे आणि मला पेमेंटसाठी मदत हवी आहे.",
    ],
    "bn": [
        "নমস্কার, আজ আমার অর্ডারের ডেলিভারির খবর জানতে আমি ফোন করেছি।",
        "দয়া করে বলুন কাছের পরিষেবা কেন্দ্রটি আগামীকাল কখন খুলবে।",
        "আমার অ্যাকাউন্ট নম্বর প্রস্তুত আছে এবং পেমেন্টে সাহায্য দরকার।",
    ],
    "ta": [
        "வணக்கம், இன்று எனது ஆர்டர் எப்போது வரும் என்பதை அறிய அழைத்தேன்.",
        "அருகிலுள்ள சேவை மையம் நாளை எப்போது திறக்கும் என்று சொல்லுங்கள்.",
        "எனது கணக்கு எண் தயாராக உள்ளது, பணம் செலுத்த உதவி தேவை.",
    ],
    "te": [
        "నమస్కారం, నా ఆర్డర్ ఈరోజు ఎప్పుడు వస్తుందో తెలుసుకోవడానికి ఫోన్ చేశాను.",
        "దయచేసి దగ్గరలోని సేవా కేంద్రం రేపు ఎప్పుడు తెరుస్తుందో చెప్పండి.",
        "నా ఖాతా సంఖ్య సిద్ధంగా ఉంది, చెల్లింపు విషయంలో సహాయం కావాలి.",
    ],
    "gu": [
        "નમસ્તે, મારો ઓર્ડર આજે ક્યારે આવશે તે જાણવા માટે મેં ફોન કર્યો છે.",
        "કૃપા કરીને નજીકનું સેવા કેન્દ્ર આવતીકાલે ક્યારે ખુલશે તે જણાવો.",
        "મારો ખાતા નંબર તૈયાર છે અને મને ચુકવણીમાં મદદ જોઈએ છે.",
    ],
}


def synthesize(text: str, language: str, output_path: Path) -> None:
    """Fetch gTTS MP3, decode in-process, trim silence, and save PCM WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp3", prefix="tts-", dir=output_path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
        gTTS(text=text, lang=language, slow=False).save(temporary_name)
        decoded = miniaudio.decode_file(
            temporary_name,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1,
            sample_rate=SAMPLE_RATE,
        )
        samples = np.frombuffer(decoded.samples, dtype=np.float32).copy()
        active = np.flatnonzero(np.abs(samples) > 0.003)
        if len(active):
            margin = int(0.08 * SAMPLE_RATE)
            begin = max(0, int(active[0]) - margin)
            end = min(len(samples), int(active[-1]) + margin + 1)
            samples = samples[begin:end]
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        if peak > 0.95:
            samples *= 0.95 / peak
        sf.write(output_path, samples, SAMPLE_RATE, subtype="PCM_16")
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def four_second_segment(samples: np.ndarray, seconds: float = 4.0) -> np.ndarray:
    wanted = int(seconds * SAMPLE_RATE)
    if len(samples) >= wanted:
        return samples[:wanted]
    return np.pad(samples, (0, wanted - len(samples)))


def write_switch(
    first_path: Path,
    second_path: Path,
    output_path: Path,
    first_language: str,
    second_language: str,
) -> dict:
    first, first_rate = sf.read(first_path, dtype="float32")
    second, second_rate = sf.read(second_path, dtype="float32")
    if first_rate != SAMPLE_RATE or second_rate != SAMPLE_RATE:
        raise ValueError("source clips must already be 16 kHz")
    segment_seconds = 4.0
    joined = np.concatenate([four_second_segment(first), four_second_segment(second)])
    sf.write(output_path, joined, SAMPLE_RATE, subtype="PCM_16")
    return {
        "duration_seconds": len(joined) / SAMPLE_RATE,
        "segments": [
            {
                "language": first_language,
                "start_seconds": 0.0,
                "end_seconds": segment_seconds,
            },
            {
                "language": second_language,
                "start_seconds": segment_seconds,
                "end_seconds": 2 * segment_seconds,
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument(
        "--force", action="store_true", help="regenerate existing monolingual WAVs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    heldout_ids: dict[str, str] = {}
    for language, name in LANGUAGES.items():
        for sentence_index, text in enumerate(SENTENCES[language]):
            split = (
                "heldout" if sentence_index == len(SENTENCES[language]) - 1 else "train"
            )
            clip_id = f"{language}_{split}_{sentence_index}"
            if split == "heldout":
                heldout_ids[language] = clip_id
            output_path = audio_dir / f"{clip_id}.wav"
            if args.force or not output_path.exists():
                print(f"synthesizing {clip_id}: {text}", flush=True)
                synthesize(text, language, output_path)
            samples, rate = sf.read(output_path, dtype="float32")
            if rate != SAMPLE_RATE:
                raise ValueError(f"unexpected sample rate for {output_path}: {rate}")
            records.append(
                {
                    "id": clip_id,
                    "audio_path": f"audio/{output_path.name}",
                    "split": split,
                    "language": language,
                    "language_name": name,
                    "duration_seconds": round(len(samples) / SAMPLE_RATE, 4),
                    "source": "gTTS",
                    "text": text,
                    "segments": [
                        {
                            "language": language,
                            "start_seconds": 0.0,
                            "end_seconds": round(len(samples) / SAMPLE_RATE, 4),
                        }
                    ],
                }
            )

    by_id = {record["id"]: record for record in records}
    switches = [
        ("switch_hi_en_train", "hi_train_0", "en_train_0", "hi", "en", "train"),
        (
            "switch_hi_en_eval",
            heldout_ids["hi"],
            heldout_ids["en"],
            "hi",
            "en",
            "switch",
        ),
        (
            "switch_en_hi_eval",
            heldout_ids["en"],
            heldout_ids["hi"],
            "en",
            "hi",
            "switch",
        ),
    ]
    for (
        clip_id,
        first_id,
        second_id,
        first_language,
        second_language,
        split,
    ) in switches:
        output_path = audio_dir / f"{clip_id}.wav"
        switch_info = write_switch(
            args.output_dir / by_id[first_id]["audio_path"],
            args.output_dir / by_id[second_id]["audio_path"],
            output_path,
            first_language,
            second_language,
        )
        records.append(
            {
                "id": clip_id,
                "audio_path": f"audio/{output_path.name}",
                "split": split,
                "language": "mixed",
                "language_name": f"{LANGUAGES[first_language]} to {LANGUAGES[second_language]}",
                "source": "self-concatenated gTTS",
                "text": f"{by_id[first_id]['text']} | {by_id[second_id]['text']}",
                **switch_info,
            }
        )

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "heldout", "switch")
    }
    print(f"wrote {manifest_path} with {len(records)} clips: {counts}")


if __name__ == "__main__":
    main()
