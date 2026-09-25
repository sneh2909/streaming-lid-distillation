#!/usr/bin/env python3
"""Synthesize a speaker-disjoint multilingual corpus and switch clips."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import edge_tts
import miniaudio
import numpy as np
import soundfile as sf
from gtts import gTTS

from streaming_lid.config import LANGUAGES, SAMPLE_RATE
from streaming_lid.data import require_speaker_disjoint

TRAIN_CLIPS_PER_LANGUAGE = 10
HELDOUT_CLIPS_PER_LANGUAGE = 3
# Each language uses a female and a male Edge voice in training. The gTTS locale
# voice is held out entirely. These are synthetic voice identities, not claims
# about unique human speakers; both gender coverage and identity separation are
# explicit so the evaluation split cannot accidentally be "all unseen males".
EDGE_VOICES = {
    "en": {"female": "en-IN-NeerjaNeural", "male": "en-IN-PrabhatNeural"},
    "hi": {"female": "hi-IN-SwaraNeural", "male": "hi-IN-MadhurNeural"},
    "mr": {"female": "mr-IN-AarohiNeural", "male": "mr-IN-ManoharNeural"},
    "bn": {"female": "bn-IN-TanishaaNeural", "male": "bn-IN-BashkarNeural"},
    "ta": {"female": "ta-IN-PallaviNeural", "male": "ta-IN-ValluvarNeural"},
    "te": {"female": "te-IN-ShrutiNeural", "male": "te-IN-MohanNeural"},
    "gu": {"female": "gu-IN-DhwaniNeural", "male": "gu-IN-NiranjanNeural"},
}

SENTENCES = {
    "en": [
        "Hello, I am calling to check the delivery status of my order today.",
        "Please tell me when the nearest service center will open tomorrow.",
        "My account number is ready, and I need help with a payment problem.",
        "I would like to change the address connected to my monthly bill.",
        "Can you explain why the balance shown in the mobile application is different?",
        "The verification code has arrived, so we can continue with the request.",
        "I lost my bank card, so please block it immediately.",
        "The refund for my cancelled order has not arrived yet.",
        "I need to schedule the technician visit for next Monday.",
        "Please add my new mobile number to the account.",
        "The same payment appears twice on my statement.",
        "I forgot my personal identification number and need to reset it.",
        "I need to speak with a representative about this problem.",
    ],
    "hi": [
        "नमस्ते, मैं आज अपने ऑर्डर की डिलीवरी की जानकारी लेने के लिए फ़ोन कर रहा हूँ।",
        "कृपया बताइए कि नज़दीकी सेवा केंद्र कल कितने बजे खुलेगा।",
        "मेरा खाता नंबर तैयार है और मुझे भुगतान की समस्या में सहायता चाहिए।",
        "मैं अपने मासिक बिल से जुड़ा पता बदलना चाहता हूँ।",
        "क्या आप बता सकते हैं कि मोबाइल ऐप में दिखाई गई शेष राशि अलग क्यों है?",
        "सत्यापन कोड आ गया है, इसलिए हम अनुरोध पर आगे बढ़ सकते हैं।",
        "मेरा बैंक कार्ड खो गया है, कृपया इसे तुरंत बंद कर दीजिए।",
        "मेरे रद्द किए गए ऑर्डर की धनवापसी अभी तक नहीं मिली है।",
        "मैं तकनीशियन की यात्रा अगले सोमवार के लिए तय करना चाहता हूँ।",
        "कृपया मेरे खाते में नया मोबाइल नंबर जोड़ दीजिए।",
        "मेरे विवरण में एक ही भुगतान दो बार दिखाई दे रहा है।",
        "मैं अपना गुप्त पहचान नंबर भूल गया हूँ और उसे बदलना चाहता हूँ।",
        "मुझे इस समस्या के बारे में किसी प्रतिनिधि से बात करनी है।",
    ],
    "mr": [
        "नमस्कार, माझ्या ऑर्डरची आजची स्थिती जाणून घेण्यासाठी मी फोन केला आहे.",
        "कृपया जवळचे सेवा केंद्र उद्या किती वाजता उघडेल ते सांगा.",
        "माझा खाते क्रमांक तयार आहे आणि मला पेमेंटसाठी मदत हवी आहे.",
        "मला मासिक बिलाशी जोडलेला पत्ता बदलायचा आहे.",
        "मोबाइल अनुप्रयोगात दिसणारी शिल्लक वेगळी का आहे ते सांगाल का?",
        "पडताळणीचा कोड आला आहे, त्यामुळे आपण विनंती पुढे नेऊ शकतो.",
        "माझे बँक कार्ड हरवले आहे, कृपया ते लगेच बंद करा.",
        "रद्द केलेल्या ऑर्डरचा परतावा अजून मिळालेला नाही.",
        "तंत्रज्ञाची भेट पुढच्या सोमवारसाठी ठरवायची आहे.",
        "कृपया माझ्या खात्यात नवीन मोबाइल क्रमांक जोडा.",
        "एकच पेमेंट माझ्या विवरणात दोनदा दिसत आहे.",
        "मी माझा गुप्त क्रमांक विसरलो आहे आणि तो बदलायचा आहे.",
        "या समस्येबद्दल मला प्रतिनिधीशी बोलायचे आहे.",
    ],
    "bn": [
        "নমস্কার, আজ আমার অর্ডারের ডেলিভারির খবর জানতে আমি ফোন করেছি।",
        "দয়া করে বলুন কাছের পরিষেবা কেন্দ্রটি আগামীকাল কখন খুলবে।",
        "আমার অ্যাকাউন্ট নম্বর প্রস্তুত আছে এবং পেমেন্টে সাহায্য দরকার।",
        "আমি মাসিক বিলের সঙ্গে যুক্ত ঠিকানাটি বদলাতে চাই।",
        "মোবাইল অ্যাপে দেখানো ব্যালেন্স আলাদা কেন তা বুঝিয়ে বলবেন?",
        "যাচাইয়ের কোড এসেছে, তাই আমরা অনুরোধটি এগিয়ে নিতে পারি।",
        "আমার ব্যাংক কার্ড হারিয়ে গেছে, দয়া করে এখনই বন্ধ করুন।",
        "বাতিল করা অর্ডারের টাকা এখনো ফেরত পাইনি।",
        "আগামী সোমবারের জন্য প্রযুক্তিবিদের আসা ঠিক করতে চাই।",
        "দয়া করে আমার অ্যাকাউন্টে নতুন মোবাইল নম্বর যোগ করুন।",
        "একই পেমেন্ট আমার বিবরণীতে দুবার দেখা যাচ্ছে।",
        "আমি গোপন নম্বর ভুলে গেছি এবং সেটি বদলাতে চাই।",
        "এই সমস্যা নিয়ে একজন প্রতিনিধির সঙ্গে কথা বলতে চাই।",
    ],
    "ta": [
        "வணக்கம், இன்று எனது ஆர்டர் எப்போது வரும் என்பதை அறிய அழைத்தேன்.",
        "அருகிலுள்ள சேவை மையம் நாளை எப்போது திறக்கும் என்று சொல்லுங்கள்.",
        "எனது கணக்கு எண் தயாராக உள்ளது, பணம் செலுத்த உதவி தேவை.",
        "மாதாந்திர ரசீதுடன் இணைந்த முகவரியை மாற்ற விரும்புகிறேன்.",
        "கைபேசி செயலியில் காட்டும் இருப்பு ஏன் வேறாக உள்ளது என்று விளக்குங்கள்.",
        "சரிபார்ப்பு குறியீடு வந்துவிட்டது, கோரிக்கையைத் தொடரலாம்.",
        "எனது வங்கி அட்டை தொலைந்துவிட்டது, அதை உடனே முடக்குங்கள்.",
        "ரத்து செய்த ஆர்டருக்கான பணம் இன்னும் திரும்ப வரவில்லை.",
        "அடுத்த திங்கட்கிழமை தொழில்நுட்ப நிபுணர் வர ஏற்பாடு செய்ய வேண்டும்.",
        "எனது கணக்கில் புதிய கைபேசி எண்ணைச் சேருங்கள்.",
        "ஒரே பணம் எனது கணக்கு அறிக்கையில் இரண்டு முறை உள்ளது.",
        "எனது ரகசிய எண்ணை மறந்துவிட்டேன், அதை மாற்ற வேண்டும்.",
        "இந்தப் பிரச்சினையைப் பற்றி ஒரு பிரதிநிதியிடம் பேச வேண்டும்.",
    ],
    "te": [
        "నమస్కారం, నా ఆర్డర్ ఈరోజు ఎప్పుడు వస్తుందో తెలుసుకోవడానికి ఫోన్ చేశాను.",
        "దయచేసి దగ్గరలోని సేవా కేంద్రం రేపు ఎప్పుడు తెరుస్తుందో చెప్పండి.",
        "నా ఖాతా సంఖ్య సిద్ధంగా ఉంది, చెల్లింపు విషయంలో సహాయం కావాలి.",
        "నా నెలవారీ బిల్లుకు సంబంధించిన చిరునామాను మార్చాలనుకుంటున్నాను.",
        "మొబైల్ అనువర్తనంలో చూపిన నిల్వ ఎందుకు భిన్నంగా ఉందో వివరించండి.",
        "ధృవీకరణ సంకేతం వచ్చింది, కాబట్టి అభ్యర్థనను కొనసాగించవచ్చు.",
        "నా బ్యాంకు కార్డు పోయింది, దయచేసి వెంటనే నిలిపివేయండి.",
        "రద్దు చేసిన ఆర్డర్ డబ్బు ఇంకా తిరిగి రాలేదు.",
        "వచ్చే సోమవారానికి సాంకేతిక నిపుణుడి సందర్శనను ఏర్పాటు చేయాలి.",
        "దయచేసి నా ఖాతాలో కొత్త మొబైల్ సంఖ్యను చేర్చండి.",
        "ఒకే చెల్లింపు నా ఖాతా వివరాల్లో రెండుసార్లు కనిపిస్తోంది.",
        "నా రహస్య సంఖ్య మర్చిపోయాను, దానిని మార్చాలి.",
        "ఈ సమస్య గురించి ఒక ప్రతినిధితో మాట్లాడాలి.",
    ],
    "gu": [
        "નમસ્તે, મારો ઓર્ડર આજે ક્યારે આવશે તે જાણવા માટે મેં ફોન કર્યો છે.",
        "કૃપા કરીને નજીકનું સેવા કેન્દ્ર આવતીકાલે ક્યારે ખુલશે તે જણાવો.",
        "મારો ખાતા નંબર તૈયાર છે અને મને ચુકવણીમાં મદદ જોઈએ છે.",
        "હું મારા માસિક બિલ સાથે જોડાયેલું સરનામું બદલવા માગું છું.",
        "મોબાઇલ એપમાં બતાવેલી બાકી રકમ અલગ કેમ છે તે સમજાવો.",
        "ચકાસણીનો કોડ આવી ગયો છે, તેથી વિનંતી આગળ વધારી શકીએ છીએ.",
        "મારું બેંક કાર્ડ ખોવાઈ ગયું છે, કૃપા કરીને તરત બંધ કરો.",
        "રદ કરેલા ઓર્ડરના પૈસા હજી પાછા મળ્યા નથી.",
        "આવતા સોમવારે ટેકનિશિયનની મુલાકાત ગોઠવવી છે.",
        "કૃપા કરીને મારા ખાતામાં નવો મોબાઇલ નંબર ઉમેરો.",
        "એક જ ચુકવણી મારા હિસાબમાં બે વાર દેખાય છે.",
        "હું મારો ગુપ્ત નંબર ભૂલી ગયો છું અને તેને બદલવો છે.",
        "આ સમસ્યા વિશે મારે પ્રતિનિધિ સાથે વાત કરવી છે.",
    ],
}


def decode_and_write(temporary_name: str, output_path: Path) -> None:
    """Decode an MP3 response, trim silence, peak-limit, and write PCM WAV."""
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


def synthesize(text: str, language: str, voice: str, output_path: Path) -> None:
    """Fetch one TTS response and store a normalized 16 kHz WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp3", prefix="tts-", dir=output_path.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
        if voice.startswith("gtts:"):
            gTTS(text=text, lang=language, slow=False).save(temporary_name)
        else:
            edge_tts.Communicate(text=text, voice=voice).save_sync(temporary_name)
        decode_and_write(temporary_name, output_path)
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

    expected_sentences = TRAIN_CLIPS_PER_LANGUAGE + HELDOUT_CLIPS_PER_LANGUAGE
    for language, name in LANGUAGES.items():
        if len(SENTENCES[language]) != expected_sentences:
            raise ValueError(
                f"{language} needs {expected_sentences} sentences, "
                f"found {len(SENTENCES[language])}"
            )
        for sentence_index, text in enumerate(SENTENCES[language]):
            split = (
                "train"
                if sentence_index < TRAIN_CLIPS_PER_LANGUAGE
                else "heldout"
            )
            if split == "heldout":
                voice = f"gtts:{language}:default"
                source = "gTTS"
            else:
                gender = "female" if sentence_index < 5 else "male"
                voice = EDGE_VOICES[language][gender]
                source = "Microsoft Edge TTS via edge-tts"
            clip_id = f"{language}_{split}_{sentence_index:02d}"
            output_path = audio_dir / f"{clip_id}.wav"
            if args.force or not output_path.exists():
                print(f"synthesizing {clip_id} with {voice}: {text}", flush=True)
                synthesize(text, language, voice, output_path)
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
                    "source": source,
                    "speaker_id": voice,
                    "synthetic_voice": True,
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
        ("switch_hi_en_train", "hi_train_00", "en_train_00", "hi", "en", "train"),
        (
            "switch_hi_en_eval",
            "hi_heldout_10",
            "en_heldout_10",
            "hi",
            "en",
            "switch",
        ),
        (
            "switch_en_hi_eval",
            "en_heldout_10",
            "hi_heldout_10",
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
                "language_name": (
                    f"{LANGUAGES[first_language]} to {LANGUAGES[second_language]}"
                ),
                "source": "self-concatenated synthetic speech",
                "speaker_ids": [
                    by_id[first_id]["speaker_id"],
                    by_id[second_id]["speaker_id"],
                ],
                "synthetic_voice": True,
                "text": f"{by_id[first_id]['text']} | {by_id[second_id]['text']}",
                **switch_info,
            }
        )

    speaker_audit = require_speaker_disjoint(records)
    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "heldout", "switch")
    }
    print(
        f"wrote {manifest_path} with {len(records)} clips: {counts}; "
        f"train/evaluation speakers disjoint={speaker_audit['speaker_disjoint']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
