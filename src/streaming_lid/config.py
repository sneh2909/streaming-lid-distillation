"""One source of truth for timing, language, and model constants."""

from collections import OrderedDict

SAMPLE_RATE = 16_000
N_FFT = 400
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 40
FRAME_MS = 1_000 * HOP_LENGTH / SAMPLE_RATE
WINDOW_MS = 1_000 * WIN_LENGTH / SAMPLE_RATE

LANGUAGES = OrderedDict(
    [
        ("en", "English"),
        ("hi", "Hindi"),
        ("mr", "Marathi"),
        ("bn", "Bengali"),
        ("ta", "Tamil"),
        ("te", "Telugu"),
        ("gu", "Gujarati"),
    ]
)
LANGUAGE_CODES = tuple(LANGUAGES)

TEACHER_NAME = "speechbrain/lang-id-voxlingua107-ecapa"
TEACHER_REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
TEACHER_ARTIFACT_SHA256 = (
    "f193a0548951e8fbd6ca438a492b98b5c48bf7d98c86451ca3878dd4a7c0f706"
)
TEACHER_ARTIFACT_FILES = OrderedDict(
    [
        (
            "classifier.ckpt",
            {
                "sha256": "a50d9024ff58d317031c9787d4c6c614d454a87a8ef32f9d36338cd3ff57adbc",
                "bytes": 762_555,
            },
        ),
        (
            "embedding_model.ckpt",
            {
                "sha256": "ab750d5c06d713477045fa798fab5d33e959dbc0dfe4de510a9a47844c79a19a",
                "bytes": 84_474_355,
            },
        ),
        (
            "hyperparams.yaml",
            {
                "sha256": "88fec9791a8416a152fb10834327e18d38e5bf7a351e9b714e08cdc4af05de6f",
                "bytes": 1_519,
            },
        ),
        (
            "label_encoder.txt",
            {
                "sha256": "9f566d83c4f19168be4a0bf86c0c7dac7d3264a95105bcbf33a7c32b83ccc17f",
                "bytes": 2_204,
            },
        ),
    ]
)
TEACHER_LANGUAGE_INDICES = (20, 35, 63, 9, 91, 92, 31)
TEACHER_TEMPERATURE = 2.0
TEACHER_PAST_MS = 1_750
TEACHER_FUTURE_MS = 250
TEACHER_WINDOW_MS = TEACHER_PAST_MS + TEACHER_FUTURE_MS
TEACHER_HOP_FRAMES = 25  # Run the expensive teacher every 250 ms, then interpolate.

# A prediction at student index i + 21 consumes explicit feature lookahead through
# i + 21 + 4. Thus it has exactly the teacher's 250 ms of future evidence.
LABEL_DELAY_FRAMES = 21
MODEL_LOOKAHEAD_FRAMES = 4
EVIDENCE_LOOKAHEAD_MS = (LABEL_DELAY_FRAMES + MODEL_LOOKAHEAD_FRAMES) * FRAME_MS
CHUNK_FRAMES = 16
CHUNK_MS = CHUNK_FRAMES * FRAME_MS
ALGORITHMIC_LATENCY_MS = WINDOW_MS + EVIDENCE_LOOKAHEAD_MS + CHUNK_MS
EARLY_RAMP_FRAMES = 100  # First second is down-weighted: little speech context exists.

STUDENT_HIDDEN_SIZE = 64
STUDENT_DILATIONS = (1, 2, 4, 8, 16, 32)

assert EVIDENCE_LOOKAHEAD_MS == TEACHER_FUTURE_MS
