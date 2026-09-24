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
