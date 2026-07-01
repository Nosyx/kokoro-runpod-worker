import base64
import io
import re

import numpy as np
import runpod
import soundfile as sf
from kokoro import KPipeline

pipeline = KPipeline(lang_code="a")

SAMPLE_RATE = 24000

# Kokoro's own punctuation timing is unreliable (see hexgrad/kokoro#59, #202),
# so pause length is enforced here explicitly via silence padding instead.
COMMA_PAUSE_S = 0.15    # comma
SENTENCE_PAUSE_S = 0.35 # . ! ?
TENSION_PAUSE_S = 0.55  # ellipsis / dash (hesitation, tension)
LONG_PAUSE_S = 0.9      # paragraph break (new thought)

# Split into segments, each tagged with the pause that should follow it.
SEGMENT_PATTERN = re.compile(r"(\.\.\.|[,.!?]|\n\s*\n)")


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def split_with_pauses(text: str):
    """Yield (segment_text, trailing_pause_seconds) pairs."""
    parts = SEGMENT_PATTERN.split(text)
    segment = ""
    for part in parts:
        if part is None or part == "":
            continue
        if SEGMENT_PATTERN.fullmatch(part):
            if "\n" in part:
                pause = LONG_PAUSE_S
            elif part in ("...", "-", "—"):
                pause = TENSION_PAUSE_S
            elif part == ",":
                pause = COMMA_PAUSE_S
            else:  # . ! ?
                pause = SENTENCE_PAUSE_S
            cleaned = segment.strip()
            if cleaned:
                yield cleaned, pause
            segment = ""
        else:
            segment += part
    cleaned = segment.strip()
    if cleaned:
        yield cleaned, 0.0


def synthesize(text: str, voice: str, speed: float) -> np.ndarray:
    chunks = []
    for segment_text, pause_s in split_with_pauses(text):
        for _, _, audio in pipeline(segment_text, voice=voice, speed=speed):
            chunks.append(audio)
        if pause_s > 0:
            chunks.append(silence(pause_s))

    if not chunks:
        return silence(0.1)

    return np.concatenate(chunks)


def handler(job):
    job_input = job["input"]
    text = job_input.get("text")
    voice = job_input.get("voice", "af_bella")
    speed = job_input.get("speed", 1.0)

    if not text:
        return {"error": "Missing 'text' in input"}

    full_audio = synthesize(text, voice, speed)

    buffer = io.BytesIO()
    sf.write(buffer, full_audio, samplerate=SAMPLE_RATE, format="WAV")
    audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"audio_base64": audio_base64}


runpod.serverless.start({"handler": handler})
