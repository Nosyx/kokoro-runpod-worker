import io
import os
import re
import time
import uuid

import numpy as np
import requests
import runpod
import soundfile as sf
from kokoro import KPipeline

pipeline = KPipeline(lang_code="a")

SAMPLE_RATE = 24000

BUNNY_STORAGE_ZONE = os.environ.get("BUNNY_STORAGE_ZONE", "videocrm")
BUNNY_ACCESS_KEY = os.environ.get("BUNNY_ACCESS_KEY")
BUNNY_PULL_ZONE_URL = os.environ.get("BUNNY_PULL_ZONE_URL", "https://videocrm.b-cdn.net")
BUNNY_STORAGE_HOST = os.environ.get("BUNNY_STORAGE_HOST", "storage.bunnycdn.com")


def upload_to_bunny(data: bytes, content_type: str, extension: str) -> str:
    if not BUNNY_ACCESS_KEY:
        raise RuntimeError("BUNNY_ACCESS_KEY env var is not set on the worker")

    remote_path = f"voiceovers/{int(time.time())}-{uuid.uuid4().hex}.{extension}"
    url = f"https://{BUNNY_STORAGE_HOST}/{BUNNY_STORAGE_ZONE}/{remote_path}"

    resp = requests.put(
        url,
        data=data,
        headers={"AccessKey": BUNNY_ACCESS_KEY, "Content-Type": content_type},
        timeout=60,
    )
    resp.raise_for_status()

    return f"{BUNNY_PULL_ZONE_URL.rstrip('/')}/{remote_path}"


# Kokoro's own punctuation timing is unreliable (see hexgrad/kokoro#59, #202),
# so pause length is enforced here explicitly via silence padding instead.
COMMA_PAUSE_S = 0.045    # comma
SENTENCE_PAUSE_S = 0.105 # . ! ?
TENSION_PAUSE_S = 0.165  # ellipsis / dash (hesitation, tension)
LONG_PAUSE_S = 0.27      # paragraph break (new thought)

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
    duration_ms = round(len(full_audio) / SAMPLE_RATE * 1000)

    buffer = io.BytesIO()
    sf.write(buffer, full_audio, samplerate=SAMPLE_RATE, format="WAV")

    try:
        audio_url = upload_to_bunny(buffer.getvalue(), "audio/wav", "wav")
    except Exception as exc:  # noqa: BLE001 — surface upload failures to the caller
        return {"error": f"Bunny upload failed: {exc}"}

    return {"audio_url": audio_url, "duration_ms": duration_ms}


runpod.serverless.start({"handler": handler})
