import io
import os
import random
import re
import time
import uuid

import numpy as np
import requests
import runpod
import soundfile as sf
from kokoro import KPipeline
from num2words import num2words

pipeline = KPipeline(lang_code="a")

SAMPLE_RATE = 24000

BUNNY_STORAGE_ZONE = os.environ.get("BUNNY_STORAGE_ZONE", "videocrm")
BUNNY_ACCESS_KEY = os.environ.get("BUNNY_ACCESS_KEY")
BUNNY_PULL_ZONE_URL = os.environ.get("BUNNY_PULL_ZONE_URL", "https://videocrm.b-cdn.net")
BUNNY_STORAGE_HOST = os.environ.get("BUNNY_STORAGE_HOST", "storage.bunnycdn.com")

# Kokoro has no pitch/prosody knob, only `speed`. A fixed speed makes long narration sound
# mechanically even, so this jitters it slightly per phoneme-length chunk instead of using
# one flat value — subtle enough to stay natural-sounding, not slow it down or add pauses.
JITTER_RANGE = (0.97, 1.03)

_COMMA_GROUPED_NUMBER = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_HYPHEN_BETWEEN_WORDS = re.compile(r"(?<=\w)-(?=\w)")
_STANDALONE_DASH = re.compile(r"\s[-–—]\s")


def jittered_speed(base_speed: float):
    def speed_fn(_phoneme_len: int) -> float:
        return base_speed * random.uniform(*JITTER_RANGE)

    return speed_fn


def normalize_tts_text(text: str) -> str:
    # "19,000" -> spelled out as words. Kokoro/misaki has been observed mangling
    # comma-grouped thousands, so bypass its own number parsing for these specifically
    # (plain numbers without commas, like years, are left to Kokoro as before).
    text = _COMMA_GROUPED_NUMBER.sub(
        lambda m: num2words(int(m.group(0).replace(",", ""))), text
    )

    # Hyphens/dashes have been observed truncating the preceding word's last letter —
    # replace with plain spacing/pauses instead of relying on Kokoro's own dash handling.
    text = _STANDALONE_DASH.sub(", ", text)
    text = _HYPHEN_BETWEEN_WORDS.sub(" ", text)

    return text


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


def handler(job):
    job_input = job["input"]
    text = job_input.get("text")
    voice = job_input.get("voice", "af_bella")
    speed = job_input.get("speed", 1.0)

    if not text:
        return {"error": "Missing 'text' in input"}

    text = normalize_tts_text(text)

    audio_chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=jittered_speed(speed)):
        audio_chunks.append(audio)

    full_audio = np.concatenate(audio_chunks)
    duration_ms = round(len(full_audio) / SAMPLE_RATE * 1000)

    buffer = io.BytesIO()
    sf.write(buffer, full_audio, samplerate=SAMPLE_RATE, format="WAV")

    try:
        audio_url = upload_to_bunny(buffer.getvalue(), "audio/wav", "wav")
    except Exception as exc:  # noqa: BLE001 — surface upload failures to the caller
        return {"error": f"Bunny upload failed: {exc}"}

    return {"audio_url": audio_url, "duration_ms": duration_ms}


runpod.serverless.start({"handler": handler})
