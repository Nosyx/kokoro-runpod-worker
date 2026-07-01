import io
import os
import random
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

# Kokoro has no pitch/prosody knob, only `speed`. A fixed speed makes long narration sound
# mechanically even, so this jitters it slightly per phoneme-length chunk instead of using
# one flat value — subtle enough to stay natural-sounding, not slow it down or add pauses.
JITTER_RANGE = (0.97, 1.03)


def jittered_speed(base_speed: float):
    def speed_fn(_phoneme_len: int) -> float:
        return base_speed * random.uniform(*JITTER_RANGE)

    return speed_fn


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
