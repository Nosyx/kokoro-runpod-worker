import base64
import io

import numpy as np
import runpod
import soundfile as sf
from kokoro import KPipeline

pipeline = KPipeline(lang_code="a")


def handler(job):
    job_input = job["input"]
    text = job_input.get("text")
    voice = job_input.get("voice", "af_bella")
    speed = job_input.get("speed", 1.0)

    if not text:
        return {"error": "Missing 'text' in input"}

    audio_chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        audio_chunks.append(audio)

    full_audio = np.concatenate(audio_chunks)

    buffer = io.BytesIO()
    sf.write(buffer, full_audio, samplerate=24000, format="WAV")
    audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"audio_base64": audio_base64}


runpod.serverless.start({"handler": handler})
