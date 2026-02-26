import os
import json
import aiohttp
import asyncio
from dotenv import load_dotenv
import av
import numpy as np
import base64
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
ENDPOINT = os.getenv("ENDPOINT")

HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

# Shared aiohttp session + concurrency limiter
_session: aiohttp.ClientSession = None
_sema = asyncio.Semaphore(8)  # adjust concurrency here

async def _get_session():
    global _session
    if _session is None:
        _session = aiohttp.ClientSession(headers=HEADERS)
    return _session

async def async_generator(payload, max_retries=3, backoff=1):
    """
    Async POST to LLM endpoint with simple retry & concurrency control.
    Returns the content string.
    """
    session = await _get_session()
    async with _sema:
        for attempt in range(max_retries):
            try:
                async with session.post(ENDPOINT, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff)
                else:
                    raise

def format_message(messages, role, prompt):
    messages.append({
        "role": role,
        "content": [{"type": "text", "text": prompt}]
    })
    return messages

def create_llm_payload(messages, parameters=None):
    if parameters is None:
        parameters = {}
    return {
        "messages": messages,
        "temperature": parameters.get("temperature", 0.6),
        "max_tokens": parameters.get("max_tokens", 1000),
    }

def read_video_pyav_base64(video_path, fps=3, max_frames=250):
    """
    Unchanged from your original: extract frames as base64 JPEGs.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    if stream.duration is not None:
        duration = float(stream.duration * stream.time_base)
    else:
        frame_rate = float(stream.average_rate or 25.0)
        duration = stream.frames / frame_rate if stream.frames else 10.0

    est_total = int(duration * fps)
    if est_total <= max_frames:
        timestamps = np.arange(0, duration, 1 / fps)
    else:
        timestamps = np.linspace(0, duration, max_frames, endpoint=False)

    base64_frames = []
    ts_index = 0
    for frame in container.decode(video=0):
        if frame.pts is None:
            continue
        current_time = float(frame.pts * stream.time_base)
        if ts_index >= len(timestamps):
            break
        if current_time >= timestamps[ts_index]:
            img = Image.fromarray(frame.to_ndarray(format="rgb24"))
            buf = BytesIO()
            img.save(buf, format="JPEG")
            base64_frames.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
            ts_index += 1

    return base64_frames





# Example usage (stand-alone test):
if __name__ == "__main__":
    msgs = []
    msgs = format_message(msgs, "user", "Hello, describe AI in simple terms.")
    payload = create_llm_payload(msgs)
    print("LLM response:", generator(payload))
