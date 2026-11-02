"""Flap position from the 737's analog flap gauge, read by Muse Spark.

Text recognition cannot read a needle. Spark is asked only where the needle points, as a clock
position; code turns that into a detent with a table measured in X-Plane (needle parked at
each detent, read by Spark). The raw clock position is kept
too, so the rules can tell a needle that is still moving from one that has settled, and average
several reads: Spark's reads of a parked needle vary by up to ~1.3 clock hours between frames.

Calibrated for FLAPS_VIEW (head turned right so the camera sees the PFD, ND and the centre panel).
"""
import base64
import io
import json
import os
import re

import httpx
from dotenv import load_dotenv
from PIL import Image

from senses.panel import PanelReading

FLAPS_VIEW = {"pilots_head_psi": 22.0, "pilots_head_the": -30.0, "field_of_view_deg": 60.0}
FLAP_GAUGE = (2470, 340, 2700, 560)  # full-screen pixels in FLAPS_VIEW (3600x2338 capture)
# Where Spark reads the needle at each detent, in clock hours on a continuous scale (12 o'clock =
# 12, 3 o'clock = 15). Mean of Spark's reads with the needle parked there in X-Plane: 1, 2, 5 from
# 8 reads each (1: 9.0-10.0, 2: 11.0-11.3, 5: 12.0-12.5); the rest from 2 reads each.
DIAL = {0: 9.0, 1: 9.6, 2: 11.0, 5: 12.25, 10: 14.0, 15: 15.0, 25: 15.9, 30: 16.85, 40: 18.0}
PROMPT = ("Boeing 737 flap position indicator. The dial marks, by clock position: UP at 9 o'clock, 1 at 10, 2 at 11, "
          "5 at 12, 10 at 1, 15 at 3, 25 at 4, 30 at 5, 40 at 6. The needle is the thick white bar from the centre. "
          "Give the clock position the needle points to, then the flap value. JSON only: "
          '{"clock": <e.g. 2.5>, "flaps": <value, interpolated between marks if between them; 0 for UP>}')


def detent(clock: float) -> int:
    return min(DIAL, key=lambda d: abs(DIAL[d] - clock))


async def read_flaps_spark(frame: Image.Image, client: httpx.AsyncClient | None = None) -> tuple[PanelReading, dict]:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    img = frame.crop(FLAP_GAUGE).convert("RGB")
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    body = {"model": "muse-spark-1.3", "max_tokens": 1500, "reasoning_effort": "minimal", "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}]}]}
    own = client is None
    client = client or httpx.AsyncClient(trust_env=False, timeout=60)
    try:
        r = await client.post(os.environ.get("META_API_BASE", "https://api.meta.ai/v1") + "/chat/completions", json=body,
                              headers={"Authorization": "Bearer " + os.environ["MODEL_API_KEY"]})
        r.raise_for_status()
    finally:
        if own:
            await client.aclose()
    d = r.json()
    text = d["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", text, re.S)
    clock = None
    try:
        clock = float(json.loads(m.group())["clock"]) if m else None
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        clock = None
    if clock is not None:
        clock = clock + 12 if clock < 7 else clock  # 1 o'clock -> 13: one continuous scale from UP to 40
        if not 8.5 <= clock <= 18.5:
            clock = None
    reading = PanelReading(flaps=detent(clock) if clock is not None else None,
                           flaps_clock=round(clock, 2) if clock is not None else None)
    reading.raw = {"spark": text}
    return reading, d.get("usage", {})
