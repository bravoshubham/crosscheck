"""Panel reader on Muse Spark (Meta Model API): the cross-platform alternative to senses/panel.py,
which needs Apple's on-device text recognition.

Sends the calibrated PFD crop and asks for the values as JSON, with minimal reasoning effort
(~4 s and ~430 tokens a frame; the default effort spends thousands of tokens thinking and can
run out before answering). Values outside what a 737 PFD can show are dropped, so a bad read
becomes "unknown", never a wrong fact.
"""
import asyncio
import base64
import io
import json
import os
import re

import httpx
from dotenv import load_dotenv
from PIL import Image

from senses.panel import CALIBRATION, PanelReading

MODEL = "muse-spark-1.3"
PROMPT = (
    "This is a Boeing 737 primary flight display (PFD). Read it and reply with JSON only, no prose:\n"
    '{"ias": <airspeed in the box on the speed tape, kt>, "selected_speed": <magenta number above the speed tape, kt>, '
    '"selected_altitude": <magenta number above the altitude tape, ft>, "altitude": <altitude in the box on the '
    'altitude tape, ft, including the rolling last two digits>, "fma": [<thrust mode>, <roll mode>, <pitch mode>]}\n'
    "The fma is the green text in the three boxes along the top; use \"\" for an empty box. "
    "Use null for any value you cannot read."
)
ARMED_OK = re.compile(r"^[A-Z0-9 /]{0,12}$")


def _plausible(r: dict) -> dict:
    def num(k, lo, hi, step=1):
        v = r.get(k)
        if isinstance(v, str):
            v = re.sub(r"[^\d.]", "", v) or None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return int(v) if lo <= v <= hi and int(v) % step == 0 else None
    fma = r.get("fma") if isinstance(r.get("fma"), list) and len(r["fma"]) == 3 else None
    if fma is not None:
        fma = [str(x or "").strip().upper() for x in fma]
        fma = fma if all(ARMED_OK.match(x) for x in fma) else None
    return {"ias": num("ias", 40, 400), "selected_speed": num("selected_speed", 100, 340),
            "selected_altitude": num("selected_altitude", 0, 45000, 100),
            "altitude": num("altitude", -1000, 45000), "fma": fma}


async def read_frame_spark(frame: Image.Image, client: httpx.AsyncClient | None = None) -> tuple[PanelReading, dict]:
    """Read one full-screen frame. Returns the reading and the call's usage (for cost)."""
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    buf = io.BytesIO()
    frame.crop(CALIBRATION["pfd"]).convert("RGB").save(buf, "JPEG", quality=90)
    body = {"model": MODEL, "max_tokens": 1500, "reasoning_effort": "minimal", "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}]}]}
    own = client is None
    client = client or httpx.AsyncClient(trust_env=False, timeout=60)
    try:
        for attempt in range(3):
            r = await client.post(os.environ.get("META_API_BASE", "https://api.meta.ai/v1") + "/chat/completions", json=body,
                                  headers={"Authorization": "Bearer " + os.environ["MODEL_API_KEY"]})
            if r.status_code == 200:
                break
            await asyncio.sleep(2 * (attempt + 1))
        r.raise_for_status()
    finally:
        if own:
            await client.aclose()
    d = r.json()
    text = d["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        raw = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        raw = {}
    v = _plausible(raw)
    reading = PanelReading(ias=v["ias"], selected_speed=v["selected_speed"], selected_altitude=v["selected_altitude"],
                           altitude=v["altitude"], altitude_method="spark" if v["altitude"] is not None else "",
                           fma=v["fma"] or ["", "", ""])
    reading.raw = {"spark": text}
    return reading, d.get("usage", {})
