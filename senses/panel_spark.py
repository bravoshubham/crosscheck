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
    'altitude tape, ft, including the rolling last two digits>, "fma": [<thrust mode>, <roll mode>, <pitch mode>], '
    '"baro": <the green altimeter setting under the altitude tape, inches with two decimals as a string such as "29.92", or "STD">}\n'
    "The fma is the green text in the three boxes along the top; use \"\" for an empty box. "
    "Use null for any value you cannot read."
)
ARMED_OK = re.compile(r"^[A-Z0-9 /]{0,12}$")


def baro_hundredths(v) -> int | None:
    """The altimeter setting as read ("30.12", 30.12 or "STD") in hundredths of an inch; None if it
    is not one an airport could report."""
    if isinstance(v, str) and v.strip().upper() == "STD":
        return 2992
    try:
        b = round(float(str(v).strip()) * 100)
    except (TypeError, ValueError):
        return None
    return b if 2750 <= b <= 3150 else None


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
    d = await _call(body, client)
    text = d["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        raw = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        raw = {}
    v = _plausible(raw)
    reading = PanelReading(ias=v["ias"], selected_speed=v["selected_speed"], selected_altitude=v["selected_altitude"],
                           altitude=v["altitude"], altitude_method="spark" if v["altitude"] is not None else "",
                           fma=v["fma"] or ["", "", ""], baro=baro_hundredths(raw.get("baro")))
    reading.raw = {"spark": text}
    return reading, d.get("usage", {})


async def _call(body: dict, client: httpx.AsyncClient | None = None) -> dict:
    """One Meta Model API chat call, retried twice; returns the response JSON."""
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
    return r.json()


# The final lane on Muse Spark, for final approach (brain/live.py): the same four parts of the PFD as
# Glimmer's final lane (senses/panel_glimmer.py: selected speed, speed box, thrust and pitch mode
# boxes) and a one-line answer in a strict JSON schema, checked against Glimmer's final-lane pattern.
# Recorded frames, 36 reads: median 2.2 s and ~270 prompt + ~190 completion tokens a read, against
# 4.6-5.4 s and 966 + 280-450 for the whole PFD ("minimal" is the least reasoning Spark accepts).
# Saying that the white ARM is not a mode took the median from 3.9 s: Spark deliberated over it.
# Some reads still take 4-7 s; with a read started every second, a quicker read of a newer frame
# overtakes them.
FINAL_PROMPT = (
    "Four parts of a Boeing 737 primary flight display (PFD): on the left the magenta selected speed above the speed box, "
    'on the right the thrust mode box and the pitch mode box. Reply with JSON only: {"r": "selected speed;speed;thrust '
    'mode;pitch mode"}, for example {"r": "142;141;MCP SPD;G/S"}. speed = the number in the speed box. A mode is the green '
    "text in its box, nothing for a box with no green text: white text such as ARM is not a mode, and the thrust box has "
    'no green text when the autothrottle is off ({"r": "142;141;;ALT HOLD"}). ? for a number you cannot read.')
FINAL_FORMAT = {"type": "json_schema", "json_schema": {"name": "final_lane", "strict": True, "schema": {
    "type": "object", "properties": {"r": {"type": "string"}}, "required": ["r"], "additionalProperties": False}}}


async def read_frame_spark_final(frame: Image.Image, client: httpx.AsyncClient | None = None) -> tuple[PanelReading, dict]:
    """Speed, selected speed, thrust and pitch mode; the roll mode is not read (None), so it is
    neither a mode nor a blank box. An answer that does not match the pattern reads nothing."""
    from senses.panel_glimmer import FINAL_SCHEMA, final_parts  # panel_glimmer imports this module
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    buf = io.BytesIO()
    final_parts(frame.crop(CALIBRATION["pfd"])).convert("RGB").save(buf, "JPEG", quality=90)
    body = {"model": MODEL, "max_tokens": 1500, "reasoning_effort": "minimal", "response_format": FINAL_FORMAT,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": FINAL_PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}]}]}
    d = await _call(body, client)
    text = d["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        line = str(json.loads(m.group()).get("r", "")).strip() if m else ""
    except json.JSONDecodeError:
        line = ""
    v = line.split(";") if re.match(FINAL_SCHEMA["pattern"], line) else []
    ok = len(v) == 4
    num = lambda i: v[i] if ok and v[i].isdigit() else None
    p = _plausible({"ias": num(1), "selected_speed": num(0), "selected_altitude": None, "altitude": None,
                    "fma": [v[2], "", v[3]] if ok else None})
    fma = [p["fma"][0], None, p["fma"][2]] if p["fma"] else None
    reading = PanelReading(ias=p["ias"], selected_speed=p["selected_speed"], fma=fma or ["", "", ""])
    reading.raw = {"spark_final": text}
    return reading, d.get("usage", {})
