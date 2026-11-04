"""Panel reader on Muse Glimmer 30B (Meta open weights, vision), running on this computer.

The on-board reader: no network and no Apple framework. The same calibrated PFD crop, prompt and
plausibility filter as senses/panel_spark.py, sent to a local model server (Ollama / llama.cpp
on 127.0.0.1). Thinking is off and the reply is held to a JSON schema, so a frame costs one
short answer. Values outside what a 737 PFD can show are dropped, so a bad read becomes
"unknown", never a wrong fact.
"""
import base64
import io
import json
import re

import httpx
from PIL import Image

from senses.onboard import glimmer_json
from senses.panel import CALIBRATION, FMA_VOCAB, PanelReading
from senses.panel_spark import _plausible

# The numbers as one short string, not a JSON object per value: the reply is most of a read's time
# on a GPU the simulator is also using (recorded frames, X-Plane running: 24 tokens and 4.4 s a read
# against 49 tokens and 7.2 s, same accuracy). The three modes as a list of exactly three, each one
# of the 737's mode names or "" for an empty box: in a plain string Glimmer sometimes left out the
# empty thrust box ("122/141/HDG SEL/ALT HOLD"), which made the whole reading unreadable.
MODE = {"type": "string", "enum": [""] + FMA_VOCAB}
SCHEMA = {"type": "object", "properties": {"r": {"type": "string"},
          "m": {"type": "array", "items": MODE, "minItems": 3, "maxItems": 3}}, "required": ["r", "m"]}
MODES_HELP = ('"m" = the green text in each of the three mode boxes, left to right (thrust, roll, pitch), "" for a box with '
              "no green text: the left box is empty when the autothrottle is off. ? for a number you cannot read.")
FIELDS = ("ias", "selected_speed", "selected_altitude", "altitude")
SCALE = 0.75  # of the stacked parts: fewer image tokens, same accuracy on the recorded frames

# The three parts of the PFD the rules use, cut from the calibrated crop and stacked: the mode
# boxes on top, the speed tape and the altitude tape below. Half the image tokens of the whole PFD
# for the same accuracy (24 recorded frames: altitude 20/30, selected altitude 23/24, speed 10/13,
# identical to the whole crop), because the attitude ball in the middle carries no number.
PARTS = {"modes": (140, 0, 580, 110), "speed": (0, 50, 150, 450), "altitude": (570, 50, 760, 450)}
PARTS_PROMPT = (
    "These are three parts of a Boeing 737 primary flight display (PFD): on top the three flight mode boxes, below on the "
    'left the speed tape and on the right the altitude tape. Reply with JSON only: {"r": "speed/selected speed/selected '
    'altitude/altitude", "m": [thrust mode, roll mode, pitch mode]}, for example {"r": "192/190/9000/8900", "m": ["MCP SPD", '
    '"HDG SEL", "ALT HOLD"]}. speed = the number in the box on the speed tape; selected speed = the magenta number above the '
    "speed tape; selected altitude = the magenta number above the altitude tape; altitude = the number in the box on the "
    "altitude tape, including the rolling last two digits. " + MODES_HELP)


def parts(pfd: Image.Image) -> Image.Image:
    top, sp, al = (pfd.crop(PARTS[k]) for k in ("modes", "speed", "altitude"))
    out = Image.new("RGB", (max(top.width, sp.width + 10 + al.width), top.height + 10 + sp.height), "black")
    out.paste(top, (0, 0))
    out.paste(sp, (0, top.height + 10))
    out.paste(al, (sp.width + 10, top.height + 10))
    return out.resize((int(out.width * SCALE), int(out.height * SCALE)), Image.LANCZOS)


# The fast lane, for final approach (brain/live.py): only what the autothrottle alert needs, the
# mode boxes over the speed tape, and a shorter answer. Every third read on final is still the
# whole set of parts, so altitude and selected altitude keep arriving.
FAST_PROMPT = (
    "These are two parts of a Boeing 737 primary flight display (PFD): on top the three flight mode boxes, below the speed "
    'tape. Reply with JSON only: {"r": "speed/selected speed", "m": [thrust mode, roll mode, pitch mode]}, for example '
    '{"r": "142/143", "m": ["MCP SPD", "HDG SEL", "ALT HOLD"]}. speed = the number in the box on the speed tape; selected '
    "speed = the magenta number above the speed tape. " + MODES_HELP)


def fast_parts(pfd: Image.Image) -> Image.Image:
    top, sp = pfd.crop(PARTS["modes"]), pfd.crop(PARTS["speed"])
    out = Image.new("RGB", (top.width, top.height + 10 + sp.height), "black")
    out.paste(top, (0, 0))
    out.paste(sp, (0, top.height + 10))
    return out.resize((int(out.width * SCALE), int(out.height * SCALE)), Image.LANCZOS)


async def read_frame_glimmer_fast(frame: Image.Image, client: httpx.AsyncClient | None = None) -> tuple[PanelReading, dict]:
    """Speed, selected speed and the three modes only; the other values stay unknown."""
    text, usage = await glimmer_json(FAST_PROMPT, SCHEMA, jpeg_b64(fast_parts(frame.crop(CALIBRATION["pfd"]))), num_predict=50,
                                     client=client)
    p = _plausible(parse(text, ("ias", "selected_speed")))
    reading = PanelReading(ias=p["ias"], selected_speed=p["selected_speed"], fma=p["fma"] or ["", "", ""])
    reading.raw = {"glimmer_fast": text}
    return reading, usage


def parse(text: str, numbers: tuple = FIELDS) -> dict:
    """{"r": "192/190/9000/8900", "m": ["MCP SPD", "HDG SEL", "ALT HOLD"]} -> the fields
    senses.panel_spark._plausible expects; a number that is not one stays unknown."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group()) if m else {}
        v = [x.strip() for x in str(d.get("r", "")).split("/")]
        modes = d.get("m") if isinstance(d.get("m"), list) and len(d["m"]) == 3 else None
    except json.JSONDecodeError:
        return {}
    out = {k: None for k in FIELDS}
    if len(v) == len(numbers):
        out.update({k: (x if x.isdigit() else None) for k, x in zip(numbers, v)})
    out["fma"] = modes
    return out


def jpeg_b64(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


async def read_frame_glimmer(frame: Image.Image, client: httpx.AsyncClient | None = None) -> tuple[PanelReading, dict]:
    """Read one full-screen frame. Returns the reading and the call's counters."""
    text, usage = await glimmer_json(PARTS_PROMPT, SCHEMA, jpeg_b64(parts(frame.crop(CALIBRATION["pfd"]))), num_predict=70,
                                     client=client)
    v = _plausible(parse(text))
    reading = PanelReading(ias=v["ias"], selected_speed=v["selected_speed"], selected_altitude=v["selected_altitude"],
                           altitude=v["altitude"], altitude_method="glimmer" if v["altitude"] is not None else "",
                           fma=v["fma"] or ["", "", ""])
    reading.raw = {"glimmer": text}
    return reading, usage
