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
FIELDS = ("ias", "selected_speed", "selected_altitude", "altitude", "baro")
SCALE = 0.75  # of the stacked parts: fewer image tokens, same accuracy on the recorded frames

# The three parts of the PFD the rules use, cut from the calibrated crop and stacked: the mode
# boxes on top, the speed tape and the altitude tape below. Half the image tokens of the whole PFD
# for the same accuracy (24 recorded frames: altitude 20/30, selected altitude 23/24, speed 10/13,
# identical to the whole crop), because the attitude ball in the middle carries no number.
PARTS = {"modes": (140, 0, 580, 110), "speed": (0, 50, 150, 450), "altitude": (570, 50, 760, 450),
         "baro": (570, 615, 780, 715)}  # the altimeter setting under the altitude tape (green "29.92 IN."), with room for the view to move
PARTS_PROMPT = (
    "These are four parts of a Boeing 737 primary flight display (PFD): on top the three flight mode boxes, below on the "
    'left the speed tape, on the right the altitude tape, and along the bottom, enlarged, the altimeter setting. Reply with JSON only: {"r": '
    '"speed/selected speed/selected altitude/altitude/altimeter", "m": [thrust mode, roll mode, pitch mode]}, for example '
    '{"r": "192/190/9000/8900/29.92", "m": ["MCP SPD", "HDG SEL", "ALT HOLD"]}. speed = the number in the box on the speed '
    "tape; selected speed = the magenta number above the speed tape; selected altitude = the magenta number above the "
    "altitude tape; altitude = the number in the box on the altitude tape, including the rolling last two digits; "
    "altimeter = the green setting along the bottom, in inches with two decimals, or STD. " + MODES_HELP)


def parts(pfd: Image.Image) -> Image.Image:
    top, sp, al, ba = (pfd.crop(PARTS[k]) for k in ("modes", "speed", "altitude", "baro"))
    # the altimeter setting is small type: at 1.5 times its size, along the bottom (at its own size,
    # scaled with the rest, Glimmer could read it on 32 of 49 recorded frames)
    ba = ba.resize((ba.width * 3 // 2, ba.height * 3 // 2), Image.LANCZOS)
    tapes = max(sp.height, al.height)
    out = Image.new("RGB", (max(top.width, sp.width + 10 + al.width, ba.width), top.height + 10 + tapes + 10 + ba.height), "black")
    out.paste(top, (0, 0))
    out.paste(sp, (0, top.height + 10))
    out.paste(al, (sp.width + 10, top.height + 10))
    out.paste(ba, (0, top.height + 10 + tapes + 10))
    return out.resize((int(out.width * SCALE), int(out.height * SCALE)), Image.LANCZOS)


# The fast lane, the mode boxes over the speed tape: the reader on final until the final lane below
# replaced it (two reads in three, too slow for the autothrottle alert); kept as the comparison in
# eval/panel_final_lane.py.
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


# The final lane, for final approach on a GPU the simulator is also using: the smallest image that
# still carries the autothrottle alert, the selected speed and the speed box on the left, the thrust
# and pitch mode boxes on the right. The answer is one JSON string held to a pattern, about a third
# of the reply tokens of separate JSON fields (with the simulator running each token costs ~45 ms):
# numbers 2-3 digits or "?" (the speed box's rolling digit once came back as "1232"), and each mode
# box only its own column's modes, so a pitch mode cannot be read into the thrust box. The prompt
# goes as the system message (senses/onboard.py). With X-Plane flying: 2.15 s a read, against 3.0 s
# for the fast lane and 3.4 s for the whole set of parts.
FINAL = {"selected_speed": (30, 55, 140, 115), "speed": (15, 335, 145, 445), "thrust": (140, 0, 290, 110), "pitch": (425, 0, 580, 110)}
THRUST_MODES = ["MCP SPD", "FMC SPD", "THR HLD", "RETARD", "N1", "GA", "ARM"]
PITCH_MODES = ["MCP SPD", "TO/GA", "CWS P", "ALT HOLD", "ALT ACQ", "VNAV PTH", "VNAV SPD", "VNAV ALT", "V/S", "G/S", "FLARE"]
_NUM, _ONE = "([0-9]{2,3}|[?])", lambda modes: "(" + "|".join([""] + [re.escape(m).replace("\\ ", " ") for m in modes]) + ")"
FINAL_SCHEMA = {"type": "string", "pattern": f"^{_NUM};{_NUM};{_ONE(THRUST_MODES)};{_ONE(PITCH_MODES)}$"}
FINAL_PROMPT = (
    "Four parts of a Boeing 737 primary flight display (PFD): on the left the magenta selected speed above the speed box, "
    'on the right the thrust mode box and the pitch mode box. Reply with one JSON string only: "selected speed;speed;thrust '
    'mode;pitch mode", for example "142;141;MCP SPD;G/S". speed = the number in the speed box. A mode is the green text '
    'in its box, nothing for a box with no green text: the thrust box is empty when the autothrottle is off '
    '("142;141;;ALT HOLD"). ? for a number you cannot read.')


def final_parts(pfd: Image.Image) -> Image.Image:
    sel, sp, thr, pit = (pfd.crop(FINAL[k]) for k in ("selected_speed", "speed", "thrust", "pitch"))
    out = Image.new("RGB", (sp.width + 10 + thr.width + 10 + pit.width, sel.height + 10 + sp.height), "black")
    out.paste(sel, (0, 0))
    out.paste(sp, (0, sel.height + 10))
    out.paste(thr, (sp.width + 10, 0))
    out.paste(pit, (sp.width + 10 + thr.width + 10, 0))
    return out


async def read_frame_glimmer_final(frame: Image.Image, client: httpx.AsyncClient | None = None) -> tuple[PanelReading, dict]:
    """Speed, selected speed, thrust and pitch mode; the roll mode is not read (None), so it is
    neither a mode nor a blank box. A reply that does not parse reads nothing."""
    text, usage = await glimmer_json(FINAL_PROMPT, FINAL_SCHEMA, jpeg_b64(final_parts(frame.crop(CALIBRATION["pfd"]))),
                                     num_predict=30, client=client, system=True)
    m = re.search(r'".*"', text, re.S)
    try:
        v = str(json.loads(m.group())).split(";") if m else []
    except json.JSONDecodeError:
        v = []
    ok = len(v) == 4
    num = lambda i: v[i] if ok and v[i].isdigit() else None
    p = _plausible({"ias": num(1), "selected_speed": num(0), "selected_altitude": None, "altitude": None,
                    "fma": [v[2], "", v[3]] if ok else None})
    fma = [p["fma"][0], None, p["fma"][2]] if p["fma"] else None
    reading = PanelReading(ias=p["ias"], selected_speed=p["selected_speed"], fma=fma or ["", "", ""])
    reading.raw = {"glimmer_final": text}
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
    if "baro" in numbers and len(v) == len(numbers) - 1:  # the altimeter left out: the other numbers still count
        numbers = tuple(k for k in numbers if k != "baro")
    if len(v) == len(numbers):
        out.update({k: (x if x.isdigit() else None) for k, x in zip(numbers, v) if k != "baro"})
        if "baro" in numbers:  # "30.12" -> 3012, "STD" -> 2992 (standard pressure)
            x = v[numbers.index("baro")].upper()
            b = 2992 if x == "STD" else round(float(x) * 100) if re.fullmatch(r"\d{2}\.\d{2}", x) else None
            out["baro"] = b if b is not None and 2750 <= b <= 3150 else None
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
    p = parse(text)
    v = _plausible(p)
    reading = PanelReading(ias=v["ias"], selected_speed=v["selected_speed"], selected_altitude=v["selected_altitude"],
                           altitude=v["altitude"], altitude_method="glimmer" if v["altitude"] is not None else "",
                           fma=v["fma"] or ["", "", ""], baro=p.get("baro"))
    reading.raw = {"glimmer": text}
    return reading, usage
