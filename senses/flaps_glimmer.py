"""Flap position from the 737's analog flap gauge, read by the local Muse Glimmer (on board).

The same gauge crop as senses/flaps_spark.py, but a different question: Glimmer is asked which
marking the needle points at, from a fixed list with each marking's place on the dial, not for a
clock position (asked for the clock position it was right on 8 of 20 shipped frames). The list
alone read a needle parked at 5 as UP or 2 in live on-board runs, and a jump to UP kept the settle
check from ever settling. With each marking's place on the dial in the question: 19 of 20 shipped
frames at 2x zoom (eval/panel_onboard.py), and a 2 for a 5 is within the settle check's spread.
3x zoom read better (22 of 22 live frames) but took 6 s a read, too slow for the settle check to
see five readings before a run ended. The marking's own clock position goes into the ledger, so
the rules' settle check (brain/rules.py) is unchanged: a needle in travel reads as changing marks.
"""
import json
import re

import httpx
from PIL import Image

from senses.flaps_spark import DIAL, FLAP_GAUGE
from senses.onboard import glimmer_json
from senses.panel import PanelReading
from senses.panel_glimmer import jpeg_b64

MARKS = {"UP": 0, "1": 1, "2": 2, "5": 5, "10": 10, "15": 15, "25": 25, "30": 30, "40": 40}
PROMPT = ("Boeing 737 flap position indicator: a round dial with a white needle from the centre. The markings go clockwise from "
          "the left: UP at the 9 o'clock position (flaps retracted), 1 at 10 o'clock, 2 at 11 o'clock, 5 at 12 o'clock (the top), "
          "10 at 1 o'clock, 15 at 3 o'clock (the right), 25 at 4, 30 at 5, 40 at 6 o'clock (the bottom). Which marking is the "
          'needle pointing at? JSON only: {"mark": "UP" | "1" | "2" | "5" | "10" | "15" | "25" | "30" | "40"}')
ZOOM = 2  # 3x reads better but at 6 s a read the settle check runs out of time
SCHEMA = {"type": "object", "properties": {"mark": {"type": "string", "enum": list(MARKS)}}, "required": ["mark"]}


async def read_flaps_glimmer(frame: Image.Image, client: httpx.AsyncClient | None = None) -> tuple[PanelReading, dict]:
    img = frame.crop(FLAP_GAUGE).convert("RGB")
    img = img.resize((img.width * ZOOM, img.height * ZOOM), Image.LANCZOS)
    text, usage = await glimmer_json(PROMPT, SCHEMA, jpeg_b64(img, 92), num_predict=20, client=client)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        flaps = MARKS.get(str(json.loads(m.group())["mark"])) if m else None
    except (KeyError, TypeError, json.JSONDecodeError):
        flaps = None
    reading = PanelReading(flaps=flaps, flaps_clock=DIAL[flaps] if flaps is not None else None)
    reading.raw = {"glimmer": text}
    return reading, usage
