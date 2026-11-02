"""T1: read the 737 primary flight display from the screen, scored against X-Plane truth.

This run uses the spec's fallback reader (Muse Glimmer vision on a fixed PFD crop) and
screen capture instead of the iPhone camera; SAM 3.1 calibration replaces the fixed crop
once the Meta API key is set.

Pass (spec targets): speed +/-2 kt, altitude +/-20 ft, heading +/-2 deg.

Run with X-Plane flying the 737 and Ollama up:  uv run python tests/day_one/t1_panel.py
"""
import asyncio
import base64
import io
import json
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
import time

import httpx
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from bridge.xplane import XPlane  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[2] / "eval" / "results" / "day_one"
MODEL = "muse-glimmer:30b"
SAMPLES = 20
PFD_CROP = (1340, 1330, 2150, 2120)  # full-res screen pixels for head pitch -22 deg, FOV 50 deg
VIEW = {"pilots_head_the": -22.0, "pilots_head_psi": 0.0, "field_of_view_deg": 50.0}

TRUTH = {"ias": "sim/cockpit2/gauges/indicators/airspeed_kts_pilot",
         "alt": "sim/cockpit2/gauges/indicators/altitude_ft_pilot",
         "hdg": "sim/cockpit2/gauges/indicators/heading_AHARS_deg_mag_pilot"}
MODES = ["autothrottle_enabled", "heading_status", "altitude_hold_status", "speed_status", "vvi_status"]

SCHEMA = {"type": "object", "properties": {
    "airspeed_kt": {"type": "integer"},
    "altitude_thousands_and_hundreds": {"type": "integer", "description": "large digits in the altitude box, e.g. 92"},
    "altitude_last_two": {"type": "integer", "description": "the rolling two digits at the centre line, e.g. 80"},
    "fma_thrust": {"type": "string"}, "fma_roll": {"type": "string"}, "fma_pitch": {"type": "string"}},
    "required": ["airspeed_kt", "altitude_thousands_and_hundreds", "altitude_last_two", "fma_thrust", "fma_roll", "fma_pitch"]}
PROMPT = ("This is a Boeing 737 primary flight display. Read the CURRENT values, not the magenta selected values. "
          "Airspeed: the number in the white box on the left tape. Altitude box on the right tape: give the large "
          "digits and the smaller rolling last two digits (the pair on the centre line) separately. Also read the "
          "three green mode boxes across the top, left to right. Reply with JSON only.")
COMPASS_CROP = (140, 620, 670, 790)  # within the PFD crop
HDG_SCHEMA = {"type": "object", "properties": {"heading_deg": {"type": "integer"}}, "required": ["heading_deg"]}
HDG_PROMPT = ("Heading scale of a Boeing 737 display. White labels are tens of degrees (18 = 180, 21 = 210); "
              "long ticks every 10 degrees, short ticks every 5. The aircraft's heading is where the white "
              "triangle and the vertical white line at the top centre cross the scale; interpolate between ticks. "
              "Ignore the magenta number and magenta marks: they are the selected heading, not the current one. "
              "Reply with JSON only.")


def capture() -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        subprocess.run(["screencapture", "-x", f.name], check=True)
        return Image.open(f.name).crop(PFD_CROP).copy()


async def ask(http: httpx.AsyncClient, img: Image.Image, prompt: str, schema: dict) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = await http.post("http://127.0.0.1:11434/api/chat", json={
        "model": MODEL, "stream": False, "think": False, "format": schema, "keep_alive": "30m",
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(buf.getvalue()).decode()]}]})
    raw = r.json()["message"]["content"]
    return json.loads(re.sub(r"<\|[a-z_]+\|>\s*$", "", raw.strip()))


async def read_pfd(http: httpx.AsyncClient, img: Image.Image) -> tuple[dict, float]:
    t = time.monotonic()
    read = await ask(http, img, PROMPT, SCHEMA)
    compass = img.crop(COMPASS_CROP)
    compass = compass.resize((compass.width * 2, compass.height * 2))
    read["heading_deg"] = (await ask(http, compass, HDG_PROMPT, HDG_SCHEMA))["heading_deg"] % 360
    read["altitude_ft"] = read["altitude_thousands_and_hundreds"] * 100 + read["altitude_last_two"]
    return read, time.monotonic() - t


def heading_err(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


async def truth(xp: XPlane) -> dict:
    return {k: await xp.get(n) for k, n in TRUTH.items()}


async def main() -> int:
    xp = XPlane()
    http = httpx.AsyncClient(timeout=120)
    rows = []
    try:
        for k, v in VIEW.items():
            await xp.set("sim/graphics/view/" + k, v)
        # Make the numbers move: descend to 7,000 ft in level change, turn to 240.
        await xp.set("sim/cockpit2/autopilot/altitude_dial_ft", 7000.0)
        await xp.set("sim/cockpit2/autopilot/heading_dial_deg_mag_pilot", 240.0)
        await xp.command("laminar/B738/autopilot/lvl_chg_press")
        await asyncio.sleep(3)
        for i in range(SAMPLES):
            before = await truth(xp)
            img = capture()
            after = await truth(xp)
            modes = {m: await xp.get("sim/cockpit2/autopilot/" + m) for m in MODES}
            read, latency = await read_pfd(http, img)
            t = {k: (before[k] + after[k]) / 2 for k in before}
            t["hdg"] = after["hdg"] if heading_err(before["hdg"], after["hdg"]) > 10 else t["hdg"]
            row = {"i": i, "truth": {k: round(v, 1) for k, v in t.items()},
                   "change_during_capture": {k: round(abs(after[k] - before[k]), 1) for k in before},
                   "read": read, "latency_s": round(latency, 2), "modes": modes,
                   "err": {"ias": abs(read["airspeed_kt"] - t["ias"]),
                           "alt": abs(read["altitude_ft"] - t["alt"]),
                           "hdg": heading_err(read["heading_deg"], t["hdg"])}}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("i", "truth", "read", "err", "latency_s")}))
            if i == 0:
                OUT.mkdir(parents=True, exist_ok=True)
                img.save(OUT / "t1_pfd_sample.png")
    finally:
        await http.aclose()
        await xp.close()

    def stats(key: str, tol: float) -> dict:
        errs = [r["err"][key] for r in rows]
        return {"median": round(statistics.median(errs), 1), "max": round(max(errs), 1),
                "within_target": f"{sum(e <= tol for e in errs)}/{len(errs)}", "target": tol}
    summary = {"reader": f"{MODEL} vision, fixed crop + 2x compass crop, split altitude digits, screen capture", "samples": len(rows),
               "ias_kt": stats("ias", 2), "alt_ft": stats("alt", 20), "hdg_deg": stats("hdg", 2),
               "latency_s_median": round(statistics.median(r["latency_s"] for r in rows), 2)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t1_rows.json").write_text(json.dumps(rows, indent=1))
    (OUT / "t1_results.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
