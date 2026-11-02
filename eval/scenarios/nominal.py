"""False-alarm test: 15 minutes of a correctly flown approach, Boeing 737-800 near Portland.

Everything a scenario fault would break is done right here, many times over: six ATC descent
clearances, each read back correctly and set correctly; heading and speed changes; flap callouts
each followed by the matching selection; a descent through 2,500 ft (where the autothrottle watch
switches on) with the autothrottle in speed. The racetrack keeps the aircraft inside the demo area.
Every line is spoken and heard through Muse Voice Transcribe, so misheard speech is part of the
test. Any callout is a false alarm. Pass: none.

The panel camera looks at the PFD (altitude, MCP, speed, thrust mode); the flap gauge is not in
this view, so the flap callouts test the speech side only (a callout with no gauge reading must
stay silent).

Run with X-Plane 12 open:  uv run --group eval python eval/scenarios/nominal.py [--speak] [--minutes 15]
"""
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brain.live import Crosscheck  # noqa: E402
from bridge.xplane import XPlane, bring_to_front  # noqa: E402
from eval.audio.make_radio_set import radio  # noqa: E402
from eval.scoring import score  # noqa: E402
from senses.speech import transcribe  # noqa: E402
from senses.speech_spark import read_line  # noqa: E402
from voice.crew import CREW, say  # noqa: E402

VIEW = {"pilots_head_the": -22.0, "pilots_head_psi": 0.0, "field_of_view_deg": 50.0}
FLIGHT = {
    "aircraft": {"path": "Aircraft/Laminar Research/Boeing 737-800/b738.acf"},
    "lle_air_start": {"latitude": 45.62, "longitude": -122.75, "elevation_in_meters": 2134,
                      "heading_true": 300, "speed_in_meters_per_second": 113},
    "engine_status": {"all_engines": {"running": True}},
    "local_time": {"day_of_year": 180, "time_in_24_hours": 12.0},
}
OWN = "421"
WORDS = {7000: "seven thousand", 6000: "six thousand", 5000: "five thousand", 4000: "four thousand",
         3000: "three thousand", 2500: "two thousand five hundred", 2000: "two thousand"}
FLAPS_WORDS = {1: "one", 5: "five", 10: "ten", 15: "fifteen"}
RATIO = {0: 0.0, 1: 0.125, 5: 0.375, 10: 0.5, 15: 0.625}
# (seconds into the run, event): "alt" = clearance + readback + PF sets it; "hdg" = ATC heading
# + readback + PF sets it; "spd" = PF sets MCP speed; "flaps" = PF calls, PM selects
PLAN = [(20, "alt", 6000), (90, "hdg", 120), (150, "alt", 5000), (230, "spd", 210), (290, "hdg", 300),
        (330, "alt", 4000), (400, "flaps", 1), (450, "alt", 3000), (510, "spd", 190), (530, "flaps", 5),
        (600, "hdg", 120), (640, "alt", 2000), (720, "spd", 170), (740, "flaps", 10), (800, "hdg", 300),
        (840, "flaps", 15)]
SET_AFTER_S = 3  # the PF dials the MCP this long after the readback is heard
TRUTH = {"alt": "sim/cockpit2/gauges/indicators/altitude_ft_pilot",
         "sel_alt": "sim/cockpit2/autopilot/altitude_dial_ft",
         "ias": "sim/cockpit2/gauges/indicators/airspeed_kts_pilot",
         "sel": "sim/cockpit2/autopilot/airspeed_dial_kts",
         "at": "sim/cockpit2/autopilot/autothrottle_enabled",
         "deploy": "sim/cockpit2/controls/flap_handle_deploy_ratio", "paused": "sim/time/paused"}


def play(pcm: np.ndarray, sr: int) -> None:
    path = pathlib.Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(path, pcm, sr)
    subprocess.Popen(["afplay", str(path)])


def lines_for(kind: str, value: int) -> list[tuple[str, str]]:
    if kind == "alt":
        return [("atc", f"alaska four two one descend and maintain {WORDS[value]}"),
                ("pm", f"descend and maintain {WORDS[value]} alaska four two one")]
    if kind == "hdg":
        spoken = " ".join("zero one two three four five six seven eight nine".split()[int(d)] for d in f"{value:03d}")
        return [("atc", f"alaska four two one turn {'left' if value == 120 else 'right'} heading {spoken}"),
                ("pm", f"{'left' if value == 120 else 'right'} heading {spoken} alaska four two one")]
    if kind == "flaps":
        return [("pf", f"flaps {FLAPS_WORDS[value]}")]
    return []


def camera_check_or_stop(out) -> dict:
    """SAM 3.1 must find the primary flight display where the readers expect it, or the run stops:
    a camera that has moved, or a window in front of the simulator, would make every reading wrong."""
    from brain.live import capture_screen
    from senses.panel import CALIBRATION
    from senses.sam31 import camera_check
    r = camera_check(capture_screen(), {"pfd": CALIBRATION["pfd"]})
    (out / "camera_check.json").write_text(json.dumps(r, indent=1))
    if not r["ok"]:
        raise SystemExit(f"camera check failed: SAM 3.1 did not find the PFD where expected ({r['matches']})")
    return r


async def main(speak: bool, minutes: float) -> int:
    from voice.tts import synth
    run_s = minutes * 60
    plan = [p for p in PLAN if p[0] < run_s - 30]
    audio = {}
    for _, kind, value in plan:  # synthesise everything first: synthesis must not delay the timeline
        for who, text in lines_for(kind, value):
            if who == "pf":
                audio[text] = say(text, CREW)
            else:
                pcm, sr = synth(text)
                assert sr == 16000, sr
                audio[text] = radio(pcm) if who == "atc" else pcm * 0.9 / (np.abs(pcm).max() + 1e-9)
    name = "nominal"
    out = pathlib.Path(os.environ.get("CROSSCHECK_OUT", ROOT / "replay")) / name
    out.mkdir(parents=True, exist_ok=True)

    xp = XPlane()
    await xp.ensure_running()
    await xp.start_flight(FLIGHT)
    await xp.wait_until_loaded()
    await asyncio.sleep(4)
    await xp.set("sim/cockpit2/autopilot/altitude_dial_ft", 7000.0)
    await xp.set("sim/cockpit2/autopilot/airspeed_dial_kts", 220.0)
    await xp.set("sim/cockpit2/autopilot/heading_dial_deg_mag_pilot", 285.0)
    for c in ("cmd_a_press", "speed_press", "hdg_sel_press", "alt_hld_press"):
        await xp.command("laminar/B738/autopilot/" + c)
        await asyncio.sleep(0.8)
    for k, v in VIEW.items():
        await xp.set("sim/graphics/view/" + k, v)
    await asyncio.sleep(8)

    say_cc = None
    if speak:
        from voice.tts import speak as say_cc
    cc = Crosscheck(out, speak=say_cc, frames=True, explain=True)
    heard, truth_rows = [], []
    bring_to_front()
    cam = camera_check_or_stop(out)  # SAM 3.1 pre-flight check of the panel camera
    t0 = time.time()
    worker = threading.Thread(target=cc.run, args=(run_s, t0), daemon=True)
    worker.start()

    async def sample_truth():
        while worker.is_alive():
            truth_rows.append({"t": time.time(), **{k: await xp.get(n) for k, n in TRUTH.items()}})
            await asyncio.sleep(1.0)
    sampler = asyncio.create_task(sample_truth())

    async def transmit(who: str, text: str):
        t_start = time.time()
        if speak:
            play(audio[text], 16000)
        r = await transcribe(audio[text])
        t = time.time()
        facts, spark = await read_line(r["text"], who, OWN)
        heard.append({"speaker": who, "spoken": text, "transcript": r["text"], "audio_s": r["audio_s"],
                      "s_from_start": round(t - t_start, 2), "facts": [(f.key, f.value) for f in facts], "spark": spark})
        cc.event("heard", t, speaker=who, transcript=r["text"], spoken=text)
        for f in facts:
            cc.said(f, transcript=r["text"])

    for at_s, kind, value in plan:
        await asyncio.sleep(max(0.0, t0 + at_s - time.time()))
        for who, text in lines_for(kind, value):
            await transmit(who, text)
            await asyncio.sleep(1.0)
        if kind in ("alt", "hdg", "spd"):
            await asyncio.sleep(SET_AFTER_S)
        t = time.time()
        if kind == "alt":
            await xp.set("sim/cockpit2/autopilot/altitude_dial_ft", float(value))
            await xp.command("laminar/B738/autopilot/lvl_chg_press")
            cc.event("action", t, line=f"PF sets MCP {value} and descends")
        elif kind == "hdg":
            await xp.set("sim/cockpit2/autopilot/heading_dial_deg_mag_pilot", float(value))
            cc.event("action", t, line=f"PF sets heading {value:03d}")
        elif kind == "spd":
            await xp.set("sim/cockpit2/autopilot/airspeed_dial_kts", float(value))
            cc.event("action", t, line=f"PF sets MCP speed {value}")
        elif kind == "flaps":
            await xp.set("sim/cockpit2/controls/flap_ratio", RATIO[value])
            cc.event("action", t, line=f"PM selects flaps {value}")
    while worker.is_alive():
        await asyncio.sleep(1.0)
    await sampler
    await xp.park()
    await xp.close()

    with (out / "truth.jsonl").open("w") as f:
        for row in truth_rows:
            f.write(json.dumps(row) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "scenario": name, "scorer": "nominal", "panel_reader": "apple-vision", "aircraft": "B738", "area": "Portland",
        "t_start": t0, "run_s": run_s, "plan": plan,
        "speech": "Muse Voice Transcribe, live (MMS-TTS ATC radio-filtered and readbacks; macOS voice for flap callouts)",
        "heard": heard, "camera_check": {"model": cam["model"], **cam["matches"]}, "recorded": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=1))
    result = score(out)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    minutes = float(sys.argv[sys.argv.index("--minutes") + 1]) if "--minutes" in sys.argv else 15.0
    sys.exit(asyncio.run(main(speak="--speak" in sys.argv, minutes=minutes)))
