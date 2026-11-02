"""Scenarios 1 and 1b: altitude bust and readback error, Boeing 737-800 near Portland.

Level at 9,000 ft. ATC clears "climb and maintain one one thousand". Variants:
  bust      - readback correct, MCP set to 12,000  -> "Cleared one-one thousand, set one-two thousand."
  readback  - readback "one two thousand", MCP 11,000 -> "Readback one-two thousand, cleared one-one thousand."
  correct   - readback and MCP both 11,000          -> silence (false-alarm check)

Both sides are real. Panel: the MCP altitude is read from the screen. Speech: the clearance
and readback are spoken (MMS-TTS; the ATC line radio-filtered), each streamed on its own
channel to Muse Voice Transcribe, and only what comes back is parsed into the ledger
(senses/speech.py). --scripted puts the values in directly instead (no API key needed).
--speak also plays the lines and callouts aloud.

--spark reads the panel with Muse Spark instead of Apple Vision (recorded as altitude_<variant>_spark).

Run with X-Plane 12 open:  uv run python eval/scenarios/altitude_bust.py [bust|readback|correct] [--speak] [--scripted] [--spark]
"""
import asyncio
import json
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
from brain.ledger import Fact  # noqa: E402
from brain.live import Crosscheck  # noqa: E402
from bridge.xplane import XPlane, bring_to_front  # noqa: E402
from eval.audio.make_radio_set import radio  # noqa: E402
from eval.scoring import score  # noqa: E402
from senses.speech import said_facts, transcribe  # noqa: E402

VIEW = {"pilots_head_the": -22.0, "pilots_head_psi": 0.0, "field_of_view_deg": 50.0}
FLIGHT = {
    "aircraft": {"path": "Aircraft/Laminar Research/Boeing 737-800/b738.acf"},
    "lle_air_start": {"latitude": 45.62, "longitude": -122.75, "elevation_in_meters": 2743,
                      "heading_true": 300, "speed_in_meters_per_second": 115},
    "engine_status": {"all_engines": {"running": True}},
    "local_time": {"day_of_year": 180, "time_in_24_hours": 12.0},
}
VARIANTS = {"bust": (11000, 11000, 12000), "readback": (11000, 12000, 11000), "correct": (11000, 11000, 11000)}
EXPECT = {"bust": "clearance_vs_set", "readback": "readback_vs_clearance", "correct": None}
T_CLEAR, T_SET, RUN_S = 12, 26, 62  # readback follows the clearance; the PF sets the MCP after both
OWN = "421"
SPOKEN = {11000: "one one thousand", 12000: "one two thousand"}
TRUTH = {"alt": "sim/cockpit2/gauges/indicators/altitude_ft_pilot",
         "sel_alt": "sim/cockpit2/autopilot/altitude_dial_ft", "paused": "sim/time/paused"}


def play(pcm: np.ndarray, sr: int) -> None:
    path = pathlib.Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(path, pcm, sr)
    subprocess.Popen(["afplay", str(path)])


async def main(variant: str, speak: bool, scripted: bool, reader: str = "apple") -> int:
    cleared, readback, mcp = VARIANTS[variant]
    lines = {"atc": f"alaska four two one climb and maintain {SPOKEN[cleared]}",
             "pm": f"climb and maintain {SPOKEN[readback]} alaska four two one"}
    audio = {}
    if not scripted:
        from voice.tts import synth
        for who, text in lines.items():
            pcm, sr = synth(text)
            assert sr == 16000, sr
            audio[who] = radio(pcm) if who == "atc" else pcm * 0.9 / (np.abs(pcm).max() + 1e-9)
    heard = []
    name = f"altitude_{variant}" + ("_spark" if reader == "spark" else "")
    out = ROOT / "replay" / name
    out.mkdir(parents=True, exist_ok=True)
    xp = XPlane()
    await xp.ensure_running()
    await xp.start_flight(FLIGHT)
    await xp.wait_until_loaded()
    await asyncio.sleep(4)
    await xp.set("sim/cockpit2/autopilot/altitude_dial_ft", 9000.0)
    await xp.set("sim/cockpit2/autopilot/airspeed_dial_kts", 250.0)
    await xp.set("sim/cockpit2/autopilot/heading_dial_deg_mag_pilot", 285.0)
    for c in ("cmd_a_press", "speed_press", "hdg_sel_press", "alt_hld_press"):
        await xp.command("laminar/B738/autopilot/" + c)
        await asyncio.sleep(0.8)
    for k, v in VIEW.items():
        await xp.set("sim/graphics/view/" + k, v)
    await asyncio.sleep(6)

    say = None
    if speak:
        from voice.tts import speak as say
    cc = Crosscheck(out, speak=say, frames=True, reader=reader)
    truth_rows = []
    bring_to_front()  # the panel reader captures the screen
    t0 = time.time()
    worker = threading.Thread(target=cc.run, args=(RUN_S, t0), daemon=True)
    worker.start()

    async def at(seconds: float):
        await asyncio.sleep(max(0.0, t0 + seconds - time.time()))

    async def sample_truth():
        while worker.is_alive():
            truth_rows.append({"t": time.time(), **{k: await xp.get(n) for k, n in TRUTH.items()}})
            await asyncio.sleep(0.5)
    sampler = asyncio.create_task(sample_truth())

    async def transmit(who: str, value: int):
        """One radio or intercom line: speech -> Voice Transcribe -> parsed facts -> ledger."""
        t_start = time.time()
        if scripted:
            key = "clearance.altitude" if who == "atc" else "readback.altitude"
            cc.said(Fact("said", key, value, t_start, who, source="script"), line=lines[who])
            return
        if speak:
            play(audio[who], 16000)
        r = await transcribe(audio[who])
        t = time.time()
        facts = said_facts(r["text"], who, t, OWN)
        heard.append({"speaker": who, "spoken": lines[who], "transcript": r["text"], "audio_s": r["audio_s"],
                      "s_from_start": round(t - t_start, 2), "facts": [(f.key, f.value) for f in facts]})
        cc.event("heard", t, speaker=who, transcript=r["text"], spoken=lines[who])
        for f in facts:
            cc.said(f, transcript=r["text"])

    await at(T_CLEAR)
    await transmit("atc", cleared)
    await asyncio.sleep(1.0)
    await transmit("pm", readback)
    await at(T_SET)
    t_set = time.time()
    await xp.set("sim/cockpit2/autopilot/altitude_dial_ft", float(mcp))
    await xp.command("laminar/B738/autopilot/lvl_chg_press")
    cc.event("action", t_set, line=f"PF sets MCP {mcp} and climbs")
    while worker.is_alive():
        await asyncio.sleep(0.5)
    await sampler
    await xp.park()
    await xp.close()

    with (out / "truth.jsonl").open("w") as f:
        for r in truth_rows:
            f.write(json.dumps(r) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "scenario": name, "scorer": "altitude", "panel_reader": "muse-spark" if reader == "spark" else "apple-vision", "aircraft": "B738", "area": "Portland",
        "variant": variant, "cleared": cleared, "readback": readback, "mcp_set": mcp,
        "expect_rule": EXPECT[variant], "t_start": t0, "t_clear": t0 + T_CLEAR, "t_set": t_set,
        "speech": "scripted (said facts from the scenario script)" if scripted else
                  "Muse Voice Transcribe, live (MMS-TTS lines; ATC radio-filtered), parsed by senses/speech.py",
        "heard": heard,
        "recorded": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=1))
    result = score(out)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(asyncio.run(main(args[0] if args else "bust", speak="--speak" in sys.argv,
                              scripted="--scripted" in sys.argv, reader="spark" if "--spark" in sys.argv else "apple")))
