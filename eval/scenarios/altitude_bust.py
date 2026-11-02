"""Scenarios 1 and 1b: altitude bust and readback error, Boeing 737-800 near Portland.

Level at 9,000 ft. ATC clears "climb and maintain one one thousand". Variants:
  bust      - readback correct, MCP set to 12,000  -> "Cleared one-one thousand, set one-two thousand."
  readback  - readback "one two thousand", MCP 11,000 -> "Readback one-two thousand, cleared one-one thousand."
  correct   - readback and MCP both 11,000          -> silence (false-alarm check)

The panel side is real: the MCP altitude is read from the screen. The speech side is scripted
until Muse Voice Transcribe is connected: clearance and readback enter the ledger as "said"
facts with source "script" at the moment the line would be spoken.

Run with X-Plane 12 open:  uv run python eval/scenarios/altitude_bust.py [bust|readback|correct] [--speak]
"""
import asyncio
import json
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brain.ledger import Fact  # noqa: E402
from brain.live import Crosscheck  # noqa: E402
from bridge.xplane import XPlane  # noqa: E402
from eval.scoring import score  # noqa: E402

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
T_CLEAR, T_READBACK, T_SET, RUN_S = 12, 15, 18, 55
TRUTH = {"alt": "sim/cockpit2/gauges/indicators/altitude_ft_pilot",
         "sel_alt": "sim/cockpit2/autopilot/altitude_dial_ft", "paused": "sim/time/paused"}


async def main(variant: str, speak: bool) -> int:
    cleared, readback, mcp = VARIANTS[variant]
    out = ROOT / "replay" / f"altitude_{variant}"
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
    cc = Crosscheck(out, speak=say, frames=True)
    truth_rows = []
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

    await at(T_CLEAR)
    cc.said(Fact("said", "clearance.altitude", cleared, time.time(), "atc", source="script"))
    await at(T_READBACK)
    cc.said(Fact("said", "readback.altitude", readback, time.time(), "pm", source="script"))
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
        "scenario": f"altitude_{variant}", "scorer": "altitude", "aircraft": "B738", "area": "Portland",
        "variant": variant, "cleared": cleared, "readback": readback, "mcp_set": mcp,
        "expect_rule": EXPECT[variant], "t_start": t0, "t_clear": t0 + T_CLEAR, "t_set": t_set,
        "speech": "scripted (said facts from the scenario script)",
        "recorded": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=1))
    result = score(out)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(asyncio.run(main(args[0] if args else "bust", speak="--speak" in sys.argv)))
