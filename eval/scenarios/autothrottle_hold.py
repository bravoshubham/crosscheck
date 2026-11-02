"""Scenario 3: autothrottle not in speed on final (the Asiana 214 pattern), Boeing 737-800.

On a 2,500 ft approach to Portland 10R with the autopilot holding altitude and the autothrottle
holding 145 kt, Crosscheck watches for QUIET_S seconds (any callout here is a false alarm). Then
the autothrottle drops out of speed control (autothrottle_enabled 0 = armed / HOLD) and the
throttles go to idle, so speed decays. Crosscheck reads everything from the screen; the X-Plane
truth log is used only for scoring.

Pass: no callout before the fault; "Speed. Autothrottle not in speed." before true IAS falls
10 kt below the selected speed.

Run with X-Plane 12 open:  uv run python eval/scenarios/autothrottle_hold.py [--silent]
"""
import asyncio
import json
import os
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brain.live import Crosscheck  # noqa: E402
from bridge.xplane import XPlane, bring_to_front  # noqa: E402
from eval.scoring import score  # noqa: E402

TARGET_KT = 145.0
QUIET_S, FAULT_S = 25, 45
VIEW = {"pilots_head_the": -22.0, "pilots_head_psi": 0.0, "field_of_view_deg": 50.0}
FLIGHT = {  # 8 NM north-west of KPDX 10R on the extended centreline (runway 119 deg true)
    "aircraft": {"path": "Aircraft/Laminar Research/Boeing 737-800/b738.acf"},
    "lle_air_start": {"latitude": 45.660, "longitude": -122.788, "elevation_in_meters": 762,
                      "heading_true": 119, "speed_in_meters_per_second": 75},
    "engine_status": {"all_engines": {"running": True}},
    "local_time": {"day_of_year": 180, "time_in_24_hours": 12.0},
}
TRUTH = {"ias": "sim/cockpit2/gauges/indicators/airspeed_kts_pilot",
         "alt": "sim/cockpit2/gauges/indicators/altitude_ft_pilot",
         "at": "sim/cockpit2/autopilot/autothrottle_enabled",
         "sel": "sim/cockpit2/autopilot/airspeed_dial_kts",
         "paused": "sim/time/paused"}


async def truth_logger(xp: XPlane, path: pathlib.Path, stop: asyncio.Event) -> None:
    with path.open("w") as f:
        async for ts, vals in xp.subscribe(list(TRUTH.values())):
            f.write(json.dumps({"t": time.time(), **{k: vals.get(n) for k, n in TRUTH.items()}}) + "\n")
            if stop.is_set():
                return


async def main(speak: bool, reader: str = "apple") -> int:
    name = "autothrottle_hold" + ("_spark" if reader == "spark" else "")
    out = pathlib.Path(os.environ.get("CROSSCHECK_OUT", ROOT / "replay")) / name  # eval/repeat.py sets it
    out.mkdir(parents=True, exist_ok=True)
    xp = XPlane()
    await xp.ensure_running()
    await xp.start_flight(FLIGHT)
    await xp.wait_until_loaded()
    await asyncio.sleep(4)
    await xp.set("sim/cockpit2/controls/gear_handle_down", 1)
    await xp.set("sim/cockpit2/controls/flap_handle_request_ratio", 0.75)  # flaps 25
    await xp.set("sim/cockpit2/autopilot/airspeed_dial_kts", TARGET_KT)
    await xp.set("sim/cockpit2/autopilot/altitude_dial_ft", 2500.0)
    await xp.set("sim/cockpit2/autopilot/heading_dial_deg_mag_pilot", 104.0)
    for c in ("cmd_a_press", "speed_press", "hdg_sel_press", "alt_hld_press"):
        await xp.command("laminar/B738/autopilot/" + c)
        await asyncio.sleep(0.8)
    for k, v in VIEW.items():
        await xp.set("sim/graphics/view/" + k, v)
    await asyncio.sleep(8)  # let speed settle on the bug

    stop = asyncio.Event()
    logger = asyncio.create_task(truth_logger(xp, out / "truth.jsonl", stop))
    say = None
    if speak:
        from voice.tts import speak as say
    cc = Crosscheck(out, speak=say, frames=True, reader=reader)
    bring_to_front()  # the panel reader captures the screen
    t0 = time.time()
    worker = threading.Thread(target=cc.run, args=(QUIET_S + FAULT_S, t0), daemon=True)
    worker.start()
    await asyncio.sleep(QUIET_S)
    t_fault = time.time()
    await xp.set("sim/cockpit2/autopilot/autothrottle_enabled", 0)
    await xp.set("sim/cockpit2/engine/actuators/throttle_ratio_all", 0.0)
    while worker.is_alive():
        await asyncio.sleep(0.5)
    stop.set()
    logger.cancel()  # it waits on the next update, which never comes if the sim is paused
    try:
        await logger
    except asyncio.CancelledError:
        pass
    await xp.park()
    await xp.close()

    (out / "meta.json").write_text(json.dumps({
        "scenario": name, "scorer": "autothrottle_hold", "panel_reader": "muse-spark" if reader == "spark" else "apple-vision", "aircraft": "B738", "airport": "KPDX 10R",
        "t_start": t0, "t_fault": t_fault, "target_kt": TARGET_KT, "quiet_s": QUIET_S, "fault_s": FAULT_S,
        "speech": "none", "recorded": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=1))
    result = score(out)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(speak="--silent" not in sys.argv, reader="spark" if "--spark" in sys.argv else "apple")))
