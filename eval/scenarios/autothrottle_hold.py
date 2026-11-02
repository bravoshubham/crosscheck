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
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brain.live import Crosscheck  # noqa: E402
from bridge.xplane import XPlane  # noqa: E402

OUT = ROOT / "eval" / "results" / "scenarios" / "autothrottle_hold"
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


async def main(speak: bool) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    xp = XPlane()
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
    logger = asyncio.create_task(truth_logger(xp, OUT / "truth.jsonl", stop))
    say = None
    if speak:
        from voice.tts import speak as say
    cc = Crosscheck(OUT / "events.jsonl", speak=say)
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
    await asyncio.wait_for(logger, 5)
    await xp.close()

    truth = [json.loads(l) for l in (OUT / "truth.jsonl").read_text().splitlines()]
    events = [json.loads(l) for l in (OUT / "events.jsonl").read_text().splitlines()]
    callouts = [e for e in events if e["kind"] == "callout"]
    # subscriptions send only changes: carry the last value forward
    last, rows = {}, []
    for r in truth:
        last.update({k: v for k, v in r.items() if v is not None})
        rows.append(dict(last))
    sel = next((r["sel"] for r in reversed(rows) if r["t"] <= t_fault and "sel" in r), TARGET_KT)
    ias = [(r["t"], r["ias"]) for r in rows if "ias" in r]
    def t_below(margin):
        return next((t for t, v in ias if t > t_fault and v < sel - margin), None)
    first = next((c for c in callouts if c["t"] >= t_fault and c["rule"] == "mode_vs_phase"), None)
    t10 = t_below(10)
    result = {
        "selected_speed_truth": round(sel, 1),
        "false_callouts_before_fault": [c["callout"] for c in callouts if c["t"] < t_fault],
        "callout": first["callout"] if first else None,
        "callout_s_after_fault": round(first["t"] - t_fault, 1) if first else None,
        "truth_below_target_minus_5_s": round(t_below(5) - t_fault, 1) if t_below(5) else None,
        "truth_below_target_minus_10_s": round(t10 - t_fault, 1) if t10 else None,
        "readings": sum(e["kind"] == "reading" for e in events),
        "paused_during_run": any(r.get("paused") for r in truth),
    }
    result["pass"] = (not result["false_callouts_before_fault"] and first is not None
                      and (t10 is None or first["t"] < t10) and not result["paused_during_run"])
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(speak="--silent" not in sys.argv)))
