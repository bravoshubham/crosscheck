"""Record PFD frames with X-Plane truth for offline T1 work (record once, replay many times).

Flies the 737 through a turn plus a climb or descent and saves, per frame, the PFD crop and
the truth read just before and after the capture.

Run with X-Plane flying the 737:  uv run python tests/day_one/t1_record.py [frames]
"""
import asyncio
import json
import pathlib
import subprocess
import sys
import tempfile
import time

from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from bridge.xplane import XPlane  # noqa: E402

REC = pathlib.Path(__file__).resolve().parents[2] / "eval" / "recordings" / "t1_pfd"
PFD_CROP = (1340, 1330, 2150, 2120)
VIEW = {"pilots_head_the": -22.0, "pilots_head_psi": 0.0, "field_of_view_deg": 50.0}
TRUTH = {"ias": "sim/cockpit2/gauges/indicators/airspeed_kts_pilot",
         "alt": "sim/cockpit2/gauges/indicators/altitude_ft_pilot",
         "hdg": "sim/cockpit2/gauges/indicators/heading_AHARS_deg_mag_pilot",
         "trk_true": "sim/flightmodel/position/hpath",
         "magvar": "sim/flightmodel/position/magnetic_variation"}
MODES = ["autothrottle_enabled", "heading_status", "altitude_hold_status", "speed_status", "vvi_status",
         "pitch_status", "roll_status"]


async def truth(xp: XPlane) -> dict:
    return {k: await xp.get(n) for k, n in TRUTH.items()}


# Fresh air start next to KPDX: the demo pauses the sim with a "leaving the demo area"
# dialog if the aircraft wanders off, so each recording starts here and turns back.
FLIGHT = {
    "aircraft": {"path": "Aircraft/Laminar Research/Boeing 737-800/b738.acf"},
    "lle_air_start": {"latitude": 45.60, "longitude": -122.70, "elevation_in_meters": 2743,
                      "heading_true": 270, "speed_in_meters_per_second": 110},
    "engine_status": {"all_engines": {"running": True}},
    "local_time": {"day_of_year": 180, "time_in_24_hours": 12.0},
}


async def main(frames: int) -> int:
    xp = XPlane()
    REC.mkdir(parents=True, exist_ok=True)
    meta = []
    try:
        await xp.start_flight(FLIGHT)
        await xp.wait_until_loaded()
        await asyncio.sleep(5)
        await xp.set("sim/cockpit2/autopilot/airspeed_dial_kts", 220.0)
        for c in ("cmd_a_press", "speed_press", "hdg_sel_press", "alt_hld_press"):
            await xp.command("laminar/B738/autopilot/" + c)
            await asyncio.sleep(1)
        for k, v in VIEW.items():
            await xp.set("sim/graphics/view/" + k, v)
        alt = await xp.get(TRUTH["alt"])
        hdg = await xp.get(TRUTH["hdg"])
        target_alt = 11000.0 if alt < 9000 else 7000.0
        await xp.set("sim/cockpit2/autopilot/altitude_dial_ft", target_alt)
        await xp.set("sim/cockpit2/autopilot/heading_dial_deg_mag_pilot", (hdg + 150) % 360)
        await xp.command("laminar/B738/autopilot/hdg_sel_press")
        await xp.command("laminar/B738/autopilot/lvl_chg_press")
        await asyncio.sleep(2)
        for i in range(frames):
            before = await truth(xp)
            with tempfile.NamedTemporaryFile(suffix=".png") as f:
                subprocess.run(["screencapture", "-x", f.name], check=True)
                Image.open(f.name).save(REC / f"{i:03d}.png")  # full frame: the reader crops
            after = await truth(xp)
            modes = {m: await xp.get("sim/cockpit2/autopilot/" + m) for m in MODES}
            if await xp.get("sim/time/paused"):
                raise RuntimeError(f"sim paused at frame {i} (demo-area dialog?); recording discarded")
            meta.append({"i": i, "t": time.time(), "before": before, "after": after, "modes": modes})
            if i == frames // 2:  # second half: turn back and change altitude again
                await xp.set("sim/cockpit2/autopilot/heading_dial_deg_mag_pilot", hdg)
            await asyncio.sleep(0.7)
    finally:
        await xp.close()
    (REC / "truth.json").write_text(json.dumps(meta, indent=1))
    print(f"recorded {len(meta)} frames to {REC}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)))
