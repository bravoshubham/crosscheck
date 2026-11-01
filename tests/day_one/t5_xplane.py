"""T5: X-Plane API round trip.

Pass when: truth reads at ~10 Hz; heading, altitude and squawk can be set; an engine
failure can be injected. Also dumps the 737's own datarefs so FMA sources can be found.

Run with X-Plane 12.4+ open:  uv run python tests/day_one/t5_xplane.py
"""
import asyncio
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from bridge.xplane import XPlane  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[2] / "eval" / "results" / "day_one"

# Air start east of Portland (inside the X-Plane 12 demo scenery), 10,000 ft, heading 090.
FLIGHT = {
    "aircraft": {"path": "Aircraft/Laminar Research/Boeing 737-800/b738.acf"},
    "lle_air_start": {"latitude": 45.55, "longitude": -122.95, "elevation_in_meters": 3048,
                      "heading_true": 90, "speed_in_meters_per_second": 130},
    "engine_status": {"all_engines": {"running": True}},
    "local_time": {"day_of_year": 180, "time_in_24_hours": 12.0},
}

TRUTH = ["sim/flightmodel/position/latitude", "sim/flightmodel/position/longitude",
         "sim/cockpit2/gauges/indicators/airspeed_kts_pilot",
         "sim/cockpit2/gauges/indicators/altitude_ft_pilot",
         "sim/cockpit2/gauges/indicators/heading_AHARS_deg_mag_pilot"]


async def main() -> int:
    xp = XPlane()
    results: dict[str, object] = {}
    try:
        t = time.monotonic()
        await xp.start_flight(FLIGHT)
        n = await xp.wait_until_loaded()
        await asyncio.sleep(5)  # let the flight settle
        results["load"] = f"{n} datarefs after {time.monotonic() - t:.1f}s"

        # 1. Truth at 10 Hz
        stamps, last = [], {}
        async def collect():
            async for ts, vals in xp.subscribe(TRUTH):
                stamps.append(ts)
                last.update(vals)
                if ts - stamps[0] >= 10:
                    return
        await asyncio.wait_for(collect(), 30)
        rate = (len(stamps) - 1) / (stamps[-1] - stamps[0])
        results["truth_rate_hz"] = round(rate, 2)
        results["truth_sample"] = {k.rsplit("/", 1)[1]: round(v, 2) for k, v in last.items()}

        # 2. Set heading, altitude, squawk and read back
        writes = {"sim/cockpit2/autopilot/heading_dial_deg_mag_pilot": 180.0,
                  "sim/cockpit2/autopilot/altitude_dial_ft": 12000.0,
                  "sim/cockpit2/radios/actuators/transponder_code": 7700}
        for name, value in writes.items():
            await xp.set(name, value)
        await asyncio.sleep(1)
        results["writes"] = {name.rsplit("/", 1)[1]: {"set": v, "read": await xp.get(name)}
                             for name, v in writes.items()}

        # 3. Inject engine 1 failure (6 = fail now) and watch N1 fall
        n1_before = await xp.get("sim/flightmodel/engine/ENGN_N1_", 0)
        await xp.set("sim/operation/failures/rel_engfai0", 6)
        await asyncio.sleep(8)
        results["engine_failure"] = {"rel_engfai0": await xp.get("sim/operation/failures/rel_engfai0"),
                                     "N1_eng1_before": round(n1_before, 1),
                                     "N1_eng1_after_8s": round(await xp.get("sim/flightmodel/engine/ENGN_N1_", 0), 1)}
        await xp.set("sim/operation/failures/rel_engfai0", 0)

        # 4. Dump the aircraft's own datarefs for FMA / MCP discovery
        all_refs = await xp.list_datarefs()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "t5_datarefs.json").write_text(json.dumps(all_refs, indent=1))
        b738 = [d["name"] for d in all_refs if d["name"].startswith("laminar/B738")]
        results["b738_datarefs"] = len(b738)
        results["all_datarefs"] = len(all_refs)
    finally:
        await xp.close()

    (OUT / "t5_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
