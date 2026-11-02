"""Score the Muse Spark panel reader against X-Plane truth, next to the Apple reader, on the
frames saved with each recording in replay/.

A value counts as right if it falls within tolerance of X-Plane's value at some moment in the
second before the reading (the capture happens just before the reading is logged, and truth is
sampled every 0.1-0.5 s). Tolerances: airspeed 2 kt, altitude 20 ft, selected altitude exact.
Thrust mode is checked against the scenario's own fault time (speed mode before, none after).
Spark answers are cached in eval/results/day_one/panel_spark_cache.json; reruns are free.

Run:  uv run python eval/panel_spark.py [--every 3]
"""
import argparse
import asyncio
import json
import pathlib
import sys
import time

import httpx
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.live import frame_from_crop  # noqa: E402
from senses.panel_spark import read_frame_spark  # noqa: E402

OUT = ROOT / "eval" / "results" / "day_one"
CACHE = OUT / "panel_spark_cache.json"
USD_IN, USD_OUT = 1.25 / 1e6, 4.25 / 1e6  # Muse Spark per token
PARALLEL = 6
CHECKS = {  # reading field: (truth field, tolerance)
    "ias": ("ias", 2), "altitude": ("alt", 20), "selected_altitude": ("sel_alt", 0)}


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def right(value, truth_rows, t, field, tol):
    window = [r[field] for r in truth_rows if t - 1.0 <= r["t"] <= t + 0.1 and r.get(field) is not None]
    if not window or value is None:
        return None if not window else False
    return min(window) - tol <= value <= max(window) + tol


async def main(every: int) -> int:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    jobs = []
    for rec in sorted(p.parent for p in (ROOT / "replay").glob("*/meta.json")):
        readings = [e for e in load_jsonl(rec / "events.jsonl") if e["kind"] == "reading" and e.get("frame")]
        for e in readings[::every]:
            jobs.append((rec, e))
    sem = asyncio.Semaphore(PARALLEL)
    async with httpx.AsyncClient(trust_env=False, timeout=90) as client:
        async def one(rec, e):
            key = f"{rec.name}/{e['frame']}"
            if key in cache:
                return
            async with sem:
                t = time.time()
                r, usage = await read_frame_spark(frame_from_crop(Image.open(rec / e["frame"])), client)
                cache[key] = {"ias": r.ias, "selected_speed": r.selected_speed, "selected_altitude": r.selected_altitude,
                              "altitude": r.altitude, "fma": r.fma, "raw": r.raw["spark"], "usage": usage,
                              "seconds": round(time.time() - t, 2)}
                CACHE.write_text(json.dumps(cache, indent=1))
        await asyncio.gather(*(one(rec, e) for rec, e in jobs))

    score = {}
    for rec, e in jobs:
        truth = load_jsonl(rec / "truth.jsonl")
        s = cache[f"{rec.name}/{e['frame']}"]
        for field, (tf, tol) in CHECKS.items():
            for reader, val in (("spark", s[field]), ("apple", e.get(field))):
                ok = right(val, truth, e["t"], tf, tol)
                if ok is None:
                    continue
                c = score.setdefault(field, {}).setdefault(reader, [0, 0])
                c[0] += ok
                c[1] += 1
    # thrust mode: X-Plane logs the autothrottle state only at start and fault, but the script
    # knows it: in speed mode before the fault, not after (3 s skipped while the display updates)
    for rec, e in jobs:
        meta = json.loads((rec / "meta.json").read_text())
        if "t_fault" not in meta or meta["t_fault"] <= e["t"] < meta["t_fault"] + 3:
            continue
        speed = e["t"] < meta["t_fault"]
        for reader, fma in (("spark", cache[f"{rec.name}/{e['frame']}"]["fma"]), ("apple", e.get("fma"))):
            c = score.setdefault("thrust_mode", {}).setdefault(reader, [0, 0])
            c[0] += (bool(fma) and fma[0] == "MCP SPD") == speed
            c[1] += 1
    usage = [cache[f"{r.name}/{e['frame']}"]["usage"] for r, e in jobs]
    secs = sorted(cache[f"{r.name}/{e['frame']}"]["seconds"] for r, e in jobs)
    cost = sum(u.get("prompt_tokens", 0) * USD_IN + u.get("completion_tokens", 0) * USD_OUT for u in usage)
    result = {"frames": len(jobs), "every": every,
              "right": {f: {k: f"{a}/{b}" for k, (a, b) in v.items()} for f, v in score.items()},
              "spark_median_s": secs[len(secs) // 2], "spark_cost_usd": round(cost, 3),
              "tolerance": {f: tol for f, (_, tol) in CHECKS.items()}}
    (OUT / "panel_spark.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3)
    sys.exit(asyncio.run(main(ap.parse_args().every)))
