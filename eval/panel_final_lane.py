"""Score Glimmer's final lane (senses/panel_glimmer.py: selected speed, speed, thrust and pitch mode,
for the autothrottle alert on final) against X-Plane truth on recorded 737 frames, next to the fast
lane it replaced: airspeed within 2 kt, selected speed exact, thrust in a speed mode or not (truth:
the autothrottle state). Answers are cached in eval/results/day_one/panel_final_lane_cache.json.
Timing is per read on this computer's GPU, with whatever else is on it.

Run (Ollama with muse-glimmer:30b up):  uv run python eval/panel_final_lane.py
"""
import asyncio
import bisect
import json
import pathlib
import sys
import time

import httpx
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.live import frame_from_crop  # noqa: E402
from brain.rules import SPEED_MODES  # noqa: E402
from senses import panel_glimmer as pg  # noqa: E402

OUT = ROOT / "eval" / "results" / "day_one"
CACHE = OUT / "panel_final_lane_cache.json"
RECS = {"autothrottle_hold": 3, "autothrottle_hold_spark": 2, "nominal": 25}  # recording: every nth reading's frame
LANES = {"final": pg.read_frame_glimmer_final, "fast": pg.read_frame_glimmer_fast}


def truth_at(truth: list[dict], t: float, key: str):
    """The latest value of key at time t (the truth log writes some fields only now and then)."""
    i = bisect.bisect_left([x["t"] for x in truth], t)
    for j in range(min(i, len(truth) - 1), max(-1, i - 40), -1):
        if truth[j].get(key) is not None and truth[j]["t"] <= t + 0.05:
            return truth[j][key]
    return None


async def main() -> int:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    score = {lane: {"ias": [0, 0], "selected_speed": [0, 0], "thrust": [0, 0], "false_not_in_speed": 0, "s": []} for lane in LANES}
    async with httpx.AsyncClient(trust_env=False, timeout=300) as client:
        for rec, step in RECS.items():
            d = ROOT / "replay" / rec
            meta = json.loads((d / "meta.json").read_text())
            truth = [json.loads(l) for l in (d / "truth.jsonl").read_text().splitlines()]
            rows = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()]
            for e in [e for e in rows if e["kind"] == "reading" and e.get("frame")][::step]:
                t = e.get("captured", e["t"])
                img = Image.open(d / e["frame"])
                frame = frame_from_crop(img, tuple(meta["frame_box"])[:2]) if "frame_box" in meta else frame_from_crop(img)
                ias, at, sel = truth_at(truth, t, "ias"), truth_at(truth, t, "at"), truth_at(truth, t, "sel")
                for lane, read in LANES.items():
                    key = f"{lane}:{rec}/{e['frame']}"
                    if key not in cache:
                        t0 = time.time()
                        r, usage = await read(frame, client)
                        cache[key] = {"ias": r.ias, "sel": r.selected_speed, "fma": r.fma, "s": round(time.time() - t0, 2),
                                      "usage": usage, "raw": r.raw}
                        CACHE.write_text(json.dumps(cache, indent=1))
                    g, c = cache[key], score[lane]
                    c["s"].append(g["s"])
                    if ias is not None:
                        c["ias"][1] += 1
                        c["ias"][0] += g["ias"] is not None and abs(g["ias"] - ias) <= 2
                    if sel is not None:
                        c["selected_speed"][1] += 1
                        c["selected_speed"][0] += g["sel"] == round(sel)
                    if at in (0, 1) and g["fma"]:
                        in_speed = g["fma"][0] in SPEED_MODES
                        c["thrust"][1] += 1
                        c["thrust"][0] += in_speed == (at == 1)
                        c["false_not_in_speed"] += at == 1 and not in_speed
    result = {}
    for lane, c in score.items():
        s = sorted(c.pop("s"))
        result[lane] = {**{k: f"{v[0]}/{v[1]}" if isinstance(v, list) else v for k, v in c.items()}, "median_s": s[len(s) // 2]}
    (OUT / "panel_final_lane.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
