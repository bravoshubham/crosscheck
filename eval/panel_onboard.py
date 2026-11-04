"""Score the on-board readers (the local Muse Glimmer) against X-Plane truth on the frames saved
with each recording in replay/, next to the reader that recorded them.

PFD: every 6th saved frame of the recordings read by on-device text recognition or Muse Spark,
the same tolerances as eval/panel_spark.py (airspeed 2 kt, altitude 20 ft, selected altitude
exact). Flap gauge: every 3rd frame of the flap recordings with the lever at rest, the detent
against the simulator's flap handle. Answers are cached in eval/results/day_one/panel_glimmer_cache.json.
Timing is only meaningful with nothing else on the GPU.

Run (Ollama with muse-glimmer:30b up):  uv run python eval/panel_onboard.py
"""
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
from eval.panel_spark import CHECKS, load_jsonl, right  # noqa: E402
from senses.flaps_glimmer import read_flaps_glimmer  # noqa: E402
from senses.panel_glimmer import read_frame_glimmer  # noqa: E402

OUT = ROOT / "eval" / "results" / "day_one"
CACHE = OUT / "panel_glimmer_cache.json"
RATIO = {0: 0.0, 1: 0.125, 2: 0.25, 5: 0.375, 10: 0.5, 15: 0.625}  # flap handle deploy ratio per detent


async def main() -> int:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    pfd, flaps = [], []
    for rec in sorted(p.parent for p in (ROOT / "replay").glob("*/meta.json")):
        meta = json.loads((rec / "meta.json").read_text())
        reader = meta.get("panel_reader", "")
        rows = [e for e in load_jsonl(rec / "events.jsonl") if e["kind"] == "reading" and e.get("frame") and (rec / e["frame"]).exists()]
        if reader in ("apple-vision", "muse-spark") and (rec / "truth.jsonl").exists():
            pfd += [(rec, meta, e) for e in rows[::6]]
        elif reader == "muse-spark-flaps":
            flaps += [(rec, meta, e) for e in rows[::3]]
    async with httpx.AsyncClient(trust_env=False, timeout=300) as client:
        for kind, jobs, read in (("pfd", pfd, read_frame_glimmer), ("flaps", flaps, read_flaps_glimmer)):
            for rec, meta, e in jobs:
                key = f"{kind}:{rec.name}/{e['frame']}"
                if key in cache:
                    continue
                t = time.time()
                img = Image.open(rec / e["frame"])  # the crop saved with the frame, put back where it was on screen
                frame = frame_from_crop(img, tuple(meta["frame_box"])[:2]) if "frame_box" in meta else frame_from_crop(img)
                r, usage = await read(frame, client)
                cache[key] = {"ias": r.ias, "selected_altitude": r.selected_altitude, "altitude": r.altitude, "fma": r.fma,
                              "flaps": r.flaps, "seconds": round(time.time() - t, 2), "usage": usage}
                CACHE.write_text(json.dumps(cache, indent=1))
    score = {}
    for rec, meta, e in pfd:
        truth = load_jsonl(rec / "truth.jsonl")
        g = cache[f"pfd:{rec.name}/{e['frame']}"]
        base = "muse-spark" if meta["panel_reader"] == "muse-spark" else "apple"
        for field, (tf, tol) in CHECKS.items():
            for reader, val in (("glimmer", g[field]), (base, e.get(field))):
                ok = right(val, truth, e["t"], tf, tol)
                if ok is not None:
                    c = score.setdefault(field, {}).setdefault(reader, [0, 0])
                    c[0] += ok
                    c[1] += 1
    fl = {"glimmer": [0, 0], "muse-spark": [0, 0]}
    for rec, meta, e in flaps:
        truth = [x for x in load_jsonl(rec / "truth.jsonl") if e["t"] - 1 <= x["t"] <= e["t"] + 0.1 and x.get("deploy") is not None]
        if not truth or max(x["deploy"] for x in truth) - min(x["deploy"] for x in truth) > 0.01:
            continue  # the lever is moving: no single right answer
        true = min(RATIO, key=lambda k: abs(RATIO[k] - truth[-1]["deploy"]))
        for reader, val in (("glimmer", cache[f"flaps:{rec.name}/{e['frame']}"]["flaps"]), ("muse-spark", e.get("flaps"))):
            fl[reader][0] += val == true
            fl[reader][1] += 1
    secs = sorted(v["seconds"] for k, v in cache.items() if k.startswith("pfd:"))
    result = {"pfd_frames": len(pfd), "right": {f: {k: f"{a}/{b}" for k, (a, b) in v.items()} for f, v in score.items()},
              "flap_gauge_right": {k: f"{a}/{b}" for k, (a, b) in fl.items()},
              "glimmer_median_s": secs[len(secs) // 2] if secs else None, "tolerance": {f: tol for f, (_, tol) in CHECKS.items()}}
    (OUT / "panel_glimmer.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
