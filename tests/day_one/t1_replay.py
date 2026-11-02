"""T1 scored on a recording: run senses.panel.read_frame on every recorded frame and compare
with the X-Plane truth logged around each capture. No simulator or API needed.

Run:  uv run python tests/day_one/t1_replay.py [eval/recordings/t1_pfd]
"""
import json
import pathlib
import statistics
import sys
import time

from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from senses.panel import read_frame  # noqa: E402

TARGETS = {"ias": 2, "alt": 20, "trk": 2}
THRUST = {1: "MCP SPD", 0: "ARM", -1: ""}  # autothrottle_enabled -> expected thrust cell (LVL CHG may show RETARD/N1)


def ang(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def main(rec: pathlib.Path) -> int:
    meta = json.loads((rec / "truth.json").read_text())
    rows, errs = [], {k: [] for k in TARGETS}
    methods, fma_seen = {}, {}
    for m in meta:
        t0 = time.monotonic()
        r = read_frame(Image.open(rec / f"{m['i']:03d}.png"))
        dt = time.monotonic() - t0
        b, a = m["before"], m["after"]
        truth = {"ias": (b["ias"] + a["ias"]) / 2, "alt": (b["alt"] + a["alt"]) / 2,
                 "trk": (a["trk_true"] - a["magvar"]) % 360}
        e = {}
        if r.ias is not None:
            e["ias"] = abs(r.ias - truth["ias"])
        if r.altitude is not None:
            e["alt"] = abs(r.altitude - truth["alt"])
        if r.track_mag is not None:
            e["trk"] = ang(r.track_mag, truth["trk"])
        for k, v in e.items():
            errs[k].append(v)
        methods[r.altitude_method] = methods.get(r.altitude_method, 0) + 1
        fma_seen[" | ".join(r.fma)] = fma_seen.get(" | ".join(r.fma), 0) + 1
        rows.append({"i": m["i"], "truth": {k: round(v, 1) for k, v in truth.items()},
                     "read": {"ias": r.ias, "alt": r.altitude, "alt_method": r.altitude_method,
                              "trk": r.track_mag, "fma": r.fma},
                     "err": {k: round(v, 1) for k, v in e.items()}, "modes": m["modes"], "read_s": round(dt, 3)})
    n = len(meta)
    summary = {"frames": n, "reader": "senses.panel (Apple Vision OCR on calibrated crops)",
               "read_s_median": round(statistics.median(r["read_s"] for r in rows), 3)}
    for k, tol in TARGETS.items():
        v = errs[k]
        summary[k] = {"read": f"{len(v)}/{n}", "median_err": round(statistics.median(v), 1) if v else None,
                      "max_err": round(max(v), 1) if v else None,
                      "within_target": f"{sum(x <= tol for x in v)}/{n}", "target": tol}
    summary["altitude_method"] = methods
    summary["fma_strings"] = fma_seen
    out = pathlib.Path(__file__).resolve().parents[2] / "eval" / "results" / "day_one"
    (out / "t1_replay_rows.json").write_text(json.dumps(rows, indent=1))
    (out / "t1_replay_results.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                               pathlib.Path(__file__).resolve().parents[2] / "eval" / "recordings" / "t1_pfd")))
