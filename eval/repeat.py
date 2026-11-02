"""Run scenarios several times live and report how often Crosscheck got each one right.

  uv run python eval/repeat.py --n 3 altitude_bust:bust altitude_bust:readback altitude_bust:correct \
      autothrottle_hold autothrottle_hold+spark flaps:wrong flaps:forgot flaps:correct

Each run goes to eval/results/repeat/<scenario>/<run>/ (frames and logs git-ignored); the
per-scenario summary, eval/results/repeat/<scenario>.json, is kept. A run that could not be
scored (simulator stuck, script error) is reported as an error, never as a pass or a miss.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results" / "repeat"


def parse(spec: str) -> tuple[str, str, str, str]:
    """spec: script[:variant][+flag], e.g. flaps:wrong or autothrottle_hold+spark (adds --spark).
    Returns (script, variant, flag, result name)."""
    spec, _, flag = spec.partition("+")
    script, _, variant = spec.partition(":")
    name = (f"{script.replace('_bust', '')}_{variant}" if variant else script) + (f"_{flag}" if flag else "")
    return script, variant, flag, name


def run(spec: str, i: int) -> dict:
    script, variant, flag, name = parse(spec)
    base = OUT / name / f"run{i:02d}"
    base.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "--group", "eval", "python", f"eval/scenarios/{script}.py"] + ([variant] if variant else []) + ([f"--{flag}"] if flag else []) + ["--silent"]
    t = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "CROSSCHECK_OUT": str(base)}, capture_output=True, text=True)
    results = list(base.glob("*/result.json"))
    if not results:
        return {"run": i, "error": (p.stderr or p.stdout)[-600:], "seconds": round(time.time() - t)}
    r = json.loads(results[0].read_text())
    return {"run": i, "pass": r["pass"], "callouts": r.get("callouts") or r.get("callout"),
            "seconds": round(time.time() - t), **{k: r[k] for k in ("callout_s_after_fault", "false_callouts_before_fault",
                                                                      "false_callouts_before_clearance", "false_callouts_before_call")
                                                  if k in r}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    for spec in a.specs:
        name = parse(spec)[3]
        summary_p = OUT / f"{name}.json"
        prev = json.loads(summary_p.read_text())["runs"] if summary_p.exists() else []
        runs = prev[:]
        for i in range(len(prev), len(prev) + a.n):
            r = run(spec, i)
            runs.append(r)
            print(name, json.dumps(r)[:300], flush=True)
            scored = [x for x in runs if "pass" in x]
            summary_p.write_text(json.dumps({"scenario": name, "scored": len(scored),
                                             "passed": sum(x["pass"] for x in scored),
                                             "errors": len(runs) - len(scored), "runs": runs}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
