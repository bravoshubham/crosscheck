"""Presenting Crosscheck live: one command for the stage (docs/DEMO.md is the runbook).

  uv run python cli.py demo altitude    # ATC clears 11,000 on the speakers; you read it back (right or wrong)
  uv run python cli.py demo flaps       # you call a flap setting; the pilot monitoring gets it wrong
  uv run python cli.py demo pilot       # the generated pilot slumps; answer Crosscheck when it asks
  uv run python cli.py demo replay altitude_bust   # fallback: re-enact a shipped recording in real time

Each starts the tablet display first (port 8770; the address to open on an iPad is printed) and
writes to demo_runs/, which is git-ignored and never part of the debrief: your voice stays on
this Mac. A re-enactment writes a recording's events again at their original pace, so the tablet
animates exactly as it did live, and Crosscheck speaks its callouts.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / "demo_runs"
LIVE = {  # what each live demo runs
    "altitude": (["eval/scenarios/altitude_bust.py", "bust", "--mic"], "altitude_bust_mic",
                 "ATC will clear you to one one thousand. Read it back: right, or wrong on purpose."),
    "flaps": (["eval/scenarios/flaps.py", "wrong", "--mic"], "flaps_wrong_mic",
              "When prompted, call a flap setting: \"flaps fifteen\". The pilot monitoring will set the wrong one."),
    "pilot": (["eval/scenarios/pilot_watch.py", "answers", "--video", "--mic", "--speak"], "pilot_answers",
              "The captain will slump. When Crosscheck asks \"Captain, are you OK?\", answer it out loud."),
}


def lan_address() -> str | None:
    for iface in ("en0", "en1"):
        r = subprocess.run(["ipconfig", "getifaddr", iface], capture_output=True, text=True)
        if r.stdout.strip():
            return r.stdout.strip()
    return None


def start_tablet(port: int = 8770) -> None:
    from ui.server import serve
    threading.Thread(target=serve, kwargs={"port": port}, daemon=True).start()
    time.sleep(1.5)
    ip = lan_address()
    print(f"Tablet display: http://localhost:{port}" + (f"  (iPad on the same Wi-Fi: http://{ip}:{port})" if ip else ""))


def live(which: str) -> int:
    args, name, brief = LIVE[which]
    RUNS.mkdir(exist_ok=True)
    print(brief)
    cmd = ["uv", "run", "--group", "eval", "python", *args] + ([] if "--speak" in args else ["--speak"])
    r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "CROSSCHECK_OUT": str(RUNS)})
    if r.returncode not in (0, 1):  # 0 pass, 1 scored but not as expected; anything else: it did not run
        print(f"\nThe live run stopped (exit {r.returncode}). Fallback: uv run python cli.py demo replay {name.removesuffix('_mic')}")
    return r.returncode


def reenact(name: str, speed: float = 1.0) -> int:
    """Write a recording's events again at their original pace, into demo_runs/<name>_reenacted."""
    src = ROOT / "replay" / name
    if not (src / "events.jsonl").exists():
        print(f"no recording replay/{name}; choose one of: " + ", ".join(sorted(p.parent.name for p in (ROOT / "replay").glob("*/events.jsonl"))))
        return 2
    out = RUNS / f"{name}_reenacted"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if (src / "frames").exists():
        (out / "frames").symlink_to((src / "frames").resolve())
    meta = json.loads((src / "meta.json").read_text())
    events = [json.loads(line) for line in (src / "events.jsonl").read_text().splitlines() if line.strip()]
    shift = time.time() + 1.0 - meta["t_start"]
    meta = {**{k: (v + shift if k.startswith("t_") and isinstance(v, (int, float)) else v) for k, v in meta.items()},
            "scenario": out.name, "reenacted_from": name}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    from voice.tts import speak
    with (out / "events.jsonl").open("w") as f:
        for e in sorted(events, key=lambda e: e["t"]):
            t = e["t"] + shift
            time.sleep(max(0.0, (t - time.time()) / speed))
            f.write(json.dumps({**e, "t": t, **({"captured": e["captured"] + shift} if "captured" in e else {})}) + "\n")
            f.flush()
            if e["kind"] == "callout":
                speak(e["callout"])
    time.sleep(4)  # let the last callout finish
    print(f"re-enacted replay/{name} ({len(events)} events)")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in (*LIVE, "replay") or (argv[0] == "replay" and len(argv) < 2):
        print(__doc__)
        return 2
    start_tablet()
    return reenact(argv[1]) if argv[0] == "replay" else live(argv[0])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
