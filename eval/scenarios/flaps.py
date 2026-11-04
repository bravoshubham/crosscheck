"""Scenario 2: flap callout not matched by the flaps, Boeing 737-800 near Portland.

Level at 3,000 ft, 180 kt, flaps 5. The pilot flying calls "flaps fifteen" on the intercom.
Variants, for what the pilot monitoring then does with the lever:
  wrong    - selects 10  -> "Flaps fifteen called, ten set."
  forgot   - leaves it   -> "Flaps fifteen called, five set."
  correct  - selects 15  -> silence (the needle travels ~10 s; it must not be flagged on the way)

Speech: the callout is spoken (macOS voice, voice/crew.py) and heard through Muse Voice Transcribe.
Panel: the camera looks right so it sees the centre panel; Muse Spark reads the analog flap gauge
(senses/flaps_spark.py). Nothing is read from X-Plane except to score the run.

--mic: you are the pilot flying. Call any flap setting into the Mac's microphone ("flaps fifteen",
"flaps ten"...) when prompted; the pilot monitoring (this script) then mis-selects it (wrong), leaves
the lever (forgot) or selects it (correct). Recorded as flaps_<variant>_mic.

Run with X-Plane 12 open:  uv run python eval/scenarios/flaps.py [wrong|forgot|correct] [--speak] [--mic]
"""
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time

import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brain.live import Crosscheck  # noqa: E402
from senses.onboard import network_proof, ONBOARD, ONBOARD_SPEECH  # noqa: E402
from bridge.xplane import XPlane, bring_to_front  # noqa: E402
from eval.scoring import score  # noqa: E402
from senses.flaps_spark import FLAPS_VIEW  # noqa: E402
from senses.mic import Mic  # noqa: E402
from senses.speech import transcribe  # noqa: E402
from senses.speech_spark import read_line  # noqa: E402
from voice.crew import CREW, SR, say  # noqa: E402

FLIGHT = {
    "aircraft": {"path": "Aircraft/Laminar Research/Boeing 737-800/b738.acf"},
    "lle_air_start": {"latitude": 45.62, "longitude": -122.75, "elevation_in_meters": 914,
                      "heading_true": 300, "speed_in_meters_per_second": 93},
    "engine_status": {"all_engines": {"running": True}},
    "local_time": {"day_of_year": 180, "time_in_24_hours": 12.0},
}
RATIO = {0: 0.0, 1: 0.125, 2: 0.25, 5: 0.375, 10: 0.5, 15: 0.625, 25: 0.75, 30: 0.875, 40: 1.0}
VARIANTS = {"wrong": 10, "forgot": None, "correct": 15}
EXPECT = {"wrong": "callout_vs_set", "forgot": "callout_vs_set", "correct": None}
CALLED = 15
FRAME_BOX = (2300, 250, 2900, 650)  # the flap gauge with the gear and LE flap lights around it
T_CALL, RUN_S = 8, 42
CALL_WAIT_S = 15  # --mic: how long the PF has to make the callout
DETENTS = [1, 2, 5, 10, 15, 25, 30, 40]


def wrong_detent(called: int) -> int:
    """The PM's slip: one detent short of the call (from flaps 5 the needle must still move)."""
    i = DETENTS.index(called)
    return DETENTS[i - 1] if called > 5 else DETENTS[i + 1]
TRUTH = {"handle": "sim/cockpit2/controls/flap_ratio", "deploy": "sim/cockpit2/controls/flap_handle_deploy_ratio",
         "ias": "sim/cockpit2/gauges/indicators/airspeed_kts_pilot", "paused": "sim/time/paused"}


def play(pcm, sr: int) -> None:
    path = pathlib.Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(path, pcm, sr)
    subprocess.Popen(["afplay", str(path)])


async def main(variant: str, speak: bool, mic: Mic | None = None) -> int:
    selects = VARIANTS[variant]
    called = CALLED
    name = f"flaps_{variant}" + ("_onboard" if ONBOARD else "") + ("_mic" if mic else "")
    out = pathlib.Path(os.environ.get("CROSSCHECK_OUT", ROOT / "replay")) / name  # eval/repeat.py sets it
    out.mkdir(parents=True, exist_ok=True)
    line = "flaps fifteen"
    audio = say(line, CREW)

    xp = XPlane()
    await xp.ensure_running()
    await xp.start_flight(FLIGHT)
    await xp.wait_until_loaded()
    await asyncio.sleep(4)
    await xp.set("sim/cockpit2/autopilot/altitude_dial_ft", 3000.0)
    await xp.set("sim/cockpit2/autopilot/airspeed_dial_kts", 180.0)
    await xp.set("sim/cockpit2/autopilot/heading_dial_deg_mag_pilot", 285.0)
    for c in ("cmd_a_press", "speed_press", "hdg_sel_press", "alt_hld_press"):
        await xp.command("laminar/B738/autopilot/" + c)
        await asyncio.sleep(0.8)
    await xp.set("sim/cockpit2/controls/flap_ratio", RATIO[5])
    for k, v in FLAPS_VIEW.items():
        await xp.set("sim/graphics/view/" + k, v)
    t_wait = time.time()  # flaps 0 -> 5 take ~15 s; start only once they are there
    while abs(await xp.get(TRUTH["deploy"]) - RATIO[5]) > 0.003 and time.time() - t_wait < 40:
        await asyncio.sleep(0.5)
    await asyncio.sleep(2)

    say_cc = mic.speak if mic else None
    if speak and not mic:
        from voice.tts import speak as say_cc
    # facts at capture time: a slow Spark once delivered four readings of a moving needle together, and
    # stamped on arrival they looked settled (flaps_wrong, 3 Nov 11:22: "five set" for 10)
    cc = Crosscheck(out, speak=say_cc, frames=True, reader="glimmer-flaps" if ONBOARD else "spark-flaps", frame_box=FRAME_BOX, explain=True, fact_time="captured")
    heard, truth_rows = [], []
    bring_to_front()  # the camera captures the screen
    from brain.live import capture_screen
    from senses.flaps_spark import FLAP_GAUGE
    if ONBOARD:  # nothing leaves the aircraft: the local Glimmer must read the gauge instead
        from senses.onboard import camera_check
        cam = camera_check(capture_screen(), "flaps")
    else:
        from senses.sam31 import inside_check
        cam = inside_check(capture_screen(), "gauge", FLAP_GAUGE)  # SAM 3.1 pre-flight check of the camera
    (out / "camera_check.json").write_text(json.dumps(cam, indent=1))
    if not cam["ok"]:
        raise SystemExit(f"camera check failed: SAM 3.1 found no gauge in the flap reader's view ({cam['found']})")
    t0 = time.time()
    worker = threading.Thread(target=cc.run, args=(RUN_S, t0), daemon=True)
    worker.start()

    async def at(seconds: float):
        await asyncio.sleep(max(0.0, t0 + seconds - time.time()))

    async def sample_truth():
        while worker.is_alive():
            truth_rows.append({"t": time.time(), **{k: await xp.get(n) for k, n in TRUTH.items()}})
            await asyncio.sleep(0.5)
    sampler = asyncio.create_task(sample_truth())

    await at(T_CALL)
    t_call = time.time()
    if mic:
        print(">>> Call the flaps now (e.g. \"flaps fifteen\")", flush=True)
        mic.flush()
        facts = []
        while not facts and time.time() - t_call < CALL_WAIT_S:  # anything else said is logged, not acted on
            got = await mic.next_line(CALL_WAIT_S - (time.time() - t_call))
            if got is None:
                break
            f_audio = f"mic{len(heard)}.wav"
            sf.write(out / f_audio, got["pcm"], SR)
            facts, spark = await read_line(got["text"], "pf", "421")
            facts = [f for f in facts if f.key == "callout.flaps" and f.value in DETENTS]
            heard.append({"speaker": "pf", "spoken": None, "audio_file": f_audio, "transcript": got["text"],
                          "audio_s": got["audio_s"], "s_from_start": round(got["t"] - got["t_start"], 2),
                          "facts": [(f.key, f.value) for f in facts], "spark": spark})
            cc.event("heard", got["t"], speaker="pf", transcript=got["text"], spoken=None, mic=True)
            t_call = got["t_start"]
            for f in facts:
                cc.said(f, transcript=got["text"])
        if not facts:
            print("no flap callout heard; stopping", flush=True)
            await xp.park()
            return 2
        called = facts[0].value
        selects = {"wrong": wrong_detent(called), "forgot": None, "correct": called}[variant]
    else:
        if speak:
            play(audio, SR)
        r = await transcribe(audio)
        t = time.time()
        facts, spark = await read_line(r["text"], "pf", "421", alt=r.get("alt"))  # on board: both recognizers
        heard.append({"speaker": "pf", "spoken": line, "transcript": r["text"], "audio_s": r["audio_s"],
                      **({"alt_transcript": r["alt"]} if "alt" in r else {}),
                      "s_from_start": round(t - t_call, 2), "facts": [(f.key, f.value) for f in facts], "spark": spark})
        cc.event("heard", t, speaker="pf", transcript=r["text"], spoken=line)
        for f in facts:
            cc.said(f, transcript=r["text"])
    await asyncio.sleep(1.0)
    t_move = time.time()
    if selects is not None:
        await xp.set("sim/cockpit2/controls/flap_ratio", RATIO[selects])
        cc.event("action", t_move, line=f"PM selects flaps {selects}")
    else:
        cc.event("action", t_move, line="PM does not move the flap lever")
    while worker.is_alive():
        await asyncio.sleep(0.5)
    await sampler
    await xp.park()
    await xp.close()

    with (out / "truth.jsonl").open("w") as f:
        for row in truth_rows:
            f.write(json.dumps(row) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "scenario": name, "scorer": "flaps", "aircraft": "B738", "area": "Portland", "variant": variant,
        "called": called, "selected": selects,
        "expect_rule": ("callout_vs_set" if (selects if selects is not None else 5) != called else None) if mic else EXPECT[variant],
        "t_start": t0, "t_call": t_call, "t_move": t_move, "frame_box": list(FRAME_BOX),
        "panel_reader": "muse-glimmer-flaps" if ONBOARD else "muse-spark-flaps", "view": FLAPS_VIEW, "fact_time": cc.fact_time,
        "speech": ONBOARD_SPEECH if ONBOARD else "Muse Voice Transcribe, live: a person on the Mac's microphone, parsed by senses/speech.py" if mic else
                  "Muse Voice Transcribe, live (macOS voice on the intercom), parsed by senses/speech.py",
        "heard": heard, "camera_check": {"model": cam["model"], **cam["matches"]},
        **({"network": network_proof()} if ONBOARD else {}), "recorded": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=1))
    result = score(out)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


async def _run(variant: str, speak: bool) -> int:
    if "--mic" not in sys.argv:
        return await main(variant, speak)
    async with Mic() as mic:
        return await main(variant, speak, mic)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(asyncio.run(_run(args[0] if args else "wrong", speak="--speak" in sys.argv)))
