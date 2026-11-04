"""Scenario 5: the altimeter setting, Boeing 737-800 near Portland.

Level at 9,000 ft with the altimeter at 29.92. ATC: "Alaska four two one, descend and maintain
seven thousand, Portland altimeter three zero one two." The crew reads it back, sets the MCP to
7,000 and descends. The altimeter setting is the local air pressure; set wrong, the altimeter
reads wrong by about 100 ft for every 0.10 inch, and nothing on the flight deck says so. Variants:
  wrong    - set to 30.21 (two digits swapped)  -> "Altimeter three-zero-one-two given, three-zero-two-one set."
  forgot   - left at 29.92                      -> "Altimeter three-zero-one-two given, two-niner-niner-two set."
  correct  - set to 30.12                       -> silence (false-alarm check)

As in the altitude scenarios, both sides are real: the setting is read from the box under the
captain's altitude tape, and the clearance and readback are spoken (MMS-TTS; the ATC line
radio-filtered) and transcribed live (senses/speech.py parses the altimeter setting).

--spark reads the panel with Muse Spark; on board (onboard/run.sh) Muse Glimmer reads it and the
two open recognizers hear it. Recorded as altimeter_<variant>[_spark|_onboard].

Run with X-Plane 12 open:  uv run python eval/scenarios/altimeter.py [wrong|forgot|correct] [--speak] [--spark]
"""
import asyncio
import json
import os
import pathlib
import sys
import threading
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval" / "scenarios"))
from altitude_bust import FLIGHT, VIEW, camera_check_or_stop, play  # noqa: E402  the same aircraft, place and camera
from brain import phraseology as say  # noqa: E402
from brain.live import READER_LABEL, Crosscheck  # noqa: E402
from bridge.xplane import XPlane, bring_to_front  # noqa: E402
from eval.audio.make_radio_set import radio  # noqa: E402
from eval.scoring import score  # noqa: E402
from senses.onboard import ONBOARD, ONBOARD_SPEECH, network_proof  # noqa: E402
from senses.speech_spark import hear  # noqa: E402

GIVEN = 3012
VARIANTS = {"wrong": 3021, "forgot": None, "correct": 3012}  # what the crew sets; None: not touched
EXPECT = {"wrong": "altimeter_vs_set", "forgot": "altimeter_vs_set", "correct": None}
T_CLEAR, T_SET, RUN_S = 12, 26, 66
OWN = "421"
BARO = "laminar/B738/barometer/capt_setting"  # the 737's own captain altimeter (its plugin overrides X-Plane's generic one)
TRUTH = {"alt": "sim/cockpit2/gauges/indicators/altitude_ft_pilot", "sel_alt": "sim/cockpit2/autopilot/altitude_dial_ft",
         "baro": BARO, "paused": "sim/time/paused"}
LINES = {"atc": "alaska four two one descend and maintain seven thousand portland altimeter three zero one two",
         "pm": "descend and maintain seven thousand altimeter three zero one two alaska four two one"}


async def turn_baro(xp: XPlane, target: int, pause: float = 0.1) -> None:
    """Turn the captain's baro knob to target (inches x 100), a click at a time, as a pilot does: 0.01 inch a click."""
    for _ in range(80):
        now = round(await xp.get(BARO) * 100)
        if now == target:
            return
        await xp.command("sim/instruments/barometer_up" if now < target else "sim/instruments/barometer_down")
        await asyncio.sleep(pause)


def expected_callout(variant: str) -> str | None:
    if EXPECT[variant] is None:
        return None
    return f"Altimeter {say.altimeter(GIVEN)} given, {say.altimeter(VARIANTS[variant] or 2992)} set."


async def main(variant: str, speak: bool, reader: str = "apple") -> int:
    baro_set = VARIANTS[variant]
    from voice.tts import synth
    audio = {}
    for who, text in LINES.items():
        pcm, sr = synth(text)
        assert sr == 16000, sr
        audio[who] = radio(pcm) if who == "atc" else pcm * 0.9 / (np.abs(pcm).max() + 1e-9)
    heard = []
    name = f"altimeter_{variant}" + ("_spark" if reader == "spark" else "_onboard" if reader == "glimmer" else "")
    out = pathlib.Path(os.environ.get("CROSSCHECK_OUT", ROOT / "replay")) / name  # eval/repeat.py sets it
    out.mkdir(parents=True, exist_ok=True)
    xp = XPlane()
    await xp.ensure_running()
    await xp.start_flight(FLIGHT)
    await xp.wait_until_loaded()
    await asyncio.sleep(4)
    await turn_baro(xp, 2992)  # this 737 keeps its setting from flight to flight, and ignores barometer_2992
    if round(await xp.get(BARO) * 100) != 2992:
        raise SystemExit(f"altimeter did not reach 29.92 before the run: {await xp.get(BARO):.2f}")
    await xp.set("sim/cockpit2/autopilot/altitude_dial_ft", 9000.0)
    await xp.set("sim/cockpit2/autopilot/airspeed_dial_kts", 250.0)
    await xp.set("sim/cockpit2/autopilot/heading_dial_deg_mag_pilot", 285.0)
    for c in ("cmd_a_press", "speed_press", "hdg_sel_press", "alt_hld_press"):
        await xp.command("laminar/B738/autopilot/" + c)
        await asyncio.sleep(0.8)
    for k, v in VIEW.items():
        await xp.set("sim/graphics/view/" + k, v)
    await asyncio.sleep(6)

    say = None
    if speak:
        from voice.tts import speak as say
    cc = Crosscheck(out, speak=say, frames=True, reader=reader, explain=True,
                    fact_time="captured" if reader in ("spark", "glimmer") else "arrival")
    truth_rows = []
    bring_to_front()  # the panel reader captures the screen
    cam = camera_check_or_stop(out)
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

    async def transmit(who: str):
        """One radio or intercom line: speech -> recognizer -> parsed facts -> ledger."""
        t_start = time.time()
        if speak:
            play(audio[who], 16000)
        r, facts, spark, first = await hear(audio[who], who, OWN, cc.ledger)
        t = time.time()
        heard.append({"speaker": who, "spoken": LINES[who], "transcript": r["text"], "audio_s": r["audio_s"],
                      **({"alt_transcript": r["alt"]} if "alt" in r else {}),
                      "s_from_start": round(t - t_start, 2), "facts": [(f.key, f.value) for f in facts], "spark": spark,
                      **({"first_heard": first} if first else {})})
        cc.event("heard", t, speaker=who, transcript=r["text"], spoken=LINES[who])
        for f in facts:
            cc.said(f, transcript=r["text"])

    await at(T_CLEAR)
    await transmit("atc")
    await asyncio.sleep(1.0)
    await transmit("pm")
    await at(T_SET)
    t_set = time.time()
    await xp.set("sim/cockpit2/autopilot/altitude_dial_ft", 7000.0)
    await xp.command("laminar/B738/autopilot/lvl_chg_press")
    if baro_set is not None:
        await turn_baro(xp, baro_set)
    cc.event("action", t_set, line=f"PF sets MCP 7000 and descends; altimeter " +
             (f"set to {baro_set / 100:.2f}" if baro_set is not None else "left at 29.92"))
    while worker.is_alive():
        await asyncio.sleep(0.5)
    await sampler
    await xp.park()
    await xp.close()

    with (out / "truth.jsonl").open("w") as f:
        for r in truth_rows:
            f.write(json.dumps(r) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "scenario": name, "scorer": "altimeter", "panel_reader": READER_LABEL[reader], "aircraft": "B738", "area": "Portland",
        "variant": variant, "fact_time": cc.fact_time, "given": GIVEN, "baro_set": baro_set if baro_set is not None else 2992,
        "cleared": 7000, "expect_rule": EXPECT[variant], "expect_callout": expected_callout(variant), "t_start": t0, "t_clear": t0 + T_CLEAR, "t_set": t_set,
        "speech": ONBOARD_SPEECH if reader == "glimmer" else "Muse Voice Transcribe, live (MMS-TTS lines; ATC radio-filtered), parsed by senses/speech.py",
        "heard": heard, "camera_check": {"model": cam["model"], **cam["matches"]},
        **({"network": network_proof()} if ONBOARD else {}), "recorded": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=1))
    result = score(out)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(asyncio.run(main(args[0] if args else "wrong", speak="--speak" in sys.argv,
                              reader="glimmer" if ONBOARD else "spark" if "--spark" in sys.argv else "apple")))
