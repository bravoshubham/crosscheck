"""Scenario 4: pilot incapacitation, through the opt-in pilot-watch camera.

The camera's view is a sequence of images of one pilot (generated with Muse Image, labelled by
the state they show, eval/pilot_watch/): upright until T_CHANGE, then the variant's state.
Every frame goes to SAM 3D Body on Modal as it is taken; senses/pilot_pose.py turns the body
keypoints into "slumped or not". A slumped posture alone never alerts: Crosscheck asks first.
  slumped    - set e, slumped forward, no answer   -> "Captain, are you OK?", then "...take control."
  asleep     - set f, asleep against the window     -> the same
  checklist  - set b, reading a checklist           -> silence
  answers    - set d, slumped, answers when asked   -> the question only, no alert
The answer is spoken (macOS voice) and heard through Muse Voice Transcribe, like all crew speech.
No simulator is needed.

Needs:  uv run modal deploy cloud/sam3d_body_app.py
Run:    uv run --group eval python eval/scenarios/pilot_watch.py [slumped|asleep|checklist|answers] [--speak]
"""
import asyncio
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import soundfile as sf
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brain.ledger import Fact  # noqa: E402
from brain.live import Crosscheck  # noqa: E402
from eval.scoring import score  # noqa: E402
from senses.pilot_pose import posture  # noqa: E402
from senses.speech import transcribe  # noqa: E402
from voice.crew import SR, say  # noqa: E402

IMAGES = ROOT / "eval" / "pilot_watch" / "images"
BOXES = json.loads((ROOT / "eval" / "pilot_watch" / "boxes.json").read_text())
VARIANTS = {  # image set, state after the change, what the pilot says when asked
    "slumped": ("e", "slumped", None),
    "asleep": ("f", "slumped_side", None),
    "checklist": ("b", "looking_down", None),
    "answers": ("d", "slumped", "I'm fine, just reaching for my pen."),
}
EXPECT = {"slumped": (True, True), "asleep": (True, True), "checklist": (False, False), "answers": (True, False)}
T_CHANGE, RUN_S = 10, 45
SCALE = 0.5  # frames go to the GPU at 800 x 800


def frame_bytes(image_set: str, state: str) -> bytes:
    im = Image.open(IMAGES / image_set / f"{state}.jpg").convert("RGB")
    buf = io.BytesIO()
    im.resize((int(im.width * SCALE), int(im.height * SCALE))).save(buf, "JPEG", quality=85)
    return buf.getvalue()


def play(pcm, sr: int) -> None:
    path = pathlib.Path(tempfile.mkstemp(suffix=".wav")[1])
    sf.write(path, pcm, sr)
    subprocess.Popen(["afplay", str(path)])


async def main(variant: str, speak: bool) -> int:
    import modal
    image_set, state, answer = VARIANTS[variant]
    name = f"pilot_{variant}"
    out = pathlib.Path(os.environ.get("CROSSCHECK_OUT", ROOT / "replay")) / name  # eval/repeat.py sets it
    (out / "frames").mkdir(parents=True, exist_ok=True)
    frames = {s: frame_bytes(image_set, s) for s in ("upright", state)}
    box = [v * SCALE for v in BOXES[image_set]]
    pose = modal.Cls.from_name("crosscheck-sam3d-body", "Pose")()
    await pose.one.remote.aio(frames["upright"], box)  # wake the GPU container before the clock starts

    say_cc = None
    if speak:
        from voice.tts import speak as say_cc
    cc = Crosscheck(out, speak=say_cc)
    truth, heard, n = [], [], 0
    answer_task = None

    async def respond():
        """The pilot answers 1.5 s after Crosscheck asks."""
        await asyncio.sleep(1.5)
        audio = say(answer)
        t_start = time.time()
        if speak:
            play(audio, SR)
        r = await transcribe(audio)
        t = time.time()
        heard.append({"speaker": "capt", "spoken": answer, "transcript": r["text"], "s_from_start": round(t - t_start, 2)})
        cc.event("heard", t, speaker="capt", transcript=r["text"], spoken=answer)
        if r["text"].strip():
            cc.said(Fact("said", "pilot.voice", r["text"].strip(), t, "capt", source="muse-voice-transcribe"),
                    transcript=r["text"])

    t0 = time.time()
    while time.time() - t0 < RUN_S:
        captured = time.time()
        now_state = "upright" if captured - t0 < T_CHANGE else state
        data = frames[now_state]
        fname = f"frames/{n:05d}.jpg"
        (out / fname).write_bytes(data)
        n += 1
        res = await pose.one.remote.aio(data, box)
        t = time.time()
        truth.append({"t": captured, "state": now_state})
        if "pred_keypoints_3d" not in res:
            cc.event("pilot_error", t, frame=fname, error=res.get("error"))
            continue
        cc.observe_pilot(posture(res["pred_keypoints_3d"]), t, fname, captured=round(captured, 3))
        new = cc.decide(t)
        if answer and answer_task is None and any(f.key == "challenge" for f in new):
            answer_task = asyncio.create_task(respond())
    if answer_task:
        await answer_task

    with (out / "truth.jsonl").open("w") as f:
        for row in truth:
            f.write(json.dumps(row) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "scenario": name, "scorer": "pilot_watch", "variant": variant, "image_set": image_set, "state": state,
        "answer": answer, "expect_ask": EXPECT[variant][0], "expect_alert": EXPECT[variant][1],
        "t_start": t0, "t_change": t0 + T_CHANGE, "frame_box": [0, 0, 800, 800], "pilot_camera": True,
        "pose": "SAM 3D Body (facebook/sam-3d-body-vith) on Modal T4, one call per frame",
        "speech": "Muse Voice Transcribe, live (macOS voice for the pilot's answer)" if answer else "none",
        "heard": heard, "recorded": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=1))
    result = score(out)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(asyncio.run(main(args[0] if args else "slumped", speak="--speak" in sys.argv)))
