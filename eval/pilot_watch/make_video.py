"""Pilot-camera video for scenario 4: the Muse Image photos brought to life with LTX-2.5 (Lightricks,
open weights), run locally on the Mac through ltx-2-mlx.

Each clip is anchored on the labelled photos at both ends (image-to-video with a first and a last
keyframe), so the footage starts and ends on exactly the poses the ground truth names; LTX only
fills in the movement between. Clips per scenario, which the scenario cuts together live
(eval/scenarios/pilot_watch.py --video), playing idle and hold back and forth to fill the time:
  idle        upright -> upright: awake at the controls, nearly still (5 s)
  change      upright -> the scenario's state (5 s)
  hold        state -> state: stays like that (5 s)
  recover     (answers only) slumped -> upright: lifts the head to answer, cut in when asked (5 s)
Each transition is also pinned halfway, on a third photo Muse Image makes from the upright one
(<state>_mid.jpg: eyes closing, head starting to drop, hands loosening), because with only the two
ends pinned LTX filled the middle with purposeful movement: the captain turning to the camera and
gesturing on his way to unconsciousness. Generated footage, labelled as such everywhere it is
shown (the LTX-2 licence requires it); no real person is filmed.

  uv run python eval/pilot_watch/make_video.py [slumped asleep checklist answers]
writes eval/pilot_watch/video/<variant>/<clip>.mp4 (skips clips that exist).
Needs ltx-2-mlx (LTX_MLX_DIR, default ~/llm-stack/ltx-2-mlx) and the mlx-community/ltx-2.5-mlx weights.
"""
import base64
import io
import os
import pathlib
import subprocess
import sys
import time

import httpx
from dotenv import load_dotenv
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent
IMAGES = ROOT / "images"
OUT = ROOT / "video"
LTX = pathlib.Path(os.environ.get("LTX_MLX_DIR", pathlib.Path.home() / "llm-stack" / "ltx-2-mlx"))
MODEL = os.environ.get("LTX_MODEL", "mlx-community/ltx-2.5-mlx")
SIZE, FPS = 704, 24
# 5 s each (LTX wants 8k+1 frames). Longer clips drift: a first 10 s idle clip had the captain
# gesturing, talking and turning to the camera, which no pilot hand-flying does.
FRAMES = {"idle": 121, "change": 121, "hold": 121, "recover": 121}

WHO = {  # from make_images.py SETS
    "e": "a man in his fifties with a shaved head, white shirt and dark tie, bright midday sun with hard shadows",
    "f": "a woman in her thirties with short curly hair, dark blue uniform jacket, soft blue dawn light",
    "b": "a woman in her forties with tied-back hair, white uniform shirt with four stripes, at night, the cockpit lit by the displays and a dim dome light",
    "d": "a young man with short dark hair, short-sleeved white shirt, overcast daylight",
}
CAMERA = "Fixed camera on the glareshield of a Boeing 737 flight deck, looking back at the captain in the left seat"
STYLE = "Static camera, realistic, natural light, documentary footage."
# Prompts say only what happens: "does not gesture" put gestures into the clip.
IDLE = ("sits calmly in the seat flying the aircraft, eyes on the instruments and the windscreen ahead, both hands "
        "resting on the control column, breathing slowly, blinking now and then.")
MID = {  # the halfway photo for each state, as a Muse Image edit of the upright photo
    "slumped": "Same photo, same pilot and cockpit, but the pilot is losing consciousness: eyes half closed, head starting "
               "to drop forward, shoulders beginning to sag, hands loosening on the control column.",
    "slumped_side": "Same photo, same pilot and cockpit, but the pilot is falling asleep: eyes closing, head starting to "
                    "tilt towards the side window, upper body beginning to lean sideways, hands loosening on the controls.",
    "looking_down": "Same photo, same pilot and cockpit, but the pilot is lifting a paper checklist with the left hand and "
                    "starting to look down at it, sitting upright, right hand on the control column.",
}
VARIANTS = {  # image set, state, what happens in "change", what happens in "hold"
    "slumped": ("e", "slumped",
                "is flying, eyes forward, then slowly loses consciousness: his eyes close, his head nods forward and sinks until his chin rests on his chest, his body goes limp, his shoulders sag and both hands slip from the control column into his lap.",
                "sits unconscious, head down on his chest, eyes closed, hands in his lap, completely still except for slow breathing."),
    "asleep": ("f", "slumped_side",
               "is flying, eyes forward, then grows drowsy: her eyelids droop and close, her head and upper body slowly sag sideways until they rest against the side window, her arms go loose and her hands slip from the controls.",
               "is fast asleep against the side window, eyes closed, arms loose, completely still except for slow breathing."),
    "checklist": ("b", "looking_down",
                  "is flying, eyes forward, then calmly lifts a paper checklist with her left hand and lowers her eyes to read it, still sitting upright and alert, her right hand on the control column.",
                  "reads the paper checklist in her left hand, eyes moving down the page, sitting upright, alert."),
    "answers": ("d", "slumped",
                "is flying, eyes forward, then slowly slumps: his head drops forward onto his chest, his shoulders sag and his hands slide off the control column into his lap.",
                "sits slumped with his head down on his chest and his hands in his lap, still except for slow breathing."),
}
RECOVER = ("lifts his head, opens his eyes, sits back up in the seat, looks ahead out of the windscreen and says a few words, "
           "then puts both hands back on the control column.")


def he(text: str, image_set: str) -> str:
    return text if image_set in ("e", "d") else text.replace(" his ", " her ").replace("His ", "Her ").replace(" he ", " she ")


def midpoint(image_set: str, state: str) -> pathlib.Path:
    """The halfway photo, made once with Muse Image (Meta Model API, ~$0.01) and kept with the sets."""
    out = IMAGES / image_set / f"{state}_mid.jpg"
    if out.exists():
        return out
    load_dotenv(ROOT.parents[1] / ".env")
    buf = io.BytesIO()
    Image.open(IMAGES / image_set / "upright.jpg").save(buf, "PNG")
    with httpx.Client(trust_env=False, timeout=240) as c:
        for attempt in range(3):  # the content filter refuses some edits at random; a retry usually works
            r = c.post(os.environ.get("META_API_BASE", "https://api.meta.ai/v1") + "/images/edits",
                       headers={"Authorization": "Bearer " + os.environ["MODEL_API_KEY"]},
                       data={"model": "muse-image-1.0", "prompt": MID[state], "n": "1"},
                       files={"image": ("upright.png", buf.getvalue(), "image/png")})
            if r.status_code == 200:
                break
            print(f"  Muse Image edit refused ({r.status_code}), retrying", flush=True)
            time.sleep(3)
        r.raise_for_status()
    x = r.json()["data"][0]
    raw = base64.b64decode(x["b64_json"]) if x.get("b64_json") else httpx.get(x["url"], trust_env=False).content
    Image.open(io.BytesIO(raw)).convert("RGB").resize((1600, 1600)).save(out, quality=90)
    print(f"  {out.relative_to(ROOT)} (Muse Image)", flush=True)
    return out


def anchors(name: str, first: pathlib.Path, last: pathlib.Path, frames: int, mid: pathlib.Path | None) -> list[tuple[pathlib.Path, int, float]]:
    """Keyframes. idle and hold: their photo at both ends and at the thirds. change and recover:
    the start pose for the first ~0.7 s, the halfway photo at the middle, the end pose from ~4 s."""
    if name in ("change", "recover"):
        return [(first, 0, 1.0), (first, 16, 0.9), (mid, frames // 2, 0.8), (last, frames - 24, 0.9), (last, frames - 1, 1.0)]
    return [(first, 0, 1.0), (first, frames // 3, 0.8), (last, 2 * frames // 3, 0.8), (last, frames - 1, 1.0)]


def render(name: str, prompt: str, first: pathlib.Path, last: pathlib.Path, frames: int, out: pathlib.Path, seed: int,
           mid: pathlib.Path | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(LTX / ".venv" / "bin" / "ltx-2-mlx"), "generate", "--distilled", "--model", MODEL, "--no-audio",
           "-H", str(SIZE), "-W", str(SIZE), "-f", str(frames), "--frame-rate", str(FPS), "--seed", str(seed),
           "-p", prompt, "-o", str(out)]
    for img, idx, strength in anchors(name, first, last, frames, mid):
        cmd += ["--image", str(img), str(idx), str(strength)]
    t = time.time()
    # offline: the weights are already in the Hugging Face cache, and a fetch would try to pull every variant
    subprocess.run(cmd, check=True, cwd=LTX, env={**os.environ, "HF_HUB_OFFLINE": "1"},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  {out.relative_to(ROOT)}  {time.time() - t:.0f} s", flush=True)


def make(variant: str) -> None:
    image_set, state, change, hold = VARIANTS[variant]
    img = lambda name: IMAGES / image_set / f"{name}.jpg"
    d = OUT / variant
    lead = f"{CAMERA}. The captain, {WHO[image_set]},"
    clips = {"idle": (he(IDLE, image_set), "upright", "upright"),
             "change": (he(change, image_set), "upright", state),
             "hold": (he(hold, image_set), state, state)}
    if variant == "answers":
        clips["recover"] = (he(RECOVER, image_set), state, "upright")
    for i, (name, (motion, a, b)) in enumerate(clips.items()):
        if not (d / f"{name}.mp4").exists():
            mid = midpoint(image_set, state) if name in ("change", "recover") else None
            render(name, f"{lead} {motion} {STYLE}", img(a), img(b), FRAMES[name], d / f"{name}.mp4", seed=421 + i, mid=mid)


if __name__ == "__main__":
    for v in sys.argv[1:] or list(VARIANTS):
        print(v, flush=True)
        make(v)
