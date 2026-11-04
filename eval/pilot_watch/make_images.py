"""Generate the pilot-camera images for scenario 4 with Muse Image (Meta Model API, ~$0.01 each).

Each set is one generated photo of a captain at the controls (upright.jpg, the camera install the
seat box in boxes.json is calibrated on), then edits of that same photo into the states the pilot
watch must tell apart. Set "a" was used to choose the posture thresholds; sets "b"-"d" (different
people, light and camera angle) were held out to test them, and "e"-"f" were made after the
thresholds were revised, as a clean test. Synthetic, so no real person is filmed; each image is
labelled with the state it was asked to show, which is the ground truth for scoring.

The states were remade on 5 Nov 2026 (the first edits are in images_before_remake/): "slumped"
read as a pilot looking down at his lap, almost the same as reading the checklist, and the
asleep image as a pilot resting his head. The prompts now ask for poses nobody would mistake:
  slumped       incapacitated: slumped low and sideways, head on the shoulder, arms hanging off the armrests
  slumped_side  asleep (name kept, the code and results use it): slouched, head rolled sideways against
                the seat, eyes shut, hands loose in the lap. Chin-on-chest versions were tried first and
                every one read as a pilot looking down at something in his lap from this camera.
  looking_down  awake: upright, eyes on a paper checklist held up in front of the chest
Every image was looked at after it was made; the unclear ones were made again with --redo (the
rejected tries are moved to images_rejected/, never deleted).

Run:  uv run python eval/pilot_watch/make_images.py                 (skips images that already exist)
      uv run python eval/pilot_watch/make_images.py --redo e/slumped  (moves that image aside, makes it again)
"""
import base64
import io
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import load_dotenv
from PIL import Image

DIR = pathlib.Path(__file__).resolve().parent / "images"
REJECTED = DIR.parent / "images_rejected"
BASE = "upright"
SCENE = ("Photo from a small camera mounted on the glareshield of a Boeing 737 flight deck, looking back at the captain "
         "in the left seat. The pilot sits upright, alert, eyes open, looking ahead out of the windscreen, hands on the "
         "control column, headset on, {who}. {light}, realistic, wide-angle, whole upper body and head in frame{angle}.")
SETS = {
    "a": None,  # the first generated photo (prompt as SCENE with a white-shirted man, daylight)
    "b": dict(who="a woman in her forties with tied-back hair, white uniform shirt with four stripes", light="Night, cockpit lit by the displays and a dim dome light", angle=""),
    "c": dict(who="an older man with a grey beard and glasses, dark uniform jacket", light="Low evening sun through the side window", angle=", camera slightly higher and further right"),
    "d": dict(who="a young man with short dark hair, short-sleeved white shirt", light="Overcast daylight", angle=", camera closer, pilot fills more of the frame"),
    # made after the thresholds were revised, as a clean test of them
    "e": dict(who="a man in his fifties with a shaved head, white shirt and dark tie", light="Bright midday sun with hard shadows", angle=", camera slightly to the left and lower"),
    "f": dict(who="a woman in her thirties with short curly hair, dark blue uniform jacket", light="Dawn, soft blue light", angle=", camera at head height to the pilot's right"),
}
HER = {"b", "f"}  # the pilots' pronouns, for the edit prompts
EDITS = {  # each state: prompts tried in order, the next one only if the content filter refuses the first
    # Incapacitated. Words like "limp", "fainted", "passed out" or "mouth open" are refused by the
    # content filter; "unresponsive" and "unconscious" are not.
    "slumped": ["Same photo, same pilot and cockpit, but the pilot is unresponsive, slumped low and sideways in the seat, "
                "{his} head drooping onto {his} shoulder, eyes closed, {his} arms hanging down off the armrests with the "
                "hands dangling near the floor.",
                "Same photo, same pilot and cockpit, but the pilot is unresponsive, slumped low and sideways in the seat "
                "with {his} head drooping onto {his} shoulder, eyes closed, arms hanging at {his} sides."],
    # Asleep (the file name is kept: the code and the results use it). Chin-on-chest tries all read
    # as a pilot looking down at something in the lap from this camera, so the head rolls sideways.
    "slumped_side": ["Same photo, same pilot and cockpit, but the pilot is fast asleep, slouched low in the seat: {his} "
                     "head has rolled sideways and rests against the side of the seat, tilted far over, eyes shut, face "
                     "relaxed, and {his} hands lie loose and apart in {his} lap, holding nothing.",
                     "Same photo, same pilot and cockpit, but the pilot is fast asleep, slouched low in the seat with {his} "
                     "head drooping onto {his} shoulder, eyes shut, {his} hands loose in {his} lap, holding nothing."],
    "looking_down": ["Same photo, same pilot and cockpit, but the pilot is awake and alert, sitting upright, holding a "
                     "white paper checklist up in front of {his} chest with {his} left hand and reading it, eyes open, "
                     "right hand on the control column."],
}
# Set e's camera has the glareshield in the foreground; the edits that showed his arms hanging took
# it away (the camera would jump in the video), and the ones that kept it read as looking down. So
# e/slumped is a second edit of a first one that kept the camera (images_rejected/e/slumped.first_edit_of_chain.jpg).
CHAIN = {"e/slumped": "Same photo, same framing, but his head has now tipped over sideways and rests on his shoulder, "
                      "his upper body leaning sideways against the side of the seat, eyes closed."}
RETRIES = 3  # the content filter refuses some edits at random; a retry usually works


def save(data: dict, path: pathlib.Path) -> None:
    x = data["data"][0]
    raw = base64.b64decode(x["b64_json"]) if x.get("b64_json") else httpx.get(x["url"], trust_env=False).content
    Image.open(io.BytesIO(raw)).convert("RGB").save(path, quality=90)


def edit(c: httpx.Client, base_url: str, headers: dict, upright: pathlib.Path, prompts: list[str], out: pathlib.Path) -> bool:
    """One Muse Image edit of the set's upright photo; True when written."""
    buf = io.BytesIO()
    Image.open(upright).save(buf, "PNG")
    his = "her" if out.parent.name in HER else "his"
    for i, prompt in enumerate(prompts):
        for attempt in range(RETRIES):
            r = c.post(base_url + "/images/edits", headers=headers,
                       data={"model": "muse-image-1.0", "prompt": prompt.format(his=his), "n": "1"},
                       files={"image": ("upright.png", buf.getvalue(), "image/png")})
            if r.status_code == 200:
                save(r.json(), out)
                return True
            print(f"  {out.parent.name}/{out.name} prompt {i + 1}: {r.status_code} {r.text[:200]}", flush=True)
            time.sleep(3)
    return False


def set_aside(path: pathlib.Path) -> None:
    """A rejected image (and its pose) goes to images_rejected/<set>/<state>.<n>.jpg; nothing is deleted."""
    d = REJECTED / path.parent.name
    d.mkdir(parents=True, exist_ok=True)
    n = 1 + sum(1 for _ in d.glob(f"{path.stem}.*.jpg"))
    for p in (path, path.with_suffix(".pose.json")):
        if p.exists():
            p.rename(d / f"{path.stem}.{n}{''.join(p.suffixes)}")


def main(redo: list[str]) -> int:
    load_dotenv(DIR.parents[2] / ".env")
    headers = {"Authorization": "Bearer " + os.environ["MODEL_API_KEY"]}
    base_url = os.environ.get("META_API_BASE", "https://api.meta.ai/v1")
    for item in redo:
        set_aside(DIR / f"{item}.jpg")
    failed = 0
    with httpx.Client(trust_env=False, timeout=240) as c:
        for set_name, scene in SETS.items():
            d = DIR / set_name
            d.mkdir(parents=True, exist_ok=True)
            if not (d / f"{BASE}.jpg").exists():
                r = c.post(base_url + "/images/generations", headers=headers,
                           json={"model": "muse-image-1.0", "prompt": SCENE.format(**scene), "n": 1, "size": "1024x1024"})
                if r.status_code != 200:
                    print(set_name, BASE, r.status_code, r.text[:500])
                    return 1
                save(r.json(), d / f"{BASE}.jpg")
                print("wrote", set_name, BASE)
        todo = [(DIR / s / f"{BASE}.jpg", prompts, DIR / s / f"{name}.jpg")
                for s in SETS for name, prompts in EDITS.items() if not (DIR / s / f"{name}.jpg").exists()]
        def make(upright, prompts, out):
            ok = edit(c, base_url, headers, upright, prompts, out)
            key = f"{out.parent.name}/{out.stem}"
            if ok and key in CHAIN:  # edit the first edit again
                first = out.with_name(out.stem + ".first.jpg")
                out.rename(first)
                ok = edit(c, base_url, headers, first, [CHAIN[key]], out)
                (REJECTED / out.parent.name).mkdir(parents=True, exist_ok=True)
                first.rename(REJECTED / out.parent.name / f"{out.stem}.first_edit_of_chain.jpg" if ok else out)
            return ok

        with ThreadPoolExecutor(6) as pool:  # each edit takes ~25 s
            done = list(pool.map(lambda job: make(*job), todo))
        for (_, _, out), ok in zip(todo, done):
            print("wrote" if ok else "FAILED", f"{out.parent.name}/{out.name}")
            failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    redo = [args[i + 1] for i, a in enumerate(args) if a == "--redo"]
    sys.exit(main(redo))
