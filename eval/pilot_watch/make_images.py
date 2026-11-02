"""Generate the pilot-camera images for scenario 4 with Muse Image (Meta Model API, ~$0.01 each).

Each set is one generated photo of a captain at the controls, then edits of that same photo into
the states the pilot watch must tell apart. Set "a" was used to choose the posture thresholds;
sets "b"-"d" (different people, light and camera angle) were held out to test them, and
"e"-"f" were made after the thresholds were revised, as a clean test. Synthetic, so
no real person is filmed; each image is labelled with the state it was asked to show, which is
the ground truth for scoring.

Run:  uv run python eval/pilot_watch/make_images.py   (skips images that already exist)
"""
import base64
import io
import os
import pathlib
import sys

import httpx
from dotenv import load_dotenv
from PIL import Image

DIR = pathlib.Path(__file__).resolve().parent / "images"
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
EDITS = {
    "slumped": "Same photo, same pilot and cockpit, but the pilot is unconscious: head dropped forward, chin on chest, "
               "eyes closed, shoulders slumped, both hands fallen off the control column into the lap.",
    # "passed out" is refused by the content filter; asleep against the window is the same posture
    "slumped_side": "Same photo, same pilot and cockpit, but the pilot is fast asleep: head and upper body leaning "
                    "sideways against the side window, eyes closed, arms resting loosely, hands off the controls.",
    "looking_down": "Same photo, same pilot and cockpit, but the pilot is awake and alert, looking down at a paper "
                    "checklist held in the left hand, eyes open, sitting upright.",
}


def save(data: dict, path: pathlib.Path) -> None:
    x = data["data"][0]
    raw = base64.b64decode(x["b64_json"]) if x.get("b64_json") else httpx.get(x["url"], trust_env=False).content
    Image.open(io.BytesIO(raw)).convert("RGB").save(path, quality=90)


def main() -> int:
    load_dotenv(DIR.parents[2] / ".env")
    headers = {"Authorization": "Bearer " + os.environ["MODEL_API_KEY"]}
    base_url = os.environ.get("META_API_BASE", "https://api.meta.ai/v1")
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
        for name, prompt in EDITS.items():
            out = d / f"{name}.jpg"
            if out.exists():
                continue
            buf = io.BytesIO()
            Image.open(d / f"{BASE}.jpg").save(buf, "PNG")
            r = c.post(base_url + "/images/edits", headers=headers,
                       data={"model": "muse-image-1.0", "prompt": prompt, "n": "1"},
                       files={"image": ("upright.png", buf.getvalue(), "image/png")})
            if r.status_code != 200:
                print(set_name, name, r.status_code, r.text[:500])
                return 1
            save(r.json(), out)
            print("wrote", set_name, out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
