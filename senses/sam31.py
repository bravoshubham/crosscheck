"""SAM 3.1 (Meta Model API): find things in a camera frame from a one- or two-word concept.

Two jobs here:
  - the panel camera's pre-flight check (camera_check): both flight displays must be found where
    the readers' calibration expects them, or the run does not start. A real camera is never
    exactly where it was calibrated, and a window in front of the simulator shows no displays.
  - the pilot camera: the box around the captain that SAM 3D Body needs (pilot_box).

The API is the Responses API with model "sam-3.1": an input_image and an input_text concept. It
answers in text, one "<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|>" per instance (and a mask, not
used here). Short nouns work ("screen", "person"); "primary flight display" found nothing.
"""
import base64
import io
import os
import re

import httpx
from dotenv import load_dotenv
from PIL import Image

MODEL = "sam-3.1"


def find(image: Image.Image | bytes, concept: str, client: httpx.Client | None = None) -> list[tuple[int, int, int, int]]:
    """Boxes (x1, y1, x2, y2) in the image's own pixels for every instance of the concept."""
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "JPEG", quality=90)
        image = buf.getvalue()
    body = {"model": MODEL, "input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(image).decode()},
        {"type": "input_text", "text": concept}]}]}
    own = client is None
    client = client or httpx.Client(trust_env=False, timeout=120)
    try:
        r = client.post(os.environ.get("META_API_BASE", "https://api.meta.ai/v1") + "/responses", json=body,
                        headers={"Authorization": "Bearer " + os.environ["MODEL_API_KEY"]})
        r.raise_for_status()
    finally:
        if own:
            client.close()
    text = "".join(c.get("text", "") for o in r.json().get("output", []) for c in o.get("content", []))
    boxes = []
    for b in re.findall(r"<\|box;([^|]*)\|>", text):
        v = {k: float(x) for k, x in re.findall(r"(\w+)=([\d.]+)", b)}
        boxes.append((int(v["x1"]), int(v["y1"]), int(v["x2"]), int(v["y2"])))
    return boxes


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def camera_check(frame: Image.Image, expected: dict[str, tuple[int, int, int, int]], min_iou: float = 0.7) -> dict:
    """Find the screens in a full-screen capture and match each expected display box to one.
    Returns {"ok", "found", "matches": {name: (box, iou)}}."""
    found = find(frame, "screen")
    matches = {}
    for name, want in expected.items():
        best = max(found, key=lambda b: iou(b, want), default=None)
        matches[name] = {"box": best, "iou": round(iou(best, want), 2) if best else 0.0}
    return {"ok": all(m["iou"] >= min_iou for m in matches.values()), "found": found, "matches": matches, "model": MODEL}


def pilot_box(image: Image.Image | bytes) -> tuple[int, int, int, int] | None:
    """The captain: the largest person SAM 3.1 finds."""
    boxes = find(image, "person")
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), default=None)


def inside_check(frame: Image.Image, concept: str, region: tuple[int, int, int, int], slack: int = 12) -> dict:
    """For a reader whose crop has margins (the flap gauge): the concept must be found within it."""
    found = find(frame, concept)
    hit = next((b for b in found if b[0] >= region[0] - slack and b[1] >= region[1] - slack
                and b[2] <= region[2] + slack and b[3] <= region[3] + slack), None)
    return {"ok": hit is not None, "found": found, "matches": {concept: {"box": hit, "inside": list(region)}}, "model": MODEL}
