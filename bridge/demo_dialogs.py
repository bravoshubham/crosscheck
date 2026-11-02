"""Dismiss the X-Plane 12 demo's own dialogs so scenarios can run unattended.

Only three known dialogs, identified by their titles, and only their harmless button:
  "UNLOCK X-PLANE"         -> "Use Demo"   (never "Buy" or "Unlock")
  "DEMO MODE ACTIVATED"    -> "Understood" (the demo allows 60 minutes of flying at a time)
  "LEAVING THE DEMO AREA"  -> "OK"
Anything else on screen: do nothing. One more is recognised but not clicked: "Your demo time is
up" (also titled UNLOCK X-PLANE, with only Buy / Unlock buttons). The flight controls stop
working until X-Plane restarts, so bridge/xplane.py restarts it. The button is found by its text with Apple's on-device
text recognition, and clicked with a synthetic mouse event (X-Plane draws its own UI, so
accessibility clicks do not reach it).

  uv run python -m bridge.demo_dialogs     # one pass: prints what it saw and did
"""
import subprocess
import tempfile
import time

import Quartz
from ocrmac import ocrmac
from PIL import Image

DIALOGS = {"UNLOCK X-PLANE": "Use Demo", "DEMO MODE ACTIVATED": "Understood", "LEAVING THE DEMO AREA": "OK"}


def _screen() -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        subprocess.run(["screencapture", "-x", f.name], check=True)
        return Image.open(f.name).copy()


def _click(x_pt: float, y_pt: float) -> None:
    for kind in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
        ev = Quartz.CGEventCreateMouseEvent(None, kind, (x_pt, y_pt), Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(0.08)


def find_dialog(img: Image.Image) -> tuple[str, tuple[float, float]] | None:
    """Return (dialog title, button centre in screenshot pixels) if a known dialog is showing."""
    words = ocrmac.OCR(img, recognition_level="accurate").recognize(px=True)
    titles = [(t.strip().upper(), box) for t, _, box in words]
    for title, button in DIALOGS.items():
        hit = next((box for t, box in titles if t == title), None)
        if hit is None:
            continue
        # the button must sit below the title, inside the same dialog (within ~900 px)
        cands = [box for t, _, box in words if t.strip() == button and hit[1] < box[1] < hit[1] + 900
                 and abs((box[0] + box[2]) / 2 - (hit[0] + hit[2]) / 2) < 900]
        if cands:
            b = cands[0]
            return title, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return None


def demo_time_up() -> bool:
    """True if the demo's 60 minutes are over (the sim keeps drawing, but ignores its controls)."""
    words = ocrmac.OCR(_screen(), recognition_level="fast").recognize()
    return "demo time is up" in " ".join(t for t, _, _ in words).lower()


def dismiss_once() -> str | None:
    """Bring X-Plane forward; if a known demo dialog is showing, click its harmless button."""
    subprocess.run(["open", "-a", "X-Plane"], check=False)
    time.sleep(1.0)
    img = _screen()
    found = find_dialog(img)
    if found is None:
        return None
    title, (px, py) = found
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    scale = img.width / bounds.size.width  # screenshot pixels per screen point (Retina = 2)
    _click(px / scale, py / scale)
    time.sleep(1.0)
    return title


def dismiss_all(max_passes: int = 3) -> list[str]:
    done = []
    for _ in range(max_passes):
        title = dismiss_once()
        if title is None:
            break
        done.append(title)
    return done


if __name__ == "__main__":
    print("dismissed:", dismiss_all() or "nothing (no known demo dialog on screen)")
