"""Read the 737 panel from a camera or screen frame.

Per-frame reading is plain code: crop each calibrated region and run Apple's on-device text
recognition (Vision framework via ocrmac). Regions come from a calibration step: fixed boxes
for now, located by SAM 3.1 once the Meta API key is set. The ND shows track, not heading,
as a digital readout, so direction is reported as magnetic track.
"""
import re
from dataclasses import dataclass, field

from ocrmac import ocrmac
from PIL import Image

# Full-screen pixel boxes for the X-Plane 737 view used in T1 (head pitch -22 deg, FOV 50 deg,
# 3600x2338 capture). (left, top, right, bottom)
CALIBRATION = {
    "pfd": (1340, 1330, 2150, 2120),
    "nd_top": (2275, 1340, 3080, 1440),
}
# Boxes inside the PFD crop, with an upscale factor for small digits.
PFD_REGIONS = {
    "fma_thrust": ((140, 20, 290, 72), 1),
    "fma_roll": ((290, 20, 425, 72), 1),
    "fma_pitch": ((425, 20, 580, 72), 1),
    "speed": ((10, 340, 140, 440), 3),
}
ALT_BIG = (630, 345, 690, 420)   # thousands and hundreds, large digits
ALT_ROLL = (686, 345, 740, 420)  # rolling tens column, 20 ft labels
ROLL_PX_PER_20FT = 22.4          # measured on the T1 view
FMA_WORDS = re.compile(r"[A-Z][A-Z /]+")


@dataclass
class PanelReading:
    ias: int | None = None
    altitude: int | None = None
    altitude_method: str = ""  # "roll" (to the pointer line) or "hundreds" (+/-50 ft)
    track_mag: int | None = None
    fma: list[str] = field(default_factory=list)
    raw: dict[str, list[str]] = field(default_factory=dict)


def _ocr(img: Image.Image) -> list[tuple[str, float, tuple]]:
    return ocrmac.OCR(img, recognition_level="accurate").recognize(px=True)


def _crop(img: Image.Image, box: tuple, scale: int) -> Image.Image:
    c = img.crop(box)
    return c.resize((c.width * scale, c.height * scale), Image.LANCZOS) if scale > 1 else c


def _number_near_centre(results, height: float, lo: int, hi: int) -> int | None:
    """The number whose box is closest to the vertical centre of the crop (the pointer line)."""
    best = None
    for text, conf, (x0, y0, x1, y1) in results:
        digits = re.sub(r"\D", "", text)
        if not digits or not lo <= int(digits) <= hi:
            continue
        dist = abs((y0 + y1) / 2 - height / 2)
        if best is None or dist < best[0]:
            best = (dist, int(digits))
    return None if best is None else best[1]


def _altitude(pfd: Image.Image, out: PanelReading) -> None:
    big = _ocr(_crop(pfd, ALT_BIG, 4))
    out.raw["alt_big"] = [t for t, _, _ in big]
    hundreds = [int(d) for t, _, _ in big if (d := re.sub(r"\D", "", t)) and len(d) <= 3]
    if not hundreds:
        return
    base = hundreds[0] * 100
    scale = 5
    roll = _ocr(_crop(pfd, ALT_ROLL, scale))
    out.raw["alt_roll"] = [t for t, _, _ in roll]
    centre = (ALT_ROLL[3] - ALT_ROLL[1]) * scale / 2
    px_per_ft = ROLL_PX_PER_20FT * scale / 20
    estimates = []
    for text, _, (x0, y0, x1, y1) in roll:
        d = re.sub(r"\D", "", text)
        if len(d) != 2 or int(d) % 20:
            continue
        v = int(d)
        # label above the pointer line reads higher; a label of 80-ish below "00" belongs to the hundred below
        offset_ft = ((y0 + y1) / 2 - centre) / px_per_ft
        at_centre = v + offset_ft
        if at_centre > 110:      # e.g. "80" sitting just below a 00 that belongs to this hundred
            at_centre -= 100
        elif at_centre < -10:
            at_centre += 100
        estimates.append(at_centre)
    if estimates:
        out.altitude = int(round(base + sorted(estimates)[len(estimates) // 2]))
        out.altitude_method = "roll"
    else:
        out.altitude = base + 50
        out.altitude_method = "hundreds"


def read_frame(frame: Image.Image) -> PanelReading:
    out = PanelReading()
    pfd = frame.crop(CALIBRATION["pfd"])
    for name, (box, scale) in PFD_REGIONS.items():
        crop = _crop(pfd, box, scale)
        res = _ocr(crop)
        out.raw[name] = [t for t, _, _ in res]
        if name.startswith("fma_"):
            words = " ".join(t for t, _, _ in sorted(res, key=lambda r: r[2][0]))
            out.fma.append(m.group(0).strip() if (m := FMA_WORDS.search(words)) else "")
        elif name == "speed":
            out.ias = _number_near_centre(res, crop.height, 30, 400)
    _altitude(pfd, out)
    nd = _ocr(frame.crop(CALIBRATION["nd_top"]))
    out.raw["nd_top"] = [t for t, _, _ in nd]
    for text, _, _ in nd:
        if m := re.search(r"(?:TRK|HDG)\s*(\d{3})", text):
            out.track_mag = int(m.group(1)) % 360
    return out
