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
    "speed": ((10, 340, 140, 440), 3),
    "selected_speed": ((30, 60, 140, 115), 2),  # magenta MCP speed above the speed tape
}
ALT_BIG = (608, 345, 690, 420)   # thousands and hundreds, large digits (room for 5 digits)
ALT_ROLL = (686, 345, 740, 420)  # rolling tens column, 20 ft labels
# Fitted against X-Plane truth on the even frames of eval/recordings/t1_pfd (held-out odd
# frames: 29/30 within 20 ft, vs 27/30 with the eyeballed 37.5 px / 22.4 px).
ROLL_CENTRE_PX = 43.0            # pointer line, native px from the top of ALT_ROLL
ROLL_PX_PER_20FT = 25.8
# Flight mode annunciator: engaged modes on the top line (green), armed modes below (white).
FMA_ENGAGED = (140, 12, 580, 68)
FMA_ARMED = (140, 64, 580, 106)
FMA_COLUMNS = (290, 425)  # x boundaries thrust | roll | pitch, in PFD-crop px
FMA_VOCAB = ["MCP SPD", "FMC SPD", "THR HLD", "RETARD", "N1", "GA", "ARM", "HDG SEL", "HDG HOLD", "LNAV",
             "VOR/LOC", "LOC", "ROLLOUT", "TO/GA", "CWS R", "CWS P", "ALT HOLD", "ALT ACQ", "VNAV PTH",
             "VNAV SPD", "VNAV ALT", "V/S", "G/S", "FLARE"]
FMA_RE = re.compile("|".join(re.escape(w) for w in sorted(FMA_VOCAB, key=len, reverse=True)))
# Anchor: the barometric setting under the altitude tape ("29.92 IN." or "STD"), always present.
# Its top-left in the calibration PFD crop; regions shift with it when the view moves.
BARO_ANCHOR = (590, 665)
BARO = re.compile(r"^(\d{2}\.\d{2}|STD\b|\d{3,4}\s*HPA)")


@dataclass
class PanelReading:
    ias: int | None = None
    selected_speed: int | None = None
    altitude: int | None = None
    altitude_method: str = ""  # "roll" (to the pointer line) or "hundreds" (+/-50 ft)
    track_mag: int | None = None
    fma: list[str] = field(default_factory=list)        # engaged [thrust, roll, pitch]; "" = blank
    fma_armed: list[str] = field(default_factory=list)  # armed line, same columns
    raw: dict[str, list[str]] = field(default_factory=dict)
    alt_base: int | None = None
    roll_labels: list[tuple[int, float]] = field(default_factory=list)  # (label, y in native crop px)


def _ocr(img: Image.Image) -> list[tuple[str, float, tuple]]:
    return ocrmac.OCR(img, recognition_level="accurate").recognize(px=True)


def _shift(box: tuple, dx: int, dy: int) -> tuple:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def _anchor_offset(pfd: Image.Image) -> tuple[int, int]:
    """Offset of this frame's view from the calibration view, from the baro readout position."""
    for text, _, (x0, y0, x1, y1) in _ocr(pfd):
        if y0 > pfd.height * 0.7 and x0 > pfd.width * 0.6 and BARO.match(text.strip()):
            return round(x0 - BARO_ANCHOR[0]), round(y0 - BARO_ANCHOR[1])
    return 0, 0


def _crop(img: Image.Image, box: tuple, scale: int) -> Image.Image:
    # clamp to the image: black padding from an out-of-bounds crop degrades text recognition
    box = (max(0, box[0]), max(0, box[1]), min(img.width, box[2]), min(img.height, box[3]))
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


def _fma_line(pfd: Image.Image, box: tuple, scale: int = 2) -> list[str]:
    """Read one FMA line and assign each recognised mode word to a column by its x position."""
    cols = ["", "", ""]
    for text, _, (x0, _, x1, _) in _ocr(_crop(pfd, box, scale)):
        text = text.upper()
        for m in FMA_RE.finditer(text):
            # x of the word's centre, assuming roughly even character width across the box
            frac = (m.start() + m.end()) / 2 / max(len(text), 1)
            x = max(0, box[0]) + (x0 + frac * (x1 - x0)) / scale
            col = 0 if x < FMA_COLUMNS[0] else 1 if x < FMA_COLUMNS[1] else 2
            cols[col] = cols[col] or m.group(0)
    return cols


def _altitude(pfd: Image.Image, out: PanelReading, dx: int = 0, dy: int = 0) -> None:
    big = _ocr(_crop(pfd, _shift(ALT_BIG, dx, dy), 4))
    out.raw["alt_big"] = [t for t, _, _ in big]
    hundreds = [int(d) for t, _, _ in big if (d := re.sub(r"\D", "", t)) and len(d) <= 3]
    if not hundreds:
        return
    base = hundreds[0] * 100
    out.alt_base = base
    estimates: list[float] = []
    for scale in (5, 4, 6, 3):  # text recognition on the rolling digits is scale-sensitive
        roll = _ocr(_crop(pfd, _shift(ALT_ROLL, dx, dy), scale))
        out.raw.setdefault("alt_roll", []).extend(t for t, _, _ in roll)
        centre = ROLL_CENTRE_PX * scale
        px_per_ft = ROLL_PX_PER_20FT * scale / 20
        for text, _, (x0, y0, x1, y1) in roll:
            d = re.sub(r"\D", "", text)
            if len(d) != 2 or int(d) % 20:
                continue
            out.roll_labels.append((int(d), (y0 + y1) / 2 / scale))
            # labels above the pointer line read higher. The pointer's place within the hundred is
            # each label's estimate mod 100: a "00" above the line is the next hundred, an "80"
            # below it may be the previous one.
            estimates.append((int(d) + ((y0 + y1) / 2 - centre) / px_per_ft) % 100)
        if estimates:
            break
    if estimates:
        # unwrap around the first estimate so 98 and 2 average to 100, not 50
        ref = estimates[0]
        unwrapped = sorted(e - 100 if e - ref > 50 else e + 100 if ref - e > 50 else e for e in estimates)
        out.altitude = int(round(base + unwrapped[len(unwrapped) // 2]))
        out.altitude_method = "roll"
    else:
        out.altitude = base + 50
        out.altitude_method = "hundreds"


def read_frame(frame: Image.Image) -> PanelReading:
    out = PanelReading()
    pfd = frame.crop(CALIBRATION["pfd"])
    dx, dy = _anchor_offset(pfd)
    out.raw["offset"] = [f"{dx},{dy}"]
    for name, (box, scale) in PFD_REGIONS.items():
        crop = _crop(pfd, _shift(box, dx, dy), scale)
        res = _ocr(crop)
        out.raw[name] = [t for t, _, _ in res]
        if name == "selected_speed":
            out.selected_speed = _number_near_centre(res, crop.height, 80, 400)
        elif name == "speed":
            out.ias = _number_near_centre(res, crop.height, 30, 400)
            for retry in (4, 2, 5):  # the rolling last digit makes recognition scale-sensitive
                if out.ias is not None:
                    break
                c2 = _crop(pfd, _shift(box, dx, dy), retry)
                out.ias = _number_near_centre(_ocr(c2), c2.height, 30, 400)
    out.fma = _fma_line(pfd, _shift(FMA_ENGAGED, dx, dy))
    out.fma_armed = _fma_line(pfd, _shift(FMA_ARMED, dx, dy))
    _altitude(pfd, out, dx, dy)
    nd = _ocr(frame.crop(_shift(CALIBRATION["nd_top"], dx, dy)))
    out.raw["nd_top"] = [t for t, _, _ in nd]
    # the readout box border often splits or prefixes the digits: "TRK", "1246] MAG"
    text = " ".join(t for t, _, _ in sorted(nd, key=lambda r: r[2][0]))
    if m := re.search(r"(?:TRK|HDG)\D{0,6}(\d{3,4})", text):
        out.track_mag = int(m.group(1)[-3:]) % 360
    return out
