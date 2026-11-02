"""Render a demo video from the recordings: the cockpit camera, what was said / set / doing,
and each callout, with the radio lines and Crosscheck's voice at the times they happened.

  uv run python cli.py debrief && uv run python cli.py video     # writes site/crosscheck_demo.mp4

Everything shown comes from site/data.js (built from replay/ by cli.py debrief); nothing is
staged. Each scenario plays in real time, trimmed to a window around what happens in it.
"""
import json
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import resample_poly

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
W, H, FPS, SR = 1280, 720, 10, 22050
BG, PANEL, LINE = (10, 13, 17), (17, 22, 29), (35, 44, 56)
TEXT, MUTED, SAID, SET, OK, CALL = (231, 236, 242), (137, 149, 163), (121, 184, 255), (228, 107, 240), (79, 209, 139), (255, 92, 92)
SANS = "/System/Library/Fonts/HelveticaNeue.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
PLAN = [  # scenario, seconds before the first event, seconds after the last callout
    ("altitude_bust", 3, 6), ("altitude_readback", 3, 5), ("altitude_correct", 3, 4), ("autothrottle_hold", 2, 6),
    ("flaps_wrong", 2, 5), ("flaps_correct", 2, 4), ("pilot_slumped", 3, 5), ("pilot_answers", 3, 6), ("pilot_checklist", 3, 3),
]
CARDS = {
    "intro": ["Crosscheck", "An AI that monitors the pilots.",
              "It hears what was said, reads what was set on the panel, watches what the aircraft is doing,",
              "and speaks up when they disagree. Flown in X-Plane 12; nothing below is read from the simulator."],
    "altitude_bust": ["Altitude", "ATC clears 11,000. Speech: Muse Voice Transcribe. Panel: camera."],
    "autothrottle_hold": ["Autothrottle", "On final to Portland, the autothrottle drops out (the Asiana 214 pattern)."],
    "flaps_wrong": ["Flaps", "“Flaps fifteen” called. Muse Spark reads the analog flap gauge."],
    "pilot_slumped": ["Pilot watch", "Opt-in pilot camera; posture from SAM 3D Body. It asks before it alerts.",
                      "Pilot images generated with Muse Image."],
    "outro": ["Crosscheck", "Meta models: Muse Voice Transcribe · Muse Spark · SAM 3D Body · Muse Image · Muse Glimmer · MMS-TTS",
              "29/29 repeated live runs pass with the current rules.",
              "Every run replays from the repository: uv run python cli.py replay --all"],
}


def font(path: str, size: int):
    return ImageFont.truetype(path, size)


F = {"title": font(SANS, 56), "big": font(SANS, 30), "body": font(SANS, 22), "small": font(SANS, 17),
     "lab": font(MONO, 15), "val": font(MONO, 34), "call": font(SANS, 28)}


def last(items, t):
    out = None
    for x in items:
        if x["t"] <= t:
            out = x
        else:
            break
    return out


def wrap(d: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if d.textlength((cur + " " + w).strip(), font=f) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + ([cur] if cur else [])


def card(lines: list[str]) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y = 250
    d.text((100, y), lines[0], font=F["title"], fill=TEXT)
    y += 90
    for i, line in enumerate(lines[1:]):
        for l in wrap(d, line, F["big"] if i == 0 else F["body"], W - 200):
            d.text((100, y), l, font=F["big"] if i == 0 else F["body"], fill=TEXT if i == 0 else MUTED)
            y += 44 if i == 0 else 34
    return im


def nearest_truth(rec, t):
    out, best = {}, {}
    for r in rec["truth"]:
        for k, v in r.items():
            if k != "t" and v is not None and (k not in best or abs(r["t"] - t) < best[k]):
                best[k], out[k] = abs(r["t"] - t), v
    return out


def block(d, x, y, w, label, color, lines, sub=None):
    d.rounded_rectangle((x, y, x + w, y + 118), 10, fill=PANEL, outline=LINE)
    d.text((x + 14, y + 10), label, font=F["lab"], fill=color)
    yy = y + 34
    for i, (txt, f, c) in enumerate(lines):
        d.text((x + 14, yy), txt, font=F[f], fill=c)
        yy += {"val": 42, "big": 38, "body": 28}.get(f, 24)


def frame_image(rec, t, cache):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((40, 24), "CROSSCHECK", font=F["lab"], fill=TEXT)
    d.text((170, 20), rec["title"].replace(" · Spark reader", ""), font=F["body"], fill=TEXT)
    d.text((40, 52), rec["blurb"][:120], font=F["small"], fill=MUTED)
    # camera
    r = last(rec["readings"], t) or (rec["readings"][0] if rec["readings"] else None)
    box = (40, 90, 820, 590)
    if r and r.get("frame"):
        if r["frame"] not in cache:
            cache[r["frame"]] = Image.open(SITE / r["frame"]).convert("RGB")
        src = cache[r["frame"]].copy()
        src.thumbnail((box[2] - box[0], box[3] - box[1]))
        im.paste(src, (box[0] + (box[2] - box[0] - src.width) // 2, box[1] + (box[3] - box[1] - src.height) // 2))
    onair = next((h for h in rec["heard"] if h["start"] <= t < h["t"]), None)
    if onair:
        who = {"atc": "ATC · RADIO", "capt": "CAPTAIN · INTERCOM"}.get(onair["speaker"], "CREW · INTERCOM")
        d.rectangle((box[0], box[1], box[2], box[1] + 54), fill=(10, 13, 17))
        d.text((box[0] + 12, box[1] + 6), who, font=F["lab"], fill=SAID)
        d.text((box[0] + 12, box[1] + 26), onair["spoken"], font=F["small"], fill=TEXT)
    speaking = next((c for c in rec["callouts"] if c["t"] <= t < c["t"] + 6), None)
    if speaking:
        lines = wrap(d, "“" + speaking["text"] + "”", F["call"], box[2] - box[0] - 60)
        y0 = box[3] - 20 - 40 * len(lines)
        d.rounded_rectangle((box[0] + 20, y0 - 12, box[2] - 20, box[3] - 14), 10, fill=(40, 14, 16), outline=CALL, width=2)
        for i, l in enumerate(lines):
            d.text((box[0] + 40, y0 + 40 * i), l, font=F["call"], fill=TEXT)
    # said / set / doing / crosscheck
    x, w = 840, 400
    heard = last(rec["heard"], t)
    facts = [f for f in rec["said"] if f["t"] <= t]
    said_lines = [("“" + heard["transcript"].strip()[:44] + "”", "small", TEXT)] if heard else [("nothing yet" if rec["heard"] else "no speech", "small", MUTED)]
    for f in facts[-2:]:
        v = f["value"]
        said_lines.append(("› " + {"clearance.altitude": f"clearance {v:,} ft", "readback.altitude": f"readback {v:,} ft",
                                         "callout.flaps": f"callout flaps {v}", "pilot.voice": "captain answered"}.get(f["key"], f["key"]), "small", SAID))
    block(d, x, 90, w, "SAID · Muse Voice Transcribe", SAID, said_lines[:3])
    tr = nearest_truth(rec, t)
    k = rec["kind"]
    if k == "altitude":
        set_l = [(f"{r['selected_altitude']:,} ft" if r and r.get("selected_altitude") else "—", "val", SET), ("MCP altitude, read by the camera", "small", MUTED)]
        do_l = [(f"{r['altitude']:,} ft" if r and r.get("altitude") else "—", "val", TEXT), (f"X-Plane {tr.get('alt', 0):,.0f} ft (scoring only)", "small", MUTED)]
    elif k == "autothrottle_hold":
        thr = (r["fma"][0] or "BLANK") if r and r.get("fma") and any(r["fma"]) else "—"
        set_l = [(thr, "val", CALL if thr == "BLANK" else SET), ("thrust mode, read by the camera", "small", MUTED)]
        do_l = [(f"{r['ias']} kt" if r and r.get("ias") else "—", "val", TEXT), (f"target {rec['meta'].get('target_kt')} kt · X-Plane {tr.get('ias', 0):.0f} kt", "small", MUTED)]
    elif k == "flaps":
        g = last([x for x in rec["readings"] if x.get("flaps") is not None], t)
        set_l = [("lever not in view", "body", MUTED), ("judged by the flap gauge", "small", MUTED)]
        do_l = [(f"flaps {g['flaps']}" if g else "—", "val", TEXT), ("flap gauge, read by Muse Spark", "small", MUTED)]
    else:
        set_l = [(f"{r['neck_flex']:.0f}°" if r and r.get("neck_flex") is not None else "—", "val", SET), ("neck flexion · alert above 40°", "small", MUTED)]
        do_l = [(f"{r['wrists']:.2f}" if r and r.get("wrists") is not None else "—", "val", TEXT), ("hand height · alert below 0.31", "small", MUTED)]
    block(d, x, 222, w, "SET" if k != "pilot_watch" else "HEAD · SAM 3D Body", SET, set_l)
    block(d, x, 354, w, "DOING" if k != "pilot_watch" else "HANDS · SAM 3D Body", TEXT, do_l)
    c = last(rec["callouts"], t)
    if c:
        cl = [(l, "body", CALL) for l in wrap(d, "“" + c["text"] + "”", F["body"], w - 30)[:3]]
    else:
        cl = [("Monitoring", "big", OK), ("said, set and doing agree", "small", MUTED)]
    block(d, x, 486, w, "CROSSCHECK", CALL, cl)
    # timeline
    y = 640
    d.rounded_rectangle((40, y, 1240, y + 34), 6, fill=PANEL)
    span = rec["_t1"] - rec["_t0"]
    px = lambda tt: 40 + (tt - rec["_t0"]) / span * 1200
    for h in rec["heard"]:
        d.rectangle((px(h["start"]), y + 6, max(px(h["start"]) + 3, px(h["t"])), y + 14), fill=SAID)
    for m in rec["marks"]:
        d.line((px(m["t"]), y + 2, px(m["t"]), y + 32), fill=MUTED, width=1)
    for cc in rec["callouts"]:
        d.ellipse((px(cc["t"]) - 5, y + 19, px(cc["t"]) + 5, y + 29), fill=CALL)
    d.line((px(t), y - 4, px(t), y + 38), fill=TEXT, width=2)
    d.text((1150, y + 40), f"{t - rec['_t0']:.0f} s", font=F["lab"], fill=MUTED)
    return im


def load_audio(path: pathlib.Path) -> np.ndarray:
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return resample_poly(x, SR, sr) if sr != SR else x


def build() -> pathlib.Path:
    data = json.loads((SITE / "data.js").read_text().removeprefix("window.DEBRIEF = ").rstrip().rstrip(";"))
    recs = {r["id"]: r for r in data}
    tmp = pathlib.Path(tempfile.mkdtemp())
    n, audio, cache = 0, [], {}

    def emit(im: Image.Image, seconds: float):
        nonlocal n
        for _ in range(int(round(seconds * FPS))):
            im.save(tmp / f"{n:05d}.png")
            n += 1
        audio.append(np.zeros(int(round(seconds * FPS)) * SR // FPS))

    emit(card(CARDS["intro"]), 6)
    for name, before, after in PLAN:
        if name not in recs:
            continue
        rec = recs[name]
        if name in CARDS:
            emit(card(CARDS[name]), 3.5)
        events = [h["start"] for h in rec["heard"]] + [m["t"] for m in rec["marks"]] + [c["t"] for c in rec["callouts"]]
        t0 = max(0.0, min(events) - before) if events else 0.0
        t1 = min(rec["duration"], (max(c["t"] for c in rec["callouts"]) if rec["callouts"] else max(events)) + after)
        rec["_t0"], rec["_t1"] = t0, t1
        frames = int((t1 - t0) * FPS)
        track = np.zeros(frames * SR // FPS)
        for a in rec["audio"]:
            if t0 <= a["t"] < t1:
                x = load_audio(SITE / a["file"]) * (0.8 if a["kind"] != "callout" else 1.0)
                i = int((a["t"] - t0) * SR)
                track[i:i + len(x)] += x[: max(0, len(track) - i)]
        for i in range(frames):
            frame_image(rec, t0 + i / FPS, cache).save(tmp / f"{n:05d}.png")
            n += 1
        audio.append(track)
    emit(card(CARDS["outro"]), 7)
    wav = tmp / "audio.wav"
    sf.write(wav, np.clip(np.concatenate(audio), -1, 1), SR)
    out = SITE / "crosscheck_demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "%05d.png"), "-i", str(wav),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "128k", "-shortest", str(out)], check=True)
    shutil.rmtree(tmp)
    return out
