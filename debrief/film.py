"""The film: the real debrief pages, filmed frame by frame, with narration.

  uv run python cli.py debrief && uv run python cli.py film      # writes site/crosscheck_film.mp4

Nothing in the picture is drawn for the film: headless Chrome loads the debrief site (site/), and
each page's own film hook (window.film.select / seek) sets its clock to the frame's time before a
screenshot, at 30 frames a second, 1920 x 1080. The recordings play in real time: the narration
sets each one up over its quiet lead-in, then the recording's own radio lines and callouts play
with nothing over them.

Narration is docs/film_script.md, one section per scene. A recording of your own voice in
docs/narration/NN.wav is used if it exists; otherwise OmniVoice (k2-fsa, CC-BY-NC, local, if
OMNIVOICE_DIR has it), otherwise the Mac's voice.
"""
import asyncio
import base64
import functools
import http.server
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SCRIPT = ROOT / "docs" / "film_script.md"
OWN_VOICE = ROOT / "docs" / "narration"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OMNIVOICE = pathlib.Path(os.environ.get("OMNIVOICE_DIR", pathlib.Path.home() / "llm-stack" / "omnivoice"))
W, H, FPS, SR = 1920, 1080, 30, 48000
NARRATOR = "male, middle-aged, low pitch, american accent"
# scene: (section in the script, page, what to select, recording window or None)
SCENES = [
    ("01", "film.html", "title", None),
    ("02", "film.html", "why", None),
    ("03", "film.html", "how", None),
    ("04", "debrief.html?film=1", "altitude_bust", "clearance"),
    ("05", "debrief.html?film=1", "autothrottle_hold_spark", "fault"),
    ("06", "debrief.html?film=1", "flaps_wrong", "call"),
    ("07", "debrief.html?film=1", "pilot_slumped", "change"),
    ("08", "index.html?film=1", "report", None),
    ("09", "film.html", "models", None),
    ("10", "film.html", "close", None),
]
AFTER_S = {"clearance": 4.0, "fault": 4.0, "call": 4.0, "change": 3.0}  # after the last callout in the window


def sections() -> dict[str, str]:
    text = SCRIPT.read_text()
    return {m.group(1): " ".join(m.group(2).split()) for m in re.finditer(r"^## (\d\d) · [^\n]*\n(.*?)(?=^## |\Z)", text, re.S | re.M)}


# ---------------------------------------------------------------- narration
CACHE = ROOT / ".cache" / "film_voice"  # synthetic narration, keyed on the text: re-renders reuse it


def tts(texts: dict[str, str], out: pathlib.Path) -> dict[str, np.ndarray]:
    import hashlib
    out.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    voices = {}
    todo = {}
    key = lambda text: hashlib.sha1((NARRATOR + "|" + text).encode()).hexdigest()[:16]
    for k, text in texts.items():
        own = OWN_VOICE / f"{k}.wav"
        if own.exists():
            voices[k] = load(own)
        elif (CACHE / f"{key(text)}.wav").exists():
            voices[k] = load(CACHE / f"{key(text)}.wav")
        else:
            todo[k] = text
    if todo and (OMNIVOICE / ".venv" / "bin" / "python").exists():
        # two takes of each line on the CPU (the Apple GPU path crashes here); the take Muse Voice
        # Transcribe hears closest to the script is kept, so the narration is measured, not hoped for
        job = out / "omnivoice.json"
        job.write_text(json.dumps({"instruct": NARRATOR, "lines": todo, "out": str(out), "takes": TAKES}))
        code = (
            "import json, sys, torch, soundfile as sf\n"
            "from omnivoice.models.omnivoice import OmniVoice\n"
            "j = json.load(open(sys.argv[1]))\n"
            "m = OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map='cpu', dtype=torch.float32)\n"
            "for k, text in j['lines'].items():\n"
            "    for take in range(j['takes']):\n"
            "        torch.manual_seed(7 + 100 * take)\n"
            "        a = m.generate(text=text, language='en', instruct=j['instruct'], num_step=32, guidance_scale=2.2, speed=0.95)[0]\n"
            "        sf.write(f\"{j['out']}/{k}_{take}.wav\", a, m.sampling_rate)\n")
        subprocess.run([str(OMNIVOICE / ".venv" / "bin" / "python"), "-c", code, str(job)], cwd=OMNIVOICE,
                       env={**os.environ, "HF_HUB_OFFLINE": "1"}, check=False)
        for k, text in todo.items():
            takes = [out / f"{k}_{i}.wav" for i in range(TAKES) if (out / f"{k}_{i}.wav").exists()]
            if takes:
                best = max(takes, key=lambda f: heard_as(f, text))
                shutil.copy(best, out / f"{k}.wav")
    for k, text in todo.items():
        f = out / f"{k}.wav"
        if not f.exists():  # the Mac's own voice, as a last resort (not cached: OmniVoice should be tried again)
            aiff = out / f"{k}.aiff"
            subprocess.run(["say", "-v", "Samantha", "-r", "175", "-o", str(aiff), text], check=True)
            voices[k] = load(aiff)
        else:
            shutil.copy(f, CACHE / f"{key(text)}.wav")
            voices[k] = load(f)
    return voices


TAKES = 2


def heard_as(path: pathlib.Path, text: str) -> float:
    """How closely Muse Voice Transcribe hears a take as the script (0-1, word level)."""
    import difflib
    from senses.speech import normalise, transcribe
    x, sr = sf.read(path, always_2d=True)
    x = resample_poly(x.mean(axis=1), 16000, sr)
    heard = asyncio.run(transcribe(x.astype(np.float32)))["text"]
    score = difflib.SequenceMatcher(None, normalise(text).split(), normalise(heard).split()).ratio()
    print(f"    {path.name}: {score:.2f}  \"{heard[:70]}\"", flush=True)
    return score


def load(path: pathlib.Path) -> np.ndarray:
    x, sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = resample_poly(x, SR, sr)
    return x.astype(np.float32)


# ---------------------------------------------------------------- browser
class Page:
    """The few DevTools-protocol calls the film needs, over one websocket."""

    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def call(self, method: str, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

    async def js(self, expr: str):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"{expr}: {r['exceptionDetails']}")
        return r.get("result", {}).get("value")

    async def open(self, url: str):
        await self.call("Page.navigate", url=url)
        for _ in range(200):
            await asyncio.sleep(0.1)
            if await self.js("document.readyState") == "complete":
                break
        await self.js("document.fonts.ready.then(() => true)")
        await asyncio.sleep(0.5)

    async def shot(self, path: pathlib.Path):
        r = await self.call("Page.captureScreenshot", format="jpeg", quality=92)
        path.write_bytes(base64.b64decode(r["data"]))


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def serve_site() -> int:
    port = free_port()
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Quiet, directory=str(SITE)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return port


# ---------------------------------------------------------------- scenes
def window_first(rec: dict, anchor: str) -> float:
    events = {"clearance": [h["start"] for h in rec["heard"]], "call": [h["start"] for h in rec["heard"]],
              "fault": [m["t"] for m in rec["marks"]], "change": [m["t"] for m in rec["marks"]]}[anchor]
    return min(events) if events else 2.0


def window(rec: dict, anchor: str, lead: float) -> tuple[float, float]:
    """The part of a recording to show: the narration's length before the first thing that
    happens, then until a little after the last callout of the event."""
    events = {"clearance": [h["start"] for h in rec["heard"]], "call": [h["start"] for h in rec["heard"]],
              "fault": [m["t"] for m in rec["marks"]], "change": [m["t"] for m in rec["marks"]]}[anchor]
    first = min(events) if events else 2.0
    t0 = max(0.0, first - lead - 1.0)
    calls = [c["t"] for c in rec["callouts"]] or [first + 10]
    t1 = min(rec["duration"], max(calls) + AFTER_S[anchor] + (4.0 if rec.get("video") else 0.0))
    return t0, t1


def build(repo: str | None = None) -> pathlib.Path:
    import websockets
    data = {r["id"]: r for r in json.loads((SITE / "data.js").read_text().removeprefix("window.DEBRIEF = ").rstrip().rstrip(";"))}
    for name in ("film.html",):
        shutil.copy(pathlib.Path(__file__).parent / name, SITE / name)
    shutil.copy(ROOT / "docs" / "how-it-works.svg", SITE / "how-it-works.svg")
    text = sections()
    only = os.environ.get("FILM_SCENES")  # e.g. "01,04" for a quick check of some scenes
    scenes = [sc for sc in SCENES if not only or sc[0] in only.split(",")]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="film_"))
    voice = tts({k: text[k] for k, *_ in scenes}, tmp / "voice")
    port = serve_site()
    dbg = free_port()
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={dbg}", f"--window-size={W},{H}",
                               "--hide-scrollbars", "--force-device-scale-factor=1", "--mute-audio", "--autoplay-policy=no-user-gesture-required",
                               f"--user-data-dir={tmp / 'chrome'}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    audio, n = [], 0
    frames = tmp / "frames"
    frames.mkdir()

    async def run():
        nonlocal n
        import httpx
        for _ in range(100):
            try:
                targets = httpx.get(f"http://127.0.0.1:{dbg}/json/list", trust_env=False).json()
                ws_url = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
                break
            except Exception:
                await asyncio.sleep(0.2)
        async with websockets.connect(ws_url, max_size=50_000_000, proxy=None) as ws:
            page = Page(ws)
            current, scale = None, None
            for key, url, what, anchor in scenes:
                full = f"http://127.0.0.1:{port}/{url}" + (f"{'&' if '?' in url else '?'}repo={repo}" if repo and "film.html" in url else "")
                # the film's own scenes are drawn at 1920 x 1080; the site's pages at 1.5x, so their text reads on a screen
                want = 1.0 if "film.html" in url else 1.5  # 1280 x 720 layout, exactly 1920 x 1080 pixels
                if want != scale:
                    await page.call("Emulation.setDeviceMetricsOverride", width=round(W / want), height=round(H / want),
                                    deviceScaleFactor=want, mobile=False)
                    scale, current = want, None
                if full != current:
                    await page.open(full)
                    current = full
                await page.js(f"window.film.select({json.dumps(what)})")
                v = voice[key]
                speak_s = len(v) / SR
                track = []
                if anchor:  # a recording: narration over its lead-in, then its own sound
                    rec = data[what]
                    t0, t1 = window(rec, anchor, speak_s)
                    # if the recording's quiet lead-in is shorter than the narration, hold its first frame
                    # until the narration is done, so no radio line or callout plays under it
                    hold = max(0.0, 0.6 + speak_s + 0.8 - (window_first(rec, anchor) - t0))
                    dur = t1 - t0 + hold
                    seek = lambda i, t0=t0, hold=hold: t0 + max(0.0, i / FPS - hold)
                    for a in rec["audio"]:
                        if t0 - 0.5 <= a["t"] <= t1:
                            track.append((a["t"] - t0 + hold, load(SITE / a["file"]), 0.9 if a["kind"] == "callout" else 0.8))
                    narr_at = 0.6
                elif what == "report":
                    dur = speak_s + 3.0
                    await page.js(f"window.film.span = {dur - 3.5:.2f}")
                    seek = lambda i: i / FPS
                    narr_at = 0.8
                else:
                    dur = speak_s + (2.5 if what in ("title", "close") else 1.6)
                    seek = lambda i: i / FPS
                    narr_at = 1.0 if what in ("title",) else 0.5
                track.append((narr_at, v, 1.0))
                count = int(round(dur * FPS))
                t_scene = time.time()
                # a recording opens on the whole cockpit and moves in to the instruments 2.5 s before its first event
                switch = (window_first(data[what], anchor) - 2.5 - t0 + hold) * FPS if anchor else -1
                shown = None
                for i in range(count):
                    if anchor:
                        want_view = "wide" if i < switch else "inst"
                        if want_view != shown:
                            shown = await page.js(f"window.film.view({json.dumps(want_view)})")
                    await page.js(f"window.film.seek({seek(i):.4f})")
                    await page.shot(frames / f"{n:06d}.jpg")
                    n += 1
                audio.append((count, track))
                print(f"  {key} {what}: {dur:.1f} s, {count} frames in {time.time() - t_scene:.0f} s", flush=True)
    try:
        asyncio.run(run())
    finally:
        chrome.terminate()

    # sound: each scene's narration and recording audio, mixed, then the whole track
    mix = []
    for count, track in audio:
        seg = np.zeros(int(count / FPS * SR), dtype=np.float32)
        for at, x, gain in track:
            i = int(at * SR)
            if i < len(seg):
                seg[i:i + len(x)] += gain * x[:len(seg) - i]
        mix.append(seg)
    wav = np.concatenate(mix)
    wav = np.clip(wav / max(1.0, np.abs(wav).max() / 0.95), -1, 1)
    sf.write(tmp / "sound.wav", wav, SR)
    out = SITE / "crosscheck_film.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "%06d.jpg"),
                    "-i", str(tmp / "sound.wav"), "-vf", f"scale={W}:{H}", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return out
