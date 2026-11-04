"""The film: the real debrief pages, filmed frame by frame, with narration.

  uv run python cli.py debrief && uv run python cli.py film      # writes site/crosscheck_film.mp4
  uv run python cli.py film --cut short                          # under 3 minutes: site/crosscheck_film_short.mp4

Nothing in the picture is drawn for the film: headless Chrome loads the debrief site (site/), and
each page's own film hook (window.film.select / seek) sets its clock to the frame's time before a
screenshot, at 30 frames a second, 1920 x 1080. The recordings play in real time: the narration
sets each one up over its quiet lead-in, then the recording's own radio lines and callouts play
with nothing over them.

Narration is docs/film_script.md, one section per scene. A recording of your own voice in
docs/narration/NN.wav is used if it exists; otherwise Fish Audio Drama 3 (local, if DRAMA3_DIR has
it) reads each section in the narrator's voice, docs/voice/narrator.wav: a voice Drama 3 designed
from a written description (docs/voice/narrator.json), not a copy of anyone's. Without Drama 3,
OmniVoice (k2-fsa, CC-BY-NC, local, if OMNIVOICE_DIR has it), otherwise the Mac's voice.
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
# each cut: its script and its file; a cut films the sections its script has, in the order of SCENES
CUTS = {"full": (SCRIPT, "crosscheck_film.mp4"), "short": (ROOT / "docs" / "film_script_short.md", "crosscheck_film_short.mp4")}
OWN_VOICE = ROOT / "docs" / "narration"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OMNIVOICE = pathlib.Path(os.environ.get("OMNIVOICE_DIR", pathlib.Path.home() / "llm-stack" / "omnivoice"))
DRAMA3 = pathlib.Path(os.environ.get("DRAMA3_DIR", pathlib.Path.home() / "llm-stack" / "drama3"))
VOICE = ROOT / "docs" / "voice"
W, H, FPS, SR = 1920, 1080, 30, 48000
NARRATOR = "male, middle-aged, low pitch, american accent"
# scene: (section in the script, page, what to select, recording window or None)
SCENES = [
    ("01", "film.html", "title", None),
    ("02", "film.html", "why", None),
    ("03", "film.html", "how", None),
    ("04", "debrief.html?film=1", "altitude_bust_spark", "clearance"),  # the panel read by Muse Spark, as the narration says
    ("05", "debrief.html?film=1", "autothrottle_hold_spark", "fault"),
    ("06", "debrief.html?film=1", "flaps_wrong", "call"),
    ("07", "debrief.html?film=1", "pilot_slumped", "change"),
    ("11", "debrief.html?film=1", "altitude_bust_onboard", "clearance"),  # full cut only: no network
    ("08", "index.html?film=1", "report", None),
    ("09", "film.html", "models", None),
    ("10", "film.html", "close", None),
]
# after the last callout in the window: long enough for the callout itself (the bank's longest,
# "Captain not responding. First officer, take control.", is 3.1 s) and a beat after it
AFTER_S = {"clearance": 2.9, "fault": 3.2, "call": 3.4, "change": 2.0}
VIDEO_AFTER_S = 1.5  # the pilot camera: a moment more of the first officer's view


def sections(script: pathlib.Path = SCRIPT) -> dict[str, str]:
    text = script.read_text()
    return {m.group(1): " ".join(m.group(2).split()) for m in re.finditer(r"^## (\d\d) · [^\n]*\n(.*?)(?=^## |\Z)", text, re.S | re.M)}


# ---------------------------------------------------------------- narration
CACHE = ROOT / ".cache" / "film_voice"  # synthetic narration, keyed on the text: re-renders reuse it
# How each section is performed: Drama 3 takes a bracketed direction before the words and does not
# read it out (read without one, every line came out evenly paced and flat). "Speaking naturally ...
# confident but relaxed" is the direction the narrator's voice was chosen on, by ear (docs/voice/narrator.json)
DIRECTION = {
    "01": "[speaking naturally and warmly, confident but relaxed, a quiet opening]",
    "02": "[speaking naturally, serious and sincere, like telling a friend something that matters, natural pauses]",
    "03": "[speaking naturally and warmly, clear and friendly, explaining how something works]",
    "04": "[speaking naturally, like telling a friend a story, a little tension, then quiet relief]",
    "05": "[speaking naturally, steady, with quiet urgency, like a pilot describing a close call]",
    "06": "[speaking naturally, clear and matter-of-fact, explaining]",
    "07": "[speaking naturally, gentle and serious, caring]",
    "11": "[speaking naturally, calm and confident, explaining something impressive without boasting]",
    "08": "[speaking naturally, confident and plain, stating facts]",
    "09": "[speaking naturally and warmly, understated, a little proud]",
    "10": "[speaking naturally and warmly, quietly confident, a closing line]",
}
CLEAR_ENOUGH = 0.9  # a take this close to the script counts as clear; among those, the most natural pace wins
# the chosen take, lightly denoised: what is left of the reference's hiss, under the words and in the pauses
CLEAN = "highpass=f=70,afftdn=nr=12:nf=-60:tn=1"
NARRATION_LUFS = -16.0  # every line's loudness: the first narrator's, and about the callouts' (voice/bank/)


def tts(texts: dict[str, str], out: pathlib.Path) -> dict[str, np.ndarray]:
    import hashlib
    out.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    engine = narrator()
    voices = {}
    todo = {}
    said = lambda k, text: (DIRECTION[k] + " " + text) if k in DIRECTION and engine["id"].startswith("drama3") else text
    key = lambda k, text: hashlib.sha1((engine["id"] + "|" + said(k, text)).encode()).hexdigest()[:16]
    for k, text in texts.items():
        own = OWN_VOICE / f"{k}.wav"
        if own.exists():
            voices[k] = load(own)
        elif (CACHE / f"{key(k, text)}.wav").exists():
            voices[k] = load(CACHE / f"{key(k, text)}.wav")
        else:
            todo[k] = text
    if todo and engine["run"]:
        # several takes of each line; of those Muse Voice Transcribe hears clearly enough, the one nearest
        # the narrator's own pace is kept, so the narration is measured, not hoped for, and not the flattest
        engine["run"]({k: said(k, text) for k, text in todo.items()}, out)
        for k, text in todo.items():
            takes = [out / f"{k}_{i}.wav" for i in range(engine["takes"]) if (out / f"{k}_{i}.wav").exists()]
            if takes:
                best = max(takes, key=lambda f: (min(round(heard_as(f, text), 2), CLEAR_ENOUGH), -abs(pace(f, text) - engine["pace"])))
                if engine["id"].startswith("drama3"):
                    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(best), "-af", CLEAN, str(out / f"{k}.wav")], check=True)
                else:
                    shutil.copy(best, out / f"{k}.wav")
    for k, text in todo.items():
        f = out / f"{k}.wav"
        if not f.exists():  # the Mac's own voice, as a last resort (not cached: the engine should be tried again)
            aiff = out / f"{k}.aiff"
            subprocess.run(["say", "-v", "Samantha", "-r", "175", "-o", str(aiff), text], check=True)
            voices[k] = load(aiff)
        else:
            shutil.copy(f, CACHE / f"{key(k, text)}.wav")
            voices[k] = load(f)
    return {k: level(v) for k, v in voices.items()}


def narrator() -> dict:
    """The synthetic narrator: Drama 3 in the designed voice if both are here, else OmniVoice from
    the description in NARRATOR, else none (the Mac's voice)."""
    import hashlib
    ref = VOICE / "narrator.wav"
    if (DRAMA3 / ".venv" / "bin" / "python").exists() and ref.exists():
        about = json.loads((VOICE / "narrator.json").read_text())
        ref_text = about["text"]
        return {"id": "drama3:" + hashlib.sha1(ref.read_bytes()).hexdigest()[:12], "takes": 3,
                "pace": about.get("pace") or pace(ref, ref_text),
                "run": lambda todo, out: drama3(todo, out, ref, ref_text, 3)}
    if (OMNIVOICE / ".venv" / "bin" / "python").exists():
        return {"id": NARRATOR, "takes": 2, "pace": 2.6, "run": lambda todo, out: omnivoice(todo, out, 2)}
    return {"id": "say", "takes": 0, "pace": 0, "run": None}


def drama3(todo: dict[str, str], out: pathlib.Path, ref: pathlib.Path, ref_text: str, takes: int) -> None:
    """Fish Audio Drama 3 (bf16, about 74 GB while generating) reads each line in the voice of ref."""
    job = out / "drama3.json"
    job.write_text(json.dumps({"lines": todo, "out": str(out), "takes": takes, "ref": str(ref), "ref_text": ref_text}))
    code = (
        "import json, sys, numpy as np, soundfile as sf, mlx.core as mx\n"
        "from mlx_audio.tts.utils import load_model\n"
        "j = json.load(open(sys.argv[1]))\n"
        "m = load_model('mlx-community/fish-audio-drama-3-bf16')\n"
        "for k, text in j['lines'].items():\n"
        "    for take in range(j['takes']):\n"
        "        mx.random.seed(7 + 100 * take)\n"
        "        outs = list(m.generate(text=text, ref_audio=j['ref'], ref_text=j['ref_text'], temperature=0.85))\n"
        "        a = np.concatenate([np.array(o.audio, dtype=np.float32).reshape(-1) for o in outs])\n"
        "        sf.write(f\"{j['out']}/{k}_{take}.wav\", a, outs[0].sample_rate)\n"
        "        print(k, take, f'{len(a) / outs[0].sample_rate:.1f} s', flush=True)\n")
    subprocess.run([str(DRAMA3 / ".venv" / "bin" / "python"), "-c", code, str(job)], cwd=DRAMA3,
                   env={**os.environ, "HF_HUB_OFFLINE": "1"}, check=False)


def omnivoice(todo: dict[str, str], out: pathlib.Path, takes: int) -> None:
    """OmniVoice on the CPU (the Apple GPU path crashes here), voice from the NARRATOR description."""
    job = out / "omnivoice.json"
    job.write_text(json.dumps({"instruct": NARRATOR, "lines": todo, "out": str(out), "takes": takes}))
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


def pace(path: pathlib.Path, text: str) -> float:
    """Words a second, counting the speech from its first sound to its last."""
    x, sr = sf.read(path, always_2d=True)
    x = np.abs(x.mean(axis=1))
    loud = np.flatnonzero(x > 0.02 * x.max())
    return len(text.split()) / max(0.5, (loud[-1] - loud[0]) / sr) if len(loud) else 0.0


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


def level(x: np.ndarray) -> np.ndarray:
    """A narration line at NARRATION_LUFS, whatever voice or take it came from (5 Nov: Drama 3 reads
    the second narrator 7.5 dB quieter than the first, under the callouts). loudnorm also holds the
    peaks under -1.5 dB, which plain gain cannot: these takes peak high for their loudness."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                        "-af", f"loudnorm=I={NARRATION_LUFS}:TP=-1.5:LRA=11", "-ar", str(SR), "-f", "f32le", "-"],
                       input=x.astype("<f4").tobytes(), capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype="<f4").copy()


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
    t1 = min(rec["duration"], max(calls) + AFTER_S[anchor] + (VIDEO_AFTER_S if rec.get("video") else 0.0))
    return t0, t1


def build(repo: str | None = None, cut: str = "full") -> pathlib.Path:
    import websockets
    data = {r["id"]: r for r in json.loads((SITE / "data.js").read_text().removeprefix("window.DEBRIEF = ").rstrip().rstrip(";"))}
    for name in ("film.html",):
        shutil.copy(pathlib.Path(__file__).parent / name, SITE / name)
    shutil.copy(ROOT / "docs" / "how-it-works.svg", SITE / "how-it-works.svg")
    script, out_name = CUTS[cut]
    text = sections(script)
    only = os.environ.get("FILM_SCENES")  # e.g. "01,04" for a quick check of some scenes
    scenes = [sc for sc in SCENES if sc[0] in text and (not only or sc[0] in only.split(","))]
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
                    dur = speak_s + 2.0
                    await page.js(f"window.film.span = {dur - 2.5:.2f}")
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
    out = SITE / out_name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "%06d.jpg"),
                    "-i", str(tmp / "sound.wav"), "-vf", f"scale={W}:{H}", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return out
