"""The monitoring loop: frame -> panel reading -> ledger facts -> rules -> callout.

Live and replay share this code. Live: capture the screen, read the panel, decide. Replay:
feed the readings and "said" facts saved with a recording back in at their original times,
so a judge can reproduce every callout without X-Plane, a GPU or an API key.

Each flag is spoken once, then again only after it has cleared and come back. Everything is
written to <dir>/events.jsonl; with frames=True the panel crops go to <dir>/frames/.
"""
import asyncio
import collections
import json
import pathlib
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields

from PIL import Image

from brain.ledger import Fact, Ledger
from brain.rules import Flag, evaluate
from senses.panel import PanelReading, read_frame

FINAL_BELOW_FT = 2500  # crude phase detection until checklists and the FMS phase are read
CLEAR_AFTER_S = 4.0    # a flag counts as cleared only after this long absent: one flickering
                       # reading must not make Crosscheck repeat a callout
# Region of the screen kept with a recording: every calibrated box plus a margin for view shifts.
FRAME_ORIGIN = (1240, 1230)
FRAME_BOX = (1240, 1230, 3180, 2200)
SCREEN_SIZE = (3600, 2338)
READING_FIELDS = [f.name for f in fields(PanelReading) if f.name not in ("raw", "roll_labels")]
SPARK_PERIOD_S = 1.0  # capture rate with the Muse Spark reader; reads overlap (~3 s each)


def capture_screen() -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        subprocess.run(["screencapture", "-x", f.name], check=True)
        return Image.open(f.name).copy()


def frame_from_crop(crop: Image.Image) -> Image.Image:
    """Rebuild a full-screen frame from a recorded crop, so read_frame's calibration applies."""
    canvas = Image.new("RGB", SCREEN_SIZE)
    canvas.paste(crop, FRAME_ORIGIN)
    return canvas


class Crosscheck:
    def __init__(self, out_dir: pathlib.Path, speak=None, frames: bool = False, log_name: str = "events.jsonl",
                 reader: str = "apple"):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger()
        self.active: dict[tuple[str, str], Flag] = {}
        self.last_seen: dict[tuple[str, str], float] = {}
        self.log = (out_dir / log_name).open("w")
        self.speak = speak
        self.frames = frames
        self.reader = reader
        self.n = 0
        if frames:
            (out_dir / "frames").mkdir(exist_ok=True)

    def event(self, kind: str, t: float, **data) -> None:
        self.log.write(json.dumps({"t": t, "kind": kind, **data}) + "\n")
        self.log.flush()

    def said(self, fact: Fact, **extra) -> None:
        """A spoken clearance, readback or callout (from speech, or from a scenario script).
        extra (e.g. the transcript) is logged with the event but does not enter the ledger."""
        self.ledger.add(fact)
        self.event("said", fact.t, key=fact.key, value=fact.value, speaker=fact.speaker, source=fact.source, **extra)

    def observe(self, r: PanelReading, t: float, frame_file: str | None = None, **extra) -> None:
        facts = [("doing", "ias", r.ias), ("doing", "altitude", r.altitude),
                 ("doing", "track", r.track_mag), ("set", "mcp.speed", r.selected_speed),
                 ("set", "mcp.altitude", r.selected_altitude)]
        if r.fma and any(r.fma):
            # the line was read: a blank thrust cell means no engaged thrust mode, not "unknown"
            facts.append(("doing", "fma.thrust", r.fma[0] or "BLANK"))
        for stream, key, value in facts:
            if value is not None:
                self.ledger.add(Fact(stream, key, value, t, source="panel"))
        if r.altitude is not None:
            self.ledger.add(Fact("doing", "phase", "final" if r.altitude < FINAL_BELOW_FT else "other", t, source="derived"))
        self.event("reading", t, frame=frame_file, **{k: getattr(r, k) for k in READING_FIELDS}, **extra)

    def decide(self, t: float) -> list[Flag]:
        flags = evaluate(self.ledger, t)
        new = []
        current = {(f.rule, f.key): f for f in flags}
        for k, f in current.items():
            self.last_seen[k] = t
            if k not in self.active:
                new.append(f)
                self.active[k] = f
                self.event("callout", t, **asdict(f))
                if self.speak:
                    self.speak(f.callout)
        for k in [k for k in self.active if k not in current and t - self.last_seen[k] >= CLEAR_AFTER_S]:
            del self.active[k]
            self.event("cleared", t, rule=k[0], key=k[1])
        return new

    def _save(self, frame: Image.Image) -> str | None:
        name = None
        if self.frames:
            name = f"frames/{self.n:05d}.jpg"
            frame.crop(FRAME_BOX).convert("RGB").save(self.dir / name, quality=85)
        self.n += 1
        return name

    def step(self, frame: Image.Image, t: float) -> list[Flag]:
        self.observe(read_frame(frame), t, self._save(frame))
        return self.decide(t)

    def run(self, seconds: float, t0: float | None = None) -> None:
        t0 = t0 or time.time()
        if self.reader == "spark":
            return self._run_spark(seconds, t0)
        while time.time() - t0 < seconds:
            self.step(capture_screen(), time.time())

    def _run_spark(self, seconds: float, t0: float) -> None:
        """Capture every SPARK_PERIOD_S, read on Muse Spark with several calls in flight, and
        apply readings in capture order, timestamped when they arrive: the reader's latency
        is part of the record, so replay reproduces the live run exactly."""
        from senses.panel_spark import read_frame_spark
        pending: collections.deque = collections.deque()
        next_capture = time.time()
        with ThreadPoolExecutor(max_workers=6) as pool:
            while time.time() - t0 < seconds or pending:
                if time.time() - t0 < seconds and time.time() >= next_capture:
                    frame = capture_screen()
                    captured = time.time()
                    name = self._save(frame)
                    pending.append((captured, name, pool.submit(lambda f=frame: asyncio.run(read_frame_spark(f))[0])))
                    next_capture = captured + SPARK_PERIOD_S
                while pending and pending[0][2].done():
                    captured, name, fut = pending.popleft()
                    t = time.time()
                    try:
                        r = fut.result()
                    except Exception as e:  # one failed call: the frame stays unread, the run goes on
                        self.event("reader_error", t, frame=name, error=str(e)[:200])
                        continue
                    self.observe(r, t, name, captured=round(captured, 3), reader="muse-spark")
                    self.decide(t)
                time.sleep(0.05)


def _spark_readings(rec_dir: pathlib.Path, frames: list[str]) -> dict[str, PanelReading]:
    import asyncio
    import httpx
    from senses.panel_spark import read_frame_spark

    async def go():
        sem = asyncio.Semaphore(6)
        async with httpx.AsyncClient(trust_env=False, timeout=90) as client:
            async def one(f):
                async with sem:
                    return f, (await read_frame_spark(frame_from_crop(Image.open(rec_dir / f)), client))[0]
            return dict(await asyncio.gather(*(one(f) for f in frames)))
    return asyncio.run(go())


def replay(rec_dir: pathlib.Path, out_dir: pathlib.Path, recompute: bool = False, reader: str = "apple") -> Crosscheck:
    """Re-run a recording through the same ledger and rules.

    Default: use the panel readings saved with the recording (works on any OS, no models).
    recompute=True: re-read every saved frame first, with senses.panel (reader="apple", macOS)
    or Muse Spark through the Meta Model API (reader="spark", any OS, needs MODEL_API_KEY).
    """
    events = [json.loads(l) for l in (rec_dir / "events.jsonl").read_text().splitlines()]
    spark = None
    if recompute and reader == "spark":
        spark = _spark_readings(rec_dir, [e["frame"] for e in events if e["kind"] == "reading" and e.get("frame")])
    cc = Crosscheck(out_dir, log_name="replay_events.jsonl")
    for e in sorted((e for e in events if e["kind"] in ("reading", "said")), key=lambda e: (e["t"], e["kind"] != "said")):
        if e["kind"] == "said":
            cc.said(Fact("said", e["key"], e["value"], e["t"], e["speaker"], source=e["source"]))
            continue
        if spark is not None:
            r = spark[e["frame"]]
        elif recompute:
            r = read_frame(frame_from_crop(Image.open(rec_dir / e["frame"])))
        else:
            r = PanelReading(**{k: e[k] for k in READING_FIELDS if k in e})
        cc.observe(r, e["t"], e.get("frame"))
        cc.decide(e["t"])
    return cc
