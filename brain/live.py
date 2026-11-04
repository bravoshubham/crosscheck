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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields

from PIL import Image

from brain.ledger import Fact, Ledger
from brain.rules import SPEED_MODES, Flag, evaluate
from senses.panel import PanelReading, read_frame

FINAL_BELOW_FT = 2500  # crude phase detection until checklists and the FMS phase are read
CLEAR_AFTER_S = 4.0    # a flag counts as cleared only after this long absent: one flickering
                       # reading must not make Crosscheck repeat a callout
# Region of the screen kept with a recording: every calibrated box plus a margin for view shifts.
FRAME_ORIGIN = (1240, 1230)
FRAME_BOX = (1240, 1230, 3180, 2200)
SCREEN_SIZE = (3600, 2338)
READING_FIELDS = [f.name for f in fields(PanelReading) if f.name not in ("raw", "roll_labels")]
WIDE_W = 1024  # the whole-cockpit frame kept with each reading, for the debrief and the film
SPARK_PERIOD_S = 1.0  # capture rate with a model reader; reads overlap (~3 s each)
SPARK_FINAL_PERIOD_S = 0.5  # Muse Spark's final lane: its reads take 2-7 s, so the more are in flight, the sooner a fresh one
                            # arrives (at one a second, one cloud run's alert waited 4.3 s for a reading, and was late)
SPARK_WORKERS = 16  # calls in flight at once: at 6, slow reads queued the next ones behind them
SPARK_READERS = ("spark", "spark-flaps", "glimmer", "glimmer-flaps")  # PFD values / flap gauge, on Muse Spark or the local Muse Glimmer
IN_FLIGHT = {"glimmer": 1, "glimmer-flaps": 1}  # one busy local GPU: a second call in flight doubles the delay, not the rate
FULL_EVERY_FINAL_S = 30.0  # on final with Glimmer, a read of the whole set of parts (altitude, selected altitude) at most
                           # this often, and none while the thrust box is not in a speed mode: the autothrottle alert
                           # needs every read, and a fault in the seconds before a whole read (3.4-5.7 s) is seen one
                           # whole read late
READER_LABEL = {"apple": "apple-vision", "spark": "muse-spark", "spark-flaps": "muse-spark-flaps",
                "glimmer": "muse-glimmer", "glimmer-flaps": "muse-glimmer-flaps"}


def _spark_reader(reader: str):
    if reader == "spark-flaps":
        from senses.flaps_spark import read_flaps_spark
        return read_flaps_spark
    if reader == "glimmer-flaps":
        from senses.flaps_glimmer import read_flaps_glimmer
        return read_flaps_glimmer
    if reader == "glimmer":
        from senses.panel_glimmer import read_frame_glimmer
        return read_frame_glimmer
    from senses.panel_spark import read_frame_spark
    return read_frame_spark


MISHEARD_MARGIN_FT = 300


def implausible(fact: Fact, led: Ledger, transcript: str) -> str | None:
    """Why a heard altitude cannot be what was said, or None. A climb to below the aircraft, or a
    descent to above it, is a misheard number (altitude_correct_spark, 3 Nov: "one one thousand"
    came back from Voice Transcribe as "climb and maintain 1000", at 9,000 ft)."""
    if fact.key not in ("clearance.altitude", "readback.altitude"):
        return None
    alts = sorted(f.value for f in [f for f in led.since("doing", "altitude", 0) if f.t <= fact.t][-5:])
    if not alts:
        return None
    # the median of the last five: one misread (68,193 ft at 9,000) must not reject a clearance, and
    # nor must three in a row (altimeter_forgot, 5 Nov: 18,999, 605 and 6,106 ft at 9,000 made the
    # median of three 6,106, and "descend and maintain 7000" looked like a climb)
    alt = alts[len(alts) // 2]
    words = transcript.lower()
    if "climb" in words and fact.value < alt - MISHEARD_MARGIN_FT:
        return f"a climb to {fact.value} ft from {alt} ft"
    if "descend" in words and fact.value > alt + MISHEARD_MARGIN_FT:
        return f"a descent to {fact.value} ft from {alt} ft"
    if fact.key == "readback.altitude":  # a readback on the wrong side of the aircraft from its clearance
        clr = led.latest("said", "clearance.altitude", before=fact.t, speaker="atc")
        if clr is not None and (clr.value > alt + MISHEARD_MARGIN_FT) != (fact.value > alt + MISHEARD_MARGIN_FT) \
                and abs(clr.value - alt) > MISHEARD_MARGIN_FT and abs(fact.value - alt) > MISHEARD_MARGIN_FT:
            return f"a readback of {fact.value} ft at {alt} ft, to a clearance to {clr.value} ft"
    return None


def capture_screen() -> Image.Image:
    """The whole screen, in the display's own colour values. Straight from the display (0.08 s a
    Retina screen) when Quartz is there (it comes with ocrmac), else through screencapture's PNG
    (0.56 s, the same values; its BMP is colour-converted, so not that)."""
    try:
        import Quartz
    except ImportError:
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            subprocess.run(["screencapture", "-x", f.name], check=True)
            return Image.open(f.name).copy()
    img = Quartz.CGDisplayCreateImage(Quartz.CGMainDisplayID())
    w, h = Quartz.CGImageGetWidth(img), Quartz.CGImageGetHeight(img)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
    return Image.frombuffer("RGBA", (w, h), data, "raw", "BGRA", Quartz.CGImageGetBytesPerRow(img), 1).copy()


def frame_from_crop(crop: Image.Image, origin: tuple[int, int] = FRAME_ORIGIN) -> Image.Image:
    """Rebuild a full-screen frame from a recorded crop, so the readers' calibration applies."""
    canvas = Image.new("RGB", SCREEN_SIZE)
    canvas.paste(crop, origin)
    return canvas


class Crosscheck:
    def __init__(self, out_dir: pathlib.Path, speak=None, frames: bool = False, log_name: str = "events.jsonl",
                 reader: str = "apple", frame_box: tuple[int, int, int, int] = FRAME_BOX, fact_time: str = "arrival",
                 explain: bool = False):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger()
        self.active: dict[tuple[str, str], Flag] = {}
        self.last_seen: dict[tuple[str, str], float] = {}
        self.log = (out_dir / log_name).open("w")
        self.speak = speak
        self.frames = frames
        self.reader = reader
        self.frame_box = frame_box
        # "captured": a Spark reading's facts carry the time its frame was taken (it is about that
        # moment), applied when the reading arrives. "arrival": stamped on arrival (the recordings
        # made before 3 Nov, and the flap gauge, whose settle rule was tuned that way).
        self.fact_time = fact_time
        self.explain = explain  # live runs: Muse Spark writes a sentence under each callout (brain/explain.py)
        self._log_lock = threading.Lock()  # the explanation arrives on another thread
        self.n = 0
        if frames:
            (out_dir / "frames").mkdir(exist_ok=True)
            (out_dir / "wide").mkdir(exist_ok=True)

    def event(self, kind: str, t: float, **data) -> None:
        with self._log_lock:
            self.log.write(json.dumps({"t": t, "kind": kind, **data}) + "\n")
            self.log.flush()

    def said(self, fact: Fact, **extra) -> None:
        """A spoken clearance, readback or callout (from speech, or from a scenario script).
        extra (e.g. the transcript) is logged with the event but does not enter the ledger.
        A value the aircraft's own state contradicts is logged as rejected and not used."""
        why = implausible(fact, self.ledger, extra.get("transcript") or "")
        if why:
            self.event("said_rejected", fact.t, key=fact.key, value=fact.value, speaker=fact.speaker, source=fact.source,
                       reason=why, **extra)
            if fact.key == "clearance.altitude":  # Crosscheck could not understand ATC: it says so (brain/rules.py)
                self.ledger.add(Fact("said", "clearance.unclear", fact.value, fact.t, fact.speaker, source="guard"))
            return
        self.ledger.add(fact)
        self.event("said", fact.t, key=fact.key, value=fact.value, speaker=fact.speaker, source=fact.source, **extra)

    def observe(self, r: PanelReading, t: float, frame_file: str | None = None, fact_t: float | None = None, **extra) -> None:
        """t: when the reading is applied (logged); fact_t: the time its facts are about (default t)."""
        t_event, t = t, (fact_t if fact_t is not None else t)
        facts = [("doing", "ias", r.ias), ("doing", "altitude", r.altitude),
                 ("doing", "track", r.track_mag), ("set", "mcp.speed", r.selected_speed),
                 ("set", "mcp.altitude", r.selected_altitude), ("set", "baro", r.baro)]
        if r.flaps is not None:
            facts += [("doing", "flaps", r.flaps), ("doing", "flaps.clock", r.flaps_clock)]
        if r.fma and (any(r.fma) or None in r.fma):
            # the line was read: a blank thrust cell means no engaged thrust mode, not "unknown"; a box
            # the reader did not look at is None (Glimmer's final lane reads thrust and pitch only)
            facts.append(("doing", "fma.thrust", r.fma[0] or "BLANK"))
            if r.fma[2] is not None:
                facts.append(("doing", "fma.pitch", r.fma[2] or "BLANK"))
        for stream, key, value in facts:
            if value is not None:
                self.ledger.add(Fact(stream, key, value, t, source="panel"))
        if r.altitude is not None:
            # the median of the last three altitudes: one misread (918 ft at 6,000) must not make it "final"
            alts = sorted(f.value for f in self.ledger.since("doing", "altitude", 0)[-3:])
            phase = "final" if alts[len(alts) // 2] < FINAL_BELOW_FT else "other"
            self.ledger.add(Fact("doing", "phase", phase, t, source="derived"))
        self.event("reading", t_event, frame=frame_file, **{k: getattr(r, k) for k in READING_FIELDS}, **extra)

    def observe_pilot(self, p: dict, t: float, frame_file: str | None = None, **extra) -> None:
        """One pilot-camera posture (senses/pilot_pose.py) from the opt-in pilot watch."""
        self.ledger.add(Fact("doing", "pilot.slumped", p["slumped"], t, source="pilot-camera"))
        self.event("pilot", t, frame=frame_file, **p, **extra)

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
                if self.explain:
                    from brain.explain import explain_later
                    explain_later(self.ledger, f, t, lambda r, f=f: self.event("explain", time.time(), rule=f.rule, key=f.key, **r))
        for k in [k for k in self.active if k not in current and t - self.last_seen[k] >= CLEAR_AFTER_S]:
            del self.active[k]
            self.event("cleared", t, rule=k[0], key=k[1])
        return new

    def _save(self, frame: Image.Image) -> str | None:
        """The crop the readers use, and (for people looking at the recording) the whole cockpit,
        downscaled to WIDE_W: wide/<same name>."""
        name = None
        if self.frames:
            name = f"frames/{self.n:05d}.jpg"
            frame.crop(self.frame_box).convert("RGB").save(self.dir / name, quality=85)
            w = frame.convert("RGB").resize((WIDE_W, round(frame.height * WIDE_W / frame.width)), Image.LANCZOS)
            w.save(self.dir / name.replace("frames/", "wide/"), quality=78)
        self.n += 1
        return name

    def step(self, frame: Image.Image, t: float) -> list[Flag]:
        self.observe(read_frame(frame), t, self._save(frame))
        return self.decide(t)

    def run(self, seconds: float, t0: float | None = None) -> None:
        t0 = t0 or time.time()
        if self.reader in SPARK_READERS:
            return self._run_spark(seconds, t0)
        while time.time() - t0 < seconds:
            self.step(capture_screen(), time.time())

    def _run_spark(self, seconds: float, t0: float) -> None:
        """Capture every SPARK_PERIOD_S, read on Muse Spark with several calls in flight, and
        apply readings in capture order, timestamped when they arrive: the reader's latency
        is part of the record, so replay reproduces the live run exactly. With
        fact_time="captured", each reading is applied as soon as it arrives (one slow call does
        not hold back newer ones) with its facts at capture time, and older readings that
        arrive after a newer one are dropped."""
        read = _spark_reader(self.reader)
        fast, last_full = None, 0.0
        if self.reader == "glimmer":  # on final, the final lane (senses/panel_glimmer.py): see FULL_EVERY_FINAL_S
            from senses.panel_glimmer import read_frame_glimmer_final as fast
        elif self.reader == "spark":  # the same on Muse Spark (senses/panel_spark.py)
            from senses.panel_spark import read_frame_spark_final as fast
        pending: collections.deque = collections.deque()
        next_capture = time.time()
        last_captured = last_full_applied = 0.0  # newest capture applied; newest whole read whose altitudes are applied
        with ThreadPoolExecutor(max_workers=SPARK_WORKERS) as pool:
            while time.time() - t0 < seconds or pending:
                if time.time() - t0 < seconds and time.time() >= next_capture and len(pending) < IN_FLIGHT.get(self.reader, 99):
                    frame = capture_screen()
                    captured = time.time()
                    name = self._save(frame)
                    fn, lane = read, None
                    if fast is not None:
                        lane = "full"
                        if self.ledger.value_at("doing", "phase", captured) == "final":
                            watching = self.ledger.value_at("doing", "fma.thrust", captured) not in SPEED_MODES
                            if watching or captured - last_full < FULL_EVERY_FINAL_S:
                                fn, lane = fast, "final"
                        last_full = captured if lane == "full" else last_full
                    pending.append((captured, name, lane, pool.submit(lambda f=frame, fn=fn: asyncio.run(fn(f))[0])))
                    next_capture = captured + (SPARK_FINAL_PERIOD_S if lane == "final" and self.reader == "spark" else SPARK_PERIOD_S)
                by_arrival = self.fact_time == "captured"
                while pending and (pending[0][3].done() or by_arrival and any(p[3].done() for p in pending)):
                    i = 0 if pending[0][3].done() else next(i for i, p in enumerate(pending) if p[3].done())
                    captured, name, lane, fut = pending[i]
                    del pending[i]
                    t = time.time()
                    try:
                        r = fut.result()
                    except Exception as e:  # one failed call: the frame stays unread, the run goes on
                        self.event("reader_error", t, frame=name, error=str(e)[:200])
                        continue
                    if by_arrival and captured < last_captured:  # a newer frame was read first: this one is stale
                        if lane == "full" and captured > last_full_applied and (r.altitude is not None or r.selected_altitude is not None):
                            # a whole read overtaken by quicker final-lane reads (Spark reads in parallel) still
                            # brings what only it reads, unless a newer whole read already has; its speeds and
                            # modes are older than the ones applied
                            last_full_applied = captured
                            self.observe(PanelReading(selected_altitude=r.selected_altitude, altitude=r.altitude,
                                                      altitude_method=r.altitude_method), t, name, fact_t=captured,
                                         captured=round(captured, 3), reader=READER_LABEL[self.reader], lane="full", late=True)
                            self.decide(t)
                            continue
                        self.event("late_reading", t, frame=name, captured=round(captured, 3))
                        continue
                    last_captured = captured
                    last_full_applied = captured if lane != "final" else last_full_applied
                    self.observe(r, t, name, fact_t=captured if by_arrival else None, captured=round(captured, 3),
                                 reader=READER_LABEL[self.reader].removesuffix("-flaps"), **({"lane": lane} if lane else {}))
                    self.decide(t)
                time.sleep(0.05)


def _spark_readings(rec_dir: pathlib.Path, frames: list[str], reader: str = "spark",
                    origin: tuple[int, int] = FRAME_ORIGIN) -> dict[str, PanelReading]:
    import httpx
    read_frame_spark = _spark_reader(reader)

    async def go():
        sem = asyncio.Semaphore(6)
        async with httpx.AsyncClient(trust_env=False, timeout=90) as client:
            async def one(f):
                async with sem:
                    return f, (await read_frame_spark(frame_from_crop(Image.open(rec_dir / f), origin), client))[0]
            return dict(await asyncio.gather(*(one(f) for f in frames)))
    return asyncio.run(go())


def replay(rec_dir: pathlib.Path, out_dir: pathlib.Path, recompute: bool = False, reader: str = "apple") -> Crosscheck:
    """Re-run a recording through the same ledger and rules.

    Default: use the panel readings saved with the recording (works on any OS, no models).
    recompute=True: re-read every saved frame first, with senses.panel (reader="apple", macOS)
    or Muse Spark through the Meta Model API (reader="spark", any OS, needs MODEL_API_KEY).
    """
    events = [json.loads(l) for l in (rec_dir / "events.jsonl").read_text().splitlines()]
    meta_p = rec_dir / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    if meta.get("panel_reader") in ("muse-spark-flaps", "muse-glimmer-flaps") and reader in ("spark", "spark-flaps"):
        reader = "spark-flaps"  # the flaps view has no PFD calibration; only the gauge reader applies
    if meta.get("panel_reader") in ("muse-spark-flaps", "muse-glimmer-flaps") and reader in ("glimmer", "glimmer-flaps"):
        reader = "glimmer-flaps"
    spark = None
    if recompute and reader in SPARK_READERS:
        frames = [e["frame"] for e in events if e["kind"] == "reading" and e.get("frame")]
        spark = _spark_readings(rec_dir, frames, reader, tuple(meta.get("frame_box", FRAME_BOX))[:2])
    cc = Crosscheck(out_dir, log_name="replay_events.jsonl")
    for e in sorted((e for e in events if e["kind"] in ("reading", "said", "said_rejected", "pilot")),
                    key=lambda e: (e["t"], not e["kind"].startswith("said"))):
        if e["kind"] in ("said", "said_rejected"):  # the plausibility guard decides again, from the same transcript
            cc.said(Fact("said", e["key"], e["value"], e["t"], e["speaker"], source=e["source"]), transcript=e.get("transcript"))
            continue  # live, decisions happen on camera frames only
        if e["kind"] == "pilot":  # postures are always replayed as recorded (re-posing needs the GPU)
            cc.observe_pilot({k: e[k] for k in ("neck_flex", "wrists", "slumped", "reasons")}, e["t"], e.get("frame"))
            cc.decide(e["t"])
            continue
        if spark is not None:
            r = spark[e["frame"]]
        elif recompute:
            if not (rec_dir / e["frame"]).exists():
                raise FileNotFoundError(f"{rec_dir / e['frame']}: this recording ships only some of its frames "
                                        "(replay/nominal keeps every 5th), so it replays from its saved readings only")
            r = read_frame(frame_from_crop(Image.open(rec_dir / e["frame"])))
        else:
            r = PanelReading(**{k: e[k] for k in READING_FIELDS if k in e})
        captured = e.get("captured") if meta.get("fact_time") == "captured" else None
        cc.observe(r, e["t"], e.get("frame"), fact_t=captured)
        cc.decide(e["t"])
    return cc
