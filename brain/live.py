"""The monitoring loop: frame -> panel reading -> ledger facts -> rules -> callout.

Live and replay share this code. Live: capture the screen, read the panel, decide. Replay:
feed the readings and "said" facts saved with a recording back in at their original times,
so a judge can reproduce every callout without X-Plane, a GPU or an API key.

Each flag is spoken once, then again only after it has cleared and come back. Everything is
written to <dir>/events.jsonl; with frames=True the panel crops go to <dir>/frames/.
"""
import json
import pathlib
import subprocess
import tempfile
import time
from dataclasses import asdict, fields

from PIL import Image

from brain.ledger import Fact, Ledger
from brain.rules import Flag, evaluate
from senses.panel import PanelReading, read_frame

FINAL_BELOW_FT = 2500  # crude phase detection until checklists and the FMS phase are read
# Region of the screen kept with a recording: every calibrated box plus a margin for view shifts.
FRAME_ORIGIN = (1240, 1230)
FRAME_BOX = (1240, 1230, 3180, 2200)
SCREEN_SIZE = (3600, 2338)
READING_FIELDS = [f.name for f in fields(PanelReading) if f.name not in ("raw", "roll_labels")]


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
    def __init__(self, out_dir: pathlib.Path, speak=None, frames: bool = False, log_name: str = "events.jsonl"):
        self.dir = out_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger()
        self.active: dict[tuple[str, str], Flag] = {}
        self.log = (out_dir / log_name).open("w")
        self.speak = speak
        self.frames = frames
        self.n = 0
        if frames:
            (out_dir / "frames").mkdir(exist_ok=True)

    def event(self, kind: str, t: float, **data) -> None:
        self.log.write(json.dumps({"t": t, "kind": kind, **data}) + "\n")
        self.log.flush()

    def said(self, fact: Fact) -> None:
        """A spoken clearance, readback or callout (from speech, or from a scenario script)."""
        self.ledger.add(fact)
        self.event("said", fact.t, key=fact.key, value=fact.value, speaker=fact.speaker, source=fact.source)

    def observe(self, r: PanelReading, t: float, frame_file: str | None = None) -> None:
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
        self.event("reading", t, frame=frame_file, **{k: getattr(r, k) for k in READING_FIELDS})

    def decide(self, t: float) -> list[Flag]:
        flags = evaluate(self.ledger, t)
        new = []
        current = {(f.rule, f.key): f for f in flags}
        for k, f in current.items():
            if k not in self.active:
                new.append(f)
                self.event("callout", t, **asdict(f))
                if self.speak:
                    self.speak(f.callout)
        for k in set(self.active) - set(current):
            self.event("cleared", t, rule=k[0], key=k[1])
        self.active = current
        return new

    def step(self, frame: Image.Image, t: float) -> list[Flag]:
        name = None
        if self.frames:
            name = f"frames/{self.n:05d}.jpg"
            frame.crop(FRAME_BOX).convert("RGB").save(self.dir / name, quality=85)
        self.n += 1
        self.observe(read_frame(frame), t, name)
        return self.decide(t)

    def run(self, seconds: float, t0: float | None = None) -> None:
        t0 = t0 or time.time()
        while time.time() - t0 < seconds:
            self.step(capture_screen(), time.time())


def replay(rec_dir: pathlib.Path, out_dir: pathlib.Path, recompute: bool = False) -> Crosscheck:
    """Re-run a recording through the same ledger and rules.

    Default: use the panel readings saved with the recording (works on any OS, no models).
    recompute=True: re-read every saved frame with senses.panel first (macOS: Apple Vision).
    """
    events = [json.loads(l) for l in (rec_dir / "events.jsonl").read_text().splitlines()]
    cc = Crosscheck(out_dir, log_name="replay_events.jsonl")
    for e in sorted((e for e in events if e["kind"] in ("reading", "said")), key=lambda e: (e["t"], e["kind"] != "said")):
        if e["kind"] == "said":
            cc.said(Fact("said", e["key"], e["value"], e["t"], e["speaker"], source=e["source"]))
            continue
        if recompute:
            r = read_frame(frame_from_crop(Image.open(rec_dir / e["frame"])))
        else:
            r = PanelReading(**{k: e[k] for k in READING_FIELDS if k in e})
        cc.observe(r, e["t"], e.get("frame"))
        cc.decide(e["t"])
    return cc
