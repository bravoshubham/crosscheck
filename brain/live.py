"""The live monitoring loop: frame -> panel reading -> ledger facts -> rules -> callout.

Each flag is spoken once, then again only after it has cleared and come back. Every step is
written to an event log so the evaluator can score it against simulator truth.
"""
import json
import pathlib
import subprocess
import tempfile
import time
from dataclasses import asdict

from PIL import Image

from brain.ledger import Fact, Ledger
from brain.rules import Flag, evaluate
from senses.panel import PanelReading, read_frame

FINAL_BELOW_FT = 2500  # crude phase detection until checklists and the FMS phase are read


def capture_screen() -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        subprocess.run(["screencapture", "-x", f.name], check=True)
        return Image.open(f.name).copy()


class Crosscheck:
    def __init__(self, log_path: pathlib.Path, speak=None):
        self.ledger = Ledger()
        self.active: dict[tuple[str, str], Flag] = {}
        self.log = log_path.open("w")
        self.speak = speak

    def event(self, kind: str, t: float, **data) -> None:
        self.log.write(json.dumps({"t": t, "kind": kind, **data}) + "\n")
        self.log.flush()

    def observe(self, r: PanelReading, t: float) -> None:
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
        self.event("reading", t, ias=r.ias, altitude=r.altitude, track=r.track_mag,
                   selected_speed=r.selected_speed, selected_altitude=r.selected_altitude,
                   fma=r.fma, fma_armed=r.fma_armed)

    def step(self, frame: Image.Image, t: float) -> list[Flag]:
        self.observe(read_frame(frame), t)
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

    def run(self, seconds: float, t0: float | None = None) -> None:
        t0 = t0 or time.time()
        while time.time() - t0 < seconds:
            t = time.time()
            self.step(capture_screen(), t)
