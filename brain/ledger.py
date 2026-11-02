"""The cross-check ledger: typed facts from three streams.

said  - what was spoken: ATC clearances, crew readbacks and callouts (from speech)
set   - what the crew selected: MCP windows, levers, selectors (from the panel camera)
doing - what the aircraft is doing: speed, altitude, heading, autopilot modes (from the panel camera)

Keys are dotted names shared across streams so rules can compare them, e.g.
said `clearance.altitude` vs set `mcp.altitude`, said `callout.flaps` vs set `lever.flaps`.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

Stream = Literal["said", "set", "doing"]


@dataclass(frozen=True)
class Fact:
    stream: Stream
    key: str
    value: object
    t: float
    speaker: str | None = None  # "atc", "pf", "pm" for said facts
    source: str = ""            # e.g. "speech", "panel", "replay"


@dataclass
class Ledger:
    facts: list[Fact] = field(default_factory=list)
    _by_key: dict[tuple[str, str], list[Fact]] = field(default_factory=lambda: defaultdict(list))

    def add(self, fact: Fact) -> None:
        self.facts.append(fact)
        self._by_key[(fact.stream, fact.key)].append(fact)

    def latest(self, stream: Stream, key: str, before: float | None = None, speaker: str | None = None) -> Fact | None:
        for f in reversed(self._by_key[(stream, key)]):
            if (before is None or f.t <= before) and (speaker is None or f.speaker == speaker):
                return f
        return None

    def since(self, stream: Stream, key: str, t0: float, speaker: str | None = None) -> list[Fact]:
        return [f for f in self._by_key[(stream, key)] if f.t >= t0 and (speaker is None or f.speaker == speaker)]

    def value_at(self, stream: Stream, key: str, t: float):
        f = self.latest(stream, key, before=t)
        return None if f is None else f.value
