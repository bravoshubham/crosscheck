"""Deterministic cross-check rules: they fire instantly and word standard callouts.

Each rule compares facts in the ledger and returns Flags. The model (Muse Spark or Glimmer)
only decides what to do with a flag in non-standard cases; standard callouts use the
template text here, so the wording is always exact phraseology.
"""
from dataclasses import dataclass

from brain import phraseology as say
from brain.ledger import Ledger

SPEED_MODES = {"MCP SPD", "SPEED", "FMC SPD", "SPD"}


@dataclass(frozen=True)
class Flag:
    rule: str
    key: str
    expected: object
    actual: object
    since: float
    callout: str


def readback_vs_clearance(led: Ledger, now: float, window: float = 20.0) -> list[Flag]:
    """ATC cleared one altitude, the crew read back another."""
    clr = led.latest("said", "clearance.altitude", before=now, speaker="atc")
    if clr is None:
        return []
    rbs = [f for f in led.since("said", "readback.altitude", clr.t) if f.t <= min(now, clr.t + window)]
    if not rbs or rbs[-1].value == clr.value:
        return []
    rb = rbs[-1]
    return [Flag("readback_vs_clearance", "altitude", clr.value, rb.value, rb.t,
                 f"Readback {say.altitude(rb.value)}, cleared {say.altitude(clr.value)}.")]


def _stable_set(led: Ledger, key: str, now: float, votes: int = 3) -> tuple[object, float] | None:
    """The selected value as the majority of the last few camera readings (one misread such as
    "1200" for "12000" must not count), and when that value was first seen in the current run."""
    hist = [f for f in led.since("set", key, float("-inf")) if f.t <= now]
    if not hist:
        return None
    recent = [f.value for f in hist[-votes:]]
    value = max(set(recent), key=recent.count)
    if recent.count(value) * 2 <= len(recent) and len(recent) > 1:
        return None
    start = None
    for f in reversed(hist):
        if f.value == value:
            start = f.t
        elif start is not None and abs(f.value - value) >= 50:
            # a different value read twice in a row ends the run; a lone misread does not
            prev = hist[hist.index(f) - 1] if hist.index(f) > 0 else None
            if prev is not None and prev.value == f.value:
                break
    return value, start


def clearance_vs_set(led: Ledger, now: float, grace: float = 8.0) -> list[Flag]:
    """The MCP altitude differs from the latest cleared altitude for longer than the grace period,
    timed from when the mismatching value first appeared, not from the latest reading."""
    clr = led.latest("said", "clearance.altitude", before=now, speaker="atc")
    stable = _stable_set(led, "mcp.altitude", now)
    if clr is None or stable is None:
        return []
    value, first_seen = stable
    if abs(value - clr.value) < 50:
        return []
    since = max(clr.t, first_seen)
    if now - since < grace:
        return []
    return [Flag("clearance_vs_set", "altitude", clr.value, value, since,
                 f"Cleared {say.altitude(clr.value)}, set {say.altitude(value)}.")]


def callout_vs_set(led: Ledger, now: float, window: float = 6.0) -> list[Flag]:
    """A flap or gear callout was made but the lever did not reach that position in time."""
    flags = []
    for item, word in (("flaps", say.flaps), ("gear", str)):
        call = led.latest("said", f"callout.{item}", before=now)
        if call is None or now - call.t < window:
            continue
        lever = led.latest("set", f"lever.{item}", before=now)
        if lever is None or lever.value == call.value:
            continue
        if item == "flaps":
            text = f"Flaps {word(call.value)} called, {word(lever.value)} set."
        else:
            text = f"Gear {call.value} called, gear still {lever.value}."
        flags.append(Flag("callout_vs_set", item, call.value, lever.value, call.t + window, text))
    return flags


def mode_vs_phase(led: Ledger, now: float, speed_margin: float = 5.0) -> list[Flag]:
    """On final, the autothrottle is not controlling speed and speed is below target."""
    if led.value_at("doing", "phase", now) != "final":
        return []
    thr = led.value_at("doing", "fma.thrust", now)
    ias = led.value_at("doing", "ias", now)
    target = led.value_at("set", "mcp.speed", now)
    if thr is None or ias is None or target is None or thr in SPEED_MODES or ias > target - speed_margin:
        return []
    return [Flag("mode_vs_phase", "autothrottle", "speed mode", thr, now, "Speed. Autothrottle not in speed.")]


RULES = (readback_vs_clearance, clearance_vs_set, callout_vs_set, mode_vs_phase)


def evaluate(led: Ledger, now: float) -> list[Flag]:
    return [f for rule in RULES for f in rule(led, now)]
