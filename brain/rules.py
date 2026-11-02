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
    timed from when the mismatching value first appeared, not from the latest reading. The crew
    normally sets the MCP after reading the clearance back, so the grace starts at the readback
    when there is one (live speech made this visible: clearance plus readback take ~12 s)."""
    clr = led.latest("said", "clearance.altitude", before=now, speaker="atc")
    stable = _stable_set(led, "mcp.altitude", now)
    if clr is None or stable is None:
        return []
    value, first_seen = stable
    if abs(value - clr.value) < 50:
        return []
    rb = led.latest("said", "readback.altitude", before=now)
    since = max(clr.t, first_seen, rb.t if rb is not None and rb.t >= clr.t else clr.t)
    if now - since < grace:
        return []
    return [Flag("clearance_vs_set", "altitude", clr.value, value, since,
                 f"Cleared {say.altitude(clr.value)}, set {say.altitude(value)}.")]


def _settled_flaps(led: Ledger, after: float, now: float, n: int = 5, span: float = 5.0, spread: float = 1.3,
                   max_slope: float = 0.1):
    """The flap gauge's detent once the needle has stopped, from the last n needle positions
    (clock hours, read from the camera) taken after `after`: they must cover at least `span`
    seconds, lie within `spread`, and show no trend (fitted slope under `max_slope` hours a
    second); the detent is taken from their mean. Spark's reads of a parked needle scatter by up
    to ~1.3 h with no trend (noise slope ~0.08 h/s over five reads); the 737's needle moves at
    0.15-0.39 h/s, slowest near the end of travel, where the range test alone once mistook
    14.0, 14.0, 14.5, 15.0, 15.0 for a needle parked between 10 and 15."""
    from senses.flaps_spark import detent
    clock = [f for f in led.since("doing", "flaps.clock", after) if f.t <= now][-n:]
    if len(clock) < n or clock[-1].t - clock[0].t < span:
        return None
    values = [f.value for f in clock]
    if max(values) - min(values) > spread:
        return None
    tm = sum(f.t for f in clock) / n
    vm = sum(values) / n
    slope = sum((f.t - tm) * (f.value - vm) for f in clock) / sum((f.t - tm) ** 2 for f in clock)
    if abs(slope) > max_slope:
        return None
    return detent(vm)


FLAP_RESPONSE_S = 5.0  # time the crew gets to move the lever after the call is heard: needle
                      # readings arriving sooner may show the old setting, not a missed one


def _flaps_position(led: Ledger, after: float, now: float):
    """The most recent settled flap detent since `after`. Once the needle has settled it stays
    the answer until it settles somewhere else, so a jittery read cannot clear a flag."""
    times = [f.t for f in led.since("doing", "flaps.clock", after) if f.t <= now]
    for t in reversed(times):
        v = _settled_flaps(led, after, t)
        if v is not None:
            return v
    return None


def callout_vs_set(led: Ledger, now: float, window: float = 6.0) -> list[Flag]:
    """A flap or gear callout was made but the lever (or, for flaps, the settled flap gauge)
    did not reach that position."""
    flags = []
    for item, word in (("flaps", say.flaps), ("gear", str)):
        call = led.latest("said", f"callout.{item}", before=now)
        if call is None or now - call.t < window:
            continue
        lever = led.latest("set", f"lever.{item}", before=now)
        actual = lever.value if lever is not None else None
        if lever is None and item == "flaps":
            actual = _flaps_position(led, call.t + FLAP_RESPONSE_S, now)
        if actual is None or actual == call.value:
            continue
        if item == "flaps":
            text = f"Flaps {word(call.value)} called, {word(actual)} set."
        else:
            text = f"Gear {call.value} called, gear still {actual}."
        flags.append(Flag("callout_vs_set", item, call.value, actual, call.t + window, text))
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


SLUMP_HOLD_S = 5.0       # slumped this long before Crosscheck asks
RESPONSE_WAIT_S = 10.0   # no answer this long after asking before it alerts


def pilot_watch(led: Ledger, now: float) -> list[Flag]:
    """Opt-in pilot camera. A slumped posture alone never alerts: Crosscheck first asks the pilot,
    and alerts the other pilot only if there is no answer. Anything the pilot says, or sitting
    back up, clears it."""
    hist = [f for f in led.since("doing", "pilot.slumped", float("-inf")) if f.t <= now]
    if not hist or not hist[-1].value:
        return []
    start = hist[-1].t
    for f in reversed(hist):
        if not f.value:
            break
        start = f.t
    if now - start < SLUMP_HOLD_S:
        return []
    if [f for f in led.since("said", "pilot.voice", start) if f.t <= now]:
        return []
    asked = start + SLUMP_HOLD_S
    flags = [Flag("pilot_watch", "challenge", "upright", "slumped", start, "Captain, are you OK?")]
    if now - asked >= RESPONSE_WAIT_S:
        flags.append(Flag("pilot_watch", "incapacitated", "answer", "none", asked,
                          "Captain not responding. First officer, take control."))
    return flags


RULES = (readback_vs_clearance, clearance_vs_set, callout_vs_set, mode_vs_phase, pilot_watch)


def evaluate(led: Ledger, now: float) -> list[Flag]:
    return [f for rule in RULES for f in rule(led, now)]
