from brain import phraseology as say
from brain.ledger import Fact, Ledger
from brain.rules import evaluate


def ledger(*facts):
    led = Ledger()
    for f in facts:
        led.add(Fact(*f))
    return led


def test_phraseology():
    assert say.altitude(24000) == "two-four-zero"
    assert say.altitude(25000) == "two-five-zero"
    assert say.altitude(11000) == "one-one thousand"
    assert say.altitude(3500) == "three thousand five hundred"
    assert say.heading(270) == "two-seven-zero"
    assert say.heading(5) == "zero-zero-five"
    assert say.flaps(15) == "fifteen"


def test_altitude_bust_flagged_after_grace():
    led = ledger(("said", "clearance.altitude", 24000, 0.0, "atc"),
                 ("said", "readback.altitude", 24000, 3.0, "pm"),
                 ("set", "mcp.altitude", 25000, 5.0))
    assert evaluate(led, 10.0) == []  # inside the grace period
    flags = evaluate(led, 14.0)
    assert [f.callout for f in flags] == ["Cleared two-four-zero, set two-five-zero."]


def test_altitude_set_correctly_is_silent():
    led = ledger(("said", "clearance.altitude", 24000, 0.0, "atc"),
                 ("set", "mcp.altitude", 24000, 4.0))
    assert evaluate(led, 30.0) == []


def test_readback_error():
    led = ledger(("said", "clearance.altitude", 24000, 0.0, "atc"),
                 ("said", "readback.altitude", 25000, 3.0, "pm"),
                 ("set", "mcp.altitude", 24000, 4.0))
    assert [f.callout for f in evaluate(led, 5.0)] == ["Readback two-five-zero, cleared two-four-zero."]


def test_flap_callout_mismatch_and_fix():
    led = ledger(("said", "callout.flaps", 15, 0.0, "pf"), ("set", "lever.flaps", 5, 2.0))
    assert evaluate(led, 3.0) == []
    assert [f.callout for f in evaluate(led, 7.0)] == ["Flaps fifteen called, five set."]
    led.add(Fact("set", "lever.flaps", 15, 8.0))
    assert evaluate(led, 9.0) == []


def test_gear_callout_mismatch():
    led = ledger(("said", "callout.gear", "up", 0.0, "pf"), ("set", "lever.gear", "down", 1.0))
    assert [f.callout for f in evaluate(led, 7.0)] == ["Gear up called, gear still down."]


def test_autothrottle_not_in_speed_on_final():
    led = ledger(("doing", "phase", "final", 0.0), ("set", "mcp.speed", 140, 0.0),
                 ("doing", "fma.thrust", "HOLD", 1.0), ("doing", "ias", 132, 2.0))
    assert [f.callout for f in evaluate(led, 2.0)] == ["Speed. Autothrottle not in speed."]


def test_autothrottle_in_speed_is_silent():
    led = ledger(("doing", "phase", "final", 0.0), ("set", "mcp.speed", 140, 0.0),
                 ("doing", "fma.thrust", "MCP SPD", 1.0), ("doing", "ias", 132, 2.0))
    assert evaluate(led, 2.0) == []
