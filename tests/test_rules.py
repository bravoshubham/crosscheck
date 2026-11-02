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


def test_altitude_bust_with_a_reading_every_frame_and_a_misread():
    """Regression: the grace period must run from when 25000 first appeared, even though a new
    camera reading arrives every second, and one misread ("2500") must not reset it."""
    led = ledger(("said", "clearance.altitude", 24000, 0.0, "atc"))
    for i in range(20):
        t = 1.0 + i
        value = 2500 if i == 6 else (24000 if i < 3 else 25000)
        led.add(Fact("set", "mcp.altitude", value, t))
    flags = evaluate(led, 20.0)
    assert [f.callout for f in flags] == ["Cleared two-four-zero, set two-five-zero."]
    assert flags[0].since == 4.0


def test_correct_setting_with_readings_every_frame_is_silent():
    led = ledger(("said", "clearance.altitude", 11000, 0.0, "atc"))
    for i in range(20):
        led.add(Fact("set", "mcp.altitude", 9000 if i < 3 else 11000, 1.0 + i))
    assert evaluate(led, 20.0) == []


def test_one_flickering_reading_does_not_repeat_a_callout(tmp_path):
    from brain.live import Crosscheck
    from senses.panel import PanelReading
    cc = Crosscheck(tmp_path)
    cc.said(Fact("said", "clearance.altitude", 11000, 0.0, "atc", source="test"))
    said = []
    for i in range(30):
        t = 1.0 + i
        alt = 11000 if i == 15 else 12000  # one misread that matches the clearance
        cc.observe(PanelReading(selected_altitude=alt), t)
        said += [f.callout for f in cc.decide(t)]
    assert said == ["Cleared one-one thousand, set one-two thousand."]


def test_grace_starts_at_the_readback():
    # live speech timing: clearance heard at 18 s, readback at 25 s, MCP still 9000 until set at 27 s
    led = Ledger()
    for t in range(0, 40):
        led.add(Fact("set", "mcp.altitude", 9000 if t < 27 else 11000, float(t), source="panel"))
    led.add(Fact("said", "clearance.altitude", 11000, 18.0, "atc"))
    led.add(Fact("said", "readback.altitude", 11000, 25.0, "pm"))
    assert not [f for t in range(18, 40) for f in evaluate(led, float(t))]
    # never set: flagged 8 s after the readback
    led2 = Ledger()
    for t in range(0, 40):
        led2.add(Fact("set", "mcp.altitude", 9000, float(t), source="panel"))
    led2.add(Fact("said", "clearance.altitude", 11000, 18.0, "atc"))
    led2.add(Fact("said", "readback.altitude", 11000, 25.0, "pm"))
    assert not evaluate(led2, 32.0)
    assert [f.rule for f in evaluate(led2, 33.0)] == ["clearance_vs_set"]
