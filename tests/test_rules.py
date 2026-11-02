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
                 ("doing", "fma.thrust", "HOLD", 1.0), ("doing", "fma.thrust", "HOLD", 1.3), ("doing", "fma.thrust", "HOLD", 1.6),
                 ("doing", "ias", 132, 2.0))
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


def _gauge(led, readings):
    for t, clock in readings:
        led.add(Fact("doing", "flaps.clock", clock, t, source="panel"))


def test_flaps_from_the_gauge_wait_for_the_needle_to_settle():
    # 15 called at 0 s; the needle moves at the measured ~0.39 clock-hours a second from 5
    # (12.25) and stops at 10 (14.0); one reading every 1.5 s
    led = ledger(("said", "callout.flaps", 15, 0.0, "pf"))
    _gauge(led, [(1.5, 12.2), (3.0, 12.2), (4.5, 12.6), (6.0, 13.2), (7.5, 13.8), (9.0, 14.0), (10.5, 14.1),
                 (12.0, 14.0), (13.5, 14.0)])
    assert evaluate(led, 10.5) == []  # only four readings after the crew's response time
    assert evaluate(led, 12.0) == []  # 13.2 .. 14.0: still a trend as the needle slows
    assert [f.callout for f in evaluate(led, 13.5)] == ["Flaps fifteen called, ten set."]


def test_flaps_travelling_to_the_called_setting_is_silent():
    led = ledger(("said", "callout.flaps", 15, 0.0, "pf"))
    for i, clock in enumerate([12.2, 12.8, 13.4, 14.0, 14.6, 15.0, 15.0, 15.0, 15.1, 15.0]):
        _gauge(led, [(1.5 * (i + 1), clock)])
        assert evaluate(led, 1.5 * (i + 1)) == [], clock


def test_flaps_never_moved():
    # Spark's reads of a needle parked at 5, as seen live (11.5 .. 12.5)
    led = ledger(("said", "callout.flaps", 15, 0.0, "pf"))
    _gauge(led, [(1.5 * (i + 1), c) for i, c in enumerate([12.0, 12.5, 11.5, 11.5, 12.0, 11.5, 12.5, 12.2, 11.7, 11.5])])
    assert evaluate(led, 10.5) == []  # four readings since the crew's response time
    assert [f.callout for f in evaluate(led, 13.5)] == ["Flaps fifteen called, five set."]


def test_flaps_not_yet_moved_is_not_a_missed_selection():
    # live, the camera's readings lag ~3.7 s: just after the call they still show the old setting
    led = ledger(("said", "callout.flaps", 15, 0.0, "pf"))
    for t, clock in [(1.5, 12.1), (3.0, 12.2), (4.5, 12.1), (6.0, 12.6), (7.5, 13.2), (9.0, 13.8), (10.5, 14.4),
                     (12.0, 15.0), (13.5, 15.0), (15.0, 15.1), (16.5, 15.0)]:
        _gauge(led, [(t, clock)])
        assert evaluate(led, t) == [], t


def test_jitter_on_a_settled_needle_does_not_clear_the_flag():
    led = ledger(("said", "callout.flaps", 15, 0.0, "pf"))
    _gauge(led, [(6.0 + 1.5 * i, c) for i, c in enumerate([12.0, 11.5, 12.5, 11.5, 12.0, 11.5, 12.8, 11.5, 11.5, 12.5])])
    for t in (12.0, 15.0, 18.0, 19.5):
        assert [f.callout for f in evaluate(led, t)] == ["Flaps fifteen called, five set."], t


def _posture(led, states):
    for t, slumped in states:
        led.add(Fact("doing", "pilot.slumped", slumped, t, source="pilot-camera"))


def test_slumped_pilot_is_asked_then_reported():
    led = Ledger()
    _posture(led, [(float(t), t >= 10) for t in range(0, 40, 2)])
    assert evaluate(led, 14.0) == []                      # slumped 4 s
    assert [f.callout for f in evaluate(led, 16.0)] == ["Captain, are you OK?"]
    assert [f.callout for f in evaluate(led, 26.0)] == ["Captain, are you OK?",
                                                        "Captain not responding. First officer, take control."]


def test_an_answer_or_sitting_up_clears_it():
    led = Ledger()
    _posture(led, [(float(t), t >= 10) for t in range(0, 40, 2)])
    led.add(Fact("said", "pilot.voice", "I'm fine, just dropped my pen", 18.0, "capt"))
    assert evaluate(led, 30.0) == []
    led2 = Ledger()
    _posture(led2, [(float(t), 10 <= t < 20) for t in range(0, 40, 2)])
    assert evaluate(led2, 22.0) == []


def test_needle_creeping_to_the_called_setting_is_not_settled():
    # seen live (repeat run): slow end of travel, two identical reads; range 1.0, mean 14.5 (between 10 and 15)
    led = ledger(("said", "callout.flaps", 15, 0.0, "pf"))
    _gauge(led, [(11.0, 14.0), (14.1, 14.0), (16.8, 14.5), (16.9, 15.0), (18.3, 15.0)])
    assert evaluate(led, 18.3) == []
    _gauge(led, [(19.6, 15.0), (23.0, 15.0), (23.1, 15.0), (24.8, 15.0)])
    assert evaluate(led, 24.8) == []


def _decaying(led, speeds, t0=1.0):
    for i, v in enumerate(speeds):
        led.add(Fact("doing", "ias", v, t0 + i))


def test_autothrottle_speed_decaying_fast_is_called_before_5_kt_low():
    # 140 kt target, speed falling 1.5 kt/s: still only 3 kt low, but heading past 135 within 10 s
    led = ledger(("doing", "phase", "final", 0.0), ("set", "mcp.speed", 140, 0.0), ("doing", "fma.thrust", "ARM", 0.5),
                 ("doing", "fma.thrust", "ARM", 0.7), ("doing", "fma.thrust", "ARM", 0.9))
    _decaying(led, [141.5, 140, 138.5, 137])
    assert [f.rule for f in evaluate(led, 4.0)] == ["mode_vs_phase"]


def test_autothrottle_steady_slightly_slow_is_silent():
    # a steady 2 kt below target with the thrust mode not in speed: no trend, inside the margin
    led = ledger(("doing", "phase", "final", 0.0), ("set", "mcp.speed", 140, 0.0), ("doing", "fma.thrust", "ARM", 0.5))
    _decaying(led, [138, 138, 138, 138])
    assert evaluate(led, 4.0) == []


def test_autothrottle_above_target_and_falling_is_silent():
    led = ledger(("doing", "phase", "final", 0.0), ("set", "mcp.speed", 140, 0.0), ("doing", "fma.thrust", "ARM", 0.5))
    _decaying(led, [150, 147, 144, 141])
    assert evaluate(led, 4.0) == []


def test_level_change_descent_is_silent():
    # RETARD with MCP SPD in the pitch column: the elevator holds the speed (nominal flight, 666 s)
    led = ledger(("doing", "phase", "final", 0.0), ("set", "mcp.speed", 190, 0.0),
                 ("doing", "fma.thrust", "RETARD", 0.5), ("doing", "fma.thrust", "RETARD", 0.7), ("doing", "fma.thrust", "RETARD", 0.8),
                 ("doing", "fma.pitch", "MCP SPD", 0.7), ("doing", "fma.pitch", "MCP SPD", 0.8))
    _decaying(led, [189, 187, 185, 183])
    assert evaluate(led, 4.0) == []


def test_two_misread_blank_thrust_cells_are_silent():
    # slowing to a new speed bug, MCP SPD on every frame but two (nominal flight, run 2, 748 s)
    led = ledger(("doing", "phase", "final", 0.0), ("set", "mcp.speed", 170, 0.0),
                 ("doing", "fma.thrust", "MCP SPD", 0.5), ("doing", "fma.thrust", "BLANK", 3.5), ("doing", "fma.thrust", "BLANK", 3.9))
    _decaying(led, [172, 171, 169, 167])
    assert evaluate(led, 4.0) == []


def test_without_a_readback_the_crew_gets_the_time_a_readback_takes():
    # 15-minute flight, run 3: the readback was lost to recognition; the PF set the MCP 10 s after the clearance
    led = ledger(("said", "clearance.altitude", 4000, 0.0, "atc"))
    for i in range(10):
        led.add(Fact("set", "mcp.altitude", 5000, i * 1.2))
    assert evaluate(led, 10.0) == []
    for i in range(10, 14):
        led.add(Fact("set", "mcp.altitude", 5000, i * 1.2))
    assert [f.rule for f in evaluate(led, 16.0)] == ["clearance_vs_set"]
