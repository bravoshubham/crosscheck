from senses.speech import normalise, parse_altitude, said_facts


def test_numbers_collapse_to_values():
    assert normalise("alaska four two one descend and maintain one one thousand") == "alaska 421 descend and maintain 11000"
    assert normalise("Cesna, 345 Charlie Papa 3500.") == normalise("cessna three four five charlie papa three thousand five hundred").replace("cessna", "cesna")
    assert normalise("vats 15.") == "vats 15"


def test_soundalikes_only_inside_numbers():
    assert normalise("reduce speed to two one zero knots") == "reduce speed to 210 knots"
    assert normalise("Alaska for 21 contact Portland approach 124.35") == "alaska 421 contact portland approach 124 point 35"
    assert normalise("Alas, for two one clear deals") == "alas 421 clear deals"
    assert normalise("cleared for takeoff") == "cleared for takeoff"


def test_parse_altitude_from_real_transcripts():
    # Voice Transcribe outputs from the radio set (eval/results/day_one/t2_muse_cache.json)
    assert parse_altitude("Alaska 421 descend and maintain one one thousand.") == 11000
    assert parse_altitude("Alaska four two one climbed and maintain flight level two four zero") == 24000
    assert parse_altitude(" Plum, flight level 240, Oscar 421.") == 24000
    assert parse_altitude("Cesna, 345 Charlie Papa 3500.") == 3500  # verb dropped: the only altitude-shaped number
    assert parse_altitude(" plan and mention one 2000 Alaska 421..") == 12000  # live readback, verb misheard
    assert parse_altitude("Cessna 345 Charlie Papa Squawk 4321") is None
    assert parse_altitude("Alaska 421 contact Portland approach 124.35") is None
    assert parse_altitude("Alaska 421 reduce speed to 210 knots.") is None
    assert parse_altitude("Alaska 421 turn left hitting 270") is None


def test_said_facts_channel_and_callsign():
    f = said_facts("Oscar 421 climb and maintain one one thousand", "atc", 1.0, "421")
    assert [(x.key, x.value, x.speaker) for x in f] == [("clearance.altitude", 11000, "atc")]
    assert said_facts("Delta 88 climb and maintain one one thousand", "atc", 1.0, "421") == []
    rb = said_facts(" plan and mention one 2000 Alaska 421..", "pm", 2.0, "421")
    assert [(x.key, x.value) for x in rb] == [("readback.altitude", 12000)]
    assert said_facts("One thousand stabilized.", "pm", 3.0, "421") == []  # a callout, not a readback
