from senses.speech import normalise, parse_altitude, parse_flaps, said_facts


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


def test_flap_callouts():
    # Voice Transcribe outputs for the Mac's "Samantha" voice
    assert parse_flaps("Flaps 15.") == 15
    assert parse_flaps("Flaps five.") == 5
    assert parse_flaps("flaps up") == 0
    assert parse_flaps("Flaps 12") is None  # not a detent
    assert parse_flaps("Pledge five.") is None
    f = said_facts("Flaps 15.", "pf", 1.0, "421")
    assert [(x.key, x.value, x.speaker) for x in f] == [("callout.flaps", 15, "pf")]


def test_callsign_joined_by_a_misheard_for_still_counts():
    # "for" before the callsign is read as a 4: "LAS for 421" -> 4421 after normalising
    assert [(f.key, f.value) for f in said_facts("descend and maintain 4,000 LAS for 421.", "pm", 0, "421")] == [
        ("readback.altitude", 4000)]


def test_altimeter_setting_is_parsed():
    from senses.speech import parse_altimeter, said_facts
    assert parse_altimeter("Alaska 421, Portland altimeter three zero one two.") == 3012
    assert parse_altimeter("altimeter 30.12") == 3012
    assert parse_altimeter("descend and maintain six thousand, altimeter two niner niner two") == 2992
    assert parse_altimeter("altimeter 4000") is None
    assert parse_altimeter("climb and maintain one one thousand") is None
    facts = said_facts("Alaska 421 descend and maintain 6000 Portland altimeter 3012", "atc", 0.0, "421")
    assert [(f.key, f.value) for f in facts] == [("clearance.altitude", 6000), ("clearance.altimeter", 3012)]
    assert said_facts("Portland altimeter 3012", "atc", 0.0, "421") == []  # not our callsign
    # as Voice Transcribe heard the scenario's lines (eval/scenarios/altimeter.py, 5 Nov)
    assert parse_altimeter("Alaska 421 descend and maintain 7000 Portland altitude 3012") == 3012
    assert parse_altimeter("descend and maintained seven thousand autometer three zero one two Alaska four two one") == 3012
    assert parse_altimeter("climb to altitude 3000") is None  # a whole hundred is an altitude
