from brain.summary import unsupported

TIMELINE = """ 18.2 s  heard (atc): "Alaska 421 climb and maintain one one thousand."
 18.2 s  parsed from speech: clearance.altitude = 11000
 35.0 s  CROSSCHECK SAID: "Cleared one-one thousand, set one-two thousand." (expected 11000, saw 12000)"""


def test_numbers_in_the_timeline_are_supported():
    assert unsupported("ATC cleared Alaska 421 to 11,000; the crew set 12000. Crosscheck spoke at 35 s, 18.2 s after.", TIMELINE) == []


def test_a_number_not_in_the_timeline_is_flagged():
    assert unsupported("The aircraft climbed to 13,000 ft.", TIMELINE) == ["13000"]
