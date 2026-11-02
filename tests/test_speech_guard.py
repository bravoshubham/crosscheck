from brain.speech_guard import guard, spoken_digits, spoken_values

FIELDS = ["altitude_ft", "heading_deg", "speed_kt", "frequency_mhz", "squawk"]


def test_spoken_digits():
    assert spoken_digits("descend and maintain one one thousand") == "11000"
    assert spoken_digits("flight level two four zero") == "240"
    assert spoken_digits("one two four point three five") == "12435"
    assert spoken_digits("niner thousand five hundred") == "9000500"


def test_supported_values_are_kept():
    x = {"callsign": "alaska four two one", "altitude_ft": 24000, "heading_deg": None,
         "speed_kt": None, "frequency_mhz": None, "squawk": None}
    kept, dropped = guard(x, "alaska four two one climb and maintain flight level two four zero", FIELDS)
    assert kept["altitude_ft"] == 24000 and dropped == []
    x2 = dict(x, altitude_ft=None, frequency_mhz=124.35)
    kept, dropped = guard(x2, "contact portland approach one two four point three five", FIELDS)
    assert kept["frequency_mhz"] == 124.35 and "frequency_mhz" not in dropped


def test_copied_example_is_dropped():
    """The T3 failure: a worked example's values on a line that said none of them."""
    x = {"callsign": "delta four", "altitude_ft": None, "heading_deg": 310, "speed_kt": None,
         "frequency_mhz": 132.5, "squawk": None}
    kept, dropped = guard(x, "i'm ask afor two one term lap hitting two seven serv", FIELDS)
    assert kept["heading_deg"] is None and kept["frequency_mhz"] is None and kept["callsign"] is None
    assert set(dropped) == {"callsign", "heading_deg", "frequency_mhz"}


def test_compound_numbers():
    assert 3500 in spoken_values("maintain three thousand five hundred")
    assert 11000 in spoken_values("descend and maintain one one thousand")
    x = {"callsign": "cessna three four five charlie papa", "altitude_ft": 3500, "heading_deg": None,
         "speed_kt": None, "frequency_mhz": None, "squawk": None}
    kept, dropped = guard(x, "cessna three four five charlie papa maintain three thousand five hundred", FIELDS)
    assert kept["altitude_ft"] == 3500 and dropped == []
