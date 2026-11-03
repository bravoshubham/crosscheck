from senses.speech_spark import guarded

OURS = {"for_us": True}


def kinds(facts):
    return [(f.key, f.value) for f in facts]


def test_a_mangled_readback_is_kept_when_the_transcript_holds_the_digits():
    # radio set, jet noise 0 dB: "flight level two four zero" came back as "white level two and four zero"
    text = "One, white level two and four zero Alaska four two one"
    assert kinds(guarded({**OURS, "kind": "readback", "value": 24000}, text, "pm", 0, "421")) == [("readback.altitude", 24000)]


def test_a_value_the_transcript_does_not_state_is_dropped():
    assert guarded({**OURS, "kind": "readback", "value": 25000}, "One, white level two and four zero Alaska four two one", "pm", 0, "421") == []


def test_another_aircrafts_clearance_is_not_ours():
    text = "Cessna three four five Charlie Papa maintain three thousand five hundred"
    assert guarded({**OURS, "kind": "clearance", "value": 3500}, text, "atc", 0, "421") == []


def test_a_bare_number_from_the_crew_is_not_a_flap_call():
    assert guarded({**OURS, "kind": "flaps", "value": 15}, "15.", "pm", 0, "421") == []
    assert kinds(guarded({**OURS, "kind": "flaps", "value": 15}, "vats 15.", "pm", 0, "421")) == [("callout.flaps", 15)]


def test_the_channel_decides_clearance_or_readback():
    text = "Alaska four two one descend and maintain one one thousand"
    assert guarded({**OURS, "kind": "readback", "value": 11000}, text, "atc", 0, "421") == []


def test_listen_again_when_the_first_hearing_is_impossible(monkeypatch, tmp_path):
    # Voice Transcribe dropped a "one" (3 Nov): heard again, the same audio usually comes back right
    import asyncio
    import senses.speech as speech
    from brain.ledger import Fact, Ledger
    from senses.speech_spark import hear
    heard = iter(["Alaska 421 climb and maintain 1000.", "Alaska 421 climb and maintain one one thousand."])

    async def fake(pcm, key=None):
        return {"text": next(heard), "audio_s": 4.0}
    monkeypatch.setattr(speech, "transcribe", fake)
    led = Ledger()
    for t in (1.0, 2.0, 3.0):
        led.add(Fact("doing", "altitude", 9000, t))
    r, facts, spark, first = asyncio.run(hear(None, "atc", "421", led))
    assert first == "Alaska 421 climb and maintain 1000."
    assert [(f.key, f.value) for f in facts] == [("clearance.altitude", 11000)]
