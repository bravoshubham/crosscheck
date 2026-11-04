"""Crosscheck's callout voice: a fixed bank of recorded phrases, the way an aircraft's aural alerts
are recorded once and played, never synthesised in flight.

Every callout the rules make (brain/rules.py) is either a whole phrase in the bank, or is put
together from the bank's words ("Cleared" + "one" + "one" + "thousand" + ...). Anything else falls
back to MMS-TTS. The bank is recorded once with voice/render_bank.py and chosen by how well a
recognizer hears it (MMS-TTS's own callouts came back as "Fleered one 1000", "flats 15 cult").
Playing it needs no model and no network.
"""
import json
import pathlib
import re

import numpy as np
import soundfile as sf

BANK = pathlib.Path(__file__).resolve().parent / "bank"
SR = 24000
GAP_S, COMMA_S, STOP_S = 0.04, 0.18, 0.32  # between words, at a comma, at a full stop

# Whole phrases (the shipped scenarios' callouts and the fixed ones), recorded as sentences
PHRASES = ["Cleared one-one thousand, set one-two thousand.", "Readback one-two thousand, cleared one-one thousand.",
           "Flaps fifteen called, ten set.", "Flaps fifteen called, five set.", "Speed. Autothrottle not in speed.",
           "Captain, are you OK?", "Captain not responding. First officer, take control.", "Confirm cleared altitude.",
           "Altimeter three-zero-one-two given, three-zero-two-one set.", "Altimeter three-zero-one-two given, two-niner-niner-two set."]
# Words for every other altitude and flap callout (brain/phraseology.py's vocabulary)
WORDS = ["Cleared", "cleared", "set", "Readback", "Flaps", "called", "thousand", "hundred",
         "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "niner",
         "up", "ten", "fifteen", "twenty-five", "thirty", "forty", "Altimeter", "given"]
_cache: dict | None = None


def key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _load() -> dict:
    global _cache
    if _cache is None:
        index = BANK / "bank.json"
        _cache = {}
        if index.exists():
            for text, f in json.loads(index.read_text()).items():
                x, sr = sf.read(BANK / f, dtype="float32")
                assert sr == SR, (f, sr)
                _cache[text] = x
    return _cache


def audio(text: str) -> np.ndarray | None:
    """The callout from the bank (24 kHz), or None if the bank cannot say it."""
    bank = _load()
    if text in bank:
        return bank[text]
    out = []
    for tok in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*|[,.?]", text):
        if tok in ",.?":
            out.append(np.zeros(int(SR * (COMMA_S if tok == "," else STOP_S)), np.float32))
            continue
        parts = [tok] if tok in bank else tok.split("-")  # "one-one" is said as two words
        for p in parts:
            clip = bank.get(p) if p in bank else bank.get(p.lower())
            if clip is None:
                return None
            out += [clip, np.zeros(int(SR * GAP_S), np.float32)]
    return np.concatenate(out) if out else None
