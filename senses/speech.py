"""Speech in: Muse Voice Transcribe (Meta Model API, realtime) -> transcript -> said facts.

Each audio channel is transcribed separately, so the channel says who spoke: the radio feed is
ATC, the intercom is the crew. Protocol (wss://api.meta.ai/v1/asr/realtime): a JSON handshake,
PCM16 audio in 80 ms frames at real-time pace, then {"type":"endStream"}; the server answers
with partial and final "transcript" events.

Clearance values are parsed from the transcript by code, not a model, so every value is one
the transcript states; a line without our callsign is ignored. Own-ship is matched on
the callsign's digits, because the recognizer often mishears the word ("Alaska" -> "Oscar",
"Alas, for") but keeps the digits.
"""
import asyncio
import json
import os
import re

import numpy as np
import websockets
from dotenv import load_dotenv

from brain.ledger import Fact
from brain.speech_guard import SCALE, WORD_DIGITS, spoken_values

URL = "wss://api.meta.ai/v1/asr/realtime"
MODEL = "muse-voice-transcribe-1.0"
SR = 16000
PACKET_S = 0.08
TEENS = {w: str(10 + i) for i, w in enumerate(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}


async def transcribe_stream(chunks, key: str | None = None) -> dict:
    """Transcribe audio as it arrives: `chunks` is an async iterator of 16 kHz mono int16 bytes
    (a live microphone, or a recording paced at real time). Returns the final text and billing."""
    if key is None:
        load_dotenv()
        key = os.environ["MODEL_API_KEY"]
    handshake = {"authorization": {"accessToken": key},
                 "audioEncoding": "PCM_16KHZ", "model": MODEL, "mode": "PUSH_TO_TALK",
                 "partialMode": "CUMULATIVE", "emitAudioProgress": True}
    finals, processed_ms = [], 0.0
    async with websockets.connect(URL, proxy=None) as ws:
        await ws.send(json.dumps(handshake))

        async def receive():
            nonlocal processed_ms
            async for m in ws:
                e = json.loads(m)
                processed_ms = max(processed_ms, e.get("audioProcessedMs", 0))
                if e.get("type") == "error":
                    raise RuntimeError(f"Voice Transcribe: {e}")
                if e.get("type") == "transcript" and e.get("final"):
                    finals.append(e["transcript"])

        rx = asyncio.create_task(receive())
        async for c in chunks:
            await ws.send(c)
        await ws.send('{"type":"endStream"}')
        await asyncio.wait_for(rx, 30)
    return {"text": " ".join(finals), "audio_s": processed_ms / 1000}


async def transcribe(pcm: np.ndarray, key: str | None = None) -> dict:
    """Stream float or int16 16 kHz mono audio as a live feed; return the final text and billing.
    On board (senses/onboard.py), the line goes to the local hearing service instead and the
    result also carries the second recognizer's transcript as "alt"."""
    from senses.onboard import ONBOARD
    if ONBOARD:
        return await transcribe_onboard(pcm)
    if pcm.dtype != np.int16:
        pcm = (np.clip(pcm, -1, 1) * 32767).astype("<i2")
    step = int(SR * PACKET_S)

    async def paced():
        for i in range(0, len(pcm), step):
            yield pcm[i:i + step].tobytes()
            await asyncio.sleep(PACKET_S)
    return await transcribe_stream(paced(), key)


async def transcribe_onboard(pcm: np.ndarray) -> dict:
    """Both on-board recognizers on one line (onboard/hearing_service.py on 127.0.0.1). "text" is
    SeamlessM4T v2's transcript, "alt" Omnilingual ASR's; senses/speech_spark.py decides which
    values to keep."""
    from senses.onboard import HEARING, local_client
    x = (pcm.astype(np.float32) / 32768.0) if pcm.dtype == np.int16 else pcm.astype(np.float32)
    async with local_client(60) as c:
        r = await c.post(HEARING + "/transcribe", content=x.astype("<f4").tobytes())
        r.raise_for_status()
    d = r.json()
    return {"text": d["seamless"].strip(), "alt": d["omni"].strip(), "audio_s": d["audio_s"],
            "engine": "onboard", "seconds": d["seconds"]}


def normalise(text: str) -> str:
    """Lowercase, no punctuation, and every spoken number as one value, so "two seven zero",
    "270" and "two seventy" style differences don't count as errors: "one one thousand" ->
    11000, "fifteen" -> 15, "124.35" -> 124 point 35."""
    t = text.lower().replace("-", " ")
    t = re.sub(r"\bfl\s*(?=\d)", "flight level ", t)
    t = re.sub(r"(?<=\d)[.,](?=\d)", lambda m: " point " if m.group() == "." else "", t)
    words = re.findall(r"[a-z]+|\d+", t) + [""]
    out, run = [], []
    for i, w in enumerate(words):
        if w in SOUNDALIKES:
            # "to" / "for" are digits only inside a number, or ("for", "won") right before one:
            # "speed to two one zero" stays "to 210"; "Alaska for 21" and "for two one" become 421;
            # "maintain won one thousand" (Omnilingual ASR, on board) is one one thousand
            nxt = words[i + 1]
            digit = bool(run) or (w in ("for", "fower", "won") and (nxt.isdigit() or nxt in WORD_DIGITS)
                                  and nxt not in SOUNDALIKES)
        else:
            digit = w.isdigit() or w in WORD_DIGITS or w in TEENS or (w in SCALE and bool(run))
        if digit:
            run.append(TEENS.get(w, w))
            continue
        if run:
            out.append(str(max(spoken_values(" ".join(run)))))
            run = []
        if w:
            out.append(w)
    return " ".join(out)


def numbers(text: str) -> list[str]:
    return [w for w in normalise(text).split() if w.isdigit()]


SOUNDALIKES = {"to", "too", "for", "fower", "won", "oh"}
ALT_VERBS = {"maintain", "maintained", "climb", "climbed", "descend", "descended", "to"}


def parse_altitude(text: str) -> int | None:
    """Altitude instructed in a clearance or readback: "climb and maintain one one thousand"
    -> 11000, "flight level two four zero" -> 24000. None if the line states no altitude, or
    more than one candidate without a verb to choose between them."""
    w = normalise(text).split()
    for i, tok in enumerate(w):
        if not tok.isdigit():
            continue
        v = int(tok)
        if i >= 2 and w[i - 2:i] == ["flight", "level"] and 10 <= v <= 600:
            return v * 100
        if i >= 1 and w[i - 1] in ALT_VERBS and 1000 <= v <= 45000 and v % 100 == 0:
            return v
    # verb misheard or dropped ("plan and mention one 2000"): accept the line's only
    # altitude-shaped number; speeds, headings, squawks and frequencies are not altitude-shaped
    shaped = {int(t) for t in w if t.isdigit() and 1000 <= int(t) <= 45000 and int(t) % 100 == 0}
    return shaped.pop() if len(shaped) == 1 else None


FLAP_DETENTS = {0, 1, 2, 5, 10, 15, 25, 30, 40}


def parse_flaps(text: str) -> int | None:
    """A flap callout: "flaps fifteen" / "Flaps 15." / "flabs 15" -> 15, "flaps up" -> 0."""
    # "flabs 15": how Voice Transcribe heard "flaps fifteen" through a laptop speaker and microphone
    m = re.search(r"\bfla[pb]s?\s+(up|\d+)\b", normalise(text))
    if not m:
        return None
    v = 0 if m.group(1) == "up" else int(m.group(1))
    return v if v in FLAP_DETENTS else None


def said_facts(text: str, speaker: str, t: float, own_digits: str, source: str = "muse-voice-transcribe") -> list[Fact]:
    """Facts for the ledger from one final transcript on one channel."""
    if speaker != "atc":
        flaps = parse_flaps(text)
        if flaps is not None:  # an intercom callout: our own crew, no callsign
            return [Fact("said", "callout.flaps", flaps, t, speaker, source=source)]
    alt = parse_altitude(text)
    if alt is None:
        return []
    if own_digits not in numbers(text) and not re.search(rf"\b{own_digits}\b", text):
        # (the raw transcript too: in "descend and maintain 4,000 LAS for 421" the "for" joins the
        # callsign into "4421" after normalising)
        # a clearance names us and a readback ends with our callsign; the digits survive even
        # when the word does not. This also keeps callouts ("one thousand stabilized") out.
        return []
    key = "clearance.altitude" if speaker == "atc" else "readback.altitude"
    return [Fact("said", key, alt, t, speaker, source=source)]
