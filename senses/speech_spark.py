"""Muse Spark as the second reader of a radio or intercom line, for the lines code cannot parse.

senses/speech.py parses transcripts with fixed rules, which never invent a value but give up on
a mangled line ("flight level two four zero" heard as "white level 240", a verb lost to noise).
Only then is the transcript sent to Muse Spark, which says whether the line assigns or reads back
an altitude for our aircraft, or calls a flap setting. Its answer is kept only if the guard
agrees (brain/speech_guard.py's rule, the reason the guard exists): the value must be one the
transcript itself states, and an altitude must carry our callsign's digits or the word of our
callsign, so a clearance for another aircraft cannot be taken as ours. Measured in
eval/audio/speech_fallback.py.
"""
import json
import os
import re

import httpx
from dotenv import load_dotenv

from brain.ledger import Fact
from brain.speech_guard import spoken_digits, spoken_values
from senses.speech import FLAP_DETENTS, numbers, said_facts

MODEL = "muse-spark-1.3"
PROMPT = (
    "This is a speech-recognition transcript of one line of aviation radio or intercom; it may contain "
    "recognition errors. Our aircraft is Alaska 421 (\"Alaska four two one\"). The line came from {who}.\n"
    "Transcript: \"{text}\"\n"
    "Reply with JSON only: {{\"kind\": \"clearance\" | \"readback\" | \"flaps\" | \"none\", \"value\": <number or null>, "
    "\"for_us\": true | false}}. kind is clearance when ATC assigns Alaska 421 an altitude, readback when our crew "
    "reads an altitude back, flaps when our crew calls a flap setting; value is the altitude in feet (flight level "
    "240 = 24000) or the flap setting. Use none when the line does neither, or when it is for another aircraft."
)
WHO = {"atc": "ATC, on the radio", "pm": "our crew, on the intercom", "pf": "our crew, on the intercom", "crew": "our crew, on the intercom"}
CALLSIGN_WORDS = re.compile(r"\balaska\b|\balaskan\b", re.I)
# a flap value needs a flap-like word: a bare "15." from the crew could be anything
FLAP_WORDS = re.compile(r"\b(fla\w*|vat\w*|laps?)\b", re.I)


def worth_asking(text: str) -> bool:
    """Only lines that might hold a value: a number, or a word for one."""
    return bool(numbers(text)) or bool(re.search(r"\b(thousand|hundred|level|flaps?)\b", text, re.I))


def supported(value: int, text: str) -> bool:
    """The guard: the transcript itself must state the value, as a number or as its digits in order
    ("white level two and four zero" holds 2, 4, 0 after "level": flight level 240)."""
    said = {int(n) for n in numbers(text)} | spoken_values(text)
    if value in said or str(value) in spoken_digits(text):
        return True
    level = re.search(r"\blevel\b(.*)", text, re.I)
    return value % 100 == 0 and level is not None and (value // 100 in said or str(value // 100) in spoken_digits(level.group(1)))


def guarded(answer: dict, text: str, speaker: str, t: float, own_digits: str) -> list[Fact]:
    kind, value = answer.get("kind"), answer.get("value")
    if kind not in ("clearance", "readback", "flaps") or not isinstance(value, (int, float)) or not answer.get("for_us"):
        return []
    value = int(value)
    if not supported(value, text):
        return []
    if kind == "flaps":
        if speaker == "atc" or value not in FLAP_DETENTS or not FLAP_WORDS.search(text):
            return []
        return [Fact("said", "callout.flaps", value, t, speaker, source="muse-spark")]
    if not (1000 <= value <= 45000 and value % 100 == 0):
        return []
    if (kind == "clearance") != (speaker == "atc"):
        return []
    # ours only if the line names us: the callsign digits, or the word of the callsign
    if own_digits not in numbers(text) and not re.search(rf"\b{own_digits}\b", text) and not CALLSIGN_WORDS.search(text):
        return []
    return [Fact("said", "clearance.altitude" if kind == "clearance" else "readback.altitude", value, t, speaker, source="muse-spark")]


ANSWER = {"type": "object", "properties": {"kind": {"type": "string", "enum": ["clearance", "readback", "flaps", "none"]},
          "value": {"type": ["integer", "null"]}, "for_us": {"type": "boolean"}}, "required": ["kind", "value", "for_us"]}


async def glimmer_facts(texts: list[str], speaker: str, t: float, own_digits: str) -> tuple[list[Fact], dict]:
    """On board: the same question to the local Muse Glimmer, with both recognizers' transcripts.
    The guard checks the value against each transcript, and keeps it if either states it."""
    from senses.onboard import glimmer_json
    heard = " / ".join(f'"{x}"' for x in texts if x)
    prompt = PROMPT.format(who=WHO.get(speaker, speaker), text=heard.replace('"', "'")).replace(
        "This is a speech-recognition transcript", "These are two speech-recognition transcripts (separated by /)").replace(
        "it may contain recognition errors", "each may contain recognition errors")
    out, _ = await glimmer_json(prompt, ANSWER, num_predict=60)
    m = re.search(r"\{.*\}", out, re.S)
    try:
        answer = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        answer = {}
    for x in texts:
        facts = guarded(answer, x, speaker, t, own_digits)
        if facts:
            return [Fact(f.stream, f.key, f.value, f.t, f.speaker, source="muse-glimmer") for f in facts], answer
    return [], answer


async def spark_facts(text: str, speaker: str, t: float, own_digits: str, client: httpx.AsyncClient | None = None) -> tuple[list[Fact], dict]:
    """Ask Muse Spark about one line; returns (guarded facts, its raw answer)."""
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    body = {"model": MODEL, "max_tokens": 800, "reasoning_effort": "minimal",
            "messages": [{"role": "user", "content": PROMPT.format(who=WHO.get(speaker, speaker), text=text.replace('"', "'"))}]}
    own = client is None
    client = client or httpx.AsyncClient(trust_env=False, timeout=60)
    try:
        r = await client.post(os.environ.get("META_API_BASE", "https://api.meta.ai/v1") + "/chat/completions", json=body,
                              headers={"Authorization": "Bearer " + os.environ["MODEL_API_KEY"]})
        r.raise_for_status()
    finally:
        if own:
            await client.aclose()
    out = r.json()["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", out, re.S)
    try:
        answer = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        answer = {}
    return guarded(answer, text, speaker, t, own_digits), answer


def onboard_facts(text: str, alt: str, speaker: str, t: float, own_digits: str, led=None) -> list[Fact]:
    """On board, two recognizers heard the line. A value that contradicts the aircraft (a climb to
    below it: brain/live.py's implausible) is dropped first; then a value counts when both heard
    it, or when only one heard a value at all. When they heard different values, neither is kept
    (scenario lines: eval/audio/wer_local.py)."""
    from brain.live import implausible
    a = said_facts(text, speaker, t, own_digits, source="seamless-m4t-v2")
    b = said_facts(alt, speaker, t, own_digits, source="omnilingual-asr")
    if led is not None:
        a = [f for f in a if not implausible(f, led, text)]
        b = [f for f in b if not implausible(f, led, alt)]
    # value by value: a line can carry two ("descend and maintain seven thousand, altimeter three zero
    # one two"), and one recognizer missing one of them must not cost the other
    ka, kb = {f.key: f for f in a}, {f.key: f for f in b}
    out = []
    for k in list(dict.fromkeys(list(ka) + list(kb))):
        fa, fb = ka.get(k), kb.get(k)
        if fa and fb:
            if fa.value == fb.value:
                out.append(Fact(fa.stream, fa.key, fa.value, fa.t, fa.speaker, source="seamless+omnilingual"))
        else:
            out.append(fa or fb)
    return out


async def read_line(text: str, speaker: str, own_digits: str, client: httpx.AsyncClient | None = None,
                    alt: str | None = None, led=None) -> tuple[list[Fact], dict | None]:
    """The live path: code first; Muse Spark only for a line code got nothing from. Facts are
    timed when they are known, so Spark's extra second is part of the record. On board, `alt` is
    the second recognizer's transcript and the second reader is the local Muse Glimmer."""
    import time
    if alt is not None:
        a = said_facts(text, speaker, time.time(), own_digits)
        b = said_facts(alt, speaker, time.time(), own_digits)
        facts = onboard_facts(text, alt, speaker, time.time(), own_digits, led)
        if facts or (a and b) or not (worth_asking(text) or worth_asking(alt)):
            return facts, None if facts or not (a and b) else {"disagree": [[f.key, f.value] for f in a + b]}
        try:
            facts, answer = await glimmer_facts([text, alt], speaker, 0.0, own_digits)
        except (httpx.HTTPError, KeyError) as e:
            return [], {"error": str(e)[:200]}
        t = time.time()
        return [Fact(f.stream, f.key, f.value, t, f.speaker, source=f.source) for f in facts], answer
    facts = said_facts(text, speaker, time.time(), own_digits)
    if facts or not worth_asking(text):
        return facts, None
    from senses.onboard import ONBOARD
    try:  # on board even a single transcript (a microphone line) goes to the local Glimmer, never the API
        facts, answer = await (glimmer_facts([text], speaker, 0.0, own_digits) if ONBOARD
                               else spark_facts(text, speaker, 0.0, own_digits, client))
    except (httpx.HTTPError, KeyError) as e:  # no answer: the line stays unread, as with code alone
        return [], {"error": str(e)[:200]}
    t = time.time()
    return [Fact(f.stream, f.key, f.value, t, f.speaker, source=f.source) for f in facts], answer


async def hear(pcm, speaker: str, own_digits: str, led=None) -> tuple[dict, list[Fact], dict | None, str | None]:
    """One line, heard: Voice Transcribe, then code and the guarded Spark reader. If what was heard
    contradicts the aircraft (brain/live.py's implausible: a climb to below it), listen again: the
    same audio goes to Voice Transcribe once more, which often hears it right the second time.
    Returns (transcript result, facts, Spark's answer, the first transcript if it was heard again)."""
    from brain.live import implausible
    from senses.speech import transcribe
    r = await transcribe(pcm)
    facts, spark = await read_line(r["text"], speaker, own_digits, alt=r.get("alt"), led=led)
    if led is None or not any(implausible(f, led, r["text"]) for f in facts):
        return r, facts, spark, None
    if "alt" in r:  # on board the recognizers are deterministic: hearing it again changes nothing
        return r, facts, spark, r["text"]
    first = r["text"]
    r2 = await transcribe(pcm)
    facts2, spark2 = await read_line(r2["text"], speaker, own_digits)
    if facts2 and not any(implausible(f, led, r2["text"]) for f in facts2):
        return r2, facts2, spark2, first
    return r, facts, spark, first  # heard wrong twice: the guard rejects it and Crosscheck asks
