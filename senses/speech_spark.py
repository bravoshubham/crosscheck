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


async def read_line(text: str, speaker: str, own_digits: str, client: httpx.AsyncClient | None = None) -> tuple[list[Fact], dict | None]:
    """The live path: code first; Muse Spark only for a line code got nothing from. Facts are
    timed when they are known, so Spark's extra second is part of the record."""
    import time
    facts = said_facts(text, speaker, time.time(), own_digits)
    if facts or not worth_asking(text):
        return facts, None
    try:
        facts, answer = await spark_facts(text, speaker, 0.0, own_digits, client)
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
    facts, spark = await read_line(r["text"], speaker, own_digits)
    if led is None or not any(implausible(f, led, r["text"]) for f in facts):
        return r, facts, spark, None
    first = r["text"]
    r2 = await transcribe(pcm)
    facts2, spark2 = await read_line(r2["text"], speaker, own_digits)
    if facts2 and not any(implausible(f, led, r2["text"]) for f in facts2):
        return r2, facts2, spark2, first
    return r, facts, spark, first  # heard wrong twice: the guard rejects it and Crosscheck asks
