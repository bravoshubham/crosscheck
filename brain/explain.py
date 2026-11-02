"""One sentence from Muse Spark under each callout, for the tablet: why Crosscheck spoke.

The callout itself stays fixed phraseology from brain/rules.py and is spoken at once. In the
background, Muse Spark gets the flag and the ledger's last 45 seconds (what was said, set and
done, with times) and writes one plain sentence for the crew. Every number in the sentence must
appear in what it was given (brain/summary.py's check); otherwise the sentence is not shown.
"""
import atexit
import os
import threading
import time

import httpx
from dotenv import load_dotenv

from brain.ledger import Ledger
from brain.rules import Flag
from brain.summary import unsupported

MODEL = "muse-spark-1.3"
WINDOW_S = 45.0
STEP = {"altitude": 100, "ias": 3}  # smaller changes are left out of the facts
KEEP = ("clearance.altitude", "readback.altitude", "callout.flaps", "pilot.voice", "mcp.altitude", "mcp.speed",
        "altitude", "ias", "fma.thrust", "fma.pitch", "flaps", "pilot.slumped")
PROMPT = (
    "You are Crosscheck, an AI monitoring pilot on a Boeing 737 flight deck. You just made the standard callout "
    "below. Write ONE short sentence (under 30 words) for the crew's tablet saying what you saw that made you speak, "
    "using only the facts given, with numbers as a pilot would say them (11,000 ft, 140 kt, flaps 15). No advice "
    "beyond what the facts show, no greeting.\n\nCallout: \"{callout}\" (rule {rule}: expected {expected}, found {actual})\n"
    "Facts, newest last (seconds before now):\n{facts}"
)


def context(led: Ledger, flag: Flag, now: float) -> str:
    rows, last = [], {}
    for f in led.facts:
        if f.t < now - WINDOW_S or f.t > now or f.key not in KEEP:
            continue
        prev = last.get(f.key)
        if f.stream != "said" and (prev == f.value or (f.key in STEP and isinstance(prev, (int, float))
                                                        and isinstance(f.value, (int, float)) and abs(f.value - prev) < STEP[f.key])):
            continue  # panel values only when they move
        last[f.key] = f.value
        who = f" ({f.speaker})" if f.speaker else ""
        rows.append(f"-{now - f.t:4.0f} s  {f.stream} {f.key}{who} = {f.value}")
    return PROMPT.format(callout=flag.callout, rule=flag.rule, expected=flag.expected, actual=flag.actual,
                         facts="\n".join(rows[-40:]) or "(none)")


def explain(prompt: str) -> dict:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    body = {"model": MODEL, "max_tokens": 600, "reasoning_effort": "minimal", "messages": [{"role": "user", "content": prompt}]}
    t = time.time()
    r = httpx.post(os.environ.get("META_API_BASE", "https://api.meta.ai/v1") + "/chat/completions", json=body, timeout=30,
                   headers={"Authorization": "Bearer " + os.environ["MODEL_API_KEY"]}, trust_env=False)
    r.raise_for_status()
    text = (r.json()["choices"][0]["message"].get("content") or "").strip().strip('"')
    bad = unsupported(text, prompt)
    return {"text": text if not bad else None, "unsupported_numbers": bad, "s": round(time.time() - t, 2), "model": MODEL}


def explain_later(led: Ledger, flag: Flag, now: float, emit) -> None:
    """Build the prompt now (the ledger keeps changing), ask in a background thread, and emit(result)
    when the answer comes. Never blocks the monitoring loop."""
    prompt = context(led, flag, now)

    def run():
        try:
            emit(explain(prompt))
        except (httpx.HTTPError, KeyError) as e:
            emit({"text": None, "error": str(e)[:200]})
    th = threading.Thread(target=run, daemon=True)
    th.start()
    _pending.append(th)


_pending: list[threading.Thread] = []


@atexit.register
def _finish() -> None:
    """A scenario that ends right after a callout still gets its explanation written."""
    for th in _pending:
        th.join(timeout=15)
