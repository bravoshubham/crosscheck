"""Debrief summary: Muse Spark writes a few plain sentences about one recorded flight.

The model sees only a timeline built by code from the recording (events.jsonl and meta.json):
what was heard and the facts parsed from it, what the panel showed when it changed, the crew's
actions, the pilot camera, and every callout. It is asked to use nothing else, and every number
in its answer is checked against the timeline; a summary with a number the timeline does not
hold is retried, then kept but marked unverified, so the page can say so.

Summaries are cached in <recording>/summary.json, so the debrief needs no API key to show them.

  uv run python cli.py summary [replay/<name> ...]
"""
import asyncio
import json
import os
import pathlib
import re

import httpx
from dotenv import load_dotenv

MODEL = "muse-spark-1.3"
SAMPLE_S = 4.0  # speed and altitude at most this often; the other panel values when they change
STEADY = 3      # a changed panel value is reported once it is read this many times in a row
PROMPT = (
    "You write the debrief line an airline safety officer reads after a flight. Crosscheck is an AI that "
    "listens to the radio and intercom and watches the cockpit; it cross-checks what was said against what "
    "was set and what the aircraft did, and speaks up when they disagree.\n"
    "Using ONLY the timeline below, write 2 to 4 short sentences in plain English: what was cleared or "
    "called, what the crew set or did, what happened to the aircraft, and what Crosscheck said and when "
    "(\"at 14 s\"). If Crosscheck stayed silent, say so and why that was right or wrong. Do not use any "
    "number, name or event that is not in the timeline. Describe only what happened: never write that "
    "something is absent, missing or \"not listed\". Use set-up lines and the score only to explain, "
    "without the words \"set-up\", \"timeline\" or \"parsed\". No headings, no bullet points.\n\nTimeline:\n"
)


def _load(p: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def timeline(rec: pathlib.Path) -> str:
    meta = json.loads((rec / "meta.json").read_text())
    result = json.loads((rec / "result.json").read_text()) if (rec / "result.json").exists() else {}
    t0 = meta["t_start"]
    from debrief.build import TITLES
    base = meta["scenario"].removesuffix("_spark").removesuffix("_mic").removesuffix("_webcam")
    title, about = TITLES.get(base, (meta["scenario"].replace("_", " "), ""))
    where = "pilot camera only" if meta.get("pilot_camera") else (
        f"Boeing 737-800" + (f" near {meta['area']}" if meta.get("area") else "") + (f", {meta['airport']}" if meta.get("airport") else ""))
    lines = [f"Scenario: {title}. {about} ({where})"]
    last: dict[str, object] = {}
    seen: dict[str, tuple[object, int]] = {}  # field -> (candidate value, times read in a row)
    last_sample = -1e9

    def steady(f: str, v) -> bool:
        """True once v, different from the last reported value, has been read STEADY times in a row:
        one misread frame (an MCP window read as 1200) must not appear in the timeline."""
        cand, n = seen.get(f, (None, 0))
        n = n + 1 if v == cand else 1
        seen[f] = (v, n)
        if v != last.get(f) and n >= STEADY:
            last[f] = v
            return True
        return False

    for e in sorted(_load(rec / "events.jsonl"), key=lambda e: e["t"]):
        s = f"{e['t'] - t0:5.1f} s"
        k = e["kind"]
        if k == "heard":
            who = {"atc": "ATC on the radio", "pm": "pilot monitoring on the intercom", "pf": "pilot flying on the intercom",
                   "capt": "the captain"}.get(e["speaker"], e["speaker"])
            lines.append(f"{s}  heard from {who}: \"{e['transcript'].strip()}\"")
        elif k == "said":
            lines.append(f"{s}  parsed from speech: {e['key']} = {e['value']}")
        elif k == "action":
            lines.append(f"{s}  crew action: {e['line']}")
        elif k == "callout":
            lines.append(f"{s}  CROSSCHECK SAID: \"{e['callout']}\" (expected {e['expected']}, saw {e['actual']})")
        elif k == "cleared":
            lines.append(f"{s}  Crosscheck flag cleared: {e['rule']}")
        elif k == "pilot":
            state = "slumped" if e["slumped"] else "upright"
            if last.get("pilot") != state:
                lines.append(f"{s}  pilot camera: captain {state}" + (f" ({', '.join(e['reasons'])})" if e.get("reasons") else ""))
                last["pilot"] = state
        elif k == "reading":
            changes = []
            for f, label in (("selected_altitude", "MCP altitude"), ("selected_speed", "MCP speed"), ("flaps", "flap gauge")):
                if e.get(f) is not None and steady(f, e[f]):
                    changes.append(f"{label} {e[f]}")
            if e.get("fma") and any(e["fma"]):
                thr = e["fma"][0] if e["fma"][0] != "BLANK" else ""
                if steady("thr", thr):
                    changes.append(f"autothrottle mode {thr or 'blank'}")
            if e["t"] - last_sample >= SAMPLE_S:
                changes += [f"{label} {e[f]}" for f, label in (("ias", "speed"), ("altitude", "altitude")) if e.get(f) is not None]
                last_sample = e["t"]
            if changes:
                lines.append(f"{s}  panel: " + ", ".join(changes))
    for k, label in (("t_clear", "ATC clearance sent"), ("t_set", "MCP changed by the crew"), ("t_fault", "autothrottle fault injected"),
                     ("t_call", "callout spoken"), ("t_change", "pilot camera scene changes")):
        if k in meta:
            lines.append(f"(set-up: {label} at {meta[k] - t0:.1f} s)")
    if "pass" in result:
        lines.append(f"Scored against the recorded ground truth: {'pass' if result['pass'] else 'fail'}")
    return "\n".join(lines)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".").lstrip("0") or "0" for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def unsupported(summary: str, tl: str) -> list[str]:
    """Numbers in the summary that the timeline does not contain (times are matched to 1 s)."""
    have = _numbers(tl)
    have |= {str(round(float(n))) for n in have if re.fullmatch(r"\d+(\.\d+)?", n)}
    return sorted(n for n in _numbers(summary) if n not in have)


async def summarise(rec: pathlib.Path, client: httpx.AsyncClient) -> dict:
    tl = timeline(rec)
    body = {"model": MODEL, "max_tokens": 1500, "reasoning_effort": "minimal",
            "messages": [{"role": "user", "content": PROMPT + tl}]}
    url = os.environ.get("META_API_BASE", "https://api.meta.ai/v1") + "/chat/completions"
    out = {}
    for attempt in range(2):
        r = await client.post(url, json=body, headers={"Authorization": "Bearer " + os.environ["MODEL_API_KEY"]})
        r.raise_for_status()
        d = r.json()
        text = (d["choices"][0]["message"].get("content") or "").strip()
        bad = unsupported(text, tl)
        out = {"text": text, "verified": not bad, "unsupported_numbers": bad, "model": MODEL,
               "usage": d.get("usage", {}), "attempts": attempt + 1}
        if not bad:
            break
    return out


def run(recs: list[pathlib.Path], force: bool = False) -> None:
    load_dotenv(pathlib.Path(__file__).resolve().parents[1] / ".env")

    async def go():
        async with httpx.AsyncClient(trust_env=False, timeout=90) as client:
            async def one(rec):
                p = rec / "summary.json"
                if p.exists() and not force and json.loads(p.read_text()).get("timeline") == timeline(rec):
                    return rec.name, json.loads(p.read_text())
                s = {**await summarise(rec, client), "timeline": timeline(rec)}
                p.write_text(json.dumps(s, indent=1))
                return rec.name, s
            for name, s in await asyncio.gather(*(one(r) for r in recs)):
                print(f"{name} [{'verified' if s['verified'] else 'UNVERIFIED ' + str(s['unsupported_numbers'])}]\n  {s['text']}\n")
    asyncio.run(go())
