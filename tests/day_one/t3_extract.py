"""T3: extract clearances from radio text into ledger facts, with Muse Glimmer on the Mac.

The spec names Muse Spark for this; Glimmer runs it here so development costs no API credits,
and the same prompt moves to Spark later. Scored on the exact text of the T2 radio lines and on
the noisy MMS ASR transcripts (eval/results/day_one/t2_wer_rows.json).

Run with Ollama up:  uv run python tests/day_one/t3_extract.py
"""
import json
import pathlib
import re
import sys
import time

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = "muse-glimmer:30b"
FIELDS = ["altitude_ft", "heading_deg", "speed_kt", "frequency_mhz", "squawk"]
SCHEMA = {"type": "object", "properties": {
    "callsign": {"type": "string"},
    "altitude_ft": {"type": ["integer", "null"]}, "heading_deg": {"type": ["integer", "null"]},
    "speed_kt": {"type": ["integer", "null"]}, "frequency_mhz": {"type": ["number", "null"]},
    "squawk": {"type": ["string", "null"]}},
    "required": ["callsign"] + FIELDS}
SYSTEM = ("You extract air traffic control instructions from a radio transcript into JSON. The transcript "
          "may contain speech-recognition spelling errors; read it as a pilot would. Flight levels are hundreds "
          "of feet (flight level two four zero = 24000). Use null for anything not instructed. Reply with JSON only.")
SHOTS = [
    ("united six one two descend and maintain flight level one niner zero",
     {"callsign": "united six one two", "altitude_ft": 19000, "heading_deg": None, "speed_kt": None, "frequency_mhz": None, "squawk": None}),
    ("delta four fly heding three one zero contact seatle center one tree two point five",
     {"callsign": "delta four", "altitude_ft": None, "heading_deg": 310, "speed_kt": None, "frequency_mhz": 132.5, "squawk": None}),
]
TRUTH = {  # expected values for each T2 line that carries an instruction
    "alaska four two one climb and maintain flight level two four zero": {"altitude_ft": 24000},
    "alaska four two one turn left heading two seven zero": {"heading_deg": 270},
    "alaska four two one descend and maintain one one thousand": {"altitude_ft": 11000},
    "alaska four two one contact portland approach one two four point three five": {"frequency_mhz": 124.35},
    "alaska four two one reduce speed to two one zero knots": {"speed_kt": 210},
    "cessna three four five charlie papa squawk four three two one": {"squawk": "4321"},
    "cessna three four five charlie papa maintain three thousand five hundred": {"altitude_ft": 3500},
}


def extract(http: httpx.Client, text: str) -> tuple[dict, float]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for u, a in SHOTS:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": json.dumps(a)}]
    msgs.append({"role": "user", "content": text})
    t = time.monotonic()
    r = http.post("http://127.0.0.1:11434/api/chat", json={"model": MODEL, "messages": msgs, "format": SCHEMA,
                  "think": False, "stream": False, "keep_alive": "30m", "options": {"temperature": 0}}).json()
    raw = re.sub(r"<\|[a-z_]+\|>\s*$", "", r["message"]["content"].strip())
    return json.loads(raw), time.monotonic() - t


def correct(got: dict, want: dict) -> bool:
    for f in FIELDS:
        w, g = want.get(f), got.get(f)
        if f == "frequency_mhz" and w is not None:
            if g is None or abs(float(g) - w) > 0.001:
                return False
        elif f == "squawk" and w is not None:
            if str(g or "").replace(" ", "") != w:
                return False
        elif g != w:
            return False
    return True


def main() -> int:
    rows = json.loads((ROOT / "eval" / "results" / "day_one" / "t2_wer_rows.json").read_text())
    cases = [("exact", ref, ref) for ref in TRUTH]
    for r in rows:
        if r["ref"] in TRUTH:
            snr = r["file"].split("_snr")[1].split(".")[0]
            cases.append((f"mms snr{snr}", r["ref"], r["raw"]))
    out, score, times = [], {}, []
    with httpx.Client(timeout=120) as http:
        for group, ref, text in cases:
            got, dt = extract(http, text)
            ok = correct(got, TRUTH[ref])
            times.append(dt)
            score.setdefault(group, [0, 0])
            score[group][0] += ok
            score[group][1] += 1
            out.append({"group": group, "ref": ref, "input": text, "got": got, "ok": ok, "s": round(dt, 2)})
            print(f"{group:10s} {'OK ' if ok else 'BAD'} {dt:4.1f}s  {text[:60]!r} -> {({k: v for k, v in got.items() if v is not None})}")
    summary = {"model": MODEL, "correct": {k: f"{a}/{b}" for k, (a, b) in score.items()},
               "median_s": round(sorted(times)[len(times) // 2], 2)}
    res = ROOT / "eval" / "results" / "day_one"
    (res / "t3_rows.json").write_text(json.dumps(out, indent=1))
    (res / "t3_results.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
