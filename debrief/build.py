"""Build the debrief viewer: a static page that replays every recording in replay/.

  uv run python cli.py debrief            # writes site/, open site/index.html

For each recording it copies the cockpit frames, writes the readings, transcripts, callouts and
simulator truth into site/data.js (a script, not JSON, so the page also works from file://),
and synthesises each radio line and callout with Crosscheck's voice so the page can play them
at the moment they happened. The page never computes a result: it shows what the recording
holds, and the pass/fail line comes from result.json, written by the scorer.
"""
import json
import pathlib
import shutil

import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
TEMPLATE = pathlib.Path(__file__).resolve().parent / "index.html"

TITLES = {
    "altitude_bust": ("Altitude bust", "Cleared to 11,000; the MCP is set to 12,000."),
    "altitude_readback": ("Readback error", "Cleared to 11,000; the crew reads back 12,000."),
    "altitude_correct": ("Correct setting", "Cleared, read back and set to 11,000. Crosscheck should stay silent."),
    "autothrottle_hold": ("Autothrottle not in speed", "737 on final to Portland 10R; the autothrottle drops out and speed decays (the Asiana 214 pattern)."),
}
ORDER = ["altitude_bust", "altitude_readback", "altitude_correct", "autothrottle_hold"]


def load_jsonl(p: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def build_one(rec: pathlib.Path, synth) -> dict:
    meta = json.loads((rec / "meta.json").read_text())
    name = meta["scenario"]
    t0 = meta["t_start"]
    events = load_jsonl(rec / "events.jsonl")
    truth = load_jsonl(rec / "truth.jsonl")
    result = json.loads((rec / "result.json").read_text())
    out = SITE / "rec" / name
    (out / "frames").mkdir(parents=True, exist_ok=True)

    def rel(t: float) -> float:
        return round(t - t0, 2)

    readings, said, heard, callouts, actions = [], [], [], [], []
    for e in events:
        k = e["kind"]
        if k == "reading":
            if e.get("frame"):
                shutil.copy(rec / e["frame"], out / e["frame"])
            readings.append({"t": rel(e["t"]), "frame": f"rec/{name}/{e['frame']}" if e.get("frame") else None,
                             **{f: e.get(f) for f in ("ias", "selected_speed", "selected_altitude", "altitude", "track_mag")},
                             "fma": e.get("fma"), "fma_armed": e.get("fma_armed")})
        elif k == "said":
            said.append({"t": rel(e["t"]), "key": e["key"], "value": e["value"], "speaker": e["speaker"],
                         "source": e["source"], "transcript": e.get("transcript")})
        elif k == "heard":
            heard.append({"t": rel(e["t"]), "speaker": e["speaker"], "transcript": e["transcript"], "spoken": e["spoken"]})
        elif k == "callout":
            callouts.append({"t": rel(e["t"]), "rule": e["rule"], "text": e["callout"],
                             "expected": e["expected"], "actual": e["actual"], "since": rel(e["since"])})
        elif k == "action":
            actions.append({"t": rel(e["t"]), "line": e["line"]})

    # audio: radio lines (as spoken, started when the line began) and callouts
    audio = []
    for i, h in enumerate(heard):
        wav, sr = synth(h["spoken"])
        f = f"rec/{name}/line{i}.wav"
        sf.write(SITE / f, wav, sr)
        dur = len(wav) / sr
        hh = next((x for x in meta.get("heard", []) if x["spoken"] == h["spoken"] and x["speaker"] == h["speaker"]), None)
        start = h["t"] - (hh["s_from_start"] if hh else dur)
        audio.append({"t": round(start, 2), "dur": round(dur, 2), "file": f, "kind": h["speaker"]})
        h["start"] = round(start, 2)
    for i, c in enumerate(callouts):
        wav, sr = synth(c["text"])
        f = f"rec/{name}/callout{i}.wav"
        sf.write(SITE / f, wav, sr)
        audio.append({"t": c["t"], "dur": round(len(wav) / sr, 2), "file": f, "kind": "callout"})

    marks = []
    if "t_fault" in meta:
        marks.append({"t": rel(meta["t_fault"]), "label": "Autothrottle drops out (fault injected)"})
    if "t_set" in meta:
        marks.append({"t": rel(meta["t_set"]), "label": next((a["line"] for a in actions), "MCP set")})

    tr = []
    for r in truth:
        row = {"t": rel(r["t"])}
        for k in ("alt", "sel_alt", "ias", "sel", "at"):
            if r.get(k) is not None:
                row[k] = round(r[k], 1) if isinstance(r[k], float) else r[k]
        if len(row) > 1:
            tr.append(row)

    title, blurb = TITLES.get(name, (name, ""))
    return {"id": name, "title": title, "blurb": blurb, "kind": meta["scorer"], "meta": {
        k: meta[k] for k in ("aircraft", "area", "airport", "recorded", "speech", "cleared", "readback",
                             "mcp_set", "target_kt", "variant") if k in meta},
        "duration": round(max([r["t"] for r in readings] + [r["t"] for r in tr]), 1),
        "readings": readings, "said": said, "heard": heard, "callouts": callouts, "marks": marks,
        "truth": tr, "audio": audio, "result": result}


def build() -> pathlib.Path:
    from voice.tts import synth
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir()
    recs = {p.parent.name: p.parent for p in (ROOT / "replay").glob("*/meta.json")}
    data = [build_one(recs[n], synth) for n in ORDER if n in recs]
    data += [build_one(p, synth) for n, p in sorted(recs.items()) if n not in ORDER]
    (SITE / "data.js").write_text("window.DEBRIEF = " + json.dumps(data, separators=(",", ":")) + ";\n")
    shutil.copy(TEMPLATE, SITE / "index.html")
    return SITE / "index.html"
