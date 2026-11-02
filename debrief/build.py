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
    "flaps_wrong": ("Flaps: wrong setting", "\u201cFlaps fifteen\u201d called; the lever goes to 10."),
    "flaps_forgot": ("Flaps: not moved", "\u201cFlaps fifteen\u201d called; nobody moves the lever."),
    "pilot_slumped": ("Pilot slumped", "The captain slumps forward and does not answer."),
    "pilot_asleep": ("Pilot asleep", "The captain falls asleep against the side window."),
    "pilot_checklist": ("Pilot reading", "The captain looks down at a checklist. Crosscheck should stay silent."),
    "pilot_answers": ("Pilot answers", "The captain slumps but answers when asked: no alert."),
    "nominal": ("15-minute normal flight", "Six descents, four turns, three speed changes and four flap calls, all done right. Any callout here is a false alarm."),
    "flaps_correct": ("Flaps: correct", "\u201cFlaps fifteen\u201d called and selected. The needle takes ~13 s; Crosscheck should stay silent."),
}
ORDER = ["nominal", "altitude_bust", "altitude_readback", "altitude_correct", "autothrottle_hold",
         "flaps_wrong", "flaps_forgot", "flaps_correct",
         "pilot_slumped", "pilot_asleep", "pilot_checklist", "pilot_answers",
         "altitude_bust_spark", "altitude_readback_spark", "altitude_correct_spark", "autothrottle_hold_spark"]


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
    last_frame = None
    for e in events:
        k = e["kind"]
        if k == "reading":
            # long recordings ship every 5th frame (replay/nominal): show the latest one kept
            if e.get("frame") and (rec / e["frame"]).exists():
                shutil.copy(rec / e["frame"], out / e["frame"])
                last_frame = e["frame"]
            readings.append({"t": rel(e["t"]), "frame": f"rec/{name}/{last_frame}" if last_frame else None,
                             **{f: e.get(f) for f in ("ias", "selected_speed", "selected_altitude", "altitude", "track_mag",
                                                      "flaps", "flaps_clock")},
                             "fma": e.get("fma"), "fma_armed": e.get("fma_armed")})
        elif k == "said":
            said.append({"t": rel(e["t"]), "key": e["key"], "value": e["value"], "speaker": e["speaker"],
                         "source": e["source"], "transcript": e.get("transcript")})
        elif k == "heard":
            heard.append({"t": rel(e["t"]), "speaker": e["speaker"], "transcript": e["transcript"], "spoken": e["spoken"]})
        elif k == "callout":
            callouts.append({"t": rel(e["t"]), "rule": e["rule"], "key": e["key"], "text": e["callout"],
                             "expected": e["expected"], "actual": e["actual"], "since": rel(e["since"])})
        elif k == "action":
            actions.append({"t": rel(e["t"]), "line": e["line"]})
        elif k == "pilot":  # pilot-watch camera: posture from SAM 3D Body
            have = bool(e.get("frame")) and (rec / e["frame"]).exists()  # webcam frames are git-ignored
            if have:
                shutil.copy(rec / e["frame"], out / e["frame"])
            readings.append({"t": rel(e["t"]), "frame": f"rec/{name}/{e['frame']}" if have else None,
                             **{f: e.get(f) for f in ("neck_flex", "wrists", "slumped", "reasons")}})

    # audio: radio lines (as spoken, started when the line began) and callouts
    audio = []
    mac_voice = "macOS voice" in meta.get("speech", "")  # flaps and pilot watch: the crew spoke in a macOS voice
    for i, h in enumerate(heard):
        hh = meta.get("heard", [])[i] if i < len(meta.get("heard", [])) else None
        if hh and hh.get("audio_file"):  # a person on the microphone: play their own recorded voice
            h["spoken"] = "(live microphone)"
            if (rec / hh["audio_file"]).exists():
                wav, sr = sf.read(rec / hh["audio_file"])
            else:  # voice recordings are git-ignored: elsewhere, Crosscheck's voice reads the transcript
                wav, sr = synth(h["transcript"])
        elif mac_voice:
            from voice.crew import SR as crew_sr, say
            wav, sr = say(h["spoken"]), crew_sr
        else:
            wav, sr = synth(h["spoken"])
        f = f"rec/{name}/line{i}.wav"
        sf.write(SITE / f, wav, sr)
        dur = len(wav) / sr
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
    if "t_change" in meta:
        marks.append({"t": rel(meta["t_change"]), "label": {"slumped": "Pilot slumps", "slumped_side": "Pilot falls asleep",
                                                            "looking_down": "Pilot looks down at a checklist"}.get(meta["state"], meta["state"])})
    if "t_call" in meta:
        marks.append({"t": rel(meta["t_move"]), "label": next((a["line"] for a in actions), "lever")})
    if "t_set" in meta:
        marks.append({"t": rel(meta["t_set"]), "label": next((a["line"] for a in actions), "MCP set")})

    tr = []
    for r in truth:
        row = {"t": rel(r["t"])}
        for k in ("alt", "sel_alt", "ias", "sel", "at", "handle", "deploy", "state"):
            if r.get(k) is not None:
                digits = 4 if k in ("handle", "deploy") else 1  # ratios: 0.625 is flaps 15, 0.6 is not
                row[k] = round(r[k], digits) if isinstance(r[k], float) else r[k]
        if len(row) > 1:
            tr.append(row)

    base = name.removesuffix("_spark").removesuffix("_mic").removesuffix("_webcam")
    title, blurb = TITLES.get(base, (name, ""))
    if name.endswith("_mic"):
        title += " · live voice"
        blurb = "A person speaking into the Mac's microphone. " + blurb
    if name.endswith("_webcam"):
        title += " · webcam"
        blurb = "A person in front of the Mac's camera, following spoken instructions. " + blurb
    if name.endswith("_spark"):
        title += " · Spark reader"
        if meta.get("known_result"):
            blurb = "Too slow here: Spark's ~5 s reads put the callout after speed was 10 kt low."
    box = meta.get("frame_box", [1240, 1230, 3180, 2200])
    sp = rec / "summary.json"  # Muse Spark debrief summary (brain/summary.py), if written
    summary = {k: v for k, v in json.loads(sp.read_text()).items() if k in ("text", "verified", "unsupported_numbers", "model")} if sp.exists() else None
    return {"id": name, "title": title, "blurb": blurb, "kind": meta["scorer"],
            "aspect": "4 / 3" if meta.get("pilot_camera") else f"{box[2] - box[0]} / {box[3] - box[1]}", "meta": {
        k: meta[k] for k in ("aircraft", "area", "airport", "recorded", "speech", "cleared", "readback",
                             "mcp_set", "target_kt", "variant", "panel_reader", "known_result",
                             "called", "selected", "state", "answer", "pose") if k in meta},
        "duration": round(max([r["t"] for r in readings] + [r["t"] for r in tr]), 1),
        "readings": readings, "said": said, "heard": heard, "callouts": callouts, "marks": marks,
        "truth": tr, "audio": audio, "result": result, "summary": summary}


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
