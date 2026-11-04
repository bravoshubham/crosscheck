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
         "altitude_bust_spark", "altitude_readback_spark", "altitude_correct_spark", "autothrottle_hold_spark",
         "altitude_bust_onboard", "altitude_readback_onboard", "altitude_correct_onboard", "autothrottle_hold_onboard",
         "flaps_wrong_onboard", "flaps_forgot_onboard", "flaps_correct_onboard", "nominal_onboard",
         "pilot_slumped_onboard", "pilot_asleep_onboard", "pilot_checklist_onboard", "pilot_answers_onboard"]
ONBOARD_NOTE = "On board: every model on this computer, and the network refused for the whole run. "


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
    last_frame = last_wide = None
    for e in events:
        k = e["kind"]
        if k == "reading":
            # long recordings ship every 5th frame (replay/nominal): show the latest one kept
            if e.get("frame") and (rec / e["frame"]).exists():
                shutil.copy(rec / e["frame"], out / e["frame"])
                last_frame = e["frame"]
                w = e["frame"].replace("frames/", "wide/")
                if (rec / w).exists():  # the whole cockpit, for people (the readers use the crop)
                    (out / "wide").mkdir(exist_ok=True)
                    shutil.copy(rec / w, out / w)
                    last_wide = w
            readings.append({"t": rel(e["t"]), "frame": f"rec/{name}/{last_frame}" if last_frame else None,
                             "wide": f"rec/{name}/{last_wide}" if last_wide else None,
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
        elif k == "explain" and e.get("text"):  # Muse Spark's sentence under the latest matching callout
            c = next((c for c in reversed(callouts) if c["rule"] == e["rule"] and c["key"] == e["key"]), None)
            if c:
                c["explain"], c["explain_t"] = e["text"], rel(e["t"])
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
    from voice.tts import callout
    for i, c in enumerate(callouts):
        wav, sr = callout(c["text"])  # Crosscheck's own voice, from the bank
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

    base = name.removesuffix("_onboard").removesuffix("_spark").removesuffix("_mic").removesuffix("_webcam")
    title, blurb = TITLES.get(base, (name, ""))
    if name.endswith("_onboard"):
        title += " · on board"
        blurb = ONBOARD_NOTE + blurb
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
    video = None
    if meta.get("video") and (rec / meta["video"]).exists():  # pilot watch on generated footage
        shutil.copy(rec / meta["video"], out / meta["video"])
        video = f"rec/{name}/{meta['video']}"
    wide = None
    if any(r.get("wide") for r in readings):  # where SAM 3.1 found the displays, and the crop the readers use
        from brain.live import SCREEN_SIZE, WIDE_W
        cc = json.loads((rec / "camera_check.json").read_text()) if (rec / "camera_check.json").exists() else {}
        concept = next(iter(cc.get("matches", {})), None)
        match = next((m.get("box") for m in cc.get("matches", {}).values() if m.get("box")), None)
        wide = {"scale": WIDE_W / SCREEN_SIZE[0], "aspect": f"{SCREEN_SIZE[0]} / {SCREEN_SIZE[1]}", "found": cc.get("found", []), "match": match,
                "concept": "gauge" if concept == "gauge" else "screen", "crop": meta.get("frame_box", [1240, 1230, 3180, 2200])}
    rv = rec / "review.json"  # Muse Wonder's independent review of the pilot camera (brain/review.py)
    review = {k: v for k, v in json.loads(rv.read_text()).items() if k in ("model", "state", "from_s", "evidence", "agrees_with_truth", "expected_state", "s")} if rv.exists() else None
    sp = rec / "summary.json"  # Muse Spark debrief summary (brain/summary.py), if written
    summary = {k: v for k, v in json.loads(sp.read_text()).items() if k in ("text", "verified", "unsupported_numbers", "model")} if sp.exists() else None
    return {"id": name, "title": title, "blurb": blurb, "kind": meta["scorer"],
            "aspect": "4 / 3" if meta.get("pilot_camera") else f"{box[2] - box[0]} / {box[3] - box[1]}", "meta": {
        k: meta[k] for k in ("aircraft", "area", "airport", "recorded", "speech", "cleared", "readback",
                             "mcp_set", "target_kt", "variant", "panel_reader", "known_result",
                             "called", "selected", "state", "answer", "pose", "network") if k in meta},
        "duration": round(max([r["t"] for r in readings] + [r["t"] for r in tr]), 1),
        "readings": readings, "said": said, "heard": heard, "callouts": callouts, "marks": marks,
        "truth": tr, "audio": audio, "result": result, "summary": summary, "video": video, "actions": actions, "review": review, "wide": wide}


# The test matrix on the overview: recording, repeat-run summaries (eval/results/repeat/), what each
# run's time is measured from, and the deadline field it had to beat (if the scenario has one).
MATRIX = [
    ("altitude_bust", ["altitude_bust"], "Cleared to 11,000; the pilot flying sets 12,000.", "after the MCP was set",
     "aircraft_passed_cleared_s_after_set", "the 737 passed 11,000"),
    ("altitude_readback", ["altitude_readback"], "Cleared to 11,000; the crew reads back 12,000.", "after the MCP was set", None, None),
    ("altitude_correct", ["altitude_correct"], "Cleared, read back and set to 11,000.", None, None, None),
    ("altitude_bust_spark", ["altitude_bust_spark"], "The bust, with Muse Spark reading the panel.", "after the MCP was set",
     "aircraft_passed_cleared_s_after_set", "the 737 passed 11,000"),
    ("altitude_readback_spark", ["altitude_readback_spark"], "The readback error, with Muse Spark reading the panel.", "after the MCP was set", None, None),
    ("altitude_correct_spark", ["altitude_correct_spark"], "The correct setting, with Muse Spark reading the panel.", None, None, None),
    ("autothrottle_hold", ["autothrottle_hold"], "On final the autothrottle drops out and speed decays.", "after the fault",
     "truth_below_target_minus_10_s", "speed was 10 kt low"),
    ("autothrottle_hold_spark", ["autothrottle_hold_spark"], "The same, with Muse Spark reading the panel.", "after the fault",
     "truth_below_target_minus_10_s", "speed was 10 kt low"),
    ("flaps_wrong", ["flaps_wrong"], "\u201cFlaps fifteen\u201d called; the lever goes to 10.", "after the call", None, None),
    ("flaps_forgot", ["flaps_forgot"], "\u201cFlaps fifteen\u201d called; nobody moves the lever.", "after the call", None, None),
    ("flaps_correct", ["flaps_correct"], "\u201cFlaps fifteen\u201d called and selected.", None, None, None),
    ("pilot_slumped", ["pilot_watch_slumped_video"], "The captain slumps forward and does not answer.", "after the slump", None, None),
    ("pilot_asleep", ["pilot_watch_asleep_video"], "The captain falls asleep against the side window.", "after the slump", None, None),
    ("pilot_checklist", ["pilot_watch_checklist_video"], "The captain looks down at a checklist.", None, None, None),
    ("pilot_answers", ["pilot_watch_answers_video"], "The captain slumps, then answers when asked.", "after the slump", None, None),
    ("nominal", [], "Fifteen minutes flown correctly: six descents, four turns, four flap calls.", None, None, None),
]
# The same scenarios on board (onboard/README.md): local models, network refused for the whole run
MATRIX += [(rid + "_onboard", [rp + "_onboard" for rp in reps], what, measured, dk, dl)
           for rid, reps, what, measured, dk, dl in MATRIX if not rid.endswith("_spark")]
SAYS = {"altitude_bust": "Cleared one-one thousand, set one-two thousand.",
        "altitude_readback": "Readback one-two thousand, cleared one-one thousand.",
        "altitude_bust_spark": "Cleared one-one thousand, set one-two thousand.",
        "altitude_readback_spark": "Readback one-two thousand, cleared one-one thousand.",
        "autothrottle_hold": "Speed. Autothrottle not in speed.", "autothrottle_hold_spark": "Speed. Autothrottle not in speed.",
        "flaps_wrong": "Flaps fifteen called, ten set.", "flaps_forgot": "Flaps fifteen called, five set.",
        "pilot_slumped": "Captain, are you OK?\u201d then \u201cFirst officer, take control.",
        "pilot_asleep": "Captain, are you OK?\u201d then \u201cFirst officer, take control.",
        "pilot_answers": "Captain, are you OK?"}


def nominal_since_fixes() -> list[pathlib.Path]:
    return [ROOT / "eval" / "results" / f"nominal_run{i}" / "result.json" for i in range(4, 10)
            if (ROOT / "eval" / "results" / f"nominal_run{i}" / "result.json").exists()]


def _first_time(run: dict) -> float | None:
    """When the first callout came, in the run's own terms (seconds after the set / fault / call / change)."""
    if run.get("callout_s_after_fault") is not None:
        return run["callout_s_after_fault"]
    c = run.get("callouts")
    if isinstance(c, list) and c:
        return next(v for k, v in c[0].items() if k.startswith("s_after"))
    return None


def models(data: list[dict]) -> list[dict]:
    """Each Meta model's job here and what was measured, read from the result files (nothing typed by hand)."""
    R = ROOT / "eval" / "results"
    load = lambda p: json.loads(p.read_text()) if p.exists() else None
    out = []
    t2, nom = load(R / "day_one" / "t2_muse.json"), next((r for r in data if r["id"] == "nominal"), None)
    m = []
    if t2:
        m.append("Radio test lines with every number right, clean audio, three passes: " + ", ".join(t2["numbers_exact_by_pass"]["muse clean"]))
    if nom:
        m.append(f"15-minute flight: {nom['result']['lines_heard']} lines heard live, {nom['result']['altitude_and_flap_facts_parsed']} values right")
    out.append({"model": "Muse Voice Transcribe", "job": "Hears the radio and the intercom, one channel each, as they are spoken; the person at the microphone in a live demo.", "measured": m})
    ps, fb = load(R / "day_one" / "panel_spark.json"), load(R / "speech_fallback.json")
    m = []
    if ps:
        m.append(f"Reading the flight displays: altitude within 20 ft {ps['right']['altitude']['spark']} frames (on-device text recognition {ps['right']['altitude']['apple']})")
    if fb:
        m.append(f"Second reader for lines code cannot parse: {fb['code_only']['right']} \u2192 {fb['code_then_spark']['right']} of {fb['code_only']['of']} values, {fb['code_then_spark']['wrong']} wrong")
    sums = [r["summary"] for r in data if r.get("summary")]
    if sums:
        m.append(f"Debrief summaries: {sum(s['verified'] for s in sums)} of {len(sums)} with every number found in the log")
    out.append({"model": "Muse Spark", "job": "Reads the flight displays and the analog flap gauge from camera frames on any computer; reads radio lines the parser cannot; writes a sentence under each callout and each recording's debrief.", "measured": m})
    reviews = [load(ROOT / "replay" / r["id"] / "review.json") for r in data if r["kind"] == "pilot_watch"]
    reviews = [x for x in reviews if x]
    for sp in sorted((R / "repeat").glob("pilot_watch_*_video.json")):  # and the repeat runs
        reviews += [run["wonder"] for run in json.loads(sp.read_text())["runs"] if run.get("wonder")]
    out.append({"model": "Muse Wonder", "job": "Watches the pilot-camera recording after each run and judges it on its own, without the poses or Crosscheck's alerts.",
                "measured": [f"Agreed with the ground truth on {sum(x['agrees_with_truth'] for x in reviews)} of {len(reviews)} pilot-camera runs "
                             "(slumped, asleep, reading a checklist, slumped then answering)"] if reviews else []})
    checks = [load(ROOT / "replay" / r["id"] / "camera_check.json") for r in data]
    checks = [c for c in checks if c]
    sb = load(R / "pilot_sam_boxes.json")
    m = ([f"Pre-flight camera check passed on {sum(c['ok'] for c in checks)} of {len(checks)} recordings"] if checks else []) + \
        ([f"Finding the captain for SAM 3D Body: posture right on {sb['right']} of {sb['of']} labelled images"] if sb else [])
    out.append({"model": "SAM 3.1", "job": "Finds the flight displays before each run (the camera check) and the captain in the pilot camera, from one-word prompts.", "measured": m})
    pp = load(R / "day_one" / "pilot_posture.json")
    out.append({"model": "SAM 3D Body", "job": "Turns every pilot-camera frame into a 3D body pose; code measures the head and hands from it.",
                "measured": [f"Labelled images right: {k} {v}" for k, v in pp.items()] if pp else []})
    imgs = len([p for p in (ROOT / "eval" / "pilot_watch" / "images").glob("*/*.jpg")])
    out.append({"model": "Muse Image", "job": "Makes the synthetic pilots the pilot camera is tested on, and the halfway poses the video is anchored on.", "measured": [f"{imgs} images of 6 synthetic pilots"]})
    out.append({"model": "MMS-TTS", "job": "The synthetic radio and intercom voices of the scenarios and test sets; Crosscheck's voice for any callout not in its recorded bank (voice/bank/).", "measured": []})
    pg, hs = load(R / "day_one" / "panel_glimmer.json"), load(R / "day_one" / "t2_local_scenario.json")
    m = []
    if pg:
        alt = pg["right"].get("altitude", {})
        m.append(f"Reading the flight displays: altitude within 20 ft {alt.get('glimmer')} frames "
                 f"(the reader that recorded them: {alt.get('apple') or alt.get('muse-spark')})")
        m.append(f"The flap gauge: {pg['flap_gauge_right']['glimmer']} frames right (Muse Spark {pg['flap_gauge_right']['muse-spark']})")
    fl = load(R / "day_one" / "panel_final_lane.json")
    if fl:
        f = fl["final"]
        m.append(f"On final, speeds and modes only: airspeed within 2 kt {f['ias']} frames, thrust in speed or not {f['thrust']}; "
                 "2.15 s a read with X-Plane flying")
    out.append({"model": "Muse Glimmer 30B (open weights, on board)",
                "job": "On board, with no network: reads the flight displays and the flap gauge, reads radio lines the parser cannot, writes the sentence under each callout.",
                "measured": m})
    if hs:
        pair, vt = hs["on-board pair"], hs["voice transcribe (cloud)"]
        n = pair["right"] + pair["missed"] + pair["wrong"]
        out.append({"model": "SeamlessM4T v2 + Omnilingual ASR 7B (open weights, on board)",
                    "job": "On board, with no network: both hear every radio and intercom line; a value counts when they do not disagree.",
                    "measured": [f"The scenarios' own lines: {pair['right']} of {n} values right, {pair['wrong']} wrong "
                                 f"(Muse Voice Transcribe: {vt['right']} right, {vt['wrong']} wrong)"]})
    return out


def overview(data: list[dict]) -> dict:
    """Everything the overview and test report show, from the recordings and the repeat summaries."""
    by_id = {r["id"]: r for r in data}
    rows, totals, onboard = [], {"runs": 0, "right": 0}, {"runs": 0, "right": 0}
    for rid, repeats, what, measured, deadline_key, deadline_label in MATRIX:
        if rid not in by_id:
            continue
        runs = []
        for rp in repeats:
            p = ROOT / "eval" / "results" / "repeat" / f"{rp}.json"
            if p.exists():
                runs += [{"pass": x["pass"], "t": _first_time(x), "deadline": x.get(deadline_key) if deadline_key else None}
                         for x in json.loads(p.read_text())["runs"] if "pass" in x]
        if rid == "nominal":  # the earlier flights since the fixes count too; runs 1-3 found the fixes and are evidence
            runs += [{"pass": json.loads(p.read_text())["pass"], "t": None, "deadline": None} for p in nominal_since_fixes()]
        rec = by_id[rid]
        res = rec["result"]
        live = {"pass": res["pass"], "t": _first_time(res), "deadline": res.get(deadline_key) if deadline_key else None, "recording": True}
        runs = [live] + runs
        tally = onboard if rid.endswith("_onboard") else totals  # on board is reported on its own line
        tally["runs"] += len(runs)
        tally["right"] += sum(r["pass"] for r in runs)
        rows.append({"id": rid, "title": rec["title"], "what": what, "says": SAYS.get(rid.removesuffix("_onboard")), "measured": measured,
                     "deadline_label": deadline_label, "runs": runs, "kind": rec["kind"]})
    hero = by_id.get("altitude_bust_spark") or by_id.get("altitude_bust")  # the panel read by Muse Spark, if recorded
    if hero:
        c = hero["callouts"][0]
        at = next((r for r in hero["readings"] if r["t"] >= c["t"] and r.get("frame")), {})
        hero = {"id": hero["id"], "frame": at.get("wide") or at.get("frame"), "wide": hero.get("wide") if at.get("wide") else None,
                "reader": hero["meta"].get("panel_reader"), "callout": c, "duration": hero["duration"],
                "said": [f for f in hero["said"]],
                "set": [[r["t"], r["selected_altitude"]] for r in hero["readings"] if r.get("selected_altitude")],
                "doing": [[r["t"], r["altitude"]] for r in hero["readings"] if r.get("altitude")],
                "aspect": hero["aspect"]}
    nominal = None
    if "nominal" in by_id:
        n = by_id["nominal"]
        res = n["result"]
        earlier = []
        for d, label in (("nominal_before_fixes", "Run 1"), ("nominal_run2", "Run 2"), ("nominal_run3", "Run 3"), ("nominal_run4", "Run 4"), ("nominal_run5", "Run 5"), ("nominal_run6", "Run 6")):
            p = ROOT / "eval" / "results" / d / "result.json"
            if p.exists():
                earlier.append({"label": label, "false_callouts": json.loads(p.read_text())["false_callouts"]})
        nominal = {"id": "nominal", "duration": n["duration"], "result": res, "earlier": earlier,
                   "alt": [[r["t"], r["altitude"]] for r in n["readings"] if r.get("altitude")],
                   "heard": [[h["start"], h["t"], h["speaker"]] for h in n["heard"]],
                   "actions": [[a["t"], a["line"]] for a in n.get("actions", [])]}
    return {"hero": hero, "matrix": rows, "totals": totals, "onboard": onboard, "nominal": nominal, "models": models(data),
            "findings": json.loads((TEMPLATE.parent / "findings.json").read_text())}


def build() -> pathlib.Path:
    from voice.tts import synth
    films = {p.name: p.read_bytes() for p in SITE.glob("crosscheck_film*.mp4")}  # cli.py film made these: keep them
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir()
    for name, b in films.items():
        (SITE / name).write_bytes(b)
    recs = {p.parent.name: p.parent for p in (ROOT / "replay").glob("*/meta.json")}
    data = [build_one(recs[n], synth) for n in ORDER if n in recs]
    data += [build_one(p, synth) for n, p in sorted(recs.items()) if n not in ORDER]
    (SITE / "data.js").write_text("window.DEBRIEF = " + json.dumps(data, separators=(",", ":")) + ";\n")
    (SITE / "overview.js").write_text("window.OVERVIEW = " + json.dumps(overview(data), separators=(",", ":")) + ";\n")
    shutil.copy(TEMPLATE, SITE / "debrief.html")
    shutil.copy(TEMPLATE.parent / "overview.html", SITE / "index.html")
    return SITE / "index.html"
