"""Scenario scoring shared by live runs and replays: callouts from Crosscheck's event log,
judged against the X-Plane truth log and the scenario's key times in meta.json."""
import json
import pathlib


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def carry_forward(rows: list[dict]) -> list[dict]:
    """Subscriptions send only changed values: carry the last known value forward."""
    last, out = {}, []
    for r in rows:
        last.update({k: v for k, v in r.items() if v is not None})
        out.append(dict(last))
    return out


def score_autothrottle(events: list[dict], truth: list[dict], meta: dict) -> dict:
    rows = carry_forward(truth)
    t_fault = meta["t_fault"]
    sel = next((r["sel"] for r in reversed(rows) if r["t"] <= t_fault and "sel" in r), meta["target_kt"])
    ias = [(r["t"], r["ias"]) for r in rows if "ias" in r]

    def t_below(margin):
        return next((t for t, v in ias if t > t_fault and v < sel - margin), None)

    callouts = [e for e in events if e["kind"] == "callout"]
    first = next((c for c in callouts if c["t"] >= t_fault and c["rule"] == "mode_vs_phase"), None)
    t5, t10 = t_below(5), t_below(10)
    result = {
        "selected_speed_truth": round(sel, 1),
        "false_callouts_before_fault": [c["callout"] for c in callouts if c["t"] < t_fault],
        "callout": first["callout"] if first else None,
        "callout_s_after_fault": round(first["t"] - t_fault, 1) if first else None,
        "truth_below_target_minus_5_s": round(t5 - t_fault, 1) if t5 else None,
        "truth_below_target_minus_10_s": round(t10 - t_fault, 1) if t10 else None,
        "readings": sum(e["kind"] == "reading" for e in events),
        "paused_during_run": any(r.get("paused") for r in rows),
    }
    result["pass"] = (not result["false_callouts_before_fault"] and first is not None
                      and (t10 is None or first["t"] < t10) and not result["paused_during_run"])
    return result


def score_altitude(events: list[dict], truth: list[dict], meta: dict) -> dict:
    rows = carry_forward(truth)
    t_clear, t_set, want = meta["t_clear"], meta["t_set"], meta["expect_rule"]
    callouts = [e for e in events if e["kind"] == "callout"]
    hit = next((c for c in callouts if c["rule"] == want), None) if want else None
    t_pass = next((r["t"] for r in rows if r["t"] > t_set and r.get("alt", 0) > meta["cleared"]), None)
    result = {
        "variant": meta["variant"], "cleared": meta["cleared"], "readback": meta["readback"], "mcp_set": meta["mcp_set"],
        "false_callouts_before_clearance": [c["callout"] for c in callouts if c["t"] < t_clear],
        "callouts": [{"s_after_set": round(c["t"] - t_set, 1), "text": c["callout"]} for c in callouts],
        "aircraft_passed_cleared_s_after_set": round(t_pass - t_set, 1) if t_pass else None,
        "max_altitude_in_run": round(max(r["alt"] for r in rows if "alt" in r)),
        "mcp_read_from_screen": sorted({e["selected_altitude"] for e in events
                                        if e["kind"] == "reading" and e.get("selected_altitude")}),
        "paused_during_run": any(r.get("paused") for r in rows),
    }
    if want is None:
        result["pass"] = not callouts and not result["paused_during_run"]
    else:
        result["pass"] = (hit is not None and not result["false_callouts_before_clearance"]
                          and (t_pass is None or hit["t"] < t_pass) and not result["paused_during_run"])
    return result


DETENT_OF_RATIO = {0.0: 0, 0.125: 1, 0.25: 2, 0.375: 5, 0.5: 10, 0.625: 15, 0.75: 25, 0.875: 30, 1.0: 40}


def score_flaps(events: list[dict], truth: list[dict], meta: dict) -> dict:
    t_call, want = meta["t_call"], meta["expect_rule"]
    callouts = [e for e in events if e["kind"] == "callout"]
    hit = next((c for c in callouts if c["rule"] == want and c["key"] == "flaps"), None) if want else None
    rows = [r for r in truth if r.get("deploy") is not None]
    final = rows[-1]["deploy"] if rows else None
    settled = next((r["t"] for r in rows if r["t"] > meta["t_move"] and abs(r["deploy"] - final) < 0.003), None)
    gauge = [e for e in events if e["kind"] == "reading" and e.get("flaps") is not None]
    result = {
        "variant": meta["variant"], "called": meta["called"], "selected": meta["selected"],
        "flaps_final_truth": DETENT_OF_RATIO.get(round(final * 8) / 8) if final is not None else None,
        "flaps_settled_s_after_call": round(settled - t_call, 1) if settled else None,
        "heard_callout": [e["value"] for e in events if e["kind"] == "said" and e["key"] == "callout.flaps"],
        "false_callouts_before_call": [c["callout"] for c in callouts if c["t"] < t_call],
        "callouts": [{"s_after_call": round(c["t"] - t_call, 1), "text": c["callout"]} for c in callouts],
        "gauge_read": sorted({e["flaps"] for e in gauge}),
        "paused_during_run": any(r.get("paused") for r in truth),
    }
    if want is None:
        result["pass"] = not callouts and not result["paused_during_run"]
    else:
        result["pass"] = (hit is not None and hit["actual"] == result["flaps_final_truth"]
                          and not result["false_callouts_before_call"] and not result["paused_during_run"])
    return result


def score_pilot_watch(events: list[dict], truth: list[dict], meta: dict) -> dict:
    t_change = meta["t_change"]
    callouts = [e for e in events if e["kind"] == "callout"]
    asked = next((c for c in callouts if c["key"] == "challenge"), None)
    alert = next((c for c in callouts if c["key"] == "incapacitated"), None)
    result = {
        "variant": meta["variant"], "state": meta["state"], "answer": meta.get("answer"),
        "postures": [{"s_after_change": round(e["t"] - t_change, 1), "slumped": e["slumped"], "neck_flex": e["neck_flex"],
                      "wrists": e["wrists"]} for e in events if e["kind"] == "pilot"][::3],
        "heard": [e["transcript"] for e in events if e["kind"] == "heard"],
        "false_callouts_before_change": [c["callout"] for c in callouts if c["t"] < t_change],
        "callouts": [{"s_after_change": round(c["t"] - t_change, 1), "text": c["callout"]} for c in callouts],
        "asked_s_after_change": round(asked["t"] - t_change, 1) if asked else None,
        "alert_s_after_change": round(alert["t"] - t_change, 1) if alert else None,
    }
    want_ask, want_alert = meta["expect_ask"], meta["expect_alert"]
    result["pass"] = (bool(asked) == want_ask and bool(alert) == want_alert
                      and not result["false_callouts_before_change"])
    return result


def score_nominal(events: list[dict], truth: list[dict], meta: dict) -> dict:
    """A correctly flown approach: every callout is a false alarm."""
    rows = carry_forward(truth)
    callouts = [e for e in events if e["kind"] == "callout"]
    heard = meta.get("heard", [])
    expected = sum(1 for _, kind, _ in meta["plan"] if kind == "alt") * 2 + sum(1 for _, k, _ in meta["plan"] if k == "flaps")
    parsed = sum(1 for h in heard for k, _ in h["facts"] if k in ("clearance.altitude", "readback.altitude", "callout.flaps"))
    below = [r["t"] for r in rows if r.get("alt") is not None and r["alt"] < 2500]
    result = {
        "minutes": round(meta["run_s"] / 60, 1),
        "false_callouts": [{"s": round(c["t"] - meta["t_start"], 1), "text": c["callout"], "rule": c["rule"]} for c in callouts],
        "lines_heard": len(heard),
        "altitude_and_flap_facts_parsed": f"{parsed}/{expected}",
        "minutes_below_2500ft": round((below[-1] - below[0]) / 60, 1) if below else 0,
        "readings": sum(e["kind"] == "reading" for e in events),
        "paused_during_run": any(r.get("paused") for r in rows),
    }
    result["pass"] = not callouts and not result["paused_during_run"]
    return result


SCORERS = {"autothrottle_hold": score_autothrottle, "nominal": score_nominal, "altitude": score_altitude, "flaps": score_flaps,
           "pilot_watch": score_pilot_watch}


MIN_READABLE = 0.5  # below this the panel was not on screen (another window in front): not a valid run


def score(rec_dir: pathlib.Path, events_file: str = "events.jsonl") -> dict:
    meta = json.loads((rec_dir / "meta.json").read_text())
    events = load_jsonl(rec_dir / events_file)
    result = SCORERS[meta["scorer"]](events, load_jsonl(rec_dir / "truth.jsonl"), meta)
    readings = [e for e in events if e["kind"] == "reading"]
    if not readings:  # the pilot-watch scenario has no panel camera
        return result
    readable = sum(e.get("ias") is not None or e.get("flaps") is not None for e in readings) / max(1, len(readings))
    result["panel_readable"] = round(readable, 2)
    # a frame much brighter than the dark cockpit means another window was captured (and would
    # be published with the recording); Spark can return numbers for any image, so check pixels
    frames = [rec_dir / e["frame"] for e in readings if e.get("frame") and (rec_dir / e["frame"]).exists()]
    result["non_cockpit_frames"] = sum(_bright(f) for f in frames)
    result["pass"] = result["pass"] and readable >= MIN_READABLE and not result["non_cockpit_frames"]
    return result


def _bright(path: pathlib.Path) -> bool:
    from PIL import Image, ImageStat
    return ImageStat.Stat(Image.open(path).convert("L").resize((64, 32))).mean[0] > 110
