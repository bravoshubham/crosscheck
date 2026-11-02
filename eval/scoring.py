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


SCORERS = {"autothrottle_hold": score_autothrottle, "altitude": score_altitude}


def score(rec_dir: pathlib.Path, events_file: str = "events.jsonl") -> dict:
    meta = json.loads((rec_dir / "meta.json").read_text())
    return SCORERS[meta["scorer"]](load_jsonl(rec_dir / events_file), load_jsonl(rec_dir / "truth.jsonl"), meta)
