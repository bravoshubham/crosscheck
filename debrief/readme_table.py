"""The README's "What it catches" table, generated from the same files as the test report.

  uv run python -m debrief.readme_table      # rewrites the table between the markers in README.md

Every count is every live run of that scenario (the shipped recording and its repeats, misses
included); the timing is the shipped recording's.
"""
import json
import pathlib
import re

from debrief.build import MATRIX, SAYS, _first_time, nominal_since_fixes

ROOT = pathlib.Path(__file__).resolve().parents[1]
START, END = "<!-- what-it-catches:start -->", "<!-- what-it-catches:end -->"


def rows() -> list[str]:
    out = ["| Scenario | What goes wrong | Crosscheck says | Live runs right | The shipped recording |",
           "| --- | --- | --- | --- | --- |"]
    for rid, repeats, what, measured, deadline_key, deadline_label in MATRIX:
        rec = ROOT / "replay" / rid
        if not (rec / "result.json").exists():
            continue
        res = json.loads((rec / "result.json").read_text())
        meta = json.loads((rec / "meta.json").read_text())
        runs = [res["pass"]]
        for rp in repeats:
            p = ROOT / "eval" / "results" / "repeat" / f"{rp}.json"
            if p.exists():
                runs += [x["pass"] for x in json.loads(p.read_text())["runs"] if "pass" in x]
        if rid == "nominal":
            runs += [json.loads(p.read_text())["pass"] for p in nominal_since_fixes()]
        t = _first_time(res)
        if rid == "nominal":
            shipped = f"{len(res['false_callouts'])} false alarms in {res['minutes']} minutes; {res['altitude_and_flap_facts_parsed']} spoken values right"
        elif not measured:
            shipped = "Silent, as it should be" if res["pass"] else "A false callout"
        elif t is None:
            shipped = "No callout (a miss)"
        else:
            shipped = f"{t:.1f} s {measured}"
            if deadline_key and res.get(deadline_key) is not None:
                shipped += f"; {deadline_label} at {res[deadline_key]:.1f} s"
            if not res["pass"]:
                shipped += " (a miss)"
        title = {"altitude_bust": "Altitude bust", "altitude_readback": "Readback error", "altitude_correct": "Correct setting",
                 "altitude_bust_spark": "Altitude bust, panel read by Muse Spark", "altitude_readback_spark": "Readback error, Muse Spark",
                 "altitude_correct_spark": "Correct setting, Muse Spark", "autothrottle_hold": "Autothrottle not in speed on final (the Asiana 214 pattern)",
                 "autothrottle_hold_spark": "The same, panel read by Muse Spark", "flaps_wrong": "Flaps: wrong setting",
                 "flaps_forgot": "Flaps: not moved", "flaps_correct": "Flaps: correct", "pilot_slumped": "Pilot slumped",
                 "pilot_asleep": "Pilot asleep", "pilot_checklist": "Pilot reading a checklist", "pilot_answers": "Pilot slumps, then answers",
                 "nominal": "15 minutes of normal flying"}.get(rid, rid)
        says = f"“{SAYS[rid]}”" if rid in SAYS else "(silence)"
        out.append(f"| {title} | {what} | {says} | {sum(runs)} of {len(runs)} | {shipped} |")
    return out


def main() -> None:
    p = ROOT / "README.md"
    s = p.read_text()
    table = "\n".join([START] + rows() + [END])
    s = re.sub(re.escape(START) + r".*?" + re.escape(END), table, s, flags=re.S) if START in s else s
    p.write_text(s)
    print(table)


if __name__ == "__main__":
    main()
