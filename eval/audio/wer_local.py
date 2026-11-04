"""Score the on-board recognizers (eval/audio/asr_local.py caches) next to Muse Voice Transcribe,
on the same clips, with the same normalisation and the same "every number exact" count.

Run:  uv run python eval/audio/wer_local.py [--open] [engine ...]
      uv run python eval/audio/wer_local.py --scenario    the scenarios' own lines: ledger facts
                                                           right / missed / wrong, per recognizer and
                                                           for the on-board pair (senses/speech_spark.py)
"""
import collections
import json
import pathlib
import sys

import jiwer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from senses.speech import normalise, numbers  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OPEN = "--open" in sys.argv
SET = ROOT / "eval" / "audio" / ("radio_set_open" if OPEN else "radio_set")
RES = ROOT / "eval" / "results" / "day_one"
SUFFIX = "_open" if OPEN else ""


def score(manifest: list[dict], text: dict[str, str]) -> dict:
    """{condition: (lines with every number exact, WER, median seconds)} for one recognizer."""
    pairs, exact = collections.defaultdict(list), collections.defaultdict(list)
    seen = set()
    for m in manifest:
        for clip, cond in ((m["clean"], "clean"), (m["file"], f"{m['snr_db']:+d} dB")):
            if clip in seen or clip not in text:
                continue
            seen.add(clip)
            pairs[cond].append((normalise(m["text"]), normalise(text[clip])))
            if numbers(m["text"]):
                exact[cond].append(numbers(text[clip]) == numbers(m["text"]))
    order = ["clean", "+5 dB", "+0 dB", "-5 dB"]
    return {c: {"numbers_exact": f"{sum(exact[c])}/{len(exact[c])}",
                "wer": round(jiwer.wer([r for r, _ in pairs[c]], [h for _, h in pairs[c]]), 3)}
            for c in order if pairs[c]}


def scenario_facts() -> int:
    """What reaches the ledger from the scenarios' lines: each recognizer alone, and the on-board
    pair through onboard_facts (no aircraft context here, so no plausibility check)."""
    from senses.speech import said_facts
    from senses.speech_spark import onboard_facts
    manifest = json.loads((ROOT / "eval" / "audio" / "scenario_set" / "manifest.json").read_text())
    text = {n: {k: v["text"] for k, v in json.loads((RES / f"t2_{c}_scenario_cache.json").read_text()).items()}
            for n, c in (("voice transcribe (cloud)", "muse"), ("seamless-m4t-v2", "seamless"), ("omnilingual-asr-7b", "omni-llm-7b"))}
    out = {}
    for m in manifest:
        ref = sorted(tuple(f) for f in m["facts"])
        if not ref:
            continue
        got = {n: sorted((f.key, f.value) for f in said_facts(t[m["file"]], m["speaker"], 0, "421")) for n, t in text.items()}
        got["on-board pair"] = sorted((f.key, f.value) for f in onboard_facts(
            text["seamless-m4t-v2"][m["file"]], text["omnilingual-asr-7b"][m["file"]], m["speaker"], 0, "421"))
        for n, g in got.items():
            c = out.setdefault(n, {"right": 0, "missed": 0, "wrong": 0, "wrong_lines": []})
            v = "right" if g == ref else "missed" if not g else "wrong"
            c[v] += 1
            if v == "wrong":
                c["wrong_lines"].append(m["file"])
    for n, c in out.items():
        print(f"{n:28s} right {c['right']:2d}  missed {c['missed']:2d}  wrong {c['wrong']:2d}  of {c['right'] + c['missed'] + c['wrong']}")
    (RES / "t2_local_scenario.json").write_text(json.dumps(out, indent=1))
    return 0


def main() -> int:
    if "--scenario" in sys.argv:
        return scenario_facts()
    manifest = json.loads((SET / "manifest.json").read_text())
    engines = [a for a in sys.argv[1:] if not a.startswith("--")] or sorted(
        p.name[3:-len(f"{SUFFIX}_cache.json")] for p in RES.glob(f"t2_*{SUFFIX}_cache.json")
        if not p.name.startswith("t2_muse") and (OPEN or "_open" not in p.name))
    out = {}
    muse = json.loads((RES / f"t2_muse{SUFFIX}_cache.json").read_text())
    passes = sorted({k.split("#")[1] if "#" in k else "0" for k in muse})
    for r in passes:
        out[f"muse voice transcribe (cloud), pass {int(r) + 1}"] = score(
            manifest, {k.split("#")[0]: v["text"] for k, v in muse.items() if (k.split("#")[1] if "#" in k else "0") == r})
    for e in engines:
        cache = json.loads((RES / f"t2_{e}{SUFFIX}_cache.json").read_text())
        s = score(manifest, {k: v["text"] for k, v in cache.items()})
        secs = sorted(v["s"] for v in cache.values())
        out[f"{e} (on board), median {secs[len(secs) // 2]} s a line"] = s
    for name, s in out.items():
        print(f"{name}\n  " + "  ".join(f"{c}: {v['numbers_exact']} exact, WER {v['wer']}" for c, v in s.items()))
    (RES / f"t2_local{SUFFIX}.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
