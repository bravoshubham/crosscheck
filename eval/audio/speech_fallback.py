"""Does Muse Spark as a second reader recover values the code parser misses, without adding wrong ones?

Uses the Voice Transcribe transcripts already cached for the radio set (3 passes of every clean and
noisy clip, eval/audio/wer_muse.py), so no audio is re-sent. The expected facts come from parsing
each line's script. For each transcript: the code parser alone (senses/speech.py), then the same
with Muse Spark asked about the lines code got nothing from (senses/speech_spark.py, guarded).

  uv run python eval/audio/speech_fallback.py
"""
import asyncio
import json
import pathlib
import sys

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from senses.speech import said_facts  # noqa: E402
from senses.speech_spark import spark_facts, worth_asking  # noqa: E402

SET = ROOT / "eval" / "audio" / "radio_set"
CACHE = ROOT / "eval" / "results" / "day_one" / "t2_muse_cache.json"
OUT = ROOT / "eval" / "results" / "speech_fallback.json"
PASSES = 3


def key(f) -> tuple:
    return (f.key, f.value)


async def main() -> int:
    manifest = json.loads((SET / "manifest.json").read_text())
    cache = json.loads(CACHE.read_text())
    clips = {}
    for m in manifest:  # every raw clip once (clean and noisy), with its script and channel
        speaker = "atc" if m["speaker"] == "atc" else "pm"
        clips[m["clean"]] = (m["text"], speaker, "clean")
        clips[m["file"]] = (m["text"], speaker, f"{m['engine']} {m['snr_db']:+d} dB")
    rows = []
    sem = asyncio.Semaphore(6)
    async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
        async def one(name, r):
            text, speaker, cond = clips[name]
            k = name if r == 0 else f"{name}#{r}"
            if k not in cache:
                return
            heard = cache[k]["text"]
            want = {key(f) for f in said_facts(text, speaker, 0, "421")}
            code = {key(f) for f in said_facts(heard, speaker, 0, "421")}
            spark, answer = set(), None
            if not code and worth_asking(heard):
                async with sem:
                    facts, answer = await spark_facts(heard, speaker, 0, "421", client)
                spark = {key(f) for f in facts}
            rows.append({"clip": name, "pass": r, "condition": cond, "script": text, "heard": heard, "want": sorted(want),
                         "code": sorted(code), "spark": sorted(spark), "spark_answer": answer})
        await asyncio.gather(*(one(n, r) for n in clips for r in range(PASSES)))

    def tally(use_spark: bool) -> dict:
        t = {"right": 0, "missed": 0, "wrong": 0, "of": 0}
        for row in rows:
            want = {tuple(x) for x in row["want"]}
            got = {tuple(x) for x in row["code"]} | ({tuple(x) for x in row["spark"]} if use_spark else set())
            t["of"] += len(want)
            t["right"] += len(want & got)
            t["missed"] += len(want - got)
            t["wrong"] += len(got - want)
        return t
    asked = sum(r["spark_answer"] is not None for r in rows)
    result = {"transcripts": len(rows), "spark_asked": asked, "code_only": tally(False), "code_then_spark": tally(True),
              "recovered": [r for r in rows if set(map(tuple, r["spark"])) - set(map(tuple, r["code"]))],
              "rows": rows}
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "recovered")}, indent=1))
    for r in result["recovered"]:
        print(f"  recovered {r['spark']} from \"{r['heard']}\" ({r['condition']})")
    wrong = [r for r in rows if set(map(tuple, r["spark"])) - {tuple(x) for x in r["want"]}]
    for r in wrong:
        print(f"  WRONG {r['spark']} from \"{r['heard']}\" (script: {r['script']})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
