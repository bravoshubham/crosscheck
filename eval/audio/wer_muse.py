"""Word error rate on the radio set with Muse Voice Transcribe (Meta Model API, realtime).

Streams each clip to wss://api.meta.ai/v1/asr/realtime the way a live cockpit feed would:
a JSON handshake, 80 ms PCM16 frames at real-time pace, then endStream; the last final
transcript is the result. Scores clean speech, raw mixes and SAM Audio outputs
(<mix>.sam.wav), grouped by SNR, and rescores the MMS transcripts with the same text
normalisation so the two recognizers are compared like for like.

Transcripts are cached in eval/results/day_one/t2_muse_cache.json, so a rerun only pays for
clips not yet transcribed. Price used for the cost line: $3 per 1,000 audio minutes.

Run:  uv run --with websockets python eval/audio/wer_muse.py
"""
import asyncio
import collections
import json
import os
import pathlib
import sys

import jiwer
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from senses.speech import normalise, numbers, transcribe  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
SET = ROOT / "eval" / "audio" / "radio_set"
OUT = ROOT / "eval" / "results" / "day_one"
CACHE = OUT / "t2_muse_cache.json"
USD_PER_MIN = 3 / 1000
PARALLEL = 6
async def run_all(files: list[pathlib.Path], cache: dict, key: str) -> None:
    sem = asyncio.Semaphore(PARALLEL)

    async def one(p: pathlib.Path):
        async with sem:
            for attempt in range(3):
                try:
                    x, sr = sf.read(p)
                    assert sr == 16000, sr
                    cache[p.name] = await transcribe(x, key)
                    print(f"  {p.name}: {cache[p.name]['text']}")
                    CACHE.write_text(json.dumps(cache, indent=1))
                    return
                except Exception as e:  # network or server error: retry, then report
                    print(f"  {p.name}: attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(2)

    await asyncio.gather(*(one(p) for p in files))


def main() -> int:
    key = os.environ["MODEL_API_KEY"]
    OUT.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    manifest = json.loads((SET / "manifest.json").read_text())
    files = sorted({SET / m["clean"] for m in manifest} | {SET / m["file"] for m in manifest}
                   | {SET / m["file"].replace(".wav", ".sam.wav") for m in manifest})
    files = [p for p in files if p.exists()]
    todo = [p for p in files if p.name not in cache]
    print(f"{len(files)} clips, {len(todo)} to transcribe")
    asyncio.run(run_all(todo, cache, key))
    missing = [p.name for p in files if p.name not in cache]
    if missing:
        print("not transcribed:", missing)
        return 1

    mms = {r["file"]: r for r in json.loads((OUT / "t2_wer_rows.json").read_text())}
    groups: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    exact: dict[str, list[bool]] = collections.defaultdict(list)
    rows, seen_clean = [], set()
    for m in manifest:
        ref = normalise(m["text"])
        sam = m["file"].replace(".wav", ".sam.wav")
        if m["clean"] not in seen_clean:
            seen_clean.add(m["clean"])
            groups["muse clean"].append((ref, normalise(cache[m["clean"]]["text"])))
            if numbers(m["text"]):
                exact["muse clean"].append(numbers(cache[m["clean"]]["text"]) == numbers(m["text"]))
        snr = f"snr{m['snr_db']:+d}"
        groups[f"muse raw {snr}"].append((ref, normalise(cache[m["file"]]["text"])))
        groups[f"muse sam {snr}"].append((ref, normalise(cache[sam]["text"])))
        if numbers(m["text"]):
            exact[f"muse raw {snr}"].append(numbers(cache[m["file"]]["text"]) == numbers(m["text"]))
            exact[f"muse sam {snr}"].append(numbers(cache[sam]["text"]) == numbers(m["text"]))
        old = mms.get(m["file"], {})
        for kind in ("raw", "sam"):
            if kind in old:
                groups[f"mms {kind} {snr}"].append((ref, normalise(old[kind])))
                if numbers(m["text"]):
                    exact[f"mms {kind} {snr}"].append(numbers(old[kind]) == numbers(m["text"]))
        rows.append({"file": m["file"], "ref": m["text"], "muse_raw": cache[m["file"]]["text"],
                     "muse_sam": cache[sam]["text"]})
    summary = {k: round(jiwer.wer([r for r, _ in v], [h for _, h in v]), 3) for k, v in sorted(groups.items())}
    audio_s = sum(cache[p.name]["audio_s"] for p in files)
    result = {"asr": "muse-voice-transcribe-1.0", "normalisation": "lowercase, no punctuation, each spoken number as one value",
              "wer": summary,
              "numbers_exact": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(exact.items())},
              "audio_minutes": round(audio_s / 60, 2),
              "est_cost_usd": round(audio_s / 60 * USD_PER_MIN, 4)}
    (OUT / "t2_muse_rows.json").write_text(json.dumps(rows, indent=1))
    (OUT / "t2_muse.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
