"""Word error rate on the radio set with Muse Voice Transcribe (Meta Model API, realtime).

Streams each clip to wss://api.meta.ai/v1/asr/realtime the way a live cockpit feed would:
a JSON handshake, 80 ms PCM16 frames at real-time pace, then endStream; the last final
transcript is the result. Scores clean speech, raw mixes and SAM Audio outputs
(<mix>.sam.wav), grouped by SNR, and rescores the MMS transcripts with the same text
normalisation so the two recognizers are compared like for like.

Voice Transcribe is not deterministic: the same clip can come back differently on another
pass ("Alaska 421" / "Alaska 4, two one"), so --passes N transcribes every clip N times and
reports each pass and the spread. Transcripts are cached in eval/results/day_one/<tag>_cache.json,
so a rerun only pays for what is not there yet. Price used for the cost line: $3 per 1,000 audio minutes.

Run:  uv run --group eval python eval/audio/wer_muse.py [--open] [--passes 3]
      (--open: radio_set_open, the publishable set with synthetic engine noise)
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
OPEN = "--open" in sys.argv  # the publishable set with synthetic engine noise
SET = ROOT / "eval" / "audio" / ("radio_set_open" if OPEN else "radio_set")
OUT = ROOT / "eval" / "results" / "day_one"
TAG = "t2_muse_open" if OPEN else "t2_muse"
CACHE = OUT / f"{TAG}_cache.json"
USD_PER_MIN = 3 / 1000
PARALLEL = 6
PASSES = int(sys.argv[sys.argv.index("--passes") + 1]) if "--passes" in sys.argv else 1
DIGITS_OK = "lowercase, no punctuation, each spoken number as one value"


def cache_key(name: str, r: int) -> str:
    """Pass 0 keeps the plain file name (the first, single-pass results); later passes add #r."""
    return name if r == 0 else f"{name}#{r}"


async def run_all(jobs: list[tuple[str, pathlib.Path]], cache: dict, key: str) -> None:
    sem = asyncio.Semaphore(PARALLEL)

    async def one(k: str, p: pathlib.Path):
        async with sem:
            for attempt in range(3):
                try:
                    x, sr = sf.read(p)
                    assert sr == 16000, sr
                    cache[k] = await transcribe(x, key)
                    print(f"  {k}: {cache[k]['text']}")
                    CACHE.write_text(json.dumps(cache, indent=1))
                    return
                except Exception as e:  # network or server error: retry, then report
                    print(f"  {k}: attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(2)

    await asyncio.gather(*(one(k, p) for k, p in jobs))


def score_pass(manifest: list[dict], text: dict[str, str], mms: dict) -> tuple[dict, dict, list]:
    """WER and numbers-exact for one pass; text maps clip file name -> transcript."""
    groups: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    exact: dict[str, list[bool]] = collections.defaultdict(list)
    rows, seen_clean = [], set()
    for m in manifest:
        ref = normalise(m["text"])
        sam = m["file"].replace(".wav", ".sam.wav")
        if m["clean"] not in seen_clean:
            seen_clean.add(m["clean"])
            groups["muse clean"].append((ref, normalise(text[m["clean"]])))
            if numbers(m["text"]):
                exact["muse clean"].append(numbers(text[m["clean"]]) == numbers(m["text"]))
        snr = f"snr{m['snr_db']:+d}"
        groups[f"muse raw {snr}"].append((ref, normalise(text[m["file"]])))
        if sam in text:
            groups[f"muse sam {snr}"].append((ref, normalise(text[sam])))
        if numbers(m["text"]):
            exact[f"muse raw {snr}"].append(numbers(text[m["file"]]) == numbers(m["text"]))
            if sam in text:
                exact[f"muse sam {snr}"].append(numbers(text[sam]) == numbers(m["text"]))
        old = mms.get(m["file"], {})
        for kind in ("raw", "sam"):
            if kind in old:
                groups[f"mms {kind} {snr}"].append((ref, normalise(old[kind])))
                if numbers(m["text"]):
                    exact[f"mms {kind} {snr}"].append(numbers(old[kind]) == numbers(m["text"]))
        rows.append({"file": m["file"], "ref": m["text"], "muse_raw": text[m["file"]], "muse_sam": text.get(sam)})
    wer = {k: round(jiwer.wer([r for r, _ in v], [h for _, h in v]), 3) for k, v in sorted(groups.items())}
    return wer, {k: f"{sum(v)}/{len(v)}" for k, v in sorted(exact.items())}, rows


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ["MODEL_API_KEY"]
    OUT.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    manifest = json.loads((SET / "manifest.json").read_text())
    files = sorted({SET / m["clean"] for m in manifest} | {SET / m["file"] for m in manifest}
                   | {SET / m["file"].replace(".wav", ".sam.wav") for m in manifest})
    files = [p for p in files if p.exists()]
    jobs = [(cache_key(p.name, r), p) for r in range(PASSES) for p in files if cache_key(p.name, r) not in cache]
    print(f"{len(files)} clips x {PASSES} passes, {len(jobs)} to transcribe")
    asyncio.run(run_all(jobs, cache, key))
    missing = [cache_key(p.name, r) for r in range(PASSES) for p in files if cache_key(p.name, r) not in cache]
    if missing:
        print("not transcribed:", missing)
        return 1

    mms = {} if OPEN else {r["file"]: r for r in json.loads((OUT / "t2_wer_rows.json").read_text())}
    passes = []
    for r in range(PASSES):
        wer, exact, rows = score_pass(manifest, {p.name: cache[cache_key(p.name, r)]["text"] for p in files}, mms)
        passes.append({"wer": wer, "numbers_exact": exact})
        if r == 0:
            (OUT / f"{TAG}_rows.json").write_text(json.dumps(rows, indent=1))
    audio_s = sum(cache[cache_key(p.name, r)]["audio_s"] for r in range(PASSES) for p in files)
    result = {"asr": "muse-voice-transcribe-1.0", "normalisation": DIGITS_OK,
              "wer": passes[0]["wer"], "numbers_exact": passes[0]["numbers_exact"], "passes": passes,
              "audio_minutes": round(audio_s / 60, 2), "est_cost_usd": round(audio_s / 60 * USD_PER_MIN, 4)}
    if PASSES > 1:
        keys = passes[0]["numbers_exact"].keys()
        result["numbers_exact_by_pass"] = {k: [p["numbers_exact"].get(k) for p in passes] for k in keys if k.startswith("muse")}
        result["wer_mean"] = {k: round(sum(p["wer"][k] for p in passes) / PASSES, 3) for k in passes[0]["wer"] if k.startswith("muse")}
    (OUT / f"{TAG}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "passes"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
