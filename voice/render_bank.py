"""Record Crosscheck's callout bank (voice/bank.py) once, with Fish Audio Drama 3 run locally.

  uv run python voice/render_bank.py design     # candidate voices from descriptions, each heard
                                                # back by Muse Voice Transcribe: voice/bank/candidates/
  uv run python voice/render_bank.py render <candidate.wav> [--missing]
                                                # every phrase and word in that voice, two takes each,
                                                # the take Voice Transcribe hears best kept: voice/bank/
                                                # (--missing: only what the bank does not have yet)

Drama 3 needs about 75 GB while it runs: stop other large models first. Built with Fish Audio
(Fish Audio Community License: non-commercial use; see README).
"""
import asyncio
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from senses.speech import normalise, transcribe  # noqa: E402
from voice.bank import BANK, PHRASES, SR, WORDS, key  # noqa: E402

DRAMA3 = pathlib.Path(os.environ.get("DRAMA3_DIR", pathlib.Path.home() / "llm-stack" / "drama3"))
TEST = "Cleared one-one thousand, set one-two thousand."
DESIGNS = {
    "alert_f": "[a calm, clear woman's voice: an aircraft cockpit alerting system, precise neutral diction, every word distinct, unhurried]",
    "alert_m": "[a calm, clear man's voice: an aircraft cockpit alerting system, precise neutral diction, every word distinct, unhurried]",
    "fo": "[an experienced airline first officer making a standard callout, calm and firm, every word distinct]",
}
CODE = (
    "import json, sys, numpy as np, soundfile as sf, mlx.core as mx\n"
    "from mlx_audio.tts.utils import load_model\n"
    "j = json.load(open(sys.argv[1]))\n"
    "m = load_model('mlx-community/fish-audio-drama-3-bf16')\n"
    "for k, text in j['lines'].items():\n"
    "    for take in range(j['takes']):\n"
    "        mx.random.seed(11 + 100 * take)\n"
    "        kw = {'ref_audio': j['ref'], 'ref_text': j['ref_text']} if j.get('ref') else {}\n"
    "        outs = list(m.generate(text=text, temperature=0.7, **kw))\n"
    "        a = np.concatenate([np.array(o.audio, dtype=np.float32).reshape(-1) for o in outs])\n"
    "        sf.write(f\"{j['out']}/{k}_{take}.wav\", a, outs[0].sample_rate)\n"
    "        print(k, take, f'{len(a) / outs[0].sample_rate:.1f} s', flush=True)\n")


def drama3(lines: dict[str, str], out: pathlib.Path, takes: int, ref: pathlib.Path | None = None, ref_text: str = "") -> None:
    out.mkdir(parents=True, exist_ok=True)
    job = out / "job.json"
    job.write_text(json.dumps({"lines": lines, "out": str(out), "takes": takes, "ref": str(ref) if ref else None, "ref_text": ref_text}))
    subprocess.run([str(DRAMA3 / ".venv" / "bin" / "python"), "-c", CODE, str(job)], cwd=DRAMA3,
                   env={**os.environ, "HF_HUB_OFFLINE": "1"}, check=True)


def to_bank_rate(path: pathlib.Path) -> np.ndarray:
    """24 kHz mono, silence trimmed to 30 ms each side, peak at 0.9."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    x = resample_poly(x, SR, sr).astype(np.float32) if sr != SR else x
    loud = np.flatnonzero(np.abs(x) > 0.02 * np.abs(x).max())
    pad = int(0.03 * SR)
    x = x[max(0, loud[0] - pad): loud[-1] + pad] if len(loud) else x
    return x * (0.9 / (np.abs(x).max() + 1e-9))


async def heard(x: np.ndarray, text: str) -> float:
    """Share of the words Voice Transcribe hears back (normalised, as the parser would)."""
    import difflib
    pcm = resample_poly(x, 16000, SR).astype(np.float32)
    got = normalise((await transcribe(np.concatenate([np.zeros(4000, np.float32), pcm, np.zeros(4000, np.float32)])))["text"])
    want = normalise(text)
    return difflib.SequenceMatcher(None, want.split(), got.split()).ratio()


def design() -> None:
    out = BANK / "candidates"
    drama3({name: f"{d} {TEST}" for name, d in DESIGNS.items()}, out, takes=3)
    scores = {}
    for f in sorted(out.glob("*.wav")):
        x = to_bank_rate(f)
        scores[f.name] = round(asyncio.run(heard(x, TEST)), 2)
        print(f.name, scores[f.name], f"{len(x) / SR:.1f} s")
    (out / "scores.json").write_text(json.dumps(scores, indent=1))


def render(ref: pathlib.Path, only_missing: bool = False) -> None:
    """only_missing: record just the phrases and words not in the bank yet, and keep the rest as
    they are (a new callout must not change how the old ones sound)."""
    raw = BANK / "raw"
    old_index = json.loads((BANK / "bank.json").read_text()) if only_missing and (BANK / "bank.json").exists() else {}
    old_report = json.loads((BANK / "report.json").read_text())["phrases"] if old_index else {}
    items = {key(t): t for t in PHRASES + WORDS if t not in old_index}
    if not items:
        print("nothing missing")
        return
    lines = {k: (t if t[-1] in ".?" else t.capitalize() + ".") for k, t in items.items()}  # a word alone, said as a word
    drama3(lines, raw, takes=2, ref=ref, ref_text=TEST)
    index, report = dict(old_index), dict(old_report)
    for k, text in items.items():
        best = None
        for take in range(2):
            f = raw / f"{k}_{take}.wav"
            if not f.exists():
                continue
            x = to_bank_rate(f)
            s = asyncio.run(heard(x, text)) if text in PHRASES else 1.0  # single words: kept by length below
            if best is None or s > best[0] or (s == best[0] and len(x) < len(best[1])):
                best = (s, x)
        if best is None:
            continue
        sf.write(BANK / f"{k}.wav", best[1], SR)
        index[text] = f"{k}.wav"
        report[text] = {"heard": round(best[0], 2), "s": round(len(best[1]) / SR, 2)}
    (BANK / "bank.json").write_text(json.dumps(index, indent=1))
    (BANK / "report.json").write_text(json.dumps({"reference": ref.name, "phrases": report}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "design":
        design()
    else:  # render <candidate.wav> [--missing]
        render(pathlib.Path(sys.argv[2]).resolve(), only_missing="--missing" in sys.argv)  # Drama 3 runs in its own folder
